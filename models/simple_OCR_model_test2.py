#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接 OCR 預測模型：使用 CRNN+CTC 模型直接從 region2 圖像預測 OCR 結果（數字字串），
可處理 0~3 個整數數字（數字間以空格隔開）。
此腳本利用 K‑折交叉驗證充分利用有限數據，同時也方便未來進行集成預測。

使用方法:
    1. 訓練模式:
       python train_direct_ocr.py --mode train --jsonl region2/region2.jsonl --base-dir 2024-11-20_h --output-dir ocr_models
    2. 預測模式:
       python train_direct_ocr.py --mode predict --jsonl region2/region2_test.jsonl --base-dir 2024-11-20_h --output-dir ocr_results --model-dir ocr_models
"""

import os
import json
import re
import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from simple_OCR_model_test import save_prediction_results, ensemble_prediction
# -------------------------------
# 0. 定義字元表與編碼
# -------------------------------
VOCAB = "0123456789 "  # 10 個數字及一個空白，字元總數 11
# 約定 0 為 CTC blank，其它字元編號從 1 開始
char_to_idx = {char: i+1 for i, char in enumerate(VOCAB)}
idx_to_char = {i+1: char for i, char in enumerate(VOCAB)}
# blank index = 0
n_class = len(VOCAB) + 1  # 包含 blank

# -------------------------------
# 1. 定義資料集（CTC 版本）
# -------------------------------
class RegionOCRDatasetCTC(Dataset):
    def __init__(self, data_samples, transform=None, char_to_idx=char_to_idx):
        """
        初始化 OCR 數據集（CTC 版本），每筆資料包含圖像與對應的數字字串標籤
        Args:
            data_samples: 列表，每個元素為 (image_path, response, frame_num)
            transform: 圖像前處理 transform
            char_to_idx: 字元到索引的對應字典（不包含 blank，blank 固定為 0）
        """
        self.samples = data_samples
        self.transform = transform
        self.char_to_idx = char_to_idx
        
        print(f"資料集共 {len(self.samples)} 個樣本")
        # 顯示常見標籤（原始字串）
        response_counts = defaultdict(int)
        for _, response, _ in data_samples:
            response_counts[response] += 1
        print("前10種最常見標籤:")
        for resp, count in sorted(response_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - '{resp}': {count} 次")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, response, frame_num = self.samples[idx]
        # 載入圖像
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # 預處理標籤：strip 並轉換成字元索引序列（僅保留在 vocab 中的字元）
        response = response.strip()
        label_indices = [self.char_to_idx[c] for c in response if c in self.char_to_idx]
        label_length = len(label_indices)
        return image, torch.tensor(label_indices, dtype=torch.long), label_length, frame_num

# 自定義 collate function 處理變長標籤
def collate_fn_ctc(batch):
    images, labels, label_lengths, frames = zip(*batch)
    images = torch.stack(images, 0)
    # 將所有標籤連接成一個 1D tensor
    targets = torch.cat(labels)
    target_lengths = torch.tensor(label_lengths, dtype=torch.long)
    frames = torch.tensor(frames, dtype=torch.long)
    return images, targets, target_lengths, frames

# -------------------------------
# 2. 定義 CRNN 模型（使用簡單 CNN+LSTM 結構）
# -------------------------------
class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        """
        Args:
            imgH: 輸入圖像高度
            nc: 輸入通道數（例如 3）
            nclass: 分類數量（包含 blank）
            nh: LSTM 隱藏層維度
        """
        super(CRNN, self).__init__()
        # 簡單 CNN 提取特徵
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/2, W/2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)   # H/4, W/2
        )
        # 自適應池化將高度固定為 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        # LSTM 層（雙向）
        self.rnn = nn.LSTM(128, nh, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(nh * 2, nclass)
    
    def forward(self, x):
        # x: (batch, nc, H, W)
        conv = self.cnn(x)  # (batch, 128, H/4, W/4)
        conv = self.adaptive_pool(conv)  # (batch, 128, 1, W_out)
        conv = conv.squeeze(2)  # (batch, 128, W_out)
        conv = conv.permute(0, 2, 1)  # (batch, W_out, 128)
        recurrent, _ = self.rnn(conv)  # (batch, W_out, nh*2)
        output = self.fc(recurrent)    # (batch, W_out, nclass)
        output = output.permute(1, 0, 2)  # (T, batch, nclass) 符合 CTCLoss 輸入要求
        return output

# -------------------------------
# 3. 數據載入函數（同之前）
# -------------------------------
def load_data_from_jsonl(jsonl_path, base_dir=""):
    data_samples = []
    if isinstance(jsonl_path, str):
        jsonl_paths = [jsonl_path]
    else:
        jsonl_paths = jsonl_path
    for path in jsonl_paths:
        print(f"正在處理 JSONL 文件: {path}")
        if not os.path.exists(path):
            print(f"警告: JSONL 文件不存在: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    rel_path = data.get('images', '')
                    response = data.get('response', '').strip()
                    # 提取幀號
                    frame_match = re.search(r'frame_(\d+)\.png', rel_path)
                    frame_num = int(frame_match.group(1)) if frame_match else line_num
                    img_path = rel_path
                    if base_dir and not rel_path.startswith(base_dir):
                        img_path = os.path.join(base_dir, rel_path)
                    if os.path.exists(img_path):
                        data_samples.append((img_path, response, frame_num))
                    else:
                        # 嘗試其他可能的路徑組合
                        alternate_paths = [
                            img_path,
                            os.path.join(os.path.dirname(path), rel_path),
                            os.path.join(base_dir, os.path.basename(rel_path))
                        ]
                        found = False
                        for alt_path in alternate_paths:
                            if os.path.exists(alt_path):
                                data_samples.append((alt_path, response, frame_num))
                                found = True
                                break
                        if not found:
                            print(f"警告: 找不到圖像 {img_path} (行 {line_num})")
                except Exception as e:
                    print(f"處理 JSONL 數據時出錯 (行 {line_num}): {e}")
                    continue
    print(f"共從 {len(jsonl_paths)} 個 JSONL 文件中加載 {len(data_samples)} 個有效樣本")
    data_samples.sort(key=lambda x: x[2])
    return data_samples

# -------------------------------
# 4. 訓練與驗證函數
# -------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets, target_lengths, _ in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # outputs: (T, batch, nclass)
        T, batch, _ = outputs.size()
        input_lengths = torch.full(size=(batch,), fill_value=T, dtype=torch.long).to(device)
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def greedy_decode(outputs, idx_to_char):
    # outputs: (T, batch, nclass)
    outputs = outputs.argmax(dim=2)  # (T, batch)
    outputs = outputs.transpose(0,1)  # (batch, T)
    pred_strings = []
    for seq in outputs:
        pred = []
        prev = None
        for idx in seq.cpu().numpy():
            if idx != 0 and idx != prev:
                pred.append(idx_to_char.get(idx, ''))
            prev = idx
        pred_strings.append(''.join(pred))
    return pred_strings

def validate_epoch(model, dataloader, criterion, device, idx_to_char):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets, target_lengths, _ in tqdm(dataloader, desc="Val", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            outputs = model(images)  # (T, batch, nclass)
            T, batch, _ = outputs.size()
            input_lengths = torch.full(size=(batch,), fill_value=T, dtype=torch.long).to(device)
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            running_loss += loss.item() * images.size(0)
            preds = greedy_decode(outputs, idx_to_char)
            # 將 targets 轉回字串（需用到 target_lengths與拆分 targets）
            # 由於 targets 是連接的，需要按 batch 拆分
            start = 0
            for tl in target_lengths.cpu().numpy():
                t_seq = targets[start:start+tl]
                t_str = ''.join([idx_to_char.get(int(idx), '') for idx in t_seq.cpu().numpy()])
                all_targets.append(t_str)
                start += tl
            all_preds.extend(preds)
    epoch_loss = running_loss / len(dataloader.dataset)
    # 計算字串準確率（僅作參考，因資料可能很少）
    correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
    acc = correct / len(all_preds)
    return epoch_loss, acc, all_targets, all_preds

# -------------------------------
# 5. K-折交叉驗證訓練函數
# -------------------------------
def train_with_cross_validation(data_samples, transform, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    
    # 為 CTC 訓練，採用標籤字串長度作為分層依據（這裡僅以標籤長度簡單分層）
    labels = [response for _, response, _ in data_samples]
    label_lengths = [len(response.strip()) for response in labels]
    
    n_splits = min(5, len(set(label_lengths)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    
    fold_results = []
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_samples, label_lengths)):
        print(f"\n=== 訓練第 {fold+1}/{n_splits} 折 ===")
        train_samples = [data_samples[i] for i in train_idx]
        val_samples = [data_samples[i] for i in val_idx]
        print(f"訓練集: {len(train_samples)} 個樣本，驗證集: {len(val_samples)} 個樣本")
        
        train_dataset = RegionOCRDatasetCTC(train_samples, transform=transform, char_to_idx=char_to_idx)
        val_dataset = RegionOCRDatasetCTC(val_samples, transform=transform, char_to_idx=char_to_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_ctc)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_ctc)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 假設圖像高度固定為 32（可依實際情況調整），輸入通道 3
        model = CRNN(imgH=32, nc=3, nclass=n_class, nh=256)
        model = model.to(device)
        print(f"模型參數: {sum(p.numel() for p in model.parameters())}")
        
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        
        best_val_acc = 0.0
        best_model_state = None
        train_losses = []
        val_losses = []
        patience = 0
        max_patience = 10
        
        for epoch in tqdm(range(args.epochs), desc=f"Fold {fold+1} Epochs"):
            t_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, _, _ = validate_epoch(model, val_loader, criterion, device, idx_to_char)
            scheduler.step(v_loss)
            train_losses.append(t_loss)
            val_losses.append(v_loss)
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={t_loss:.4f} | Val Loss={v_loss:.4f} | Val Acc={v_acc*100:.2f}%")
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_model_state = model.state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"提前停止於 epoch {epoch+1}")
                    break
        
        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        model_path = os.path.join(fold_output_dir, "best_model_OCR_region2.pth")
        torch.save(best_model_state, model_path)
        # 儲存字元映射
        mapping_path = os.path.join(fold_output_dir, "char_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({'char_to_idx': char_to_idx, 'idx_to_char': {str(k): v for k, v in idx_to_char.items()}}, f, ensure_ascii=False, indent=2)
        
        model.load_state_dict(best_model_state)
        _, final_acc, gt_strings, pred_strings = validate_epoch(model, val_loader, criterion, device, idx_to_char)
        print(f"折 {fold+1} 驗證準確率: {final_acc*100:.2f}%")
        fold_results.append(final_acc)
        best_models.append((best_model_state, char_to_idx, idx_to_char))
        
        # 畫出損失曲線
        plt.figure(figsize=(8,4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Fold {fold+1} Loss Curve")
        plt.savefig(os.path.join(fold_output_dir, "loss_curve.png"), dpi=300)
        plt.close()
    
    print("\n=== 交叉驗證結果 ===")
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    print(f"平均準確率: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    
    ensemble_info = {
        'num_folds': n_splits,
        'fold_accuracies': [float(acc) for acc in fold_results],
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc)
    }
    with open(os.path.join(output_dir, "ensemble_info.json"), 'w', encoding='utf-8') as f:
        json.dump(ensemble_info, f, ensure_ascii=False, indent=2)
    
    return best_models

# -------------------------------
# 6. 預測函數
# -------------------------------
def predict_with_ensemble(data_samples, transform, model_dir, output_dir, args):
    """
    使用集成模型進行預測
    """
    os.makedirs(output_dir, exist_ok=True)
    test_dataset = RegionOCRDatasetCTC(data_samples, transform=transform, char_to_idx=char_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_ctc)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 找到所有模型文件夾
    fold_dirs = [d for d in os.listdir(model_dir) if d.startswith("fold_") and os.path.isdir(os.path.join(model_dir, d))]
    if not fold_dirs:
        print(f"錯誤: 在 {model_dir} 中沒有找到模型文件夾")
        return None
    
    # 載入ensemble_info獲取fold準確率
    ensemble_info_path = os.path.join(model_dir, "ensemble_info.json")
    fold_accuracies = {}
    if os.path.exists(ensemble_info_path):
        with open(ensemble_info_path, 'r', encoding='utf-8') as f:
            ensemble_info = json.load(f)
            if 'fold_accuracies' in ensemble_info:
                fold_accuracies = {f"fold_{i}": acc for i, acc in enumerate(ensemble_info['fold_accuracies'])}
    
    # 所有fold的預測結果 {frame_num: {fold_id: prediction}}
    all_fold_predictions = {}
    # 我們不使用真實標籤，因為CTC模型的標籤可能沒有直接映射
    ground_truth = {}
    
    # 載入標籤映射
    mapping_path = os.path.join(model_dir, fold_dirs[0], "char_mapping.json")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
        idx_to_char_loaded = {int(k): v for k, v in mapping['idx_to_char'].items()}
    
    # 對每個fold的模型進行預測
    for fold_id, fold_dir in enumerate(fold_dirs):
        fold_path = os.path.join(model_dir, fold_dir)
        model_path = os.path.join(fold_path, "best_model_OCR_region2.pth")
        
        try:
            # 載入模型
            model = CRNN(imgH=32, nc=3, nclass=n_class, nh=256)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            
            # 預測
            with torch.no_grad():
                for images, targets, target_lengths, frames in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = greedy_decode(outputs, idx_to_char_loaded)
                    
                    # 提取真實標籤
                    start = 0
                    for i, (frame, pred, length) in enumerate(zip(frames, preds, target_lengths)):
                        frame_num = int(frame.item())
                        t_seq = targets[start:start+length]
                        true_label = ''.join([idx_to_char_loaded.get(int(idx), '') for idx in t_seq.cpu().numpy()])
                        start += length
                        
                        # 保存真實標籤
                        if frame_num not in ground_truth:
                            ground_truth[frame_num] = true_label
                        
                        # 保存預測結果
                        if frame_num not in all_fold_predictions:
                            all_fold_predictions[frame_num] = {}
                        all_fold_predictions[frame_num][fold_dir] = pred
            
            print(f"完成了 {fold_dir} 的預測")
            
        except Exception as e:
            print(f"處理 {fold_dir} 時出錯: {str(e)}")
            continue
    
    # 使用集成方法得到最終預測
    ensemble_method = args.ensemble_method if hasattr(args, 'ensemble_method') else "weighted"
    final_predictions = ensemble_prediction(
        all_fold_predictions, 
        ensemble_method=ensemble_method,
        fold_accuracies=fold_accuracies
    )
    
    # 保存結果
    save_prediction_results(
        final_predictions, 
        output_dir,
        "CRNN+CTC",
        ground_truth
    )
    
    # 可視化錯誤案例
    result_df = pd.DataFrame([
        {'frame': frame, 'true_response': ground_truth.get(frame, ""), 'pred_response': pred, 
         'is_correct': ground_truth.get(frame, "") == pred}
        for frame, pred in final_predictions.items()
    ])
    error_df = result_df[result_df['is_correct'] == False]
    visualize_prediction_errors(error_df, data_samples, output_dir)
    
    return result_df
# -------------------------------
# 7. 可視化錯誤案例（如有真實標籤時可用）
# -------------------------------
def visualize_prediction_errors(error_df, data_samples, output_dir):
    if error_df.empty:
        print("沒有預測錯誤，跳過可視化")
        return
    frame_to_sample = {frame: (img_path, resp) for img_path, resp, frame in data_samples}
    n_errors = min(25, len(error_df))
    grid_size = int(np.ceil(np.sqrt(n_errors)))
    plt.figure(figsize=(15,15))
    plt.suptitle(f'預測錯誤範例 ({len(error_df)} 個錯誤)', fontsize=20)
    for i, (_, error) in enumerate(error_df.head(n_errors).iterrows()):
        frame = error['frame']
        if frame in frame_to_sample:
            img_path, _ = frame_to_sample[frame]
            plt.subplot(grid_size, grid_size, i+1)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(f"Frame {frame}\nTrue: {error['true_response']}\nPred: {error['pred_response']}", fontsize=8)
            plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(output_dir, "error_samples.png"), dpi=300)
    plt.close()

# -------------------------------
# 8. 主函數
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="直接 OCR 預測模型訓練與預測 (CRNN+CTC)")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"],
                        help="操作模式: train (訓練) 或 predict (預測)")
    parser.add_argument("--jsonl", type=str, nargs='+', default=["2024-11-20_h/region2/region2.jsonl"],
                        help="JSONL 文件路徑，可指定多個文件")
    parser.add_argument("--base-dir", type=str, default="",
                        help="圖像路徑的基礎目錄")
    parser.add_argument("--output-dir", type=str, default="simple_ocr_models2",
                        help="輸出目錄")
    parser.add_argument("--model-dir", type=str, default="simple_ocr_models2",
                        help="模型目錄 (predict 模式需要)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="訓練輪數")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="學習率")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機種子")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    transform = transforms.Compose([
        transforms.Resize((32, 128)),  # 固定高度 32, 寬度調整至 128（可依實際資料調整）
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    data_samples = load_data_from_jsonl(args.jsonl, args.base_dir)
    if not data_samples:
        print("錯誤: 沒有找到有效的數據樣本")
        return
    
    if args.mode == "train":
        print("=== 開始訓練 ===")
        train_with_cross_validation(data_samples, transform, args.output_dir, args)
    else:
        print("=== 開始預測 ===")
        predict_with_ensemble(data_samples, transform, args.model_dir, args.output_dir, args)

if __name__ == "__main__":
    main()
