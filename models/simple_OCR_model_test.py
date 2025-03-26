from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import models
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import argparse
import random
from torchvision import transforms

class RegionOCRDataset(Dataset):
    def __init__(self, data_samples, transform=None, is_training=True, response_mapping=None):
        """
        初始化 OCR 數據集
        
        Args:
            data_samples: 列表，每個元素是 (image_path, response, frame_num) 的元組
            transform: 圖像轉換
            is_training: 是否為訓練數據集
            response_mapping: 用於驗證/測試集的響應映射字典
        """
        self.samples = data_samples
        self.transform = transform
        self.is_training = is_training
        
        # 統計所有可能的響應
        if is_training or response_mapping is None:
            # 在訓練模式下建立映射
            self.unique_responses = sorted(set(response for _, response, _ in data_samples))
            # 添加一個特殊的 UNKNOWN 標籤用於處理未見過的響應
            self.unique_responses = ['<UNKNOWN>'] + self.unique_responses
            self.response_to_idx = {resp: idx for idx, resp in enumerate(self.unique_responses)}
            self.idx_to_response = {idx: resp for idx, resp in enumerate(self.unique_responses)}
        else:
            # 在驗證/測試模式下使用提供的映射
            self.response_to_idx = response_mapping['response_to_idx']
            self.idx_to_response = response_mapping['idx_to_response']
            self.unique_responses = list(self.response_to_idx.keys())
        
        print(f"數據集包含 {len(self.samples)} 個樣本，{len(self.unique_responses)} 種不同的響應")
        # 顯示前10種最常見的響應
        response_counts = defaultdict(int)
        for _, response, _ in data_samples:
            response_counts[response] += 1
        
        print("最常見的10種響應:")
        for resp, count in sorted(response_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - '{resp}': {count} 個樣本 ({count/len(self.samples)*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, response, frame_num = self.samples[idx]
        
        # 載入圖像
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # 將響應轉換為索引（分類問題）
        # 如果響應不在映射中，使用 UNKNOWN 的索引
        response_idx = self.response_to_idx.get(response, self.response_to_idx.get('<UNKNOWN>', 0))
        
        return image, response_idx, frame_num 

# -------------------------------
# 2. 模型定義
# -------------------------------
class DirectOCRModel(nn.Module):
    def __init__(self, num_classes):
        """
        初始化直接 OCR 預測模型
        
        Args:
            num_classes: 不同響應的數量（分類目標數）
        """
        super(DirectOCRModel, self).__init__()
        # 使用 EfficientNet-B0 作為基礎模型
        self.base_model = models.efficientnet_b0(pretrained=True)
        in_features = self.base_model.classifier[1].in_features
        
        # 替換最後的分類頭
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

class TwoStageOCRModel(nn.Module):
    def __init__(self, num_classes, digit_classifier_path=None):
        """
        初始化兩階段 OCR 預測模型
        
        Args:
            num_classes: 不同響應的數量（分類目標數）
            digit_classifier_path: 數字個數分類器的路徑
        """
        super(TwoStageOCRModel, self).__init__()
        # 使用 EfficientNet-B0 作為特徵提取器
        self.base_model = models.efficientnet_b0(pretrained=True)
        self.in_features = self.base_model.classifier[1].in_features
        
        # 第一階段: 數字個數分類器 (0,1,2,3個數字)
        self.digit_count_classifier = nn.Linear(self.in_features, 4)
        
        # 如果提供了數字個數分類器，載入之
        if digit_classifier_path and os.path.exists(digit_classifier_path):
            digit_classifier_state = torch.load(digit_classifier_path)
            # 根據實際模型結構載入參數
            # 這邊需要根據實際情況調整，例如可能只載入部分權重
            try:
                print(f"載入數字個數分類器: {digit_classifier_path}")
                # 這裡假設你的數字個數分類器是一個完整的模型，我們只取其最後的線性層
                # 實際情況可能需要調整
                self.base_model.load_state_dict(digit_classifier_state, strict=False)
            except Exception as e:
                print(f"載入數字個數分類器失敗: {e}")
        
        # 第二階段: 針對不同數字個數的專門分類器
        self.digit_specific_classifiers = nn.ModuleDict({
            '0': nn.Linear(self.in_features, num_classes),  # 處理無數字情況
            '1': nn.Linear(self.in_features, num_classes),  # 處理1個數字
            '2': nn.Linear(self.in_features, num_classes),  # 處理2個數字
            '3': nn.Linear(self.in_features, num_classes)   # 處理3個數字
        })
        
    def forward(self, x):
        # 提取特徵
        features = self.base_model.features(x)
        features = self.base_model.avgpool(features)
        features = torch.flatten(features, 1)
        
        # 第一階段: 數字個數預測
        digit_count_logits = self.digit_count_classifier(features)
        digit_count_probs = F.softmax(digit_count_logits, dim=1)
        predicted_digit_count = torch.argmax(digit_count_probs, dim=1)
        
        # 第二階段: 基於數字個數的具體預測
        batch_size = x.size(0)
        all_logits = []
        
        # 根據每個樣本預測的數字個數選擇對應的分類器
        for i in range(batch_size):
            count = predicted_digit_count[i].item()
            classifier = self.digit_specific_classifiers[str(count)]
            logits = classifier(features[i:i+1])
            all_logits.append(logits)
        
        # 將所有樣本的輸出堆疊在一起
        final_logits = torch.cat(all_logits, dim=0)
        
        # 返回數字個數預測和最終預測結果
        return final_logits, digit_count_logits

# -------------------------------
# 4. 訓練和評估函數
# -------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, is_two_stage=False):
    """訓練一個 epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if is_two_stage:
            # 對於兩階段模型，需要處理兩個輸出
            outputs, digit_count_outputs = model(images)
            # 我們只考慮最終的OCR輸出的損失
            loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss, accuracy

def validate_epoch(model, dataloader, criterion, device, is_two_stage=False):
    """評估一個 epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_frames = []
    all_digit_counts = [] if is_two_stage else None
    
    with torch.no_grad():
        for images, labels, frames in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            if is_two_stage:
                # 對於兩階段模型，需要處理兩個輸出
                outputs, digit_count_outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 保存數字個數預測結果
                digit_counts = digit_count_outputs.argmax(dim=1)
                all_digit_counts.extend(digit_counts.cpu().numpy())
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_frames.extend(frames.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    if is_two_stage:
        return epoch_loss, accuracy, all_labels, all_preds, all_frames, all_digit_counts
    else:
        return epoch_loss, accuracy, all_labels, all_preds, all_frames

# -------------------------------
# 5. K-折交叉驗證訓練函數
# -------------------------------
def train_with_cross_validation(data_samples, transform, output_dir, args):
    """
    使用 K-折交叉驗證訓練模型
    
    Args:
        data_samples: 數據樣本列表
        transform: 圖像轉換
        output_dir: 輸出目錄
        args: 參數
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取標籤用於分層劃分
    labels = [response for _, response, _ in data_samples]
    
    # 使用標籤長度作為分層標準，確保每個折中包含各種長度的響應
    label_lengths = [len(label) for label in labels]
    
    # 初始化分層 K-折交叉驗證
    n_splits = min(5, len(set(label_lengths)))  # 最多5折，最少與不同標籤長度數量相同
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    
    # 準備存儲每折的結果
    fold_results = []
    best_models = []
    
    # 循環訓練每一折
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_samples, label_lengths)):
        print(f"\n=== 訓練第 {fold+1}/{n_splits} 折 ===")
        
        # 創建當前折的訓練集和驗證集
        train_samples = [data_samples[i] for i in train_idx]
        val_samples = [data_samples[i] for i in val_idx]
        
        print(f"訓練集: {len(train_samples)} 個樣本，驗證集: {len(val_samples)} 個樣本")
        
        # 創建數據集和數據加載器
        train_dataset = RegionOCRDataset(train_samples, transform=transform, is_training=True)
        
        # 從訓練集獲取響應映射
        response_mapping = {
            'response_to_idx': train_dataset.response_to_idx,
            'idx_to_response': train_dataset.idx_to_response
        }
        
        # 創建驗證集，使用訓練集的響應映射
        val_dataset = RegionOCRDataset(val_samples, transform=transform, 
                                     is_training=False, response_mapping=response_mapping)
        
        # 數據加載器
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # 創建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DirectOCRModel(num_classes=len(train_dataset.unique_responses))
        model = model.to(device)
        print(f"模型參數: {sum(p.numel() for p in model.parameters())}")
        # 優化器和損失函數
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        
        # 訓練循環
        best_val_acc = 0.0
        best_model_state = None
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        patience = 0
        max_patience = 10  # 提前停止的耐心值
        
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}:")
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, y_true, y_pred, _ = validate_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            
            # 檢查是否是最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"提前停止於 epoch {epoch+1}，驗證準確率未改善")
                    break
        
        # 保存最佳模型及標籤映射
        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        model_path = os.path.join(fold_output_dir, "best_model.pth")
        torch.save(best_model_state, model_path)
        
        # 保存標籤映射
        mapping_path = os.path.join(fold_output_dir, "label_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'response_to_idx': {k: int(v) if isinstance(v, (int, np.integer)) else v 
                                  for k, v in train_dataset.response_to_idx.items()},
                'idx_to_response': {int(k) if isinstance(k, (int, np.integer)) else k: v 
                                  for k, v in train_dataset.idx_to_response.items()}
            }, f, ensure_ascii=False, indent=2)
        
        # 評估最佳模型
        model.load_state_dict(best_model_state)
        _, final_acc, y_true, y_pred, frames = validate_epoch(model, val_loader, criterion, device)
        
        # 生成混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap='Blues')  # annot=False 因為類別可能很多
        plt.xlabel('預測響應')
        plt.ylabel('真實響應')
        plt.title(f'折 {fold+1} 混淆矩陣 (準確率: {final_acc*100:.2f}%)')
        plt.savefig(os.path.join(fold_output_dir, "confusion_matrix.png"), dpi=300)
        plt.close()
        
        # 錯誤分析
        errors = [] 

# -------------------------------
# 6. 預測函數
# -------------------------------
def predict_with_ensemble(data_samples, transform, model_dir, output_dir, args):
    """
    使用集成模型進行預測
    
    Args:
        data_samples: 數據樣本列表
        transform: 圖像轉換
        model_dir: 模型目錄
        output_dir: 輸出目錄
        args: 參數
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 找到所有模型文件夾
    fold_dirs = [d for d in os.listdir(model_dir) if d.startswith("fold_") and os.path.isdir(os.path.join(model_dir, d))]
    
    # 結果存儲
    all_frame_nums = []
    all_true_responses = []
    all_pred_responses = []
    ensemble_votes = {}  # 每個幀號對應的預測投票
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 對每個折的模型進行預測
    for fold_dir in fold_dirs:
        fold_path = os.path.join(model_dir, fold_dir)
        model_path = os.path.join(fold_path, "best_model.pth")
        mapping_path = os.path.join(fold_path, "label_mapping.json")
        
        # 載入標籤映射
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            # 確保鍵是字符串，值是整數
            response_to_idx = {k: int(v) if isinstance(v, str) and v.isdigit() else v 
                             for k, v in mapping['response_to_idx'].items()}
            idx_to_response = {int(k) if isinstance(k, str) and k.isdigit() else k: v 
                             for k, v in mapping['idx_to_response'].items()}
        
        # 創建測試數據集，使用載入的映射
        test_dataset = RegionOCRDataset(
            data_samples, 
            transform=transform, 
            is_training=False, 
            response_mapping={'response_to_idx': response_to_idx, 'idx_to_response': idx_to_response}
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # 載入模型
        model = DirectOCRModel(num_classes=len(response_to_idx))
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        
        # 對測試集進行預測
        all_preds = []
        all_frames = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, frames in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_frames.extend(frames.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 將索引轉換為響應
        pred_responses = [idx_to_response.get(str(idx), idx_to_response.get(idx, "<UNKNOWN>")) for idx in all_preds]
        true_responses = [test_dataset.idx_to_response.get(str(idx), test_dataset.idx_to_response.get(idx, "<UNKNOWN>")) for idx in all_labels]
        
        # 更新集成投票
        for frame, pred in zip(all_frames, pred_responses):
            frame_num = int(frame)
            if frame_num not in ensemble_votes:
                ensemble_votes[frame_num] = {}
            
            if pred not in ensemble_votes[frame_num]:
                ensemble_votes[frame_num][pred] = 0
            ensemble_votes[frame_num][pred] += 1
        
        # 保存第一折的信息用於輸出
        if fold_dir == fold_dirs[0]:
            all_frame_nums = all_frames
            all_true_responses = true_responses
    
    # 根據投票確定最終預測
    for frame_num in ensemble_votes:
        votes = ensemble_votes[frame_num]
        # 選擇獲得最多票的響應
        max_votes = max(votes.values())
        best_responses = [resp for resp, vote in votes.items() if vote == max_votes]
        # 如果有多個響應得票相同，選擇第一個（可以改進為更複雜的策略）
        final_pred = best_responses[0]
        all_pred_responses.append(final_pred)
    
    # 確保結果按照幀號排序
    sorted_results = sorted(zip(all_frame_nums, all_true_responses, all_pred_responses), key=lambda x: x[0])
    all_frame_nums, all_true_responses, all_pred_responses = zip(*sorted_results)
    
    # 計算準確率
    accuracy = sum(1 for true, pred in zip(all_true_responses, all_pred_responses) if true == pred) / len(all_true_responses)
    print(f"集成模型準確率: {accuracy*100:.2f}%")
    
    # 創建結果表格
    results = []
    for frame, true, pred in zip(all_frame_nums, all_true_responses, all_pred_responses):
        is_correct = true == pred
        results.append({
            'frame': int(frame),
            'true_response': true,
            'pred_response': pred,
            'is_correct': is_correct
        })
    
    # 保存結果到CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(output_dir, "ensemble_predictions.csv"), index=False)
    
    # 保存錯誤案例
    error_df = result_df[result_df['is_correct'] == False]
    error_df.to_csv(os.path.join(output_dir, "ensemble_errors.csv"), index=False)
    
    # 可視化錯誤案例
    visualize_prediction_errors(error_df, data_samples, output_dir)
    
    print(f"預測完成! 總共處理了 {len(all_frame_nums)} 個樣本.")
    print(f"錯誤案例數量: {len(error_df)} ({len(error_df)/len(all_frame_nums)*100:.2f}%)")
    print(f"所有結果已保存到 {output_dir} 目錄")
    
    return result_df 

# -------------------------------
# 8. 主函數
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="直接 OCR 預測模型訓練與預測")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"],
                        help="操作模式: train (訓練) 或 predict (預測)")
    parser.add_argument("--jsonl", type=str, nargs='+', default=["2024-11-20_h/region2/region2.jsonl"], 
                        help="JSONL 文件路徑，可指定多個文件")
    parser.add_argument("--base-dir", type=str, default="", 
                        help="圖像路徑的基礎目錄 (如路徑已完整則留空)")
    parser.add_argument("--output-dir", type=str, default="ocr_models", 
                        help="輸出目錄")
    parser.add_argument("--model-dir", type=str, default="ocr_models",
                        help="模型目錄 (predict 模式需要)")
    parser.add_argument("--epochs", type=int, default=30, 
                        help="訓練輪數")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="學習率")
    parser.add_argument("--seed", type=int, default=42, 
                        help="隨機種子")
    parser.add_argument("--two-stage", action="store_true", 
                        help="使用兩階段模型 (先預測數字個數然後預測具體數字)")
    parser.add_argument("--digit-classifier", type=str, default="", 
                        help="數字個數分類器路徑 (用於兩階段模型)")
    args = parser.parse_args()
    
    # 設置隨機種子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 定義圖像前處理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # 載入數據
    data_samples = load_data_from_jsonl(args.jsonl, args.base_dir)
    
    if not data_samples:
        print("錯誤: 沒有找到有效的數據樣本")
        return
    
    # 根據模式執行不同操作
    if args.mode == "train":
        print("=== 開始訓練 ===")
        if args.two_stage:
            print("使用兩階段模型進行訓練")
            if args.digit_classifier:
                print(f"載入數字個數分類器: {args.digit_classifier}")
            # 這裡需要更新 train_with_cross_validation 函數支持兩階段模型
            # 暫時使用同一函數，但需實際修改
            train_with_cross_validation(data_samples, transform, args.output_dir, args)
        else:
            train_with_cross_validation(data_samples, transform, args.output_dir, args)
    else:  # predict
        print("=== 開始預測 ===")
        if args.two_stage:
            print("使用兩階段模型進行預測")
            if args.digit_classifier:
                print(f"載入數字個數分類器: {args.digit_classifier}")
            # 這裡需要更新 predict_with_ensemble 函數支持兩階段模型
            # 暫時使用同一函數，但需實際修改
            predict_with_ensemble(data_samples, transform, args.model_dir, args.output_dir, args)
        else:
            predict_with_ensemble(data_samples, transform, args.model_dir, args.output_dir, args)

if __name__ == "__main__":
    main() 