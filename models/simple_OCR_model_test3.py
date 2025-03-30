#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR 预测模型：使用预训练 Hezar CRNN 模型进行微调，针对数字识别优化
可处理 0~3 个整数数字（数字间以空格隔开），并支持负号识别。
此脚本利用 K 折交叉验证充分利用有限数据，并基于预训练模型进行微调以提高通用性。

使用方法:
    1. 训练模式:
       python train_finetune_hezar.py --mode train --jsonl data/2024-11-20_h/region2/region2.jsonl --output-dir ocr_models
    
    2. 预测模式:
       python train_finetune_hezar.py --mode predict --jsonl data/2024-11-20_h/region2/region2_test.jsonl --output-dir ocr_results --model-dir ocr_models
       
    3. 带帧变化检测的预测:
       python train_finetune_hezar.py --mode predict --jsonl data/2024-11-20_h/region2/region2_test.jsonl --output-dir ocr_results --model-dir ocr_models --detect-changes --similarity-threshold 0.92
    
    4. 指定感兴趣区域的预测:
       python train_finetune_hezar.py --mode predict --jsonl data/2024-11-20_h/region2/region2_test.jsonl --output-dir ocr_results --model-dir ocr_models --roi 100,200,400,300
"""

import os
import json
import re
import argparse
import random
from collections import defaultdict
import datetime

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoFeatureExtractor
from sklearn.model_selection import StratifiedKFold

# 导入 Hezar
try:
    from hezar.models import Model
except ImportError:
    print("请安装 Hezar 库: pip install hezar")
    print("临时使用替代方法加载...")

# -------------------------------
# 0. 定义字符表与编码（包含负号）
# -------------------------------
VOCAB = "-0123456789 "  # 包含负号、10个数字和空白
char_to_idx = {char: i+1 for i, char in enumerate(VOCAB)}  # blank 为 0
idx_to_char = {i+1: char for i, char in enumerate(VOCAB)}
n_class = len(VOCAB) + 1  # 包含 blank

# -------------------------------
# 1. 图像预处理函数：自适应二值化与形态学操作
# -------------------------------
def preprocess_image(image):
    """对输入 PIL.Image 进行灰度、自适应二值化及简单形态学处理"""
    gray = ImageOps.grayscale(image)
    gray_cv = np.array(gray)
    bin_img = cv2.adaptiveThreshold(gray_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(bin_img, kernel, iterations=1)
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(processed)

# -------------------------------
# 2. Hezar 模型包装类
# -------------------------------
class HezarCRNNWrapper(nn.Module):
    """包装 Hezar CRNN 模型以便于微调"""
    def __init__(self, model_name="hezarai/crnn-fa-printed-96-long", vocab_size=None):
        super(HezarCRNNWrapper, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # 使用 Hezar 加载模型
            self.model = Model.load(model_name)
            # 获取模型内部结构
            self.backbone = self.model.model
            
            # 替换最后一层以适应新的词汇表大小
            if vocab_size is not None:
                # 通常 CRNN 最后一层是线性层，连接到词汇表大小
                if hasattr(self.backbone, 'classifier'):
                    old_fc = self.backbone.classifier
                    in_features = old_fc.in_features if hasattr(old_fc, 'in_features') else old_fc.inplanes
                    self.backbone.classifier = nn.Linear(in_features, vocab_size)
                elif hasattr(self.backbone, 'fc'):
                    old_fc = self.backbone.fc
                    in_features = old_fc.in_features
                    self.backbone.fc = nn.Linear(in_features, vocab_size)
                else:
                    print("警告: 无法确定如何替换最后一层，模型可能不支持词汇表大小调整")
        except Exception as e:
            print(f"加载 Hezar 模型时出错: {e}")
            print("初始化基础 CRNN 模型...")
            self._init_basic_crnn(vocab_size or n_class)
    
    def _init_basic_crnn(self, vocab_size):
        """初始化简单的 CRNN 模型作为后备"""
        # CNN 特征提取部分
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        # 修改 RNN 部分以避免递归错误
        # 使用单个 GRU 层而不是使用 Sequential
        self.rnn1 = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.rnn2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        
        # 分类器
        self.classifier = nn.Linear(512, vocab_size)
        
        # 取消引用自身，改为直接设置各个组件
        # 不再设置 self.backbone = self
    
    def forward(self, x):
        """前向传播"""
        try:
            # 尝试使用 Hezar 模型
            if hasattr(self, 'model') and hasattr(self.model, 'predict_batch'):
                return self.model.predict_batch(x)
        except Exception as e:
            print(f"Hezar 模型前向传播出错，使用后备模型: {e}")
            
        # 使用后备 CRNN 模型
        if hasattr(self, 'cnn'):
            # CNN 特征提取
            x = self.cnn(x)  # [batch_size, channels, height, width]
            
            # 调整形状以便 RNN 处理
            b, c, h, w = x.size()
            x = x.squeeze(2)  # 压缩高度维度
            x = x.permute(0, 2, 1)  # [batch_size, width, channels]
            
            # RNN 序列处理 - 使用单独的 RNN 层而不是 Sequential
            x, _ = self.rnn1(x)
            x, _ = self.rnn2(x)
            
            # 分类
            x = self.classifier(x)  # [batch_size, width, vocab_size]
            
            # 转置以匹配 CTC 损失的预期格式: [sequence_length, batch_size, vocab_size]
            x = x.permute(1, 0, 2)
            return x
        
        # 如果没有 CNN 和 RNN，则返回错误
        raise RuntimeError("模型初始化失败，既没有 Hezar 模型也没有基础 CRNN 模型")
    
    def predict(self, x):
        """预测方法"""
        if hasattr(self, 'model') and hasattr(self.model, 'predict'):
            return self.model.predict(x)
        
        # 后备预测
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            # output: [width, batch_size, vocab_size]
            output = output.permute(1, 0, 2)  # [batch_size, width, vocab_size]
            
            # 贪婪解码
            _, indices = torch.max(output, dim=2)  # [batch_size, width]
            indices = indices.cpu().numpy()
            
            results = []
            for idx in indices:
                # CTC 解码: 合并重复字符并移除空白
                collapsed = []
                prev = -1
                for i in idx:
                    if i != 0 and i != prev:  # 0 是空白符
                        collapsed.append(i)
                    prev = i
                
                # 转换为字符
                text = ''.join([idx_to_char.get(int(i), '') for i in collapsed])
                results.append(text)
            
            return results

# -------------------------------
# 3. 數據集（基於 torch.utils.data.Dataset）
# -------------------------------
class RegionOCRDatasetCTC(Dataset):
    def __init__(self, data_samples, transform=None, preproc_fn=None, char_to_idx=char_to_idx):
        """
        data_samples: 列表，每個元素為 (image_path, response, frame_num)
        transform: 圖像預處理 transform
        preproc_fn: 額外的圖像預處理函數（例如 preprocess_image）
        """
        self.samples = data_samples
        self.transform = transform
        self.preproc_fn = preproc_fn
        self.char_to_idx = char_to_idx
        
        print(f"數據集共 {len(self.samples)} 個樣本")
        counter = defaultdict(int)
        for _, resp, _ in data_samples:
            counter[resp.strip()] += 1
        print("前10種最常見標籤:")
        for k, v in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - '{k}': {v} 次")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text, frame_num = self.samples[idx]
        
        # 使用 PIL 打開圖像
        try:
            image = Image.open(img_path).convert('RGB')
            
            # 如果提供了預處理函數，應用它（確保它返回 PIL 圖像）
            if self.preproc_fn:
                image = self.preproc_fn(image)
                
                # 確保預處理後的圖像是 PIL 格式
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image.astype('uint8'))
            
            # 應用轉換
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"處理圖像 {img_path} 出錯: {e}")
            # 創建一個空白的 PIL 圖像作為替代
            image = Image.new('RGB', (128, 32), color='white')
            if self.transform:
                image = self.transform(image)
        
        # 編碼文本標籤
        target = []
        for char in text:
            if char in self.char_to_idx:
                target.append(self.char_to_idx[char])
            else:
                # 如果字符不在映射表中，可以跳過或用特殊標記替代
                pass
        
        # 返回圖像、目標序列、目標長度和幀號
        return image, torch.tensor(target, dtype=torch.long), len(target), frame_num

def collate_fn_ctc(batch):
    """
    自定義 collate 函數，用於批次處理 CTC 數據
    
    Args:
        batch: 一個批次的數據，每個元素為 (image, target, target_length, frame_num)
        
    Returns:
        images: 批次圖像張量
        targets: 所有目標序列的連接張量
        target_lengths: 每個目標序列的長度張量
        frame_nums: 每個樣本的幀號列表
    """
    images, targets, target_lengths, frame_nums = zip(*batch)
    
    # 堆疊圖像
    images = torch.stack(images, 0)
    
    # 連接所有目標序列
    targets = torch.cat(targets)
    
    # 轉換目標長度為張量
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return images, targets, target_lengths, frame_nums

# -------------------------------
# 4. 数据载入函数
# -------------------------------
def load_data_from_jsonl(jsonl_path, base_dir=""):
    data_samples = []
    if isinstance(jsonl_path, str):
        jsonl_paths = [jsonl_path]
    else:
        jsonl_paths = jsonl_path
    for path in jsonl_paths:
        print(f"正在处理 JSONL 文件: {path}")
        if not os.path.exists(path):
            print(f"警告: JSONL 文件不存在: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    rel_path = data.get("images", "")
                    response = data.get("response", "").strip()
                    frame_match = re.search(r"frame_(\d+)\.png", rel_path)
                    frame_num = int(frame_match.group(1)) if frame_match else line_num
                    possible_paths = [
                        rel_path,
                        os.path.join(base_dir, rel_path),
                        os.path.join("data", rel_path),
                        os.path.join("data", os.path.basename(rel_path)),
                        os.path.join("../data", rel_path),
                        os.path.join("../data", os.path.basename(rel_path))
                    ]
                    video_match = re.search(r"([\d-]+_[a-zA-Z]+)", rel_path)
                    if video_match:
                        video_name = video_match.group(1)
                        possible_paths.append(os.path.join("data", video_name, os.path.basename(rel_path)))
                        possible_paths.append(os.path.join("../data", video_name, os.path.basename(rel_path)))
                    found = False
                    for img_path in possible_paths:
                        if os.path.exists(img_path):
                            data_samples.append((img_path, response, frame_num))
                            found = True
                            break
                    if not found:
                        print(f"警告: 找不到图像 {rel_path} (行 {line_num})")
                except Exception as e:
                    print(f"处理 JSONL 数据时出错 (行 {line_num}): {e}")
                    continue
    print(f"共从 {len(jsonl_paths)} 个 JSONL 文件中加载 {len(data_samples)} 个有效样本")
    data_samples.sort(key=lambda x: x[2])
    return data_samples

# -------------------------------
# 5. 训练与验证函数
# -------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        
    Returns:
        epoch_loss: 本轮训练的平均损失
        epoch_acc: 本轮训练的准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, targets, target_lengths, _) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算输出长度 (预设所有样本输出长度相同)
        output_lengths = torch.full(size=(outputs.size(1),), 
                                fill_value=outputs.size(0), 
                                dtype=torch.long).to(device)
        
        # 计算损失
        loss = criterion(outputs, targets, output_lengths, target_lengths)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        # 记录损失
        running_loss += loss.item()
        
        # 计算准确率 (使用贪婪解码)
        batch_size = images.size(0)
        total += batch_size
        
        # 贪婪解码
        _, indices = torch.max(outputs.transpose(0, 1), dim=2)
        indices = indices.cpu().numpy()
        
        # 计算正确预测
        for i, (idx, length) in enumerate(zip(indices, target_lengths.cpu().numpy())):
            # CTC 解码: 合并重复字符并移除空白
            collapsed = []
            prev = -1
            for j in idx:
                if j != 0 and j != prev:  # 0 是空白符
                    collapsed.append(j)
                prev = j
            
            # 获取真实标签
            true_label = targets[sum(target_lengths[:i]):sum(target_lengths[:i])+length].cpu().numpy()
            
            # 比较预测与真实标签
            if len(collapsed) == len(true_label) and np.array_equal(collapsed, true_label):
                correct += 1
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    # 计算本轮的平均损失和准确率
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device, idx_to_char):
    """
    验证一个 epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        idx_to_char: 索引到字符的映射
        
    Returns:
        epoch_loss: 本轮验证的平均损失
        epoch_acc: 本轮验证的准确率
        val_targets: 真实标签列表
        val_preds: 预测结果列表
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    val_targets = []
    val_preds = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating")
        for batch_idx, (images, targets, target_lengths, _) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算输出长度 (预设所有样本输出长度相同)
            output_lengths = torch.full(size=(outputs.size(1),), 
                                    fill_value=outputs.size(0), 
                                    dtype=torch.long).to(device)
            
            # 计算损失
            loss = criterion(outputs, targets, output_lengths, target_lengths)
            
            # 记录损失
            running_loss += loss.item()
            
            # 计算准确率 (使用贪婪解码)
            batch_size = images.size(0)
            total += batch_size
            
            # 贪婪解码
            _, indices = torch.max(outputs.transpose(0, 1), dim=2)
            indices = indices.cpu().numpy()
            
            # 计算正确预测并收集预测结果
            for i, (idx, length) in enumerate(zip(indices, target_lengths.cpu().numpy())):
                # CTC 解码: 合并重复字符并移除空白
                collapsed = []
                prev = -1
                for j in idx:
                    if j != 0 and j != prev:  # 0 是空白符
                        collapsed.append(j)
                    prev = j
                
                # 获取真实标签
                true_label = targets[sum(target_lengths[:i]):sum(target_lengths[:i])+length].cpu().numpy()
                
                # 转换为字符串
                true_text = ''.join([idx_to_char.get(int(i), '') for i in true_label])
                pred_text = ''.join([idx_to_char.get(int(i), '') for i in collapsed])
                
                # 保存真实标签和预测结果
                val_targets.append(true_text)
                val_preds.append(pred_text)
                
                # 比较预测与真实标签
                if true_text == pred_text:
                    correct += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    # 计算本轮的平均损失和准确率
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, val_targets, val_preds

# -------------------------------
# 6. K 折交叉验证训练函数
# -------------------------------
def train_with_cross_validation(data_samples, transform, output_dir, args):
    """
    使用 K 折交叉驗證訓練模型
    
    Args:
        data_samples: 數據樣本列表
        transform: 圖像轉換
        output_dir: 輸出目錄
        args: 參數
    """
    print("\n開始訓練 K 折交叉驗證模型...")
    
    # 檢查數據樣本數量
    if len(data_samples) == 0:
        print("錯誤: 沒有可用的訓練數據!")
        return

    # 提取特徵和標籤
    features = []
    labels = []
    frame_nums = []
    for img_path, response, frame_num in data_samples:
        features.append(img_path)
        labels.append(response)
        frame_nums.append(frame_num)
    
    print(f"數據集包含 {len(features)} 個樣本，{len(set(labels))} 個不同的標籤")
    
    # 建立字符到索引的映射
    all_chars = set()
    for label in labels:
        all_chars.update(label)
    all_chars = sorted(list(all_chars))
    
    # 添加空白字符用於 CTC 損失
    all_chars = ['<blank>'] + all_chars
    
    # 建立映射
    char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(all_chars)}
    
    print(f"字符集大小: {len(all_chars)}，包括空白字符")
    print(f"前10個字符: {all_chars[:10] if len(all_chars) > 10 else all_chars}")
    
    # 設置 n_splits 至少為 2
    n_splits = min(5, len(set(labels)))  # 使用最多 5 折，但不超過唯一標籤數量
    if n_splits < 2:
        n_splits = 2  # 確保至少有一個訓練/測試分割
    
    print(f"使用 {n_splits} 折交叉驗證")
    
    # 創建 StratifiedKFold 對象
    # 由於標籤可能是字符串，無法直接用於分層抽樣，創建標籤組
    label_groups = {}
    for i, label in enumerate(labels):
        if label not in label_groups:
            label_groups[label] = len(label_groups)
        labels[i] = label_groups[label]
    
    # 現在 labels 包含數字，可以用於 StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    
    # 初始化模型性能追蹤
    fold_accuracies = []
    fold_losses = []
    best_model_paths = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 對於每一折
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"\n===== 訓練第 {fold+1}/{n_splits} 折 =====")
        
        # 創建當前折的輸出目錄
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # 保存字符映射
        mapping = {
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char
        }
        with open(os.path.join(fold_dir, "char_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)
        
        # 準備訓練集和驗證集
        train_features = [features[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        train_frame_nums = [frame_nums[i] for i in train_idx]
        
        val_features = [features[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        val_frame_nums = [frame_nums[i] for i in val_idx]
        
        # 將特徵、標籤和幀號組合回數據樣本格式
        train_samples = []
        for i in range(len(train_features)):
            img_path = train_features[i]
            label = list(label_groups.keys())[list(label_groups.values()).index(train_labels[i])]
            frame_num = train_frame_nums[i]
            train_samples.append((img_path, label, frame_num))
        
        val_samples = []
        for i in range(len(val_features)):
            img_path = val_features[i]
            label = list(label_groups.keys())[list(label_groups.values()).index(val_labels[i])]
            frame_num = val_frame_nums[i]
            val_samples.append((img_path, label, frame_num))
        
        print(f"訓練集: {len(train_samples)} 樣本，驗證集: {len(val_samples)} 樣本")
        
        # 創建數據集
        train_dataset = RegionOCRDatasetCTC(
            [(features[i], data_samples[train_idx[i]][1], frame_nums[i]) for i in range(len(train_idx))],
            transform=transform,
            preproc_fn=preprocess_image,
            char_to_idx=char_to_idx
        )

        val_dataset = RegionOCRDatasetCTC(
            [(features[i], data_samples[val_idx[i]][1], frame_nums[i]) for i in range(len(val_idx))],
            transform=transform,
            preproc_fn=preprocess_image,
            char_to_idx=char_to_idx
        )
        
        # 創建數據加載器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn_ctc,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn_ctc,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 初始化模型
        vocab_size = len(all_chars)
        model = HezarCRNNWrapper(vocab_size=vocab_size)
        model = model.to(device)
        
        # 設置優化器和損失函數
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CTCLoss(blank=0, reduction='mean')
        
        # 設置學習率調度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        # 訓練模型
        best_val_loss = float('inf')
        best_accuracy = 0.0
        best_model_path = os.path.join(fold_dir, "best_model_OCR_region2.pth")
        
        for epoch in range(args.epochs):
            # 訓練一個 epoch
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # 驗證
            val_loss, val_acc, val_targets, val_preds = validate_epoch(
                model, val_loader, criterion, device, idx_to_char
            )
            
            # 調整學習率
            scheduler.step(val_loss)
            
            # 打印進度
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"保存最佳模型 (精度: {best_accuracy:.4f})")
            
            # 保存一些錯誤預測的樣本進行分析
            if epoch == args.epochs - 1 or epoch % 10 == 0:
                errors = [(t, p) for t, p in zip(val_targets, val_preds) if t != p]
                if errors:
                    with open(os.path.join(fold_dir, f"errors_epoch_{epoch+1}.txt"), "w", encoding="utf-8") as f:
                        f.write("真實標籤\t預測標籤\n")
                        for true, pred in errors[:100]:  # 最多保存100個錯誤
                            f.write(f"{true}\t{pred}\n")
                            
                # 保存一些混淆矩陣分析
                confusion = {}
                for t, p in zip(val_targets, val_preds):
                    if t != p:
                        key = f"{t} -> {p}"
                        if key not in confusion:
                            confusion[key] = 0
                        confusion[key] += 1
                
                with open(os.path.join(fold_dir, f"confusion_epoch_{epoch+1}.txt"), "w", encoding="utf-8") as f:
                    f.write("錯誤類型\t計數\n")
                    for k, v in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"{k}\t{v}\n")
        
        # 記錄該折的性能
        fold_accuracies.append(best_accuracy)
        fold_losses.append(best_val_loss)
        best_model_paths.append(best_model_path)
        
        # 釋放內存
        del model, optimizer, scheduler, train_dataset, val_dataset, train_loader, val_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"第 {fold+1} 折完成，最佳驗證精度: {best_accuracy:.4f}")
    
    # 打印總結
    print("\n===== 交叉驗證結果 =====")
    for fold, (acc, loss) in enumerate(zip(fold_accuracies, fold_losses)):
        print(f"Fold {fold+1}: 精度 = {acc:.4f}, 損失 = {loss:.4f}")
    
    mean_acc = sum(fold_accuracies) / len(fold_accuracies)
    mean_loss = sum(fold_losses) / len(fold_losses)
    print(f"平均精度: {mean_acc:.4f}, 平均損失: {mean_loss:.4f}")
    
    # 保存集成信息
    ensemble_info = {
        "fold_accuracies": fold_accuracies,
        "fold_losses": fold_losses,
        "model_paths": [os.path.basename(p) for p in best_model_paths],
        "mean_accuracy": mean_acc,
        "mean_loss": mean_loss,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": len(all_chars),
        "completed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, "ensemble_info.json"), "w", encoding="utf-8") as f:
        json.dump(ensemble_info, f, ensure_ascii=False, indent=4)
    
    print(f"訓練完成，集成信息已保存到 {os.path.join(output_dir, 'ensemble_info.json')}")
    return ensemble_info

# -------------------------------
# 7. 预测函数（含帧变化检测）
# -------------------------------
def predict_with_ensemble(data_samples, transform, model_dir, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    data_samples.sort(key=lambda x: x[2])
    frame_paths = [sample[0] for sample in data_samples]
    
    # 帧变化检测
    if hasattr(args, "detect_changes") and args.detect_changes:
        print("检测帧变化...")
        roi = args.roi
        changed_indices = detect_frame_changes(frame_paths, threshold=args.similarity_threshold, region_of_interest=roi)
        print(f"检测到 {len(changed_indices)} 个变化帧，从总共 {len(frame_paths)} 帧中")
        data_samples = [data_samples[i] for i in changed_indices]
    
    # 准备数据集和数据加载器
    test_dataset = RegionOCRDatasetCTC(data_samples, transform=transform, preproc_fn=preprocess_image, char_to_idx=char_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_ctc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 查找模型文件夹
    fold_dirs = [d for d in os.listdir(model_dir) if d.startswith("fold_") and os.path.isdir(os.path.join(model_dir, d))]
    if not fold_dirs:
        print(f"错误: 在 {model_dir} 中没有找到模型文件夹")
        return None
    
    # 加载集成模型信息
    ensemble_info_path = os.path.join(model_dir, "ensemble_info.json")
    fold_accuracies = {}
    if os.path.exists(ensemble_info_path):
        with open(ensemble_info_path, "r", encoding="utf-8") as f:
            ensemble_info = json.load(f)
            if "fold_accuracies" in ensemble_info:
                fold_accuracies = {f"fold_{i}": acc for i, acc in enumerate(ensemble_info["fold_accuracies"])}
    
    # 加载字符映射
    mapping_path = os.path.join(model_dir, fold_dirs[0], "char_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
        idx_to_char_loaded = {int(k): v for k, v in mapping["idx_to_char"].items()}
    
    # 存储所有模型的预测结果
    all_fold_predictions = {}
    ground_truth = {}
    
    # 对每个模型进行预测
    for fold_id, fold_dir in enumerate(fold_dirs):
        fold_path = os.path.join(model_dir, fold_dir)
        model_path = os.path.join(fold_path, "best_model_OCR_region2.pth")
        
        try:
            # 加载模型
            model = HezarCRNNWrapper(vocab_size=n_class)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            
            # 使用模型进行预测
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Predicting with {fold_dir}"):
                    images, targets, target_lengths, frames = batch
                    images = images.to(device)
                    
                    # 前向传播
                    outputs = model(images)
                    
                    # 解码预测结果
                    preds = greedy_decode(outputs, idx_to_char_loaded)
                    
                    # 获取真实标签
                    start = 0
                    for i, (frame, pred, length) in enumerate(zip(frames, preds, target_lengths)):
                        frame_num = int(frame.item())
                        true_indices = targets[start:start+length].numpy()
                        true_label = ''.join([idx_to_char_loaded.get(int(idx), '') for idx in true_indices])
                        start += length
                        
                        if frame_num not in ground_truth:
                            ground_truth[frame_num] = true_label
                        
                        if frame_num not in all_fold_predictions:
                            all_fold_predictions[frame_num] = {}
                        
                        all_fold_predictions[frame_num][fold_dir] = pred
            
            print(f"完成了 {fold_dir} 的预测")
        
        except Exception as e:
            print(f"处理 {fold_dir} 时出错: {str(e)}")
            continue
    
    # 集成预测结果
    ensemble_method = args.ensemble_method if hasattr(args, "ensemble_method") else "weighted"
    final_predictions = ensemble_prediction(all_fold_predictions, ensemble_method=ensemble_method, fold_accuracies=fold_accuracies)
    
    # 保存预测结果
    save_prediction_results(final_predictions, output_dir, "Hezar-CRNN", ground_truth)
    
    # 生成结果数据框
    result_df = pd.DataFrame([
        {"frame": frame, "true_response": ground_truth.get(frame, ""), "pred_response": pred,
         "is_correct": ground_truth.get(frame, "") == pred}
        for frame, pred in final_predictions.items()
    ])
    
    # 可视化预测错误
    wrong_predictions = result_df[result_df["is_correct"] == False]
    if not wrong_predictions.empty:
        visualize_prediction_errors(wrong_predictions, data_samples, output_dir)
    
    return result_df

# -------------------------------
# 集成預測函數
# -------------------------------
def ensemble_prediction(all_fold_predictions, ensemble_method="majority", fold_accuracies=None):
    """
    集成多個模型的預測結果
    
    Args:
        all_fold_predictions: 字典的字典，第一層鍵為幀號，第二層鍵為模型名，值為預測結果
        ensemble_method: 集成方法，可選"majority"(多數投票),"weighted"(加權),"average"(平均)
        fold_accuracies: 字典，鍵為模型名，值為該模型的準確率（用於加權集成）
        
    Returns:
        final_predictions: 字典，鍵為幀號，值為最終預測結果
    """
    final_predictions = {}
    
    for frame_num in all_fold_predictions:
        fold_preds = all_fold_predictions[frame_num]
        
        if ensemble_method == "majority":
            # 多數投票
            votes = {}
            for fold, pred in fold_preds.items():
                if pred not in votes:
                    votes[pred] = 0
                votes[pred] += 1
                
            max_votes = max(votes.values())
            candidates = [pred for pred, count in votes.items() if count == max_votes]
            final_predictions[frame_num] = candidates[0]  # 如果票數相同，取第一個
            
        elif ensemble_method == "weighted" and fold_accuracies:
            # 加權投票
            weighted_votes = {}
            for fold, pred in fold_preds.items():
                weight = fold_accuracies.get(fold, 1.0)
                if pred not in weighted_votes:
                    weighted_votes[pred] = 0
                weighted_votes[pred] += weight
                
            final_predictions[frame_num] = max(weighted_votes.items(), key=lambda x: x[1])[0]
            
        else:
            # 默認使用多數投票
            votes = {}
            for fold, pred in fold_preds.items():
                if pred not in votes:
                    votes[pred] = 0
                votes[pred] += 1
                
            max_votes = max(votes.values())
            candidates = [pred for pred, count in votes.items() if count == max_votes]
            final_predictions[frame_num] = candidates[0]
    
    return final_predictions

# -------------------------------
# 保存預測結果
# -------------------------------
def save_prediction_results(predictions, output_dir, model_name, ground_truth=None):
    """
    保存預測結果到 CSV 文件並返回結果數據框
    
    Args:
        predictions: 字典，鍵為幀號，值為預測結果
        output_dir: 輸出目錄
        model_name: 模型名稱，用於文件命名
        ground_truth: 可選，字典，鍵為幀號，值為真實標籤
        
    Returns:
        result_df: 包含預測結果的數據框
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 準備結果
    results = []
    for frame, pred in predictions.items():
        result = {
            'frame': frame,
            'pred_response': pred
        }
        
        # 如果有真實標籤，添加對比信息
        if ground_truth and frame in ground_truth:
            true_text = ground_truth[frame]
            result['true_response'] = true_text
            result['is_correct'] = (pred == true_text)
        
        results.append(result)
    
    # 按幀號排序
    results.sort(key=lambda x: x['frame'])
    
    # 保存為CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(output_dir, f"{model_name}_predictions.csv"), index=False)
    
    # 如果有真實標籤，計算準確率
    if ground_truth:
        correct = sum(1 for r in results if 'is_correct' in r and r['is_correct'])
        accuracy = correct / len([r for r in results if 'is_correct' in r]) if [r for r in results if 'is_correct' in r] else 0
        print(f"{model_name} 準確率: {accuracy*100:.2f}%")
    
    return result_df

# -------------------------------
# 可視化預測錯誤
# -------------------------------
def visualize_prediction_errors(wrong_predictions, data_samples, output_dir):
    """
    可視化預測錯誤的樣本
    
    Args:
        wrong_predictions: 數據框，包含預測錯誤的樣本信息
        data_samples: 原始數據樣本列表，每個元素為 (image_path, response, frame_num)
        output_dir: 輸出目錄
    """
    # 創建輸出目錄
    error_vis_dir = os.path.join(output_dir, "error_visualization")
    os.makedirs(error_vis_dir, exist_ok=True)
    
    # 創建幀號到圖像路徑的映射
    frame_to_path = {frame_num: img_path for img_path, _, frame_num in data_samples}
    
    # 限制可視化的錯誤樣本數量
    max_samples = min(20, len(wrong_predictions))
    samples_to_vis = wrong_predictions.iloc[:max_samples]
    
    fig, axes = plt.subplots(max_samples, 1, figsize=(8, 4 * max_samples))
    if max_samples == 1:
        axes = [axes]
    
    for i, (_, row) in enumerate(samples_to_vis.iterrows()):
        frame = row['frame']
        true_text = row['true_response']
        pred_text = row['pred_response']
        
        if frame in frame_to_path:
            # 讀取圖像
            img_path = frame_to_path[frame]
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 顯示圖像和預測信息
                axes[i].imshow(img)
                axes[i].set_title(f"Frame {frame}\nTrue: '{true_text}' | Pred: '{pred_text}'")
                axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(error_vis_dir, "prediction_errors.png"), dpi=150)
    plt.close()
    
    # 保存詳細的錯誤分析
    wrong_predictions.to_csv(os.path.join(error_vis_dir, "error_details.csv"), index=False)
    
    # 統計錯誤類型
    error_types = defaultdict(int)
    for _, row in wrong_predictions.iterrows():
        true_text = row['true_response']
        pred_text = row['pred_response']
        error_type = f"{true_text} -> {pred_text}"
        error_types[error_type] += 1
    
    # 保存錯誤類型統計
    error_stats = pd.DataFrame([{"error_type": k, "count": v} for k, v in error_types.items()])
    error_stats = error_stats.sort_values("count", ascending=False)
    error_stats.to_csv(os.path.join(error_vis_dir, "error_statistics.csv"), index=False)
    
    # 可視化最常見的幾種錯誤類型
    top_errors = error_stats.head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="count", y="error_type", data=top_errors)
    plt.title("Top 10 Error Types")
    plt.tight_layout()
    plt.savefig(os.path.join(error_vis_dir, "top_errors.png"), dpi=150)
    plt.close()

# -------------------------------
# 幀變化檢測函數
# -------------------------------
def detect_frame_changes(frame_paths, threshold=0.9, region_of_interest=None):
    """
    檢測視頻幀之間的變化
    
    Args:
        frame_paths: 幀圖像路徑列表
        threshold: 相似度閾值，低於此值視為變化
        region_of_interest: 感興趣區域 (x, y, w, h)
        
    Returns:
        changed_indices: 檢測到變化的幀索引列表
    """
    if len(frame_paths) < 2:
        return list(range(len(frame_paths)))
    
    changed_indices = [0]  # 第一幀總是包含
    prev_img = None
    
    for i, path in enumerate(tqdm(frame_paths, desc="檢測幀變化")):
        # 讀取當前幀
        current_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if current_img is None:
            print(f"警告: 無法讀取圖像 {path}")
            continue
        
        # 提取感興趣區域
        if region_of_interest:
            x, y, w, h = region_of_interest
            current_roi = current_img[y:y+h, x:x+w]
        else:
            current_roi = current_img
        
        # 與前一幀比較
        if prev_img is not None:
            # 計算結構相似性
            similarity = cv2.matchTemplate(current_roi, prev_img, cv2.TM_CCOEFF_NORMED)
            similarity_score = np.max(similarity)
            
            # 如果相似度低於閾值，認為發生了變化
            if similarity_score < threshold:
                changed_indices.append(i)
        
        prev_img = current_roi
    
    return changed_indices

# -------------------------------
# 處理器和模型加載函數
# -------------------------------
def preprocess_image(image, target_size=(32, 128)):
    """
    預處理圖像為模型輸入格式
    
    Args:
        image: 輸入圖像，PIL 圖像或 numpy 數組
        target_size: 目標尺寸 (height, width)
        
    Returns:
        processed_img: 預處理後的圖像
    """
    if isinstance(image, str):
        # 如果輸入是路徑，讀取圖像
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if isinstance(image, np.ndarray):
        # 轉換為 PIL 圖像
        image = Image.fromarray(image)
    
    # 調整大小
    image = image.resize(target_size[::-1])  # PIL 使用 (width, height) 格式
    
    # 轉換為灰度圖
    if image.mode != 'L':
        image = image.convert('L')
    
    # 標準化
    image = np.array(image).astype(np.float32) / 255.0
    
    return image

# -------------------------------
# 主函數
# -------------------------------
def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='OCR 模型訓練與評估')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train',
                      help='訓練模式或預測模式')
    parser.add_argument('--jsonl', type=str, default='../data/2024-11-20_h/region2/region2.jsonl',
                      help='JSONL 數據文件路徑')
    parser.add_argument('--output-dir', type=str, default='simple_ocr_models3',
                      help='輸出目錄')
    parser.add_argument('--model-dir', type=str, default='simple_ocr_models3',
                      help='模型目錄(僅預測模式需要)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='學習率')
    parser.add_argument('--epochs', type=int, default=30,
                      help='訓練輪數')
    parser.add_argument('--seed', type=int, default=42,
                      help='隨機種子')
    parser.add_argument('--model-name', type=str, default="resnet18_bilstm",
                      help='骨幹網絡名稱')
    parser.add_argument('--detect-changes', action='store_true',
                      help='是否檢測幀變化')
    parser.add_argument('--similarity-threshold', type=float, default=0.95,
                      help='幀變化檢測的相似度閾值')
    parser.add_argument('--roi', type=str,
                      help='感興趣區域, 格式: x,y,w,h')
    parser.add_argument('--ensemble-method', type=str, default='weighted',
                      choices=['majority', 'weighted'],
                      help='集成方法')
    
    args = parser.parse_args()
    
    # 打印參數
    print("\n=== 訓練參數 ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("================\n")
    
    # 檢查 JSONL 文件是否存在
    if not os.path.exists(args.jsonl):
        print(f"錯誤: JSONL 文件不存在: {args.jsonl}")
        print("請確保提供正確的 JSONL 文件路徑，或檢查當前工作目錄。")
        print(f"當前工作目錄: {os.getcwd()}")
        print("尋找可用的 JSONL 文件...")
        found_files = []
        for root, dirs, files in os.walk('.', topdown=True):
            for name in files:
                if name.endswith('.jsonl'):
                    found_files.append(os.path.join(root, name))
        
        if found_files:
            print("找到以下 JSONL 文件:")
            for file in found_files[:10]:  # 最多顯示10個
                print(f"  - {file}")
            if len(found_files) > 10:
                print(f"  以及其他 {len(found_files)-10} 個文件...")
        else:
            print("未找到任何 JSONL 文件。")
        return
    
    # 解析 ROI
    if args.roi:
        try:
            args.roi = tuple(map(int, args.roi.split(',')))
            if len(args.roi) != 4:
                raise ValueError("ROI 格式應為 'x,y,w,h'")
        except:
            print(f"錯誤: 無效的 ROI 格式 '{args.roi}'，應為 'x,y,w,h'")
            return
    
    # 設置隨機種子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設置圖像轉換
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 載入數據
    print(f"從 {args.jsonl} 載入數據...")
    data_samples = load_data_from_jsonl(args.jsonl)
    
    # 訓練模式
    if args.mode == 'train':
        print("開始訓練...")
        train_with_cross_validation(data_samples, transform, args.output_dir, args)
    
    # 預測模式
    elif args.mode == 'predict':
        if not args.model_dir:
            print("錯誤: 預測模式需要指定模型目錄 (--model-dir)")
            return
        
        print("開始預測...")
        result_df = predict_with_ensemble(data_samples, transform, args.model_dir, args.output_dir, args)
        
        if result_df is not None:
            print(f"預測完成，結果已保存至 {args.output_dir}")
    
    print("程序執行完畢!")

if __name__ == "__main__":
    main()