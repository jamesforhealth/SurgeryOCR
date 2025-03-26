#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本程式用於對 region2 圖像進行分類，分類依據為 JSONL 文件中 response 所包含的數字個數（0～3）。
我們使用預訓練的 EfficientNet‑B0 模型進行 fine‑tune，並隨機切分資料集為訓練集與驗證集，
觀察分類效果，進而為後續針對不同數字個數的 OCR 策略提供依據。

執行範例:
    python test_classify_region2.py 
或自定義參數:
    python test_classify_region2.py --jsonl region2/region2.jsonl --base-dir 2024-11-20_h --output-dir region2_output --epochs 20 --batch-size 32 --lr 1e-3
"""

import os
import json
import re
import argparse
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. 自定義 Dataset
# -------------------------------
class RegionDataset(Dataset):
    def __init__(self, jsonl_path, base_dir="", transform=None):
        """
        讀取 JSONL 文件，每一行包含圖像路徑和 OCR response。
        標籤取 response 中數字的個數（0～3）。
        """
        self.samples = []  # 每個樣本：(image_path, label)
        self.transform = transform

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    rel_path = data.get('images', '')
                    # 若 base_dir 非空，組合路徑；否則直接使用 JSONL 中的路徑
                    img_path = os.path.join(base_dir, rel_path) if base_dir else rel_path
                    if os.path.exists(img_path):
                        response = data.get('response', '').strip()
                        # 計算 response 中數字的個數
                        digit_count = len(re.findall(r'\d', response))
                        # 為保證標籤在 0～3 之間，若超過 3 則取 3
                        label = min(digit_count, 3)
                        self.samples.append((img_path, label))
                    else:
                        print(f"警告: 找不到圖像文件 {img_path}")
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------------
# 2. 建立模型 (使用 EfficientNet‑B0 進行 finetune)
# -------------------------------
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientNetClassifier, self).__init__()
        # 載入預訓練 EfficientNet‑B0 模型
        self.model = models.efficientnet_b0(pretrained=True)
        # 替換分類頭：EfficientNet‑B0 預設分類頭輸出 1000 類
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# -------------------------------
# 3. 訓練與驗證函數
# -------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
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

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss, accuracy, all_labels, all_preds

# -------------------------------
# 4. 主程序
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Region2 Image Classification by Digit Count")
    parser.add_argument("--jsonl", type=str, default="region2/region2.jsonl", 
                        help="Path to the JSONL file (e.g., region2.jsonl)")
    parser.add_argument("--base-dir", type=str, default="", 
                        help="Base directory for image paths (leave empty if paths in JSONL are already complete)")
    parser.add_argument("--output-dir", type=str, default="region2_output", 
                        help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 設置隨機種子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 定義圖像前處理：使用 EfficientNet-B0 的標準 transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 載入資料集
    dataset = RegionDataset(args.jsonl, base_dir=args.base_dir, transform=transform)
    total_samples = len(dataset)
    # 隨機切分 50% 訓練, 50% 驗證
    train_size = total_samples // 2
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 建立模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetClassifier(num_classes=4)
    model = model.to(device)

    # 定義損失和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_acc = 0.0
    best_model_state = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 開始訓練
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    # 保存最佳模型
    model_path = os.path.join(args.output_dir, "best_classifier.pth")
    torch.save(best_model_state, model_path)
    print(f"最佳模型保存到: {model_path}")

    # 輸出分類報告
    print("分類報告:")
    print(classification_report(y_true, y_pred, digits=4))

    # 繪製訓練曲線
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
