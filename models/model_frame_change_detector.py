#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
帧变化检测器：使用自监督学习方法训练模型识别连续帧之间是否存在实质变化
正样本：连续两帧且OCR结果相同的帧对
负样本：连续两帧但OCR结果不同的帧对

使用方法:
    1. 训练模式:
       python model_frame_change_detector.py --mode train --jsonl data/2024-11-20_h/region2/region2.jsonl --output-dir change_detector_models --backbone cbam_resnet18 --loss-type contrastive
    
    2. 预测模式:
       python model_frame_change_detector.py --mode predict --jsonl data/2024-11-20_h/region2/region2_test.jsonl --output-dir change_predictions --model-path change_detector_models/best_model.pth --backbone cbam_resnet18 --loss-type contrastive
"""

import os
import json
import re
import argparse
import random
from collections import defaultdict
import time
import copy

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# -------------------------------
# CBAM 模块实现
# -------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out

def cbam_resnet18(pretrained=True):
    """构造带 CBAM 注意力模块的 ResNet18"""
    resnet18 = models.resnet18(pretrained=pretrained)
    # 对 ResNet18 的四个主要层分别添加 CBAM 模块
    resnet18.layer1 = nn.Sequential(
        resnet18.layer1,
        CBAM(64)
    )
    resnet18.layer2 = nn.Sequential(
        resnet18.layer2,
        CBAM(128)
    )
    resnet18.layer3 = nn.Sequential(
        resnet18.layer3,
        CBAM(256)
    )
    resnet18.layer4 = nn.Sequential(
        resnet18.layer4,
        CBAM(512)
    )
    return resnet18

# -------------------------------
# 数据集准备
# -------------------------------
class FrameChangeDataset(Dataset):
    """帧变化数据集"""
    def __init__(self, frame_pairs, transform=None, return_original=False):
        """
        Args:
            frame_pairs: 每个元素为 (frame1_path, frame2_path, label, frame_num)
            transform: 图像预处理转换
            return_original: 若 True，则返回原始 PIL 图像（用于预测时计算 SSIM）
        """
        self.frame_pairs = frame_pairs
        self.transform = transform
        self.return_original = return_original
        
    def __len__(self):
        return len(self.frame_pairs)
    
    def __getitem__(self, idx):
        frame1_rel_path, frame2_rel_path, label, frame_num = self.frame_pairs[idx]
        frame1_path = self._find_image_path(frame1_rel_path)
        frame2_path = self._find_image_path(frame2_rel_path)
        img1 = Image.open(frame1_path).convert("RGB")
        img2 = Image.open(frame2_path).convert("RGB")
        orig1, orig2 = img1.copy(), img2.copy()
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        if self.return_original:
            return img1, img2, torch.tensor(label, dtype=torch.float32), frame_num, (orig1, orig2)
        else:
            return img1, img2, torch.tensor(label, dtype=torch.float32), frame_num
    
    def _find_image_path(self, rel_path):
        possible_paths = [
            rel_path,
            os.path.join("data", rel_path),
            os.path.join("..", "data", rel_path),
            os.path.join("data", os.path.basename(rel_path)),
        ]
        video_match = re.search(r'([\w-]+)/region\d+/frame_\d+\.png', rel_path)
        if video_match:
            possible_paths.append(os.path.join("..", "data", rel_path))
        for path in possible_paths:
            if os.path.exists(path):
                return path
        print(f"警告: 找不到图像文件 {rel_path}")
        print(f"尝试的路径: {possible_paths}")
        print(f"当前工作目录: {os.getcwd()}")
        return rel_path

def create_frame_pairs(jsonl_path, balance_classes=True, negative_ratio=1.0):
    frames_data = {}
    jsonl_dir = os.path.dirname(jsonl_path) or "."
    print(f"从 {jsonl_path} 加载数据...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                rel_image_path = data.get('images', '')
                response = data.get('response', '').strip()
                if os.path.exists(rel_image_path):
                    image_path = rel_image_path
                elif os.path.exists(os.path.join("data", rel_image_path)):
                    image_path = os.path.join("data", rel_image_path)
                elif os.path.exists(os.path.join("..", "data", rel_image_path)):
                    image_path = os.path.join("..", "data", rel_image_path)
                else:
                    image_path = rel_image_path
                frame_match = re.search(r'frame_(\d+)', rel_image_path)
                if frame_match:
                    frame_num = int(frame_match.group(1))
                    frames_data[frame_num] = (image_path, response)
            except json.JSONDecodeError:
                continue
    sorted_frames = sorted(list(frames_data.keys()))
    frame_pairs = []
    for i in range(len(sorted_frames) - 1):
        current_frame = sorted_frames[i]
        next_frame = sorted_frames[i + 1]
        if next_frame - current_frame > 5:
            continue
        current_img_path, current_response = frames_data[current_frame]
        next_img_path, next_response = frames_data[next_frame]
        is_unchanged = 1 if current_response == next_response else 0
        frame_pairs.append((current_img_path, next_img_path, is_unchanged, current_frame))
    positive_pairs = [pair for pair in frame_pairs if pair[2] == 1]
    negative_pairs = [pair for pair in frame_pairs if pair[2] == 0]
    print(f"原始数据集: 正样本 {len(positive_pairs)}个 ({len(positive_pairs)/len(frame_pairs)*100:.4f}%), "
          f"负样本 {len(negative_pairs)}个 ({len(negative_pairs)/len(frame_pairs)*100:.4f}%)")
    if balance_classes and len(negative_pairs) > 0:
        target_neg_count = int(len(positive_pairs) * negative_ratio)
        if len(negative_pairs) < target_neg_count:
            oversampled_neg = []
            for _ in range(target_neg_count // len(negative_pairs)):
                oversampled_neg.extend(negative_pairs)
            oversampled_neg.extend(random.sample(negative_pairs, target_neg_count % len(negative_pairs)))
            balanced_pairs = positive_pairs + oversampled_neg
        else:
            if negative_ratio < 1.0:
                sampled_pos = random.sample(positive_pairs, int(len(negative_pairs) / negative_ratio))
                balanced_pairs = sampled_pos + negative_pairs
            else:
                sampled_neg = random.sample(negative_pairs, target_neg_count)
                balanced_pairs = positive_pairs + sampled_neg
        print(f"平衡后数据集: 正样本 {len([p for p in balanced_pairs if p[2]==1])}个, "
              f"负样本 {len([p for p in balanced_pairs if p[2]==0])}个")
        return balanced_pairs
    return frame_pairs

# -------------------------------
# 损失函数定义
# -------------------------------
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=8.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    def forward(self, outputs, targets):
        loss = 0
        for i in range(len(outputs)):
            if targets[i] == 1:
                loss += self.pos_weight * (-torch.log(outputs[i] + 1e-7))
            else:
                loss += self.neg_weight * (-torch.log(1 - outputs[i] + 1e-7))
        return loss / len(outputs)

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss:
      对于成对输入，若标签为1（代表无变化，相似）则希望距离小，
      若标签为0（代表有变化，不相似）则距离至少大于 margin。
    注意：标准公式通常将相似标签设为0，因此此处将标签转换为 standard_label = 1 - label。
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, emb1, emb2, label):
        standard_label = 1 - label  # 使得 0 表示相似，1 表示不相似
        distance = F.pairwise_distance(emb1, emb2)
        loss = torch.mean(0.5 * (1 - standard_label) * distance**2 + 
                          0.5 * standard_label * torch.clamp(self.margin - distance, min=0.0)**2)
        return loss

# TripletLoss 与 ContrastiveLoss 的主要区别：
# - ContrastiveLoss 使用样本对进行训练，目标是使相似样本之间距离小，不相似的样本距离大于 margin。
# - TripletLoss 需要构造 (anchor, positive, negative) 三元组，目标是保证 anchor 与 positive 的距离比 anchor 与 negative 的距离小至少 margin。
# 由于本数据集以成对数据为主，这里直接集成 ContrastiveLoss；如有需要，可扩展 TripletLoss 模块。

# -------------------------------
# 模型定义（支持 BCE 与 Contrastive Loss 两种训练模式）
# -------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self, pretrained=True, feature_dim=128, loss_type="BCE", backbone="resnet18"):
        """
        Args:
            loss_type: "BCE" 或 "contrastive"
            backbone: "resnet18" 或 "cbam_resnet18"
        """
        super(SiameseNetwork, self).__init__()
        self.loss_type = loss_type
        if backbone == "cbam_resnet18":
            backbone_model = cbam_resnet18(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
        else:
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True)
        )
        if loss_type == "BCE":
            self.prediction_layer = nn.Sequential(
                nn.Linear(feature_dim, 1),
                nn.Sigmoid()
            )
    def forward_one(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        embedding = self.fc(features)
        return embedding
    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        if self.loss_type == "BCE":
            diff = torch.abs(emb1 - emb2)
            score = self.prediction_layer(diff)
            return score
        elif self.loss_type == "contrastive":
            return emb1, emb2

# -------------------------------
# 训练函数（根据损失类型区分处理）
# -------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                num_epochs=25, patience=5, threshold=0.5):
    model = model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 40)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_train_preds = []
        all_train_labels = []
        all_train_probs = []
        
        for batch in tqdm(train_loader, desc="Training"):
            if len(batch) == 4:
                inputs1, inputs2, labels, _ = batch
            else:
                inputs1, inputs2, labels = batch
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            if model.loss_type == "contrastive":
                emb1, emb2 = model(inputs1, inputs2)
                loss = criterion(emb1, emb2, labels)
                distance = F.pairwise_distance(emb1, emb2)
                # 若距离小于阈值，则预测为相似（1=无变化）
                predicted = (distance < threshold).float()
                all_train_probs.extend(distance.detach().cpu().numpy())
            else:
                outputs = model(inputs1, inputs2)
                loss = criterion(outputs, labels)
                predicted = (outputs > threshold).float()
                all_train_probs.extend(outputs.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        train_loss = running_loss / len(train_loader)
        train_acc = (correct / total) * 100
        train_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
        try:
            train_auc = roc_auc_score(all_train_labels, all_train_probs)
        except:
            train_auc = 0.5
        neg_mask = np.array(all_train_labels) == 0
        train_neg_recall = np.mean((np.array(all_train_preds) == 0)[neg_mask]) if neg_mask.sum() > 0 else 0.0
        
        val_metrics = evaluate_model(model, val_loader, criterion, device, threshold)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        history['train_auc'].append(train_auc)
        history['train_neg_recall'].append(train_neg_recall)
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_neg_recall'].append(val_metrics['neg_recall'])
        
        print(f"训练损失: {train_loss:.4f}, 准确率: {train_acc*100:.2f}%")
        print(f"验证损失: {val_metrics['loss']:.4f}, 准确率: {val_metrics['accuracy']*100:.2f}%")
        print(f"无变化类召回率: {val_metrics['pos_recall']:.4f}, 有变化类召回率: {val_metrics['neg_recall']:.4f}")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n早停! 验证损失 {patience} 轮未改善")
                break
                
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    
    return model, dict(history)

# -------------------------------
# 添加自定义合并函数
# -------------------------------
def custom_collate_fn(batch):
    """自定义合并函数，用于处理包含PIL图像的批次数据"""
    frames1 = []
    frames2 = []
    labels = []
    frame_nums = []
    orig1_list = []
    orig2_list = []
    
    for item in batch:
        if len(item) == 6:  # 带原始图像的格式
            frame1, frame2, label, frame_num, orig1, orig2 = item
            orig1_list.append(orig1)
            orig2_list.append(orig2)
        elif len(item) == 4:  # 带帧号但无原始图像的格式
            frame1, frame2, label, frame_num = item
            orig1_list.append(None)
            orig2_list.append(None)
        else:  # 基本格式
            frame1, frame2, label = item
            frame_num = 0  # 默认帧号
            orig1_list.append(None)
            orig2_list.append(None)
            
        frames1.append(frame1)
        frames2.append(frame2)
        labels.append(label)
        frame_nums.append(frame_num)
    
    # 将张量数据进行合并
    frames1 = torch.stack(frames1)
    frames2 = torch.stack(frames2)
    labels = torch.tensor(labels)
    
    return frames1, frames2, labels, frame_nums, orig1_list, orig2_list

def predict_changes(model, test_loader, device, threshold=0.5, model_high=0.3, model_low=0.7, ssim_threshold=0.95):
    """预测帧变化"""
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # 解包数据批次，现在预期有6个返回值
            inputs1, inputs2, _, frame_nums, orig1_list, orig2_list = batch
            
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            
            if model.loss_type == "contrastive":
                emb1, emb2 = model(inputs1, inputs2)
                distance = F.pairwise_distance(emb1, emb2)
            else:
                outputs = model(inputs1, inputs2)
                
            for i, frame_num in enumerate(frame_nums):
                if orig1_list[i] is not None and orig2_list[i] is not None:
                    ssim_score = ssim(cv2.cvtColor(np.array(orig1_list[i]), cv2.COLOR_RGB2GRAY),
                                      cv2.cvtColor(np.array(orig2_list[i]), cv2.COLOR_RGB2GRAY),
                                      full=True)[0]
                    if model.loss_type == "contrastive":
                        model_output = distance[i].item()
                        # 双门槛策略
                        if model_output > model_low and ssim_score < ssim_threshold:
                            final_pred = 0
                        elif model_output < model_high and ssim_score > ssim_threshold:
                            final_pred = 1
                        else:
                            final_pred = 1 if model_output < threshold else 0
                    else:
                        model_output = outputs[i].item()
                        if model_output < model_low and ssim_score < ssim_threshold:
                            final_pred = 0
                        elif model_output > model_high and ssim_score > ssim_threshold:
                            final_pred = 1
                        else:
                            final_pred = 1 if model_output > threshold else 0
                else:
                    if model.loss_type == "contrastive":
                        final_pred = 1 if distance[i].item() < threshold else 0
                    else:
                        final_pred = 1 if outputs[i].item() > threshold else 0
                predictions[int(frame_num)] = bool(final_pred)
    return predictions

# -------------------------------
# 可视化函数（略，保持原有代码）
# -------------------------------
def plot_training_history(history, output_dir):
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 无法设置中文字体")
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='训练F1')
    plt.plot(history['val_f1'], label='验证F1')
    plt.title('F1分数曲线')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 4)
    plt.plot(history['train_neg_recall'], label='训练负样本召回率')
    plt.plot(history['val_neg_recall'], label='验证负样本召回率')
    plt.title('负样本召回率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('召回率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_precision'], label='训练精确率')
    plt.plot(history['val_precision'], label='验证精确率')
    plt.title('精确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('精确率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 2)
    plt.plot(history['train_recall'], label='训练召回率')
    plt.plot(history['val_recall'], label='验证召回率')
    plt.title('召回率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('召回率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 3)
    plt.plot(history['train_auc'], label='训练AUC')
    plt.plot(history['val_auc'], label='验证AUC')
    plt.title('AUC曲线')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 4)
    metrics = ['acc', 'precision', 'recall', 'f1', 'neg_recall']
    values = [history['val_acc'][-1]/100, history['val_precision'][-1], 
              history['val_recall'][-1], history['val_f1'][-1], history['val_neg_recall'][-1]]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    plt.bar(metrics, values, color=colors)
    plt.title('最终模型性能')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v+0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_metrics.png'))
    plt.close()

def visualize_predictions(predictions, output_dir, ground_truth=None):
    """可视化预测结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体支持
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

    # 保存预测结果为文本文件
    with open(os.path.join(output_dir, 'predictions.txt'), 'w', encoding='utf-8') as f:
        for frame_num, is_unchanged in sorted(predictions.items()):
            f.write(f"帧 {frame_num}: {'无变化' if is_unchanged else '有变化'}\n")
    
    # 如果有真实标签，计算并显示指标
    if ground_truth is not None:
        common_frames = set(predictions.keys()) & set(ground_truth.keys())
        if common_frames:
            y_true = [ground_truth[f] for f in common_frames]
            y_pred = [predictions[f] for f in common_frames]
            
            # 由于数据是布尔值，需要转换为整数计算指标
            y_true_int = [1 if x else 0 for x in y_true]
            y_pred_int = [1 if x else 0 for x in y_pred]
            
            acc = accuracy_score(y_true_int, y_pred_int)
            prec = precision_score(y_true_int, y_pred_int, zero_division=1)
            recall = recall_score(y_true_int, y_pred_int, zero_division=1)
            f1 = f1_score(y_true_int, y_pred_int, zero_division=1)
            
            print(f"准确率: {acc*100:.4f}%")
            print(f"精确率: {prec:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
            
            # 保存指标到文件，指定UTF-8编码
            with open(os.path.join(output_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
                f.write(f"准确率: {acc*100:.4f}%\n")
                f.write(f"精确率: {prec:.4f}\n")
                f.write(f"召回率: {recall:.4f}\n")
                f.write(f"F1分数: {f1:.4f}\n")

def evaluate_model(model, data_loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 4:
                inputs1, inputs2, labels, _ = batch
            else:
                inputs1, inputs2, labels = batch
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            labels = labels.to(device).float()
            if model.loss_type == "contrastive":
                emb1, emb2 = model(inputs1, inputs2)
                outputs = F.pairwise_distance(emb1, emb2)
                loss = criterion(emb1, emb2, labels)
                predicted = (outputs < threshold).float()
            else:
                outputs = model(inputs1, inputs2)
                loss = criterion(outputs, labels)
                predicted = (outputs > threshold).float()
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    pos_precision = precision_score(all_labels, all_preds, pos_label=1)
    pos_recall = recall_score(all_labels, all_preds, pos_label=1)
    neg_precision = precision_score(all_labels, all_preds, pos_label=0)
    neg_recall = recall_score(all_labels, all_preds, pos_label=0)
    print(f"无变化类(正样本) - 精确率: {pos_precision:.4f}, 召回率: {pos_recall:.4f}")
    print(f"有变化类(负样本) - 精确率: {neg_precision:.4f}, 召回率: {neg_recall:.4f}")
    return {
        'loss': running_loss / len(data_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'pos_precision': pos_precision,
        'pos_recall': pos_recall,
        'neg_precision': neg_precision,
        'neg_recall': neg_recall
    }

# -------------------------------
# 主函数
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='帧变化检测模型')
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'predict'], 
                        help='运行模式: train 或 predict')
    parser.add_argument('--jsonl', type=str, default="../data/2024-11-20_h/region2/region2.jsonl", help='JSONL文件路径')
    parser.add_argument('--output-dir', type=str, default="frame_change_detector", help='输出目录')
    parser.add_argument('--model-path', type=str, default="frame_change_detector/frame_change_detector.pth", help='模型路径 (用于预测模式)')
    
    parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    parser.add_argument('--epochs', type=int, default=600, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    
    parser.add_argument('--balance-classes', action='store_true', help='是否平衡正负样本')
    parser.add_argument('--negative-ratio', type=float, default=1.0, help='负样本:正样本目标比例')
    parser.add_argument('--pos-weight', type=float, default=1.0, help='正样本权重')
    parser.add_argument('--neg-weight', type=float, default=8.0, help='负样本权重')
    
    parser.add_argument('--threshold', type=float, default=0.5, help='变化判断阈值')
    parser.add_argument('--loss-type', type=str, default="contrastive", choices=["BCE", "contrastive", "triplet"], help='损失函数类型')
    parser.add_argument('--backbone', type=str, default="resnet18", choices=["resnet18", "cbam_resnet18"], help='backbone 选择')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frame_pairs = create_frame_pairs(args.jsonl, balance_classes=args.balance_classes, 
                                     negative_ratio=args.negative_ratio)
    
    if args.mode == 'train':
        train_pairs, val_pairs = train_test_split(frame_pairs, test_size=0.2, random_state=42, 
                                                  stratify=[p[2] for p in frame_pairs])
        train_dataset = FrameChangeDataset(train_pairs, transform=transform, return_original=False)
        val_dataset = FrameChangeDataset(val_pairs, transform=transform, return_original=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        model = SiameseNetwork(pretrained=True, loss_type=args.loss_type, backbone=args.backbone)
        #sum of params
        print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")
        if args.loss_type == "BCE":
            criterion = WeightedBCELoss(pos_weight=args.pos_weight, neg_weight=args.neg_weight)
        elif args.loss_type == "contrastive":
            criterion = ContrastiveLoss(margin=1.0)
        else:
            raise NotImplementedError("Triplet loss 需构造三元组，未在本代码中实现。")
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        print("开始训练...")
        start_time = time.time()
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                                     num_epochs=args.epochs, patience=args.patience, threshold=args.threshold)
        training_time = time.time() - start_time
        print(f"训练完成! 耗时: {training_time:.4f} 秒")
        model_save_path = os.path.join(args.output_dir, 'frame_change_detector.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"模型保存至 {model_save_path}")
        plot_training_history(history, args.output_dir)
        print("在验证集上评估模型...")
        predictions = predict_changes(model, val_loader, device, threshold=args.threshold)
        ground_truth = {frame_num: bool(label) for _, _, label, frame_num in val_pairs}
        visualize_predictions(predictions, args.output_dir, ground_truth)
        
    elif args.mode == 'predict':
        if not args.model_path:
            raise ValueError("预测模式需要指定模型路径 (--model-path)")
        model = SiameseNetwork(pretrained=False, loss_type=args.loss_type, backbone=args.backbone)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        test_dataset = FrameChangeDataset(frame_pairs, transform=transform, return_original=True)
        # 添加自定义collate_fn
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
        print("开始预测...")
        predictions = predict_changes(model, test_loader, device, threshold=args.threshold)
        visualize_predictions(predictions, args.output_dir)
        pred_df = pd.DataFrame([(frame, int(is_unchanged)) for frame, is_unchanged in predictions.items()],
                               columns=['frame', 'is_unchanged'])
        pred_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
        print(f"预测结果已保存至 {os.path.join(args.output_dir, 'predictions.csv')}")
        change_frames = [frame for frame, is_unchanged in predictions.items() if not is_unchanged]
        change_frames.sort()
        print(f"检测到 {len(change_frames)} 个变化帧:")
        for i, frame in enumerate(change_frames[:20]):
            print(f"  - {frame}", end=", " if i < len(change_frames[:20])-1 else "\n")
        if len(change_frames) > 20:
            print(f"  ... 以及 {len(change_frames)-20} 个更多的变化帧")

if __name__ == "__main__":
    main()
