# models/finetune_surgery_crnn.py
import os
#openMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import CTCLoss
import random
import json
import torch.nn.functional as F

# 從新的 pretrain_crnn 導入模型和預訓練配置
from pretrain_crnn import CRNN, PretrainConfig
# 導入數據增強 (如果需要)
try:
    from dataset_augmentations import OCRDataAugmentation, set_seed, FinetuneAugmentedDataset
except ImportError:
    print("警告: 無法導入 dataset_augmentations。如果需要數據增強，請確保文件存在。")
    # 定義佔位符或引發錯誤，取決於是否嚴格要求增強
    class OCRDataAugmentation: pass
    class FinetuneAugmentedDataset(Dataset): pass
    def set_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# --- 配置 ---
class FinetuneConfig:
    # --- 路徑 ---
    # 使用 PretrainConfig 中的模型保存路徑作為預訓練模型路徑的默認值
    pretrain_model_path = os.path.join(PretrainConfig.model_save_path, 'best_crnn_model.pth')
    surgery_data_dir = "../data/"  # 指向包含 JSONL 和圖像的目錄
    surgery_jsonl_filename = "2024-11-20_h/region2/region2.jsonl" # JSONL 文件名
    finetune_model_save_path = "./finetuned_models_surgery" # 微調後模型保存路徑 (可以改個名區分)

    # --- 新字符集 ---
    # 包含 0-9, 負號 '-', 空格 ' '
    characters = '0123456789- '
    num_classes = len(characters) + 1  # 字符數 + 1 (CTC 空白符)
    ctc_blank_char_index = len(characters) # 將空白符放在最後

    # --- 模型與圖像參數 (應與預訓練保持一致或根據需要調整) ---
    input_channels = PretrainConfig.input_channels # 灰度圖
    hidden_size = PretrainConfig.hidden_size # 與預訓練模型一致
    img_height = PretrainConfig.img_height
    img_width = PretrainConfig.img_width
    max_label_length = 30 # 根據手術數據中最長標籤調整 (可能需要增加)

    # --- 微調參數 ---
    batch_size = 32   # 微調時通常使用較小的 batch size
    epochs = 50       # 微調所需的 epoch 通常較少，可以先設少一點
    learning_rate = 5e-5 # 微調時使用更小的學習率
    validation_split = 0.15 # 可以稍微增加驗證集比例
    freeze_cnn = False # 是否凍結 CNN 層 (可以先嘗試 False)
    early_stopping_patience = 10 # 早停耐心

    # --- 數據增強 ---
    use_augmentation = False # 微調時通常也建議使用增強
    augmentation_multiplier = 2 # 微調時增強倍數可以少一些
    augmentation_prob = 0.6

    # --- 設備 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = FinetuneConfig()
set_seed(42) # 設置隨機種子

# 創建字符映射
char_to_idx = {char: idx for idx, char in enumerate(config.characters)}
idx_to_char = {idx: char for idx, char in enumerate(config.characters)}
print(f"字符集: '{config.characters}'")
print(f"字符到索引映射: {char_to_idx}")
print(f"CTC 空白符索引: {config.ctc_blank_char_index}")
print(f"總類別數 (含空白符): {config.num_classes}")

# 確保保存路徑存在
os.makedirs(config.finetune_model_save_path, exist_ok=True)

# --- 手術數據集類 (修改以讀取 JSONL) ---
class SurgeryDataset(Dataset):
    def __init__(self, root_dir, jsonl_filename, char_map, max_len, transform=None):
        self.root_dir = root_dir
        self.jsonl_path = os.path.join(root_dir, jsonl_filename)
        self.char_map = char_map
        self.max_len = max_len
        self.transform = transform # 基礎轉換或 None
        self.data = []

        print(f"嘗試從 {self.jsonl_path} 加載數據...")
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        # 假設 JSONL 每行有 'images' 和 'response' 鍵
                        if 'images' in item and 'response' in item:
                            img_rel_path = item['images']
                            # 確保 response 是字符串，處理 None 或數字
                            label_str = str(item['response']) if item['response'] is not None else ""

                            # 檢查圖像文件是否存在
                            img_full_path = os.path.join(self.root_dir, img_rel_path)
                            if os.path.exists(img_full_path):
                                self.data.append({'images': img_rel_path, 'label': label_str})
                            else:
                                print(f"警告: 圖像文件未找到 {img_full_path}，已跳過此條目。")
                        else:
                             print(f"警告: JSONL 行缺少 'images' 或 'response' 鍵: {line.strip()}")

                    except json.JSONDecodeError:
                        print(f"警告: 無法解析 JSONL 行: {line.strip()}")
            print(f"從 {self.jsonl_path} 加載了 {len(self.data)} 個有效樣本")
        except FileNotFoundError:
            print(f"錯誤: 在 {root_dir} 中未找到 {jsonl_filename}")
            # self.data 保持為空列表
        except Exception as e:
             print(f"讀取 JSONL 文件時發生錯誤: {e}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
             raise IndexError("索引超出範圍")
        item = self.data[idx]
        img_rel_path = item['images']
        label_str = item['label']
        img_full_path = os.path.join(self.root_dir, img_rel_path)

        try:
            image = Image.open(img_full_path).convert('L') # 轉為灰度圖
        except FileNotFoundError:
            print(f"警告: 圖像文件未找到 {img_full_path}")
            # 返回一個佔位符圖像和空標籤
            image = Image.new('L', (config.img_width, config.img_height), color=128)
            label_str = ""
        except Exception as e:
            print(f"警告: 加載圖像時出錯 {img_full_path}: {e}")
            image = Image.new('L', (config.img_width, config.img_height), color=128)
            label_str = ""


        # 將標籤字符串轉換為索引列表
        label = []
        for char in label_str:
            if char in self.char_map:
                label.append(self.char_map[char])
            else:
                # 忽略未知字符，但可以打印警告
                # print(f"警告: 標籤 '{label_str}' 中包含未知字符 '{char}'，已忽略。")
                pass # 或者你可以選擇映射到一個特殊符號

        # 限制標籤長度
        label = label[:self.max_len]
        label_length = len(label)

        # 根據是否使用數據增強返回不同格式
        if self.transform:
             # 如果 transform 存在 (基礎轉換)，應用它並返回 Tensor
             image = self.transform(image)
             return image, torch.LongTensor(label), label_length
        else:
             # 如果 transform 為 None (用於增強)，返回 PIL Image 和標籤字符串
             return image, label_str


# --- 數據預處理 (基礎轉換，用於驗證集或不增強的訓練集) ---
base_transform = transforms.Compose([
    transforms.Resize((config.img_height, config.img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # 歸一化到 [-1, 1]
])

# --- Collate Function (保持不變) ---
def collate_fn_finetune(batch):
    # 過濾掉可能為 None 的樣本 (例如圖像加載失敗)
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch:
        return None, None, None # 或者返回空的 tensors

    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.IntTensor(label_lengths)

    max_len_batch = max(label_lengths) if label_lengths else 0

    padded_labels = torch.full((len(labels), max_len_batch),
                               fill_value=config.ctc_blank_char_index,
                               dtype=torch.long)

    for i, label in enumerate(labels):
        if len(label) > 0:
             padded_labels[i, :len(label)] = label

    return images, padded_labels, target_lengths


# --- 加載數據 ---
print("加載手術數據集...")
# 初始加載時不應用 transform，因為增強器會處理
full_dataset = SurgeryDataset(
    config.surgery_data_dir,
    config.surgery_jsonl_filename,
    char_to_idx,
    config.max_label_length,
    transform=None # 讓 FinetuneAugmentedDataset 或後續步驟處理 transform
)

if len(full_dataset) == 0:
    print("錯誤: 未能加載任何手術數據，請檢查數據路徑和 JSONL 文件。")
    exit()

# 拆分訓練集和驗證集
val_size = int(len(full_dataset) * config.validation_split)
# 確保 val_size 至少為 1 (如果數據集很小)
val_size = max(1, val_size)
train_size = len(full_dataset) - val_size

if train_size <= 0:
     print("錯誤: 數據集太小，無法劃分訓練集和驗證集。")
     exit()

train_subset_indices, val_subset_indices = random_split(range(len(full_dataset)), [train_size, val_size])

# --- 數據增強器 (如果啟用) ---
augmenter = None
train_dataset = None
val_dataset = None

if config.use_augmentation:
    print("創建數據增強器...")
    try:
        augmenter = OCRDataAugmentation(
            img_height=config.img_height,
            img_width=config.img_width,
            prob=config.augmentation_prob,
            grayscale=True # 確保增強器處理灰度圖
        )
        # 創建增強後的訓練集
        train_dataset = FinetuneAugmentedDataset(
            dataset=full_dataset,
            indices=train_subset_indices.indices,
            char_map=char_to_idx,
            max_len=config.max_label_length,
            augmenter=augmenter,
            multiplier=config.augmentation_multiplier,
            base_transform=base_transform # 基礎轉換應用於原始樣本和增強樣本
        )
    except NameError:
        print("警告: OCRDataAugmentation 未定義，無法使用數據增強。")
        config.use_augmentation = False # 禁用增強

# 如果不使用增強，或者增強器創建失敗
if not config.use_augmentation:
    print("不使用數據增強。")
    # 創建普通的 Subset，並應用基礎 transform
    from torch.utils.data import Subset
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            img, label, length = self.subset[index] # 假設 SurgeryDataset 返回處理好的 Tensor
            # 如果 SurgeryDataset 返回 PIL Image, label_str
            # img_pil, label_str = self.subset[index]
            # img = self.transform(img_pil)
            # label = torch.LongTensor([char_to_idx[c] for c in label_str if c in char_to_idx][:config.max_label_length])
            # length = len(label)
            # --- 需要確保 SurgeryDataset 返回的是可以直接 collate 的格式 ---
            # 假設 SurgeryDataset 已經應用了 base_transform
            return img, label, length # 直接返回
        def __len__(self):
            return len(self.subset)

    # 創建 Subset 實例
    train_subset = Subset(full_dataset, train_subset_indices.indices)
    val_subset = Subset(full_dataset, val_subset_indices.indices)

    # 應用基礎 transform (如果 SurgeryDataset 沒有應用)
    # 需要修改 SurgeryDataset 的 __getitem__ 返回 PIL Image 和 label_str
    # 然後在這裡應用 transform
    # --- 假設 SurgeryDataset 已經應用了 base_transform ---
    # 創建 Dataset 實例 (這裡假設 SurgeryDataset 返回的是 Tensor)
    class SimpleDatasetWrapper(Dataset):
        def __init__(self, subset, transform):
             self.subset = subset
             self.transform = transform # 這裡的 transform 應該是 base_transform
        def __getitem__(self, index):
             # 原始數據獲取 - 假設 self.subset.dataset.data[index] 是字典或者元組
             item = self.subset.dataset.data[self.subset.indices[index]]
             
             # 獲取圖像路徑和標籤 - 假設 item 是字典格式
             if isinstance(item, dict):
                 img_rel_path = item['images'] # 或者其他圖像路徑的鍵名
                 label_str = item['label']
                 img_full_path = os.path.join(self.subset.dataset.root_dir, img_rel_path)
             else:
                 # 假設 item 是元組 (img_path, label)
                 img_rel_path, label_str = item
                 img_full_path = os.path.join(self.subset.dataset.root_dir, img_rel_path)
             
             # 打印一些信息以便調試 - 可以在第一個 batch 後刪除
             if index == 0:
                 print(f"圖像路徑示例: {img_full_path}")
                 print(f"標籤示例: {label_str}")
             
             # 嘗試加載圖像
             try:
                 img_pil = Image.open(img_full_path).convert('L')
             except Exception as e:
                 print(f"無法加載圖像 {img_full_path}: {e}")
                 # 創建一個空白圖像作為替代
                 img_pil = Image.new('L', (self.subset.dataset.config.img_width, 
                                         self.subset.dataset.config.img_height), 128)
                 label_str = ""

             img = self.transform(img_pil)
             
             # 轉換標籤為索引
             label = []
             for char in label_str:
                 if char in self.subset.dataset.char_map:
                     label.append(self.subset.dataset.char_map[char])
             label = label[:self.subset.dataset.max_len]
             label_length = len(label)
             
             return img, torch.LongTensor(label), label_length

        def __len__(self):
             return len(self.subset)

    train_dataset = SimpleDatasetWrapper(train_subset, base_transform)
    val_dataset = SimpleDatasetWrapper(val_subset, base_transform)


else: # 如果使用了增強
    # 驗證集不使用乘數增強，只應用基礎轉換
    from torch.utils.data import Subset
    class ValidationDatasetWrapper(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform # 這裡應該是 base_transform

        def __getitem__(self, index):
            # 從原始數據集獲取數據
            item = self.dataset.data[self.indices[index]]
            img_rel_path = item['images']
            label_str = item['label']
            img_full_path = os.path.join(self.dataset.root_dir, img_rel_path)

            try:
                image = Image.open(img_full_path).convert('L')
            except Exception as e:
                print(f"警告: 加載驗證圖像時出錯 {img_full_path}: {e}")
                image = Image.new('L', (config.img_width, config.img_height), color=128)
                label_str = ""

            # 應用基礎轉換
            image = self.transform(image)

            # 轉換標籤
            label = []
            for char in label_str:
                if char in self.dataset.char_map:
                    label.append(self.dataset.char_map[char])
            label = label[:self.dataset.max_len]
            label_length = len(label)

            return image, torch.LongTensor(label), label_length

        def __len__(self):
            return len(self.indices)

    val_dataset = ValidationDatasetWrapper(full_dataset, val_subset_indices.indices, base_transform)


print(f"原始訓練樣本數: {train_size}")
print(f"原始驗證樣本數: {val_size}")
print(f"最終訓練集大小: {len(train_dataset)}")
print(f"最終驗證集大小: {len(val_dataset)}")

# 創建 DataLoader
# 檢查 dataset 是否為空
if len(train_dataset) == 0 or len(val_dataset) == 0:
    print("錯誤：訓練集或驗證集為空，無法創建 DataLoader。")
    exit()

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn_finetune,
    num_workers=0,  # 改為 0，避免多進程問題
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collate_fn_finetune,
    num_workers=0,  # 改為 0，避免多進程問題
    pin_memory=True
)

# --- 加載預訓練模型並修改最後一層 ---
print("加載預訓練模型...")
# 首先使用預訓練時的配置創建模型結構
# 假設 PretrainConfig 存在且定義了 num_classes
try:
    pretrain_config = PretrainConfig()
    pretrain_num_classes = pretrain_config.num_classes
except NameError:
    print("警告: 無法加載 PretrainConfig，將假設預訓練類別數為 11 (0-9 + blank)。")
    pretrain_num_classes = 11 # 假設值

# 使用配置創建模型實例
model = CRNN(
    input_channels=config.input_channels,
    hidden_size=config.hidden_size,
    num_classes=pretrain_num_classes # 先使用預訓練的類別數
).to(config.device)

# 加載預訓練權重
try:
    checkpoint = torch.load(config.pretrain_model_path, map_location=config.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # 可能是直接保存的模型狀態
        model.load_state_dict(checkpoint, strict=False)
    print(f"成功從 {config.pretrain_model_path} 加載預訓練權重。")
except Exception as e:
    print(f"加載預訓練模型時出錯: {e}")
    print("將使用隨機初始化的模型。")

# --- 打印模型結構以便調試 ---
print("模型結構:")
print(model)

# --- 修改最後一層以匹配新的類別數 ---
# 檢查模型是否有fc層，如果沒有，嘗試找到最後的線性層
if hasattr(model, 'fc'):
    # 如果有fc層
    print("找到fc層，替換為新的類別數輸出層...")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.num_classes).to(config.device)
else:
    # 嘗試尋找模型中的最後一個線性層
    print("未找到fc層，嘗試尋找和替換最後一個線性層...")
    last_linear_layer = None
    last_module_name = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_layer = module
            last_module_name = name
    
    if last_linear_layer is not None:
        print(f"找到最後一個線性層: {last_module_name}")
        num_features = last_linear_layer.in_features
        # 創建新的線性層並移動到正確的設備上
        new_linear = nn.Linear(num_features, config.num_classes).to(config.device)
        # 通過遞歸找到並替換最後一個線性層
        name_parts = last_module_name.split('.')
        if len(name_parts) == 1:
            # 直接是模型的屬性
            setattr(model, name_parts[0], new_linear)
        else:
            # 嵌套在子模塊中
            parent_module = model
            for part in name_parts[:-1]:
                if part.isdigit():  # 如果是數字，則是列表索引
                    parent_module = parent_module[int(part)]
                else:
                    parent_module = getattr(parent_module, part)
            if name_parts[-1].isdigit():  # 如果最後一部分是數字，則是列表索引
                parent_module[int(name_parts[-1])] = new_linear
            else:
                setattr(parent_module, name_parts[-1], new_linear)
        print(f"已將最後一個線性層替換為輸出類別數為 {config.num_classes} 的新層")
        print(f"新層已移至設備：{config.device}")
    else:
        print("錯誤: 在模型中找不到任何線性層。無法適應新的類別數。")
        print("可能需要手動檢查和修改模型結構。")

# --- (可選) 凍結 CNN 部分的權重 ---
if config.freeze_cnn:
    print("凍結 CNN 部分的權重...")
    for name, param in model.named_parameters():
        if 'cnn' in name:  # 假設 CNN 部分的參數名中包含 'cnn'
            param.requires_grad = False

# --- 準備訓練 ---
# 篩選需要優化的參數
params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())

# 優化器 (使用較小的學習率)
optimizer = optim.AdamW(params_to_optimize, lr=config.learning_rate, weight_decay=1e-5)

# 學習率調度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# 損失函數 (確保 blank 索引正確)
criterion = CTCLoss(blank=config.ctc_blank_char_index, reduction='mean', zero_infinity=True)


# --- 解碼函數修訂 ---
def decode_predictions(preds, idx_to_char_map, blank_idx):
    """使用 CTC 解碼模型輸出"""
    # 確保 preds 是形狀 [T, B, C] 的張量
    if preds.dim() == 3:
        if preds.size(1) > preds.size(0):  # 如果 B > T，則需要轉置
            preds = preds.transpose(0, 1)  # [B, T, C] -> [T, B, C]
    
    preds = preds.permute(1, 0, 2)  # [T, B, C] -> [B, T, C]
    preds_argmax = preds.argmax(dim=2)  # [B, T]
    preds_argmax = preds_argmax.cpu().numpy()

    decoded_strings = []
    for pred_seq in preds_argmax:
        decoded = []
        last_char_idx = -1
        for char_idx in pred_seq:
            if char_idx != blank_idx and char_idx != last_char_idx:
                if char_idx in idx_to_char_map:  # 確保索引有效
                    decoded.append(idx_to_char_map[char_idx])
            last_char_idx = char_idx
        decoded_strings.append("".join(decoded))
    return decoded_strings

def decode_targets(targets, target_lengths, idx_to_char_map):
    """解碼目標標籤"""
    decoded_strings = []
    targets_cpu = targets.cpu().numpy()
    
    for i, length in enumerate(target_lengths):
        if isinstance(length, torch.Tensor):
            length = length.item()
        target_seq = targets_cpu[i, :length]
        decoded = ''.join([idx_to_char_map[idx] for idx in target_seq if idx in idx_to_char_map])
        decoded_strings.append(decoded)
    
    return decoded_strings

# --- 修改訓練函數 ---
def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    progress_bar = tqdm(data_loader, desc="Train")
    
    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None:
            continue
            
        images, labels, label_lengths = batch_data
        
        # 確保數據在正確的設備上
        images = images.to(device)
        labels = labels.to(device)
        
        # 處理 label_lengths
        if isinstance(label_lengths, torch.Tensor):
            label_lengths = label_lengths.to(device)
        else:
            # 如果 label_lengths 是列表，轉換為張量
            label_lengths = torch.tensor(label_lengths, device=device)
        
        # 過濾零長度標籤
        valid_indices = label_lengths > 0
        if not valid_indices.all():
            images = images[valid_indices]
            labels = labels[valid_indices]
            label_lengths = label_lengths[valid_indices]
        
        batch_size = images.size(0)
        if batch_size == 0:
            continue
        
        optimizer.zero_grad()
        
        try:
            # 前向傳播
            outputs = model(images)  # 應該是 [T, B, C] 形狀
            
            # 確保輸出維度正確 - 模型輸出可能是 [B, T, C] 或 [T, B, C]
            if outputs.size(0) == batch_size:
                # 如果第一維是批次大小，則轉換為 [T, B, C]
                outputs = outputs.transpose(0, 1)
            
            # 確保輸出維度正確 - 模型輸出可能是 [T, B, C] 或 [B, T, C]
            T, B, C = outputs.size()
            if B != batch_size:
                print(f"警告: 輸出批次維度 {B} 與輸入批次大小 {batch_size} 不匹配！在第 {batch_idx} 批次。")
                continue
            
            # 創建輸入長度張量 - 對每個樣本都設置為時間步長 T
            input_lengths = torch.full((batch_size,), T, dtype=torch.long, device=device)
            
            # 計算損失
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            # 更新統計信息
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 分別解碼預測和目標
            decoded_preds = decode_predictions(outputs, idx_to_char, config.ctc_blank_char_index)
            decoded_targets = decode_targets(labels, label_lengths, idx_to_char)
            
            all_preds.extend(decoded_preds)
            all_targets.extend(decoded_targets)
            
            # 計算準確率
            for pred, target in zip(decoded_preds, decoded_targets):
                if pred == target:
                    total_correct += 1
                    
            # 更新進度條
            progress_bar.set_postfix(loss=loss.item(), 
                                  acc=total_correct/total_samples if total_samples > 0 else 0)
                                  
        except Exception as e:
            print(f"在訓練批次 {batch_idx} 中發生錯誤: {e}")
            print(f"圖像形狀: {images.shape}, 標籤形狀: {labels.shape}, 標籤長度形狀: {label_lengths.shape}")
            if torch.isnan(images).any():
                print("警告: 圖像中包含 NaN 值")
            continue
    
    # 計算平均損失和準確率
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_acc, all_preds, all_targets

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    progress_bar = tqdm(loader, desc='Validate', leave=False)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue
                
            images, labels, label_lengths = batch_data
            
            # 確保數據在正確的設備上
            images = images.to(device)
            labels = labels.to(device)
            
            # 處理 label_lengths
            if isinstance(label_lengths, torch.Tensor):
                label_lengths = label_lengths.to(device)
            else:
                # 如果 label_lengths 是列表，轉換為張量
                label_lengths = torch.tensor(label_lengths, device=device)
            
            # 過濾零長度標籤
            valid_indices = label_lengths > 0
            if not valid_indices.all():
                images = images[valid_indices]
                labels = labels[valid_indices]
                label_lengths = label_lengths[valid_indices]
            
            batch_size = images.size(0)
            if batch_size == 0:
                continue
            
            try:
                # 前向傳播
                outputs = model(images)  # 應該是 [T, B, C] 形狀
                
                # 確保輸出維度正確 - 模型輸出可能是 [B, T, C] 或 [T, B, C]
                if outputs.size(0) == batch_size:
                    # 如果第一維是批次大小，則轉換為 [T, B, C]
                    outputs = outputs.transpose(0, 1)
                
                # 確保輸出維度正確
                T, B, C = outputs.size()
                if B != batch_size:
                    print(f"警告: 驗證時輸出批次維度 {B} 與輸入批次大小 {batch_size} 不匹配！在第 {batch_idx} 批次。")
                    continue
                
                # 創建輸入長度張量 - 對每個樣本都設置為時間步長 T
                input_lengths = torch.full((batch_size,), T, dtype=torch.long, device=device)
                
                # 計算損失
                loss = criterion(outputs, labels, input_lengths, label_lengths)
                
                # 更新統計信息
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 分別解碼預測和目標
                decoded_preds = decode_predictions(outputs, idx_to_char, config.ctc_blank_char_index)
                decoded_targets = decode_targets(labels, label_lengths, idx_to_char)
                
                all_preds.extend(decoded_preds)
                all_targets.extend(decoded_targets)
                
                # 計算準確率
                for pred, target in zip(decoded_preds, decoded_targets):
                    if pred == target:
                        total_correct += 1
                        
                # 更新進度條
                progress_bar.set_postfix(loss=loss.item(), 
                                      acc=total_correct/total_samples if total_samples > 0 else 0)
                                      
            except Exception as e:
                print(f"在驗證批次 {batch_idx} 中發生錯誤: {e}")
                print(f"圖像形狀: {images.shape}, 標籤形狀: {labels.shape}, 標籤長度形狀: {label_lengths.shape}")
                continue
    
    # 計算平均損失和準確率
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_acc, all_preds, all_targets

# --- 微調循環 ---
print("開始微調...")
best_val_acc = 0.0
early_stop_counter = 0
train_losses, val_losses, val_accs = [], [], []

for epoch in range(1, config.epochs + 1):
    print(f"\nEpoch {epoch}/{config.epochs}")

    train_loss, train_acc, train_preds, train_targets = train_epoch(model, train_loader, optimizer, criterion, config.device)
    val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, config.device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 輸出一些預測樣例
    print("\n預測樣例 (預測 -> 真實):")
    for i in range(min(5, len(val_preds))):
        print(f"'{val_preds[i]}' -> '{val_targets[i]}'")

    # 更新學習率
    scheduler.step(val_loss)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(config.finetune_model_save_path, 'best_finetuned_model.pth')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'char_map': char_to_idx, # 保存字符映射以備後用
            'config': {k: v for k, v in vars(config).items() if not k.startswith('_') and not callable(v)} # 保存可序列化的配置
        }
        torch.save(checkpoint, best_model_path)
        print(f"保存最佳模型，驗證準確率: {best_val_acc:.4f}")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"驗證準確率未提升 ({early_stop_counter}/{config.early_stopping_patience})")
        if early_stop_counter >= config.early_stopping_patience:
            print(f"\n觸發早停。")
            break

    # 繪製並保存曲線圖 (頻率降低)
    if epoch % 5 == 0 or epoch == 1 or epoch == config.epochs: # 每 5 個 epoch 或第一個/最後一個 epoch 保存
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(1, epoch + 1), train_losses, label='Train Loss')
            plt.plot(range(1, epoch + 1), val_losses, label='Val Loss')
            plt.title('Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(range(1, epoch + 1), val_accs, label='Val Accuracy')
            plt.title('Validation Accuracy over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(config.finetune_model_save_path, 'finetuning_curves.png'))
            plt.close()
        except Exception as e:
            print(f"繪製圖表時出錯: {e}")


print("\n微調完成！")
print(f"最佳驗證準確率: {best_val_acc:.4f}")
print(f"最佳模型保存在: {os.path.join(config.finetune_model_save_path, 'best_finetuned_model.pth')}")

# (可選) 加載最佳模型並在驗證集上顯示一些預測示例
print("\n加載最佳模型進行示例預測...")
best_model_path = os.path.join(config.finetune_model_save_path, 'best_finetuned_model.pth')
if os.path.exists(best_model_path):
    try:
        best_checkpoint = torch.load(best_model_path, map_location=config.device)
        # 重新創建模型結構 (確保類別數正確)
        # 從 checkpoint 中加載配置來確定類別數
        saved_config_dict = best_checkpoint.get('config', {})
        saved_num_classes = saved_config_dict.get('num_classes', config.num_classes) # 使用保存的或當前的
        saved_hidden_size = saved_config_dict.get('hidden_size', config.hidden_size)
        saved_input_channels = saved_config_dict.get('input_channels', config.input_channels)

        final_model = CRNN(saved_input_channels, saved_hidden_size, saved_num_classes).to(config.device)
        final_model.load_state_dict(best_checkpoint['model_state_dict'])
        final_model.eval()

        # 獲取一些驗證樣本進行預測
        # 確保 val_loader 仍然可用且數據有效
        if len(val_dataset) > 0:
             val_loss, val_acc, val_preds, val_targets = validate(final_model, val_loader, criterion, config.device)
             print(f"\n最終驗證集表現: Loss={val_loss:.4f}, Acc={val_acc:.4f}")

             print("\n部分預測示例 (預測 -> 真實):")
             num_examples = min(15, len(val_preds))
             for i in range(num_examples):
                 print(f"'{val_preds[i]}' -> '{val_targets[i]}'")
        else:
             print("驗證集為空，無法進行示例預測。")

    except Exception as e:
        print(f"加載最佳模型或進行預測時出錯: {e}")
else:
    print(f"未找到最佳模型文件: {best_model_path}")
