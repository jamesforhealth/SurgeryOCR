import os
import json
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
import torch.nn.functional as F
from torch.nn import CTCLoss
# --- 從預訓練腳本導入 CRNN 模型 ---
# 假設 pretrain_crnn.py 在 models/ 目錄下，並且 finetune_surgery_crnn.py 在項目根目錄
# 或者根據您的實際路徑調整
try:
    from models.pretrain_crnn import CRNN
except ImportError:
    print("錯誤: 無法從 models.pretrain_crnn 導入 CRNN 模型。請確保路徑正確且該文件包含 CRNN 類定義。")
    # 作為後備，如果您想在此處也保留一份CRNN定義（不推薦，最好是導入）
    # class CRNN(nn.Module): ... (在此處粘貼CRNN定義)
    raise # 重新拋出導入錯誤，以便用戶知道問題所在

# --- 配置類 ---
class FinetuneConfig:
    def __init__(self, args):
        # 路徑相關
        self.data_dir = args.data_dir # 微調數據的根目錄 (包含影片子目錄)
        self.val_video_dir_name = args.val_video_dir_name
        self.pretrained_model_path = args.pretrained_model_path
        self.pretrained_char_map_path = args.pretrained_char_map_path
        self.finetuned_model_save_dir = args.finetuned_model_save_dir
        os.makedirs(self.finetuned_model_save_dir, exist_ok=True)
        self.finetuned_model_name = args.finetuned_model_name

        # 圖像和模型結構參數
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.input_channels = args.input_channels # 應與預訓練模型一致
        self.hidden_size = args.hidden_size     # 應與預訓練模型一致

        # 微調超參數
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.freeze_cnn_layers = args.freeze_cnn_layers

        # 混合預訓練數據相關
        self.pretrain_data_dir_for_mix = args.pretrain_data_dir_for_mix
        self.pretrain_labels_file_for_mix = args.pretrain_labels_file_for_mix # 例如 'labels.jsonl'
        self.pretrain_mix_ratio = args.pretrain_mix_ratio # 預訓練數據混合比例

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 從加載的字符映射表獲取字符信息
        if not os.path.exists(self.pretrained_char_map_path):
            raise FileNotFoundError(f"預訓練字符映射文件未找到: {self.pretrained_char_map_path}")
        with open(self.pretrained_char_map_path, 'r', encoding='utf-8') as f:
            # 假設 char_map 是 {原始索引: 字符}
            # 例如 pretrain_crnn.py 中保存的是 config.idx_to_char
            # {0: '0', 1: '1', ..., 9: '9'}
            loaded_idx_to_char_map = json.load(f)
            # 將 json key (字符串) 轉為 int
            self.idx_to_char = {int(k): v for k, v in loaded_idx_to_char_map.items()}

        # 根據加載的 idx_to_char 構建 char_to_idx
        self.char_to_idx = {v: k for k, v in self.idx_to_char.items()}
        
        # 類別數 = 字符數量 (來自加載的映射表) + 1 (CTC空白符)
        self.num_model_classes = len(self.idx_to_char) + 1
        # CTC 空白符號的索引 (假設為字符集之後的下一個索引)
        self.blank_index = len(self.idx_to_char)

        print(f"從 {self.pretrained_char_map_path} 加載的字符集大小: {len(self.idx_to_char)}")
        print(f"模型將使用的類別數 (含空白): {self.num_model_classes}")
        print(f"CTC 空白符號索引: {self.blank_index}")

        print("--- FinetuneConfig ---")
        print(f"Loaded idx_to_char: {self.idx_to_char}")
        print(f"Calculated num_model_classes: {self.num_model_classes}")
        print(f"Calculated blank_index: {self.blank_index}")
        print(f"Expected characters: {''.join(self.idx_to_char.values())}")


# --- 數據集定義 ---
class SurgeryFrameDataset(Dataset):
    def __init__(self, image_paths, labels, char_to_idx_map, img_height, img_width, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_idx = char_to_idx_map
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.Grayscale(num_output_channels=1), # 確保是單通道
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]) # 與預訓練時的標準化一致
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_str = self.labels[idx]
        try:
            image = Image.open(img_path).convert('L') # 確保以灰度模式打開
        except FileNotFoundError:
            print(f"錯誤: 圖像文件未找到 {img_path}")
            # 返回一個占位符或引發錯誤
            return torch.randn(1, self.img_height, self.img_width), torch.IntTensor([]), torch.IntTensor([])
        except Exception as e:
            print(f"錯誤: 加載圖像 {img_path} 時出錯: {e}")
            return torch.randn(1, self.img_height, self.img_width), torch.IntTensor([]), torch.IntTensor([])


        if self.transform:
            image = self.transform(image)

        encoded_label = encode_text(label_str, self.char_to_idx)
        label_length = torch.IntTensor([len(encoded_label)])
        return image, torch.IntTensor(encoded_label), label_length

class PretrainMixDataset(Dataset):
    """用於混合的原始預訓練數據集"""
    def __init__(self, root_dir, labels_file, char_to_idx_map, img_height, img_width, transform=None):
        self.root_dir = root_dir
        self.char_to_idx = char_to_idx_map
        self.img_height = img_height
        self.img_width = img_width
        self.image_paths = []
        self.labels = []

        if not os.path.isdir(root_dir):
            print(f"警告: 預訓練數據目錄 {root_dir} 不存在或不是一個目錄。將無法混合預訓練數據。")
            return
        if not os.path.isfile(labels_file):
            print(f"警告: 預訓練標籤文件 {labels_file} 未找到。將無法混合預訓練數據。")
            return

        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # 假設預訓練標籤文件格式與手術數據類似: {"filename": "img.png", "label": "text"}
                        # 或者根據您的實際格式調整
                        img_filename = data.get("filename")
                        label_str = data.get("label")
                        if img_filename and label_str is not None: # 確保標籤可以是空字符串
                            full_img_path = os.path.join(self.root_dir, img_filename)
                            if os.path.exists(full_img_path):
                                self.image_paths.append(full_img_path)
                                self.labels.append(str(label_str)) # 確保是字符串
                            else:
                                print(f"警告: 預訓練數據中的圖像文件 {full_img_path} 未找到。")
                        else:
                            print(f"警告: 預訓練標籤文件中的行格式不正確: {line.strip()}")
                    except json.JSONDecodeError:
                        print(f"警告: 無法解析預訓練標籤文件中的JSON行: {line.strip()}")
        except Exception as e:
            print(f"錯誤: 讀取預訓練標籤文件 {labels_file} 時出錯: {e}")
            self.image_paths = [] # 清空以避免後續問題
            self.labels = []

        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        if not self.image_paths:
            print("警告: 未能從預訓練數據源加載任何圖像和標籤。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_str = self.labels[idx]
        try:
            image = Image.open(img_path).convert('L')
        except FileNotFoundError:
            print(f"錯誤: 預訓練圖像文件未找到 {img_path}")
            return torch.randn(1, self.img_height, self.img_width), torch.IntTensor([]), torch.IntTensor([])
        except Exception as e:
            print(f"錯誤: 加載預訓練圖像 {img_path} 時出錯: {e}")
            return torch.randn(1, self.img_height, self.img_width), torch.IntTensor([]), torch.IntTensor([])

        if self.transform:
            image = self.transform(image)
        
        encoded_label = encode_text(label_str, self.char_to_idx)
        label_length = torch.IntTensor([len(encoded_label)])
        return image, torch.IntTensor(encoded_label), label_length

# --- 輔助函數 ---
def encode_text(text, char_to_idx):
    """將文本字符串編碼為數字序列"""
    encoded = []
    for char in str(text): # 確保是字符串
        idx = char_to_idx.get(char)
        if idx is not None:
            encoded.append(idx)
        # else: # 可選：處理不在字符集中的字符
        #     print(f"警告: 字符 '{char}' 在文本 '{text}' 中未找到於字符映射表中，將被忽略。")
    return encoded

def decode_predictions(preds_tensor, idx_to_char_map, blank_idx):
    """
    解碼模型的原始輸出 (通常在 log_softmax 之後)。
    preds_tensor: [SeqLen, BatchSize, NumClasses]
    """
    preds_tensor = F.log_softmax(preds_tensor, dim=2) # 確保在log_softmax空間
    preds_idx = torch.argmax(preds_tensor, dim=2)  # [SeqLen, BatchSize]
    preds_idx = preds_idx.transpose(0, 1).cpu().numpy()  # [BatchSize, SeqLen]

    decoded_texts = []
    for batch_item_preds in preds_idx:
        text = []
        last_char_idx = None
        for char_idx in batch_item_preds:
            if char_idx != blank_idx and char_idx != last_char_idx:
                char = idx_to_char_map.get(char_idx)
                if char:
                    text.append(char)
            last_char_idx = char_idx
        decoded_texts.append("".join(text))
    return decoded_texts

def load_data_ft(config: FinetuneConfig):
    """
    加載微調數據集 (手術幀數據) 和可選的預訓練混合數據。
    根據 config.data_dir 下的 .mp4 文件名篩選有效的影片數據目錄。
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config.img_height, config.img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_items_ft = []
    val_items_ft = []

    # 1. 識別 data_dir 下所有 .mp4 文件，並獲取其基本名稱作為有效的影片目錄名
    mp4_files = glob.glob(os.path.join(config.data_dir, "*.mp4"))
    valid_video_dir_names = {os.path.splitext(os.path.basename(f))[0] for f in mp4_files}
    
    print(f"在 {config.data_dir} 中找到的 .mp4 文件對應的有效影片目錄名: {valid_video_dir_names}")
    if not valid_video_dir_names:
        print(f"警告: 在 {config.data_dir} 中沒有找到 .mp4 文件，無法確定有效的影片數據目錄。")
        # 根據需求，這裡可以選擇拋出錯誤或返回空的 DataLoader
        # return None, None # 或者根據您的錯誤處理策略

    all_video_subdirs = [d for d in os.listdir(config.data_dir) if os.path.isdir(os.path.join(config.data_dir, d))]
    
    processed_video_dirs_for_ft = set()

    for video_dir_name in all_video_subdirs:
        if video_dir_name not in valid_video_dir_names:
            # print(f"跳過目錄 {video_dir_name}，因為在 {config.data_dir} 中沒有找到對應的 .mp4 文件。")
            continue # 只處理與 .mp4 文件名對應的目錄

        video_path = os.path.join(config.data_dir, video_dir_name)
        labels_file = os.path.join(video_path, "region2.jsonl")

        if not os.path.exists(labels_file):
            print(f"警告: 在目錄 {video_path} 中未找到 region2.jsonl，跳過此目錄。")
            continue
        
        current_video_items = []
        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        img_name = data['filename']
                        label = data['label']
                        img_path = os.path.join(video_path, img_name)
                        if os.path.exists(img_path):
                            current_video_items.append({'image_path': img_path, 'label': label})
                        else:
                            print(f"警告: 圖像文件 {img_path} 未找到，跳過此條目。")
                    except json.JSONDecodeError:
                        print(f"警告: 解析JSON行失敗: {line.strip()}，文件: {labels_file}")
                    except KeyError:
                        print(f"警告: JSON行缺少 'filename' 或 'label' 鍵: {line.strip()}，文件: {labels_file}")
        except Exception as e:
            print(f"錯誤: 讀取或處理標籤文件 {labels_file} 失敗: {e}")
            continue

        if not current_video_items:
            print(f"警告: 目錄 {video_dir_name} 未能加載任何有效的圖像-標籤對。")
            continue

        if config.val_video_dir_name and video_dir_name == config.val_video_dir_name:
            val_items_ft.extend(current_video_items)
            print(f"目錄 {video_dir_name} ({len(current_video_items)} 項) 已分配給驗證集。")
        else:
            train_items_ft.extend(current_video_items)
            print(f"目錄 {video_dir_name} ({len(current_video_items)} 項) 已分配給訓練集。")
        processed_video_dirs_for_ft.add(video_dir_name)

    if config.val_video_dir_name and config.val_video_dir_name not in processed_video_dirs_for_ft:
        print(f"警告: 指定的驗證影片目錄 '{config.val_video_dir_name}' 未在 {config.data_dir} 下作為有效數據源被處理 (可能缺少對應的 .mp4 文件或 region2.jsonl)。")

    if not train_items_ft:
        print("警告: 未能加載任何微調訓練數據。請檢查您的數據目錄和配置。")
        # 根據情況，可能需要返回 None 或拋出錯誤
        # return None, None

    # --- 創建 SurgeryFrameDataset 實例 ---
    # 訓練集 (僅手術幀數據)
    train_surgery_dataset = SurgeryFrameDataset(train_items_ft, config.char_to_idx, transform, config.img_height, config.img_width)
    
    final_train_dataset = train_surgery_dataset # 初始化最終訓練集

    # --- 混合預訓練數據 (如果配置了) ---
    if config.pretrain_mix_ratio > 0 and config.pretrain_data_dir_for_mix:
        print(f"嘗試混合預訓練數據，比例: {config.pretrain_mix_ratio}")
        pretrain_labels_path = os.path.join(config.pretrain_data_dir_for_mix, config.pretrain_labels_file_for_mix)
        if not os.path.exists(config.pretrain_data_dir_for_mix) or not os.path.exists(pretrain_labels_path):
            print(f"警告: 預訓練混合數據目錄 '{config.pretrain_data_dir_for_mix}' 或標籤文件 '{pretrain_labels_path}' 未找到。跳過混合。")
        else:
            pretrain_mix_items = []
            try:
                with open(pretrain_labels_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            img_name = data['filename']
                            label = data['label']
                            img_path = os.path.join(config.pretrain_data_dir_for_mix, img_name)
                            if os.path.exists(img_path):
                                pretrain_mix_items.append({'image_path': img_path, 'label': label})
                        except json.JSONDecodeError:
                            print(f"警告: 解析預訓練標籤JSON行失敗: {line.strip()}")
                        except KeyError:
                            print(f"警告: 預訓練標籤JSON行缺少 'filename' 或 'label' 鍵: {line.strip()}")
            except Exception as e:
                print(f"錯誤: 讀取或處理預訓練混合標籤文件 {pretrain_labels_path} 失敗: {e}")

            if pretrain_mix_items:
                num_surgery_train_samples = len(train_surgery_dataset)
                num_pretrain_to_mix = int(num_surgery_train_samples * config.pretrain_mix_ratio)
                
                if num_pretrain_to_mix == 0 and config.pretrain_mix_ratio > 0 and num_surgery_train_samples > 0:
                    num_pretrain_to_mix = 1 # 至少混合一個樣本，如果比例允許但計算結果為0
                
                print(f"手術幀訓練樣本數: {num_surgery_train_samples}")
                print(f"期望混合的預訓練樣本數: {num_pretrain_to_mix}")

                if num_pretrain_to_mix > len(pretrain_mix_items):
                    print(f"警告: 期望混合的預訓練樣本數 ({num_pretrain_to_mix}) 大于可用的預訓練樣本數 ({len(pretrain_mix_items)})。將使用所有可用的預訓練樣本。")
                    num_pretrain_to_mix = len(pretrain_mix_items)

                if num_pretrain_to_mix > 0:
                    # 隨機抽取預訓練數據子集
                    # random.shuffle(pretrain_mix_items) # 如果 PretrainMixDataset 內部沒有隨機化，可以在這裡 shuffle
                    # pretrain_subset_items = random.sample(pretrain_mix_items, num_pretrain_to_mix) # 可能導致重複，如果樣本少
                    
                    # 創建完整的 PretrainMixDataset，然後使用 Subset 隨機抽樣
                    full_pretrain_mix_dataset = PretrainMixDataset(
                        root_dir=config.pretrain_data_dir_for_mix, # PretrainMixDataset 需要 root_dir
                        labels_file=pretrain_labels_path,          # 和 labels_file
                        char_to_idx=config.char_to_idx,
                        transform=transform,
                        img_height=config.img_height,
                        img_width=config.img_width
                    )
                    if len(full_pretrain_mix_dataset) > 0:
                        indices_to_sample = random.sample(range(len(full_pretrain_mix_dataset)), min(num_pretrain_to_mix, len(full_pretrain_mix_dataset)))
                        pretrain_dataset_subset = Subset(full_pretrain_mix_dataset, indices_to_sample)
                        
                        print(f"實際混合的預訓練樣本數: {len(pretrain_dataset_subset)}")
                        final_train_dataset = ConcatDataset([train_surgery_dataset, pretrain_dataset_subset])
                        print(f"訓練集已與 {len(pretrain_dataset_subset)} 個預訓練樣本混合。總訓練樣本數: {len(final_train_dataset)}")
                    else:
                        print("警告: PretrainMixDataset 為空，無法混合預訓練數據。")
                else:
                    print("計算出的預訓練混合樣本數為0，不進行混合。")
            else:
                print("警告: 未能從預訓練數據源加載任何有效的圖像-標籤對，跳過混合。")

    # 驗證集
    val_dataset_ft = None
    if val_items_ft:
        val_dataset_ft = SurgeryFrameDataset(val_items_ft, config.char_to_idx, transform, config.img_height, config.img_width)
        print(f"已創建驗證集，包含 {len(val_dataset_ft)} 個樣本。")
    else:
        print("未加載驗證數據 (可能是因為未指定 val_video_dir_name，或指定的目錄無效/無數據)。")

    collate_fn_to_use = CollateFN(config.blank_index) # 使用統一的 collate_fn

    train_loader_ft = DataLoader(
        final_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4, # 根據您的CPU核心數調整
        pin_memory=True,
        collate_fn=collate_fn_to_use
    ) if final_train_dataset and len(final_train_dataset) > 0 else None

    val_loader_ft = DataLoader(
        val_dataset_ft,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_to_use
    ) if val_dataset_ft and len(val_dataset_ft) > 0 else None
    
    if train_loader_ft:
        print(f"訓練 DataLoader 已創建，每個 epoch 將有 {len(train_loader_ft)} 個批次。")
    if val_loader_ft:
        print(f"驗證 DataLoader 已創建，每個 epoch 將有 {len(val_loader_ft)} 個批次。")

    return train_loader_ft, val_loader_ft

def collate_fn_ft(batch):
    images, encoded_labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    
    # 對 encoded_labels 進行填充，使其具有相同的長度
    max_len = max(len(lab) for lab in encoded_labels)
    padded_labels = torch.zeros(len(encoded_labels), max_len, dtype=torch.long)
    for i, lab in enumerate(encoded_labels):
        padded_labels[i, :len(lab)] = lab
        
    label_lengths = torch.cat(label_lengths) # 應該已經是 [B]
    return images, padded_labels, label_lengths # 返回填充後的標籤和原始長度

# --- 訓練和驗證循環 ---
def train_epoch_ft(model, data_loader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="訓練中", leave=False)

    for batch_idx, (images, targets, target_lengths) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device) # targets 已經是編碼後的索引序列
        target_lengths = target_lengths.to(device) # [B]

        optimizer.zero_grad()
        # 模型輸出: [SeqLen, BatchSize, NumClasses]
        preds = model(images)
        preds_log_softmax = F.log_softmax(preds, dim=2) # CTCLoss 期望 log_softmax 輸入

        # CTCLoss 輸入:
        # log_probs: (T, N, C) where T is input length, N is batch size, C is number of classes
        # targets: (N, S) or (sum(target_lengths))
        # input_lengths: (N) -> 模型的序列輸出長度 (通常是固定的，由CNN下採樣決定)
        # target_lengths: (N) -> 每個目標序列的真實長度
        
        input_lengths = torch.full(size=(preds.size(1),), fill_value=preds.size(0), dtype=torch.long).to(device)

        # 展平 targets 以匹配 CTCLoss 的期望 (sum(target_lengths))
        flat_targets = []
        for i in range(targets.size(0)): # 遍歷 batch
            flat_targets.extend(targets[i, :target_lengths[i]].tolist())
        flat_targets = torch.IntTensor(flat_targets).to(device)

        loss = criterion(preds_log_softmax, flat_targets, input_lengths, target_lengths)
        
        # 處理可能的 inf 或 nan 損失
        if torch.isinf(loss) or torch.isnan(loss):
            print(f"警告: 檢測到 Inf/NaN 損失在批次 {batch_idx}。跳過此批次的梯度更新。")
            # 可選：打印更多調試信息
            # print(f"Preds shape: {preds.shape}, max: {preds.max()}, min: {preds.min()}")
            # print(f"Targets: {targets}, Target lengths: {target_lengths}")
            # print(f"Input lengths: {input_lengths}")
            optimizer.zero_grad() # 確保清除任何可能導致NaN的梯度
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5) # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)

def validate_ft(model, data_loader, criterion, device, config):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc="驗證中", leave=False)

    with torch.no_grad():
        for images, targets, target_lengths in progress_bar:
            images = images.to(device)
            targets_encoded = targets.to(device) # 編碼後的標籤
            target_lengths_val = target_lengths.to(device)

            preds = model(images)
            preds_log_softmax = F.log_softmax(preds, dim=2)
            input_lengths_val = torch.full(size=(preds.size(1),), fill_value=preds.size(0), dtype=torch.long).to(device)

            flat_targets_val = []
            for i in range(targets_encoded.size(0)):
                flat_targets_val.extend(targets_encoded[i, :target_lengths_val[i]].tolist())
            flat_targets_val = torch.IntTensor(flat_targets_val).to(device)
            
            loss = criterion(preds_log_softmax, flat_targets_val, input_lengths_val, target_lengths_val)
            total_loss += loss.item()

            # 解碼預測和目標以計算準確率 (完全匹配)
            decoded_preds = decode_predictions(preds.cpu(), config.idx_to_char, config.blank_index)
            
            # 解碼真實標籤
            target_texts = []
            for i in range(targets_encoded.size(0)): # 遍歷 batch
                true_encoded_label = targets_encoded[i, :target_lengths_val[i]].cpu().tolist()
                true_text = "".join([config.idx_to_char.get(idx, '') for idx in true_encoded_label])
                target_texts.append(true_text)

            for pred_text, true_text in zip(decoded_preds, target_texts):
                if pred_text == true_text:
                    correct_predictions += 1
            total_samples += len(target_texts)
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"驗證 - 平均損失: {avg_loss:.4f}, 準確率 (完全匹配): {accuracy:.4f} ({correct_predictions}/{total_samples})")
    return avg_loss, accuracy

# --- 主微調流程 ---
def run_finetuning(config):
    print(f"使用設備: {config.device}")
    print(f"從字符映射加載的 idx_to_char: {config.idx_to_char}")

    train_loader, val_loader = load_data_ft(config)
    if train_loader is None:
        print("數據加載失敗，終止微調。")
        return

    # 實例化模型
    # num_model_classes 和 blank_index 已經在 config 中根據加載的 char_map 設置好了
    model = CRNN(
        img_height=config.img_height,
        input_channels=config.input_channels,
        hidden_size=config.hidden_size,
        num_classes=config.num_model_classes # 使用從 char_map 推斷的類別數
    ).to(config.device)

    # 加載預訓練權重
    if os.path.exists(config.pretrained_model_path):
        try:
            print(f"從 {config.pretrained_model_path} 加載預訓練權重...")
            # model.load_state_dict(torch.load(config.pretrained_model_path, map_location=config.device))
            
            # 為了處理可能的鍵名不匹配 (例如 DataParallel 引入的 'module.' 前綴)
            # 或者最後一層 fc 的大小不匹配 (如果字符集改變了 - 但我們這裡假設字符集通過 char_map 保持一致)
            pretrained_dict = torch.load(config.pretrained_model_path, map_location=config.device)
            model_dict = model.state_dict()

            # 1. 過濾掉不匹配的鍵 (例如，如果預訓練模型的最後一層與當前模型不同)
            #    在我們的例子中，num_classes 是基於加載的 char_map，所以 fc 層應該匹配
            #    除非預訓練模型本身的 num_classes 與 char_map 不符，這是不應該發生的
            
            # 處理 'module.' 前綴 (如果預訓練模型是用 DataParallel 保存的)
            clean_pretrained_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith('module.'):
                    clean_pretrained_dict[k[7:]] = v
                else:
                    clean_pretrained_dict[k] = v
            
            # 過濾掉尺寸不匹配的層 (主要是為了安全，理論上 fc 層應該匹配)
            final_load_dict = {}
            for k, v in clean_pretrained_dict.items():
                if k in model_dict and model_dict[k].size() == v.size():
                    final_load_dict[k] = v
                else:
                    print(f"跳過加載權重: {k} (尺寸不匹配或不在當前模型中)")
            
            model_dict.update(final_load_dict)
            model.load_state_dict(model_dict)
            print("預訓練權重加載成功。")

            print("\n--- Model Structure (after loading weights and potential replacement) ---")
            print(model)
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
                print(f"Final classifier out_features: {model.classifier.out_features}")
            elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                 print(f"Final fc out_features: {model.fc.out_features}")

        except Exception as e:
            print(f"加載預訓練權重失敗: {e}")
            print("將從隨機權重開始訓練（或部分加載的權重）。如果預期加載權重，請檢查模型結構和權重文件。")
    else:
        print(f"警告: 預訓練模型文件 {config.pretrained_model_path} 未找到。模型將從隨機初始化的權重開始。")

    # 凍結CNN層 (如果需要)
    if config.freeze_cnn_layers:
        print("凍結CNN層的權重...")
        for param_name, param in model.named_parameters():
            if param_name.startswith("cnn."):
                param.requires_grad = False
            else:
                param.requires_grad = True
        # 確保優化器只更新需要梯度的參數
        optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
        print("以下層將被訓練:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"\t{name}")
    else:
        print("訓練所有模型層。")
        optimizer_params = model.parameters()


    criterion = CTCLoss(blank=config.blank_index, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(optimizer_params, lr=config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    train_losses, val_losses, val_accuracies = [], [], []

    print(f"\n開始在 {config.device} 上微調模型...")
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        train_loss = train_epoch_ft(model, train_loader, criterion, optimizer, config.device, config)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1} - 訓練損失: {train_loss:.4f}")

        current_val_loss = float('inf')
        current_val_accuracy = 0.0
        if val_loader:
            current_val_loss, current_val_accuracy = validate_ft(model, val_loader, criterion, config.device, config)
            val_losses.append(current_val_loss)
            val_accuracies.append(current_val_accuracy)
            scheduler.step(current_val_loss)

            if current_val_loss < best_val_loss: # 或 current_val_accuracy > best_val_accuracy
                best_val_loss = current_val_loss
                # best_val_accuracy = current_val_accuracy # 如果以準確率為標準
                save_path = os.path.join(config.finetuned_model_save_dir, config.finetuned_model_name)
                torch.save(model.state_dict(), save_path)
                print(f"在 Epoch {epoch+1} 保存了新的最佳模型到 {save_path} (Val Loss: {best_val_loss:.4f}, Val Acc: {current_val_accuracy:.4f})")
        else:
            save_path = os.path.join(config.finetuned_model_save_dir, f"{os.path.splitext(config.finetuned_model_name)[0]}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"在 Epoch {epoch+1} 保存模型到 {save_path} (無驗證集)")


    print("\n微調完成！")

    # 繪製損失和準確率曲線
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='訓練損失')
    if val_losses:
        plt.plot(val_losses, label='驗證損失')
    plt.title('損失曲線')
    plt.xlabel('Epoch')
    plt.ylabel('損失')
    plt.legend()

    if val_accuracies:
        plt.subplot(2, 1, 2)
        plt.plot(val_accuracies, label='驗證準確率 (完全匹配)')
        plt.title('驗證準確率曲線')
        plt.xlabel('Epoch')
        plt.ylabel('準確率')
        plt.legend()

    plt.tight_layout()
    plot_save_path = os.path.join(config.finetuned_model_save_dir, "finetuning_curves.png")
    plt.savefig(plot_save_path)
    print(f"訓練曲線圖已保存至: {plot_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRNN 微調腳本")
    
    # 路徑參數
    parser.add_argument("--data_dir", type=str, default="./data/", help="包含影片子目錄的微調數據根目錄")
    parser.add_argument("--val_video_dir_name", type=str, default="./data/2024-12-04-wu-003", help="用作驗證集的影片目錄名稱。如果為 None，則不使用驗證集。")
    parser.add_argument("--pretrained_model_path", type=str, default="./models/OCR_interface/simpleocr/best_crnn_model.pth", help="預訓練CRNN模型權重文件 (.pth) 的路徑")
    parser.add_argument("--pretrained_char_map_path", type=str, required=True, help="預訓練字符映射表 (char_mapping.json) 的路徑")
    parser.add_argument("--finetuned_model_save_dir", type=str, default="./models/OCR_interface/simpleocr/", help="保存微調後模型的目錄")
    parser.add_argument("--finetuned_model_name", type=str, default="best_finetuned_model.pth", help="保存的最佳微調模型的文件名")

    # 圖像和模型結構參數
    parser.add_argument("--img_height", type=int, default=32, help="輸入圖像高度 (應與CRNN模型期望一致)")
    parser.add_argument("--img_width", type=int, default=100, help="輸入圖像寬度")
    parser.add_argument("--input_channels", type=int, default=1, help="模型輸入通道數 (例如1表示灰度圖，應與預訓練模型一致)")
    parser.add_argument("--hidden_size", type=int, default=256, help="CRNN中RNN的隱藏單元數 (應與預訓練模型一致)")

    # 微調超參數
    parser.add_argument("--epochs", type=int, default=30, help="微調的epoch數量")
    parser.add_argument("--batch_size", type=int, default=16, help="批處理大小")
    parser.add_argument("--lr", type=float, default=5e-5, help="學習率 (微調時通常較小)")
    parser.add_argument("--freeze_cnn_layers", action='store_true', help="是否凍結預訓練模型的CNN層")

    # 混合預訓練數據參數
    parser.add_argument("--pretrain_data_dir_for_mix", type=str, default=None, help="用於混合的原始預訓練數據的根目錄")
    parser.add_argument("--pretrain_labels_file_for_mix", type=str, default="labels.jsonl", help="在 pretrain_data_dir_for_mix 中預訓練數據的標籤文件名 (例如 'labels.jsonl')")
    parser.add_argument("--pretrain_mix_ratio", type=float, default=0.0, help="混合到每個epoch訓練數據中的預訓練數據的比例 (0.0 到 1.0)。例如0.1表示添加相當於微調訓練集10%大小的預訓練數據。")

    args = parser.parse_args()
    config = FinetuneConfig(args)
    
    # 設置隨機種子 (可選)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True # 可能影響性能
        # torch.backends.cudnn.benchmark = False

    run_finetuning(config) 