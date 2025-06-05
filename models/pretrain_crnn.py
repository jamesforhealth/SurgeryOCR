import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import CTCLoss
import json
import random
import argparse


characters = '0123456789-'
# 配置參數
class PretrainConfig:
    # 數據路徑
    train_dir = "../data/pretrain/merged_train"#train"
    test_dir = "../data/pretrain/test"
    
    # --- 修改：擴展字符集和類別數 ---
    characters = '0123456789-' # 添加負號
    num_classes = len(characters) + 1  # 10個數字 + 1個負號 + 1個空白符號CTC = 12
    blank_index = len(characters) # 空白符號的索引現在是 11
    
    # 模型參數
    input_channels = 1  # 灰度圖像
    hidden_size = 256
    # num_classes = 11   # 10個數字（0-9）+ 空白符號CTC <-- 舊的設置
    
    # 訓練參數
    batch_size = 512
    epochs = 200
    learning_rate = 0.001
    validation_split = 0.1
    
    # 圖像預處理
    img_height = 32
    img_width = 128
    
    # 保存路徑
    model_save_path = "./OCR_interface/simpleocr"
    pretrain_model_path = os.path.join(model_save_path, 'best_crnn_model.pth')
    # 設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 修改：增加最大標籤長度以滿足 CTC Loss 要求 ---
    # 添加最大標籤長度
    # 即使實際最長標籤可能小於6，但 CTC Loss 的 CUDA 實現要求此維度至少為6
    max_label_length = 6 # 原本是 5




# 數據集類
class SVHNDataset(Dataset):
    def __init__(self, root_dir, transform=None, char_to_int=None): # 添加 char_to_int 映射
        # 讀取CSV標籤文件
        self.labels_df = pd.read_csv(os.path.join(root_dir, "labels.csv"))
        self.root_dir = root_dir
        self.transform = transform
        # --- 添加：存儲字符到索引的映射 ---
        self.char_to_int = char_to_int
        if self.char_to_int is None:
            # 如果未提供，則創建默認映射 (僅數字)
            # 注意：這假設預訓練數據只有數字。如果預訓練數據也包含負號，
            # 需要確保傳入的 char_to_int 包含負號。
            self.char_to_int = {char: i for i, char in enumerate('0123456789')}
            print("[Warning] SVHNDataset 未提供 char_to_int，使用默認數字映射。")

        # 過濾不存在的圖像文件
        valid_files = []
        for i, row in self.labels_df.iterrows():
            file_path = os.path.join(self.root_dir, row['filename'])
            if os.path.exists(file_path):
                valid_files.append(i)
        
        self.labels_df = self.labels_df.iloc[valid_files].reset_index(drop=True)
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx]['filename'])
        image = Image.open(img_name).convert('L')  # 轉為灰度圖
        
        # 獲取標籤
        label_str = str(self.labels_df.iloc[idx]['label'])
        
        # --- 修改：使用 char_to_int 映射將標籤轉換為整數列表 ---
        try:
            label = [self.char_to_int[c] for c in label_str]
        except KeyError as e:
            print(f"[Error] 標籤 '{label_str}' 包含無效字符: {e}. 跳過此樣本。")
            # 返回一個空樣本或引發異常，取決於您希望如何處理
            # 這裡我們返回 None，需要在 collate_fn 中處理
            return None, None, None
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, len(label)


# 添加自訂的 collate 函數來處理不同長度的標籤和可能的 None 值
def collate_fn(batch):
    # 過濾掉無效的樣本 (getitem 返回 None 的情況)
    batch = [item for item in batch if item[0] is not None]
    if not batch: # 如果過濾後批次為空
        return None, None, None

    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images)
    
    # 將標籤填充到最大長度
    batch_size = len(images)
    target_lengths = torch.IntTensor(label_lengths)
    # --- 修改：使用配置中的 max_label_length ---
    # max_len_in_batch = max(label_lengths) # 或者使用批次內最大長度
    max_len = PretrainConfig.max_label_length # 使用配置的最大長度
    target = torch.zeros(batch_size, max_len).long()
    
    for i, label in enumerate(labels):
        label_tensor = torch.LongTensor(label)
        # 確保標籤長度不超過填充長度
        length = min(len(label), max_len)
        target[i, :length] = label_tensor[:length]
    
    return images, target, target_lengths



# 定義CRNN模型
class CRNN(nn.Module):
    def __init__(self, input_channels, hidden_size, num_classes):
        super(CRNN, self).__init__()
        
        # CNN部分
        self.cnn = nn.Sequential(
            # 4層CNN而非6層
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        # 計算CNN輸出特徵維度
        self.cnn_output_size = 512 * 2  # 根據CNN最後的輸出高度計算
        
        # RNN部分 - 更新輸入維度為1024
        self.rnn = nn.LSTM(
            self.cnn_output_size,  # CNN的輸出維度是512*2=1024
            hidden_size, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True
        )
        
        # 分類器
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # *2 是因為雙向
    
    def forward(self, x):
        # CNN前向傳播
        conv = self.cnn(x)
        
        # 特徵圖重塑為序列
        batch, channel, height, width = conv.size()
        conv = conv.permute(0, 3, 1, 2)  # [B, W, C, H]
        conv = conv.reshape(batch, width, channel * height)  # [B, W, C*H]
        
        # RNN前向傳播 - 直接調用LSTM，不再使用Sequential
        rnn_output, _ = self.rnn(conv)
        
        # 分類器前向傳播
        output = self.classifier(rnn_output)
        
        return output



# 將標籤轉換為模型期望的格式
def labels_to_input(batch_labels, device):
    """將批次標籤轉換為模型輸入期望的格式"""
    max_len = max(len(label) for label in batch_labels)
    targets = torch.zeros(len(batch_labels), max_len, dtype=torch.long).to(device)
    target_lengths = []
    
    for i, label in enumerate(batch_labels):
        targets[i, :len(label)] = torch.tensor(label, dtype=torch.long)
        target_lengths.append(len(label))
        
    return targets, torch.tensor(target_lengths, dtype=torch.long)

# 將模型輸出解碼為數字序列
def decode_predictions(outputs, idx_to_char, blank_index): # 添加 blank_index 參數
    """將模型輸出解碼為數字序列"""
    # 獲取每個時間步最高概率的類別
    _, max_indices = torch.max(outputs, 2)
    
    # 將預測結果轉換為CPU並轉為numpy數組
    max_indices = max_indices.cpu().numpy()
    
    # 儲存解碼後的結果
    decoded_preds = []
    
    # 處理每個樣本
    for sample_indices in max_indices:
        # 移除重複和空白標籤 (CTC解碼)
        previous = -1
        sample_result = []
        
        for index in sample_indices:
            # --- 修改：使用傳入的 blank_index ---
            if index == blank_index: # 跳過空白標籤
                previous = index
                continue
            # 跳過重複標籤
            if index != previous:
                # --- 修改：使用 idx_to_char 映射 ---
                if index in idx_to_char: # 確保索引在映射表中
                    sample_result.append(idx_to_char[index])
                else:
                    print(f"[Warning] 解碼時遇到未知索引: {index}") # 可選：處理未知索引
                previous = index
        
        decoded_preds.append("".join(sample_result)) # 直接組合成字符串
    
    return decoded_preds

# 訓練一個 epoch
def train_epoch(model, loader, criterion, optimizer, device, idx_to_char, blank_index): # 添加映射和 blank 索引
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, total=len(loader))
    for batch_data in pbar:
        # --- 修改：處理 collate_fn 可能返回 None 的情況 ---
        if batch_data[0] is None:
            continue
        images, targets, target_lengths = batch_data

        # 將數據移到設備上
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        # 前向傳播
        outputs = model(images)
        
        # 計算 CTC 損失
        # log_probs: [seq_len, batch, num_classes]
        log_probs = outputs.log_softmax(2).permute(1, 0, 2)
        # input_lengths: [batch] - 每個序列的長度
        input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long).to(device)
        # target_lengths: [batch] - 每個目標標籤的長度
        # targets: [batch, max_label_length] - 填充後的目標標籤

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 計算準確率
        # --- 修改：傳遞 idx_to_char 和 blank_index ---
        predictions = decode_predictions(outputs, idx_to_char, blank_index)
        for i, pred_str in enumerate(predictions):
            # 從targets中獲取原始標籤
            target_length = target_lengths[i].item()
            label_indices = targets[i, :target_length].tolist()
            # --- 修改：使用 idx_to_char 映射真實標籤 ---
            label_str = "".join([idx_to_char[idx] for idx in label_indices if idx in idx_to_char])
            
            if pred_str == label_str:
                correct += 1
            total += 1
        
        # 更新進度條
        pbar.set_description(f"Train Loss: {loss.item():.4f} Acc: {correct/total:.4f}")
    
    return total_loss / len(loader), correct / total

# 驗證
def validate(model, loader, criterion, device, idx_to_char, blank_index): # 添加映射和 blank 索引
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in loader:
            # --- 修改：處理 collate_fn 可能返回 None 的情況 ---
            if batch_data[0] is None:
                continue
            images, targets, target_lengths = batch_data

            # 將數據移到設備上
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # 前向傳播
            outputs = model(images)
            
            # 計算 CTC 損失
            log_probs = outputs.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long).to(device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            total_loss += loss.item()
            
            # 計算準確率
            # --- 修改：傳遞 idx_to_char 和 blank_index ---
            predictions = decode_predictions(outputs, idx_to_char, blank_index)
            for i, pred_str in enumerate(predictions):
                target_length = target_lengths[i].item()
                label_indices = targets[i, :target_length].tolist()
                # --- 修改：使用 idx_to_char 映射真實標籤 ---
                label_str = "".join([idx_to_char[idx] for idx in label_indices if idx in idx_to_char])
                
                if pred_str == label_str:
                    correct += 1
                total += 1
            
    val_loss = total_loss / len(loader)
    val_acc = correct / total if total > 0 else 0
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
    return val_loss, val_acc


def run_pretraining():
    # 設定隨機種子
    torch.manual_seed(42)
    np.random.seed(42)

    config = PretrainConfig()

    # --- 添加：創建字符映射表 ---
    char_to_int = {char: i for i, char in enumerate(config.characters)}
    idx_to_char = {i: char for i, char in enumerate(config.characters)}

    # 確保保存路徑存在
    os.makedirs(config.model_save_path, exist_ok=True)

    # --- 添加：保存字符映射表 ---
    char_map_save_path = os.path.join(config.model_save_path, "char_mapping.json")
    with open(char_map_save_path, 'w', encoding='utf-8') as f:
        json.dump(idx_to_char, f, ensure_ascii=False, indent=4)
    print(f"字符映射表已保存至: {char_map_save_path}")


    # 定義數據預處理
    transform = transforms.Compose([
        transforms.Resize((config.img_height, config.img_width)),
        transforms.RandomRotation(degrees=(-2, 2)),  # 輕微旋轉
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # 輕微顏色抖動
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # 灰度圖標準化
    ])

    # 創建數據集實例
    # --- 修改：傳入 char_to_int ---
    full_dataset = SVHNDataset(root_dir=config.train_dir, transform=transform, char_to_int=char_to_int)

    # 劃分訓練集和驗證集
    val_size = int(len(full_dataset) * config.validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 創建數據加載器
    # --- 修改：使用 collate_fn ---
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # 初始化模型、損失函數和優化器
    # --- 修改：傳遞正確的 num_classes ---
    model = CRNN(input_channels=config.input_channels, hidden_size=config.hidden_size, num_classes=config.num_classes).to(config.device)
    if os.path.exists(config.pretrain_model_path):
        model.load_state_dict(torch.load(config.pretrain_model_path))
        print(f"已從 {config.pretrain_model_path} 加載預訓練模型")
    else:
        print(f"未找到預訓練模型 {config.pretrain_model_path}")
    
    # --- 修改：設置 blank 索引 ---
    criterion = CTCLoss(blank=config.blank_index, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True) # 添加學習率調度器

    # 訓練循環
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"開始在 {config.device} 上訓練...")
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        # --- 修改：傳遞映射和 blank 索引 ---
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.device, idx_to_char, config.blank_index)
        val_loss, val_acc = validate(model, val_loader, criterion, config.device, idx_to_char, config.blank_index)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss) # 更新學習率

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.pretrain_model_path)
            print(f"在 Epoch {epoch+1} 保存了新的最佳模型到 {config.pretrain_model_path} (Val Loss: {val_loss:.4f})")

    print("\n訓練完成！")

    # 繪製損失和準確率曲線
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, config.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.epochs + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, config.epochs + 1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_save_path = os.path.join(config.model_save_path, "training_curves.png")
    plt.savefig(plot_save_path)
    print(f"訓練曲線圖已保存至: {plot_save_path}")
    # plt.show() # 如果在本地運行可以取消註釋

# --- 新增：單張圖片預處理函數 ---
def preprocess_single_image(image_path, img_height, img_width, device):
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # 歸一化到 [-1, 1]
    ])
    try:
        image = Image.open(image_path).convert('L') # 確保是灰度圖
    except FileNotFoundError:
        print(f"錯誤: 測試圖片 {image_path} 未找到。")
        raise
    except Exception as e:
        print(f"錯誤: 打開或轉換圖片 {image_path} 時出錯: {e}")
        raise
        
    image = transform(image)
    image = image.unsqueeze(0) # 添加 batch 維度 [C, H, W] -> [B, C, H, W]
    return image.to(device)

# --- 新增：單張圖片測試函數 ---
def test_single_image(model_path_to_test, char_map_path_to_test, image_path_to_test, base_config):
    """
    使用指定的CRNN模型測試單張圖片。
    base_config: 用於獲取如 img_height, img_width, device, input_channels, hidden_size 等基礎模型結構參數
    """
    device = torch.device(base_config.device)
    print(f"\n--- 開始單張圖片測試 ---")
    print(f"模型權重: {model_path_to_test}")
    print(f"字符映射: {char_map_path_to_test}")
    print(f"測試圖片: {image_path_to_test}")

    # 1. 加載字符映射表
    try:
        with open(char_map_path_to_test, 'r', encoding='utf-8') as f:
            # JSON 加載後 key 是字符串，需要轉為整型
            loaded_idx_to_char_str_keys = json.load(f) 
            loaded_idx_to_char = {int(k): v for k, v in loaded_idx_to_char_str_keys.items()}
        print(f"字符映射表已從 {char_map_path_to_test} 加載。包含 {len(loaded_idx_to_char)} 個字符。")
    except FileNotFoundError:
        print(f"錯誤: 字符映射文件 {char_map_path_to_test} 未找到。")
        return
    except Exception as e:
        print(f"錯誤: 加載或解析字符映射表 {char_map_path_to_test} 時出錯: {e}")
        return

    # 2. 根據加載的字符映射表確定模型參數
    num_chars_from_map = len(loaded_idx_to_char)
    num_model_classes_for_test = num_chars_from_map + 1  # 字符數 + CTC空白
    blank_idx_for_decode = num_chars_from_map          # CTC空白索引為 M

    print(f"  推斷模型類別數: {num_model_classes_for_test}")
    print(f"  推斷CTC空白索引: {blank_idx_for_decode}")

    # 3. 初始化模型
    # 使用 base_config 中的結構參數，但 num_classes 來自加載的映射表
    try:
        model_to_test = CRNN(input_channels=base_config.input_channels,
                             hidden_size=base_config.hidden_size,
                             num_classes=num_model_classes_for_test).to(device)
    except Exception as e:
        print(f"錯誤: 初始化CRNN模型時出錯 (請檢查 base_config 中的 input_channels/hidden_size 是否與模型兼容): {e}")
        return

    # 4. 加載模型權重
    if not os.path.exists(model_path_to_test):
        print(f"錯誤: 模型權重文件 {model_path_to_test} 未找到。")
        return
    try:
        model_to_test.load_state_dict(torch.load(model_path_to_test, map_location=device))
        model_to_test.eval()
        print(f"模型權重已從 {model_path_to_test} 加載。")
    except Exception as e:
        print(f"錯誤: 加載模型權重 {model_path_to_test} 時出錯: {e}")
        print(f"  這通常意味著模型架構 (input_channels, hidden_size, num_classes) 與權重文件不匹配。")
        print(f"  期望 num_classes={num_model_classes_for_test} (基於字符映射表)。")
        return

    # 5. 預處理圖片
    try:
        img_tensor = preprocess_single_image(image_path_to_test, 
                                            base_config.img_height, 
                                            base_config.img_width, 
                                            device)
    except Exception as e:
        # preprocess_single_image 內部已打印錯誤，此處直接返回
        return

    # 6. 模型推理和解碼
    try:
        with torch.no_grad():
            preds_tensor = model_to_test(img_tensor) # Output: [T, B, C]
        
        # 使用加載的 idx_to_char 和推斷的 blank_idx 進行解碼
        decoded_texts = decode_predictions(preds_tensor.cpu(), loaded_idx_to_char, blank_idx_for_decode)
        
        if decoded_texts:
            print(f"\n識別結果: \"{decoded_texts[0]}\"")
        else:
            print("識別結果為空。")
            
    except Exception as e:
        print(f"錯誤: 模型推理或解碼時出錯: {e}")

# --- 主函數入口 ---
if __name__ == "__main__":
    # 設置隨機種子以保證可複現性 (如果需要)
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser(description="CRNN 預訓練與測試腳本")
    parser.add_argument("--run_mode", type=str, default="train", choices=["train", "test"],
                        help="運行模式: 'train' 進行訓練, 'test' 進行單圖測試。 (默認: train)")
    
    # 測試模式參數
    parser.add_argument("--test_image_path", type=str, default=None,
                        help="[測試模式] 要進行OCR測試的單張圖片路徑。")
    parser.add_argument("--test_model_path", type=str, default='OCR_interface/simpleocr/best_crnn_model.pth',
                        help="[測試模式] 用於測試的CRNN模型權重文件 (.pth) 路徑。")
    parser.add_argument("--test_char_map_path", type=str, default='OCR_interface/simpleocr/char_mapping.json',
                        help="[測試模式] 用於測試的字符映射表 (char_mapping.json) 路徑。")
    
    # 可以添加更多參數來覆蓋 PretrainConfig 中的默認值，例如：
    # parser.add_argument("--characters", type=str, help="覆蓋訓練時的字符集")
    # parser.add_argument("--epochs", type=int, help="覆蓋訓練的 epoch 數量")

    args = parser.parse_args()
    
    config = PretrainConfig() # 加載默認配置

    # 如果需要，可以在這裡用 args 中的值更新 config 對象的屬性
    # if args.characters: config.characters = args.characters (需要重新計算依賴項)
    # if args.epochs: config.epochs = args.epochs

    if args.run_mode == "test":
        print("進入測試模式...")
        if not all([args.test_image_path]):
            parser.error("[測試模式] 必須提供 --test_image_path參數。")
        else:
            test_single_image(
                model_path_to_test=args.test_model_path,
                char_map_path_to_test=args.test_char_map_path,
                image_path_to_test=args.test_image_path,
                base_config=config # 傳遞基礎配置以獲取模型結構參數
            )
    elif args.run_mode == "train":
        print("進入訓練模式...")
        # 在開始訓練前，重新基於 (可能已更新的) config.characters 初始化依賴項
        config.char_to_int = {char: i for i, char in enumerate(config.characters)}
        config.char_to_idx = {char: i for i, char in enumerate(config.characters)}
        config.idx_to_char = {i: char for i, char in enumerate(config.characters)}
        config.num_classes = len(config.characters) + 1
        config.blank_index = len(config.characters)
        config.char_map_save_path = os.path.join(config.model_save_path, f"pretrain_char_mapping_{''.join(filter(str.isalnum, config.characters))}.json") # 根據字符集命名
        config.pretrain_model_path = os.path.join(config.model_save_path, f"best_pretrain_crnn_model_{''.join(filter(str.isalnum, config.characters))}.pth")


        run_pretraining()
    else:
        print(f"錯誤: 未知的運行模式 '{args.run_mode}'")
