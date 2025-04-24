import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os

# CRNN模型定義
class CRNN(nn.Module):
    def __init__(self, imgH=32, nc=3, nclass=37, nh=256):
        super(CRNN, self).__init__()
        
        # VGG風格的卷積層
        self.cnn = nn.Sequential(
            # 與論文一致的VGG結構
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)
        )
        
        # 將特徵圖按列提取特徵
        self.map_to_seq = MapToSequence()
        
        # 雙層雙向LSTM
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )
        
    def forward(self, x):
        # 卷積特徵提取
        conv = self.cnn(x)
        
        # 特徵序列化（按列）
        b, c, h, w = conv.size()
        assert h == 1, "卷積特徵的高度應為1"
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)  # [b, w, c]
        
        # RNN序列預測
        output = self.rnn(conv)
        
        # 調整為CTC格式 [序列長度, 批次大小, 類別數]
        output = output.permute(1, 0, 2)
        
        return output

# 雙向LSTM模組
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output

# Map-to-Sequence層
class MapToSequence(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        assert h == 1, "高度必須為1"
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        return x

# 數據集類
class TextImageDataset(Dataset):
    def __init__(self, image_paths, labels, char_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 將文本標籤轉換為索引
        label_str = self.labels[idx]
        label_indices = [self.char_to_idx[c] for c in label_str if c in self.char_to_idx]
        label_length = len(label_indices)
        
        return image, torch.tensor(label_indices, dtype=torch.long), label_length

# 數據加載器的collate_fn
def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(labels)
    target_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return images, targets, target_lengths

# 訓練函數
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, targets, target_lengths in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(images)
        
        # 計算輸入序列長度
        input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long).to(device)
        
        # CTC損失
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    return running_loss / len(dataloader.dataset)

# 主要訓練邏輯
def main():
    # 設置參數
    imgH = 32  # 圖像高度
    nc = 3     # 輸入通道數（RGB=3）
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz "  # 字元表
    nclass = len(alphabet) + 1  # +1 為CTC空白符號
    nh = 256   # LSTM隱藏層大小
    
    # 創建字元映射
    char_to_idx = {char: i+1 for i, char in enumerate(alphabet)}  # 0保留給CTC空白
    idx_to_char = {i+1: char for i, char in enumerate(alphabet)}
    
    # 創建模型
    model = CRNN(imgH=imgH, nc=nc, nclass=nclass, nh=nh)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 定義損失函數和優化器
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.ADADELTA(model.parameters(), lr=1.0)  # 論文使用ADADELTA
    
    # 數據轉換
    transform = transforms.Compose([
        transforms.Resize((imgH, 100)),  # 統一高度，寬度可變
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 假設數據集已加載
    # train_dataset = TextImageDataset(image_paths, labels, char_to_idx, transform)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    
    # 訓練循環
    num_epochs = 50
    for epoch in range(num_epochs):
        # train_loss = train(model, train_loader, criterion, optimizer, device)
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')
        pass

if __name__ == "__main__":
    main()