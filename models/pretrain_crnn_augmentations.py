import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 導入原始模型和配置
from pretrain_crnn import Config, SVHNDataset, CRNN, validate, config, transform

# 導入數據增強功能
from dataset_augmentations import (
    OCRDataAugmentation,
    create_augmented_dataloaders,
    set_seed
)

# 設置隨機種子以確保可重現性
set_seed(42)

def train_with_augmentations():
    """使用數據增強訓練CRNN模型"""
    # 將配置從原始模型導入
    config.augmentation_multiplier = 50  # 每個原始樣本生成5個增強樣本
    config.early_stopping_patience = 15  # 早停耐心
    config.epochs = 500  # 總訓練時期數
    
    # 確保模型保存目錄存在
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # 加載基礎數據集
    print("加載基礎數據集...")
    base_dataset = SVHNDataset(config.train_dir, transform=transform)
    test_dataset = SVHNDataset(config.test_dir, transform=transform)
    
    # 創建數據增強器
    print("創建數據增強器...")
    augmenter = OCRDataAugmentation(
        img_height=config.img_height,
        img_width=config.img_width,
        prob=0.8  # 80%的概率應用增強
    )
    
    # 創建增強數據加載器
    print("創建數據加載器...")
    train_loader, val_loader = create_augmented_dataloaders(
        base_dataset=base_dataset,
        config=config,
        augmenter=augmenter,
        multiplier=config.augmentation_multiplier
    )
    
    # 添加這些行來確認數據集大小
    print(f"原始數據集大小: {len(base_dataset)}")
    print(f"訓練數據集大小: {len(train_loader.dataset)}")
    print(f"驗證數據集大小: {len(val_loader.dataset)}")
    print(f"訓練批次數: {len(train_loader)}")
    print(f"驗證批次數: {len(val_loader)}")
    
    # 使用與原始模型相同的collate_fn創建測試數據加載器
    def collate_fn(batch):
        images, labels, label_lengths = zip(*batch)
        images = torch.stack(images)
        
        # 將標籤填充到最大長度
        batch_size = len(images)
        target_lengths = torch.IntTensor(label_lengths)
        target = torch.zeros(batch_size, max(label_lengths)).long()
        
        for i, label in enumerate(labels):
            if isinstance(label, list):
                label_tensor = torch.LongTensor(label)
            else:
                label_tensor = label
            target[i, :len(label)] = label_tensor
        
        return images, target, target_lengths
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 創建模型
    print("創建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(
        input_channels=config.input_channels,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes
    ).to(device)
    
    # 載入預訓練權重 (使用weights_only=True以避免安全警告)
    pretrained_path = os.path.join(config.model_save_path, 'best_crnn_model.pth')
    if os.path.exists(pretrained_path):
        try:
            checkpoint = torch.load(pretrained_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("已載入預訓練模型權重")
        except Exception as e:
            print(f"載入預訓練模型時出錯: {e}")
            print("將使用隨機初始化的模型")
    
    # 創建損失函數和優化器
    criterion = torch.nn.CTCLoss(blank=10, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 檢查是否存在已訓練的模型
    best_model_path = os.path.join(config.model_save_path, 'best_model_augmented.pth')
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['val_acc']
            print(f"已載入已有的最佳模型，起始Epoch: {start_epoch}，最佳驗證準確度: {best_val_acc:.4f}")
        except Exception as e:
            print(f"嘗試載入已有模型時出錯: {e}")
            print("將使用隨機初始化的模型開始訓練。")
            start_epoch = 1
            best_val_acc = 0.0
    else:
        print("未找到已有模型，將使用隨機初始化的模型開始訓練。")
        start_epoch = 1
        best_val_acc = 0.0
    
    # 訓練函數
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        total_accuracy = 0
        batch_count = 0
        
        pbar = tqdm(loader, desc="訓練中")
        for images, targets, target_lengths in pbar:
            # 將數據移到設備上
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # 前向傳播
            outputs = model(images)
            
            # 計算 CTC 損失
            input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long).to(device)
            loss = criterion(outputs.log_softmax(2).permute(1, 0, 2), targets, input_lengths, target_lengths)
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 計算準確率 (簡化版)
            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        return total_loss / batch_count
    
    # 訓練循環
    print(f"開始訓練，總共{config.epochs}個Epoch...")
    train_losses = []
    val_losses = []
    val_accs = []
    early_stop_counter = 0
    
    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        
        # 訓練
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print(f"訓練損失: {train_loss:.4f}")
        
        # 驗證
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}")
        
        # 學習率調整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            torch.save(checkpoint, best_model_path)
            print(f"保存最佳模型，驗證準確率: {best_val_acc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"驗證準確率未提升 ({early_stop_counter}/{config.early_stopping_patience})")
            
            # 早停檢查
            if early_stop_counter >= config.early_stopping_patience:
                print(f"\n驗證準確率連續 {config.early_stopping_patience} 個 epoch 未提升，觸發早停。")
                break
        
        # 每10個epoch保存一次訓練曲線圖
        if epoch % 10 == 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(val_accs, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(config.model_save_path, 'training_curves.png'))
            plt.close()
    
    print("\n訓練完成！")
    
    # 載入最佳模型進行測試
    print("載入最佳模型進行測試...")
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # 在測試集上評估
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"測試損失: {test_loss:.4f}, 測試準確率: {test_acc:.4f}")

if __name__ == "__main__":
    train_with_augmentations() 