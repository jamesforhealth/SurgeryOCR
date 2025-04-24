import os
import random
import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter, ImageOps
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
# 設置隨機種子以保證可重現性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 自定義隨機擦除類
class CustomRandomErasing:
    """對張量圖像執行隨機擦除操作"""
    def __init__(self, scale=(0.02, 0.2), ratio=(0.3, 3.3), p=0.5, value=0):
        self.scale = scale
        self.ratio = ratio
        self.p = p
        self.value = value

    def __call__(self, img):
        if random.uniform(0, 1) >= self.p:
            return img

        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for _ in range(10):  # 嘗試最多10次
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            h = int(round(np.sqrt(erase_area * aspect_ratio)))
            w = int(round(np.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                x1 = random.randint(0, img_w - w)
                y1 = random.randint(0, img_h - h)
                if img.dtype == torch.float32:
                    img[:, y1:y1 + h, x1:x1 + w] = float(self.value)
                else:
                    img[:, y1:y1 + h, x1:x1 + w] = self.value
                return img

        return img  # 如果無法找到合適區域，返回原圖

# 強大的數據增強類
class OCRDataAugmentation:
    def __init__(self, img_height, img_width, prob=0.5):
        self.img_height = img_height
        self.img_width = img_width
        self.prob = prob
        self.to_tensor = transforms.ToTensor()
        self.random_erasing = CustomRandomErasing(p=0.3)

        # Albumentations增強流水線
        self.transform = A.Compose([
            # 1. 幾何變換
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=5, 
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7
            ),
            A.GridDistortion(
                num_steps=5, distort_limit=0.1, 
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3
            ),
            A.ElasticTransform(
                alpha=1, sigma=50, 
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2
            ),
            
            # 2. 亮度和對比度變換
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.5),
            
            # 3. 噪聲和模糊
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=3),
            ], p=0.5),
            
            # 4. 質量降低
            A.OneOf([
                A.ImageCompression(quality_lower=70, quality_upper=100),
                A.JpegCompression(quality_lower=70, quality_upper=100),
            ], p=0.3),
            
            # 5. 遮蓋和擦除
            A.CoarseDropout(
                max_holes=8, max_height=5, max_width=5, 
                fill_value=0, p=0.3
            ),
        ])

        # 額外的 PIL 特有轉換
        self.extra_transforms = [
            lambda img: img.filter(ImageFilter.MedianFilter(size=3)),
            lambda img: img.filter(ImageFilter.DETAIL),
            lambda img: img.filter(ImageFilter.UnsharpMask(radius=2, percent=150)),
            lambda img: ImageOps.autocontrast(img, cutoff=0.5),
            lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.1)),
            lambda img: ImageOps.equalize(img),
        ]

    def apply_pil_transforms(self, image, num_transforms=1):
        """應用隨機的PIL轉換"""
        if random.random() > self.prob:
            return image
            
        transforms_to_apply = random.sample(self.extra_transforms, 
                                           k=min(num_transforms, len(self.extra_transforms)))
        for transform in transforms_to_apply:
            if random.random() < 0.5:  # 每個轉換還有50%的幾率被應用
                try:
                    image = transform(image)
                except Exception as e:
                    print(f"PIL轉換出錯: {e}")
        
        return image

    def apply_aug(self, img):
        """應用增強到輸入圖像"""
        if random.random() < self.prob:
            # 將圖像格式調整為 augmentation 需要的格式
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
                img = np.repeat(img, 3, axis=2)
            
            # 應用 Albumentations 增強
            augmented = self.transform(image=img)
            img = augmented['image']
            
            # 確保圖像在正確的範圍內
            if img.dtype == np.float32:
                img = np.clip(img, 0.0, 1.0)
        
        return img

    def __call__(self, image):
        """
        對輸入圖像應用數據增強。
        Args:
            image: PIL圖像對象
        Returns:
            torch.Tensor: 增強後的圖像張量
        """
        # 1. 首先進行PIL特有的轉換
        img_pil = self.apply_pil_transforms(image, num_transforms=2)
        
        # 2. 轉換為numpy格式，供Albumentations使用
        img_np = np.array(img_pil)
        
        # 3. 應用Albumentations轉換
        if random.random() < self.prob:
            try:
                img_np = self.transform(image=img_np)['image']
            except Exception as e:
                print(f"Albumentations轉換出錯: {e}")
        
        # 4. 調整大小
        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.resize((self.img_width, self.img_height), Image.LANCZOS)
        
        # 5. 轉換為PyTorch張量
        img_tensor = self.to_tensor(img_pil)
        
        # 6. 標準化
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        img_tensor = normalize(img_tensor)
        
        # 7. 應用隨機擦除
        if random.random() < self.prob:
            img_tensor = self.random_erasing(img_tensor)
        
        return img_tensor

# 增強資料集類別
class AugmentedDataset(Dataset):
    def __init__(self, dataset, augmenter=None, multiplier=1, is_train=True):
        self.dataset = dataset
        self.augmenter = augmenter
        self.multiplier = multiplier  # 每個原始樣本產生的增強版本數量
        self.is_train = is_train
        
        # 確保至少有一個版本（原始版本）
        self.multiplier = max(1, self.multiplier)
        print(f"初始化數據集: multiplier={self.multiplier}, len(dataset)={len(dataset)}")
        
    def __len__(self):
        # 修改這一行: 乘以倍增器來擴大數據集大小
        return len(self.dataset) * self.multiplier
    
    def __getitem__(self, idx):
        # 計算原始樣本的索引和增強版本號
        original_idx = idx // self.multiplier
        augmentation_version = idx % self.multiplier
        
        # 確保原始索引在範圍內
        original_idx = original_idx % len(self.dataset)
        
        # 獲取原始樣本
        image, label, label_length = self.dataset[original_idx]
        
        # 如果是驗證集或不需要增強，直接返回原始樣本
        if not self.is_train or self.augmenter is None:
            return image, label, label_length
        
        # 如果是第一個版本(version 0)，返回原始樣本
        if augmentation_version == 0:
            return image, label, label_length
        
        # 否則進行增強
        # 將圖像轉換為numpy數組進行增強
        image_np = image.permute(1, 2, 0).numpy()  # CHW -> HWC
        
        # 使用增強器生成不同版本
        augmented = self.augmenter.apply_aug(image_np)
        
        # 轉回tensor
        augmented_tensor = torch.from_numpy(augmented).permute(2, 0, 1)  # HWC -> CHW
        
        return augmented_tensor, label, label_length

# 創建增強數據加載器
def create_augmented_dataloaders(base_dataset, config, augmenter=None, multiplier=1):
    """創建包含增強數據的訓練和驗證數據加載器"""
    
    # 獲取數據集大小
    dataset_size = len(base_dataset)
    
    # 確定訓練/驗證分割
    train_size = int(dataset_size * 0.9)  # 90% 用於訓練
    val_size = dataset_size - train_size  # 10% 用於驗證
    
    # 隨機分割數據集
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 創建訓練和驗證子集
    train_subset = torch.utils.data.Subset(base_dataset, train_indices)
    val_subset = torch.utils.data.Subset(base_dataset, val_indices)
    
    print(f"原始訓練集大小: {len(train_subset)}")
    print(f"原始驗證集大小: {len(val_subset)}")
    
    # 使用增強器創建擴增數據集
    train_augmented = AugmentedDataset(
        train_subset, augmenter=augmenter, 
        multiplier=multiplier, is_train=True
    )
    
    # 驗證集不使用數據增強
    val_augmented = AugmentedDataset(
        val_subset, augmenter=None, 
        multiplier=1, is_train=False
    )
    
    print(f"增強後訓練集大小: {len(train_augmented)}")
    print(f"增強後驗證集大小: {len(val_augmented)}")
    
    # 獲取CPU核心數用於多進程數據加載
    num_workers = 0 #max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    
    # 定義collate_fn
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
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_augmented,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True  # 丟棄最後不完整的批次
    )
    
    val_loader = DataLoader(
        val_augmented,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

# 使用示例
if __name__ == "__main__":
    from pretrain_crnn import Config, SVHNDataset, transform
    
    # 設置隨機種子
    set_seed(42)
    
    # 創建配置
    config = Config()
    
    # 加載基礎數據集
    base_dataset = SVHNDataset(config.train_dir, transform=transform)
    
    # 創建數據增強器
    augmenter = OCRDataAugmentation(
        img_height=config.img_height,
        img_width=config.img_width,
        prob=0.9
    )
    
    # 創建增強數據加載器
    train_loader, val_loader = create_augmented_dataloaders(
        base_dataset=base_dataset,
        config=config,
        augmenter=augmenter,
        multiplier=5  # 每個原始樣本生成5個增強樣本
    )
    
    print(f"訓練批次數: {len(train_loader)}")
    print(f"驗證批次數: {len(val_loader)}")
    
    # 檢查單個批次
    for images, targets, target_lengths in train_loader:
        print(f"批次大小: {images.shape}")
        print(f"目標形狀: {targets.shape}")
        print(f"目標長度: {target_lengths}")
        break 