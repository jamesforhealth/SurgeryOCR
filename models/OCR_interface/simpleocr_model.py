#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自訓練CRNN模型實現
"""
import os
import torch
# import torch.nn as nn # CRNN類將從 pretrain_crnn 導入
# import torch.nn.functional as F # 如果CRNN的forward方法使用了F，則可能仍需導入或由CRNN內部處理
import numpy as np
import cv2
from .base import BaseOCRModel
import sys

# 這個導入現在依賴於 models/pretrain_crnn.py 中有名為 CRNN 的類和名為 characters 的變量
from ..pretrain_crnn import CRNN, characters as PRETRAINED_CHARACTERS

class CRNNModel(BaseOCRModel):
    """自訓練CRNN模型"""
    
    def __init__(self, model_path, characters=None, gpu=True, **kwargs):
        """
        初始化CRNN模型
        
        參數:
            model_path: 模型權重文件路徑
            characters: 字符集。如果為 None，則使用從 pretrain_crnn.py 導入的 PRETRAINED_CHARACTERS。
            gpu: 是否使用GPU
            **kwargs: 其他參數 (例如，如果 CRNN 需要更多固定參數，可以考慮從這裡傳入)
        """
        self.model_path = model_path
        
        if characters is None:
            self.characters = PRETRAINED_CHARACTERS 
            print(f"使用從 pretrain_crnn.py 導入的字符集: {self.characters}")
        else:
            self.characters = characters

        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.characters)} # 索引從1開始，0為CTC空白
        self.idx_to_char = {i + 1: char for i, char in enumerate(self.characters)}
        self.num_classes = len(self.characters) + 1 # 實際類別數 + CTC空白符
        
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        
        # 假設 CRNN 的 input_channels 和 hidden_size 可以從 kwargs 獲取或有默認值
        # 這些值應與 pretrain_crnn.py 中 CRNN 類的期望一致
        self.crnn_input_channels = kwargs.get('input_channels', 1) # 默認為1 (灰度圖)
        self.crnn_hidden_size = kwargs.get('hidden_size', 256)   # 默認為256

        self._load_model()

    def _load_model(self):
        """加載CRNN模型"""
        try:
            # 使用 self.crnn_input_channels 和 self.crnn_hidden_size
            self.model = CRNN(
                input_channels=self.crnn_input_channels,
                hidden_size=self.crnn_hidden_size,
                num_classes=self.num_classes
                # 如果 pretrain_crnn.py 中的 CRNN.__init__ 還有其他必需參數，
                # 需要確保它們也通過 kwargs 傳遞或在此處提供默認值。
            )

            if os.path.exists(self.model_path):
                print(f"從 {self.model_path} 加載CRNN模型權重")
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            else:
                print(f"警告: CRNN模型文件 {self.model_path} 不存在。模型將使用隨機初始化的權重。")
            
            self.model.to(self.device)
            self.model.eval()
            print("CRNN模型加載成功。")

        except Exception as e:
            # 這裡捕獲了初始化失敗的錯誤
            raise RuntimeError(f"初始化CRNN模型失敗: {e}")
        
    def preprocess_image(self, image):
        """
        圖像預處理
        
        參數:
            image: 輸入圖像，numpy數組 (BGR)
            
        返回:
            tensor: 預處理後的圖像張量
        """
        # 轉為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 調整大小為固定高度，保持寬高比
        height = 32
        ratio = height / gray.shape[0]
        width = int(gray.shape[1] * ratio)
        resized = cv2.resize(gray, (width, height))
        
        # 標準化
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        
        # 轉為張量，添加批次和通道維度
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor
        
    def decode(self, pred):
        """
        解碼預測結果
        
        參數:
            pred: 模型輸出的預測結果
            
        返回:
            text: 解碼後的文本
        """
        # 獲取每個時間步的最大概率字符索引
        _, max_indices = torch.max(pred, 2)
        
        # 轉為numpy數組
        indices = max_indices.squeeze(0).cpu().numpy()
        
        # 使用CTC解碼（簡單版本 - 合併重複字符並移除空白）
        prev_idx = -1
        text = []
        for idx in indices:
            if idx != 0 and idx != prev_idx:  # 不是空白且不重複
                text.append(self.idx_to_char[idx])
            prev_idx = idx
            
        return ''.join(text)
    
    def recognize(self, image):
        """
        識別圖像中的文字
        
        參數:
            image: 輸入圖像，numpy數組 (BGR)
            
        返回:
            result: 字符串，識別結果
            confidence: 浮點數，置信度估計 (簡化版，固定為0.9)
        """
        with torch.no_grad():
            # 圖像預處理
            tensor = self.preprocess_image(image)
            tensor = tensor.to(self.device)
            
            # 模型推斷
            pred = self.model(tensor)
            
            # 解碼預測結果
            text = self.decode(pred)
            
            # 由於CRNN模型輸出沒有直接提供置信度，這裡使用一個固定值
            # 在實際應用中，可以根據預測概率分布計算置信度
            confidence = 0.9
            
        return text, confidence
    
    def get_model_info(self):
        """
        獲取模型信息
        
        返回:
            dict: 包含模型名稱、版本等信息
        """
        return {
            "name": "CRNN",
            "model_path": self.model_path,
            "characters": self.characters,
            "device": str(self.device)
        } 