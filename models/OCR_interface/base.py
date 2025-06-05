#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR模型基礎接口類
"""
from abc import ABC, abstractmethod

class BaseOCRModel(ABC):
    """OCR模型的抽象基礎類"""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """初始化模型"""
        pass
    
    @abstractmethod
    def recognize(self, image):
        """
        識別圖像中的文字
        
        參數:
            image: 輸入圖像，numpy數組 (BGR)
            
        返回:
            result: 字符串，識別結果
            confidence: 浮點數，置信度 (0-1)
        """
        pass
    
    @abstractmethod
    def get_model_info(self):
        """
        獲取模型信息
        
        返回:
            dict: 包含模型名稱、版本等信息
        """
        pass
