#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EasyOCR模型實現
"""
import easyocr
import numpy as np
from .base import BaseOCRModel
from PIL import Image
import cv2

class EasyOCRModel(BaseOCRModel):
    """EasyOCR引擎包裝類"""
    
    def __init__(self, gpu=True, lang_list=None, model_storage_directory=None,
                 confidence_threshold=0.3, allowlist="0123456789-+",
                 # 以下是 easyocr.Reader 的其他可選參數，如果需要從外部配置的話
                 download_enabled=True, user_network_directory=None,
                 recog_network='standard', detector=True, recognizer=True, verbose=True,
                 quantize=True, cudnn_benchmark=False, **kwargs_extra_ignored): # 捕獲多餘的kwargs以避免錯誤

        if lang_list is None:
            lang_list = ['en']

        self.gpu = gpu
        self.lang_list = lang_list
        self.model_storage_directory = model_storage_directory
        self.confidence_threshold = confidence_threshold # 在此存儲，用於 recognize 方法
        self.allowlist = allowlist # 在此存儲，用於 recognize 方法

        # 準備僅easyocr.Reader構造函數支持的參數
        reader_params = {
            'lang_list': self.lang_list,
            'gpu': self.gpu,
            'model_storage_directory': self.model_storage_directory,
            'download_enabled': download_enabled,
            'user_network_directory': user_network_directory,
            'recog_network': recog_network,
            'detector': detector,
            'recognizer': recognizer,
            'verbose': verbose,
            'quantize': quantize,
            'cudnn_benchmark': cudnn_benchmark
        }
        
        try:
            self.reader = easyocr.Reader(**reader_params)
        except TypeError as e:
            print(f"初始化 EasyOCR Reader 時發生 TypeError: {e}")
            print(f"使用的參數: {reader_params}")
            print("請檢查 EasyOCR 版本與所傳參數是否兼容。嘗試使用最少參數初始化...")
            try:
                self.reader = easyocr.Reader(lang_list=self.lang_list, gpu=self.gpu, verbose=verbose)
            except Exception as fallback_e:
                print(f"使用最少參數初始化 EasyOCR Reader 失敗: {fallback_e}")
                raise e # 重新拋出原始錯誤
        
    def recognize(self, image):
        """
        使用EasyOCR識別圖像中的文字。
        此方法包含置信度過濾、按x座標排序和文本拼接。
        
        參數:
            image: 輸入圖像，numpy數組 (BGR) 或 PIL.Image
            
        返回:
            result: 字符串，識別結果
            confidence: 浮點數，平均置信度 (0-1)
        """
        # 確保輸入是numpy數組
        if isinstance(image, Image.Image):
            # 如果是PIL Image，轉換為numpy array
            # EasyOCR 通常期望 BGR 格式的 numpy array
            if image.mode == 'RGB':
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image = np.array(image)
        elif not isinstance(image, np.ndarray):
            raise TypeError("輸入圖像必須是 PIL.Image 或 numpy.ndarray 類型")

        # 執行OCR
        ocr_results = self.reader.readtext(
            image,
            allowlist=self.allowlist, # 在 readtext 中使用存儲的 allowlist
            detail=1,
            paragraph=False
            # 其他 readtext 參數可以根據需要添加
        )
        
        # 如果沒有結果返回空字符串
        if not ocr_results:
            return "", 0.0
            
        # 根據置信度過濾結果，並記錄bbox的左上角x座標
        filtered_with_pos_conf = [
            (txt, bbox[0][0], conf) 
            for (bbox, txt, conf) in ocr_results
            if conf >= self.confidence_threshold # 在 recognize 中使用存儲的 confidence_threshold
        ]
        
        # 按 x 座標 (左→右) 排序
        filtered_with_pos_conf.sort(key=lambda x: x[1])
        
        # 只取文字部分拼接
        text = " ".join(item[0] for item in filtered_with_pos_conf)
        
        # 計算平均置信度
        if filtered_with_pos_conf:
            avg_confidence = sum(item[2] for item in filtered_with_pos_conf) / len(filtered_with_pos_conf)
        else:
            # 如果過濾後沒有結果，則文本為空，置信度為0
            text = ""
            avg_confidence = 0.0
            
        return text, avg_confidence
    
    def get_model_info(self):
        """
        獲取模型信息
        
        返回:
            dict: 包含模型名稱、版本等信息
        """
        return {
            "name": "EasyOCR",
            "languages": self.lang_list,
            "gpu_enabled": self.gpu,
            "allowlist": self.allowlist, # 在模型信息中包含 allowlist
            "confidence_threshold": self.confidence_threshold # 在模型信息中包含閾值
        } 