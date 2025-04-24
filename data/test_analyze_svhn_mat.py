#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解析 SVHN 數據集 MAT 檔案並顯示其內容結構
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import os
from tqdm import tqdm
import sys
def analyze_mat_file(mat_file_path):
    """
    分析 MAT 檔案的結構並顯示基本信息
    
    Args:
        mat_file_path: MAT 檔案的路徑
    """
    print(f"分析 MAT 檔案: {mat_file_path}")
    
    # 嘗試使用 scipy.io.loadmat 載入 (適用於較舊版本的 MAT 檔案)
    try:
        data = sio.loadmat(mat_file_path)
        print("成功使用 scipy.io.loadmat 載入")
        
        # 顯示頂級變量
        print("頂級變量:")
        for key in data.keys():
            if not key.startswith('__'):  # 跳過內部變量
                print(f"- {key}: {type(data[key])}")
                if isinstance(data[key], np.ndarray):
                    print(f"  形狀: {data[key].shape}")
                    print(f"  數據類型: {data[key].dtype}")
        
        # 如果有 'X' 和 'y' 變量 (常見的 SVHN 格式)
        if 'X' in data and 'y' in data:
            X = data['X']  # 圖像數據
            y = data['y']  # 標籤
            print(f"\n圖像數據 'X' 形狀: {X.shape}")
            print(f"標籤 'y' 形狀: {y.shape}")
            
            # 顯示幾個樣本
            sample_indices = np.random.choice(y.shape[0], min(5, y.shape[0]), replace=False)
            for i, idx in enumerate(sample_indices):
                print(f"樣本 {i+1}: 標籤 = {y[idx][0]}")
            
            return data
            
        # 如果有 'digitStruct' 變量 (SVHN 全版本格式)
        elif 'digitStruct' in data:
            print("\n數據包含 'digitStruct' 結構")
            digit_struct = data['digitStruct']
            print(f"digitStruct 類型: {type(digit_struct)}")
            print(f"digitStruct 形狀: {digit_struct.shape if hasattr(digit_struct, 'shape') else 'N/A'}")
            
            # 顯示 digitStruct 的字段
            if hasattr(digit_struct, 'dtype'):
                print("digitStruct 字段:")
                for field_name in digit_struct.dtype.names:
                    print(f"- {field_name}")
            
            return data
            
    except Exception as e:
        print(f"使用 scipy.io.loadmat 載入失敗: {e}")
        
    # 嘗試使用 h5py 載入 (適用於較新版本的 MAT 檔案，HDF5 格式)
    try:
        with h5py.File(mat_file_path, 'r') as f:
            print("成功使用 h5py 載入")
            
            # 顯示頂級組
            print("頂級組:")
            for key in f.keys():
                print(f"- {key}: {type(f[key])}")
                if isinstance(f[key], h5py.Dataset):
                    print(f"  形狀: {f[key].shape}")
                    print(f"  數據類型: {f[key].dtype}")
                elif isinstance(f[key], h5py.Group):
                    print(f"  群組包含: {list(f[key].keys())}")
            
            # 如果有 'digitStruct' 群組 (常見於 SVHN MAT 文件)
            if 'digitStruct' in f:
                digit_struct = f['digitStruct']
                print("\n發現 'digitStruct' 群組")
                
                # 探索內部結構
                if 'name' in digit_struct:
                    print("包含 'name' 數據集")
                    names_dataset = digit_struct['name']
                    print(f"名稱數據集形狀: {names_dataset.shape}")
                    
                    # 嘗試取得前幾個文件名
                    try:
                        name_refs = [names_dataset[i][0] for i in range(min(5, names_dataset.shape[0]))]
                        names = [f[ref].value.tobytes().decode('utf-8') for ref in name_refs]
                        print("前幾個文件名:")
                        for name in names:
                            print(f"- {name}")
                    except Exception as e:
                        print(f"獲取文件名失敗: {e}")
                
                if 'bbox' in digit_struct:
                    print("包含 'bbox' 數據集")
                    bbox_dataset = digit_struct['bbox']
                    print(f"邊界框數據集形狀: {bbox_dataset.shape}")
                    
                    # 嘗試獲取前幾個邊界框信息
                    try:
                        # 這裡根據 SVHN 格式具體結構調整
                        print("邊界框結構:")
                        for field in ['left', 'top', 'width', 'height', 'label']:
                            if field in bbox_dataset:
                                print(f"- {field} 數據集形狀: {bbox_dataset[field].shape}")
                    except Exception as e:
                        print(f"獲取邊界框信息失敗: {e}")
            
            return f
            
    except Exception as e:
        print(f"使用 h5py 載入失敗: {e}")
    
    print("無法識別 MAT 檔案格式")
    return None

def extract_svhn_data(mat_file_path, output_dir, max_samples=None):
    """
    從 SVHN MAT 檔案中提取圖像和標籤，保存為單獨的文件
    
    Args:
        mat_file_path: MAT 檔案的路徑
        output_dir: 保存提取數據的目錄
        max_samples: 最大提取樣本數 (None 表示全部提取)
    """
    print(f"從 {mat_file_path} 提取數據...")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 嘗試使用 scipy.io.loadmat 載入
    try:
        data = sio.loadmat(mat_file_path)
        
        # 檢查是否有 'X' 和 'y' 變量 (基本 SVHN 格式)
        if 'X' in data and 'y' in data:
            X = data['X']  # 圖像數據，格式可能是 [高度, 寬度, 通道, 樣本數]
            y = data['y']  # 標籤，格式可能是 [樣本數, 1]
            
            # 確保 y 的標籤中 10 代表 0 (SVHN 特性)
            y[y == 10] = 0
            
            # 確定樣本總數
            num_samples = X.shape[3] if X.ndim == 4 else X.shape[0]
            samples_to_extract = min(num_samples, max_samples or num_samples)
            
            print(f"共找到 {num_samples} 個樣本，將提取 {samples_to_extract} 個")
            
            # 創建標籤字典 {文件名: 標籤}
            labels_dict = {}
            
            # 提取並保存圖像和標籤
            for i in tqdm(range(samples_to_extract), desc="提取樣本"):
                # 獲取圖像和標籤
                if X.ndim == 4:  # [H, W, C, N] 格式
                    img = X[:, :, :, i]
                else:  # [N, H, W, C] 格式
                    img = X[i]
                
                label = str(y[i][0])
                
                # 創建文件名
                img_filename = f"svhn_{i:06d}.png"
                img_path = os.path.join(output_dir, img_filename)
                
                # 保存圖像
                plt.imsave(img_path, img)
                
                # 添加到標籤字典
                labels_dict[img_filename] = label
            
            # 保存標籤到 CSV 文件
            labels_path = os.path.join(output_dir, "labels.csv")
            with open(labels_path, 'w') as f:
                f.write("filename,label\n")
                for filename, label in labels_dict.items():
                    f.write(f"{filename},{label}\n")
            
            print(f"成功提取 {samples_to_extract} 個樣本")
            print(f"圖像保存到: {output_dir}")
            print(f"標籤保存到: {labels_path}")
            
            return True
            
        else:
            print("未找到標準 SVHN 格式的 'X' 和 'y' 變量")
            
    except Exception as e:
        print(f"使用 scipy.io.loadmat 載入失敗: {e}")
    
    # 如果前面的方法失敗，嘗試使用 h5py 載入 (多數字格式)
    try:
        with h5py.File(mat_file_path, 'r') as f:
            # 檢查是否有 digitStruct 群組
            if 'digitStruct' in f:
                digit_struct = f['digitStruct']
                
                # 檢查是否有必要的數據集
                if 'name' in digit_struct and 'bbox' in digit_struct:
                    print("解析 digitStruct 格式數據...")
                    
                    # 這部分需要根據具體的 SVHN MAT 文件格式調整
                    # 由於 h5py 讀取的 SVHN MAT 文件格式比較複雜，這裡只提供一個框架
                    
                    print("注意: 此類型的 MAT 檔案需要更複雜的解析邏輯")
                    print("請根據分析結果調整代碼以適應特定的文件結構")
                    
                    return False
            
            print("未找到 SVHN digitStruct 格式的數據")
            
    except Exception as e:
        print(f"使用 h5py 載入失敗: {e}")
    
    print("無法從提供的 MAT 檔案中提取 SVHN 數據")
    return False

if __name__ == "__main__":
    # 修改為您的 MAT 檔案路徑
    mat_file_path = sys.argv[1]
    
    # 分析文件結構
    analyze_mat_file(mat_file_path)
    
    # 提取數據到新目錄
    # extract_svhn_data(mat_file_path, "extracted_svhn_data", max_samples=10000)