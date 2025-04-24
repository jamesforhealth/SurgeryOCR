#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
將 SVHN 數據集的 MAT 文件轉換為 CSV 和圖像文件
"""

import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys
import h5py
import cv2

def convert_svhn_to_csv(mat_file_path, output_dir, sample_images=None):
    """
    將 SVHN MAT 文件轉換為 CSV 文件和樣本圖像
    
    Args:
        mat_file_path: MAT 文件路徑
        output_dir: 輸出目錄
        sample_images: 要保存的樣本圖像數量 (None 表示全部)
    """
    print(f"正在處理 MAT 文件: {mat_file_path}")
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "sample_images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 嘗試使用 scipy.io.loadmat 載入 (適用於老版本 MAT 文件)
    try:
        # 載入 MAT 文件
        data = sio.loadmat(mat_file_path)
        
        # 提取 X 和 y
        if 'X' in data and 'y' in data:
            X = data['X']  # 圖像數據
            y = data['y']  # 標籤
            
            # 處理 SVHN 特殊性：標籤 10 實際代表數字 0
            y_corrected = y.copy()
            y_corrected[y_corrected == 10] = 0
            
            print(f"圖像數據 'X' 形狀: {X.shape}")
            print(f"標籤 'y' 形狀: {y.shape}")
            
            # 確定樣本總數和要處理的樣本數
            num_samples = X.shape[3] if X.ndim == 4 else X.shape[0]
            samples_to_process = min(num_samples, sample_images or num_samples)
            
            print(f"共有 {num_samples} 個樣本，將處理 {samples_to_process} 個")
            
            # 建立標籤數據框
            labels_df = pd.DataFrame({
                'sample_id': range(num_samples),
                'label': y_corrected.flatten()
            })
            
            # 保存標籤到 CSV
            labels_csv_path = os.path.join(output_dir, "labels.csv")
            labels_df.to_csv(labels_csv_path, index=False)
            print(f"標籤已保存至: {labels_csv_path}")
            
            # 建立標籤統計信息
            label_counts = labels_df['label'].value_counts().sort_index()
            stats_df = pd.DataFrame({
                'label': label_counts.index,
                'count': label_counts.values,
                'percentage': (label_counts.values / num_samples * 100).round(2)
            })
            
            # 保存標籤統計到 CSV
            stats_csv_path = os.path.join(output_dir, "label_statistics.csv")
            stats_df.to_csv(stats_csv_path, index=False)
            print(f"標籤統計已保存至: {stats_csv_path}")
            
            # 保存樣本圖像
            if sample_images:
                print(f"正在保存 {samples_to_process} 個樣本圖像...")
                for i in tqdm(range(samples_to_process)):
                    if X.ndim == 4:  # [H, W, C, N] 格式
                        img = X[:, :, :, i]
                    else:  # [N, H, W, C] 格式
                        img = X[i]
                    
                    label = y_corrected[i][0]
                    img_path = os.path.join(images_dir, f"sample_{i:05d}_label_{label}.png")
                    plt.imsave(img_path, img)
                
                print(f"樣本圖像已保存至: {images_dir}")
            
            # 如果樣本數量較大，建立圖像像素值統計
            pixel_stats = {
                'mean_r': np.mean(X[:, :, 0, :]) if X.ndim == 4 else np.mean(X[:, :, :, 0]),
                'mean_g': np.mean(X[:, :, 1, :]) if X.ndim == 4 else np.mean(X[:, :, :, 1]),
                'mean_b': np.mean(X[:, :, 2, :]) if X.ndim == 4 else np.mean(X[:, :, :, 2]),
                'std_r': np.std(X[:, :, 0, :]) if X.ndim == 4 else np.std(X[:, :, :, 0]),
                'std_g': np.std(X[:, :, 1, :]) if X.ndim == 4 else np.std(X[:, :, :, 1]),
                'std_b': np.std(X[:, :, 2, :]) if X.ndim == 4 else np.std(X[:, :, :, 2]),
                'min': np.min(X),
                'max': np.max(X)
            }
            
            pixel_stats_df = pd.DataFrame([pixel_stats])
            pixel_stats_csv_path = os.path.join(output_dir, "pixel_statistics.csv")
            pixel_stats_df.to_csv(pixel_stats_csv_path, index=False)
            print(f"像素統計已保存至: {pixel_stats_csv_path}")
            
            return True
        
    except NotImplementedError:
        print("检测到 MATLAB v7.3 格式 (HDF5)，使用 h5py 尝试读取...")
    except Exception as e:
        print(f"使用 scipy.io.loadmat 載入失敗: {e}")
    
    # 使用 h5py 读取 MATLAB v7.3 格式文件 (HDF5)
    try:
        with h5py.File(mat_file_path, 'r') as f:
            # 检查是否是 SVHN digitStruct 格式
            if 'digitStruct' in f:
                print("检测到 SVHN digitStruct 格式数据")
                return process_svhn_digitstruct(f, mat_file_path, output_dir, sample_images)
            else:
                print("未找到 'digitStruct' 组，这可能不是 SVHN 全数据集格式")
                return False
    except Exception as e:
        print(f"使用 h5py 载入失败: {e}")
        return False

def process_svhn_digitstruct(h5f, mat_file_path, output_dir, sample_images=None):
    """處理 SVHN digitStruct 格式的數據"""
    
    # 獲取 digitStruct 組
    digit_struct = h5f['digitStruct']
    
    # 獲取 name 和 bbox 數據集
    name_dataset = digit_struct['name']
    bbox_dataset = digit_struct['bbox']
    
    # 獲取樣本總數
    num_samples = name_dataset.shape[0]
    # 處理所有樣本，而非默認的100個
    samples_to_process = min(num_samples, sample_images or num_samples)
    
    print(f"共有 {num_samples} 個樣本，將處理 {samples_to_process} 個")
    
    # 創建用於存儲結果的列表
    filenames = []
    all_labels = []
    # all_bboxes = []  # 不再需要位置信息
    
    # 獲取文件所在目錄路徑
    mat_dir = os.path.dirname(mat_file_path)
    
    # 輔助函數：從 h5py 引用中獲取數據
    def get_attr_value(obj, name):
        if isinstance(obj[name], h5py.Dataset):
            value = obj[name][()]
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()
            return value
        return None
    
    def get_name(index):
        ref = h5f[name_dataset[index][0]]
        name = ''.join([chr(v[0]) for v in ref[()]])
        return name
    
    def get_bbox_data(index):
        """獲取指定索引的邊界框數據"""
        bbox_item_ref = h5f[bbox_dataset[index][0]]
        
        # 檢查屬性
        digit_struct_label = get_attr_value(bbox_item_ref, 'label')
        digit_struct_top = get_attr_value(bbox_item_ref, 'top')
        digit_struct_left = get_attr_value(bbox_item_ref, 'left')
        digit_struct_height = get_attr_value(bbox_item_ref, 'height')
        digit_struct_width = get_attr_value(bbox_item_ref, 'width')
        
        # 單個數字的情況
        if isinstance(digit_struct_label, (int, float)):
            label = int(digit_struct_label)
            if label == 10:  # SVHN中10代表0
                label = 0
                
            return [{
                'label': label,
                'left': int(digit_struct_left)  # 僅保留左側坐標用於排序
            }]
        
        # 多個數字的情況
        num_digits = len(bbox_item_ref['label'])
        result = []
        
        for j in range(num_digits):
            # 解析標籤和左坐標(僅用於排序)
            def get_element_value(name):
                ref = bbox_item_ref[name]
                if ref.shape[0] > j:
                    element_ref = ref[j][0]
                    value = h5f[element_ref][()]
                    if isinstance(value, np.ndarray) and value.size == 1:
                        value = value.item()
                    return int(value)
                return 0
            
            label = get_element_value('label')
            if label == 10:  # SVHN中10代表0
                label = 0
                
            result.append({
                'label': label,
                'left': get_element_value('left')  # 僅保留左側坐標用於排序
            })
        
        return result
    
    print("正在讀取文件名和數字標籤信息...")
    for i in tqdm(range(samples_to_process)):
        try:
            # 獲取文件名
            filename = get_name(i)
            filenames.append(filename)
            
            # 獲取邊界框數據(僅用於獲取標籤和排序)
            bboxes = get_bbox_data(i)
            # all_bboxes.append(bboxes)  # 不再需要保存所有邊界框信息
            
            # 按從左到右順序排序數字
            bboxes.sort(key=lambda x: x['left'])
            label = ''.join([str(box['label']) for box in bboxes])
            all_labels.append(label)
            
        except Exception as e:
            print(f"處理樣本 {i} 時出錯: {e}")
            # 添加空數據以保持索引一致
            filenames.append(f"error_{i}.png")
            all_labels.append("")
    
    # 創建標籤數據框
    labels_df = pd.DataFrame({
        'filename': filenames,
        'label': all_labels,
        'num_digits': [len(label) for label in all_labels]
    })
    
    # 過濾掉空標籤
    valid_labels_df = labels_df[labels_df['label'] != ""]
    
    # 保存標籤到 CSV
    labels_csv_path = os.path.join(output_dir, "labels.csv")
    valid_labels_df.to_csv(labels_csv_path, index=False)
    print(f"標籤已保存至: {labels_csv_path}")
    
    # 創建數字統計信息（每個類別的計數）
    digit_counts = {}
    for label in all_labels:
        if label:
            for digit in label:
                if digit in digit_counts:
                    digit_counts[digit] += 1
                else:
                    digit_counts[digit] = 1
    
    # 轉換為 DataFrame 並排序
    stats_df = pd.DataFrame({
        'digit': list(digit_counts.keys()),
        'count': list(digit_counts.values()),
    })
    stats_df = stats_df.sort_values('digit').reset_index(drop=True)
    
    # 計算百分比
    total_digits = sum(digit_counts.values())
    stats_df['percentage'] = (stats_df['count'] / total_digits * 100).round(2)
    
    # 保存數字統計到 CSV
    digit_stats_csv_path = os.path.join(output_dir, "digit_statistics.csv")
    stats_df.to_csv(digit_stats_csv_path, index=False)
    print(f"數字統計已保存至: {digit_stats_csv_path}")
    
    # 創建數字數量統計信息
    num_digits_counts = {}
    for num_digits in labels_df['num_digits']:
        if num_digits > 0:  # 排除無效樣本
            if num_digits in num_digits_counts:
                num_digits_counts[num_digits] += 1
            else:
                num_digits_counts[num_digits] = 1
    
    # 轉換為 DataFrame 並排序
    num_digits_stats = pd.DataFrame({
        'num_digits': list(num_digits_counts.keys()),
        'count': list(num_digits_counts.values()),
    })
    num_digits_stats = num_digits_stats.sort_values('num_digits').reset_index(drop=True)
    
    # 計算百分比
    total_samples = sum(num_digits_counts.values())
    num_digits_stats['percentage'] = (num_digits_stats['count'] / total_samples * 100).round(2)
    
    # 保存數字數量統計到 CSV
    num_digits_csv_path = os.path.join(output_dir, "num_digits_statistics.csv")
    num_digits_stats.to_csv(num_digits_csv_path, index=False)
    print(f"數字數量統計已保存至: {num_digits_csv_path}")
    
    # 注釋掉保存樣本圖像的代碼，因為現在不關注位置信息
    # # 保存样本图像
    # sample_images_dir = os.path.join(output_dir, "sample_images")
    # os.makedirs(sample_images_dir, exist_ok=True)
    
    # if sample_images:
    #     print(f"正在保存 {samples_to_process} 个样本图像...")
    #     saved_count = 0
        
    #     for i in tqdm(range(samples_to_process)):
    #         if not all_labels[i]:  # 跳过空标签
    #             continue
                
    #         # 读取原始图像
    #         img_path = os.path.join(mat_dir, filenames[i])
    #         if not os.path.exists(img_path):
    #             print(f"警告: 未找到图像文件 {img_path}")
    #             continue
            
    #         img = cv2.imread(img_path)
    #         if img is None:
    #             print(f"警告: 无法读取图像 {img_path}")
    #             continue
            
    #         # 在图像上标注边界框
    #         img_with_boxes = img.copy()
    #         for bbox in all_bboxes[i]:
    #             left = int(bbox['left'])
    #             top = int(bbox['top'])
    #             width = int(bbox['width'])
    #             height = int(bbox['height'])
    #             label = bbox['label']
                
    #             # 绘制边界框
    #             cv2.rectangle(img_with_boxes, (left, top), (left + width, top + height), (0, 255, 0), 2)
    #             # 添加标签
    #             cv2.putText(img_with_boxes, str(label), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    #         # 保存带标注的图像
    #         annotated_img_path = os.path.join(sample_images_dir, f"annotated_{filenames[i]}")
    #         cv2.imwrite(annotated_img_path, img_with_boxes)
            
    #         # 同时保存未标注的原始图像以便后续使用
    #         original_img_path = os.path.join(sample_images_dir, filenames[i])
    #         cv2.imwrite(original_img_path, img)
            
    #         saved_count += 1
    #         if saved_count >= sample_images:
    #             break
        
    #     print(f"样本图像已保存至: {sample_images_dir}")
    
    return True

if __name__ == "__main__":
    # 输入你的 MAT 文件路径
    mat_file_path = sys.argv[1]
    
    # 设置输出目录
    output_dir = os.path.dirname(mat_file_path)
    
    # 执行转换，处理所有样本
    convert_svhn_to_csv(mat_file_path, output_dir)  # 不再限制只处理100个样本