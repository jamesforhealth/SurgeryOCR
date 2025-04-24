import os
import sys
import json
import csv
import re
from tqdm import tqdm
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import easyocr
import argparse

def process_image(reader, image_path):
    """處理單張圖像並返回原始OCR結果"""
    try:
        return reader.readtext(image_path, allowlist='0123456789- ')
    except Exception as e:
        print(f"處理 {image_path} 時出錯: {e}")
        return []

def main(args):
    # 創建 EasyOCR 閱讀器
    print("初始化 EasyOCR 模型...")
    reader = easyocr.Reader(['en'], gpu=True)
    
    # 讀取 JSONL 文件中的標籤數據
    data_dir = args.data_dir
    jsonl_path = os.path.join(data_dir, 'region2.jsonl')
    print(f"讀取標籤文件: {jsonl_path}")
    
    labels = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if 'images' in item and 'response' in item:
                # 獲取圖像名稱的基本部分
                image_filename = os.path.basename(item['images'])
                # 構建完整的圖像路徑
                image_path = os.path.join(data_dir, image_filename)
                
                labels.append({
                    'image_path': image_path,
                    'label': item['response']
                })
    
    print(f"總共讀取了 {len(labels)} 個標籤")
    
    # 創建結果 CSV 文件
    results_csv_path = os.path.join(data_dir, 'easyocr_raw_results.csv')
    with open(results_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Image', 'Ground Truth', 'OCR_Results'])
        
        # 處理每張圖像
        for item in tqdm(labels, desc="處理圖像"):
            image_path = item['image_path']
            ground_truth = item['label']
            
            # 檢查圖像是否存在
            if not os.path.exists(image_path):
                print(f"警告: 找不到圖像 {image_path}")
                continue
            
            # 處理圖像
            result = process_image(reader, image_path)
            
            # 保存結果，使用JSON字符串格式保存完整的OCR結果
            # 處理 bbox 可能是 numpy 數組或者已經是列表的情況
            serialized_result = []
            for bbox, text, prob in result:
                # 檢查 bbox 是否為 numpy 數組
                if hasattr(bbox, 'tolist'):
                    bbox = bbox.tolist()
                # 確保概率值是 Python 內建的 float 類型
                prob_value = float(prob)
                serialized_result.append([bbox, text, prob_value])
            
            result_json = json.dumps(serialized_result)
            writer.writerow([os.path.basename(image_path), ground_truth, result_json])
    
    print(f"原始OCR結果已保存到: {results_csv_path}")
    print("您可以使用這些原始結果進行不同閾值的分析")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用 EasyOCR 獲取所有圖像的原始OCR結果')
    parser.add_argument('--data_dir', type=str, default='../data/2024-11-20_h/region2', 
                        help='包含圖像的目錄')
    args = parser.parse_args()
    
    main(args) 