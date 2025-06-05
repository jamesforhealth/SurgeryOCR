#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR模型比較腳本：比較不同OCR模型在指定視頻幀上的識別效果
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import torch

# 導入OCR接口
from models.OCR_interface import recognize_text, get_ocr_model

def load_jsonl(file_path):
    """加載JSONL格式的標籤文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_frame_number(image_path):
    """從圖像路徑中提取幀號"""
    filename = os.path.basename(image_path)
    frame_num = int(filename.split('_')[1].split('.')[0])
    return frame_num

def analyze_ocr_results(results_df, model_name=None):
    """分析OCR結果，生成統計報告"""
    total_frames = len(results_df)
    diff_frames = results_df[results_df['original'] != results_df['ocr_result']]
    num_diffs = len(diff_frames)
    accuracy = (total_frames - num_diffs) / total_frames * 100 if total_frames > 0 else 0
    
    # 計算錯誤類型
    error_types = defaultdict(int)
    for _, row in diff_frames.iterrows():
        org = str(row['original'])
        ocr = str(row['ocr_result'])
        if not ocr:
            error_types['未識別'] += 1
        elif len(org) != len(ocr):
            error_types['長度不符'] += 1
        elif '-' in org and '-' not in ocr:
            error_types['負號缺失'] += 1
        elif '-' not in org and '-' in ocr:
            error_types['誤加負號'] += 1
        else:
            error_types['數字錯誤'] += 1
    
    # 生成報告
    report = {
        '模型': model_name or "未命名模型",
        '總幀數': total_frames,
        '差異幀數': num_diffs,
        '準確率': f"{accuracy:.2f}%",
        '錯誤類型': dict(error_types)
    }
    
    return report

def process_frames(model_name, model_instance, label_data, video_dir, samples=0):
    """使用指定的OCR模型實例處理幀"""
    print(f"\n>>>>> 開始使用模型: {model_name} 進行處理 <<<<<")
    
    results = []
    
    # 限制處理樣本數
    data_to_process = label_data[:samples] if samples > 0 else label_data
    
    progress_bar = tqdm(data_to_process, desc=f"處理幀 ({model_name})")
    for frame_data in progress_bar:
        image_rel_path_from_jsonl = str(frame_data['images'])
        original_label = str(frame_data['response']) # 確保標籤是字符串
        
        # 由於 labels.jsonl 中的 'images' 路徑是相對於項目根目錄的，
        # 並且假設此腳本從項目根目錄運行，
        # image_path可以直接從 image_rel_path_from_jsonl 構建。
        # video_dir 主要用於定位 labels.jsonl 文件。
        base_data_dir = video_dir.parent 
        image_path = base_data_dir / image_rel_path_from_jsonl
        
        # 讀取圖像
        # cv2.imread 可以處理相對於當前工作目錄的路徑
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"警告：無法讀取圖像 {image_path} (原始路徑: '{image_rel_path_from_jsonl}')，跳過")
            continue
        
        # 提取幀號
        frame_number = extract_frame_number(str(image_path))
        
        # 使用模型實例進行OCR識別
        # 假設 model_instance.recognize 返回 (text, confidence)
        ocr_text, confidence = model_instance.recognize(image)
        ocr_text = str(ocr_text) # 確保OCR結果是字符串

        # 添加到結果列表
        results.append({
            'frame': frame_number,
            'original': original_label,
            'ocr_result': ocr_text,
            'confidence': confidence,
            'is_different': original_label != ocr_text
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="OCR模型比較腳本")
    parser.add_argument('--video_dir', type=str, required=True, help="包含視頻幀和labels.jsonl的目錄")
    parser.add_argument('--region', type=str, default="region2", help="要處理的區域，'all' 或 'region2'")
    parser.add_argument('--models', type=str, default="easyocr,crnn", help="要測試的OCR模型列表，以逗號分隔 (例如: easyocr,crnn)")
    parser.add_argument('--crnn_model_path', type=str, default="models/OCR_interface/simpleocr/best_crnn_model.pth", help="CRNN模型權重文件路徑")
    parser.add_argument('--crnn_characters', type=str, default="0123456789-", help="CRNN模型識別的字符集")
    parser.add_argument('--output_dir', type=str, default="ocr_results", help="保存結果的目錄")
    parser.add_argument('--samples', type=int, default=0, help="限制處理的樣本數量 (0表示全部)")
    parser.add_argument('--no_gpu', action='store_true', help="禁用GPU (如果模型支持)")
    parser.add_argument('--easyocr_confidence_threshold', type=float, default=0.3, help="EasyOCR的置信度閾值")


    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    labels_file = video_dir / Path(args.region + ".jsonl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加載標籤數據
    if not labels_file.exists():
        print(f"錯誤：找不到標籤文件 {labels_file}")
        return
    label_data = load_jsonl(labels_file)
    
    # 解析要測試的模型
    model_types_to_test = [m.strip().lower() for m in args.models.split(',')]
    
    all_model_results_summary = []
    comparison_data_frames = {} # 用於存儲每個模型詳細的逐幀結果

    for model_type in model_types_to_test:
        model_instance = None
        model_name_for_report = model_type

        if model_type == "easyocr":
            try:
                model_instance = get_ocr_model(
                    model_type="easyocr",
                    gpu=not args.no_gpu,
                    lang_list=['en'], # 或者您需要的語言
                    confidence_threshold=args.easyocr_confidence_threshold, # 使用命令行參數
                    allowlist="0123456789-" # 與CRNN保持一致
                )
                model_name_for_report = f"EasyOCR (conf: {args.easyocr_confidence_threshold})"
            except Exception as e:
                print(f"初始化EasyOCR模型失敗: {e}")
                continue
        elif model_type == "crnn":
            try:
                # 檢查CRNN模型文件是否存在
                crnn_path = Path(args.crnn_model_path)
                if not crnn_path.exists():
                    print(f"警告: CRNN模型文件 {crnn_path} 不存在。將嘗試使用預訓練的 'default'。")
                    # 嘗試使用接口中定義的預訓練模型
                    model_instance = get_ocr_model(
                        model_type="crnn",
                        pretrained="default", # 假設您在 __init__.py 中定義了 "crnn_default"
                        characters=args.crnn_characters,
                        device='cuda' if not args.no_gpu and torch.cuda.is_available() else 'cpu'
                    )
                    model_name_for_report = "CRNN (pretrained default)"
                else:
                    # 確保 get_ocr_model 內部能處理新的 checkpoint 格式
                    model_instance = get_ocr_model(
                        model_type="crnn",
                        model_path=str(crnn_path), # 傳遞微調模型的路徑
                        characters=args.crnn_characters, # 確保字符集與微調時一致
                        device='cuda' if not args.no_gpu and torch.cuda.is_available() else 'cpu'
                    )
                    model_name_for_report = f"CRNN ({crnn_path.name})"
            except ImportError:
                print("CRNN模型需要PyTorch。請安裝PyTorch後重試。")
                continue
            except Exception as e:
                print(f"初始化CRNN模型失敗: {e}")
                continue
        else:
            print(f"不支持的模型類型: {model_type}，跳過。")
            continue

        if model_instance is None:
            continue

        # 使用模型處理幀
        frame_results = process_frames(
            model_name=model_name_for_report,
            model_instance=model_instance,
            label_data=label_data,
            video_dir=video_dir,
            samples=args.samples
        )
        
        if not frame_results:
            print(f"模型 {model_name_for_report} 沒有生成任何結果。")
            continue
            
        results_df = pd.DataFrame(frame_results)
        comparison_data_frames[model_name_for_report] = results_df # 存儲DataFrame用於後續比較

        # 保存單個模型的詳細結果
        model_output_path = output_dir / f"results_{model_name_for_report.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')}.csv"
        results_df.to_csv(model_output_path, index=False)
        print(f"模型 {model_name_for_report} 的結果已保存到 {model_output_path}")
        
        # 生成統計報告
        report = analyze_ocr_results(results_df, model_name=model_name_for_report)
        all_model_results_summary.append(report)
        
        # 輸出單個模型的統計
        print(f"\n===== 模型: {report['模型']} =====")
        print(f"總幀數: {report['總幀數']}")
        print(f"差異幀數: {report['差異幀數']}")
        print(f"準確率: {report['準確率']}")
        print("錯誤類型統計:")
        for error_type, count in report['錯誤類型'].items():
            print(f"  - {error_type}: {count}筆")
        
        # 清理模型以釋放資源 (如果需要且模型有實現)
        if hasattr(model_instance, 'close'):
            model_instance.close()
        del model_instance
        if not args.no_gpu and model_type == "crnn": # CRNN使用torch，可能需要清理CUDA緩存
            torch.cuda.empty_cache()

    # 生成並輸出比較表格
    if len(all_model_results_summary) > 0:
        summary_df = pd.DataFrame(all_model_results_summary)
        print("\n\n===== OCR模型比較總結 =====")
        # 選擇要顯示的列
        columns_to_show = ['模型', '總幀數', '差異幀數', '準確率']
        # 檢查 '錯誤類型' 列是否存在並包含字典
        if '錯誤類型' in summary_df.columns and isinstance(summary_df['錯誤類型'].iloc[0], dict):
            # 將錯誤類型字典扁平化為多列
            error_types_df = summary_df['錯誤類型'].apply(pd.Series).fillna(0).astype(int)
            summary_df = pd.concat([summary_df.drop('錯誤類型', axis=1), error_types_df], axis=1)
            columns_to_show.extend(error_types_df.columns)
        
        print(summary_df[columns_to_show].to_string(index=False))
        
        summary_output_path = output_dir / "comparison_summary.csv"
        summary_df.to_csv(summary_output_path, index=False)
        print(f"\n比較總結已保存到 {summary_output_path}")

    # 如果測試了多個模型，可以創建一個包含所有模型逐幀結果的合併文件
    if len(comparison_data_frames) > 1:
        # 以 'frame' 和 'original' 作為基準合併
        base_df = None
        for model_name, df in comparison_data_frames.items():
            # 為每個模型的結果列添加後綴
            df_renamed = df.rename(columns={
                'ocr_result': f'{model_name}_ocr',
                'confidence': f'{model_name}_conf',
                'is_different': f'{model_name}_diff'
            })
            if base_df is None:
                base_df = df_renamed
            else:
                # 合併時只保留每個模型特有的列，以及共享的 'frame' 和 'original'
                base_df = pd.merge(base_df, df_renamed[['frame', 'original', f'{model_name}_ocr', f'{model_name}_conf', f'{model_name}_diff']], on=['frame', 'original'], how='outer')
        
        if base_df is not None:
            all_frames_comparison_path = output_dir / "all_frames_comparison_detailed.csv"
            base_df.sort_values('frame', inplace=True)
            base_df.to_csv(all_frames_comparison_path, index=False)
            print(f"所有模型的逐幀詳細比較已保存到 {all_frames_comparison_path}")

if __name__ == "__main__":
    # 為了CRNN模型在多進程或某些環境下的兼容性
    # from torch.multiprocessing import set_start_method
    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass
    main()