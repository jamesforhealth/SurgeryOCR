import os
import shutil
import pandas as pd
import argparse

def merge_datasets(main_dir, extra_dir, output_dir):
    """
    合併兩個數據集目錄到一個新目錄
    
    參數:
        main_dir: 主數據集目錄
        extra_dir: 額外數據集目錄
        output_dir: 合併後的輸出目錄
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取主數據集的標籤文件
    main_labels_path = os.path.join(main_dir, "labels.csv")
    main_df = pd.read_csv(main_labels_path)
    
    # 讀取額外數據集的標籤文件
    extra_labels_path = os.path.join(extra_dir, "labels.csv")
    extra_df = pd.read_csv(extra_labels_path)
    
    # 檢查兩個標籤文件的列名是否一致
    if set(main_df.columns) != set(extra_df.columns):
        print(f"警告: 標籤文件列名不一致!")
        print(f"主數據集列名: {main_df.columns}")
        print(f"額外數據集列名: {extra_df.columns}")
    
    # 為額外數據集的文件名添加前綴，避免衝突
    extra_df['original_filename'] = extra_df['filename']
    extra_df['filename'] = 'extra_' + extra_df['filename']
    
    # 合併兩個數據框
    combined_df = pd.concat([main_df, extra_df], ignore_index=True)
    
    # 保存合併後的標籤文件
    combined_labels_path = os.path.join(output_dir, "labels.csv")
    combined_df.to_csv(combined_labels_path, index=False)
    
    # 複製主數據集的圖像文件
    for _, row in main_df.iterrows():
        src_path = os.path.join(main_dir, row['filename'])
        dst_path = os.path.join(output_dir, row['filename'])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # 複製額外數據集的圖像文件(添加前綴)
    for _, row in extra_df.iterrows():
        src_path = os.path.join(extra_dir, row['original_filename'])
        dst_path = os.path.join(output_dir, row['filename'])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # 打印統計信息
    print(f"合併完成!")
    print(f"主數據集樣本數: {len(main_df)}")
    print(f"額外數據集樣本數: {len(extra_df)}")
    print(f"合併後總樣本數: {len(combined_df)}")
    print(f"合併後標籤文件保存至: {combined_labels_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合併兩個SVHN數據集")
    parser.add_argument('--main', type=str, required=True, help='主數據集目錄')
    parser.add_argument('--extra', type=str, required=True, help='額外數據集目錄')
    parser.add_argument('--output', type=str, required=True, help='合併後的輸出目錄')
    
    args = parser.parse_args()
    
    merge_datasets(args.main, args.extra, args.output) 