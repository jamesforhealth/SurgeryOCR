#!/usr/bin/env python
import json
import re
import argparse
from collections import defaultdict
import os
from datetime import datetime

def extract_frame_number(image_path):
    """Extract the frame number from the image path."""
    match = re.search(r'frame_(\d+)\.png', image_path)
    if match:
        return int(match.group(1))
    return None

def load_jsonl_file(file_path):
    """Load JSONL file and return a dictionary mapping frame numbers to OCR responses."""
    results = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                frame_num = extract_frame_number(data.get('images', ''))
                if frame_num is not None:
                    results[frame_num] = data.get('response', '')
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line in {file_path}")
    return results

def compare_ocr_results(file1, file2, output_file=None):
    """Compare OCR results between two JSONL files and list the differences."""
    print(f"Loading results from {file1}...")
    results1 = load_jsonl_file(file1)
    print(f"Loading results from {file2}...")
    results2 = load_jsonl_file(file2)
    
    # Find common frame numbers
    common_frames = set(results1.keys()) & set(results2.keys())
    print(f"Found {len(common_frames)} common frames between the two files.")
    
    # Find differences
    differences = []
    for frame in sorted(common_frames):
        if results1[frame] != results2[frame]:
            differences.append({
                'frame': frame,
                'file1_response': results1[frame],
                'file2_response': results2[frame]
            })
    
    # Print and optionally save the differences
    print(f"Found {len(differences)} frames with different OCR results.")
    if differences:
        print("\nDifferences:")
        for diff in differences:
            print(f"Frame {diff['frame']}: '{diff['file1_response']}' vs '{diff['file2_response']}'")
        
        if output_file:
            # 創建包含比較檔案資訊的輸出結構
            output_data = {
                "metadata": {
                    "file1": os.path.abspath(file1),
                    "file2": os.path.abspath(file2),
                    "comparison_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_common_frames": len(common_frames),
                    "total_differences": len(differences)
                },
                "differences": differences
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\nDifferences saved to {output_file}")
    
    return differences

def main():
    parser = argparse.ArgumentParser(description='Compare OCR results between two JSONL files.')
    # 修改参数设置，将位置参数改为可选参数并添加默认值
    parser.add_argument('--file1', default="2024-11-20_h/region2/region2.jsonl", 
                        help='Path to the first JSONL file (default: 2024-11-20_h/region2/region2.jsonl)')
    parser.add_argument('--file2', default="region2/region2_test.jsonl", 
                        help='Path to the second JSONL file (default: region2/region2_test.jsonl)')
    parser.add_argument('--output', '-o', default="differences.json",
                        help='Output file to save the differences (default: differences.json)')
    args = parser.parse_args()
    
    compare_ocr_results(args.file1, args.file2, args.output)

if __name__ == "__main__":
    main() 