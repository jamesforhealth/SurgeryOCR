import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
import re
from skimage.metrics import structural_similarity as ssim # SSIM 也是一個有用的指標

def parse_roi(roi_str):
    """將 'x,y,w,h' 格式的字符串解析為 (x, y, w, h) 元組"""
    try:
        parts = list(map(int, roi_str.split(',')))
        if len(parts) == 4:
            x, y, w, h = parts
            if w > 0 and h > 0:
                return x, y, w, h
            else:
                raise ValueError("寬度和高度必須為正數")
        else:
            raise ValueError("ROI 格式必須是 x,y,w,h")
    except Exception as e:
        raise argparse.ArgumentTypeError(f"無效的 ROI 格式 '{roi_str}': {e}")

def find_image_path(rel_path, base_dirs=None):
    """嘗試在不同基本目錄下查找圖像文件"""
    if base_dirs is None:
        # 默認嘗試的路徑: 相對路徑, data/, ../data/
        base_dirs = ['.', 'data', os.path.join('..', 'data')]

    possible_paths = [os.path.join(base, rel_path) for base in base_dirs]
    possible_paths.insert(0, rel_path) # 也檢查原始相對路徑

    # 嘗試從路徑中提取視頻文件夾名，並添加到搜索路徑
    # 例如: 2024-11-20_h/region2/frame_100.png -> ../data/2024-11-20_h/region2/frame_100.png
    match = re.search(r'([\w-]+/region\d+)/frame_\d+\.png', rel_path)
    if match:
        video_folder_path = match.group(1)
        possible_paths.append(os.path.join('..', 'data', video_folder_path, os.path.basename(rel_path)))
        possible_paths.append(os.path.join('data', video_folder_path, os.path.basename(rel_path)))


    for path in possible_paths:
        abs_path = os.path.abspath(path) # 獲取絕對路徑以便調試
        # print(f"正在嘗試: {abs_path}") # 取消註釋以查看嘗試的路徑
        if os.path.exists(path):
            # print(f"找到圖像: {abs_path}")
            return path

    print(f"警告: 找不到圖像文件 {rel_path}")
    print(f"嘗試的基目錄: {base_dirs}")
    print(f"當前工作目錄: {os.getcwd()}")
    return None # 返回 None 表示未找到

def analyze_frame_diff(frame1_path, frame2_path, roi, diff_threshold=1):
    """
    分析兩個幀在指定 ROI 區域的像素差異。

    Args:
        frame1_path (str): 第一幀圖像的路徑。
        frame2_path (str): 第二幀圖像的路徑。
        roi (tuple): 感興趣區域 (x, y, w, h)。
        diff_threshold (int): 計算 T-MAD 時忽略的像素差異閾值。

    Returns:
        tuple: 包含 (roi1, roi2, diff_image, mad, t_mad, ssim_score) 的元組，
               如果圖像加載或 ROI 提取失敗則返回 None。
    """
    img1 = cv2.imread(frame1_path)
    img2 = cv2.imread(frame2_path)

    if img1 is None or img2 is None:
        print(f"錯誤: 無法加載圖像 {frame1_path} 或 {frame2_path}")
        return None

    x, y, w, h = roi
    # 確保 ROI 不超出圖像邊界
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if x < 0 or y < 0 or x + w > w1 or y + h > h1 or x + w > w2 or y + h > h2:
         print(f"錯誤: ROI {roi} 超出圖像邊界 ({w1}x{h1} 或 {w2}x{h2})")
         return None # 或者直接返回錯誤

    # 提取 ROI
    roi1 = img1[y:y+h, x:x+w]
    roi2 = img2[y:y+h, x:x+w]

    # 轉換為灰度圖進行比較
    gray_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    # 1. 計算絕對差值圖像
    diff_image = cv2.absdiff(gray_roi1, gray_roi2)

    # 2. 計算平均絕對差值 (MAD)
    mad = np.mean(diff_image)

    # --- 新增：計算閾值化平均絕對差 (T-MAD) ---
    thresholded_diff = diff_image.copy()
    thresholded_diff[thresholded_diff <= diff_threshold] = 0 # 將小於等於閾值的差異設為0
    t_mad = np.mean(thresholded_diff) # 計算閾值化後的平均差異

    # 3. 計算結構相似性指數 (SSIM)
    # data_range 是像素值的範圍
    ssim_score, _ = ssim(gray_roi1, gray_roi2, full=True, data_range=gray_roi1.max() - gray_roi1.min())

    # 返回所有指標
    return roi1, roi2, diff_image, mad, t_mad, ssim_score

def visualize_diff(roi1, roi2, diff_image, mad, t_mad, ssim_score, frame1_name, frame2_name, diff_threshold):
    """使用 Matplotlib 可視化 ROI 和差異"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Matplotlib 期望 RGB 順序，而 OpenCV 是 BGR
    axes[0].imshow(cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Frame: {frame1_name}\nROI")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Frame: {frame2_name}\nROI")
    axes[1].axis('off')

    # 顯示差異圖（使用灰度色彩映射）
    im = axes[2].imshow(diff_image, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f"Absolute Difference\nMAD: {mad:.2f} | T-MAD(>{diff_threshold}): {t_mad:.2f} | SSIM: {ssim_score:.4f}")
    axes[2].axis('off')

    # 添加顏色條以顯示差異程度
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="分析連續幀在指定 ROI 的像素差異")
    parser.add_argument("--jsonl", default='../data/2024-11-20_h/region2/region2.jsonl', help="包含幀文件路徑的 JSONL 文件")
    parser.add_argument("--roi", default='0,0,196,80', type=parse_roi, help="感興趣區域，格式為 'x,y,w,h'")
    parser.add_argument("--max_pairs", type=int, default=10, help="要分析的最大幀對數量 (設為 0 分析所有)")
    parser.add_argument("--start_frame", type=int, default=0, help="從指定的幀號開始分析")
    parser.add_argument("--diff_threshold", type=int, default=30, help="計算 T-MAD 時忽略的像素差異閾值")
    args = parser.parse_args()

    # --- 從 JSONL 加載幀數據 ---
    frames_data = {}
    print(f"從 {args.jsonl} 加載數據...")
    jsonl_dir = os.path.dirname(args.jsonl) or "."
    # 確定查找圖像的基本目錄
    base_dirs = ['.', 'data', os.path.join('..', 'data'), jsonl_dir, os.path.join(jsonl_dir, '..', 'data')]
    base_dirs = list(dict.fromkeys(base_dirs)) # 去重

    with open(args.jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                rel_image_path = data.get('images', '')
                if not rel_image_path:
                    continue

                # 嘗試查找圖像的絕對路徑
                image_path = find_image_path(rel_image_path, base_dirs)
                if not image_path:
                    continue # 如果找不到圖像，跳過

                # 從文件名提取幀號
                frame_match = re.search(r'frame_(\d+)', os.path.basename(rel_image_path))
                if frame_match:
                    frame_num = int(frame_match.group(1))
                    # 只添加幀號大於等於起始幀號的數據
                    if frame_num >= args.start_frame:
                         # 存儲絕對路徑和原始相對路徑（用於顯示）
                        frames_data[frame_num] = (image_path, rel_image_path)
                else:
                    print(f"警告: 無法從文件名 {os.path.basename(rel_image_path)} 提取幀號")

            except json.JSONDecodeError as e:
                # --- 修改：處理可能的 JSONL 格式問題 ---
                # 嘗試處理一行多個 JSON 對象的情況
                try:
                    # 查找所有可能的 JSON 對象
                    json_objects = re.findall(r'({.*?})', line.strip())
                    for json_str in json_objects:
                        data = json.loads(json_str)
                        rel_image_path = data.get('images', '')
                        if not rel_image_path:
                            continue
                        image_path = find_image_path(rel_image_path, base_dirs)
                        if not image_path:
                            continue
                        frame_match = re.search(r'frame_(\d+)', os.path.basename(rel_image_path))
                        if frame_match:
                            frame_num = int(frame_match.group(1))
                            if frame_num >= args.start_frame:
                                frames_data[frame_num] = (image_path, rel_image_path)
                        else:
                            print(f"警告: 無法從文件名 {os.path.basename(rel_image_path)} 提取幀號")
                except json.JSONDecodeError:
                    print(f"警告: 無法解析行（即使嘗試分割後）: {line.strip()}")
                continue # 跳過無法解析的行

    if not frames_data:
        print("錯誤: 未能從 JSONL 文件加載任何有效的幀數據。請檢查路徑和文件內容。")
        return

    # 按幀號排序
    sorted_frame_nums = sorted(frames_data.keys())
    print(f"成功加載 {len(sorted_frame_nums)} 個幀的數據 (從幀 {args.start_frame} 開始)。")

    # --- 遍歷幀對進行分析 ---
    pairs_analyzed = 0
    for i in range(len(sorted_frame_nums) - 1):
        current_frame_num = sorted_frame_nums[i]
        next_frame_num = sorted_frame_nums[i+1]

        # 可選：跳過幀號不連續的對
        # if next_frame_num - current_frame_num > 5: # 允許最多跳4幀
        #     print(f"跳過不連續的幀對: {current_frame_num} -> {next_frame_num}")
        #     continue

        frame1_abs_path, frame1_rel_path = frames_data[current_frame_num]
        frame2_abs_path, frame2_rel_path = frames_data[next_frame_num]

        print(f"\n--- 分析幀對: {current_frame_num} vs {next_frame_num} ---")
        print(f"  Frame 1: {frame1_rel_path}")
        print(f"  Frame 2: {frame2_rel_path}")

        analysis_result = analyze_frame_diff(frame1_abs_path, frame2_abs_path, args.roi, args.diff_threshold)

        if analysis_result:
            roi1, roi2, diff_image, mad, t_mad, ssim_score = analysis_result
            print(f"  平均絕對差 (MAD): {mad:.2f}")
            print(f"  閾值化平均絕對差 (T-MAD > {args.diff_threshold}): {t_mad:.2f}")
            print(f"  結構相似性 (SSIM): {ssim_score:.4f}")

            # 可視化
            visualize_diff(roi1, roi2, diff_image, mad, t_mad, ssim_score,
                           os.path.basename(frame1_rel_path), os.path.basename(frame2_rel_path),
                           args.diff_threshold)
        else:
            print("  分析失敗 (圖像加載或 ROI 提取錯誤)")

        pairs_analyzed += 1
        if args.max_pairs > 0 and pairs_analyzed >= args.max_pairs:
            print(f"\n已達到最大分析幀對數量 ({args.max_pairs})。")
            break

        # 添加暫停，以便觀察每個結果

    print("\n分析完成。")

if __name__ == "__main__":
    main()