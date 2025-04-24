import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
import re
import csv
from tqdm import tqdm # 用於顯示進度條
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

def calculate_tmad(frame1_path, frame2_path, roi, diff_threshold=30):
    """
    計算兩個幀在指定 ROI 區域的閾值化平均絕對差 (T-MAD)。

    Args:
        frame1_path (str): 第一幀圖像的路徑。
        frame2_path (str): 第二幀圖像的路徑。
        roi (tuple): 感興趣區域 (x, y, w, h)。
        diff_threshold (int): 忽略的像素差異閾值。

    Returns:
        float: T-MAD 值，如果圖像加載或 ROI 提取失敗則返回 None。
    """
    img1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE) # 直接讀取灰度圖
    img2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        # print(f"錯誤: 無法加載圖像 {frame1_path} 或 {frame2_path}")
        return None

    x, y, w, h = roi
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    if x < 0 or y < 0 or x + w > w1 or y + h > h1 or x + w > w2 or y + h > h2:
        # print(f"錯誤: ROI {roi} 超出圖像邊界 ({w1}x{h1} 或 {w2}x{h2})")
        return None

    roi1 = img1[y:y+h, x:x+w]
    roi2 = img2[y:y+h, x:x+w]

    diff_image = cv2.absdiff(roi1, roi2)
    thresholded_diff = diff_image.copy()
    thresholded_diff[thresholded_diff <= diff_threshold] = 0
    t_mad = np.mean(thresholded_diff)

    return t_mad

def main():
    parser = argparse.ArgumentParser(description="分析連續幀在 ROI 區域的 T-MAD 並與標籤變化比較，評估分類準確度")
    parser.add_argument('--jsonl', default='../data/2024-11-20_h/region2/region2.jsonl', help="包含幀路徑和標籤的 JSONL 文件路徑")
    parser.add_argument('--roi', default='0,0,196,80', type=parse_roi, help="感興趣區域 (ROI)，格式: x,y,w,h")
    parser.add_argument('--output_csv', default='tmad_analysis_filtered.csv', help="輸出的 CSV 文件名")
    parser.add_argument('--diff_threshold', type=int, default=30, help="計算 T-MAD 時忽略的像素差異閾值")
    # --- 新增：用於分類的 T-MAD 門檻值 ---
    parser.add_argument('--classification_threshold', type=float, default=2, help="用於判斷標籤是否變化的 T-MAD 門檻值")
    args = parser.parse_args()

    # --- 1. 加載並過濾數據 ---
    print(f"從 {args.jsonl} 加載數據...")
    frames_data = {}
    loaded_count = 0
    skipped_lines = 0
    base_dir = os.path.dirname(args.jsonl) # JSONL 文件所在目錄作為基礎

    with open(args.jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json_objects_str = line.replace('}{', '}\n{').splitlines()
                records = [json.loads(obj_str) for obj_str in json_objects_str]
            except json.JSONDecodeError:
                try:
                    records = [json.loads(line)]
                except json.JSONDecodeError:
                    skipped_lines += 1
                    continue
            for record in records:
                try:
                    rel_image_path = record.get("images")
                    response = record.get("response", "")
                    if not rel_image_path:
                        skipped_lines += 1
                        continue
                    possible_base_dirs = [base_dir, '.', 'data', os.path.join('..', 'data')]
                    image_path = find_image_path(rel_image_path, base_dirs=possible_base_dirs)
                    if image_path is None:
                        skipped_lines += 1
                        continue
                    frame_match = re.search(r'frame_(\d+)', os.path.basename(rel_image_path))
                    if frame_match:
                        frame_num = int(frame_match.group(1))
                        frames_data[frame_num] = (image_path, rel_image_path, response)
                        loaded_count += 1
                except Exception as e:
                    skipped_lines += 1

    if not frames_data:
        print("錯誤: 未能從 JSONL 文件加載任何有效的幀數據。")
        return
    print(f"成功加載 {loaded_count} 個幀的數據。跳過 {skipped_lines} 個無法處理的行/記錄。")
    sorted_frame_nums = sorted(frames_data.keys())

    # --- 2. 計算 T-MAD 並比較標籤 (僅限連續幀) ---
    results = []
    tmad_values_changed_stats = []
    tmad_values_unchanged_stats = []
    # --- 新增：用於準確度計算的計數器和列表 ---
    tp = 0 # True Positives
    tn = 0 # True Negatives
    fp = 0 # False Positives
    fn = 0 # False Negatives
    false_positives_list = []
    false_negatives_list = []
    classification_threshold = args.classification_threshold # 獲取分類門檻值

    print(f"開始計算連續幀對的 T-MAD (閾值={args.diff_threshold}) 並評估分類 (門檻={classification_threshold})...")
    skipped_discontinuous = 0
    for i in tqdm(range(len(sorted_frame_nums) - 1)):
        current_frame_num = sorted_frame_nums[i]
        next_frame_num = sorted_frame_nums[i+1]

        if next_frame_num != current_frame_num + 1:
            skipped_discontinuous += 1
            continue

        frame1_abs_path, _, frame1_label = frames_data[current_frame_num]
        frame2_abs_path, _, frame2_label = frames_data[next_frame_num]

        t_mad = calculate_tmad(frame1_abs_path, frame2_abs_path, args.roi, args.diff_threshold)

        if t_mad is not None:
            actual_label_changed = (frame1_label != frame2_label)

            # 記錄結果到 CSV
            results.append({
                'Frame_Pair': f'{current_frame_num}_vs_{next_frame_num}',
                'T_MAD': t_mad,
                'Label_Changed': actual_label_changed,
                'Label_Frame1': frame1_label,
                'Label_Frame2': frame2_label
            })

            # --- 僅在兩個標籤都非空時進行統計和準確度評估 ---
            if frame1_label and frame2_label:
                # 添加到統計列表
                if actual_label_changed:
                    tmad_values_changed_stats.append(t_mad)
                else:
                    tmad_values_unchanged_stats.append(t_mad)

                # --- 進行分類預測並計算準確度指標 ---
                predicted_label_changed = (t_mad >= classification_threshold)

                if predicted_label_changed and actual_label_changed:
                    tp += 1
                elif not predicted_label_changed and not actual_label_changed:
                    tn += 1
                elif predicted_label_changed and not actual_label_changed:
                    fp += 1
                    false_positives_list.append({
                        'Frame_Pair': f'{current_frame_num}_vs_{next_frame_num}',
                        'T_MAD': t_mad,
                        'Label_Frame1': frame1_label,
                        'Label_Frame2': frame2_label,
                        'Reason': f'預測變化 (T-MAD={t_mad:.2f} >= {classification_threshold}), 實際未變'
                    })
                elif not predicted_label_changed and actual_label_changed:
                    fn += 1
                    false_negatives_list.append({
                        'Frame_Pair': f'{current_frame_num}_vs_{next_frame_num}',
                        'T_MAD': t_mad,
                        'Label_Frame1': frame1_label,
                        'Label_Frame2': frame2_label,
                        'Reason': f'預測未變 (T-MAD={t_mad:.2f} < {classification_threshold}), 實際已變'
                    })

    print(f"跳過了 {skipped_discontinuous} 個不連續的幀對。")

    # --- 3. 將結果寫入 CSV ---
    if not results:
        print("沒有計算出任何有效的 T-MAD 結果，無法生成 CSV 文件。")
        return
    print(f"\n將 {len(results)} 條連續幀對結果寫入 CSV 文件: {args.output_csv}")
    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Frame_Pair', 'T_MAD', 'Label_Changed', 'Label_Frame1', 'Label_Frame2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print("CSV 文件寫入成功。")
    except IOError as e:
        print(f"錯誤: 無法寫入 CSV 文件 {args.output_csv}: {e}")
        return

    # --- 4. 統計 T-MAD 範圍 (使用過濾後的數據) ---
    print("\n--- T-MAD 統計 (僅包含兩個標籤都非空的連續幀對) ---")
    if tmad_values_changed_stats:
        min_tmad_c = np.min(tmad_values_changed_stats)
        max_tmad_c = np.max(tmad_values_changed_stats)
        avg_tmad_c = np.mean(tmad_values_changed_stats)
        print(f"標籤發生變化的幀對 ({len(tmad_values_changed_stats)} 個):")
        print(f"  T-MAD 範圍: [{min_tmad_c:.4f}, {max_tmad_c:.4f}]")
        print(f"  T-MAD 平均值: {avg_tmad_c:.4f}")
    else:
        print("沒有找到兩個標籤都非空且標籤發生變化的幀對。")
    if tmad_values_unchanged_stats:
        min_tmad_u = np.min(tmad_values_unchanged_stats)
        max_tmad_u = np.max(tmad_values_unchanged_stats)
        avg_tmad_u = np.mean(tmad_values_unchanged_stats)
        print(f"\n標籤未發生變化的幀對 ({len(tmad_values_unchanged_stats)} 個):")
        print(f"  T-MAD 範圍: [{min_tmad_u:.4f}, {max_tmad_u:.4f}]")
        print(f"  T-MAD 平均值: {avg_tmad_u:.4f}")
    else:
        print("沒有找到兩個標籤都非空且標籤未發生變化的幀對。")

    # --- 5. 計算並打印準確度及錯誤列表 ---
    print(f"\n--- 分類準確度評估 (T-MAD 門檻值 = {classification_threshold}) ---")
    total_evaluated = tp + tn + fp + fn
    if total_evaluated > 0:
        accuracy = (tp + tn) / total_evaluated * 100 # 轉換為百分比
        print(f"總共評估的有效幀對 (標籤非空): {total_evaluated}")
        print(f"  - True Positives (TP - 預測變化，實際變化): {tp}")
        print(f"  - True Negatives (TN - 預測未變，實際未變): {tn}")
        print(f"  - False Positives (FP - 預測變化，實際未變): {fp}")
        print(f"  - False Negatives (FN - 預測未變，實際已變): {fn}")
        print(f"\n準確度 (Accuracy): {accuracy:.2f}%")

        # 可選：計算其他指標
        if tp + fp > 0:
            precision = tp / (tp + fp) * 100
            print(f"精確率 (Precision): {precision:.2f}% (預測為變化的幀對中，實際變化的比例)")
        else:
            print("精確率 (Precision): N/A (沒有預測為變化的幀對)")

        if tp + fn > 0:
            recall = tp / (tp + fn) * 100
            print(f"召回率 (Recall): {recall:.2f}% (實際變化的幀對中，被成功預測的比例)")
        else:
            print("召回率 (Recall): N/A (沒有實際變化的幀對)")

        if precision is not None and recall is not None and precision + recall > 0:
             f1 = 2 * (precision * recall) / (precision + recall)
             print(f"F1 分數 (F1 Score): {f1:.2f}%")
        else:
             print("F1 分數 (F1 Score): N/A")

        # 打印錯誤分類的列表
        if false_positives_list:
            print(f"\n--- 錯誤分類：False Positives (預測變化，實際未變) ---")
            for err in false_positives_list:
                print(f"  - {err['Frame_Pair']}: T-MAD={err['T_MAD']:.4f}, Labels='{err['Label_Frame1']}' -> '{err['Label_Frame2']}'")
        else:
            print("\n--- 沒有 False Positives ---")

        if false_negatives_list:
            print(f"\n--- 錯誤分類：False Negatives (預測未變，實際已變) ---")
            for err in false_negatives_list:
                print(f"  - {err['Frame_Pair']}: T-MAD={err['T_MAD']:.4f}, Labels='{err['Label_Frame1']}' -> '{err['Label_Frame2']}'")
        else:
            print("\n--- 沒有 False Negatives ---")

    else:
        print("沒有找到兩個標籤都非空的連續幀對，無法進行準確度評估。")

    print("\n分析完成。")

if __name__ == "__main__":
    main()