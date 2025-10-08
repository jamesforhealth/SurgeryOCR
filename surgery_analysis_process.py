#!/usr/bin/env python3
"""
Script: surgery_analysis_process.py
-----------------------------------
A standalone utility that scans every frame of a video, detects content changes
inside a user‑defined ROI **after binarisation**, and performs OCR only on the
frames where a change is detected.  Results are stored in *jsonl* format fully
compatible with the existing GUI tool (e.g. ``region2_ocr.jsonl``).

Highlights
~~~~~~~~~~
* **Binarised diff** – each ROI frame is converted to a binary image (OTSU or a
  simple rule‑based foreground extraction).  A change is registered when the
  proportion of differing pixels w.r.t. the previous binary ROI exceeds a user
  threshold.
* **CLI‑friendly** – invoke via ``python surgery_analysis_process.py --video
  VIDEO_PATH --region REGION_NAME``.  All key parameters are configurable
  through flags.
* **Performance logging** – reports average *detection* latency, average *OCR*
  latency, frames analysed, and frames OCR‑ed.
* **Dependency‑light** – only relies on ``opencv‑python``, ``easyocr``, and
  ``numpy``.  (EasyOCR will automatically fall back to CPU if CUDA is not
  available.)

Usage
~~~~~
```bash
python surgery_analysis_process.py \
    --video /path/to/video.mp4 \
    --region region2 \
    --roi-config config/rois.json \
    --method otsu \
    --diff-thresh 0.01
```

Arguments
~~~~~~~~~
* ``--video``        : Path to the video file.
* ``--region``       : ROI name (must exist in the ROI config).
* ``--roi-config``   : JSON file containing { region_name: [x1, y1, x2, y2] }.
                       Default: ``config/rois.json``.
* ``--method``       : Binarisation method – ``otsu`` (default) | ``rule``.
* ``--diff-thresh``  : Proportion of differing pixels (0‑1) needed to flag a
                       change; default **0.005** (0.5 %).
* ``--save-dir``     : Directory to write the ``*_ocr.jsonl``.  Defaults to the
                       directory of the video file.
* ``--device``       : ``gpu`` or ``cpu`` for EasyOCR.  Auto‑detect by default.

Output
~~~~~~
* ``<save-dir>/<region>_ocr.jsonl`` – one JSON per line:
  ``{"frame": 4345, "ocr_text": "123"}``
* Console summary with timing statistics.

"""
from __future__ import annotations
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 使用與 UI app 相同的 OCR 接口
from models.OCR_interface import get_ocr_model
import torch
import matplotlib.pyplot as plt

# --- Core API Import ---
from extract_roi_images import process_video_frames
from utils.get_configs import load_roi_config, load_stage_config, load_roi_header_config, load_ocr_char_sets_config, load_setting_regions_config
from utils.get_analysis_results import get_stage_analysis_json

SINGLE_DIGIT_BORDER = 40        # px 左右黑邊寬度門檻
SINGLE_DIGIT_THRESH = 0.03      # 左或右 >3% 白點 ⇒ 不是單一數字
# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------




def calculate_binary_diff(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        return 0.0
    b1 = (img1 > 127).astype(np.uint8)
    b2 = (img2 > 127).astype(np.uint8)
    diff = np.logical_xor(b1, b2)
    return float(np.mean(diff))

def is_single_digit(bw: np.ndarray) -> bool:
    """Return True if ROI likely contains exactly *one* digit.

    Heuristic: check left & right 40‑pixel columns for white ratio.
    """
    h, w = bw.shape
    border = min(SINGLE_DIGIT_BORDER, w // 3)  # avoid over‑size for narrow ROI
    left_white  = np.count_nonzero(bw[:, :border]) / (h * border)
    #right_white = np.count_nonzero(bw[:, -border:]) / (h * border)
    return left_white < SINGLE_DIGIT_THRESH #and right_white < SINGLE_DIGIT_THRESH


def trim_black_borders(binary_img: np.ndarray, max_border: int = 1) -> np.ndarray:
    """
    去除二值化圖像四個方向的黑邊，保留中間的白色內容，最多保留指定像素的黑邊。
    
    Args:
        binary_img: 二值化圖像 (0=黑, 255=白)
        max_border: 最多保留的黑邊像素數 (預設1像素)
    
    Returns:
        裁切後的二值化圖像
    """
    if binary_img.size == 0:
        return binary_img
    
    h, w = binary_img.shape
    
    # 找到有白色像素的邊界
    white_pixels = binary_img > 127  # 白色像素的遮罩
    
    # 找到包含白色像素的行和列
    rows_with_white = np.any(white_pixels, axis=1)  # 每一行是否有白色
    cols_with_white = np.any(white_pixels, axis=0)  # 每一列是否有白色
    
    # 如果沒有白色像素，返回原圖
    if not np.any(rows_with_white) or not np.any(cols_with_white):
        return binary_img
    
    # 找到第一個和最後一個包含白色的行/列
    top = np.argmax(rows_with_white)
    bottom = len(rows_with_white) - 1 - np.argmax(rows_with_white[::-1])
    left = np.argmax(cols_with_white)
    right = len(cols_with_white) - 1 - np.argmax(cols_with_white[::-1])
    
    # 加上最多 max_border 像素的邊框，但不超出原圖範圍
    top = max(0, top - max_border)
    bottom = min(h - 1, bottom + max_border)
    left = max(0, left - max_border)
    right = min(w - 1, right + max_border)
    
    # 裁切圖像
    trimmed = binary_img[top:bottom+1, left:right+1]
    
    return trimmed


def show_before_after_comparison(original: np.ndarray, trimmed: np.ndarray, frame_idx: int, description: str):
    """顯示前後對照的圖片"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    #set figure title
    fig.suptitle(f'Frame {frame_idx}')

    ax1.imshow(original, cmap='gray')
    ax1.set_title(f'Origin Frame')
    ax1.set_xlabel(f'尺寸: {original.shape[1]}x{original.shape[0]}')
    ax1.axis('off')
    
    ax2.imshow(trimmed, cmap='gray')
    ax2.set_title(f'New Frame')
    ax2.set_xlabel(f'尺寸: {trimmed.shape[1]}x{trimmed.shape[0]}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Core processing routine
# ---------------------------------------------------------------------------

def _is_active_header(
    frame: np.ndarray,
    region_name: str,
    roi_header_dict: Dict[str, List[int]] | None,
    frame_idx: int = 0,
    diff_threshold: float = 0.2
) -> bool:
    """
    檢查指定region的header是否在當前frame中active（與快取圖像匹配）
    
    Args:
        frame: 完整的幀圖像 (BGR格式)
        region_name: ROI區域名稱
        roi_header_dict: header座標字典
        diff_threshold: 差異閾值，小於此值認為header匹配
        
    Returns:
        bool: True表示header已activate，False表示未activate
    """
    # 如果沒有header配置，默認總是active
    # if not roi_header_dict or region_name not in roi_header_dict:
    #     return True
   
    header_coords = roi_header_dict[region_name]
    cache_path = Path("data/roi_img_caches/roi_headers") / f"{region_name}.png"

    # if frame_idx > 2220:
    #     input(f"header_coords: {header_coords}, cache_path: {cache_path}, region_name: {region_name}, frame_idx: {frame_idx}, diff_threshold: {diff_threshold}")

    if len(header_coords) != 4:
        return True
    
    # 載入快取的header圖像
    if not cache_path.exists():
        print(f"警告: 找不到 {region_name} 的header快取圖像: {cache_path}")
        return True  # 沒有快取時默認active
    
    try:
        # 載入快取的二值化header圖像
        cached_header_pil = Image.open(cache_path).convert('L')
        cached_header = np.array(cached_header_pil)
    except Exception as e:
        print(f"警告: 載入 {region_name} header快取失敗: {e}")
        return True
    
    x1, y1, x2, y2 = header_coords
    
    # 確保座標在圖像範圍內
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        return True  # 無效座標，默認active
    
    # 提取當前幀的header區域
    header_roi_bgr = frame[y1:y2, x1:x2]
    
    # 應用與快取一致的二值化處理
    try:
        from utils.core_processing import binarize
        current_header = binarize(header_roi_bgr, method="rule")
    except ImportError:
        # 如果無法導入，使用簡單的OTSU二值化作為備用
        header_gray = cv2.cvtColor(header_roi_bgr, cv2.COLOR_BGR2GRAY)
        _, current_header = cv2.threshold(header_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception as e:
        print(f"警告: {region_name} header二值化失敗: {e}")
        return True
    
    # 檢查尺寸是否一致
    if cached_header.shape != current_header.shape:
        print(f"警告: {region_name} header尺寸不匹配: 快取{cached_header.shape} vs 當前{current_header.shape}")
        return True
    
    # 使用binary diff計算差異
    diff_ratio = calculate_binary_diff(cached_header, current_header)
    
    # if frame_idx > 2220:
    #     input(f"_is_active_header region_name: {region_name}, frame_idx: {frame_idx}, diff_threshold: {diff_threshold}, diff_ratio: {diff_ratio}")

    # 差異小於閾值表示header匹配（已active）
    return diff_ratio < diff_threshold




def active_in_stage(
    stage_analysis_dict: Dict[str, List[Dict[str, Any]]] | None, # stage_analysis.json
    stage_activation_dict: Dict[str, List[int]] | None,
    roi_header_dict: Dict[str, List[int]] | None,
    frame_idx: int,
    region_name: str,
    current_frame: np.ndarray | None = None  # 新增：當前幀數據
) -> bool:
    """Check if OCR should be active for the given frame and region based on stage config and header cache img"""

    # Header檢測（如果有frame數據和header配置）
    if current_frame is not None and roi_header_dict:
        if not _is_active_header(current_frame, region_name, roi_header_dict, frame_idx):
            return False  # Header not active means always inactive
    
    if not stage_activation_dict:
        return True  # No config is considered as always active

    if region_name not in stage_activation_dict:
        return True  # Region not specified is considered as always active

    active_stages = stage_activation_dict[region_name]
    if not active_stages:
        return True  # Empty list is considered as always active!
    
    current_stage = -1 
    segment_dict = stage_analysis_dict.get("regions", {}).get("STAGE")
    
    for stage_segment in segment_dict:
        start_frame = stage_segment.get("start_frame")
        if start_frame is not None and frame_idx < start_frame:
            break
        current_stage = stage_segment.get("pattern", -1)
        
    if current_stage in active_stages: 
        return True 
    return False # No matched stage segments means always none active


def active_by_header_detection(
    frame: np.ndarray,
    region_name: str,
    roi_header_dict: Dict[str, List[int]],
    diff_threshold: float = 0.2
) -> bool:
    """
    基於header區域檢測判斷region是否已activate。
    
    Args:
        frame: 完整的幀圖像 (BGR格式)
        region_name: ROI區域名稱
        roi_header_dict: header座標字典
        diff_threshold: 差異閾值，小於此值認為header存在
        
    Returns:
        bool: True表示region已activate，False表示未activate
    """
    if region_name not in roi_header_dict:
        return True  # 沒有header配置的region默認總是activate
    
    header_coords = roi_header_dict[region_name]
    return detect_header_presence(frame, header_coords, region_name, diff_threshold)


def determine_if_setting_value(
    ocr_text: str, 
    binary_image: np.ndarray, 
    region_name: str, 
    frame_idx: int
) -> bool:
    """
    判斷單一數值的情況下是否為設定值
    
    Args:
        ocr_text: OCR識別的文字
        binary_image: 二值化後的圖像
        region_name: 區域名稱
        frame_idx: 幀編號
    
    Returns:
        True if 設定值, False if 運作值
    
    判斷邏輯：
    1. 載入設定區域配置
    2. 如果區域不在需要判斷的列表中，返回預設值
    3. 分析指定子區域的白色像素比例
    4. 比例超過閾值則為運作值（大數字），否則為設定值（小數字）
    """
    
    # 載入配置
    config = load_setting_regions_config()
    
    # 檢查是否需要進行設定值檢測
    regions_with_detection = config.get("regions_with_setting_detection", [])
    if region_name not in regions_with_detection:
        # 不在檢測列表中的區域，返回預設值
        return config.get("default_setting_value", False)
    
    # 取得檢測配置
    detection_config = config.get("detection_config", {})
    sub_coords = detection_config.get("sub_region_coords", {})
    threshold = detection_config.get("white_pixel_threshold", 0.02)
    
    # 檢查子區域座標配置
    if not all(key in sub_coords for key in ["x1", "x2", "y1", "y2"]):
        print(f"警告: {region_name} 缺少子區域座標配置，使用預設值")
        return config.get("default_setting_value", False)
    
    # 取得子區域座標
    x1, x2 = sub_coords["x1"], sub_coords["x2"]
    y1, y2 = sub_coords["y1"], sub_coords["y2"]
    
    # 檢查座標是否在圖像範圍內
    h, w = binary_image.shape[:2]
    if x2 > w or y2 > h or x1 < 0 or y1 < 0:
        print(f"警告: {region_name} 子區域座標 ({x1},{y1},{x2},{y2}) 超出圖像範圍 ({w},{h})")
        return config.get("default_setting_value", False)
    
    # 裁切子區域
    sub_region = binary_image[y1:y2, x1:x2]
    
    if sub_region.size == 0:
        print(f"警告: {region_name} 子區域為空")
        return config.get("default_setting_value", False)
    
    # 計算白色像素比例
    white_pixels = np.sum(sub_region > 0)  # 二值化圖像中白色像素（值>0）
    total_pixels = sub_region.size
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    
    # 判斷：白色比例超過閾值則為運作值（大數字），否則為設定值（小數字）
    is_operation_value = white_ratio > threshold
    
    # 可選：輸出除錯資訊
    # print(f"[DEBUG] {region_name} frame {frame_idx}: 白色比例={white_ratio:.4f}, 閾值={threshold:.4f}, 判斷={'運作值' if is_operation_value else '設定值'}")
    
    return not is_operation_value  # 返回 True 表示設定值，False 表示運作值


def process_video(
    video_path: Path,
    rois: List[Tuple[str, Tuple[int, int, int, int]]],
    method: str = "rule",
    diff_thresh: float = 0.01,
    save_dir: Path | None = None,
    stage_activation_dict: Dict[str, List[int]] | None = None,
    roi_header_dict: Dict[str, List[int]] | None = None,
    char_sets_dict: Dict[str, str] | None = None
) -> None:
    """
    對單一影片的所有指定ROI區域進行變化檢測與OCR。
    使用 `process_video_frames` 作為統一的數據源。
    """
    if not save_dir:
        video_name = video_path.stem
        save_dir = Path("data") / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    stage_analysis_dict = get_stage_analysis_json( save_dir / "stage_analysis.json")

    if not char_sets_dict:
        char_sets_dict = {}

    # --- 初始化 ---
    ocr_iface = get_ocr_model(
        model_type="easyocr",
        gpu=torch.cuda.is_available(),
        lang_list=['en'],
        confidence_threshold=0.5,
        debug_output=False
    )
    
    # 為每個ROI區域維護獨立的狀態
    prev_bins = {}  # type: Dict[str, np.ndarray | None]
    prev_active_states = {}  # type: Dict[str, bool | None] # 追蹤前一幀的activate狀態
    
    # 用於分組結果的新數據結構
    multi_digit_groups = {region_name: [] for region_name, _ in rois}
    single_digit_results = {region_name: [] for region_name, _ in rois}
    cache_map = {region_name: [] for region_name, _ in rois} # type: Dict[str, List[Tuple[np.ndarray, int]]]

    stats = {
        region_name: {"ocr_time_total": 0.0, "ocr_count": 0, "change_count": 0}
        for region_name, _ in rois
    }
    
    t0 = time.perf_counter()

    # --- 核心處理：只調用一次API ---
    frame_processor = process_video_frames(
        video_path=video_path,
        roi_items=rois,
        binarize_method=method,
    )
    
    pbar = None
    total_frames = 0
    
    # --- 單一循環處理所有幀和所有ROI ---
    for frame_data in frame_processor:
        frame_idx = frame_data["frame_idx"]
        rois_data = frame_data["rois"]
        original_frame = frame_data.get("original_frame")  # 獲取原始幀

        if pbar is None:
            total_frames = frame_data["total_frames"]
            pbar = tqdm(total=total_frames, desc=f"分析 {video_path.name}", unit="frame")

        for region_name, roi_data in rois_data.items():
            curr_bin = roi_data["binary_np"]
            prev_bin = prev_bins.get(region_name)
            
            # 檢查當前幀的activate狀態
            current_active = active_in_stage(stage_analysis_dict, stage_activation_dict, roi_header_dict, frame_idx, region_name, original_frame)
            prev_active = prev_active_states.get(region_name)
            
            # 檢測 activate → deactivate 轉換
            if prev_active is True and current_active is False:
                # 記錄deactivate事件
                single_digit_results[region_name].append({
                    "type": "deactivate",
                    "frame": frame_idx,
                    "ocr_text": "(Deactivated)",
                    "confidence": 1.0,
                    "setting": False  # deactivate事件不是設定值
                })
                stats[region_name]["change_count"] += 1
                print(f"檢測到 {region_name} 在frame {frame_idx} 從activate轉為deactivate")
            
            # 更新狀態
            prev_bins[region_name] = curr_bin
            prev_active_states[region_name] = current_active
            
            # 如果當前幀不是active，跳過後續OCR處理
            if not current_active:
                continue

            change = False
            if prev_bin is None:
                change = True  # 永遠OCR第一幀
            else:
                diff = calculate_binary_diff(prev_bin, curr_bin)
                change = diff >= diff_thresh

            # if frame_idx > 2220:
            #     input(f"\ndiff_thresh: {diff_thresh}, diff: {diff} change: {change}")

            if change:
                stats[region_name]["change_count"] += 1
                t1 = time.perf_counter()
                
                from_cache = False
                ocr_calls_in_frame = 0

                # --- Step 1: 判斷是否為多數字組合並嘗試從快取查找 ---
                is_multi_digit = not is_single_digit(curr_bin)
                
                if is_multi_digit:
                    for cached_bin, group_index in cache_map[region_name]:
                        if calculate_binary_diff(curr_bin, cached_bin) < 0.01:
                            multi_digit_groups[region_name][group_index]["matched_frames"].append(frame_idx)
                            from_cache = True
                            break
                # if frame_idx > 2220:
                #     input(f"is_multi_digit: {is_multi_digit}, from_cache: {from_cache}")
                # --- Step 2: 如果不在快取中，則執行OCR ---
                if not from_cache:
                    # 根據 region 獲取特定的 allowlist，如果沒有則為 None (使用模型預設值)
                    custom_allowlist = char_sets_dict.get(region_name)
                    ocr_result, confidence = ocr_iface.recognize(Image.fromarray(curr_bin), allowlist=custom_allowlist)
                    ocr_calls_in_frame += 1

                    # --- 後備方案：如果OCR失敗，嘗試裁切 ---
                    if not ocr_result:
                        trimmed_bin = trim_black_borders(curr_bin, max_border=1)
                        if trimmed_bin.size > 0 and trimmed_bin.shape != curr_bin.shape:
                            ocr_result, confidence = ocr_iface.recognize(Image.fromarray(trimmed_bin), allowlist=custom_allowlist)
                            ocr_calls_in_frame += 1
                    
                    # --- Step 3: 記錄新結果 ---
                    if is_multi_digit:
                        if ocr_result: # 只有成功識別才創建新組
                            new_group = {
                                "type": "multi_digit_group",
                                "source_frame": frame_idx,
                                "ocr_text": ocr_result,
                                "confidence": confidence,
                                "matched_frames": [frame_idx],
                                "setting": True  # multi_digit_group 總是設定值
                            }
                            multi_digit_groups[region_name].append(new_group)
                            new_group_index = len(multi_digit_groups[region_name]) - 1
                            cache_map[region_name].append((curr_bin, new_group_index))
                    else: # 單數字
                        # TODO: 添加設定值判斷邏輯，目前預設為運作值
                        is_setting_value = determine_if_setting_value(ocr_result, curr_bin, region_name, frame_idx)
                        single_digit_results[region_name].append({
                            "type": "single_digit",
                            "frame": frame_idx,
                            "ocr_text": ocr_result,
                            "confidence": confidence,
                            "setting": is_setting_value
                        })

                ocr_elapsed = time.perf_counter() - t1

                # --- Step 4: 更新統計數據 ---
                stats[region_name]["ocr_time_total"] += ocr_elapsed
                stats[region_name]["ocr_count"] += ocr_calls_in_frame

            prev_bins[region_name] = curr_bin

        # 更新進度條
        total_ocr_count = sum(s["ocr_count"] for s in stats.values())
        pbar.set_postfix({"Total OCRs": total_ocr_count})
        pbar.update(1)

    detection_time_total = time.perf_counter() - t0
    if pbar:
        pbar.close()

    # --- 保存結果並打印報告 ---
    print("\n" + "="*20 + f" 分析報告: {video_path.name} " + "="*20)
    print(f"總耗時        : {detection_time_total:.2f} 秒")
    print(f"總幀數        : {total_frames}")
    
    avg_detection_fps = total_frames / detection_time_total if detection_time_total > 0 else 0
    print(f"平均檢測速度  : {avg_detection_fps:.2f} FPS")

    for region_name, _ in rois:
        change_count = stats[region_name]["change_count"]
        ocr_count = stats[region_name]["ocr_count"]
        if change_count == 0:
            continue
            
        save_path = save_dir / f"{region_name}_ocr_testing.jsonl"
        
        # 組合和排序結果
        region_groups = multi_digit_groups[region_name]
        region_singles = single_digit_results[region_name]
        all_results_for_region = region_groups + region_singles
        all_results_for_region.sort(key=lambda r: r.get("source_frame", r.get("frame", 0)))

        with open(save_path, "w", encoding="utf-8") as f:
            for rec in all_results_for_region:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        ocr_time_total = stats[region_name]["ocr_time_total"]
        avg_change_processing_ms = (ocr_time_total / change_count * 1000) if change_count > 0 else 0

        print(f"\n--- ROI: {region_name} ---")
        print(f"  變動幀數       : {change_count} / {total_frames} ({change_count/total_frames:.2%})")
        print(f"  實際OCR次數    : {ocr_count}")
        print(f"  平均變動處理耗時: {avg_change_processing_ms:.2f} ms/frame")
        print(f"  結果保存至     : {save_path}")

    print("="* (42 + len(video_path.name)))


# ---------------------------------------------------------------------------
# Testing API for UI integration
# ---------------------------------------------------------------------------

def test_current_frame_change(
    frame_idx: int,
    video_name: str,
    region_name: str = "region2",
    diff_threshold: float = 0.01,
    binarize_method: str = "rule",  # 保留此參數以保持接口一致性，但不使用
    roi_config_path: Path = Path("config/rois.json")
) -> Dict[str, Any]:
    """
    測試指定frame是否會被算法判定為frame change。
    直接讀取已經預處理好的二值化圖片進行測試。
    
    Args:
        frame_idx: 要測試的frame index
        video_name: 影片名稱 (對應data/目錄下的子目錄名，如："2024-10-18周建碧OS")
        region_name: ROI區域名稱 (如："region2")
        diff_threshold: 變化檢測閾值 (0-1)
        binarize_method: 二值化方法（保留參數，但不使用，因為圖片已預處理）
        roi_config_path: ROI配置文件路徑（用於驗證region_name）
    
    Returns:
        測試結果字典:
        {
            'is_change': bool,              # 是否檢測到變化
            'diff_ratio': float,            # 實際差異比例
            'threshold': float,             # 使用的閾值
            'current_frame': int,           # 當前frame
            'previous_frame': int,          # 前一frame
            'method': str,                  # 使用的方法（直接讀取）
            'reason': str,                  # 判斷原因
            'current_image_path': str,      # 當前幀圖片路徑
            'previous_image_path': str,     # 前一幀圖片路徑
            'error': str (optional)         # 錯誤信息（如果有）
        }
    """
    try:
        # 1. 驗證ROI配置（確保region_name有效）
        roi_dict = load_roi_config(roi_config_path)
        if region_name not in roi_dict:
            return {'error': f'ROI配置中找不到區域 "{region_name}"'}
        
        # 2. 構建圖片目錄路徑
        region_dir = Path("data") / video_name / region_name
        if not region_dir.exists():
            return {'error': f'找不到區域目錄: {region_dir}'}
        
        # 3. 檢查frame範圍
        if frame_idx <= 0:
            return {'error': '當前為第一幀或更早，無法與前一幀比較'}
        
        # 4. 構建圖片文件路徑
        previous_frame_idx = frame_idx - 1
        current_image_path = region_dir / f"frame_{frame_idx}_binary.png"
        previous_image_path = region_dir / f"frame_{previous_frame_idx}_binary.png"
        
        # 5. 檢查圖片文件是否存在
        if not previous_image_path.exists():
            return {'error': f'找不到前一幀二值化圖片: {previous_image_path}'}
        
        if not current_image_path.exists():
            return {'error': f'找不到當前幀二值化圖片: {current_image_path}'}
        
        # 6. 讀取二值化圖片
        try:
            prev_binary_pil = Image.open(previous_image_path).convert('L')  # 確保是灰階
            curr_binary_pil = Image.open(current_image_path).convert('L')   # 確保是灰階
            
            prev_binary = np.array(prev_binary_pil)
            curr_binary = np.array(curr_binary_pil)
            
        except Exception as e:
            return {'error': f'讀取圖片失敗: {str(e)}'}
        
        # 7. 檢查圖片尺寸是否一致
        if prev_binary.shape != curr_binary.shape:
            return {'error': f'圖片尺寸不一致: 前一幀{prev_binary.shape} vs 當前幀{curr_binary.shape}'}
        
        # 8. 計算差異並判斷（使用與主流程完全相同的邏輯）
        diff_ratio = calculate_binary_diff(prev_binary, curr_binary)
        is_change = diff_ratio >= diff_threshold
        
        # 9. 生成結果
        result = {
            'is_change': is_change,
            'diff_ratio': diff_ratio,
            'threshold': diff_threshold,
            'current_frame': frame_idx,
            'previous_frame': previous_frame_idx,
            'method': 'direct_read_binary',  # 表明是直接讀取二值化圖片
            'current_image_path': str(current_image_path),
            'previous_image_path': str(previous_image_path),
            'region_dir': str(region_dir)
        }
        
        if is_change:
            result['reason'] = f'檢測到變化：差異比例 {diff_ratio:.4f} ≥ 閾值 {diff_threshold:.4f}'
        else:
            result['reason'] = f'未檢測到變化：差異比例 {diff_ratio:.4f} < 閾值 {diff_threshold:.4f}'
        
        return result
        
    except Exception as e:
        return {'error': f'測試過程出錯: {str(e)}'}


# ---------------------------------------------------------------------------
# Entry‑point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="Binarised ROI frame‑change detection + OCR")
    p.add_argument("--video", required=True, type=Path, help="Path to video file or video files directory (e.g. mp4)")
    p.add_argument("--region", default="all", help="ROI name as in ROI config (e.g. region2)")
    p.add_argument("--roi-config", type=Path, default=Path("config/rois.json"), help="ROI config JSON path")
    p.add_argument("--stage-config", type=Path, default=Path("config/ocr_activation_stages.json"), help="Stage config JSON path")
    p.add_argument("--char-config", type=Path, default=Path("config/ocr_char_sets.json"), help="OCR character sets JSON path")
    p.add_argument("--method", choices=["otsu", "rule"], default="rule", help="Binarisation method")
    p.add_argument("--diff-thresh", type=float, default=0.01, help="Diff ratio threshold (0‑1) to flag change")
    p.add_argument("--save-dir", type=Path, help="Directory to save jsonl (defaults to video directory)")
    return p.parse_args()


def main():  # pragma: no cover
    args = parse_args()
    roi_dict = load_roi_config(args.roi_config)
    stage_activation_dict = load_stage_config(args.stage_config)
    char_sets_dict = load_ocr_char_sets_config(args.char_config)
    if char_sets_dict:
        print(f"載入 custom OCR character sets 成功: {list(char_sets_dict.keys())}")

    # 嘗試載入header配置
    roi_header_dict = {}
    try:
        roi_header_dict = load_roi_header_config(args.roi_config)
        if roi_header_dict:
            print(f"載入header配置成功，包含區域: {list(roi_header_dict.keys())}")
        else:
            print("未找到header配置，將使用stage-based activation")
    except Exception as e:
        print(f"載入header配置失敗: {e}，將使用stage-based activation")

    rois = []
    if args.region == "all":
        for region in roi_dict:
            rois.append((region, tuple(roi_dict[region])))  # type: ignore[arg-type]
            # input(f"ROI: {rois}")
    else:
        if args.region not in roi_dict:
            raise SystemExit(f"Region '{args.region}' not found in ROI config {args.roi_config}")
        rois = [(args.region, tuple(roi_dict[args.region]))]  # type: ignore[arg-type]

    if args.video.is_dir():
        video_files = list(args.video.glob("*.mp4"))
        for video_file in video_files:
            print(f"Processing video: {video_file}")
            process_video(
                video_file, rois, args.method, args.diff_thresh, args.save_dir, 
                stage_activation_dict, roi_header_dict, char_sets_dict
            )

    elif str(args.video)[-4:].lower() == ".mp4":
        process_video(
            args.video, rois, args.method, args.diff_thresh, args.save_dir, 
            stage_activation_dict, roi_header_dict, char_sets_dict
        )


if __name__ == "__main__":  # pragma: no cover
    main()