#!/usr/bin/env python3
"""
Script: surgery_analysis_process.py
-----------------------------------
Frame-change detection + OCR utility (CLI & Library Mode).
"""
from __future__ import annotations
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 使用與 UI app 相同的 OCR 接口
from models.OCR_interface import get_ocr_model
import torch

# --- Core API Import ---
from extract_roi_images import iterate_roi_binary_from_cache
from extract_frame_cache import get_video_meta
from utils.cv_processing import calculate_average_binary_diff, is_single_digit, trim_black_borders, binarize
from utils.get_analysis_results import get_stage_analysis_json
from utils.get_configs import load_roi_config, load_stage_config, load_roi_header_config, load_ocr_char_sets_config, load_setting_regions_config
from utils.get_paths import resolve_video_analysis_dir

# --------------------------------------------------------------------------
# [NEW] OCRProcessor Class for Pipeline Integration
# --------------------------------------------------------------------------
class OCRProcessor:
    def __init__(self, 
                 stage_activation_dict: Dict[str, List[int]], 
                 roi_header_dict: Dict[str, List[int]],
                 char_sets_dict: Dict[str, str],
                 diff_threshold: float = 0.01):
        
        self.stage_activation_dict = stage_activation_dict
        self.roi_header_dict = roi_header_dict
        self.char_sets_dict = char_sets_dict
        self.diff_threshold = diff_threshold
        
        # Initialize OCR Model
        self.ocr_iface = get_ocr_model(
            model_type="easyocr",
            gpu=torch.cuda.is_available(),
            lang_list=['en'],
            confidence_threshold=0.5,
            debug_output=False
        )
        
        # State containers
        self.prev_bins: Dict[str, Optional[np.ndarray]] = {}
        self.prev_active_states: Dict[str, bool] = {}
        self.cache_map: Dict[str, List[Tuple[np.ndarray, int]]] = {} # for multi-digit groups
        
        # Results containers
        self.multi_digit_groups: Dict[str, List[Dict]] = {}
        self.single_digit_results: Dict[str, List[Dict]] = {}
        
        # Stats
        self.stats: Dict[str, Dict] = {}

    def initialize_region(self, region_name: str):
        """Ensure containers are ready for a specific region"""
        if region_name not in self.prev_bins:
            self.prev_bins[region_name] = None
            self.prev_active_states[region_name] = None
            self.cache_map[region_name] = []
            self.multi_digit_groups[region_name] = []
            self.single_digit_results[region_name] = []
            self.stats[region_name] = {"ocr_time_total": 0.0, "ocr_count": 0, "change_count": 0}

    def process_frame(self, 
                      frame_bgr: np.ndarray, 
                      frame_idx: int, 
                      roi_config: Dict[str, Tuple[int, int, int, int]],
                      stage_result: Dict[str, Any] = None) -> Dict[str, List[Dict]]:
        """
        Process a single frame: crop ROIs -> binarize -> diff check -> OCR.
        Returns new OCR results found in this frame.
        """
        new_results = {} # { region_name: [result_dict, ...] }
        
        # Construct a virtual stage analysis dict for _active_in_stage
        # We assume stage_result comes from StageAnalyzer.process_frame() which returns {region: {pattern_id, rmse}}
        # But _active_in_stage expects the full JSON structure.
        # For single-pass, we simplify: we only need current STAGE pattern.
        
        current_stage_pattern = -1
        if stage_result and "STAGE" in stage_result:
            current_stage_pattern = stage_result["STAGE"].get("pattern_id", -1)
            if current_stage_pattern is None: 
                current_stage_pattern = -1

        for region_name, coords in roi_config.items():
            self.initialize_region(region_name)
            
            # 1. Active Check (Simplified for single-pass)
            is_active = self._check_active(region_name, frame_idx, current_stage_pattern, frame_bgr)
            
            # Handle Deactivation Event
            prev_active = self.prev_active_states.get(region_name)
            if prev_active is True and is_active is False:
                deact_res = {
                    "type": "deactivate",
                    "frame": frame_idx,
                    "ocr_text": "(Deactivated)",
                    "confidence": 1.0,
                    "setting": False
                }
                self.single_digit_results[region_name].append(deact_res)
                self.stats[region_name]["change_count"] += 1
                # if region_name not in new_results: new_results[region_name] = []
                # new_results[region_name].append(deact_res)
            
            self.prev_active_states[region_name] = is_active
            
            if not is_active:
                continue

            # 2. Crop & Binarize
            x1, y1, x2, y2 = coords
            try:
                roi_bgr = frame_bgr[y1:y2, x1:x2]
                if roi_bgr.size == 0: continue
                curr_bin = binarize(roi_bgr, method="rule")
            except Exception:
                continue
            
            # 3. Diff Check
            prev_bin = self.prev_bins.get(region_name)
            change = False
            if prev_bin is None:
                change = True
            else:
                diff = calculate_average_binary_diff(prev_bin, curr_bin)
                change = diff >= self.diff_threshold
            
            # Update State
            self.prev_bins[region_name] = curr_bin
            
            if change:
                self.stats[region_name]["change_count"] += 1
                t1 = time.perf_counter()
                
                from_cache = False
                ocr_calls = 0
                
                # OCR Logic
                custom_allowlist = self.char_sets_dict.get(region_name)
                is_multi_digit = not is_single_digit(curr_bin)
                
                found_res = None
                
                if is_multi_digit:
                    for cached_bin, group_index in self.cache_map[region_name]:
                        if calculate_average_binary_diff(curr_bin, cached_bin) < 0.01:
                            # Cache Hit
                            group = self.multi_digit_groups[region_name][group_index]
                            group["matched_frames"].append(frame_idx)
                            from_cache = True
                            break
                
                if not from_cache:
                    ocr_result, conf = self.ocr_iface.recognize(Image.fromarray(curr_bin), allowlist=custom_allowlist)
                    ocr_calls += 1
                    
                    if not ocr_result:
                        trimmed = trim_black_borders(curr_bin, max_border=1)
                        if trimmed.size > 0 and trimmed.shape != curr_bin.shape:
                             ocr_result, conf = self.ocr_iface.recognize(Image.fromarray(trimmed), allowlist=custom_allowlist)
                             ocr_calls += 1
                    
                    if is_multi_digit:
                        if ocr_result:
                            new_group = {
                                "type": "multi_digit_group",
                                "source_frame": frame_idx,
                                "ocr_text": ocr_result,
                                "confidence": conf,
                                "matched_frames": [frame_idx],
                                "setting": True
                            }
                            self.multi_digit_groups[region_name].append(new_group)
                            self.cache_map[region_name].append((curr_bin, len(self.multi_digit_groups[region_name])-1))
                            found_res = new_group
                    else:
                        is_setting = _determine_if_setting_value(ocr_result, curr_bin, region_name, frame_idx)
                        res = {
                            "type": "single_digit",
                            "frame": frame_idx,
                            "ocr_text": ocr_result,
                            "confidence": conf,
                            "setting": is_setting
                        }
                        self.single_digit_results[region_name].append(res)
                        found_res = res

                self.stats[region_name]["ocr_time_total"] += (time.perf_counter() - t1)
                self.stats[region_name]["ocr_count"] += ocr_calls
                
                if found_res:
                    if region_name not in new_results: new_results[region_name] = []
                    new_results[region_name].append(found_res)

        return new_results

    def _check_active(self, region_name, frame_idx, current_stage_pattern, frame_bgr):
        # 1. Check Header
        if self.roi_header_dict and frame_bgr is not None:
             if not _is_active_header(frame_bgr, region_name, self.roi_header_dict, frame_idx):
                 return False
                 
        # 2. Check Stage Config
        if not self.stage_activation_dict: return True
        if region_name not in self.stage_activation_dict: return True
        active_stages = self.stage_activation_dict[region_name]
        if not active_stages: return True
        
        return current_stage_pattern in active_stages

    def save_results(self, save_dir: Path):
        """Dump all accumulated results to jsonl files"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for region_name, _ in self.single_digit_results.items(): # Iterate known regions
            # Combine
            groups = self.multi_digit_groups.get(region_name, [])
            singles = self.single_digit_results.get(region_name, [])
            
            if not groups and not singles:
                continue
                
            all_results = groups + singles
            all_results.sort(key=lambda r: r.get("source_frame", r.get("frame", 0)))
            
            save_path = save_dir / f"{region_name}_ocr_testing.jsonl"
            with open(save_path, "w", encoding="utf-8") as f:
                for rec in all_results:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            # print(f"[OCR] Saved {len(all_results)} results to {save_path.name}")

# --------------------------------------------------------------------------
# Original Utilities (Kept for backward compatibility)
# --------------------------------------------------------------------------

def _is_active_header(
    frame: np.ndarray,
    region_name: str,
    roi_header_dict: Dict[str, List[int]] | None,
    frame_idx: int = 0,
    diff_threshold: float = 0.2
) -> bool:
    """
    檢查指定region的header是否在當前frame中active（與快取圖像匹配）
    """
    # if not roi_header_dict or region_name not in roi_header_dict:
    #     return True
    
    # For backward compatibility, we need to handle cases where dict might not have the key
    if not roi_header_dict or region_name not in roi_header_dict:
         return True

    header_coords = roi_header_dict[region_name]
    cache_path = Path("data/roi_img_caches/roi_headers") / f"{region_name}.png"

    if len(header_coords) != 4:
        return True
    
    # 載入快取的header圖像
    if not cache_path.exists():
        # print(f"警告: 找不到 {region_name} 的header快取圖像: {cache_path}")
        return True  # 沒有快取時默認active
    
    try:
        # 載入快取的二值化header圖像
        cached_header_pil = Image.open(cache_path).convert('L')
        cached_header = np.array(cached_header_pil)
    except Exception as e:
        # print(f"警告: 載入 {region_name} header快取失敗: {e}")
        return True
    
    x1, y1, x2, y2 = header_coords
    
    # 確保座標在圖像範圍內
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        return True  # 無效座標，默認active
    
    # 提取當前幀的header區域
    try:
        header_roi_bgr = frame[y1:y2, x1:x2]
        if header_roi_bgr.size == 0: return True
    except:
        return True
    
    # 應用與快取一致的二值化處理
    try:
        current_header = binarize(header_roi_bgr, method="rule")
    except Exception as e:
        # print(f"警告: {region_name} header二值化失敗: {e}")
        return True
    
    # 檢查尺寸是否一致
    if cached_header.shape != current_header.shape:
        # print(f"警告: {region_name} header尺寸不匹配")
        return True
    
    # 使用binary diff計算差異
    diff_ratio = calculate_average_binary_diff(cached_header, current_header)
    
    # 差異小於閾值表示header匹配（已active）
    return diff_ratio < diff_threshold

def _active_in_stage(
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
    segment_dict = stage_analysis_dict.get("regions", {}).get("STAGE") if stage_analysis_dict else None
    # 若沒有 stage 分析結果或缺少 STAGE 區段，視為總是啟用
    if not segment_dict:
        return True
    
    for stage_segment in segment_dict:
        start_frame = stage_segment.get("start_frame")
        if start_frame is not None and frame_idx < start_frame:
            break
        current_stage = stage_segment.get("pattern", -1)
        
    if current_stage in active_stages: 
        return True 
    return False # No matched stage segments means always none active

def _determine_if_setting_value(
    ocr_text: str, 
    binary_image: np.ndarray, 
    region_name: str, 
    frame_idx: int
) -> bool:
    """
    判斷單一數值的情況下是否為設定值
    """
    # 載入配置
    config = load_setting_regions_config()
    
    # 檢查是否需要進行設定值檢測
    regions_with_detection = config.get("regions_with_setting_detection", [])
    if region_name not in regions_with_detection:
        return config.get("default_setting_value", False)
    
    # 取得檢測配置
    detection_config = config.get("detection_config", {})
    sub_coords = detection_config.get("sub_region_coords", {})
    threshold = detection_config.get("white_pixel_threshold", 0.02)
    
    # 檢查子區域座標配置
    if not all(key in sub_coords for key in ["x1", "x2", "y1", "y2"]):
        return config.get("default_setting_value", False)
    
    # 取得子區域座標
    x1, x2 = sub_coords["x1"], sub_coords["x2"]
    y1, y2 = sub_coords["y1"], sub_coords["y2"]
    
    # 檢查座標是否在圖像範圍內
    h, w = binary_image.shape[:2]
    if x2 > w or y2 > h or x1 < 0 or y1 < 0:
        return config.get("default_setting_value", False)
    
    # 裁切子區域
    sub_region = binary_image[y1:y2, x1:x2]
    if sub_region.size == 0:
        return config.get("default_setting_value", False)
    
    # 計算白色像素比例
    white_pixels = np.sum(sub_region > 0)
    total_pixels = sub_region.size
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    
    # 判斷：白色比例超過閾值則為運作值（大數字），否則為設定值（小數字）
    is_operation_value = white_ratio > threshold
    
    return not is_operation_value  # 返回 True 表示設定值，False 表示運作值


def process_video(
    video_path: Path,
    rois: List[Tuple[str, Tuple[int, int, int, int]]],
    method: str = "rule",
    diff_thresh: float = 0.01,
    save_dir: Path | None = None,
    stage_activation_dict: Dict[str, List[int]] | None = None,
    roi_header_dict: Dict[str, List[int]] | None = None,
    char_sets_dict: Dict[str, str] | None = None,
    force: bool = False
) -> None:
    """
    Legacy CLI function: processes video file by file (Dual-Pass like).
    Kept for backward compatibility.
    """
    if not save_dir:
        save_dir = resolve_video_analysis_dir(video_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not force:
        all_exist = True
        for region_name, _ in rois:
            save_path = save_dir / f"{region_name}_ocr_testing.jsonl"
            if not save_path.exists():
                all_exist = False
                break
        if all_exist:
            print(f"⏩ 已存在所有 OCR 結果且未指定 --force，跳過: {video_path.name}")
            return

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

    # 以 ROI 提供者 API 迭代，統一取得 rois_data
    frame_provider = iterate_roi_binary_from_cache(video_path, rois, binarize_method=method)
    meta = get_video_meta(video_path)
    total_hint = meta.get('total_frames', 0) or None
    pbar = tqdm(total=total_hint, desc=f"分析 {video_path.name}", unit="frame")
    total_frames = 0

    # --- 單一循環處理所有幀和所有ROI ---
    for frame_idx, rois_data in frame_provider:
        # rois_data 已由 iterate_roi_binary_from_cache 提供

        for region_name, roi_data in rois_data.items():
            curr_bin = roi_data["binary_np"]
            prev_bin = prev_bins.get(region_name)
            
            # 檢查當前幀的activate狀態
            current_active = _active_in_stage(stage_analysis_dict, stage_activation_dict, roi_header_dict, frame_idx, region_name, None)
            prev_active = prev_active_states.get(region_name)
            
            # 檢測 activate → deactivate 轉換
            if prev_active is True and current_active is False:
                single_digit_results[region_name].append({
                    "type": "deactivate",
                    "frame": frame_idx,
                    "ocr_text": "(Deactivated)",
                    "confidence": 1.0,
                    "setting": False  # deactivate事件不是設定值
                })
                stats[region_name]["change_count"] += 1
            
            prev_bins[region_name] = curr_bin
            prev_active_states[region_name] = current_active
            
            if not current_active:
                continue

            change = False
            if prev_bin is None:
                change = True
            else:
                diff = calculate_average_binary_diff(prev_bin, curr_bin)
                change = diff >= diff_thresh

            if change:
                stats[region_name]["change_count"] += 1
                t1 = time.perf_counter()
                
                from_cache = False
                ocr_calls_in_frame = 0

                # --- Step 1: 判斷是否為多數字組合並嘗試從快取查找 ---
                is_multi_digit = not is_single_digit(curr_bin)
                
                if is_multi_digit:
                    for cached_bin, group_index in cache_map[region_name]:
                        if calculate_average_binary_diff(curr_bin, cached_bin) < 0.01:
                            multi_digit_groups[region_name][group_index]["matched_frames"].append(frame_idx)
                            from_cache = True
                            break

                # --- Step 2: 如果不在快取中，則執行OCR ---
                if not from_cache:
                    custom_allowlist = char_sets_dict.get(region_name)
                    ocr_result, confidence = ocr_iface.recognize(Image.fromarray(curr_bin), allowlist=custom_allowlist)
                    ocr_calls_in_frame += 1

                    if not ocr_result:
                        trimmed_bin = trim_black_borders(curr_bin, max_border=1)
                        if trimmed_bin.size > 0 and trimmed_bin.shape != curr_bin.shape:
                            ocr_result, confidence = ocr_iface.recognize(Image.fromarray(trimmed_bin), allowlist=custom_allowlist)
                            ocr_calls_in_frame += 1
                    
                    # --- Step 3: 記錄新結果 ---
                    if is_multi_digit:
                        if ocr_result:
                            new_group = {
                                "type": "multi_digit_group",
                                "source_frame": frame_idx,
                                "ocr_text": ocr_result,
                                "confidence": confidence,
                                "matched_frames": [frame_idx],
                                "setting": True
                            }
                            multi_digit_groups[region_name].append(new_group)
                            new_group_index = len(multi_digit_groups[region_name]) - 1
                            cache_map[region_name].append((curr_bin, new_group_index))
                    else: # 單數字
                        is_setting_value = _determine_if_setting_value(ocr_result, curr_bin, region_name, frame_idx)
                        single_digit_results[region_name].append({
                            "type": "single_digit",
                            "frame": frame_idx,
                            "ocr_text": ocr_result,
                            "confidence": confidence,
                            "setting": is_setting_value
                        })

                ocr_elapsed = time.perf_counter() - t1
                stats[region_name]["ocr_time_total"] += ocr_elapsed
                stats[region_name]["ocr_count"] += ocr_calls_in_frame

            prev_bins[region_name] = curr_bin

        total_ocr_count = sum(s["ocr_count"] for s in stats.values())
        pbar.set_postfix({"Total OCRs": total_ocr_count})
        pbar.update(1)
        total_frames += 1

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
        print(f"  變動幀數       : {change_count} / {total_frames} ({change_count/total_frames:.2f})")
        print(f"  實際OCR次數    : {ocr_count}")
        print(f"  平均變動處理耗時: {avg_change_processing_ms:.2f} ms/frame")
        print(f"  結果保存至     : {save_path}")

    print("="* (42 + len(video_path.name)))


# ---------------------------------------------------------------------------
# Testing API for UI integration
# ---------------------------------------------------------------------------
# ... (Keep test_current_frame_change as is) ...
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
        diff_ratio = calculate_average_binary_diff(prev_binary, curr_binary)
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
    p.add_argument("--force", action="store_true", help="強制重新分析，忽略已存在的結果檔案")
    return p.parse_args()


def main():  # pragma: no cover
    args = parse_args()
    
    # 加载全局配置（不依赖视频）
    stage_activation_dict = load_stage_config(args.stage_config)
    char_sets_dict = load_ocr_char_sets_config(args.char_config)
    if char_sets_dict:
        print(f"載入 custom OCR character sets 成功: {list(char_sets_dict.keys())}")

    # 处理视频 (CLI logic kept simplified for brevity as it duplicates logic in old script)
    # ... (Implementation logic remains similar to previous version for CLI) ...
    if args.video.is_dir():
        video_files = list(args.video.glob("*.mp4"))
        for video_file in video_files:
            print(f"\n{'='*60}")
            print(f"Processing video: {video_file.name}")
            print(f"{'='*60}")
            
            video_name = video_file.stem
            roi_dict = load_roi_config(path=args.roi_config, video_name=video_name)
            
            roi_header_dict = {}
            try:
                roi_header_dict = load_roi_header_config(path=args.roi_config, video_name=video_name)
            except Exception:
                pass
            
            rois = []
            if args.region == "all":
                for region in roi_dict:
                    rois.append((region, tuple(roi_dict[region])))
            else:
                if args.region in roi_dict:
                    rois = [(args.region, tuple(roi_dict[args.region]))]
            
            process_video(
                video_file, rois, args.method, args.diff_thresh, args.save_dir, 
                stage_activation_dict, roi_header_dict, char_sets_dict, args.force
            )

    elif str(args.video)[-4:].lower() == ".mp4":
        video_name = args.video.stem
        roi_dict = load_roi_config(path=args.roi_config, video_name=video_name)
        roi_header_dict = {}
        try:
            roi_header_dict = load_roi_header_config(path=args.roi_config, video_name=video_name)
        except Exception:
            pass
        
        rois = []
        if args.region == "all":
            for region in roi_dict:
                rois.append((region, tuple(roi_dict[region])))
        else:
            if args.region in roi_dict:
                rois = [(args.region, tuple(roi_dict[args.region]))]
        
        process_video(
            args.video, rois, args.method, args.diff_thresh, args.save_dir, 
            stage_activation_dict, roi_header_dict, char_sets_dict, args.force
        )

if __name__ == "__main__":  # pragma: no cover
    main()
