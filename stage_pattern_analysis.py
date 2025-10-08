#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.get_configs import load_diff_rules, load_pattern_name_mapping


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class RegionPattern:
    pattern_id: int
    array: np.ndarray  # reference pattern image (RGB for PEDAL, binary for others)


@dataclass
class Segment:
    pattern_id: int
    start_frame: int
    end_frame: int
    avg_rmse: float


# -----------------------------
# Utilities
# -----------------------------

def debug_pause(args: argparse.Namespace, message: str):
    """如果啟用互動模式，則暫停程式"""
    if args.interactive:
        input(f"    └── ⏸️  {message} (按 Enter 繼續)...")

def read_surgery_stage_rois(path: Path) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def binarize_otsu(image_rgb: np.ndarray) -> np.ndarray:
    """Return a 2D uint8 binary image (0/255) using OTSU on gray."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def calculate_rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        # 在偵錯模式下提供更詳細的形狀不匹配資訊
        print(f"    [DEBUG] 🔴 錯誤: 形狀不匹配! 圖像A: {a.shape}, 圖像B: {b.shape}")
        return float("inf")
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    mse = np.mean((a_f - b_f) ** 2)
    return float(np.sqrt(mse))


def calculate_pedal_frame_diff(prev_img: np.ndarray, curr_img: np.ndarray, sub_roi_coords: List[int]) -> float:
    """計算兩張 PEDAL ROI 圖像在指定精細區域內的平均RGB顏色差異 (與 UI 中的邏輯相同)"""
    try:
        x1, y1, x2, y2 = sub_roi_coords
        
        # 從兩張圖像中裁剪出精細區域
        prev_sub_roi = prev_img[y1:y2, x1:x2]
        curr_sub_roi = curr_img[y1:y2, x1:x2]
        
        # 檢查尺寸是否一致
        if prev_sub_roi.shape != curr_sub_roi.shape:
            return 0.0
        
        # 轉換為 float32
        prev_arr = prev_sub_roi.astype(np.float32)
        curr_arr = curr_sub_roi.astype(np.float32)
        
        # 計算每個像素RGB通道差值的平方
        squared_diff = np.square(prev_arr - curr_arr)
        
        # 計算每個像素的均方差 (MSE)
        mse_per_pixel = np.mean(squared_diff, axis=2)
        
        # 計算每個像素的均方根差 (RMSE)，即顏色距離
        rmse_per_pixel = np.sqrt(mse_per_pixel)
        average_rmse = float(np.mean(rmse_per_pixel))
        
        return average_rmse
        
    except Exception as e:
        print(f"計算 PEDAL 前後幀差異時出錯: {e}")
        return 0.0


def load_region_patterns(cache_dir: Path, region_name: str) -> List[RegionPattern]:
    region_dir = cache_dir / region_name
    if not region_dir.exists():
        return []
    patterns: List[RegionPattern] = []
    for npy_path in sorted(region_dir.glob("*.npy")):
        try:
            arr = np.load(npy_path)
            if not isinstance(arr, np.ndarray):
                continue
            try:
                pid = int(npy_path.stem)
            except ValueError:
                continue
            patterns.append(RegionPattern(pattern_id=pid, array=arr))
        except Exception:
            continue
    return patterns


def get_analysis_candidate(roi_rgb: np.ndarray, region_name: str, region_config: Dict, args: argparse.Namespace, frame_idx: int) -> np.ndarray:
    """根據區域配置，從原始ROI中準備用於分析的圖像陣列"""
    analysis_mode = region_config.get("analysis_mode", "full_roi")
    is_pedal_debug = args.debug_pedal and region_name == "PEDAL"

    if is_pedal_debug:
        print(f"\n[PEDAL DEBUG] Frame {frame_idx} | 步驟 2: 準備分析圖像")
        print(f"    - 分析模式: {analysis_mode}")

    if region_name == "PEDAL" and analysis_mode == "sub_roi":
        sub_coords = region_config.get("sub_roi_coords", [20, 13, 26, 19])
        x1, y1, x2, y2 = sub_coords
        h, w = roi_rgb.shape[:2]
        if x2 > w or y2 > h or x1 < 0 or y1 < 0:
            print(f"    [PEDAL DEBUG] 🔴 警告: 精細區域座標 {sub_coords} 超出ROI範圍 {(w, h)}")
            return roi_rgb
        
        cand_array = roi_rgb[y1:y2, x1:x2]
        if is_pedal_debug:
            print(f"    - 裁切後的候選圖像尺寸: {cand_array.shape}")
        return cand_array
    elif region_name == "PEDAL":
        return roi_rgb
    else:
        return binarize_otsu(roi_rgb)


def match_best_pattern(
    cand_array: np.ndarray,
    patterns: List[RegionPattern],
    rmse_threshold: float,
    args: argparse.Namespace,
    frame_idx: int,
    region_name: str,
    region_config: Dict[str, Any]
) -> Tuple[Optional[int], Optional[float]]:
    """從候選圖像陣列中匹配最佳樣板
    比對規則：
    1. 遍歷所有已知的樣板，計算候選圖像與每個樣板之間的 RMSE 值。
    2. 在所有計算出的 RMSE 值中，找出最小的那一個。
    3. 只有當這個最小的 RMSE 值同時也小於指定的 `rmse_threshold` 時，
       才將其視為一個確定的、唯一的匹配。
    4. 否則，即使找到了最接近的樣板，也因相似度不足而視為不匹配。
    """
    is_pedal_debug = args.debug_pedal and region_name == "PEDAL"
    analysis_mode = region_config.get("analysis_mode", "full_roi")

    if is_pedal_debug:
        print(f"[PEDAL DEBUG] Frame {frame_idx} | 步驟 3: 執行樣板比對")
        print(f"    - 候選圖像尺寸: {cand_array.shape}")
        print(f"    - 使用的門檻值: {rmse_threshold}")
        debug_pause(args, "即將開始逐一樣板比對")

    if not patterns:
        if is_pedal_debug:
            print("    [PEDAL DEBUG] 🔴 錯誤: 沒有載入任何樣板，無法比對。")
        return None, None
    
    best_pid: Optional[int] = None
    best_rmse: float = float("inf")

    for p in patterns:
        full_ref_array = p.array
        ref_for_comparison = full_ref_array

        if region_name == "PEDAL" and analysis_mode == "sub_roi":
            sub_coords = region_config.get("sub_roi_coords")
            x1, y1, x2, y2 = sub_coords
            if full_ref_array.shape[1] > x2 and full_ref_array.shape[0] > y2:
                ref_for_comparison = full_ref_array[y1:y2, x1:x2]
            else:
                if is_pedal_debug:
                    print(f"    [PEDAL DEBUG] 🔴 警告: 樣板 ID {p.pattern_id} (尺寸 {full_ref_array.shape}) 太小，無法使用 sub_roi 座標 {sub_coords} 進行裁切。")
                continue

        rmse = calculate_rmse(ref_for_comparison, cand_array)
        
        if is_pedal_debug:
            print(f"    - 正在比對樣板 ID: {p.pattern_id} (比對尺寸: {ref_for_comparison.shape}) -> 計算出的 RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_pid = p.pattern_id

    is_match = best_pid is not None and best_rmse < rmse_threshold
    
    if is_pedal_debug:
        print(f"    - 比對完成。最佳匹配樣板: {best_pid}, 最小 RMSE: {best_rmse:.4f}")
        print(f"    - 判斷結果: {best_rmse:.4f} < {rmse_threshold} ?  -> {'✅ 匹配成功' if is_match else '❌ 匹配失敗'}")
        debug_pause(args, "樣板比對結束")

    if is_match:
        return best_pid, best_rmse
    return None, None


def build_segments(
    matches: List[Tuple[int, Optional[int], Optional[float]]],
    pattern_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    從幀匹配列表中構建連續的片段。
    """
    segments: List[Dict[str, Any]] = []
    current_pattern: Optional[int] = None
    start: Optional[int] = None
    rmse_values: List[float] = []

    if current_pattern is None:
        return []

    def close_segment(end_frame: int):
        nonlocal segments, start, current_pattern, rmse_values
        if current_pattern is not None:
            avg_rmse = np.mean(rmse_values) if rmse_values else 0.0
            pattern_name = pattern_mapping.get(str(current_pattern), f"Pattern {current_pattern}")
            segments.append({
                "pattern": current_pattern,
                "pattern_name": pattern_name,
                "start_frame": start,
                "end_frame": end_frame,
                "avg_rmse": float(avg_rmse),
                "frame_count": end_frame - start + 1
            })
            rmse_values = []

    for frame_idx, pattern_id, rmse in matches:
        if pattern_id != current_pattern:
            close_segment(frame_idx - 1)
            current_pattern = pattern_id
            start = frame_idx
            if rmse is not None:
                rmse_values.append(rmse)
        else:
            if rmse is not None:
                rmse_values.append(rmse)

    if current_pattern is not None and start is not None:
        last_frame = matches[-1][0] if matches else 0
        close_segment(last_frame)

    return segments


def analyze_pedal_frame(
    current_roi: np.ndarray,
    prev_roi: Optional[np.ndarray],
    patterns: List[RegionPattern],
    region_config: Dict[str, Any],
    args: argparse.Namespace,
    frame_idx: int
) -> Tuple[Optional[int], Optional[float]]:
    """
    PEDAL 區域的優化分析函數：
    1. 第一幀：直接與 cache 比對
    2. 後續幀：先與前一幀比較差異，超過 frame_diff_threshold 才進行 cache 比對
    3. Cache 比對時使用 cache_hit_threshold 判斷是否匹配
    """
    is_pedal_debug = args.debug_pedal
    sub_coords = region_config.get("sub_roi_coords", [20, 13, 26, 19])
    
    # 從配置中獲取兩個不同的閾值
    frame_diff_threshold = region_config.get("diff_threshold", 30.0)  # 前後幀差異門檻
    cache_hit_threshold = region_config.get("cache_hit_threshold", 40.0)  # cache 匹配門檻
    
    # 記錄上一幀的匹配結果（簡單起見，使用全域變數或類屬性，這裡用靜態變數模擬）
    if not hasattr(analyze_pedal_frame, 'prev_match_result'):
        analyze_pedal_frame.prev_match_result = (None, None)
    
    if is_pedal_debug:
        print("="*60)
        print(f"[PEDAL DEBUG] Frame {frame_idx} | PEDAL 區域分析開始")
        print(f"    - 當前 ROI 尺寸: {current_roi.shape}")
        print(f"    - 前後幀差異門檻: {frame_diff_threshold}")
        print(f"    - Cache 匹配門檻: {cache_hit_threshold}")
    
    # 第一幀：直接與 cache 比對
    if prev_roi is None or frame_idx == 0:
        if is_pedal_debug:
            print(f"    - 策略: 第一幀，直接進行 cache 比對")
        
        cand_array = get_analysis_candidate(current_roi, "PEDAL", region_config, args, frame_idx)
        pid, rmse = match_best_pattern(
            cand_array, patterns, cache_hit_threshold, args, frame_idx,
            region_name="PEDAL", region_config=region_config
        )
        analyze_pedal_frame.prev_match_result = (pid, rmse)
        return pid, rmse
    
    # 後續幀：先計算前後幀差異
    frame_diff = calculate_pedal_frame_diff(prev_roi, current_roi, sub_coords)
    
    if is_pedal_debug:
        print(f"    - 與前一幀的差異值: {frame_diff:.2f}")
        print(f"    - 前後幀差異門檻: {frame_diff_threshold}")
        
    if frame_diff <= frame_diff_threshold:
        # 差異不大，沿用前一幀的結果
        prev_pid, prev_rmse = analyze_pedal_frame.prev_match_result
        if is_pedal_debug:
            print(f"    - 策略: 差異 ≤ {frame_diff_threshold}，沿用前一幀結果 (Pattern ID: {prev_pid})")
        return prev_pid, prev_rmse
    else:
        # 差異較大，進行 cache 比對
        if is_pedal_debug:
            print(f"    - 策略: 差異 > {frame_diff_threshold}，進行 cache 比對")
            print(f"    - 將使用 cache 匹配門檻: {cache_hit_threshold}")
            debug_pause(args, "即將開始 cache 比對")
        
        cand_array = get_analysis_candidate(current_roi, "PEDAL", region_config, args, frame_idx)
        pid, rmse = match_best_pattern(
            cand_array, patterns, cache_hit_threshold, args, frame_idx,
            region_name="PEDAL", region_config=region_config
        )
        
        # 更新記錄的結果
        analyze_pedal_frame.prev_match_result = (pid, rmse)
        
        if is_pedal_debug:
            print(f"    - Cache 比對結果: Pattern ID {pid}, RMSE: {rmse}")
            if pid is not None:
                print(f"    - ✅ 找到匹配的 Pattern ID {pid} (RMSE {rmse:.2f} < {cache_hit_threshold})")
            else:
                print(f"    - ❌ 沒有找到匹配的 Pattern (最小 RMSE {rmse:.2f} >= {cache_hit_threshold})")
        
        return pid, rmse


def analyse_video(
    video_path: Path,
    *,
    roi_config_path: Path,
    cache_dir: Path,
    rmse_threshold: Optional[float] = None,
    output_dir: Optional[Path] = None,
    args: argparse.Namespace,
) -> Path:
    """對單一影片進行分析"""
    
    diff_rules = load_diff_rules()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    roi_dict = read_surgery_stage_rois(roi_config_path)
    region_to_patterns: Dict[str, List[RegionPattern]] = {
        region: load_region_patterns(cache_dir, region) for region in roi_dict.keys()
    }

    region_matches: Dict[str, List[Tuple[int, Optional[int], Optional[float]]]] = {r: [] for r in roi_dict}
    
    # 儲存前一幀的 ROI 圖像（用於 PEDAL 前後幀比較）
    prev_frame_rois: Dict[str, Optional[np.ndarray]] = {r: None for r in roi_dict}
    
    print(f"🚀 開始使用「優化的 PEDAL 前後幀比較法」分析影片: {video_path.name}")
    if args.debug_pedal:
        print("🕵️  已啟用 PEDAL 偵錯模式。")
        print("📋 PEDAL 分析策略：")
        print("    1. 第一幀：直接與 cache 比對")
        print("    2. 後續幀：先與前一幀比較差異")
        print("    3. 差異 > 30：進行 cache 比對")
        print("    4. 差異 ≤ 30：沿用前一幀的結果")

    frames_to_process = total_frames
    if args.debug_pedal:
        frames_to_process = 50  # 增加到 50 幀以便觀察變化
        print(f"⚠️  偵錯模式下，僅處理前 {frames_to_process} 幀以加速除錯。")

    for frame_idx in range(frames_to_process):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if frame_idx > 0 and frame_idx % 1000 == 0:
            print(f"  ... 正在處理幀 {frame_idx}/{total_frames}")

        for region_name, coords in roi_dict.items():
            is_pedal_debug = args.debug_pedal and region_name == "PEDAL"
            region_config = diff_rules.get(region_name, {})
            threshold = rmse_threshold if rmse_threshold is not None else region_config.get("diff_threshold", 30.0)

            x1, y1, x2, y2 = map(int, coords)
            roi_rgb = frame_rgb[y1:y2, x1:x2]
            
            if region_name == "PEDAL":
                # PEDAL 區域使用新的前後幀比較策略
                pid, rmse = analyze_pedal_frame(
                    roi_rgb, prev_frame_rois[region_name], 
                    region_to_patterns.get(region_name, []),
                    region_config, args, frame_idx
                )
                # 更新前一幀的 ROI
                prev_frame_rois[region_name] = roi_rgb.copy()
            else:
                # 其他區域維持原有邏輯
                if is_pedal_debug:
                    print("="*50)
                    print(f"[PEDAL DEBUG] Frame {frame_idx} | 步驟 1: 裁切原始 ROI")
                    print(f"    - ROI 尺寸: {roi_rgb.shape}")

                cand_array = get_analysis_candidate(roi_rgb, region_name, region_config, args, frame_idx)
                
                patterns = region_to_patterns.get(region_name, [])
                pid, rmse = match_best_pattern(
                    cand_array, patterns, threshold, args, frame_idx,
                    region_name=region_name, region_config=region_config
                )
            
            region_matches[region_name].append((frame_idx, pid, rmse))

    cap.release()
    print("✅ 逐幀分析完成，開始建立區段...")

    regions_output: Dict[str, List[Dict[str, float]]] = {}
    for region_name, matches in region_matches.items():
        # 使用新的 build_segments 函數
        region_pattern_map = load_pattern_name_mapping(Path("config/pattern_name_mapping.json")).get(region_name, {})
        segments = build_segments(matches, region_pattern_map)
        regions_output[region_name] = segments

    video_name = video_path.stem
    if output_dir is None:
        output_dir = Path("data") / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "stage_analysis.json"

    payload = {
        "video": video_name,
        "total_frames": total_frames,
        "regions": regions_output,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ 分析完成，已輸出: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare ROI of each frame against cached patterns and produce stage analysis JSON")
    p.add_argument("--video", "--video-path", dest="video", required=True, type=Path, help="影片路徑或資料夾 (.mp4 檔或包含多支影片的資料夾)")
    p.add_argument("--roi-config", type=Path, default=Path("config/surgery_stage_rois.json"), help="手術階段ROI配置檔")
    p.add_argument("--cache-dir", type=Path, default=Path("data/roi_img_caches"), help="各區域快取圖片資料夾根目錄")
    p.add_argument("--threshold", type=float, default=None, help="全域RMSE門檻值 (若指定，會覆蓋diff_rule.json中的設定)")
    p.add_argument("--output-dir", type=Path, help="輸出資料夾 (預設: data/<video_name>)；若為資料夾模式將分別輸出至各自的 data/<video_name>/")
    p.add_argument("--debug-pedal", action="store_true", help="啟用針對 PEDAL 區域的詳細偵錯模式")
    p.add_argument("--interactive", action="store_true", help="在偵錯模式下啟用互動式暫停 (需搭配 --debug-pedal)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = args.video

    if target.is_dir():
        video_files = sorted(target.glob("*.mp4"))
        if not video_files:
            print(f"⚠️  資料夾中未找到 mp4 檔案: {target}")
            return
        print(f"🔎 在資料夾中找到 {len(video_files)} 支影片，開始逐一分析...")
        for idx, vf in enumerate(video_files, start=1):
            print(f"\n[{idx}/{len(video_files)}] 分析影片: {vf.name}")
            analyse_video(
                vf,
                roi_config_path=args.roi_config,
                cache_dir=args.cache_dir,
                rmse_threshold=args.threshold,
                output_dir=None,
                args=args,
            )
        print("\n✅  所有影片分析完成")
        return

    if str(target).lower().endswith(".mp4"):
        analyse_video(
            target,
            roi_config_path=args.roi_config,
            cache_dir=args.cache_dir,
            rmse_threshold=args.threshold,
            output_dir=args.output_dir,
            args=args,
        )
    else:
        print(f"❌  請指定 mp4 檔案或包含 mp4 的資料夾: {target}")


if __name__ == "__main__":
    main()