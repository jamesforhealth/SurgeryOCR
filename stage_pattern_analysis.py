#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from extract_frame_cache import iterate_frames, get_frame_cache_dir
from utils.get_configs import load_diff_rules, load_pattern_name_mapping, read_surgery_stage_rois
from utils.get_paths import resolve_video_analysis_dir
from utils.cv_processing import calculate_rmse, calculate_ndarray_diff

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

def binarize_otsu(image_rgb: np.ndarray) -> np.ndarray:
    """Return a 2D uint8 binary image (0/255) using OTSU on gray."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def _load_region_patterns(cache_dir: Path, region_name: str) -> List[RegionPattern]:
    region_dir = cache_dir / region_name
    if not region_dir.exists():
        return []
    patterns: List[RegionPattern] = []
    for npy_path in sorted(region_dir.glob("*.npy")):
        try:
            arr = np.load(npy_path)
            if not isinstance(arr, np.ndarray):
                continue
            arr = arr.astype(np.float32, copy=False)
            try:
                pid = int(npy_path.stem)
            except ValueError:
                continue
            patterns.append(RegionPattern(pattern_id=pid, array=arr))
        except Exception:
            continue
    return patterns


def _get_analysis_candidate(roi_input: np.ndarray, region_name: str, region_config: Dict, args: argparse.Namespace, frame_idx: int, input_is_bgr: bool = False) -> np.ndarray:
    """根據區域配置，從原始ROI中準備用於分析的圖像陣列"""
    analysis_mode = region_config.get("analysis_mode", "full_roi")
    is_pedal_debug = args.debug_pedal and region_name == "PEDAL"

    if is_pedal_debug:
        print(f"\n[PEDAL DEBUG] Frame {frame_idx} | 步驟 2: 準備分析圖像")
        print(f"    - 分析模式: {analysis_mode}")

    if region_name == "PEDAL":
        # PEDAL 需要 RGB 格式
        if input_is_bgr:
            roi_rgb = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)
        else:
            roi_rgb = roi_input

        if analysis_mode == "sub_roi":
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
        else:
            return roi_rgb
    else:
        # 其他區域使用 OTSU 二值化 (從 Gray)
        if input_is_bgr:
            gray = cv2.cvtColor(roi_input, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(roi_input, cv2.COLOR_RGB2GRAY)
            
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary


def _match_best_pattern(
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
    
    cand_array = cand_array.astype(np.float32, copy=False)
    best_pid: Optional[int] = None
    best_rmse: float = float("inf")

    # 優化：將循環不變量移出循環
    use_sub_roi = (region_name == "PEDAL" and analysis_mode == "sub_roi")
    sub_coords = region_config.get("sub_roi_coords") if use_sub_roi else None
    x1, y1, x2, y2 = (0, 0, 0, 0)
    if sub_coords:
         x1, y1, x2, y2 = sub_coords

    for p in patterns:
        full_ref_array = p.array
        ref_for_comparison = full_ref_array

        if use_sub_roi:
            if full_ref_array.shape[1] > x2 and full_ref_array.shape[0] > y2:
                ref_for_comparison = full_ref_array[y1:y2, x1:x2]
            else:
                if is_pedal_debug:
                    print(f"    [PEDAL DEBUG] 🔴 警告: 樣板 ID {p.pattern_id} (尺寸 {full_ref_array.shape}) 太小，無法使用 sub_roi 座標 {sub_coords} 進行裁切。")
                continue

        # 優化：內聯 RMSE 計算，減少函數調用開銷
        if ref_for_comparison.shape != cand_array.shape:
            if is_pedal_debug:
                 print(f"    [DEBUG] 🔴 錯誤: 形狀不匹配! Pattern: {ref_for_comparison.shape}, Cand: {cand_array.shape}")
            rmse = float("inf")
        else:
            rmse = calculate_rmse(ref_for_comparison, cand_array)

        if is_pedal_debug:
            print(f"    - 正在比對樣板 ID: {p.pattern_id} (比對尺寸: {ref_for_comparison.shape}) -> 計算出的 RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_pid = p.pattern_id
            # 優化：如果已經完全匹配，就不需要再找了
            if best_rmse == 0.0:
                break

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
        # None 視為空白區段，會中斷任何正在進行的片段
        if pattern_id is None:
            close_segment(frame_idx - 1)
            current_pattern = None
            start = None
            rmse_values = []
            continue

        if current_pattern is None:
            current_pattern = pattern_id
            start = frame_idx
            if rmse is not None:
                rmse_values.append(rmse)
            continue

        if pattern_id != current_pattern:
            close_segment(frame_idx - 1)
            current_pattern = pattern_id
            start = frame_idx
            rmse_values = []
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
        
        cand_array = _get_analysis_candidate(current_roi, "PEDAL", region_config, args, frame_idx)
        pid, rmse = _match_best_pattern(
            cand_array, patterns, cache_hit_threshold, args, frame_idx,
            region_name="PEDAL", region_config=region_config
        )
        analyze_pedal_frame.prev_match_result = (pid, rmse)
        return pid, rmse
    
    # 後續幀：先計算前後幀差異
    frame_diff = calculate_ndarray_diff(prev_roi, current_roi, sub_coords)
    
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
        
        cand_array = _get_analysis_candidate(current_roi, "PEDAL", region_config, args, frame_idx)
        pid, rmse = _match_best_pattern(
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


# --------------------------------------------------------------------------
# [NEW] StageAnalyzer Class for Pipeline Integration
# --------------------------------------------------------------------------
class StageAnalyzer:
    def __init__(self, roi_config_path: Path, cache_dir: Path, debug: bool = False):
        self.roi_config_path = roi_config_path
        self.cache_dir = cache_dir
        self.debug = debug
        
        # 載入配置
        self.roi_dict = read_surgery_stage_rois(roi_config_path)
        self.diff_rules = load_diff_rules()
        
        # 載入 Patterns
        self.region_to_patterns: Dict[str, List[RegionPattern]] = {
            region: _load_region_patterns(cache_dir, region) for region in self.roi_dict.keys()
        }
        
        # 狀態變數
        self.prev_frame_rois: Dict[str, Optional[np.ndarray]] = {r: None for r in self.roi_dict}
        
        # 專門為 PEDAL 優化準備的狀態
        # 我們需要維護 analyze_pedal_frame 的內部狀態 (prev_match_result)
        # 為了避免多個實例互相干擾，我們將狀態存在 instance 中
        self.pedal_prev_match = (None, None) # (pid, rmse)
        
        # [優化] 預先創建 mock_args 對象，避免每幀都創建新對象
        self._mock_args = argparse.Namespace(debug_pedal=debug, interactive=False)
        
        # [優化] 預先計算每個區域的配置和閾值
        self._region_configs: Dict[str, Dict] = {}
        self._region_thresholds: Dict[str, float] = {}
        for region_name in self.roi_dict.keys():
            config = self.diff_rules.get(region_name, {})
            self._region_configs[region_name] = config
            self._region_thresholds[region_name] = config.get("diff_threshold", 30.0)
    
    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int, rmse_threshold: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """
        處理單一幀，回傳該幀各區域的匹配結果。
        Returns:
            {
                "STAGE": {"pattern_id": 1, "rmse": 12.5},
                "PEDAL": {"pattern_id": None, "rmse": None},
                ...
            }
        """
        results = {}

        for region_name, coords in self.roi_dict.items():
            # [優化] 使用預先緩存的配置和閾值
            region_config = self._region_configs[region_name]
            threshold = rmse_threshold if rmse_threshold is not None else self._region_thresholds[region_name]

            x1, y1, x2, y2 = map(int, coords)
            h, w = frame_bgr.shape[:2]
            
            # 邊界檢查
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                if self.debug:
                    print(f"[警告] ROI 座標無效 {region_name}: {(x1,y1,x2,y2)} 超出畫面範圍 {(w,h)}，跳過")
                results[region_name] = {"pattern_id": None, "rmse": None}
                continue
                
            roi_bgr = frame_bgr[y1:y2, x1:x2]
            
            pid, rmse = None, None
            
            if region_name == "PEDAL":
                # PEDAL 需要 RGB 進行分析
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                
                # 注入狀態
                analyze_pedal_frame.prev_match_result = self.pedal_prev_match
                
                pid, rmse = analyze_pedal_frame(
                    roi_rgb, self.prev_frame_rois[region_name],
                    self.region_to_patterns.get(region_name, []),
                    region_config, self._mock_args, frame_idx
                )
                
                # 保存狀態
                self.pedal_prev_match = analyze_pedal_frame.prev_match_result
                self.prev_frame_rois[region_name] = roi_rgb.copy() # 這裡還是存 RGB 以供下一幀比較
                
            else:
                # 非 PEDAL 區域，傳入 BGR 並在 _get_analysis_candidate 內部轉 Gray -> Binary，省去 RGB 轉換
                cand_array = _get_analysis_candidate(roi_bgr, region_name, region_config, self._mock_args, frame_idx, input_is_bgr=True)
                patterns = self.region_to_patterns.get(region_name, [])
                if patterns:
                    pid, rmse = _match_best_pattern(
                        cand_array, patterns, threshold, self._mock_args, frame_idx,
                        region_name=region_name, region_config=region_config
                    )
            
            results[region_name] = {"pattern_id": pid, "rmse": rmse}
            
        return results


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
    
    # 使用新的 StageAnalyzer Class
    analyzer = StageAnalyzer(roi_config_path, cache_dir, debug=args.debug_pedal)
    
    # 簡易取得影片資訊
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    region_matches: Dict[str, List[Tuple[int, Optional[int], Optional[float]]]] = {r: [] for r in analyzer.roi_dict}
    
    print(f"🚀 開始使用「優化的 PEDAL 前後幀比較法」分析影片: {video_path.name}")
    if args.debug_pedal:
        print("🕵️  已啟用 PEDAL 偵錯模式。")

    frames_to_process = total_frames
    if args.debug_pedal:
        frames_to_process = 50
        print(f"⚠️  偵錯模式下，僅處理前 {frames_to_process} 幀以加速除錯。")

    # 優先使用 frame_cache（若存在），否則回退到直接讀取影片
    cache_path = get_frame_cache_dir(video_path)
    cache_has_files = cache_path.exists() and any(cache_path.glob("frame_*.jpg"))
    
    if cache_has_files:
        print(f"[訊息] 使用 frame_cache 來源: {cache_path}")
        frame_source = iterate_frames(video_path)
    else:
        # 嘗試從 extract_frame_cache 導入 generator，如果失敗則使用本地簡單邏輯
        try:
            from extract_frame_cache import video_frame_generator
            frame_source = video_frame_generator(video_path)
        except ImportError:
            # Fallback: 簡單的 generator
            def simple_gen():
                c = cv2.VideoCapture(str(video_path))
                i = 0
                while True:
                    r, f = c.read()
                    if not r: break
                    if f is not None and f.size > 0:
                        yield i, f
                    i += 1
                c.release()
            frame_source = simple_gen()

    # 統一的處理迴圈
    for frame_idx, frame_bgr in frame_source:
        if args.debug_pedal and frame_idx >= frames_to_process:
            break
            
        if frame_idx > 0 and frame_idx % 1000 == 0:
            print(f"  ... 正在處理幀 {frame_idx}/{total_frames}")

        # 使用 Analyzer 處理
        results = analyzer.process_frame(frame_bgr, frame_idx, rmse_threshold)
        
        # 收集結果
        for region_name, res in results.items():
            region_matches[region_name].append((frame_idx, res['pattern_id'], res['rmse']))

    print("✅ 逐幀分析完成，開始建立區段...")

    regions_output: Dict[str, List[Dict[str, float]]] = {}
    for region_name, matches in region_matches.items():
        # 保險：按 frame_idx 排序，避免任何來源造成的錯序
        matches.sort(key=lambda t: t[0])
        # 使用新的 build_segments 函數
        region_pattern_map = load_pattern_name_mapping(Path("config/pattern_name_mapping.json")).get(region_name, {})
        segments = build_segments(matches, region_pattern_map)
        # 額外防護：修正可能的負 frame_count 與錯位 end/start
        cleaned_segments: List[Dict[str, Any]] = []
        for seg in segments:
            s = int(seg.get("start_frame", 0))
            e = int(seg.get("end_frame", s))
            if e < s:
                # 以 s 作為 fallback，避免負值
                e = s
            fc = max(0, e - s + 1)
            seg["start_frame"] = s
            seg["end_frame"] = e
            seg["frame_count"] = fc
            cleaned_segments.append(seg)
        segments = cleaned_segments
        regions_output[region_name] = segments
        non_null = sum(1 for _, pid, _ in matches if pid is not None)
        print(f"[診斷] 區域 {region_name}: 匹配幀數 {non_null} / {len(matches)}，段落數 {len(segments)}")

    video_name = video_path.stem
    if output_dir is None:
        # 使用統一的路徑解析邏輯，支持子目錄結構
        output_dir = resolve_video_analysis_dir(video_path)
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
    p.add_argument("--force", action="store_true", help="強制重新分析，忽略已存在的 stage_analysis.json")
    p.add_argument("--use-stream", action="store_true", help="強制使用串流讀取模式 (不依賴 disk cache)")
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
            # 計算預期輸出路徑，若存在且未指定 --force 則跳過
            expected_out_dir = args.output_dir if args.output_dir else resolve_video_analysis_dir(vf)
            expected_out_path = expected_out_dir / "stage_analysis.json"
            if expected_out_path.exists() and not args.force:
                print(f"⏭️  偵測到已存在: {expected_out_path}，使用 --force 可強制重跑，已跳過。")
                continue

            analyse_video(
                vf,
                roi_config_path=args.roi_config,
                cache_dir=args.cache_dir,
                rmse_threshold=args.threshold,
                output_dir=args.output_dir,  # 目錄模式也尊重傳入的 output_dir
                args=args,
            )
        print("\n✅  所有影片分析完成")
        return

    if str(target).lower().endswith(".mp4"):
        # 計算預期輸出路徑，若存在且未指定 --force 則提示並跳過
        expected_out_dir = args.output_dir if args.output_dir else resolve_video_analysis_dir(target)
        expected_out_path = expected_out_dir / "stage_analysis.json"
        if expected_out_path.exists() and not args.force:
            print(f"⏭️  偵測到已存在: {expected_out_path}，使用 --force 可強制重跑，已跳過。")
            return

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
