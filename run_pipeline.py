import argparse
from pathlib import Path
import time
import cv2
import json
import traceback
from typing import List, Tuple, Any, Dict, Optional

# 引入工具
from utils.pipeline_utils import AsyncImageSaver
from utils.get_configs import (
    load_roi_config,
    load_stage_config,
    load_roi_header_config,
    load_ocr_char_sets_config,
    load_pattern_name_mapping,
    update_video_machine_mapping,
)
from utils.cv_processing import binarize

# 引入各階段模組
from extract_frame_cache import video_frame_generator
from stage_pattern_analysis import StageAnalyzer, build_segments
from auto_detect_machine_type import MachineDetector
from surgery_analysis_process import OCRProcessor
from tqdm import tqdm
import numpy as np

def run_pipeline(video_path: Path, base_output_dir: Path, force: bool = False):
    """
    單次讀取 (Single-Pass) 分析管線：
    1. 讀取影片
    2. (非同步) 儲存 Frame Cache
    3. (同步) 執行 Stage Pattern Analysis
    4. (緩衝/觸發) Machine Type Detection
    5. (即時/回溯) OCR & Change Detection
    """
    video_name = video_path.stem
    
    # 建立輸出目錄結構： base_output_dir / video_name / ...
    # 若使用者指定 base_output_dir (e.g. "data")，則輸出為 "data/video_name"
    
    if base_output_dir.name == video_name:
        # 使用者可能已經指定了完整路徑
        analysis_dir = base_output_dir
    else:
        analysis_dir = base_output_dir / video_name
        
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame_cache_dir = analysis_dir / "frame_cache"
    frame_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 開始串流分析: {video_name}")
    print(f"📂 輸出目錄: {analysis_dir}")

    # --- 1. 初始化各個組件 ---
    async_saver = AsyncImageSaver()
    
    # Configs
    stage_config_path = Path("config/surgery_stage_rois.json")
    stage_activation_path = Path("config/ocr_activation_stages.json")
    roi_config_path = Path("config/rois.json")
    char_config_path = Path("config/ocr_char_sets.json")
    cache_root = Path("data/roi_img_caches")
    
    if not stage_config_path.exists():
        print("❌ 錯誤: 找不到 config/surgery_stage_rois.json")
        return

    # Analyzers
    stage_analyzer = StageAnalyzer(stage_config_path, cache_root)
    region_matches: Dict[str, List[Tuple[int, Any, Any]]] = {
        region: [] for region in stage_analyzer.roi_dict.keys()
    }
    machine_detector = MachineDetector() # 預設讀取 region1.png
    
    # OCR Processor (初始時還不知道機型，ROI Config 稍後載入)
    # 但可以先載入與機型無關的設定
    stage_activation_dict = load_stage_config(stage_activation_path)
    char_sets_dict = load_ocr_char_sets_config(char_config_path)
    pattern_name_map = load_pattern_name_mapping(Path("config/pattern_name_mapping.json"))
    
    # 由於 OCRProcessor 需要 roi_header_dict，這取決於機型，所以我們延後初始化或動態更新
    # 這裡我們先建立一個暫存的結構，等機型確認後再實例化 Processor
    ocr_processor: Optional[OCRProcessor] = None 
    roi_dict: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    
    # --- 2. 狀態變數 ---
    t0 = time.time()
    processed_frames = 0
    
    # 機型偵測相關
    machine_detected = False
    machine_id = None
    pattern2_start_frame = None
    
    # 緩衝區：儲存 (frame_idx, frame_bgr, stage_result)
    # 用於在機型確認前暫存畫面，以便回溯 OCR
    frame_buffer: List[Tuple[int, np.ndarray, Dict]] = []
    
    # Frame Generator
    frame_gen = video_frame_generator(video_path)
    total_frames = getattr(frame_gen, "total_frames", None)
    
    try:
        for frame_idx, frame_bgr in tqdm(frame_gen, desc=f"Processing frames ({video_name})", total=total_frames):
            processed_frames += 1
            # if frame_idx % 100 == 0:
            #     print(f"Processing frame {frame_idx}...", end='\r')
            
            # [Step A] 非同步存大圖 (Frame Cache)
            # 模擬 extract_frame_cache 的行為
            cache_path = frame_cache_dir / f"frame_{frame_idx}.jpg"
            if force or not cache_path.exists():
                async_saver.save(frame_bgr, cache_path, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            
            # [Step B] 階段分析 (Stage Analysis)
            # 這是通用的，不依賴機型
            stage_res = stage_analyzer.process_frame(frame_bgr, frame_idx)
            for region_name, res in stage_res.items():
                pid = None
                rmse = None
                if isinstance(res, dict):
                    pid = res.get("pattern_id")
                    rmse = res.get("rmse")
                region_matches.setdefault(region_name, []).append((frame_idx, pid, rmse))
            current_stage_pattern = stage_res.get("STAGE", {}).get("pattern_id")
            
            # [Step C] 機型偵測與 OCR 分支邏輯
            if not machine_detected:
                # 尚未確認機型：進入緩衝模式
                frame_buffer.append((frame_idx, frame_bgr.copy(), stage_res))
                
                # 監測 Pattern 2
                if current_stage_pattern == 2 and pattern2_start_frame is None:
                    pattern2_start_frame = frame_idx
                
                # 判斷是否觸發偵測 (Pattern 2 後 5 幀)
                should_detect = (pattern2_start_frame is not None) and \
                                (frame_idx == pattern2_start_frame + 5)
                
                # 防呆：若過了很久 (e.g. 500幀) 還沒 Pattern 2，強制使用預設機型
                force_default = (frame_idx > 500 and pattern2_start_frame is None)
                
                if should_detect or force_default:
                    print(f"\n🔍 觸發機型偵測 (Frame {frame_idx})...")
                    
                    # 嘗試偵測
                    header_config = load_roi_header_config(video_name=None) # 預設機型1配置
                    
                    detected_id = None
                    if should_detect and header_config and "region1" in header_config:
                        detected_id = machine_detector.detect_from_frame(frame_bgr, header_config["region1"])
                    
                    machine_id = detected_id if detected_id else 2 # 預設為 2
                    print(f"✅ 機型確認: Type {machine_id}")
                    
                    machine_detected = True
                    
                    # --- 初始化 OCR Processor ---
                    # 1. 載入對應機型的 ROI
                    # 注意：load_roi_config 預設是讀檔，這裡我們需要根據機型 ID 直接載入
                    # 但現有的 load_roi_config 是根據 video_name 去查 rois.json
                    # 為了不修改 rois.json，我們這裡假設 rois.json 已經有 machine_1/machine_2 的模板
                    # 或者我們直接根據 ID 選擇 "machine_1_default" / "machine_2_default"
                    
                    # 這裡使用一個小技巧：直接用 machine_id 來獲取對應的 Config
                    # 假設 config/rois.json 中有 "machine_1_default" 和 "machine_2_default"
                    # 或是使用 load_roi_config 的 behavior：如果 mapping 沒找到，會 fallback
                    # 我們手動構建一個 mock video name 來騙過 load_roi_config，或者直接傳入 machine_id 邏輯
                    
                    # 為了保持乾淨，我們假設 rois.json 裡有定義：
                    # "machine_1_default": { ... }, "machine_2_default": { ... }
                    update_video_machine_mapping(video_name, machine_id)
                    roi_dict = load_roi_config(roi_config_path, video_name=video_name)
                    
                    # 載入 Header Config (用於 OCR active check)
                    try:
                        roi_header_dict = load_roi_header_config(roi_config_path, video_name=video_name)
                    except Exception:
                        roi_header_dict = {}

                    # 初始化 Processor
                    ocr_processor = OCRProcessor(
                        stage_activation_dict,
                        roi_header_dict,
                        char_sets_dict,
                        diff_threshold=0.01
                    )
                    
                    # --- 回溯處理緩衝區 (Flush Buffer) ---
                    print(f"⏪ 回溯處理緩衝區 ({len(frame_buffer)} frames)...")
                    for buf_idx, buf_frame, buf_stage in frame_buffer:
                        # 執行 OCR
                        ocr_processor.process_frame(buf_frame, buf_idx, roi_dict, buf_stage)
                        
                        # 非同步存 ROI 小圖 (如果需要)
                        _save_roi_images(async_saver, buf_frame, roi_dict, buf_idx, analysis_dir, force)
                        
                    frame_buffer = [] # 清空
                    print("⏩ 回溯完成，進入即時模式")

            else:
                # 機型已確認：即時處理模式
                # 直接執行 OCR
                if ocr_processor and roi_dict:
                    ocr_processor.process_frame(frame_bgr, frame_idx, roi_dict, stage_res)
                    
                    # 非同步存 ROI 小圖
                    _save_roi_images(async_saver, frame_bgr, roi_dict, frame_idx, analysis_dir, force)

    except KeyboardInterrupt:
        print("\n⚠️ 使用者中斷")
    except Exception as e:
        print(f"\n❌ 發生未預期的錯誤: {e}")
        traceback.print_exc()
        
    # [Step D] 結束處理：儲存結果
    if ocr_processor:
        print(f"\n💾 儲存 OCR 結果至: {analysis_dir}")
        ocr_processor.save_results(analysis_dir)

    stage_total = total_frames if total_frames is not None else processed_frames
    _write_stage_analysis(video_name, analysis_dir, region_matches, stage_total, pattern_name_map)
        
    # 等待 IO
    print("⏳ 等待背景儲存完成...")
    async_saver.stop()
    print(f"\n✅ 分析完成，總耗時: {time.time() - t0:.2f}s (Frames: {processed_frames})")


def _save_roi_images(
    saver: AsyncImageSaver,
    frame: np.ndarray,
    roi_config: Dict[str, Tuple[int, int, int, int]],
    frame_idx: int,
    base_dir: Path,
    force: bool,
) -> None:
    """儲存 ROI 原圖與二值圖，與舊流程一致。"""
    for region_name, (x1, y1, x2, y2) in roi_config.items():
        region_dir = base_dir / region_name
        region_dir.mkdir(parents=True, exist_ok=True)

        try:
            roi_bgr = frame[y1:y2, x1:x2]
            if roi_bgr.size == 0:
                continue
        except Exception:
            continue

        orig_path = region_dir / f"frame_{frame_idx}.png"
        bin_path = region_dir / f"frame_{frame_idx}_binary.png"

        if force or not orig_path.exists():
            saver.save(roi_bgr, orig_path)

        if force or not bin_path.exists():
            try:
                roi_binary = binarize(roi_bgr, method="rule")
                saver.save(roi_binary, bin_path)
            except Exception:
                continue


def _write_stage_analysis(
    video_name: str,
    analysis_dir: Path,
    region_matches: Dict[str, List[Tuple[int, Optional[int], Optional[float]]]],
    total_frames: int,
    pattern_name_map: Dict[str, Dict[str, str]],
) -> None:
    regions_output: Dict[str, List[Dict[str, Any]]] = {}
    for region_name, matches in region_matches.items():
        if not matches:
            regions_output[region_name] = []
            continue
        matches.sort(key=lambda t: t[0])
        region_map = pattern_name_map.get(region_name, {})
        segments = build_segments(matches, region_map)
        cleaned_segments: List[Dict[str, Any]] = []
        for seg in segments:
            start = int(seg.get("start_frame", 0))
            end = int(seg.get("end_frame", start))
            if end < start:
                end = start
            seg["start_frame"] = start
            seg["end_frame"] = end
            seg["frame_count"] = max(0, end - start + 1)
            cleaned_segments.append(seg)
        regions_output[region_name] = cleaned_segments

    payload = {
        "video": video_name,
        "total_frames": total_frames,
        "regions": regions_output,
    }

    out_path = analysis_dir / "stage_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-pass surgery pipeline")
    parser.add_argument("--video", type=Path, required=True, help="影片檔或包含影片的資料夾")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="輸出根目錄")
    parser.add_argument("--force", action="store_true", help="覆蓋既有 frame cache 與 ROI 圖片")
    args = parser.parse_args()

    target_path = args.video
    if not target_path.exists():
        print(f"❌ 路徑不存在: {target_path}")
        return

    if target_path.is_file():
        video_files = [target_path]
    else:
        video_files = sorted({*target_path.glob("*.mp4"), *target_path.glob("*.MP4")})
        if not video_files:
            print(f"⚠️ 在目錄 {target_path} 中未找到 .mp4 檔案")
            return
        print(f"📂 找到 {len(video_files)} 個影片檔案，準備開始批次分析...")

    for idx, vf in enumerate(video_files, start=1):
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(video_files)}] 🎬 處理影片: {vf.name}")
        print(f"{'=' * 60}")
        try:
            run_pipeline(vf, args.output_dir, args.force)
        except Exception as e:
            print(f"❌ 處理 {vf.name} 時發生錯誤: {e}")
            traceback.print_exc()

    print("\n✅ 所有任務完成")


if __name__ == "__main__":
    main()
