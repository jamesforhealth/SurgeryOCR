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


def _initialize_ocr_processor(
    video_name: str,
    roi_config_path: Path,
    stage_activation_dict: Dict,
    char_sets_dict: Dict,
    diff_threshold: float = 0.01
)  -> Tuple[OCRProcessor, Dict[str, Any]]:
    """
    初始化 OCR Processor
    回傳: (ocr_processor, roi_dict)
    """
    # 1. 載入 Config
    roi_dict = load_roi_config(roi_config_path, video_name=video_name)
    
    try:
        roi_header_dict = load_roi_header_config(roi_config_path, video_name=video_name)
    except Exception:
        roi_header_dict = {}

    # 2. 初始化 Processor
    ocr_processor = OCRProcessor(
        stage_activation_dict,
        roi_header_dict,
        char_sets_dict,
        diff_threshold=diff_threshold
    )
    return ocr_processor, roi_dict

def _flush_buffer(
    ocr_processor: OCRProcessor,
    roi_dict: Dict[str, Tuple[int, int, int, int]],
    frame_buffer: List[Tuple[int, np.ndarray, Dict]],
    async_saver: AsyncImageSaver,
    analysis_dir: Path,
    mode: str,
    force: bool
) -> None:
    """
    回溯處理 Buffer 中的幀。
    """
    for buf_idx, buf_encoded_img, buf_stage in frame_buffer:
        # 解碼 JPEG
        buf_frame = cv2.imdecode(buf_encoded_img, cv2.IMREAD_COLOR)
        if buf_frame is None: continue
        
        # 執行 OCR
        ocr_processor.process_frame(buf_frame, buf_idx, roi_dict, buf_stage)
        
        # 非同步存 ROI 小圖 
        if mode == "detail":
            _save_roi_images(async_saver, buf_frame, roi_dict, buf_idx, analysis_dir, force)
    print("⏩ 回溯完成，進入即時模式")
    return 

def run_pipeline(video_path: Path, base_output_dir: Path, mode: str = "detail", force: bool = False):
    """
    單次讀取 (Single-Pass) 分析管線
    """
    video_name = video_path.stem
    
    # ... (省略部分路徑設定代碼) ...
    if base_output_dir.name == video_name:
        analysis_dir = base_output_dir
    else:
        analysis_dir = base_output_dir / video_name
        
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # RAM mode 不需要 frame_cache 目錄
    frame_cache_dir = None
    if mode in ["detail", "frame"]:
        frame_cache_dir = analysis_dir / "frame_cache"
        frame_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 開始串流分析: {video_name} (Mode: {mode})")
    print(f"📂 輸出目錄: {analysis_dir}")

    # --- 1. 初始化各個組件 ---
    # RAM mode 不需要 AsyncImageSaver
    async_saver = AsyncImageSaver() if mode in ["detail", "frame"] else None
    
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
    machine_detector = MachineDetector()
    
    # 載入 Configs
    stage_activation_dict = load_stage_config(stage_activation_path)
    char_sets_dict = load_ocr_char_sets_config(char_config_path)
    pattern_name_map = load_pattern_name_mapping(Path("config/pattern_name_mapping.json"))
    
    # [優化] 預先檢查是否已有機型設定
    # 若 rois.json 中已有該影片的 key，表示之前已跑過或已手動設定，直接使用該設定
    pre_roi_dict = load_roi_config(roi_config_path)
    # load_roi_config 會回傳整個 dict 或特定 video 的設定，我們這裡直接讀 raw json 比較準確
    # 但為了方便，我們用一個簡單邏輯：嘗試 load_roi_config(..., video_name=video_name)
    # 觀察其回傳是否為 default fallback。但因為 fallback 邏輯在 load_roi_config 內部，
    # 最穩妥的方式是直接檢查 video_name 是否在 rois.json 的 keys 中
    
    # 機型偵測相關
    machine_detected = False
    machine_id = None
    ocr_processor: Optional[OCRProcessor] = None 
    roi_dict: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    
    try:
        with open(roi_config_path, 'r', encoding='utf-8') as f:
            full_roi_config = json.load(f)
            if video_name in full_roi_config.get('video_machine_mapping', {}):
                # 直接讀取已知的 machine_id
                machine_id = full_roi_config['video_machine_mapping'][video_name]
                print(f"ℹ️  檢測到已知設定: {video_name} (Type {machine_id})，跳過機型偵測。")
                machine_detected = True

                ocr_processor, roi_dict = _initialize_ocr_processor(video_name, roi_config_path, stage_activation_dict, char_sets_dict)
                
    except Exception as e:
        print(f"⚠️  讀取設定檔時發生警告: {e}")
    
    # --- 2. 狀態變數 ---
    t0 = time.time()
    processed_frames = 0
    
    # 緩衝區：儲存 (frame_idx, frame_bgr, stage_result)
    # 用於在機型確認前暫存畫面，以便回溯 OCR
    frame_buffer: List[Tuple[int, np.ndarray, Dict]] = []
    
    # Frame Source Setup
    cap = None
    cache_files = []
    cache_iterator = None
    total_frames = 0
    
    if mode == "read":
        if not frame_cache_dir.exists():
            print(f"❌ [Read Mode] Cache directory not found: {frame_cache_dir}")
            return
        
        # 讀取所有 cache 檔案並按 frame index 排序
        # 假設檔名格式為 frame_{idx}.jpg
        try:
            cache_files = sorted(
                frame_cache_dir.glob("frame_*.jpg"), 
                key=lambda p: int(p.stem.split('_')[1])
            )
        except Exception as e:
             print(f"❌ [Read Mode] Error parsing cache files: {e}")
             return

        if not cache_files:
            print(f"❌ [Read Mode] No cached frames found in: {frame_cache_dir}")
            return
            
        total_frames = len(cache_files)
        cache_iterator = iter(cache_files)
        print(f"📂 [Read Mode] Found {total_frames} cached frames.")
        
    else:
        # 改用直接控制 VideoCapture 以獲取時間戳並支援不同模式
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else None

    frame_timestamps: List[float] = [] # 記錄每幀的時間戳 (ms)
    
    try:
        # 初始化 tqdm
        pbar = tqdm(total=total_frames, desc=f"Processing frames ({video_name})")
        
        frame_idx = 0
        while True:
            frame_bgr = None
            ts = 0.0
            
            if mode == "read":
                try:
                    img_path = next(cache_iterator)
                    # 從檔名解析 frame_idx
                    frame_idx = int(img_path.stem.split('_')[1])
                    
                    # [Fix] Windows 下路徑含中文時，cv2.imread 會失敗，需改用 imdecode
                    # frame_bgr = cv2.imread(str(img_path))
                    img_array = np.fromfile(str(img_path), dtype=np.uint8)
                    frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame_bgr is None:
                        print(f"⚠️ [Read Mode] Failed to read image: {img_path}")
                        continue
                    # Read mode 下時間戳暫時設為 0 或依賴外部記錄 (此處簡化)
                    ts = 0.0 
                except StopIteration:
                    break
            else:
                if not cap.isOpened():
                    break
                    
                # 獲取當前時間戳 (在 read 之前獲取 POS_MSEC)
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                
                if frame_bgr is None or frame_bgr.size == 0:
                    frame_idx += 1 # 即使是壞幀，索引也要遞增，保持時間軸一致
                    continue

            # 記錄時間戳
            frame_timestamps.append(ts)
            
            processed_frames += 1
            pbar.update(1)
            
            # [Step A] 根據 mode 決定是否存 Frame Cache (大圖)
            # mode='detail' or 'frame' -> 存大圖
            # mode='ram' -> 不存
            # mode='read' -> 已經從 cache 讀了，不需要再存
            if mode in ["detail", "frame"]:
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
                # 1. 記憶體優化：將 Frame 壓縮為 JPEG Bytes 存入 Buffer
                success, encoded_img = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if success:
                    frame_buffer.append((frame_idx, encoded_img, stage_res))
                
                # 2. 逐幀偵測 (Frame-by-Frame Detection)
                header_config = load_roi_header_config(video_name=None) # 預設機型1配置
                
                detected_id = None
                if header_config and "region1" in header_config:
                    detected_id = machine_detector.detect_from_frame(frame_bgr, header_config["region1"])
                
                # 條件A: 成功偵測到機型
                # 條件B: Buffer 超過安全上限 (例如 100000 幀)，強制使用預設機型
                force_resolve = len(frame_buffer) > 100000
                
                if detected_id is not None or force_resolve:
                    if detected_id:
                        machine_id = detected_id
                        update_video_machine_mapping(video_name, machine_id)
                        print(f"\n✅ 偵測機型: Type {machine_id} (Frame {frame_idx})")
                    else:
                        machine_id = 1
                        print(f"\n⚠️ 警告：Buffer 超過 100000 幀仍未偵測到機型，強制使用預設 Type 1")

                    if 'pbar' in locals(): pbar.clear()
                    machine_detected = True
                    
                    ocr_processor, roi_dict = _initialize_ocr_processor(video_name, roi_config_path, stage_activation_dict, char_sets_dict)
                    _flush_buffer(ocr_processor, roi_dict, frame_buffer, async_saver, analysis_dir, mode, force)
                    frame_buffer = [] # 清空 Buffer

            else:
                # 機型已確認：即時處理模式
                # 直接執行 OCR
                if ocr_processor and roi_dict:
                    ocr_processor.process_frame(frame_bgr, frame_idx, roi_dict, stage_res)
                    
                    # 非同步存 ROI 小圖
                    if mode == "detail":
                        _save_roi_images(async_saver, frame_bgr, roi_dict, frame_idx, analysis_dir, force)

            frame_idx += 1 # 確保每一幀索引遞增

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
        
    # 等待 IO (只有在有 async_saver 時才需要)
    if async_saver:
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
    parser.add_argument("--mode", type=str, default="ram", choices=["detail", "frame", "ram", "read"],
                       help="存檔模式: detail (存大圖+ROI), frame (只存大圖), ram (不存圖, 最快), read (讀取快取)")
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
        print(f"[{idx}/{len(video_files)}] 🎬 處理影片: {vf.name} (Mode: {args.mode})")
        print(f"{'=' * 60}")
        try:
            run_pipeline(vf, args.output_dir, args.mode, args.force)
        except Exception as e:
            print(f"❌ 處理 {vf.name} 時發生錯誤: {e}")
            traceback.print_exc()

    print("\n✅ 所有任務完成")


if __name__ == "__main__":
    main()
