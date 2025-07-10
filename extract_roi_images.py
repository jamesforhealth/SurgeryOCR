#!/usr/bin/env python3
"""
Script: extract_roi_images.py
-----------------------------------
從影片中擷取指定 ROI 區域的圖片並儲存到對應目錄。

主要功能：
* 順序讀取影片的每一幀
* 擷取指定 ROI 區域
* 儲存為 frame_XXXXXX.png 格式
* 顯示進度條與統計資訊
* 支援多個 ROI 區域同時擷取
* 支援批量處理影片目錄
* 智能跳過已處理的影片+區域組合

Usage:
```bash
# 擷取單一區域
python extract_roi_images.py \
    --video /path/to/video.mp4 \
    --region region2

# 擷取所有區域
python extract_roi_images.py \
    --video /path/to/video.mp4 \
    --region all

# 批量處理影片目錄
python extract_roi_images.py \
    --video /path/to/videos/ \
    --region all

# 強制重新處理（忽略已存在的檔案）
python extract_roi_images.py \
    --video /path/to/videos/ \
    --region all \
    --force
```
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import traceback
from binarized_change_detection import binarize, load_roi_config

def extract_roi_images(
    video_path: Path,
    roi_items: List[Tuple[str, Tuple[int, int, int, int]]],
    save_dir: Path,
    save_binary: bool = True,
    binarize_method: str = "rule",
    force: bool = False,
) -> None:
    """擷取並儲存 ROI 圖片，逐幀檢查以支援續傳。"""
    
    if not roi_items:
        print("沒有指定 ROI 區域，結束處理")
        return
    
    # 開啟影片
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片: {video_path}")
    
    # 取得影片資訊
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n=== 影片資訊 ===")
    print(f"檔案: {video_path.name}")
    print(f"FPS: {fps:.2f}")
    print(f"解析度: {width}x{height}")
    print(f"需要處理的區域: {len(roi_items)} 個")
    
    # 驗證 ROI 座標
    for region_name, roi_coords in roi_items:
        x1, y1, x2, y2 = roi_coords
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"ROI 區域 '{region_name}' 的座標無效: {roi_coords}")
    
    # 建立輸出目錄
    for region_name, _ in roi_items:
        region_dir = save_dir / region_name
        region_dir.mkdir(parents=True, exist_ok=True)
        
    # 統計資訊
    saves_count = 0
    skips_count = 0
    save_errors = []
    
    # 進度條
    pbar = tqdm(desc="擷取 ROI 圖片", unit="frame")
    
    start_time = time.perf_counter()
    
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"\n影片讀取完畢或發生錯誤，結束處理。")
            break
        
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        for region_name, roi_coords in roi_items:
            # 必須先裁切，因為原圖和二值化圖都從這裡來
            x1, y1, x2, y2 = roi_coords
            roi_pil = frame_pil.crop((x1, y1, x2, y2))
            
            if roi_pil.size[0] == 0 or roi_pil.size[1] == 0:
                save_errors.append(f"幀 {frame_idx}, {region_name}: ROI 為空")
                continue

            # 1. 獨立處理原圖
            roi_file = save_dir / region_name / f"frame_{frame_idx}.png"
            if not roi_file.exists() or force:
                try:
                    roi_pil.save(roi_file, "PNG")
                    saves_count += 1
                except Exception as e:
                    save_errors.append(f"幀 {frame_idx}, {region_name} (原圖): {str(e)}")
            else:
                skips_count += 1

            # 2. 獨立處理二值化圖
            if save_binary:
                binary_file = save_dir / region_name / f"frame_{frame_idx}_binary.png"
                if not binary_file.exists() or force:
                    try:
                        roi_bgr = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
                        roi_binary = binarize(roi_bgr, method=binarize_method)
                        pil_binary = Image.fromarray(roi_binary)
                        pil_binary.save(binary_file, "PNG")
                        saves_count += 1
                    except Exception as e:
                        save_errors.append(f"幀 {frame_idx}, {region_name} (二值圖): {str(e)}")
                else:
                    skips_count += 1
        
        pbar.set_postfix({
            "frame": frame_idx,
            "saves": saves_count,
            "errors": len(save_errors)
        })

        frame_idx += 1
        pbar.update(1)
    
    total_time = time.perf_counter() - start_time
    pbar.close()
    cap.release()
    
    # 統計報告
    actual_frames = pbar.n
    
    print(f"\n=== 擷取完成 ===")
    print(f"總共處理幀數: {actual_frames}")
    print(f"儲存的圖片數 (新增/覆蓋): {saves_count}")
    print(f"跳過的圖片數 (已存在): {skips_count}")
    print(f"儲存錯誤: {len(save_errors)} 個")
    print(f"總耗時: {total_time:.2f} 秒")
    
    if save_errors:
        print(f"\n=== 儲存錯誤詳情 (最多顯示10個) ===")
        for error in save_errors[:10]:
            print(f"  - {error}")
    
    # 顯示各區域的檔案數量
    print(f"\n=== 輸出檔案統計 ===")
    for region_name, _ in roi_items:
        region_dir = save_dir / region_name
        original_files = [f for f in region_dir.glob("frame_*.png") if not f.name.endswith("_binary.png")]
        binary_files = list(region_dir.glob("frame_*_binary.png"))
        
        stats = f"{region_name}: {len(original_files)} 個原圖"
        if save_binary:
            stats += f", {len(binary_files)} 個二值化圖"
        stats += f" -> {region_dir}"
        print(stats)


def process_video(
    video_path: Path,
    roi_items: List[Tuple[str, Tuple[int, int, int, int]]],
    *,
    output_dir: Path | None = None,
    force: bool = False,
    save_binary: bool = True,
    binarize_method: str = "rule"
) -> None:
    """處理單一影片的 ROI 擷取"""
    
    # 決定輸出目錄
    if output_dir is None:
        video_name = video_path.stem
        save_dir = Path("data") / video_name
    else:
        save_dir = output_dir / video_path.stem
    
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"輸出目錄: {save_dir}")
    
    print(f"\n🔄 開始處理區域: {', '.join([item[0] for item in roi_items])}")
    extract_roi_images(
        video_path, 
        roi_items, 
        save_dir, 
        save_binary, 
        binarize_method, 
        force
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="從影片中擷取 ROI 區域圖片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  # 擷取單一區域
  python extract_roi_images.py --video video.mp4 --region region2
  
  # 擷取所有區域
  python extract_roi_images.py --video video.mp4 --region all
  
  # 批量處理影片目錄
  python extract_roi_images.py --video /path/to/videos/ --region all
  
  # 強制重新處理（忽略已存在的檔案）
  python extract_roi_images.py --video video.mp4 --region all --force
  
  # 指定輸出目錄
  python extract_roi_images.py --video video.mp4 --region all --output-dir ./output
        """
    )
    
    parser.add_argument(
        "--video", 
        type=Path, 
        required=True,
        help="影片檔案路徑或影片目錄路徑"
    )
    
    parser.add_argument(
        "--region", 
        default="all",
        help="要擷取的 ROI 區域名稱 (如: region2) 或 'all' 擷取所有區域 (預設: all)"
    )
    
    parser.add_argument(
        "--roi-config", 
        type=Path, 
        default=Path("data/config/rois.json"),
        help="ROI 配置檔案路徑 (預設: data/config/rois.json)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=Path,
        help="輸出目錄 (預設: data/影片名稱/)"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="強制重新處理，忽略已存在的檔案"
    )
    
    parser.add_argument(
        '--no-binary',
        dest='save_binary',
        action='store_false',
        help="不要儲存二值化後的 ROI 圖片 (預設會儲存)"
    )
    
    parser.add_argument(
        "--method",
        choices=["otsu", "rule"],
        default="rule",
        help="二值化方法 (預設: rule)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 載入 ROI 配置
    roi_dict = load_roi_config(args.roi_config)
    
    # 決定要處理的區域
    roi_items = []
    if args.region == "all":
        for region_name, coords in roi_dict.items():
            roi_items.append((region_name, tuple(coords)))
        print(f"\n將處理所有 {len(roi_items)} 個區域: {[item[0] for item in roi_items]}")
    else:
        if args.region not in roi_dict:
            available = list(roi_dict.keys())
            raise SystemExit(f"區域 '{args.region}' 不存在。可用區域: {available}")
        roi_items = [(args.region, tuple(roi_dict[args.region]))]
        print(f"\n將處理區域: {args.region}")
    
    # 統計變數
    total_videos = 0
    total_processed_regions = 0
    total_skipped_regions = 0
    failed_videos = []
    
    # 處理影片
    if args.video.is_dir():
        # 批量處理影片目錄
        video_files = list(args.video.glob("*.mp4"))
        if not video_files:
            raise SystemExit(f"目錄中未找到 .mp4 檔案: {args.video}")
        
        print(f"\n找到 {len(video_files)} 個影片檔案")
        total_videos = len(video_files)
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n{'='*80}")
            print(f"處理影片 {i}/{total_videos}: {video_file.name}")
            print(f"{'='*80}")
            
            try:
                process_video(
                    video_file,
                    roi_items,
                    output_dir=args.output_dir,
                    force=args.force,
                    save_binary=args.save_binary,
                    binarize_method=args.method
                )
                total_processed_regions += 1 # Simplified metric
                print(f"✅ {video_file.name} 處理完成")
            except Exception as e:
                print(f"❌ {video_file.name} 處理失敗: {e}")
                traceback.print_exc()
                failed_videos.append(video_file.name)
                continue
        
        # 最終統計報告
        print(f"\n{'='*80}")
        print(f"批量處理完成統計")
        print(f"{'='*80}")
        print(f"總影片數: {total_videos}")
        print(f"成功處理: {total_videos - len(failed_videos)}")
        print(f"處理失敗: {len(failed_videos)}")
        
        if failed_videos:
            print(f"\n失敗的影片:")
            for video_name in failed_videos:
                print(f"  - {video_name}")
        
    elif args.video.suffix.lower() == ".mp4":
        # 處理單一影片
        if not args.video.exists():
            raise SystemExit(f"影片檔案不存在: {args.video}")
        
        process_video(
            args.video,
            roi_items,
            output_dir=args.output_dir,
            force=args.force,
            save_binary=args.save_binary,
            binarize_method=args.method
        )
        
        print(f"\n=== 處理完成 ===")
        
    else:
        raise SystemExit(f"不支援的檔案格式或路徑: {args.video}")


if __name__ == "__main__":
    main()