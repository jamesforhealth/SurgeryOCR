#!/usr/bin/env python3
"""
Script: extract_roi_images.py
-----------------------------------
從影片中擷取指定 ROI 區域的圖片並儲存。

支持两种模式：
1. PNG模式：传统方式，逐帧保存为PNG文件
2. Tensor模式：高效方式，保存为HDF5格式的tensor结构

主要功能：
* 順序讀取影片的每一幀
* 擷取指定 ROI 區域
* PNG模式：儲存為 frame_XXXXXX.png 格式
* Tensor模式：储存为HDF5格式，支持快速随机访问
* 顯示進度條與統計資訊
* 支援多個 ROI 區域同時擷取
* 支援批量處理影片目錄
* 智能跳過已處理的影片+區域組合

Usage:
```bash
# PNG模式：擷取單一區域
python extract_roi_images.py \
    --video /path/to/video.mp4 \
    --region region2 \
    --mode png

# Tensor模式：高效存储
python extract_roi_images.py \
    --video /path/to/video.mp4 \
    --region all \
    --mode tensor

# 擷取所有區域（PNG模式）
python extract_roi_images.py \
    --video /path/to/video.mp4 \
    --region all

# 批量處理影片目錄（Tensor模式）
python extract_roi_images.py \
    --video /path/to/videos/ \
    --region all \
    --mode tensor

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
from extract_frame_cache import get_frame_cache_dir, iterate_frames, get_video_meta
from utils.cv_processing import binarize
from utils.get_configs import load_roi_config
from utils.get_paths import resolve_video_analysis_dir

# 添加HDF5支持
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("警告: 未安装h5py，tensor模式不可用")


# -------------------- Public API for analysis to get ROI data --------------------
def iterate_roi_binary_from_cache(
    video_path: Path,
    rois: List[Tuple[str, Tuple[int, int, int, int]]],
    binarize_method: str = "rule",
):
    """讀取已切好的 ROI 二值圖，逐幀輸出，不做任何影像處理。

    Yields: (frame_idx, { region_name: { 'binary_np': np.ndarray } })
    """
    base_dir = resolve_video_analysis_dir(video_path)
    # 蒐集所有區域的可用幀索引
    available_frames = set()
    region_dirs = {name: base_dir / name for name, _ in rois}
    for region_name, _ in rois:
        rdir = region_dirs[region_name]
        if not rdir.exists():
            continue
        for fp in rdir.glob("frame_*_binary.png"):
            try:
                idx = int(fp.stem.split("_")[1])
                available_frames.add(idx)
            except Exception:
                continue
    for frame_idx in sorted(available_frames):
        rois_data: Dict[str, Dict[str, np.ndarray]] = {}
        bin_file = rdir / f"frame_{frame_idx}_binary.png"
        if not bin_file.exists():
            continue
        try:
            pil_img = Image.open(bin_file).convert("L")
        except Exception:
            continue

        for region_name, _ in rois:
            rdir = region_dirs[region_name]
            if not rdir.exists():
                continue
            try:
                rois_data[region_name] = {"binary_np": np.array(pil_img)}
            except Exception:
                continue
        if rois_data:
            yield frame_idx, rois_data

def process_video(
    video_path: Path,
    roi_items: List[Tuple[str, Tuple[int, int, int, int]]],
    *,
    output_dir: Path | None = None,
    force: bool = False,
    save_binary: bool = True,
    binarize_method: str = "rule",
    mode: str = "png",
) -> None:
    """处理单个视频文件"""
    if not output_dir:
        # 對於 Tensor 模式，其 HDF5 文件預設直接放在 'data/' 下
        # 而 frame_cache 和 PNG ROI 圖片在分析目錄下（支持子目錄結構）
        if mode == "tensor":
            output_dir = Path("data")
        else:
            output_dir = resolve_video_analysis_dir(video_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"處理模式: {mode.upper()}")

    # 使用 frame_cache 來源：逐幀讀取 frame_cache，裁切 ROI（統一來源，不再解碼影片）
    video_name = video_path.stem
    cache_dir = get_frame_cache_dir(video_path)
    if not cache_dir.exists():
        print(f"❌ 缺少 frame_cache: {cache_dir}，請先執行 extract_frame_cache.py")
        return

    # 檢查是否已存在 ROI 輸出且非 --force
    if not force:
        # 精確判斷：所有區域的原圖（以及需要時的二值圖）數量都等於快取幀數，才整支影片跳過
        meta = get_video_meta(video_path)
        expected_total = int(meta.get('total_frames', 0))
        if expected_total <= 0:
            print(f"❌ frame_cache 為空: {cache_dir}")
            return
        all_regions_complete = True
        for region_name, _ in roi_items:
            out_dir = output_dir / region_name
            if not out_dir.exists():
                all_regions_complete = False
                break
            orig_count = len([f for f in out_dir.glob("frame_*.png") if not f.name.endswith("_binary.png")])
            if orig_count < expected_total:
                all_regions_complete = False
                break
            if save_binary:
                bin_count = len(list(out_dir.glob("frame_*_binary.png")))
                if bin_count < expected_total:
                    all_regions_complete = False
                    break
        if all_regions_complete:
            print(f"⏩ 已存在所有 ROI 輸出且未指定 --force，跳過: {video_name}")
            return

    # 逐幀讀 cache 並裁切（直接用 iterate_frames）
    for region_name, _ in roi_items:
        (output_dir / region_name).mkdir(parents=True, exist_ok=True)

    total_hint = int(get_video_meta(video_path).get('total_frames', 0)) or None
    pbar = tqdm(total=total_hint, desc=f"裁切ROI: {video_name}", unit="frame")
    saves_count = 0
    skips_count = 0
    save_errors = []
    for frame_idx, frame_bgr in iterate_frames(video_path):
        if frame_bgr is None or frame_bgr.size == 0:
            pbar.update(1)
            continue
        # 決定本幀是否需要處理任一區域，若全部文件都已存在可直接略過影像載入
        need_load_image = False
        per_region_write_plan = []  # (region_name, need_orig, need_bin)
        for region_name, _ in roi_items:
            out_orig = output_dir / region_name / f"frame_{frame_idx}.png"
            out_bin  = output_dir / region_name / f"frame_{frame_idx}_binary.png"
            need_orig = force or not out_orig.exists()
            need_bin  = save_binary and (force or not out_bin.exists())
            if need_orig or need_bin:
                need_load_image = True
            per_region_write_plan.append((region_name, need_orig, need_bin))

        if not need_load_image:
            # 本幀所有區域檔案都已存在
            skips_count += 1
            pbar.set_postfix({"saves": saves_count, "skips": skips_count, "errors": len(save_errors)})
            pbar.update(1)
            continue
        img_bgr = frame_bgr
        for region_name, roi_coords in roi_items:
            x1, y1, x2, y2 = roi_coords
            roi_bgr = img_bgr[y1:y2, x1:x2]
            if roi_bgr.size == 0:
                continue
            roi_pil = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
            # 按計畫逐檔判斷是否需要寫出
            out_orig = output_dir / region_name / f"frame_{frame_idx}.png"
            out_bin  = output_dir / region_name / f"frame_{frame_idx}_binary.png"

            # 原圖
            if force or not out_orig.exists():
                try:
                    roi_pil.save(out_orig)
                    saves_count += 1
                except Exception as e:
                    save_errors.append(f"{region_name}@{frame_idx}: {e}")

            # 二值圖
            if save_binary and (force or not out_bin.exists()):
                binary = binarize(roi_bgr, method=binarize_method)
                try:
                    Image.fromarray(binary).save(out_bin)
                    saves_count += 1
                except Exception as e:
                    save_errors.append(f"{region_name}@{frame_idx} binary: {e}")

        pbar.set_postfix({"saves": saves_count, "skips": skips_count, "errors": len(save_errors)})
        pbar.update(1)
    pbar.close()
    print(f"✓ 完成從 frame_cache 裁切 ROI: {video_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="從影片中擷取 ROI 區域圖片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  # PNG模式：擷取單一區域
  python extract_roi_images.py --video video.mp4 --region region2 --mode png
  
  # Tensor模式：高效存储（包含PNG兼容性）
  python extract_roi_images.py --video video.mp4 --region all --mode tensor
  
  # 擷取所有區域（默认PNG模式）
  python extract_roi_images.py --video video.mp4 --region all
  
  # 批量處理影片目錄（Tensor模式）
  python extract_roi_images.py --video /path/to/videos/ --region all --mode tensor
  
  # 強制重新處理（忽略已存在的檔案）
  python extract_roi_images.py --video video.mp4 --region all --force
  
  # 默认进行准确帧数计算。
  python extract_roi_images.py --video video.mp4 --region all
  
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
        default=Path("config/rois.json"),
        help="ROI 配置檔案路徑 (預設: config/rois.json)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=Path,
        help="輸出目錄 (預設: PNG模式為data/影片名稱/，Tensor模式為data/preprocessed/)"
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
    
    parser.add_argument(
        "--mode",
        choices=["png", "tensor"],
        default="png",
        help="輸出模式：png=传统PNG文件，tensor=HDF5格式 (預設: png)"
    )
    
    # 從 frame_cache 讀取已是唯一來源，無需額外旗標
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查tensor模式的依赖
    if args.mode == "tensor" and not HDF5_AVAILABLE:
        print("❌ 错误：tensor模式需要安装h5py库")
        print("   请运行：pip install h5py")
        return
    
    # 处理视频
    if args.video.is_file():
        # 处理单个视频文件
        video_name = args.video.stem
        
        # 根据视频名称载入对应的ROI配置
        roi_dict = load_roi_config(path=args.roi_config, video_name=video_name)
        
        # 处理区域选择
        if args.region == "all":
            roi_items = [(region, tuple(coords)) for region, coords in roi_dict.items()]
        else:
            if args.region not in roi_dict:
                print(f"错误：区域 '{args.region}' 在配置文件中不存在")
                print(f"可用区域：{list(roi_dict.keys())}")
                return
            roi_items = [(args.region, tuple(roi_dict[args.region]))]
        
        if not roi_items:
            print("错误：没有可处理的ROI区域")
            return
        
        print(f"将处理 {len(roi_items)} 个ROI区域: {[item[0] for item in roi_items]}")
        
        process_video(
            args.video, 
            roi_items, 
            output_dir=args.output_dir,
            force=args.force,
            save_binary=args.save_binary,
            binarize_method=args.method,
            mode=args.mode,
        )
        
    elif args.video.is_dir():
        # 处理目录中的所有视频文件（大小寫無關的副檔名過濾）
        allowed_exts = {'.mp4', '.avi', '.mov', '.mkv'}
        video_files = [
            p for p in args.video.iterdir()
            if p.is_file() and p.suffix.lower() in allowed_exts
        ]
        
        if not video_files:
            print(f"在目录 {args.video} 中未找到视频文件")
            return
        
        print(f"找到 {len(video_files)} 个视频文件")
        
        for video_file in sorted(video_files):
            print(f"\n{'='*60}")
            print(f"处理: {video_file.name}")
            print(f"{'='*60}")
            
            try:
                # 为每个视频加载对应的ROI配置
                video_name = video_file.stem
                roi_dict = load_roi_config(path=args.roi_config, video_name=video_name)
                
                # 处理区域选择
                if args.region == "all":
                    roi_items = [(region, tuple(coords)) for region, coords in roi_dict.items()]
                else:
                    if args.region not in roi_dict:
                        print(f"⚠ 警告：区域 '{args.region}' 在 {video_name} 的配置中不存在，跳过...")
                        continue
                    roi_items = [(args.region, tuple(roi_dict[args.region]))]
                
                if not roi_items:
                    print("⚠ 警告：没有可处理的ROI区域，跳过...")
                    continue
                
                print(f"将处理 {len(roi_items)} 个ROI区域: {[item[0] for item in roi_items]}")
                
                process_video(
                    video_file, 
                    roi_items, 
                    output_dir=args.output_dir,
                    force=args.force,
                    save_binary=args.save_binary,
                    binarize_method=args.method,
                    mode=args.mode,
                )
            except Exception as e:
                print(f"处理 {video_file.name} 时发生错误: {e}")
                traceback.print_exc()
    else:
        print(f"错误：路径不存在 - {args.video}")


if __name__ == "__main__":
    main()