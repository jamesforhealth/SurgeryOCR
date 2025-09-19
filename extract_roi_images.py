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
from utils.core_processing import binarize
from utils.get_configs import load_roi_config


# 添加HDF5支持
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("警告: 未安装h5py，tensor模式不可用")


def count_frames_accurately(cap: cv2.VideoCapture) -> int:
    """逐帧计算准确的总帧数"""
    print("正在逐帧计算准确的总帧数...")
    
    # 先尝试使用元数据
    meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 逐帧计数
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    actual_frames = 0
    
    pbar = tqdm(desc="计算总帧数", total=meta_frames if meta_frames > 0 else None, unit="frame")
    
    while True:
        ret = cap.grab()  # 使用grab()更快，不需要解码
        if not ret:
            break
        actual_frames += 1
        pbar.update(1)
        
        # 如果实际帧数明显超过元数据，可能元数据有误
        if meta_frames > 0 and actual_frames > meta_frames * 1.5:
            print(f"⚠️ 实际帧数({actual_frames})明显超过元数据({meta_frames})，继续计数...")
    
    pbar.close()
    
    if meta_frames != actual_frames:
        print(f"⚠️ 帧数校正: 元数据 {meta_frames} → 实际 {actual_frames}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开头
    return actual_frames


def process_video_frames(
    video_path: Path,
    roi_items: List[Tuple[str, Tuple[int, int, int, int]]],
    binarize_method: str = "rule",
):
    """
    一個生成器函數，逐幀處理影片並返回所需數據。
    這是影片處理的核心API。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片: {video_path}")

    total_frames = count_frames_accurately(cap)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 確保從頭開始
    
    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        rois_data = {}
        for region_name, roi_coords in roi_items:
            x1, y1, x2, y2 = roi_coords
            roi_pil = frame_pil.crop((x1, y1, x2, y2))
            
            if roi_pil.size[0] > 0 and roi_pil.size[1] > 0:
                roi_bgr_np = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
                binary_np = binarize(roi_bgr_np, method=binarize_method)
                rois_data[region_name] = {
                    "original_pil": roi_pil,
                    "binary_np": binary_np
                }

        yield {
            "frame_idx": frame_idx,
            "full_frame_pil": frame_pil,
            "original_frame": frame_bgr,  # 添加原始BGR幀用於header檢測
            "rois": rois_data,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height
        }
        
        frame_idx += 1
        
    cap.release()


def extract_roi_images_png(
    video_path: Path,
    roi_items: List[Tuple[str, Tuple[int, int, int, int]]],
    save_dir: Path,
    save_binary: bool = True,
    binarize_method: str = "rule",
    force: bool = False,
    save_full_frames: bool = False,
) -> None:
    """擷取並儲存 ROI 圖片為PNG格式，逐幀檢查以支援續傳。"""
    
    if not roi_items:
        print("沒有指定 ROI 區域，結束處理")
        return
    
    # 建立輸出目錄和子目錄
    for region_name, _ in roi_items:
        region_dir = save_dir / region_name
        region_dir.mkdir(parents=True, exist_ok=True)

    # 建立完整幀快取目錄 (如果需要)
    frame_cache_dir = None
    if save_full_frames:
        frame_cache_dir = save_dir / "frame_cache"
        frame_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"將同時儲存完整幀至: {frame_cache_dir}")
        
    # 統計資訊
    saves_count = 0
    skips_count = 0
    save_errors = []
    
    # 進度條和時間
    pbar = None
    start_time = time.perf_counter()

    # 使用新的核心API
    frame_processor = process_video_frames(
        video_path=video_path,
        roi_items=roi_items,
        binarize_method=binarize_method
    )
    
    info_printed = False
    
    for frame_data in frame_processor:
        # 首次迭代時，初始化進度條和顯示影片資訊
        if not info_printed:
            total_frames = frame_data["total_frames"]
            fps = frame_data["fps"]
            width = frame_data["width"]
            height = frame_data["height"]

            print(f"\n=== 影片資訊 ===")
            print(f"檔案: {video_path.name}")
            print(f"FPS: {fps:.2f}")
            print(f"解析度: {width}x{height}")
            print(f"總幀數: {total_frames}")
            print(f"需要處理的區域: {len(roi_items)} 個")
            
            # 驗證 ROI 座標
            for region_name, roi_coords in roi_items:
                x1, y1, x2, y2 = roi_coords
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"ROI 區域 '{region_name}' 的座標無效: {roi_coords}")

            pbar = tqdm(total=total_frames, desc="擷取 ROI 圖片 (PNG模式)", unit="frame")
            info_printed = True

        frame_idx = frame_data["frame_idx"]
        frame_pil = frame_data["full_frame_pil"]
        rois = frame_data["rois"]

        # 儲存完整幀 (如果需要)
        if save_full_frames and frame_cache_dir:
            full_frame_file = frame_cache_dir / f"frame_{frame_idx}.jpg"
            if not full_frame_file.exists() or force:
                try:
                    frame_pil.save(full_frame_file, "JPEG", quality=85)
                    saves_count += 1
                except Exception as e:
                    save_errors.append(f"幀 {frame_idx} (完整幀): {str(e)}")
            else:
                skips_count += 1
        
        for region_name, roi_data in rois.items():
            roi_pil = roi_data["original_pil"]
            binary_np = roi_data["binary_np"]

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
                        pil_binary = Image.fromarray(binary_np)
                        pil_binary.save(binary_file, "PNG")
                        saves_count += 1
                    except Exception as e:
                        save_errors.append(f"幀 {frame_idx}, {region_name} (二值圖): {str(e)}")
                else:
                    skips_count += 1
        
        if pbar:
            pbar.set_postfix({
                "frame": frame_idx,
                "saves": saves_count,
                "errors": len(save_errors)
            })
            pbar.update(1)

    total_time = time.perf_counter() - start_time
    if pbar:
        pbar.close()
    
    # 統計報告
    actual_frames = pbar.n if pbar else 0
    
    print(f"\n=== 擷取完成 (PNG模式) ===")
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


def extract_roi_images_tensor(
    video_path: Path,
    roi_items: List[Tuple[str, Tuple[int, int, int, int]]],
    save_dir: Path,
    save_binary: bool = True,
    binarize_method: str = "rule",
    force: bool = False,
    compression_level: int = 6,
    save_full_frames: bool = False,
) -> None:
    """擷取並儲存 ROI 數據為Tensor格式（HDF5文件，替代PNG文件）"""
    
    if not HDF5_AVAILABLE:
        raise SystemExit("❌ 未安装h5py库，无法使用tensor模式")
    
    if not roi_items:
        print("沒有指定 ROI 區域，結束處理")
        return
    
    # 输出文件路径 - 与PNG模式相同的目录结构
    video_title = video_path.stem
    output_file = save_dir / f"{video_title}.h5"
    
    if output_file.exists() and not force:
        print(f"输出文件已存在: {output_file}")
        print("使用 --force 强制重新处理")
        return
    
    # 确保输出目录存在
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 建立完整幀快取目錄 (如果需要)
    frame_cache_dir = None
    if save_full_frames:
        # 注意：Tensor 模式的輸出是一個檔案，其 "save_dir" 是 `data/`
        # 而 frame_cache 應該在 `data/<video_name>/` 下
        video_name = video_path.stem
        frame_cache_dir = save_dir / video_name / "frame_cache"
        frame_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"將同時儲存完整幀至: {frame_cache_dir}")

    # 開啟影片
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片: {video_path}")
    
    try:
        # 取得影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n=== 影片資訊 ===")
        print(f"檔案: {video_path.name}")
        print(f"FPS: {fps:.2f}")
        print(f"解析度: {original_width}x{original_height}")
        
        # 计算总帧数
        total_frames = count_frames_accurately(cap)
        print(f"總幀數: {total_frames}")
        
        # 验证ROI坐标
        valid_rois = []
        for region_name, coords in roi_items:
            x1, y1, x2, y2 = coords
            
            # 基本验证
            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ 跳过无效ROI '{region_name}': 坐标顺序错误 {coords}")
                continue
                
            # 边界检查
            if (x1 < 0 or y1 < 0 or x2 > original_width or y2 > original_height):
                print(f"⚠️ ROI '{region_name}' 超出视频边界: {coords} (视频尺寸: {original_width}x{original_height})")
                # 裁剪到有效范围
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_width, x2)
                y2 = min(original_height, y2)
                print(f"   已裁剪为: ({x1}, {y1}, {x2}, {y2})")
            
            valid_rois.append((region_name, (x1, y1, x2, y2)))
        
        if not valid_rois:
            print("❌ 没有有效的ROI配置")
            return
        
        print(f"有效ROI区域: {len(valid_rois)} 个")
        for region_name, coords in valid_rois:
            x1, y1, x2, y2 = coords
            print(f"  {region_name}: ({x1}, {y1}, {x2}, {y2}) [{x2-x1}x{y2-y1}]")
        
        # 创建HDF5文件并处理所有帧
        with h5py.File(output_file, 'w') as h5f:
            # 存储元数据
            meta_group = h5f.create_group('metadata')
            meta_group.attrs['video_path'] = str(video_path)
            meta_group.attrs['video_title'] = video_title
            meta_group.attrs['total_frames'] = total_frames
            meta_group.attrs['fps'] = fps
            meta_group.attrs['original_width'] = original_width
            meta_group.attrs['original_height'] = original_height
            meta_group.attrs['preprocessing_time'] = time.time()
            meta_group.attrs['data_format'] = 'tensor'  # 标记为tensor格式
            
            # 存储ROI配置
            roi_group = h5f.create_group('roi_config')
            for region_name, coords in valid_rois:
                roi_group.attrs[region_name] = coords
            
            # 为每个ROI区域创建数据组
            roi_original_data = {}
            roi_binary_data = {} if save_binary else {}
            
            for region_name, coords in valid_rois:
                x1, y1, x2, y2 = coords
                roi_height = y2 - y1
                roi_width = x2 - x1
                
                # 原始ROI数据
                roi_original_data[region_name] = h5f.create_dataset(
                    f'roi_data/{region_name}',
                    shape=(total_frames, roi_height, roi_width, 3),
                    dtype=np.uint8,
                    compression='gzip',
                    compression_opts=compression_level,
                    chunks=True
                )
                
                # 二值化ROI数据（如果需要）
                if save_binary:
                    roi_binary_data[region_name] = h5f.create_dataset(
                        f'roi_binary/{region_name}',
                        shape=(total_frames, roi_height, roi_width),
                        dtype=np.uint8,
                        compression='gzip',
                        compression_opts=compression_level,
                        chunks=True
                    )
            
            # 处理所有帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            pbar = tqdm(total=total_frames, desc="Tensor模式預處理", unit="frame")
            start_time = time.perf_counter()
            
            frame_idx = 0
            while frame_idx < total_frames:
                ret, frame_bgr = cap.read()
                if not ret:
                    print(f"⚠️ 在帧 {frame_idx} 处视频读取结束，实际处理了 {frame_idx} 帧")
                    break
                
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # 儲存完整幀 (如果需要)
                if save_full_frames and frame_cache_dir:
                    full_frame_file = frame_cache_dir / f"frame_{frame_idx}.jpg"
                    if not full_frame_file.exists() or force:
                        try:
                            # 需要將 numpy array 轉回 PIL Image 來儲存
                            Image.fromarray(frame_rgb).save(full_frame_file, "JPEG", quality=85)
                        except Exception as e:
                            print(f"⚠️ 幀 {frame_idx} (完整幀) 存檔失敗: {str(e)}")
                
                # 处理每个ROI区域
                for region_name, coords in valid_rois:
                    x1, y1, x2, y2 = coords
                    
                    # 提取ROI
                    roi_rgb = frame_rgb[y1:y2, x1:x2]
                    
                    # 存储原始ROI数据
                    roi_original_data[region_name][frame_idx] = roi_rgb
                    
                    # 处理二值化数据（如果需要）
                    if save_binary:
                        try:
                            roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
                            roi_binary = binarize(roi_bgr, method=binarize_method)
                            roi_binary_data[region_name][frame_idx] = roi_binary
                        except Exception as e:
                            print(f"⚠️ 帧 {frame_idx} 区域 {region_name} 二值化失败: {e}")
                            # 使用简单的阈值作为备用
                            gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
                            _, roi_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                            roi_binary_data[region_name][frame_idx] = roi_binary
                
                frame_idx += 1
                pbar.update(1)
                
                # 定期更新进度信息
                if frame_idx % 1000 == 0:
                    elapsed = time.perf_counter() - start_time
                    fps_processed = frame_idx / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        "处理FPS": f"{fps_processed:.1f}",
                        "预计剩余": f"{(total_frames - frame_idx) / fps_processed / 60:.1f}min" if fps_processed > 0 else "N/A"
                    })
            
            pbar.close()
            
            # 更新实际处理的帧数
            if frame_idx < total_frames:
                meta_group.attrs['actual_frames'] = frame_idx
                total_frames = frame_idx
        
        total_time = time.perf_counter() - start_time
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print(f"\n=== Tensor模式預處理完成 ===")
        print(f"處理幀數: {total_frames}")
        print(f"輸出文件: {output_file}")
        print(f"文件大小: {file_size_mb:.1f} MB")
        print(f"總耗時: {total_time:.2f} 秒")
        print(f"平均處理速度: {total_frames / total_time:.1f} FPS")
        print(f"數據格式: Tensor (HDF5)")
        
    except Exception as e:
        print(f"❌ Tensor模式預處理失敗: {e}")
        traceback.print_exc()
        raise
    finally:
        cap.release()


def extract_roi_images(
    video_path: Path,
    roi_items: List[Tuple[str, Tuple[int, int, int, int]]],
    save_dir: Path,
    save_binary: bool = True,
    binarize_method: str = "rule",
    force: bool = False,
    mode: str = "png",
    save_full_frames: bool = False,
) -> None:
    """統一的ROI擷取接口，支持PNG和Tensor两种模式"""
    
    if mode == "png":
        extract_roi_images_png(
            video_path=video_path,
            roi_items=roi_items,
            save_dir=save_dir,
            save_binary=save_binary,
            binarize_method=binarize_method,
            force=force,
            save_full_frames=save_full_frames,
        )
    elif mode == "tensor":
        extract_roi_images_tensor(
            video_path=video_path,
            roi_items=roi_items,
            save_dir=save_dir,
            save_binary=save_binary,
            binarize_method=binarize_method,
            force=force,
            save_full_frames=save_full_frames,
        )
    else:
        raise ValueError(f"不支持的模式: {mode}。支持的模式: 'png', 'tensor'")


def process_video(
    video_path: Path,
    roi_items: List[Tuple[str, Tuple[int, int, int, int]]],
    *,
    output_dir: Path | None = None,
    force: bool = False,
    save_binary: bool = True,
    binarize_method: str = "rule",
    mode: str = "png",
    save_full_frames: bool = False,
) -> None:
    """处理单个视频文件"""
    if not output_dir:
        video_name = video_path.stem
        # 對於 Tensor 模式，其 HDF5 文件預設直接放在 'data/' 下
        # 而 frame_cache 和 PNG ROI 圖片在 'data/<video_name>/' 下
        if mode == "tensor":
            output_dir = Path("data")
        else:
            output_dir = Path("data") / video_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"處理模式: {mode.upper()}")
    
    extract_roi_images(
        video_path=video_path,
        roi_items=roi_items,
        save_dir=output_dir,
        save_binary=save_binary,
        binarize_method=binarize_method,
        force=force,
        mode=mode,
        save_full_frames=save_full_frames,
    )


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
    
    parser.add_argument(
        "--save-full-frames",
        action="store_true",
        help="同時儲存完整的影片幀到 'frame_cache' 目錄以加速UI載入"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查tensor模式的依赖
    if args.mode == "tensor" and not HDF5_AVAILABLE:
        print("❌ 错误：tensor模式需要安装h5py库")
        print("   请运行：pip install h5py")
        return
    
    # 载入ROI配置
    roi_dict = load_roi_config(args.roi_config)
    
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
    
    # 处理视频
    if args.video.is_file():
        # 处理单个视频文件
        process_video(
            args.video, 
            roi_items, 
            output_dir=args.output_dir,
            force=args.force,
            save_binary=args.save_binary,
            binarize_method=args.method,
            mode=args.mode,
            save_full_frames=args.save_full_frames,
        )
    elif args.video.is_dir():
        # 处理目录中的所有视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(args.video.glob(f"*{ext}"))
            video_files.extend(args.video.glob(f"*{ext.upper()}"))
        
        if not video_files:
            print(f"在目录 {args.video} 中未找到视频文件")
            return
        
        print(f"找到 {len(video_files)} 个视频文件")
        
        for video_file in sorted(video_files):
            print(f"\n{'='*60}")
            print(f"处理: {video_file.name}")
            print(f"{'='*60}")
            
            try:
                process_video(
                    video_file, 
                    roi_items, 
                    output_dir=args.output_dir,
                    force=args.force,
                    save_binary=args.save_binary,
                    binarize_method=args.method,
                    mode=args.mode,
                    save_full_frames=args.save_full_frames,
                )
            except Exception as e:
                print(f"处理 {video_file.name} 时发生错误: {e}")
                traceback.print_exc()
    else:
        print(f"错误：路径不存在 - {args.video}")


if __name__ == "__main__":
    main()