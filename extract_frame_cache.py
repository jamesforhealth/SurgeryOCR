#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple, Optional
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np


# -------------------- Public API for direct video reading --------------------
class VideoFrameStream:
    """可反覆迭代的影片幀串流，同時提供 total_frames 屬性。"""

    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        else:
            total = None
        self.total_frames: Optional[int] = total
        cap.release()

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            return iter(())  # 空迭代器

        def _generator():
            frame_idx = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame is None or frame.size == 0:
                        frame_idx += 1
                        continue

                    yield frame_idx, frame
                    frame_idx += 1
            finally:
                cap.release()

        return _generator()


def video_frame_generator(video_path: Path) -> VideoFrameStream:
    """
    回傳可迭代的 VideoFrameStream，具備 total_frames 屬性，方便 tqdm 顯示進度。
    """
    return VideoFrameStream(video_path)


def ensure_frame_cache_for_video(video_file: Path, *, force: bool, jpeg_quality: int = 85) -> None:
    video_name = video_file.stem
    # 寫入到與影片同層的 <video_name>/frame_cache，避免子資料夾情況下路徑錯置
    out_dir = video_file.parent / video_name / "frame_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 調試輸出：顯示實際寫入路徑
    print(f"📂 目標輸出路徑: {out_dir.absolute()}")

    # 嘗試獲取總幀數用於進度條
    total_meta = 0
    try:
        cap = cv2.VideoCapture(str(video_file))
        if cap.isOpened():
            total_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except:
        pass
        
    print(f"📊 影片元數據顯示總幀數: {total_meta}")
    
    pbar = tqdm(total=total_meta if total_meta > 0 else None, desc=f"建立frame_cache: {video_name}", unit="frame")
    
    idx = 0
    write_count = 0
    fail_count = 0
    skip_count = 0
    
    # 使用 generator 重構迴圈
    for idx, frame_bgr in video_frame_generator(video_file):
        out_file = out_dir / f"frame_{idx}.jpg"
        
        if not out_file.exists() or force:
            # 使用 cv2.imencode + 文件寫入來避免中文路徑問題
            try:
                # 編碼為 JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
                success, encoded_img = cv2.imencode('.jpg', frame_bgr, encode_param)
                
                if success:
                    # 直接寫入二進制數據，避免 cv2.imwrite 的路徑編碼問題
                    with open(out_file, 'wb') as f:
                        f.write(encoded_img.tobytes())
                    
                    write_count += 1
                    # 每 1000 幀驗證一次寫入
                    if write_count == 1 or write_count % 1000 == 0:
                        if out_file.exists():
                            size = out_file.stat().st_size
                            print(f"\n✓ 第 {idx} 幀寫入成功 ({size} bytes)")
                        else:
                            print(f"\n⚠️ 第 {idx} 幀寫入報告成功但文件不存在！")
                else:
                    fail_count += 1
                    if fail_count <= 3:
                        print(f"\n⚠️ 第 {idx} 幀編碼失敗")
            except Exception as e:
                fail_count += 1
                if fail_count <= 3:
                    print(f"\n⚠️ 第 {idx} 幀寫入異常: {e}")
        else:
            skip_count += 1
            
        pbar.update(1)
        
    pbar.close()
    print(f"\n✓ 完成: {video_name}")
    print(f"  總幀數: {idx + 1}")
    print(f"  成功寫入: {write_count}")
    print(f"  跳過（已存在）: {skip_count}")
    print(f"  失敗: {fail_count}")
    print(f"  輸出路徑: {out_dir.absolute()}")


# -------------------- Public API for reading frame_cache directory --------------------
def get_frame_cache_dir(video: Path) -> Path:
    """Return the frame_cache directory for a given video path.
    Preference:
      1) <video_parent>/<video_name>/frame_cache
      2) data/<video_name>/frame_cache (backward compatibility)
    """
    video_name = video.stem if video.is_file() else video.name
    parent_dir = video.parent if video.is_file() else video.parent
    cand1 = parent_dir / video_name / "frame_cache"
    cand2 = Path("data") / video_name / "frame_cache"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    # 預設回傳 cand1 作為預期位置
    return cand1

def _list_frame_files(video: Path) -> List[Path]:
    """列出完整幀快取檔案，並以數字幀索引排序（避免字串排序造成錯序）。"""
    cache_dir = get_frame_cache_dir(video)
    if not cache_dir.exists():
        return []
    candidates: List[Path] = []
    for pattern in ("frame_*.jpg", "frame_*.jpeg", "frame_*.png"):
        candidates.extend(cache_dir.glob(pattern))

    def idx_key(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except Exception:
            return 1 << 60  # 未能解析的放在最後

    return sorted(candidates, key=idx_key)

# -------------------- Public API for iterate frames arrays --------------------
def iterate_frames(video: Path) -> Iterator[Tuple[int, np.ndarray]]:
    """依幀索引遞增產生 (frame_idx, frame_bgr)。遇到不可讀檔案則跳過。"""
    for f in _list_frame_files(video):
        try:
            idx = int(f.stem.split("_")[1])
        except Exception:
            continue
        try:
            img_rgb = Image.open(f).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            yield idx, frame_bgr
        except Exception:
            continue

# -------------------- Public API for get video meta --------------------
def get_video_meta(video: Path) -> Dict[str, Any]:
    """回傳影片（或其 frame_cache）相關的基本資訊。

    優先以 frame_cache 計數 total_frames（更可靠）；
    若可開啟影片，則補充 fps、width、height，否則為 0。

    Returns: {
      'cache_dir': Path,
      'has_cache': bool,
      'total_frames': int,
      'fps': float,
      'width': int,
      'height': int,
    }
    """
    cache_dir = get_frame_cache_dir(video)
    files = _list_frame_files(video)
    total_frames = len(files)

    fps: float = 0.0
    width: int = 0
    height: int = 0
    try:
        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
    except Exception:
        pass

    return {
        'cache_dir': cache_dir,
        'has_cache': cache_dir.exists() and total_frames > 0,
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
    }

def main():
    ap = argparse.ArgumentParser(description="建立 frame_cache（完整幀快取），供後續分析使用")
    ap.add_argument("--video", required=True, type=Path, help="影片檔或包含多支影片的目錄")
    ap.add_argument("--force", action="store_true", help="強制重建（覆蓋既有快取）")
    ap.add_argument("--jpeg-quality", type=int, default=85, help="完整幀 JPG 品質 (1-100，預設85；越低越快、檔案更小)")
    args = ap.parse_args()

    if args.video.is_file():
        ensure_frame_cache_for_video(args.video, force=args.force, jpeg_quality=args.jpeg_quality)
    elif args.video.is_dir():
        videos: List[Path] = []
        for ext in ["*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV", "*.mkv", "*.MKV"]:
            videos += list(args.video.glob(ext))
        
        # 去重：Windows 不區分大小寫，可能會找到重複的文件
        videos = list(set(videos))
        
        if not videos:
            print(f"⚠️ 未找到影片於: {args.video}")
            return
        
        print(f"找到 {len(videos)} 個影片文件")
        for vf in sorted(videos):
            ensure_frame_cache_for_video(vf, force=args.force, jpeg_quality=args.jpeg_quality)
    else:
        print(f"⚠️ 路徑不存在: {args.video}")


if __name__ == "__main__":
    main()
