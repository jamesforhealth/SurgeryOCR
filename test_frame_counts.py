#!/usr/bin/env python3
"""test_frame_counts.py

批量測試影片總幀數差異
========================

1. 依序掃描指定目錄（預設為當前工作目錄）下所有影片檔案（*.mp4, *.avi, *.mov, *.mkv）。
2. 對每支影片分別計算：
   • `prop_count`  : `cv2.CAP_PROP_FRAME_COUNT` 回傳的幀數（GUI 主要依此值）。
   • `iter_count`  : 逐幀 `grab()` 真正成功讀取的幀數（分析腳本最終實際讀到的幀數）。
3. 將結果以表格列印，並標註差異。

用法：
------
```bash
python test_frame_counts.py --root /path/to/search/root
```
未指定 `--root` 則預設為程式所在目錄。
"""
from __future__ import annotations
import argparse
import cv2
from pathlib import Path
from typing import List

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def enumerate_videos(root: Path) -> List[Path]:
    """Recursively collect video paths under *root* with supported extensions."""
    return [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]


def get_prop_frame_count(cap: cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_iter_frame_count(cap: cv2.VideoCapture) -> int:
    """Count frames by repeatedly grab()ing until失敗; 更快且不解碼內容。"""
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
    return frame_idx

def analyse_video(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return -1, -1  # 表示影片無法開啟

    prop_count = get_prop_frame_count(cap)

    # 將檔案指標重置再做逐幀統計
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    iter_count = get_iter_frame_count(cap)
    cap.release()
    return prop_count, iter_count


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare CAP_PROP_FRAME_COUNT vs actual frame count for all videos under root dir")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="Root directory to search (default: current directory)")
    args = ap.parse_args()

    videos = enumerate_videos(args.root)
    if not videos:
        print(f"在 {args.root} 找不到任何影片檔案")
        return

    print(f"共找到 {len(videos)} 支影片，開始分析…\n")
    print(f"{'Video':60} | {'prop_count':>10} | {'iter_count':>10} | {'delta':>6}")
    print("-" * 96)

    for vid_path in videos:
        prop_count, iter_count = analyse_video(vid_path)
        delta = prop_count - iter_count if prop_count >= 0 else "-"
        print(f"{vid_path.as_posix():60} | {prop_count:10} | {iter_count:10} | {delta:6}")

    print("\n分析完成。")


if __name__ == "__main__":
    main()
