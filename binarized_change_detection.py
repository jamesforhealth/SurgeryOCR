#!/usr/bin/env python3
"""
Script: binarized_change_detection.py
-----------------------------------
A standalone utility that scans every frame of a video, detects content changes
inside a user‑defined ROI **after binarisation**, and performs OCR only on the
frames where a change is detected.  Results are stored in *jsonl* format fully
compatible with the existing GUI tool (e.g. ``region2_ocr.jsonl``).

Highlights
~~~~~~~~~~
* **Binarised diff** – each ROI frame is converted to a binary image (OTSU or a
  simple rule‑based foreground extraction).  A change is registered when the
  proportion of differing pixels w.r.t. the previous binary ROI exceeds a user
  threshold.
* **CLI‑friendly** – invoke via ``python binarized_change_detection.py --video
  VIDEO_PATH --region REGION_NAME``.  All key parameters are configurable
  through flags.
* **Performance logging** – reports average *detection* latency, average *OCR*
  latency, frames analysed, and frames OCR‑ed.
* **Dependency‑light** – only relies on ``opencv‑python``, ``easyocr``, and
  ``numpy``.  (EasyOCR will automatically fall back to CPU if CUDA is not
  available.)

Usage
~~~~~
```bash
python binarized_change_detection.py \
    --video /path/to/video.mp4 \
    --region region2 \
    --roi-config data/config/rois.json \
    --method otsu \
    --diff-thresh 0.01
```

Arguments
~~~~~~~~~
* ``--video``        : Path to the video file.
* ``--region``       : ROI name (must exist in the ROI config).
* ``--roi-config``   : JSON file containing { region_name: [x1, y1, x2, y2] }.
                       Default: ``data/config/rois.json``.
* ``--method``       : Binarisation method – ``otsu`` (default) | ``rule``.
* ``--diff-thresh``  : Proportion of differing pixels (0‑1) needed to flag a
                       change; default **0.005** (0.5 %).
* ``--save-dir``     : Directory to write the ``*_ocr.jsonl``.  Defaults to the
                       directory of the video file.
* ``--device``       : ``gpu`` or ``cpu`` for EasyOCR.  Auto‑detect by default.

Output
~~~~~~
* ``<save-dir>/<region>_ocr.jsonl`` – one JSON per line:
  ``{"frame": 4345, "ocr_text": "123"}``
* Console summary with timing statistics.

"""
from __future__ import annotations
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 使用與 UI app 相同的 OCR 接口
from models.OCR_interface import get_ocr_model
import torch
import matplotlib.pyplot as plt


SINGLE_DIGIT_BORDER = 40        # px 左右黑邊寬度門檻
SINGLE_DIGIT_THRESH = 0.03      # 左或右 >3% 白點 ⇒ 不是單一數字
# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_roi_config(path: Path) -> dict[str, List[int]]:
    """Load ROI dictionary from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"ROI config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, List[int]] = json.load(f)
    return data


def binarize(image_bgr: np.ndarray, method: str = "rule", *,
             hsv_s_thresh: int = 30, gray_thresh: int = 150) -> np.ndarray:
    """Return a binary (uint8 0/255) image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    if method == "rule":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s_pct = (s / 255.0) * 100
        mask = (s_pct < hsv_s_thresh) & (gray > gray_thresh)
        binary = np.zeros_like(gray, dtype=np.uint8)
        binary[mask] = 255
        return binary

    raise ValueError(f"Unsupported binarisation method: {method}")

def calculate_binary_diff(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        return 0.0
    b1 = (img1 > 127).astype(np.uint8)
    b2 = (img2 > 127).astype(np.uint8)
    diff = np.logical_xor(b1, b2)
    return float(np.mean(diff))

def is_single_digit(bw: np.ndarray) -> bool:
    """Return True if ROI likely contains exactly *one* digit.

    Heuristic: check left & right 40‑pixel columns for white ratio.
    """
    h, w = bw.shape
    border = min(SINGLE_DIGIT_BORDER, w // 3)  # avoid over‑size for narrow ROI
    left_white  = np.count_nonzero(bw[:, :border]) / (h * border)
    right_white = np.count_nonzero(bw[:, -border:]) / (h * border)
    return left_white < SINGLE_DIGIT_THRESH and right_white < SINGLE_DIGIT_THRESH


def trim_black_borders(binary_img: np.ndarray, max_border: int = 1) -> np.ndarray:
    """
    去除二值化圖像四個方向的黑邊，保留中間的白色內容，最多保留指定像素的黑邊。
    
    Args:
        binary_img: 二值化圖像 (0=黑, 255=白)
        max_border: 最多保留的黑邊像素數 (預設1像素)
    
    Returns:
        裁切後的二值化圖像
    """
    if binary_img.size == 0:
        return binary_img
    
    h, w = binary_img.shape
    
    # 找到有白色像素的邊界
    white_pixels = binary_img > 127  # 白色像素的遮罩
    
    # 找到包含白色像素的行和列
    rows_with_white = np.any(white_pixels, axis=1)  # 每一行是否有白色
    cols_with_white = np.any(white_pixels, axis=0)  # 每一列是否有白色
    
    # 如果沒有白色像素，返回原圖
    if not np.any(rows_with_white) or not np.any(cols_with_white):
        return binary_img
    
    # 找到第一個和最後一個包含白色的行/列
    top = np.argmax(rows_with_white)
    bottom = len(rows_with_white) - 1 - np.argmax(rows_with_white[::-1])
    left = np.argmax(cols_with_white)
    right = len(cols_with_white) - 1 - np.argmax(cols_with_white[::-1])
    
    # 加上最多 max_border 像素的邊框，但不超出原圖範圍
    top = max(0, top - max_border)
    bottom = min(h - 1, bottom + max_border)
    left = max(0, left - max_border)
    right = min(w - 1, right + max_border)
    
    # 裁切圖像
    trimmed = binary_img[top:bottom+1, left:right+1]
    
    return trimmed


def show_before_after_comparison(original: np.ndarray, trimmed: np.ndarray, frame_idx: int, description: str):
    """顯示前後對照的圖片"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    #set figure title
    fig.suptitle(f'Frame {frame_idx}')

    ax1.imshow(original, cmap='gray')
    ax1.set_title(f'Origin Frame')
    ax1.set_xlabel(f'尺寸: {original.shape[1]}x{original.shape[0]}')
    ax1.axis('off')
    
    ax2.imshow(trimmed, cmap='gray')
    ax2.set_title(f'New Frame')
    ax2.set_xlabel(f'尺寸: {trimmed.shape[1]}x{trimmed.shape[0]}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Core processing routine
# ---------------------------------------------------------------------------

def analysis_video_roi(
    video_path: Path,
    save_path: Path,
    roi: Tuple[int, int, int, int],
    method: str = "rule",
    diff_thresh: float = 0.01,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    x1, y1, x2, y2 = roi

    # 使用與 UI app 相同的 OCR 接口初始化
    ocr_iface = get_ocr_model(
        model_type="easyocr",
        gpu=torch.cuda.is_available(),
        lang_list=['en'],
        confidence_threshold=0.5,
        debug_output=False
    )

    prev_bin: np.ndarray | None = None
    results: List[dict[str, object]] = []

    ocr_time_total = 0.0
    ocr_count = 0
    pbar = tqdm(desc="Processing frames", unit="frame")

    t0 = time.perf_counter()
    frame_idx = 0
    diff = 0.0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # End of stream

        roi_bgr = frame_bgr[y1:y2, x1:x2]

        curr_bin = binarize(roi_bgr, method)

        change = False
        if prev_bin is None:
            change = True  # Always OCR first frame
        else:
            diff = calculate_binary_diff(prev_bin, curr_bin)
            change = diff >= diff_thresh
        

        if change:
            # if frame_idx >= 4000:
            #     print(f"Change detected at frame {frame_idx} diff: {diff}", flush=True)
            #     show_before_after_comparison(prev_bin, curr_bin, frame_idx, "Change detected")
            t1 = time.perf_counter()
            
            # 一般情況，使用原始二值化圖像
            pil_img = Image.fromarray(curr_bin)
            ocr_result, confidence = ocr_iface.recognize(pil_img)
            if ocr_result == "":
                # 檢查是否為單一數字並進行黑邊去除處理
                if is_single_digit(curr_bin):
                    # print(f"\n檢測到單一數字 (Frame {frame_idx})，進行黑邊去除處理...", flush=True)
                    trimmed_bin = trim_black_borders(curr_bin, max_border=1)
                    
                    # 使用去除黑邊後的圖像進行 OCR
                    pil_img = Image.fromarray(trimmed_bin)
                    ocr_result, confidence = ocr_iface.recognize(pil_img)
                    # print(f"size of original: {curr_bin.shape}", flush=True)
                    # print(f"size of trimmed: {trimmed_bin.shape}", flush=True)
                    # print(f"OCR 結果: {ocr_result}, 置信度: {confidence}", flush=True)
                    #                     # # 顯示前後對照
                    # show_before_after_comparison(curr_bin, trimmed_bin, frame_idx, "Trimmed")
           
            ocr_elapsed = time.perf_counter() - t1
            ocr_time_total += ocr_elapsed
            ocr_count += 1
            results.append({"frame": frame_idx, "ocr_text": ocr_result, "confidence": confidence})
            pbar.set_postfix({"OCR count": ocr_count})

        prev_bin = curr_bin
        pbar.update(1)
        frame_idx += 1

    detection_time_total = time.perf_counter() - t0
    pbar.close()
    cap.release()

    # Save jsonl --------------------------------------------------------------
    with open(save_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -----------------------------------------------------------------------
    analysed = frame_idx
    avg_det = detection_time_total / analysed if analysed else 0.0
    avg_ocr = ocr_time_total / ocr_count if ocr_count else 0.0

    print("\n=== Summary ===")
    print(f"Video            : {video_path.name}")
    print(f"Total frames     : {analysed}")
    print(f"Frames OCR‑ed    : {ocr_count}")
    print(f"Avg detect time  : {avg_det*1000:.2f} ms/frame")
    print(f"Avg OCR time     : {avg_ocr*1000:.2f} ms/frame (over OCR frames)")
    print(f"Output           : {save_path}")


# ---------------------------------------------------------------------------
# Entry‑point
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    rois: List[str, Tuple[int, int, int, int]],
    method: str = "rule",
    diff_thresh: float = 0.01,
    save_dir: Path | None = None,
) -> None:

    if not save_dir:
        # 從影片路徑提取目錄名稱，建立 data/影片目錄名/ 結構
        video_name = video_path.stem  # 移除副檔名
        save_dir = Path("data") / video_name
    
    save_dir.mkdir(parents=True, exist_ok=True)

    for region_name, roi in rois:
        print(f"Processing ROI: {region_name} - {roi}")
        save_path = save_dir / f"{region_name}_ocr.jsonl"
        analysis_video_roi(
            video_path=video_path,
            save_path=save_path,
            roi=roi,  # type: ignore[arg-type]
            method=method,
            diff_thresh=diff_thresh,
        )    


def parse_args() -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="Binarised ROI frame‑change detection + OCR")
    p.add_argument("--video", required=True, type=Path, help="Path to video file or video files directory (e.g. mp4)")
    p.add_argument("--region", default="all", help="ROI name as in ROI config (e.g. region2)")
    p.add_argument("--roi-config", type=Path, default=Path("data/config/rois.json"), help="ROI config JSON path")
    p.add_argument("--method", choices=["otsu", "rule"], default="rule", help="Binarisation method")
    p.add_argument("--diff-thresh", type=float, default=0.01, help="Diff ratio threshold (0‑1) to flag change")
    p.add_argument("--save-dir", type=Path, help="Directory to save jsonl (defaults to video directory)")
    return p.parse_args()


def main():  # pragma: no cover
    args = parse_args()
    roi_dict = load_roi_config(args.roi_config)

    rois = []
    if args.region == "all":
        for region in roi_dict:
            rois.append((region, tuple(roi_dict[region])))  # type: ignore[arg-type]
            # input(f"ROI: {rois}")
    else:
        if args.region not in roi_dict:
            raise SystemExit(f"Region '{args.region}' not found in ROI config {args.roi_config}")
        rois = [(args.region, tuple(roi_dict[args.region]))]  # type: ignore[arg-type]

    if args.video.is_dir():
        video_files = list(args.video.glob("*.mp4"))
        for video_file in video_files:
            print(f"Processing video: {video_file}")
            process_video(video_file, rois, args.method, args.diff_thresh, args.save_dir)


    elif str(args.video)[-4:].lower() == ".mp4":
        process_video(args.video, rois, args.method, args.diff_thresh, args.save_dir)


if __name__ == "__main__":  # pragma: no cover
    main()