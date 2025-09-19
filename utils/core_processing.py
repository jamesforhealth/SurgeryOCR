from __future__ import annotations
import cv2
import numpy as np

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