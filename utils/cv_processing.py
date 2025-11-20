from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

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

def binarize_pil(image: Image.Image, method: str = "rule", *,
             hsv_s_thresh: int = 30, gray_thresh: int = 150) -> np.ndarray:
    """Return a binary (uint8 0/255) image."""
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    binary = binarize(image_bgr, method)
    return binary


def calculate_roi_diff(prev_img: Image.Image, curr_img: Image.Image, coords: Tuple[int, int, int, int]) -> tuple[float, np.ndarray | None]:
    """Calculate the difference between two ROI images."""
    try:
        # 從兩張圖像中裁剪出精細區域
        prev_sub_roi = prev_img.crop(coords)
        curr_sub_roi = curr_img.crop(coords)
        
        # 轉換為 NumPy 數組
        prev_arr = np.array(prev_sub_roi).astype(np.float32)
        curr_arr = np.array(curr_sub_roi).astype(np.float32)
        
        # 檢查尺寸是否一致
        if prev_arr.shape != curr_arr.shape:
            return 0.0, None
        
        # 計算每個像素RGB通道差值的平方
        squared_diff = np.square(prev_arr - curr_arr)
        
        # 計算每個像素的均方差 (MSE)
        mse_per_pixel = np.mean(squared_diff, axis=2)
        
        # 計算每個像素的均方根差 (RMSE)，即顏色距離
        rmse_per_pixel = np.sqrt(mse_per_pixel)
        average_rmse = float(np.mean(rmse_per_pixel))
        
        # 返回所有像素顏色距離的平均值和完整的差異矩陣
        return average_rmse, rmse_per_pixel
        
    except Exception as e:
        print(f"計算 roi_diff 時出錯: {e}")
        return 0.0, None

def calculate_average_binary_diff(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        return 0.0
    b1 = (img1 > 127).astype(np.uint8)
    b2 = (img2 > 127).astype(np.uint8)
    diff = np.logical_xor(b1, b2)
    return float(np.mean(diff))

SINGLE_DIGIT_BORDER = 40        # px 左右黑邊寬度門檻
SINGLE_DIGIT_THRESH = 0.03      # 左或右 >3% 白點 ⇒ 不是單一數字
def is_single_digit(bw: np.ndarray) -> bool:
    """Return True if ROI likely contains exactly *one* digit.

    Heuristic: check left & right 40‑pixel columns for white ratio.
    """
    h, w = bw.shape
    border = min(SINGLE_DIGIT_BORDER, w // 3)  # avoid over‑size for narrow ROI
    left_white  = np.count_nonzero(bw[:, :border]) / (h * border)
    #right_white = np.count_nonzero(bw[:, -border:]) / (h * border)
    return left_white < SINGLE_DIGIT_THRESH #and right_white < SINGLE_DIGIT_THRESH


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


def resize_keep_aspect(image: Image.Image, max_size: tuple[int, int] = (200, 150), *, max_upscale: float = 3.0) -> Image.Image:
    """等比縮放 PIL 影像至指定最大尺寸，允許最多 max_upscale 倍放大。

    - 若原圖已小於 max_size，允許放大但不超過 max_upscale 倍。
    - 若原圖大於 max_size，按比例縮小以適配。
    """
    try:
        original_width, original_height = image.size
        max_width, max_height = max_size

        if original_width <= 0 or original_height <= 0:
            return image

        if original_width <= max_width and original_height <= max_height:
            scale_x = max_width / original_width
            scale_y = max_height / original_height
            scale = min(scale_x, scale_y, float(max_upscale))
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
        else:
            scale_x = max_width / original_width
            scale_y = max_height / original_height
            scale = min(scale_x, scale_y)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

        if new_width <= 0 or new_height <= 0:
            return image

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except Exception:
        return image