import cv2
import sys
import os
import time

def get_video_info(video_path):
    if not os.path.exists(video_path):
        print(f"錯誤: 找不到文件 '{video_path}'")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤: 無法打開視頻文件 '{video_path}'")
        return None
    
    # 獲取視頻基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0
    
    # 格式化為時分秒
    minutes, seconds = divmod(duration_sec, 60)
    hours, minutes = divmod(minutes, 60)
    
    info = {
        "path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}",
        "duration_seconds": duration_sec
    }
    
    cap.release()
    return info

def print_video_info(info):
    if not info:
        return
    
    print("\n===== 視頻文件信息 =====")
    print(f"文件路徑: {info['path']}")
    print(f"總幀數: {info['total_frames']}")
    print(f"幀率: {info['fps']:.2f} fps")
    print(f"解析度: {info['width']}x{info['height']}")
    print(f"時長: {info['duration']} (共 {info['duration_seconds']:.2f} 秒)")
    print("========================\n")

def main():
    if len(sys.argv) < 2:
        print("使用方法: python get_video_info.py <視頻文件路徑>")
        return
    
    video_path = sys.argv[1]
    info = get_video_info(video_path)
    if info:
        print_video_info(info)

if __name__ == "__main__":
    main() 