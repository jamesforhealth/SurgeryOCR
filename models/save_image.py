import cv2
import sys
import os

# 確保有輸入影片檔案
if len(sys.argv) < 2:
    print("Usage: python video_control.py <video_file>")
    sys.exit(1)

video_file = sys.argv[1]
if not os.path.isfile(video_file):
    print(f"Error: File '{video_file}' does not exist.")
    sys.exit(1)

# 打開影片
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_file}'")
    sys.exit(1)

frame_width = 640
frame_height = 360
paused = False
original_frame = None

def handle_key_input(key, cap, frame, original_frame, paused):
    global frame_width, frame_height
    if key == ord('q'):
        print("Exiting.")
        return False, paused, frame
    elif key == ord(' '):  # 暫停/繼續
        paused = not paused
    elif key == ord('s'):  # 儲存當前畫面
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        output_filename = f"frame_{frame_number}.png"
        if original_frame is not None:
            cv2.imwrite(output_filename, original_frame)
            print(f"Saved original frame to {output_filename}")
        else:
            print("No frame available to save.")
    elif key == ord('f'):  # 快轉10個frame
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        target_frame = current_frame + 10
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video while fast-forwarding.")
            return False, paused, frame
        original_frame = frame.copy()
        frame = cv2.resize(frame, (frame_width, frame_height))
    elif key == ord('r'):  # 倒轉10個frame
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        target_frame = max(current_frame - 10, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            print("Error while rewinding.")
            return False, paused, frame
        original_frame = frame.copy()
        frame = cv2.resize(frame, (frame_width, frame_height))
    return True, paused, frame

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # 保留原畫面
        original_frame = frame.copy()

        # 縮放畫面
        frame = cv2.resize(frame, (frame_width, frame_height))

    # 顯示畫面
    cv2.imshow('Video Player', frame)

    # 按鍵控制
    key = cv2.waitKey(10) & 0xFF
    running, paused, frame = handle_key_input(key, cap, frame, original_frame, paused)
    if not running:
        break

cap.release()
cv2.destroyAllWindows()
