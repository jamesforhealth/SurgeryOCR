import threading
import queue
import cv2
import time
from pathlib import Path
import numpy as np

class AsyncImageSaver:
    def __init__(self, max_queue_size=200):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.total_saved = 0

    def save(self, img: np.ndarray, path: Path, params=None):
        """非同步儲存圖片"""
        if img is None or img.size == 0:
            return
        self.queue.put((img.copy(), path, params))

    def _worker(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
                img, path, params = item
                path = Path(path)
                
                # 確保目錄存在
                if not path.parent.exists():
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass # 忽略並發建立目錄時的錯誤
                
                try:
                    suffix = path.suffix.lower() or ".png"
                    # cv2.imencode 需要前導點，如 .jpg/.png
                    encode_params = params if params is not None else []
                    success, encoded_img = cv2.imencode(suffix, img, encode_params)
                    if not success:
                        raise RuntimeError("imencode failed")
                    with open(path, "wb") as f:
                        f.write(encoded_img.tobytes())
                    self.total_saved += 1
                except Exception as e:
                    print(f"[AsyncSaver] Error saving {path}: {e}")
                
                self.queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        """等待所有任務完成並停止"""
        self.queue.join()
        self.stop_event.set()
        self.worker_thread.join()
        # print(f"[AsyncSaver] Finished. Total saved: {self.total_saved}")

