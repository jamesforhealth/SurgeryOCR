import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import threading
import queue
from typing import Optional, Dict, List, Tuple
import json
import traceback

class BinaryDiffAnalyzer(tk.Toplevel):
    def __init__(self, parent, video_path: Path, region_name: str, roi_coords: tuple):
        super().__init__(parent)
        self.title(f"二值化差异分析 - {region_name}")
        self.geometry("1200x800")
        
        # 保存参数
        self.video_path = video_path
        self.region_name = region_name
        self.roi_coords = roi_coords
        self.video_title = video_path.stem
        
        # 数据存储
        self.frames: List[int] = []
        self.diffs: List[float] = []
        self.stop_event = threading.Event()
        self.result_queue = queue.Queue()
        
        # 创建UI
        self._create_widgets()
        
        # 开始分析
        self.analysis_thread = threading.Thread(
            target=self._analyze_frames,
            daemon=True
        )
        self.analysis_thread.start()
        
        # 开始更新图表
        self.after(100, self._update_plot)
    
    
    def _create_widgets(self):
        """创建UI组件"""
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.ax.set_title(f'相鄰幀二值化黑白像素差異 - {self.region_name}', fontsize=14)
        self.ax.set_xlabel('幀號', fontsize=12)
        self.ax.set_ylabel('黑白像素差異比例', fontsize=12)
        self.ax.grid(True)
        
        # 初始化空圖
        self.scatter = self.ax.scatter([], [], s=10, alpha=0.6)
        self.line, = self.ax.plot([], [], 'b-', alpha=0.3)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.canvas.draw()
        
        # 進度條
        self.progress_frame = ttk.Frame(self)
        self.progress_frame.pack(fill=tk.X, padx=10, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.status_label = ttk.Label(self.progress_frame, text="准备分析...")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # 控制按鈕
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(fill=tk.X, padx=10, pady=5)
        self.stop_button = ttk.Button(
            self.button_frame,
            text="停止分析",
            command=self._stop_analysis
        )
        self.stop_button.pack(side=tk.RIGHT, padx=5)
    
    def _analyze_frames(self):
        """分析幀差異的後台線程"""
        try:
            # 獲取ROI目錄
            roi_dir = Path("data") / self.video_title / self.region_name
            print(f"正在分析目錄: {roi_dir.absolute()}")  # 調試信息
            
            if not roi_dir.exists():
                self.result_queue.put(("error", f"ROI目錄不存在: {roi_dir.absolute()}"))
                return
                
            # 獲取所有幀圖像
            frame_files = sorted(
                [f for f in roi_dir.glob("frame_*.png")],
                key=lambda x: int(x.stem.split("_")[1])
            )
            
            if not frame_files:
                self.result_queue.put(("error", f"未找到幀圖像文件: {roi_dir.absolute()}"))
                return
                
            print(f"找到 {len(frame_files)} 個幀文件")  # 調試信息
            
            total_frames = len(frame_files)
            self.result_queue.put(("total_frames", total_frames))
            
            # 分析每一對相鄰幀
            for i in range(len(frame_files) - 1):
                if self.stop_event.is_set():
                    break
                    
                curr_frame = self._process_frame(frame_files[i])
                next_frame = self._process_frame(frame_files[i + 1])
                
                if curr_frame is not None and next_frame is not None:
                    diff = self._calculate_binary_diff(curr_frame, next_frame)
                    frame_idx = int(frame_files[i].stem.split("_")[1])
                    
                    # 發送結果
                    self.result_queue.put(("diff", (frame_idx, diff)))
                    print(f"Frame {frame_idx}: diff = {diff:.4f}")
                else:
                    print(f"跳過幀 {frame_files[i].stem} 或 {frame_files[i+1].stem} 因為處理失敗")
                    
                # 更新進度
                progress = (i + 1) / (total_frames - 1) * 100
                self.result_queue.put(("progress", progress))
                
            self.result_queue.put(("complete", None))
            
        except Exception as e:
            print(f"分析過程出錯: {str(e)}")
            print(f"錯誤詳情: {traceback.format_exc()}")
            self.result_queue.put(("error", str(e)))
    
    def _process_frame(self, frame_path: Path) -> Optional[np.ndarray]:
        try:
            if not frame_path.exists():
                print(f"文件不存在: {frame_path}")
                return None
            if frame_path.stat().st_size == 0:
                print(f"文件大小為0: {frame_path}")
                return None

            abs_path = str(frame_path.absolute())
            img = cv2.imread(abs_path)
            if img is None:
                print(f"cv2.imread 返回 None: {abs_path}")
                try:
                    pil_img = Image.open(abs_path).convert("RGB")
                    img = np.array(pil_img)
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    print(f"用 PIL 成功讀取: {abs_path}")
                except Exception as pil_e:
                    print(f"PIL 讀取也失敗: {pil_e}")
                    return None
            else:
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img

            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return binary

        except Exception as e:
            print(f"處理幀 {frame_path} 時出錯: {str(e)}")
            print(f"錯誤詳情: {traceback.format_exc()}")
            return None

    def _calculate_binary_diff(self, img1: np.ndarray, img2: np.ndarray) -> float:
        if img1.shape != img2.shape:
            return 0.0
        b1 = (img1 > 127).astype(np.uint8)
        b2 = (img2 > 127).astype(np.uint8)
        diff = np.logical_xor(b1, b2)
        return float(np.mean(diff))
    
    def _update_plot(self):
        """高效即時更新圖表"""
        updated = False
        try:
            while True:
                try:
                    result = self.result_queue.get_nowait()
                    result_type = result[0]
                    
                    if result_type == "error":
                        messagebox.showerror("错误", result[1])
                        self.destroy()
                        return
                    
                    elif result_type == "total_frames":
                        self.total_frames = result[1]
                        self.ax.set_xlim(0, self.total_frames)
                        self.canvas.draw()
                    
                    elif result_type == "diff":
                        frame_idx, diff_value = result[1]
                        self.frames.append(frame_idx)
                        self.diffs.append(diff_value)
                        updated = True
                    
                    elif result_type == "progress":
                        self.progress_var.set(result[1])
                        self.status_label.config(
                            text=f"分析进度: {result[1]:.1f}%"
                        )
                    
                    elif result_type == "complete":
                        self.status_label.config(text="分析完成")
                        self.stop_button.config(state=tk.DISABLED)
                        updated = True
                        break
                    
                except queue.Empty:
                    break
            
            # 只在有新資料時才更新圖表
            if updated and self.frames:
                self.scatter.set_offsets(np.c_[self.frames, self.diffs])
                self.line.set_data(self.frames, self.diffs)
                # 自動調整Y軸
                y_min = min(self.diffs) * 0.9 if self.diffs else 0
                y_max = max(self.diffs) * 1.1 if self.diffs else 1
                self.ax.set_ylim(y_min, y_max)
                self.ax.set_xlim(0, max(self.frames) + 1)
                self.canvas.draw_idle()
        
        except Exception as e:
            messagebox.showerror("错误", f"更新图表时出错: {e}")
            self.destroy()
            return
        
        self.after(100, self._update_plot)

    def _stop_analysis(self):
        """停止分析"""
        self.stop_event.set()
        self.status_label.config(text="正在停止分析...")
        self.stop_button.config(state=tk.DISABLED)

# 在主程序中添加测试按钮
def add_test_button(parent_frame: tk.Frame, video_annotator) -> ttk.Button:
    """添加测试按钮到主界面"""
    def on_test_click():
        if not video_annotator.video_file_path or not video_annotator.roi_coords:
            messagebox.showwarning("警告", "请先载入视频并设定ROI区域")
            return
        
        # 创建测试窗口
        analyzer = BinaryDiffAnalyzer(
            video_annotator,
            video_annotator.video_file_path,
            video_annotator.region_name,
            video_annotator.roi_coords
        )
    
    test_button = ttk.Button(
        parent_frame,
        text="测试二值化差异分析",
        command=on_test_click
    )
    return test_button
