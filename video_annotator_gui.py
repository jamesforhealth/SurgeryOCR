#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Video Annotation GUI
- 使用 T-MAD 進行幀間變化偵測
- 分離變化偵測與OCR兩條背景線程
- 定點ROI OCR，連續無變化則複製上一幀結果
- 動態放大影片視窗 (800x450)
- 右側 TreeView 可雙向滾動+垂直 Slider 快速跳轉
- 可編輯、儲存(為JSONL)表格；中途結果與ROI圖自動保存

已修正：
- 每個執行緒使用獨立 VideoCapture，避免 libavcodec async_lock 錯誤
- slider 僅在拖動釋放時讀取幀，提高 UI 流暢度
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import threading
import queue
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import tkinter.font as tkFont # For bold font in Treeview
import csv
import easyocr
import torch

# ---------------- OCR 接口 ----------------
class OCRModelInterface:
    def __init__(self):
        self.models: Dict[str, Dict] = {}
        self.current: Optional[str] = None

    def add_model(self, name: str, model, transform_fn, decode_fn, mapping=None):
        self.models[name] = {"model": model, "tf": transform_fn, "decode": decode_fn, "map": mapping}
        if self.current is None:
            self.current = name

    def get_names(self) -> List[str]:
        return list(self.models.keys())

    def set_current(self, name: str) -> bool:
        if name in self.models:
            self.current = name
            return True
        return False

    def predict(self, pil_img: Image.Image) -> str:
        info = self.models.get(self.current)
        if not info:
            return ""
        reader = info["model"]
        img_for_tf = np.array(pil_img) if not isinstance(pil_img, Image.Image) else pil_img
        img_transformed = info["tf"](img_for_tf)
        res = info["decode"](reader, img_transformed, info.get("map"))
        return res or ""

# --------------- 變化偵測接口 (僅 T-MAD) ---------------
class ChangeDetectorInterface:
    def __init__(self):
        print("變化偵測接口初始化 (使用 T-MAD)。")
        pass

    def _calculate_tmad(self, img1_pil: Image.Image, img2_pil: Image.Image, diff_threshold: int) -> float:
        """計算兩個 PIL 圖像之間的 T-MAD"""
        try:
            # 轉換為灰度 NumPy 數組
            img1_gray = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(np.array(img2_pil), cv2.COLOR_RGB2GRAY)

            # 確保圖像大小相同
            if img1_gray.shape != img2_gray.shape:
                h1, w1 = img1_gray.shape
                h2, w2 = img2_gray.shape
                # 嘗試將第二個圖像調整為第一個的大小
                print(f"警告：T-MAD 計算時圖像大小不匹配 ({h1}x{w1} vs {h2}x{w2})，嘗試調整大小...")
                img2_gray = cv2.resize(img2_gray, (w1, h1), interpolation=cv2.INTER_AREA)

            # 計算絕對差值
            diff = cv2.absdiff(img1_gray, img2_gray)

            # 應用閾值 (忽略小於等於閾值的差異)
            diff_thresholded = diff.copy()
            diff_thresholded[diff_thresholded <= diff_threshold] = 0

            # 計算閾值化後的平均絕對差
            t_mad = np.mean(diff_thresholded)
            return t_mad
        except Exception as e:
            print(f"錯誤：計算 T-MAD 時出錯: {e}")
            return float('inf') # 返回極大值表示錯誤

    def is_changed(self, prev: Image.Image, curr: Image.Image,
                   tmad_threshold: float = 3.0, diff_threshold: int = 30) -> bool:
        """
        使用 T-MAD 判斷兩個圖像之間是否發生變化。

        Args:
            prev: 上一幀的 PIL 圖像 (ROI)。
            curr: 當前幀的 PIL 圖像 (ROI)。
            tmad_threshold: T-MAD 方法的分類閾值。
            diff_threshold: T-MAD 方法中忽略的像素差異閾值。

        Returns:
            如果檢測到變化 (T-MAD >= tmad_threshold) 則返回 True，否則返回 False。
        """
        t_mad = self._calculate_tmad(prev, curr, diff_threshold)
        # print(f"  T-MAD: {t_mad:.4f} (Threshold: {tmad_threshold})") # Debugging
        return t_mad >= tmad_threshold

# -----------------------------------------------------------------
# 1) 重新定義 EasyOCRInterface 取代舊類別
# -----------------------------------------------------------------
class EasyOCRInterface:
    """
    精簡版 EasyOCR：只辨識 0–9 和 '.'；任何錯誤皆回傳空字串。
    """
    def __init__(self, use_gpu: bool = False):
        print("初始化 EasyOCR...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        except Exception as e:
            print(f"GPU 初始化失敗，改用 CPU: {e}")
            self.reader = easyocr.Reader(['en'], gpu=False)

    def predict(self, pil_img: Image.Image) -> str:
        if pil_img is None:
            return ""
        try:
            res = self.reader.readtext(
                np.array(pil_img),
                allowlist="0123456789.",
                detail=0,
                paragraph=False
            )
            return res[0].strip() if res else ""
        except Exception as e:
            print(f"OCR 失敗: {e}")
            return ""

# -------------------- 主GUI --------------------
class VideoAnnotator(tk.Frame):
    VID_W, VID_H = 800, 450
    ROI = (1640, 445, 1836, 525)   # 預設 ROI
    OCR_CONF_TH = 0.5 

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        master.title("Frame Annotation Tool (T-MAD Change Detection)") # 更新標題
        master.geometry("1350x750")
        master.protocol("WM_DELETE_WINDOW", self._on_close)
        self.pack(fill="both", expand=True)

        # 影片來源與總幀數
        self.cap_ui = None          # 用於 UI 顯示
        self.cap_detect = None      # 用於變化偵測
        self.cap_ocr = None         # 用於 OCR 辨識
        self.total_frames = 0
        self.video_file_path = None # Store video path
        self.current_frame_idx = 0
        self.roi_coords = None # 當前使用的 ROI (存儲原始坐標)
        self.original_vid_w = 0 # 存儲原始寬度
        self.original_vid_h = 0 # 存儲原始高度
        self.roi_start_coords = None # 用於拖動選擇

        # 資料暫存
        self.change_cache: Dict[int,bool] = {}
        self.ocr_cache: Dict[int,str] = {}
        self.annotations: Dict[int,str] = {}
        self.roi_image_cache: Dict[int, Image.Image] = {} # Cache ROI images for saving

        # 執行緒通訊
        self.detect_q = queue.Queue()
        self.task_q = queue.Queue()
        self.stop_event = threading.Event()
        self.save_lock = threading.Lock() # Lock for saving operations

        # 模型介面
        self.ocr_iface = EasyOCRInterface(use_gpu=False)
        self.change_iface = ChangeDetectorInterface() # 僅 T-MAD

        # --- 閾值變量 ---
        # 將 T-MAD 分類門檻值默認設為 2.0
        self.tmad_threshold_var = tk.DoubleVar(value=2.0)
        self.diff_threshold_var = tk.IntVar(value=30)

        # --- UI 元素 ---
        self._create_widgets()
        self._load_ocr_models() # 嘗試加載 OCR
        # self._update_status_bar() # Status bar is created later

        # --- 線程和隊列 ---
        self.detect_queue = queue.Queue(maxsize=100) # Limit queue size
        self.ocr_queue = queue.Queue(maxsize=100)    # Limit queue size
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.save_lock = threading.Lock() # Lock for saving operations

        # --- 啟動背景線程 ---
        # 延遲到選擇視頻後啟動
        self.analysis_thread: Optional[threading.Thread] = None   # ← 逐幀分析
        self.ocr_thread: Optional[threading.Thread] = None    # Type hinting

        # --- 啟動結果輪詢 ---
        self.after(100, self._poll_queue) # Start polling result queue

        # 分析起始幀 (由使用者指定)
        self.analysis_start_idx: int = 0

        # 鍵盤左右鍵逐格
        master.bind("<Left>",  lambda e: self._step_frame(-1))
        master.bind("<Right>", lambda e: self._step_frame(+1))

        # -------------  狀態列 (Status Bar) -------------
        self.status_var = tk.StringVar(value="就緒")
        self.lbl_status = tk.Label(
            self,
            textvariable=self.status_var,
            bd=1, relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_widgets(self):
        """創建 GUI 界面元素"""
        # --- Top Control Frame ---
        top_frame = tk.Frame(self)
        top_frame.pack(pady=5, padx=10, fill="x")

        # Video Selection
        btn_load = tk.Button(top_frame, text="選擇影片", command=self._load_video)
        btn_load.pack(side="left", padx=5)
        self.lbl_video_path = tk.Label(top_frame, text="未選擇影片")
        self.lbl_video_path.pack(side="left", padx=5)

        # OCR Model Selection
        tk.Label(top_frame, text="OCR模型:").pack(side="left", padx=(10, 2))
        self.ocr_model_var = tk.StringVar(self)
        # Initialize combobox here, values will be set in _load_ocr_models
        self.ocr_model_combobox = ttk.Combobox(top_frame, textvariable=self.ocr_model_var,
                                               values=[], state="readonly", width=15)
        self.ocr_model_combobox.pack(side="left", padx=2)
        self.ocr_model_combobox.bind("<<ComboboxSelected>>", self._on_ocr_model_change)

        # --- T-MAD 閾值設置 (改用 pack) ---
        # 將 T-MAD 相關控件放在一個子框架中，以便更好地組織
        tmad_frame = tk.Frame(top_frame)
        tmad_frame.pack(side="left", padx=(10, 0)) # Pack the sub-frame

        ttk.Label(tmad_frame, text="T-MAD 門檻:").pack(side="left", padx=(0, 2))
        self.tmad_threshold_spinbox = ttk.Spinbox(tmad_frame, from_=0.0, to=100.0, increment=0.1, width=5, textvariable=self.tmad_threshold_var)
        self.tmad_threshold_spinbox.pack(side="left", padx=(0, 5))

        ttk.Label(tmad_frame, text="忽略差異<=").pack(side="left", padx=(5, 2))
        self.diff_threshold_spinbox = ttk.Spinbox(tmad_frame, from_=0, to=255, increment=1, width=4, textvariable=self.diff_threshold_var)
        self.diff_threshold_spinbox.pack(side="left", padx=(0, 5))

        # Save Button (放在最右側)
        btn_save = tk.Button(top_frame, text="儲存標註", command=self._save_annotations)
        btn_save.pack(side="right", padx=5)

        # --- Main Area (Video + TreeView) ---
        main_area = tk.Frame(self)
        main_area.pack(fill="both", expand=True, padx=10, pady=5)

        # Video Display Frame
        video_frame = tk.Frame(main_area, width=self.VID_W, height=self.VID_H, bd=1, relief="sunken")
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        video_frame.pack_propagate(False) # Prevent resizing based on content

        self.lbl_video = tk.Label(video_frame, bg="black")
        self.lbl_video.pack(fill="both", expand=True)
        self.lbl_video.bind("<Button-1>", self._on_roi_start)
        self.lbl_video.bind("<B1-Motion>", self._on_roi_drag)
        self.lbl_video.bind("<ButtonRelease-1>", self._on_roi_end)
        self.roi_rect_id = None # For drawing ROI rectangle

        # TreeView Frame (with Scrollbars)
        tree_frame = tk.Frame(main_area)
        tree_frame.pack(side="right", fill="y")

        # Vertical Scrollbar for TreeView
        tree_yscroll = ttk.Scrollbar(tree_frame, orient="vertical")
        tree_yscroll.pack(side="right", fill="y")

        # Horizontal Scrollbar for TreeView
        tree_xscroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        tree_xscroll.pack(side="bottom", fill="x")

        # TreeView
        self.tree = ttk.Treeview(tree_frame, columns=("frame", "content"),
                                 show="headings", yscrollcommand=tree_yscroll.set,
                                 xscrollcommand=tree_xscroll.set)
        self.tree.pack(side="left", fill="y")

        self.tree.heading("frame", text="幀號")
        self.tree.heading("content", text="content")

        self.tree.column("frame", width=60, anchor="center")
        self.tree.column("content", width=200, anchor="w")

        tree_yscroll.config(command=self.tree.yview)
        tree_xscroll.config(command=self.tree.xview)

        # 行加粗標籤
        self.bold_font = tkFont.Font(weight="bold")
        self.tree.tag_configure("changed", font=self.bold_font)

        # 允許雙擊編輯 content 欄
        self.tree.bind("<Double-1>", self._on_tree_double_click)

        # --- Bottom Control Frame (Slider) ---
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(fill="x", padx=10, pady=(0, 5))

        self.slider_var = tk.DoubleVar()
        self.slider = ttk.Scale(bottom_frame, from_=0, to=100, orient="horizontal",
                                variable=self.slider_var, command=self._on_slider_move)
        # Bind release event for efficient frame loading
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.slider.pack(fill="x", padx=5, side="left", expand=True)

        self.lbl_frame_num = tk.Label(bottom_frame, text="幀: 0 / 0")
        self.lbl_frame_num.pack(side="right", padx=5)

        # ===== Video + Slider 下方再加一排控制列 =====
        nav_frame = tk.Frame(main_area)
        nav_frame.pack(side="left", fill="x", pady=3)

        # -- Slider (假設程式裡已有 self.slider_var / self.slider) --
        # 在這裡只放精確跳幀輸入框
        tk.Label(nav_frame, text="跳至幀:").pack(side="left", padx=(0,2))
        self.goto_var = tk.IntVar(value=0)
        self.goto_entry = ttk.Entry(nav_frame, textvariable=self.goto_var, width=7)
        self.goto_entry.pack(side="left")
        self.goto_entry.bind("<Return>", self._on_goto_frame)

        tk.Button(nav_frame, text="Go", command=self._on_goto_frame)\
            .pack(side="left", padx=3)

        # ======== 進度條 =========
        prog_frame = tk.Frame(self)
        prog_frame.pack(side=tk.BOTTOM, fill="x")

        self.progress_var = tk.IntVar()
        # maximum 之後在 _start_analysis() 設定
        self.progressbar = ttk.Progressbar(
            prog_frame, length=280, mode="determinate",
            variable=self.progress_var
        )
        self.progressbar.pack(side="right", padx=6)

        self.lbl_prog = tk.Label(prog_frame, text="進度: 0/0")
        self.lbl_prog.pack(side="right")

        # ---- 新增「開始分析」按鈕 ----
        btn_analyse = tk.Button(top_frame, text="開始分析",
                                command=self._start_analysis)
        btn_analyse.pack(side="right", padx=5)

    def _load_ocr_models(self):
        """
        目前僅提供 EasyOCR，一律直接選定；若日後要擴充，可在這裡 append。
        """
        try:
            model_names = ["EasyOCR"]
            self.ocr_model_combobox["values"] = model_names
            self.ocr_model_var.set("EasyOCR")
        except Exception as e:
            print(f"設定 OCR 模型下拉框失敗: {e}")

    def _on_ocr_model_change(self, event=None):
        """
        未來可根據使用者選擇切換不同 OCR；目前僅佔位，不做事。
        """
        pass

    def _load_video(self):
        """Opens a video file, initializes necessary components, and starts processing."""
        filepath = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not filepath:
            return

        self.video_file_path = Path(filepath)
        self.lbl_video_path.config(text=str(self.video_file_path.name))
        self._update_status_bar(f"正在開啟影片: {self.video_file_path.name}...")

        # --- 釋放舊的資源 (如果存在) ---
        self.stop_event.set() # Signal existing threads to stop
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0) # Wait briefly
        if self.ocr_thread and self.ocr_thread.is_alive():
            self.ocr_thread.join(timeout=1.0) # Wait briefly

        for cap in (self.cap_ui, self.cap_detect, self.cap_ocr):
            if cap:
                try:
                    cap.release()
                except Exception as e:
                    print(f"釋放舊 VideoCapture 時出錯: {e}")
        self.cap_ui = self.cap_detect = self.cap_ocr = None
        self.stop_event.clear() # Reset event for new threads

        # --- 清空緩存和數據 ---
        self.change_cache.clear()
        self.ocr_cache.clear()
        self.annotations.clear()
        self.roi_image_cache.clear()
        self.tree.delete(*self.tree.get_children()) # Clear treeview
        with self.detect_queue.mutex: self.detect_queue.queue.clear()
        with self.ocr_queue.mutex: self.ocr_queue.queue.clear()
        with self.result_queue.mutex: self.result_queue.queue.clear()
        self.current_frame_idx = 0
        self.roi_coords = None # 重置 ROI
        # --- 重置 ROI 繪圖 (如果之前有) ---
        # Since we draw on the image in Label now, just clear the reference
        self.current_display_image = None # Clear displayed image reference
        self.lbl_video.config(image='')   # Clear the label image


        # --- 初始化新的 VideoCapture ---
        try:
            self.cap_ui = cv2.VideoCapture(filepath)
            self.cap_detect = cv2.VideoCapture(filepath)
            self.cap_ocr = cv2.VideoCapture(filepath)

            if not self.cap_ui.isOpened():
                raise IOError("無法開啟影片檔案 (UI)")
            if not self.cap_detect.isOpened():
                raise IOError("無法開啟影片檔案 (Detect)")
            if not self.cap_ocr.isOpened():
                raise IOError("無法開啟影片檔案 (OCR)")

            self.total_frames = int(self.cap_ui.get(cv2.CAP_PROP_FRAME_COUNT))
            # --- 獲取並存儲原始視頻尺寸 ---
            self.original_vid_w = int(self.cap_ui.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_vid_h = int(self.cap_ui.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.total_frames <= 0 or self.original_vid_w <= 0 or self.original_vid_h <= 0:
                 raise ValueError("無法讀取有效的影片幀數或尺寸")

            print(f"影片加載成功: {self.total_frames} 幀, 原始尺寸: {self.original_vid_w}x{self.original_vid_h}")

            self.slider.config(to=self.total_frames - 1)
            self.slider_var.set(0)
            self.lbl_frame_num.config(text=f"幀: 0 / {self.total_frames - 1}")

            # --- 嘗試應用默認 ROI ---
            # 如果 self.ROI 是基於原始尺寸定義的，可以直接使用
            # 如果需要根據顯示尺寸調整，可以在這裡計算
            # 假設 self.ROI 是原始坐標
            if self.original_vid_w > 0 and self.original_vid_h > 0:
                 # 驗證默認 ROI 是否在原始邊界內
                 def_x1, def_y1, def_x2, def_y2 = self.ROI
                 if 0 <= def_x1 < self.original_vid_w and \
                    0 <= def_y1 < self.original_vid_h and \
                    def_x1 < def_x2 <= self.original_vid_w and \
                    def_y1 < def_y2 <= self.original_vid_h:
                     self.roi_coords = self.ROI
                     print(f"應用默認 ROI (原始坐標): {self.roi_coords}")
                 else:
                     print("警告：默認 ROI 超出視頻邊界，請手動選擇。")
                     self.roi_coords = None # 清除無效的默認值
            else:
                 self.roi_coords = None # 無法應用默認值

            # --- 重新讀取並顯示第一幀 ---
            self._show_frame(0) # Display the first frame

            self._update_status_bar("影片加載完成，開始自動分析…")

        except (IOError, ValueError, Exception) as e:
            messagebox.showerror("錯誤", f"加載影片失敗: {e}")
            self.lbl_video_path.config(text="未選擇影片")
            self.video_file_path = None
            self.total_frames = 0
            self.original_vid_w = 0
            self.original_vid_h = 0
            self.slider.config(to=0)
            self.slider_var.set(0)
            self.lbl_frame_num.config(text="幀: 0 / 0")
            self._update_status_bar("錯誤：加載影片失敗。")
            # 確保資源已釋放
            for cap in (self.cap_ui, self.cap_detect, self.cap_ocr):
                if cap: cap.release()
            self.cap_ui = self.cap_detect = self.cap_ocr = None

    def _start_analysis_thread(self, start_idx=0):
        if self.analysis_thread and self.analysis_thread.is_alive():
            return
        self.stop_event.clear()
        self.analysis_thread = threading.Thread(
            target=self._analysis_worker,
            args=(start_idx,),
            daemon=True
        )
        self.analysis_thread.start()
    def _analysis_worker(self, start_idx: int | None = None):
        if start_idx is None:
            start_idx = getattr(self, "analysis_start_idx", 0)

        self.cap_detect.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        prev_roi, prev_txt = None, ""
        for idx in range(start_idx, self.total_frames):
            if self.stop_event.is_set():
                break

            ret, frame_bgr = self.cap_detect.read()
            if not ret:
                break
            roi_pil = self._crop_roi(
                Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            )
            if roi_pil is None:
                continue

            # 存 PNG
            self._save_roi_image(idx, roi_pil)
            self.roi_image_cache[idx] = roi_pil

            # 判斷變化
            changed = (prev_roi is None) or self.change_iface.is_changed(
                prev_roi, roi_pil,
                self.tmad_threshold_var.get(),
                self.diff_threshold_var.get()
            )

            if changed:
                # -------- EasyOCR 與信心過濾 --------
                results = self.ocr_iface.reader.readtext(
                    np.array(roi_pil),
                    allowlist="0123456789.",
                    detail=1, paragraph=False
                )
                # 記錄 bbox
                filtered_with_pos = [(txt, bbox[0][0]) for (bbox, txt, conf) in results
                                   if conf >= self.OCR_CONF_TH]
                # 按 x 座標 (左→右) 排序
                filtered_with_pos.sort(key=lambda x: x[1])
                # 只取文字部分
                txt = " ".join(item[0] for item in filtered_with_pos)
            else:
                txt = prev_txt

            # 起始幀或變化幀 → 回傳 UI
            if changed or idx == start_idx:
                self.result_queue.put(("analysis", idx, txt, changed))

            # 進度
            self.result_queue.put(("progress", idx, self.total_frames - 1))

            prev_roi, prev_txt = roi_pil, txt

        self.result_queue.put(("done",))
        self.result_queue.put(("done",))

    def _save_roi_image(self, frame_idx: int, roi_pil: Image.Image):
        roi_dir = self._get_roi_dir()
        png_path = roi_dir / f"frame_{frame_idx}.png"
        try:
            roi_pil.save(png_path, "PNG")
        except Exception as e:
            print(f"[ERR] 儲存 ROI 失敗: {e}")

    def _load_roi_from_file(self, frame_idx: int) -> Image.Image | None:
        png_path = self._get_roi_dir() / f"frame_{frame_idx}.png"
        try:
            return Image.open(png_path) if png_path.exists() else None
        except Exception as e:
            print(f"[ERR] 讀取 ROI 失敗: {e}")
            return None

    def _show_frame(self, frame_idx: int):
        """
        1. 讀取 frame_idx 幀並顯示於 self.lbl_video。
        2. 推送 (frame_idx, PIL) 進 detect_queue 以便背景線程處理。
        """
        if not (0 <= frame_idx < self.total_frames):
            return
        # --- 讀幀 ---
        self.cap_ui.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = self.cap_ui.read()
        if not ret:
            print(f"警告：無法讀取幀 {frame_idx}")
            return
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # --- 調整尺寸顯示 ---
        disp_pil = frame_pil.resize((self.VID_W, self.VID_H), Image.BILINEAR)
        if self.roi_coords:
            # 畫 ROI 紅框
            draw = ImageDraw.Draw(disp_pil)
            scale_x = self.VID_W / self.original_vid_w
            scale_y = self.VID_H / self.original_vid_h
            x1, y1, x2, y2 = self.roi_coords
            draw.rectangle(
                [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y],
                outline="red", width=2
            )
        self.current_display_image = ImageTk.PhotoImage(disp_pil)
        self.lbl_video.config(image=self.current_display_image)

        # --- 更新 Slider/Label 顯示 (避免遞迴觸發 _on_slider_move) ---
        self.slider_var.set(frame_idx)
        self.lbl_frame_num.config(text=f"幀: {frame_idx} / {self.total_frames-1}")
        self.current_frame_idx = frame_idx

        # --- 丟進背景偵測佇列 ---
        try:
            self.detect_queue.put_nowait((frame_idx, frame_pil))
        except queue.Full:
            pass

        # 同步更新 goto_entry
        self.goto_var.set(frame_idx)

    def _on_roi_start(self, event):
        """Records the starting coordinates for ROI selection."""
        self.roi_start_coords = (event.x, event.y)
        # Optional: Delete existing temporary drag rectangle if using Canvas

    def _on_roi_drag(self, event):
        """Draws a temporary rectangle while dragging."""
        if not self.roi_start_coords:
            return

        x1, y1 = self.roi_start_coords
        x2, y2 = event.x, event.y

        # --- 在 lbl_video 上繪製拖動矩形 (需要 Canvas) ---
        # If using Canvas:
        # if self.roi_rect_id: self.lbl_video.delete(self.roi_rect_id)
        # self.roi_rect_id = self.lbl_video.create_rectangle(x1, y1, x2, y2, outline="blue", width=1, tags="roi_rect")

        # --- 如果是 Label (無法直接繪製): ---
        # 拖動時的實時反饋比較困難，可以考慮只在釋放時更新最終矩形
        pass # No easy way to draw temporary rect on Label without redrawing image constantly

    def _on_roi_end(self, event):
        """Finalizes the ROI selection, unscaling coordinates."""
        if not self.roi_start_coords:
            return

        start_x, start_y = self.roi_start_coords
        end_x, end_y = event.x, event.y
        self.roi_start_coords = None # Reset start coords

        # 確保 x1 < x2 and y1 < y2
        label_x1 = min(start_x, end_x)
        label_y1 = min(start_y, end_y)
        label_x2 = max(start_x, end_x)
        label_y2 = max(start_y, end_y)

        # 忽略太小的 ROI
        if abs(label_x2 - label_x1) < 5 or abs(label_y2 - label_y1) < 5:
            print("ROI 太小，已忽略。")
            # Optional: Redraw the current frame without the temporary rect if needed
            self._show_frame(self.current_frame_idx) # Redraw to clear artifacts if any
            return

        # --- 反向縮放坐標回原始尺寸 ---
        if self.original_vid_w > 0 and self.original_vid_h > 0:
            scale_x = self.VID_W / self.original_vid_w
            scale_y = self.VID_H / self.original_vid_h

            # 防止除以零
            if scale_x == 0 or scale_y == 0:
                print("錯誤：縮放比例為零，無法計算原始 ROI。")
                return

            orig_x1 = int(label_x1 / scale_x)
            orig_y1 = int(label_y1 / scale_y)
            orig_x2 = int(label_x2 / scale_x)
            orig_y2 = int(label_y2 / scale_y)

            # --- 邊界檢查 (確保在原始視頻範圍內) ---
            orig_x1 = max(0, orig_x1)
            orig_y1 = max(0, orig_y1)
            orig_x2 = min(self.original_vid_w, orig_x2)
            orig_y2 = min(self.original_vid_h, orig_y2)

            # 再次檢查有效性 (寬高 > 0)
            if orig_x1 >= orig_x2 or orig_y1 >= orig_y2:
                 print("錯誤：計算出的原始 ROI 無效。")
                 return

            new_roi = (orig_x1, orig_y1, orig_x2, orig_y2)

            # 只有當 ROI 實際改變時才觸發更新
            if new_roi != self.roi_coords:
                self.roi_coords = new_roi
                print(f"ROI 已更新 (原始坐標): {self.roi_coords}")
                self._update_status_bar(f"ROI 設定為: {self.roi_coords}")

                # --- ROI 改變後的操作 ---
                # 1. 清除相關緩存 (因為舊的 OCR/變化結果可能基於舊 ROI)
                self.change_cache.clear()
                self.ocr_cache.clear()
                self.roi_image_cache.clear()
                # 2. 清空 TreeView 中的 OCR 和變化列 (標註保留)
                for item_id in self.tree.get_children():
                    self.tree.set(item_id, "change", "")
                    self.tree.set(item_id, "content", "")
                # 3. 可能需要重啟或重新填充背景處理隊列
                #    一個簡單的方法是停止現有線程並重新啟動
                self.stop_event.set()
                if self.analysis_thread and self.analysis_thread.is_alive():
                    self.analysis_thread.join(timeout=0.5)
                self.stop_event.clear()
                # 清空隊列以防萬一
                with self.detect_queue.mutex: self.detect_queue.queue.clear()
                with self.ocr_queue.mutex: self.ocr_queue.queue.clear()
                with self.result_queue.mutex: self.result_queue.queue.clear()
                # 重新啟動線程
                self._start_analysis_thread()
                # 重新處理當前幀及之後的幀可能比較複雜，
                # 一個簡化的方法是讓用戶手動觸發重新分析，
                # 或者至少重新分析當前可見範圍。
                # 這裡我們先只更新顯示。

                # 4. 重新繪製當前幀的 ROI
                self._show_frame(self.current_frame_idx) # 會調用 _draw_roi
            else:
                # ROI 沒變，可能只是點擊了一下，重新繪製以確保框線可見
                 self._show_frame(self.current_frame_idx)

        else:
            print("警告：無法獲取原始視頻尺寸，無法設置 ROI。")

    def _crop_roi(self, frame_pil_full: Image.Image) -> Optional[Image.Image]:
        """Crops the ROI from the full-resolution PIL image."""
        if not self.roi_coords:
            return None
        try:
            # 使用存儲的原始坐標進行裁剪
            x1, y1, x2, y2 = self.roi_coords
            # 確保坐標是整數
            roi_pil = frame_pil_full.crop((int(x1), int(y1), int(x2), int(y2)))
            return roi_pil
        except Exception as e:
            print(f"裁剪 ROI 時出錯: {e}")
            return None
    
    def _on_tree_double_click(self, event):
        """雙擊 content 欄 → 執行 EasyOCR，並回填結果。"""
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not item or column != "#2":      # 只限 content 欄 (#2)
            return
        frame_idx = int(self.tree.set(item, "frame"))

        # 取 ROI 圖
        roi_pil = self.roi_image_cache.get(frame_idx)
        if roi_pil is None:
            roi_pil = self._load_roi_from_file(frame_idx)
        if roi_pil is None:
            messagebox.showerror("錯誤", f"讀不到 frame {frame_idx} 的 ROI 圖。")
            return

        # 直接呼叫 reader，拿完整 (bbox, text, conf)
        results = self.ocr_iface.reader.readtext(
            np.array(roi_pil),
            allowlist="0123456789.",
            detail=1, paragraph=False
        )
        print(f"[OCR] frame {frame_idx}:")
        for (bbox, txt, conf) in results:
            keep = conf >= self.OCR_CONF_TH
            flag = "✔" if keep else "✖"
            x_pos = bbox[0][0]  # 左上角 x 座標
            print(f"  {flag} '{txt}'  pos={x_pos:.1f}  conf={conf:.2f}")
        
        # 按座標排序後才組合文字
        filtered = [(txt, bbox[0][0]) for (bbox, txt, conf) in results 
                    if conf >= self.OCR_CONF_TH]
        filtered.sort(key=lambda x: x[1])  # 按 x 座標排序
        joined_txt = " ".join(item[0] for item in filtered)

        # 彈出可編輯 Entry
        x, y, w, h = self.tree.bbox(item, column)
        edit_win = tk.Toplevel(self)
        edit_win.overrideredirect(True)
        edit_win.geometry(f"{w}x{h}+{self.winfo_rootx()+x}+{self.winfo_rooty()+y}")
        entry = tk.Entry(edit_win)
        entry.insert(0, joined_txt)
        entry.select_range(0, tk.END)
        entry.focus()
        entry.pack(fill="both", expand=True)

        def _save_edit(e=None):
            self.tree.set(item, "content", entry.get())
            edit_win.destroy()
        entry.bind("<Return>", _save_edit)
        entry.bind("<Escape>", lambda e: edit_win.destroy())

    def _on_slider_move(self, value):
        """
        Scale 滑動期間 (實時) 呼叫。
        只更新右側「幀: x / y」顯示，不去真的載入幀，避免拖曳卡頓。
        `value` 由 Tk 傳入，字串或浮點皆有可能。
        """
        try:
            idx = int(float(value))
        except (ValueError, TypeError):
            return
        if self.total_frames:
            self.lbl_frame_num.config(text=f"幀: {idx} / {self.total_frames-1}")

    def _on_slider_release(self, event=None):
        """
        使用者放開滑鼠按鍵 (或完成鍵盤調整) 後呼叫。
        此時才真正載入並顯示指定幀，並推送到背景偵測/ OCR 佇列。
        """
        if self.total_frames == 0:
            return
        idx = int(float(self.slider_var.get()))
        self._show_frame(idx)  # 會自動更新 lbl_frame_num

    def _preload_frames(self, n_frames: int = 150):
        """
        將前 n_frames 幀讀出並放入 detect_queue，
        以便背景 T‑MAD/ OCR 能提早開始運算。
        若想完全停用此機能，可將函式留空或直接 return。
        """
        if not self.cap_detect or not self.cap_detect.isOpened():
            return
        # 從頭開始讀
        self.cap_detect.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for f_idx in range(min(n_frames, self.total_frames)):
            ret, f_bgr = self.cap_detect.read()
            if not ret:
                break
            f_pil = Image.fromarray(cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB))
            try:
                self.detect_queue.put_nowait((f_idx, f_pil))
            except queue.Full:
                break
        # 不改變 UI VideoCapture (self.cap_ui) 的當前位置

    def _save_annotations(self):
        """Saves the current annotations and cached results to a JSONL file."""
        if not self.video_file_path:
            messagebox.showwarning("未載入影片", "請先載入影片再儲存標註。")
            return

        save_path = self._get_save_path(".jsonl")
        if not save_path:
             messagebox.showerror("儲存失敗", "無法確定儲存路徑。")
             return

        self._save_to_file(save_path)

        # --- Also save ROI images ---
        roi_save_dir = save_path.parent / "roi_images"
        try:
            roi_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"正在儲存 ROI 圖像到: {roi_save_dir}")
            saved_roi_count = 0
            with self.save_lock: # Protect access to roi_image_cache while iterating
                roi_items = list(self.roi_image_cache.items()) # Create a copy to iterate

            for frame_idx, img_pil in tqdm(roi_items, desc="儲存 ROI 圖像"):
                if img_pil:
                    try:
                        img_filename = roi_save_dir / f"frame_{frame_idx}.png"
                        img_pil.save(img_filename, "PNG")
                        saved_roi_count += 1
                    except Exception as e:
                        print(f"警告：無法儲存幀 {frame_idx} 的 ROI 圖像: {e}")
            print(f"成功儲存 {saved_roi_count} 個 ROI 圖像。")
            self.lbl_status.config(text=f"標註和 {saved_roi_count} 個 ROI 圖像已儲存")

        except OSError as e:
            print(f"警告：無法創建 ROI 圖像儲存目錄 {roi_save_dir}: {e}")
            messagebox.showwarning("ROI 儲存警告", f"無法創建目錄儲存 ROI 圖像:\n{roi_save_dir}")
        except Exception as e:
             print(f"儲存 ROI 圖像時發生未知錯誤: {e}")

    def _save_to_file(self, file_path: Path):
        """Internal function to save data to the specified JSONL file."""
        print(f"準備儲存標註到: {file_path}")
        results_to_save = []
        # Combine annotations, OCR cache, and change cache
        all_frames = set(self.annotations.keys()) | set(self.ocr_cache.keys()) | set(self.change_cache.keys())
        if not all_frames and self.total_frames > 0: # If caches are empty but video loaded, save frame numbers
             all_frames = set(range(self.total_frames))

        for i in sorted(list(all_frames)):
            # Ensure annotation exists, default to OCR result if annotation is empty/None
            annotation_val = self.annotations.get(i)
            ocr_val = self.ocr_cache.get(i, "")
            final_annotation = annotation_val if annotation_val is not None else ocr_val # Prefer explicit annotation

            record = {
                "frame": i,
                "changed": self.change_cache.get(i, None), # Include change status if available
                "ocr_text": ocr_val,
                "annotation": final_annotation # Use combined annotation
            }
            results_to_save.append(record)

        if not results_to_save:
             print("沒有可儲存的數據。")
             # messagebox.showinfo("無需儲存", "沒有檢測到標註或處理結果需要儲存。")
             self.lbl_status.config(text="無數據可儲存")
             return

        try:
            with self.save_lock: # Use lock for file writing
                with open(file_path, 'w', encoding='utf-8') as f:
                    for record in results_to_save:
                        json.dump(record, f, ensure_ascii=False)
                        f.write('\n')
            # Don't show messagebox here, let the caller (_save_annotations) handle it
            print(f"成功儲存 {len(results_to_save)} 條記錄到 {file_path}")
            # self.lbl_status.config(text="儲存成功") # Status updated in _save_annotations
        except IOError as e:
            messagebox.showerror("儲存失敗", f"無法寫入文件:\n{file_path}\n錯誤: {e}")
            print(f"儲存失敗: {e}")
            self.lbl_status.config(text="儲存失敗")
        except Exception as e:
            messagebox.showerror("儲存失敗", f"儲存過程中發生未知錯誤: {e}")
            print(f"儲存時發生未知錯誤: {e}")
            self.lbl_status.config(text="儲存失敗")

    def _load_annotations(self):
        """Loads annotations from a JSONL file if it exists."""
        load_path = self._get_save_path(".jsonl")
        if load_path and load_path.exists():
            print(f"檢測到標註文件，正在載入: {load_path}")
            try:
                loaded_count = 0
                with open(load_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            frame_idx = record.get("frame")
                            annotation = record.get("annotation")
                            ocr_text = record.get("ocr_text") # Load OCR text too
                            changed = record.get("changed")   # Load change status

                            if frame_idx is not None:
                                # Prioritize non-empty annotation
                                if annotation:
                                    self.annotations[frame_idx] = annotation
                                elif ocr_text: # If annotation empty but OCR exists, use OCR as initial annotation
                                     self.annotations[frame_idx] = ocr_text

                                # Pre-populate caches if not already processed by background tasks
                                # Check if value exists before overwriting potentially newer cache data? No, load should overwrite.
                                if ocr_text is not None:
                                     self.ocr_cache[frame_idx] = ocr_text
                                if changed is not None:
                                     self.change_cache[frame_idx] = changed

                                loaded_count += 1
                        except json.JSONDecodeError:
                            print(f"警告：跳過無法解析的行: {line.strip()}")
                        except Exception as e:
                            print(f"警告：處理標註記錄時出錯: {e}")

                print(f"從文件載入 {loaded_count} 條記錄的標註/緩存。")
                # TreeView will be populated/updated by _show_frame and _poll_queue

            except IOError as e:
                print(f"警告：無法讀取標註文件 {load_path}: {e}")
            except Exception as e:
                print(f"警告：載入標註時發生未知錯誤: {e}")
        else:
            print("未找到現有標註文件。")

    def _get_save_path(self, suffix=".jsonl") -> Optional[Path]:
        """Determines the path for saving/loading annotations based on video path."""
        if not self.video_file_path:
            return None
        # Save in a subdirectory named after the video file, inside 'data' or script dir
        video_dir = self.video_file_path.parent
        video_name = self.video_file_path.stem
        # Try to create a 'results' subdirectory relative to the video
        save_dir = video_dir / f"{video_name}_annotations"
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            # Use ROI coords in filename for uniqueness if multiple ROIs are analyzed later
            roi_str = f"roi_{self.roi_coords[0]}_{self.roi_coords[1]}_{self.roi_coords[2]}_{self.roi_coords[3]}"
            save_filename = f"{video_name}_{roi_str}{suffix}"
            return save_dir / save_filename
        except OSError as e:
            print(f"警告：無法創建儲存目錄 {save_dir}: {e}")
            # Fallback to saving alongside the video file
            return video_dir / f"{video_name}_annotations{suffix}"

    def _on_close(self):
        """Handles window closing: stops threads and saves intermediate results."""
        print("關閉應用程式...")
        self.stop_event.set() # Signal threads to stop

        # Give threads a moment to stop (optional, adjust time as needed)
        # Consider using thread.join() for cleaner shutdown, but might hang UI
        # self.after(500, self._finalize_close) # Example: Delay final close
        self._finalize_close() # Close directly for now

    def _finalize_close(self):
        """Performs the final steps of closing after signaling threads."""
         # Optional: Ask user if they want to save before closing
        save_path = self._get_save_path(".jsonl")
        if self.video_file_path and (self.annotations or self.ocr_cache): # Check if there's data
             if messagebox.askyesno("儲存進度?", "是否在關閉前儲存當前標註進度?"):
                  if save_path:
                      self._save_to_file(save_path) # Save only JSONL, not ROIs again
                  else:
                      messagebox.showwarning("無法儲存", "無法確定儲存路徑。")

        # Release video captures
        print("釋放 VideoCapture 資源...")
        for cap in (self.cap_ui, self.cap_detect, self.cap_ocr):
            if cap:
                try:
                    cap.release()
                except Exception as e:
                    print(f"關閉時釋放 VideoCapture 出錯: {e}")

        # Destroy the main window
        print("銷毀主窗口...")
        self.master.destroy()
        print("應用程式已關閉。")
    def _poll_queue(self):
        try:
            while True:
                msg = self.result_queue.get_nowait()

                if msg[0] == "analysis":
                    _, f_idx, content, changed = msg
                    iid = f"{f_idx}"
                    if not self.tree.exists(iid):
                        self.tree.insert("", "end", iid=iid,
                                         values=(f_idx, content))
                    else:
                        self.tree.set(iid, "content", content)
                    if changed:
                        self.tree.item(iid, tags=("changed",))

                elif msg[0] == "progress":
                    _, cur, total = msg
                    self.progress_var.set(cur)
                    self.lbl_prog.config(text=f"進度: {cur}/{total}")

                elif msg[0] == "done":
                    self._update_status_bar("分析完成")

                self.result_queue.task_done()
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _start_analysis(self):
        if not self.cap_detect or not self.cap_detect.isOpened():
            messagebox.showinfo("提示", "請先載入影片再開始分析。")
            return

        self.analysis_start_idx = int(self.slider_var.get())

        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showinfo("提示", "分析已在進行中。")
            return

        self.stop_event.clear()
        self.analysis_thread = threading.Thread(
            target=self._analysis_worker,
            args=(self.analysis_start_idx,),
            daemon=True
        )
        self.analysis_thread.start()

        # 進度條
        self.progressbar.configure(maximum=self.total_frames-1)
        self.progress_var.set(self.analysis_start_idx)
        self.lbl_prog.config(
            text=f"進度: {self.analysis_start_idx}/{self.total_frames-1}"
        )
        self._update_status_bar("開始分析…")

    def _on_goto_frame(self, event=None):
        try:
            idx = int(self.goto_var.get())
        except (ValueError, TypeError):
            return
        self._show_frame(idx)

    def _step_frame(self, delta: int):
        if self.total_frames == 0:
            return
        idx = max(0, min(self.total_frames-1,
                         self.current_frame_idx + delta))
        self._show_frame(idx)

    def _easyocr_predict(self, pil_img: Image.Image) -> str:
        result = self.ocr_iface.predict(pil_img)
        return result if result else "〈未識別〉"

    def _get_roi_dir(self) -> Path:
        """
        回傳 <cwd>/data/<影片檔名>/region2 目錄並確保存在
        """
        video_name = self.video_file_path.stem
        roi_dir = Path.cwd() / "data" / video_name / "region2"
        roi_dir.mkdir(parents=True, exist_ok=True)
        return roi_dir

    def _update_status_bar(self, text: str):
        if hasattr(self, "status_var"):
            self.status_var.set(text)
            # 立即刷新，避免 UI 卡住
            self.lbl_status.update_idletasks()

if __name__ == "__main__":
    # Fix for potential blurry UI on high DPI Windows displays
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except ImportError: # Not Windows or ctypes not available
        pass
    except AttributeError: # Function might not exist on older Windows
        try:
             windll.user32.SetProcessDPIAware()
        except: # Ignore if this also fails
             pass

    root = tk.Tk()
    app = VideoAnnotator(root)
    root.mainloop()
