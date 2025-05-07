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
from tkinter import TclError
import csv
import easyocr
import torch
import shutil
import tempfile
import traceback

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
                allowlist="0123456789-",
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

        # 創建 TreeView 後添加點擊事件綁定
        self.tree.bind('<<TreeviewSelect>>', self._on_treeview_select)

        # 為 TreeView 設置雙擊事件（可以用於編輯）
        self.tree.bind("<Double-1>", self._on_edit_annotation)

        self.tree.bind("<Return>", self._on_edit_annotation)
        
        # 設置右鍵選單
        self._setup_treeview_context_menu()
        
        # 添加一個變數來追踪是否有未保存的更改
        self.changes_made = False

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

            # --- 新增：嘗試載入現有標註 ---
            self._load_annotations() # 會自動填充 TreeView 並跳轉

            # 如果 _load_annotations 沒有跳轉 (例如沒有標註檔)，則顯示第一幀
            if not self.tree.get_children():
                 self._show_frame(0)
            # ------------------------------

            self._update_status_bar(f"影片 '{self.video_file_path.name}' 已載入")
            # 移除或註解掉任何自動開始分析的舊程式碼
            # self._start_analysis() # <--- 確保這裡沒有自動開始

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
                    allowlist="0123456789-",
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
            allowlist="0123456789-",
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

    def _fill_and_get_records(self, tree_items: list) -> list:
        """
        接收 TreeView 的項目列表，填充缺失的幀，並返回包含所有連續幀記錄的完整列表。
        """
        if not tree_items:
            return []
        if not self.video_file_path: # 需要影片名稱來建構 image 路徑
            print("錯誤：無法填充記錄，因為影片路徑未設定。")
            return []

        video_name = self.video_file_path.stem
        records_by_frame = {}
        min_frame = float('inf')
        max_frame = float('-inf')

        # 1. 從 TreeView 項目中提取數據
        print("正在從 TreeView 提取標註...")
        for iid in tree_items:
            item_data = self.tree.item(iid)
            values = item_data.get("values", [])
            if not values or len(values) < 2:
                print(f"警告：跳過 TreeView 中格式不符的項目 {iid}")
                continue
            try:
                frame_idx = int(values[0])
                # 直接儲存 response 字串
                response_text = str(values[1]) if values[1] is not None else ""
                records_by_frame[frame_idx] = response_text
                min_frame = min(min_frame, frame_idx)
                max_frame = max(max_frame, frame_idx)
            except (ValueError, IndexError) as e:
                print(f"警告：解析 TreeView 項目 {iid} 時出錯 ({e})，已跳過")
                continue

        if not records_by_frame:
            print("資訊：TreeView 中沒有有效的標註可供填充。")
            return []

        # 檢查是否有有效的幀範圍
        if min_frame == float('inf') or max_frame == float('-inf'):
             print("錯誤：無法從 TreeView 確定有效的幀範圍。")
             return []

        print(f"從 TreeView 提取記錄範圍：幀 {min_frame} 到 {max_frame}")

        filled_records = []
        last_known_response = "" # 初始值

        print(f"正在填充從 {min_frame} 到 {max_frame} 的所有幀...")
        # 確保按幀號順序處理原始記錄
        sorted_original_frames = sorted(records_by_frame.keys())
        original_record_idx = 0

        for current_frame in range(min_frame, max_frame + 1):
            # 檢查原始記錄中是否有當前幀
            if original_record_idx < len(sorted_original_frames) and sorted_original_frames[original_record_idx] == current_frame:
                # 使用原始記錄 (來自 TreeView)
                response = records_by_frame[current_frame]
                last_known_response = response # 更新最後已知的 response
                region_name = getattr(self, "region_name", "region2")  # 預設為 "region2"
                image_path = f"{video_name}/{region_name}/frame_{current_frame}.png"
                record = {
                    "query": "<image>",
                    "response": response,
                    "images": image_path
                }
                filled_records.append(record)
                original_record_idx += 1
            else:
                # 填充缺失的幀
                image_path = f"{video_name}/region2/frame_{current_frame}.png"
                new_record = {
                    "query": "<image>",
                    "response": last_known_response, # 使用上一個已知幀的 response
                    "images": image_path
                }
                filled_records.append(new_record)

        print(f"填充完成，總共 {len(filled_records)} 筆記錄。")
        return filled_records

    def _save_annotations(self):
        """將 TreeView 中的資料填充後儲存為包含所有連續幀的 JSONL 格式"""
        # 同時儲存變化幀列表
        self._save_change_frames()

        save_path = self._get_save_path()
        if not save_path:
            print("錯誤：無法獲取儲存路徑。")
            self._update_status_bar("儲存失敗：無法獲取路徑")
            return

        if not self.video_file_path:
            print("錯誤：影片路徑未設定，無法儲存。")
            self._update_status_bar("儲存失敗：未載入影片")
            return

        tree_items = self.tree.get_children()
        if not tree_items:
            print("資訊：TreeView 中沒有標註可儲存。")
            # 儲存一個空的檔案
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True) # 確保目錄存在
                with open(save_path, "w", encoding="utf-8") as f:
                    pass # 寫入空內容
                self._update_status_bar(f"已儲存空的標註檔至 {save_path.name}")
                print(f"已儲存空的標註檔至：{save_path}")
            except IOError as e:
                print(f"錯誤：儲存空的標註檔失敗：{e}")
                self._update_status_bar(f"儲存失敗：{e}")
            return

        # --- 呼叫填充函數 ---
        filled_records = self._fill_and_get_records(tree_items)
        # --------------------

        if not filled_records:
             print("錯誤：填充記錄失敗或無有效記錄，無法儲存。")
             self._update_status_bar("儲存失敗：填充記錄錯誤")
             return

        # --- 使用臨時檔案安全地寫入填充後的記錄 ---
        temp_file_path = None
        try:
            # 在與目標文件相同的目錄下創建臨時文件
            save_path.parent.mkdir(parents=True, exist_ok=True) # 再次確保目錄存在
            temp_fd, temp_path_str = tempfile.mkstemp(dir=save_path.parent, prefix=save_path.stem + '_', suffix='.tmp')
            temp_file_path = Path(temp_path_str)

            print(f"正在寫入臨時文件：{temp_file_path}")
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as outfile:
                for record in filled_records:
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 用臨時文件替換原始文件
            print(f"正在用臨時文件覆寫目標文件：{save_path}")
            shutil.move(str(temp_file_path), str(save_path)) # 原子性替換
            print("文件覆寫成功。")
            self._update_status_bar(f"標註已填充並儲存至 {save_path.name} ({len(filled_records)} 幀)")

        except IOError as e:
            print(f"錯誤：寫入臨時文件或覆寫文件時出錯: {e}")
            self._update_status_bar(f"儲存失敗：{e}")
            # 出錯時嘗試刪除臨時文件
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                    print(f"已刪除臨時文件：{temp_file_path}")
                except OSError as unlink_e:
                    print(f"錯誤：刪除臨時文件失敗: {unlink_e}")
        except Exception as e:
            print(f"儲存標註時發生未知錯誤: {e}")
            self._update_status_bar(f"儲存時發生錯誤: {e}")
            if temp_file_path and temp_file_path.exists():
                 try:
                     temp_file_path.unlink()
                 except OSError:
                     pass # 可能已被移動，忽略錯誤
        finally:
            # 確保即使移動成功，也不會意外地保留對臨時文件的引用
             if temp_file_path and temp_file_path.exists():
                 try:
                     temp_file_path.unlink()
                 except OSError:
                     pass # 可能已被移動，忽略錯誤

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
        """從 JSONL 檔案載入標註並更新 TreeView"""
        load_path = self._get_save_path() # 會得到 data/<影片名>/region2.jsonl
        if not load_path or not load_path.exists():
            print(f"資訊：未找到現有標註文件。")
            return

        print(f"檢測到標註文件，正在載入: {load_path}")
        try:
            # 清空 TreeView
            for item in self.tree.get_children():
                self.tree.delete(item)

            records = []  # 儲存所有記錄
            prev_response = None
            loaded_count = 0
            change_frames = []  # 用於存儲幀內容發生變化的幀號
            
            with open(load_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        response_text = record.get("response", "")
                        image_path = record.get("images", "")

                        # 從 image_path 解析 frame_idx
                        filename = Path(image_path).name
                        if not filename.startswith("frame_") or not filename.endswith(".png"):
                            print(f"警告：跳過無效的 image_path: {image_path}")
                            continue
                        
                        frame_idx_str = filename[len("frame_"):-len(".png")]
                        try:
                            frame_idx = int(frame_idx_str)
                        except ValueError:
                            print(f"警告：無法從 {filename} 解析出有效的幀號")
                            continue
                        
                        # 記錄所有讀取到的標註
                        records.append((frame_idx, response_text))
                        
                        # 檢查這一幀的內容是否與前一幀不同
                        if prev_response != response_text:
                            change_frames.append(frame_idx)
                            prev_response = response_text
                        
                        loaded_count += 1
                    except json.JSONDecodeError:
                        print(f"警告：JSON 解析錯誤，跳過此行")
                        continue
                    except (ValueError, KeyError, AttributeError) as e:
                        print(f"警告：處理記錄時出錯 ({e})，跳過此行")
                        continue

                # 對於很大的檔案，我們只將 change_frames 添加到 TreeView
                print(f"從檔案中讀取了 {loaded_count} 條記錄，其中 {len(change_frames)} 幀內容有變化")
                
                # 如果記錄數量很大，只加載變化的幀到 TreeView
                for frame_idx in change_frames:
                    # 找到對應的 response
                    for r_idx, r_text in records:
                        if r_idx == frame_idx:
                            self.tree.insert("", "end", values=(frame_idx, r_text))
                            break
                
                # 更新 TreeView 排序
                self._sort_tree_by_frame()
                
                # 儲存變化幀列表到 region2_change.jsonl
                self._save_change_frames(change_frames)
                
                self._update_status_bar(f"已從 {load_path.name} 載入 {len(change_frames)} 個變化幀 (共 {loaded_count} 條記錄)")
                print(f"從文件載入 {len(change_frames)} 條變化幀的標註/緩存。")

                # 跳轉到第一筆記錄
                if change_frames:
                    first_frame = change_frames[0]
                    self.current_frame_idx = first_frame - 1  # 調整為前一幀
                    self._update_slider_position(first_frame)
                    self._load_and_show_frame_by_number(first_frame)  # 顯示第一個變化幀
            
            # 更新緩存
            self.annotations = {frame_idx: resp for frame_idx, resp in records}
            
        except IOError as e:
            print(f"錯誤：讀取標註文件時發生 IO 錯誤: {e}")
            self._update_status_bar(f"載入標註失敗: {e}")
        except Exception as e:
            print(f"載入標註時發生未知錯誤: {e}")
            self._update_status_bar(f"載入標註時出錯: {e}")

    def _get_save_path(self, suffix=None):
        """
        獲取標註檔案的儲存路徑
        Args:
            suffix: 可選的文件後綴，如 ".jsonl"。如果提供，則附加到文件名後。
        Returns:
            Path: data/<影片名>/region2.jsonl 或自定義路徑
        """
        if not self.video_file_path:
            return None
        
        video_name = self.video_file_path.stem
        save_dir = Path("data") / video_name
        
        # 使用類的屬性或配置來獲取區域名稱，而不是硬編碼
        region_name = getattr(self, "region_name", "region2")  # 預設為 "region2"
        
        if suffix:
            return save_dir / f"{region_name}{suffix}"
        else:
            return save_dir / f"{region_name}.jsonl"


    def _save_change_frames(self, change_frames=None):
        """將發生內容變化的幀號儲存到 region_change.jsonl 文件中"""
        if not self.video_file_path:
            return
        
        # 如果沒有提供 change_frames，則從 TreeView 中提取
        if change_frames is None:
            change_frames = []
            for iid in self.tree.get_children():
                try:
                    frame_idx = int(self.tree.item(iid)["values"][0])
                    change_frames.append(frame_idx)
                except (IndexError, ValueError):
                    continue
        
        if not change_frames:
            print("沒有檢測到變化幀，不創建 change frames 文件")
            return
        
        # 獲取區域名稱，使用 getattr 避免硬編碼
        region_name = getattr(self, "region_name", "region2")  # 預設為 "region2"
        
        # 構建變化幀文件路徑
        video_name = self.video_file_path.stem
        save_dir = Path("data") / video_name
        change_path = save_dir / f"{region_name}_change.jsonl"
        
        try:
            # 創建目錄（如果不存在）
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 使用臨時文件進行安全寫入
            temp_fd, temp_path_str = tempfile.mkstemp(dir=save_dir, prefix=f"{region_name}_change_", suffix=".tmp")
            temp_file_path = Path(temp_path_str)
            
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                # 按照幀號排序
                for frame in sorted(change_frames):
                    # 使用簡單的格式，只存儲幀號
                    f.write(json.dumps({"frame": frame}) + "\n")
            
            # 用臨時文件安全地替換目標文件
            shutil.move(str(temp_file_path), str(change_path))
            print(f"已儲存 {len(change_frames)} 個變化幀到 {change_path}")
        except Exception as e:
            print(f"儲存變化幀列表時出錯: {e}")
            if 'temp_file_path' in locals() and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except:
                    pass

    def _sort_tree_by_frame(self):
        """按幀號排序 TreeView 的項目"""
        items = [(int(self.tree.set(iid, "frame")), iid) for iid in self.tree.get_children()]
        items.sort()
        for index, (_, iid) in enumerate(items):
            self.tree.move(iid, "", index)

    def _on_close(self):
        """處理窗口關閉：停止線程並儲存標註進度。"""
        print("關閉應用程式...")
        self.stop_event.set() # 通知線程停止

        # --- 新增：儲存當前 TreeView 內容 ---
        # 檢查是否有影片載入且 TreeView 中有內容
        if self.video_file_path and self.tree.get_children():
            # --- 選項 A：總是自動儲存 ---
            print("正在自動儲存標註進度...")
            self._save_annotations()
            # --- 選項 B：詢問使用者是否儲存 (取消註解以使用) ---
            # if messagebox.askyesno("儲存進度?", "是否在關閉前儲存當前標註進度?"):
            #     print("正在儲存標註進度...")
            #     self._save_annotations()
            # else:
            #     print("使用者選擇不儲存進度。")
        else:
            print("無需儲存標註 (未載入影片或無標註內容)。")
        # ---------------------------------

        # 等待線程結束 (可選，但建議加入以確保資源完全釋放)
        if self.analysis_thread and self.analysis_thread.is_alive():
            print("等待分析線程結束...")
            self.analysis_thread.join(timeout=1.0) # 等待最多 1 秒
            if self.analysis_thread.is_alive():
                print("警告：分析線程未在預期時間內結束。")

        # 釋放影片資源 (保持不變)
        print("釋放 VideoCapture 資源...")
        for cap in (self.cap_ui, self.cap_detect, self.cap_ocr):
            if cap:
                try:
                    cap.release()
                except Exception as e:
                    print(f"關閉時釋放 VideoCapture 出錯: {e}")

        # 銷毀主窗口 (保持不變)
        print("銷毀主窗口...")
        self.master.destroy()
        print("應用程式已關閉。")
        print("關閉應用程式...")
        self.stop_event.set() # Signal threads to stop

        # Give threads a moment to stop (optional, adjust time as needed)
        # Consider using thread.join() for cleaner shutdown, but might hang UI
        # self.after(500, self._finalize_close) # Example: Delay final close
        self._finalize_close() # Close directly for now

    def _finalize_close(self):
        # """Performs the final steps of closing after signaling threads."""
        #  # Optional: Ask user if they want to save before closing
        # save_path = self._get_save_path(".jsonl")
        # if self.video_file_path and (self.annotations or self.ocr_cache): # Check if there's data
        #      if messagebox.askyesno("儲存進度?", "是否在關閉前儲存當前標註進度?"):
        #           if save_path:
        #               self._save_to_file(save_path) # Save only JSONL, not ROIs again
        #           else:
        #               messagebox.showwarning("無法儲存", "無法確定儲存路徑。")

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

        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showinfo("提示", "分析已在進行中。")
            return

        # --- 修改：決定起始幀號 ---
        last_analyzed_frame = -1
        if self.tree.get_children(): # 檢查 TreeView 是否有內容
            try:
                # 獲取 TreeView 中所有項目的幀號並找到最大值
                all_frames = [int(self.tree.set(iid, "frame")) for iid in self.tree.get_children()]
                if all_frames:
                    last_analyzed_frame = max(all_frames)
            except ValueError:
                print("警告：無法解析 TreeView 中的幀號，將從頭開始分析。")
                last_analyzed_frame = -1 # 出錯則重置

        # 如果有上次分析的記錄，從下一幀開始；否則從滑桿當前位置開始
        self.analysis_start_idx = last_analyzed_frame + 1 if last_analyzed_frame >= 0 else int(self.slider_var.get())

        # 確保起始幀號不超過總幀數
        if self.analysis_start_idx >= self.total_frames:
             messagebox.showinfo("提示", "所有幀似乎都已分析過。")
             self._update_status_bar("所有幀已分析")
             return
        # --------------------------

        self.stop_event.clear()
        self.analysis_thread = threading.Thread(
            target=self._analysis_worker,
            args=(self.analysis_start_idx,), # 使用計算出的起始幀號
            daemon=True
        )
        self.analysis_thread.start()

        # 進度條設定 (保持不變)
        self.progressbar.configure(maximum=self.total_frames-1)
        self.progress_var.set(self.analysis_start_idx)
        self.lbl_prog.config(
            text=f"進度: {self.analysis_start_idx}/{self.total_frames-1}"
        )
        self._update_status_bar(f"從幀 {self.analysis_start_idx} 開始分析…")
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

    # 1. 修改開啟影片的函數，加入清理舊資料和檢測進度的邏輯
    def _open_video(self):
        """開啟並載入影片文件"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        video_path = filedialog.askopenfilename(filetypes=filetypes, title="選擇影片文件")
        if not video_path:
            return False
        
        # 清理舊影片的資料和UI
        self._clear_previous_video_data()
        
        # 載入新影片
        self.video_file_path = Path(video_path)
        self.video_title = self.video_file_path.stem
        
        # 設置影片輸入
        try:
            success = self._setup_video_input(self.video_file_path)
            if not success:
                self._update_status_bar("影片載入失敗")
                return False
            
            self._update_status_bar(f"已載入: {self.video_title} ({self.total_frames} 幀)")
            
            # 載入現有標註（如果有）以及變化幀列表
            self._load_existing_data()
            
            # 檢查是否有分析進度，並自動跳轉
            self._check_and_jump_to_analysis_position()
            
            return True
        except Exception as e:
            messagebox.showerror("載入失敗", f"影片載入出錯:\n{e}")
            print(f"載入影片時發生錯誤: {e}")
            traceback.print_exc()  # 打印詳細錯誤堆疊
            self._update_status_bar("影片載入失敗")
            return False

    # 2. 新增函數用於清理上一個影片的資料和UI
    def _clear_previous_video_data(self):
        """清理上一個影片的資料和UI"""
        # 停止可能正在運行的處理線程
        if hasattr(self, 'tmad_detector') and self.tmad_detector:
            self.tmad_detector.stop()
        if hasattr(self, 'ocr_iface') and self.ocr_iface:
            self.ocr_iface.stop()
        
        # 清空所有緩存
        self.annotations = {}
        self.ocr_cache = {}
        self.change_cache = {}
        self.roi_image_cache = {}
        
        # 清空隊列
        while not self.detect_queue.empty():
            try:
                self.detect_queue.get_nowait()
            except:
                pass
        
        # 重置 UI 元素
        # 清空 TreeView
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 重置影像顯示
        self.lbl_img.config(image="")
        self.lbl_roi.config(image="")
        
        # 重置影片相關變數
        self.current_frame_idx = 0
        self.total_frames = 0
        self.video_file_path = None
        self.video_title = ""
        
        # 重置狀態和進度
        self.lbl_status.config(text="就緒")
        self.lbl_frame_num.config(text="幀: 0 / 0")
        self.slider_var.set(0)
        
        # 更新窗口標題
        self.title(f"影片標註工具")
        
        print("已清理上一個影片的資料和 UI")

    # 3. 新增函數用於檢查分析進度並自動跳轉
    def _check_and_jump_to_analysis_position(self):
        """檢查是否有分析進度，並自動跳轉到應該繼續的位置"""
        if not self.video_file_path or not hasattr(self, 'tree'):
            return
        
        # 檢查是否有任何標註
        if not self.tree.get_children():
            print("沒有檢測到現有標註，將從第一幀開始")
            self._load_and_show_frame_by_number(0)
            return
        
        # 從 TreeView 獲取所有標註的幀號
        frame_indices = []
        for item in self.tree.get_children():
            try:
                frame_idx = int(self.tree.item(item)["values"][0])
                frame_indices.append(frame_idx)
            except (IndexError, ValueError):
                continue
        
        if not frame_indices:
            print("無法從 TreeView 提取有效的幀號")
            self._load_and_show_frame_by_number(0)
            return
        
        # 找出最大的幀號，即當前分析進度
        latest_frame = max(frame_indices)
        
        # 檢查是否已分析到最後一幀
        if latest_frame >= self.total_frames - 1:
            print(f"已完成所有幀的分析 (最後幀: {latest_frame})")
            # 可以選擇跳到第一幀或最後一幀
            self._load_and_show_frame_by_number(0)  # 跳到第一幀
            return
        
        # 跳轉到最後一個標註的幀之後，準備繼續分析
        next_frame = latest_frame + 1
        print(f"檢測到分析進度，上次處理到幀 {latest_frame}，將跳轉到幀 {next_frame} 繼續分析")
        
        # 更新滑塊位置並載入該幀
        self._update_slider_position(next_frame)
        self._load_and_show_frame_by_number(next_frame)
        
        # 顯示提示訊息
        self._update_status_bar(f"繼續從幀 {next_frame} 開始分析 (上次處理到幀 {latest_frame})")

    # 4. 如果需要，添加一個輔助方法用於更新滑塊位置
    def _update_slider_position(self, frame_idx):
        """更新滑塊位置到指定幀"""
        if hasattr(self, 'slider_var'):
            self.slider_var.set(frame_idx)

    # 新增處理 TreeView 選擇事件的方法
    def _on_treeview_select(self, event=None):
        """當用戶點擊 TreeView 中的項目時，跳轉到對應的幀"""
        # 獲取當前選中的項目
        selected_items = self.tree.selection()
        if not selected_items:
            return  # 沒有選中項目
        
        # 使用第一個選中的項目（通常只會選中一個）
        selected_id = selected_items[0]
        
        try:
            # 獲取幀號
            frame_idx = int(self.tree.set(selected_id, "frame"))
            
            # 如果當前已經在該幀，則不需要重新載入
            if frame_idx == self.current_frame_idx:
                return
            
            # 記錄目前跳轉前的位置（方便稍後返回）
            # self.previous_frame_idx = self.current_frame_idx
            
            # 跳轉到該幀
            self._show_frame(frame_idx)
            
            # 更新狀態欄
            self._update_status_bar(f"已跳轉到幀 {frame_idx}")
            
            # 可以在此處高亮顯示 ROI 區域或標記 OCR 文本位置
            
        except (ValueError, KeyError, TclError) as e:
            print(f"跳轉到所選幀時出錯: {e}")
            
        # 確保 TreeView 保持焦點或選中狀態（避免選擇被清除）
        self.tree.focus(selected_id)
        self.tree.selection_set(selected_id)

    # 在適當位置添加這些輔助方法

    def _on_edit_annotation(self, event=None):
        """打開編輯當前選中項目的標註內容的對話框，並將其定位在適當位置"""
        try:
            # 獲取選中的項目
            selected_items = self.tree.selection()
            if not selected_items:
                print("沒有選中任何項目，無法編輯")
                messagebox.showinfo("提示", "請先選擇要編輯的項目")
                return
                
            selected_id = selected_items[0]
            
            # 獲取當前項目的值
            try:
                frame_idx = int(self.tree.item(selected_id)["values"][0])
                current_response = self.tree.item(selected_id)["values"][1]
                if not current_response:
                    current_response = ""  # 確保是字串，避免 None 值
                print(f"正在編輯幀 {frame_idx} 的標註，當前內容：{current_response}")
            except (IndexError, ValueError) as e:
                print(f"獲取項目值時出錯: {e}")
                current_response = ""
                try:
                    frame_idx = int(self.tree.set(selected_id, "frame"))
                    current_response = self.tree.set(selected_id, "response") or ""
                    print(f"透過 set 方法獲取值：幀 {frame_idx}, 內容: {current_response}")
                except Exception as e2:
                    print(f"透過 set 方法獲取值也失敗: {e2}")
                    messagebox.showerror("錯誤", f"無法獲取所選項目的內容: {e2}")
                    return
            
            # 創建編輯對話框
            edit_dialog = tk.Toplevel(self.master)
            edit_dialog.title(f"編輯幀 {frame_idx} 的標註")
            edit_dialog.geometry("400x150")  # 稍微縮小高度，因為通常只需要輸入簡短內容
            edit_dialog.resizable(True, True)
            
            # 確保對話框在主窗口之上
            edit_dialog.transient(self.master)
            edit_dialog.grab_set()
            
            # 添加文本輸入框
            lbl = tk.Label(edit_dialog, text=f"幀 {frame_idx} 的標註內容:")
            lbl.pack(pady=(10, 5), padx=10, anchor="w")
            
            # 使用 Entry 而不是 Text（因為我們主要處理單行數字）
            txt_var = tk.StringVar(value=current_response)
            txt_edit = ttk.Entry(edit_dialog, textvariable=txt_var, width=20, font=("Arial", 12))
            txt_edit.pack(fill=tk.X, expand=True, padx=10, pady=5)
            
            # 添加按鈕框架
            btn_frame = tk.Frame(edit_dialog)
            btn_frame.pack(fill=tk.X, padx=10, pady=10)
            
            # 定義保存函數
            def save_edit():
                try:
                    new_text = txt_var.get().strip()
                    print(f"保存編輯，新內容: {new_text}")
                    
                    # 更新 TreeView
                    try:
                        self.tree.item(selected_id, values=(frame_idx, new_text))
                        print(f"成功更新 TreeView 項目")
                    except Exception as e:
                        print(f"更新 TreeView 項目失敗: {e}")
                        # 備選方法
                        try:
                            self.tree.set(selected_id, "content", new_text)
                            print(f"通過 set 方法更新 TreeView 成功")
                        except Exception as e2:
                            print(f"通過 set 方法更新 TreeView 也失敗: {e2}")
                            raise
                    
                    # 同時更新記憶體中的標註
                    self.annotations[frame_idx] = new_text
                    print(f"已更新記憶體中的標註數據")
                    
                    # 標記已修改，需要保存
                    self.changes_made = True
                    
                    # 關閉對話框
                    edit_dialog.destroy()
                    
                    # 保持表格焦點，便於繼續用方向鍵導航
                    self.tree.focus_set()
                    
                    # 可選：自動選擇下一項，方便連續編輯
                    try:
                        next_item = self._get_next_item(selected_id)
                        if next_item:
                            self.tree.selection_set(next_item)
                            self.tree.see(next_item)  # 確保下一項可見
                    except Exception as e:
                        print(f"自動選擇下一項失敗: {e}")
                        # 保持當前選擇
                        self.tree.selection_set(selected_id)
                    
                except Exception as e:
                    print(f"保存編輯時出錯: {e}")
                    messagebox.showerror("錯誤", f"保存編輯時出錯: {e}")
                    # 確保對話框關閉，避免卡住界面
                    edit_dialog.destroy()
            
            # 綁定按鈕
            btn_save = tk.Button(btn_frame, text="保存", command=save_edit)
            btn_save.pack(side=tk.RIGHT, padx=5)
            
            # 取消按鈕
            btn_cancel = tk.Button(btn_frame, text="取消", command=edit_dialog.destroy)
            btn_cancel.pack(side=tk.RIGHT, padx=5)
            
            # 綁定 Enter 鍵到保存動作
            txt_edit.bind("<Return>", lambda e: save_edit())
            
            # 設置焦點
            txt_edit.focus_set()
            # 游標定位到文字末尾
            txt_edit.icursor(tk.END)
            
            # 將對話框定位到適當位置
            if event and hasattr(event, 'x_root') and hasattr(event, 'y_root'):
                # 如果有滑鼠事件，定位到滑鼠位置
                x = event.x_root
                y = event.y_root
                
                # 確保不會超出螢幕
                dialog_width = 400
                dialog_height = 150
                screen_width = edit_dialog.winfo_screenwidth()
                screen_height = edit_dialog.winfo_screenheight()
                
                # 調整 x 坐標，確保對話框不會超出右側螢幕邊緣
                if x + dialog_width > screen_width:
                    x = screen_width - dialog_width
                
                # 調整 y 坐標，確保對話框不會超出底部螢幕邊緣
                if y + dialog_height > screen_height:
                    y = screen_height - dialog_height
                
                edit_dialog.geometry(f"+{x}+{y}")
            else:
                # 如果是鍵盤事件或其他方式觸發，將對話框定位到表格項目附近
                try:
                    # 獲取當前選中項目的坐標
                    item_id = selected_id
                    tree_x, tree_y, _, _ = self.tree.bbox(item_id, "content")
                    
                    # 將對話框定位到項目右側
                    abs_x = self.tree.winfo_rootx() + tree_x + 50  # 偏移一點，避免遮擋
                    abs_y = self.tree.winfo_rooty() + tree_y
                    
                    edit_dialog.geometry(f"+{abs_x}+{abs_y}")
                except Exception as e:
                    print(f"定位對話框到項目位置失敗: {e}")
                    # 使用默認位置 - TreeView 中心
                    tree_x = self.tree.winfo_rootx() + self.tree.winfo_width() // 2
                    tree_y = self.tree.winfo_rooty() + self.tree.winfo_height() // 2
                    edit_dialog.geometry(f"+{tree_x-200}+{tree_y-75}")
        
        except Exception as e:
            print(f"創建編輯對話框時發生未知錯誤: {e}")
            messagebox.showerror("錯誤", f"編輯標註時發生錯誤: {e}")

    # 在 TreeView 上添加右鍵選單功能
    def _setup_treeview_context_menu(self):
        """為 TreeView 設置右鍵選單"""
        # 創建右鍵選單
        self.treeview_menu = tk.Menu(self.tree, tearoff=0)
        self.treeview_menu.add_command(label="編輯標註", command=self._on_edit_annotation)
        self.treeview_menu.add_command(label="跳轉到此幀", command=lambda: self._on_treeview_select(None))
        self.treeview_menu.add_separator()
        self.treeview_menu.add_command(label="刪除標註", command=self._on_delete_annotation)
        
        # 綁定右鍵事件
        self.tree.bind("<Button-3>", self._show_treeview_context_menu)
        
    def _show_treeview_context_menu(self, event):
        """顯示 TreeView 的右鍵選單"""
        # 獲取點擊位置對應的項目
        item = self.tree.identify_row(event.y)
        if item:
            # 先選中點擊的項目
            self.tree.selection_set(item)
            # 在點擊位置顯示選單
            self.treeview_menu.post(event.x_root, event.y_root)

    def _on_delete_annotation(self):
        """刪除當前選中的標註項目"""
        selected_items = self.tree.selection()
        if not selected_items:
            return
            
        if messagebox.askyesno("確認", "確定要刪除所選標註嗎？這個操作無法撤銷。"):
            for item_id in selected_items:
                try:
                    frame_idx = int(self.tree.set(item_id, "frame"))
                    # 從標註字典中刪除
                    if frame_idx in self.annotations:
                        del self.annotations[frame_idx]
                    # 從 TreeView 中刪除
                    self.tree.delete(item_id)
                    # 標記已修改
                    self.changes_made = True
                except (ValueError, KeyError, TclError) as e:
                    print(f"刪除標註時出錯: {e}")
            self._update_status_bar("已刪除所選標註")

    # 添加一個新的綜合函數來載入所有現有數據
    def _load_existing_data(self):
        """載入當前視頻的所有現有數據，包括標註和變化幀列表"""
        try:
            print(f"正在載入現有數據: {self.video_title}")
            
            # 1. 確定數據目錄
            video_name = self.video_file_path.stem
            data_dir = Path("data") / video_name
            self.dataset_dir = video_name  # 保存用於構建相對路徑
            
            if not data_dir.exists():
                print(f"數據目錄不存在: {data_dir}，可能是新影片")
                return
                
            # 2. 設置區域名稱和 ROI 名稱
            self.region_name = getattr(self, "region_name", "region2")
            self.roi_name = self.region_name  # 保持一致的命名
            
            # 3. 構建標註文件路徑
            self.jsonl_file_path = data_dir / f"{self.region_name}.jsonl"
            print(f"標註文件路徑: {self.jsonl_file_path} (存在: {self.jsonl_file_path.exists()})")
            
            # 4. 構建變化幀文件路徑
            self.change_frames_path = data_dir / f"{self.region_name}_change.jsonl"
            print(f"變化幀文件路徑: {self.change_frames_path} (存在: {self.change_frames_path.exists()})")
            
            # 5. 載入標註數據
            if self.jsonl_file_path.exists():
                self.annotations = self._load_annotations_from_jsonl(self.jsonl_file_path)
                print(f"已載入 {len(self.annotations)} 個標註")
                
                # 將標註數據更新到表格
                self._update_annotations_treeview()
            else:
                print(f"標註文件不存在: {self.jsonl_file_path}")
                self.annotations = {}
            
            # 6. 載入變化幀列表
            if self.change_frames_path.exists():
                self.change_frames = self._load_change_frames(self.change_frames_path)
                print(f"已載入 {len(self.change_frames)} 個變化幀")
            else:
                print(f"變化幀文件不存在: {self.change_frames_path}")
                self.change_frames = set()
                
        except Exception as e:
            print(f"載入現有數據時出錯: {e}")
            traceback.print_exc()
            messagebox.showwarning("警告", f"載入現有數據時發生錯誤，部分數據可能無法正確顯示: {e}")

    # 更新將標註數據顯示到表格的方法
    def _update_annotations_treeview(self):
        """將標註數據更新到表格視圖"""
        try:
            # 清空現有表格內容
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # 按幀號排序顯示
            for frame_idx, value in sorted(self.annotations.items()):
                # 將標註添加到 TreeView
                self.tree.insert("", tk.END, values=(frame_idx, value))
                
            # 更新狀態欄
            self._update_status_bar(f"已載入 {len(self.annotations)} 個標註")
            
        except Exception as e:
            print(f"更新標註表格時出錯: {e}")
            traceback.print_exc()

    # 添加載入變化幀列表的方法
    def _load_change_frames(self, change_frames_path):
        """從 JSONL 文件載入變化幀列表"""
        change_frames = set()
        try:
            if not change_frames_path.exists():
                return change_frames
                
            with open(change_frames_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'frame' in data:
                            change_frames.add(data['frame'])
                    except json.JSONDecodeError:
                        continue
                        
            print(f"從 {change_frames_path} 載入了 {len(change_frames)} 個變化幀")
        except Exception as e:
            print(f"載入變化幀列表時出錯: {e}")
            traceback.print_exc()
            
        return change_frames

    # 添加輔助方法：獲取下一個表格項目
    def _get_next_item(self, current_item):
        """獲取表格中的下一個項目"""
        try:
            # 獲取所有項目
            all_items = self.tree.get_children()
            if not all_items:
                return None
                
            # 找到當前項目的索引
            current_index = all_items.index(current_item)
            
            # 如果不是最後一項，返回下一項
            if current_index < len(all_items) - 1:
                return all_items[current_index + 1]
            else:
                # 如果是最後一項，返回第一項或 None
                # return all_items[0]  # 循環到第一項
                return None  # 保持在最後一項
        except (ValueError, IndexError) as e:
            print(f"獲取下一個項目時出錯: {e}")
            return None

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
