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
from __future__ import annotations
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import threading
import queue
from typing import Optional
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time

import cv2
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib
import tkinter.font as tkFont # For bold font in Treeview
from tkinter import TclError
import csv
import easyocr
import torch
import shutil
import tempfile
import traceback
from skimage import exposure, filters, morphology
from models.OCR_interface import get_ocr_model
from components.test_binary_frame_change import add_test_button
from typing import List, Tuple
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


"""回傳 data/config/rois.json 路徑"""
get_roi_config_path = lambda : Path("data") / "config" / "rois.json"
# -------------------- 主GUI --------------------
class VideoAnnotator(tk.Frame):
    VID_W, VID_H = 800, 450
    # ROI = (1640, 445, 1836, 525) # Wird aus roi_dict geladen
    OCR_CONF_TH = 0.5 

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        master.title("Frame Annotation Tool (T-MAD Change Detection)")
        master.geometry("1350x750")
        master.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.pack(fill="both", expand=True)

        self.cap_ui: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.video_file_path: Optional[Path] = None
        self.video_title: str = ""
        self.current_frame_idx = 0
        self.roi_coords: Optional[tuple] = None
        self.original_vid_w = 0
        self.original_vid_h = 0
        self.roi_start_coords: Optional[tuple] = None

        # OCR 測試視窗相關屬性
        self.ocr_test_active = False
        self.ocr_test_window = None

        self.region_name = "region2"
        self.roi_dict: Dict[str, list] = {
            "region2": [1640, 445, 1836, 525],
        }
        self._load_roi_config()  # 先載入配置
        
        # 設定預設region（從已載入的配置中選擇，或使用第一個可用的）
        if self.roi_dict:
            self.region_name = list(self.roi_dict.keys())[0]  # 使用第一個可用區域
        else:
            # 如果沒有配置檔案，建立一個預設的 region2
            self.region_name = "region2"
            self.roi_dict[self.region_name] = [1640, 445, 1836, 525]
                    
        self.change_cache: Dict[int, bool] = {}
        self.ocr_cache: Dict[int, str] = {}
        self.annotations: Dict[int, str] = {}
        self.roi_image_cache: Dict[int, Image.Image] = {}

        self.current_analysis_cache: Dict[int, str] = {}

        self.hsv_s_threshold_var = tk.IntVar(value=30)
        self.gray_threshold_var = tk.IntVar(value=150)

        # result_queue 仍然需要，用於從背景執行緒向UI傳遞結果和進度
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.save_lock = threading.Lock()

        self.ocr_iface = get_ocr_model(
            model_type="easyocr",
            gpu=torch.cuda.is_available(),
            lang_list=['en'],
            confidence_threshold=self.OCR_CONF_TH,
            debug_output=True  # 啟用詳細調試輸出
        )
        self.change_iface = ChangeDetectorInterface()

        self.tmad_threshold_var = tk.DoubleVar(value=2.0)
        self.diff_threshold_var = tk.IntVar(value=30)

        self.roi_x1_var = tk.IntVar(value=0)
        self.roi_y1_var = tk.IntVar(value=0)
        self.roi_x2_var = tk.IntVar(value=0)
        self.roi_y2_var = tk.IntVar(value=0)
        
        self.status_var = tk.StringVar(value="就緒")

        self._create_widgets()
        self._load_ocr_models()

        self.analysis_thread: Optional[threading.Thread] = None
        # self.ocr_thread is removed

        master.bind("<Left>", self._on_left_key)
        master.bind("<Right>", self._on_right_key)
        master.bind("<Up>", self._on_up_key)
        master.bind("<Down>", self._on_down_key)
        master.bind("<space>", self._toggle_ocr_test_window)
        
        # self.status_var = tk.StringVar(value="就緒") # Moved this line up
        # lbl_status pack in _create_widgets
        
        self.changes_made = False

    def _on_left_key(self, event=None):
        """處理左鍵事件 - 前一幀"""
        print("左鍵按下 - 前一幀")
        self._step_frame(-1)
        return "break"  # 阻止事件繼續傳播

    def _on_right_key(self, event=None):
        """處理右鍵事件 - 後一幀"""
        print("右鍵按下 - 後一幀")
        self._step_frame(+1)
        return "break"  # 阻止事件繼續傳播

    def _on_up_key(self, event=None):
        """處理上鍵事件 - 在TreeView中選擇上一項並跳轉"""
        print("上鍵按下 - TreeView上一項")
        
        # 如果TreeView沒有內容，顯示提示
        if not self.tree.get_children():
            print("TreeView中沒有項目")
            messagebox.showinfo("提示", "沒有分析結果，請先進行分析")
            return "break"
        
        # 獲取目前選中的項目
        current_selection = self.tree.selection()
        all_items = list(self.tree.get_children())
        
        if not current_selection:
            # 如果沒有選中項目，選擇最後一項
            if all_items:
                self.tree.selection_set(all_items[-1])
                self.tree.focus(all_items[-1])
                self.tree.see(all_items[-1])
                self._jump_to_selected_frame()
        else:
            # 找到當前項目的索引
            current_item = current_selection[0]
            try:
                current_index = all_items.index(current_item)
                # 選擇上一項（如果在第一項，循環到最後一項）
                if current_index > 0:
                    prev_item = all_items[current_index - 1]
                else:
                    prev_item = all_items[-1]  # 循環到最後一項
                
                self.tree.selection_set(prev_item)
                self.tree.focus(prev_item)
                self.tree.see(prev_item)
                self._jump_to_selected_frame()
                
            except ValueError:
                print("找不到當前選中項目的索引")
        
        return "break"

    def _on_down_key(self, event=None):
        """處理下鍵事件 - 在TreeView中選擇下一項並跳轉"""
        print("下鍵按下 - TreeView下一項")
        
        # 如果TreeView沒有內容，顯示提示
        if not self.tree.get_children():
            print("TreeView中沒有項目")
            messagebox.showinfo("提示", "沒有分析結果，請先進行分析")
            return "break"
        
        # 獲取目前選中的項目
        current_selection = self.tree.selection()
        all_items = list(self.tree.get_children())
        
        if not current_selection:
            # 如果沒有選中項目，選擇第一項
            if all_items:
                self.tree.selection_set(all_items[0])
                self.tree.focus(all_items[0])
                self.tree.see(all_items[0])
                self._jump_to_selected_frame()
        else:
            # 找到當前項目的索引
            current_item = current_selection[0]
            try:
                current_index = all_items.index(current_item)
                # 選擇下一項（如果在最後一項，循環到第一項）
                if current_index < len(all_items) - 1:
                    next_item = all_items[current_index + 1]
                else:
                    next_item = all_items[0]  # 循環到第一項
                
                self.tree.selection_set(next_item)
                self.tree.focus(next_item)
                self.tree.see(next_item)
                self._jump_to_selected_frame()
                
            except ValueError:
                print("找不到當前選中項目的索引")
        
        return "break"

    def _jump_to_selected_frame(self):
        """跳轉到TreeView中選中項目對應的幀"""
        selected_items = self.tree.selection()
        if not selected_items:
            return
        
        selected_id = selected_items[0]
        
        try:
            # 獲取幀號並跳轉
            frame_idx = int(self.tree.set(selected_id, "frame"))
            content = self.tree.set(selected_id, "content")
            
            print(f"跳轉到選中的幀: {frame_idx}")
            self._show_frame(frame_idx)
            
            # 更新狀態欄
            self._update_status_bar(f"已跳轉到幀 {frame_idx}: {content}")
            
        except (ValueError, KeyError, TclError) as e:
            print(f"跳轉到選中幀時出錯: {e}")

    def _create_widgets(self):
        """創建 GUI 界面元素"""
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        top_frame_config = tk.Frame(self)
        top_frame_config.pack(pady=5, padx=10, fill="x")

        self.lbl_video_path = tk.Label(top_frame_config, text="未選擇影片")
        self.lbl_video_path.pack(side="left", padx=5)

        tk.Label(top_frame_config, text="OCR模型:").pack(side="left", padx=(10, 2))
        self.ocr_model_var = tk.StringVar(self)
        self.ocr_model_combobox = ttk.Combobox(top_frame_config, textvariable=self.ocr_model_var,
                                               values=[], state="readonly", width=15)
        self.ocr_model_combobox.pack(side="left", padx=2)
        self.ocr_model_combobox.bind("<<ComboboxSelected>>", self._on_ocr_model_change)

        tk.Label(top_frame_config, text="區域:").pack(side="left", padx=(10,2))
        self.region_var = tk.StringVar()
        self.region_combobox = ttk.Combobox(top_frame_config, textvariable=self.region_var,
                                            state="readonly", width=10)
        self.region_combobox.pack(side="left")
        self.region_combobox.bind("<<ComboboxSelected>>", self._on_region_select)

        btn_new_region = tk.Button(top_frame_config, text="新增區域", command=self._on_add_region)
        btn_new_region.pack(side="left", padx=2)

        btn_save_roi_config = tk.Button(top_frame_config, text="儲存ROI組態", command=self._save_roi_config)
        btn_save_roi_config.pack(side="left", padx=2)

        roi_field = tk.Frame(top_frame_config)
        roi_field.pack(side="left", padx=(10, 0))
        for text, var_tuple in [("x1", self.roi_x1_var),
                                ("y1", self.roi_y1_var),
                                ("x2", self.roi_x2_var),
                                ("y2", self.roi_y2_var)]:
            tk.Label(roi_field, text=f"{text}:").pack(side="left")
            ttk.Spinbox(roi_field, from_=0, to=99999, width=6,
                        textvariable=var_tuple).pack(side="left") # Fixed: var_tuple instead of var
        tk.Button(roi_field, text="套用", command=self._apply_roi_from_fields)\
          .pack(side="left", padx=3)

        tmad_frame = tk.Frame(top_frame_config)
        tmad_frame.pack(side="left", padx=(10, 0)) 

        ttk.Label(tmad_frame, text="T-MAD 門檻:").pack(side="left", padx=(0, 2))
        self.tmad_threshold_spinbox = ttk.Spinbox(tmad_frame, from_=0.0, to=100.0, increment=0.1, width=5, textvariable=self.tmad_threshold_var)
        self.tmad_threshold_spinbox.pack(side="left", padx=(0, 5))

        ttk.Label(tmad_frame, text="忽略差異<=").pack(side="left", padx=(5, 2))
        self.diff_threshold_spinbox = ttk.Spinbox(tmad_frame, from_=0, to=255, increment=1, width=4, textvariable=self.diff_threshold_var)
        self.diff_threshold_spinbox.pack(side="left", padx=(0, 5))

        main_action_buttons_frame = tk.Frame(self)
        main_action_buttons_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 載入影片按鈕 (移到 main_action_buttons_frame)
        self.btn_load = tk.Button(main_action_buttons_frame, text="載入影片", command=self._load_video)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        # self.btn_test = add_test_button(top_frame_config, self)
        # self.btn_test.pack(side="left", padx=5)
        self.binarize_mode_var = tk.BooleanVar(value=False)
        self.binarize_method_var = tk.StringVar(value="rule")

        binarize_frame = tk.Frame(self)
        binarize_frame.pack(fill="x", padx=10, pady=5)

        tk.Checkbutton(binarize_frame, text="二值化顯示", variable=self.binarize_mode_var, command=self._on_binarize_toggle).pack(side="left")
        tk.Radiobutton(binarize_frame, text="OTSU", variable=self.binarize_method_var, value="otsu", command=self._on_binarize_method_change).pack(side="left")
        tk.Radiobutton(binarize_frame, text="K-means", variable=self.binarize_method_var, value="kmeans", command=self._on_binarize_method_change).pack(side="left")
        tk.Radiobutton(binarize_frame, text="規則分割", variable=self.binarize_method_var, value="rule", command=self._on_binarize_method_change).pack(side="left")

        # 規則分割參數（HSV S閾值、灰階閾值）
        tk.Label(binarize_frame, text="S閾值:").pack(side="left")
        ttk.Spinbox(binarize_frame, from_=0, to=255, width=4, textvariable=self.hsv_s_threshold_var, command=self._on_binarize_method_change).pack(side="left")
        tk.Label(binarize_frame, text="灰階閾值:").pack(side="left")
        ttk.Spinbox(binarize_frame, from_=0, to=255, width=4, textvariable=self.gray_threshold_var, command=self._on_binarize_method_change).pack(side="left")

        self.lbl_diff = tk.Label(binarize_frame, text="Diff: -")
        self.lbl_diff.pack(side="left", padx=10)
        self.lbl_change = tk.Label(binarize_frame, text="變化判定: -")
        self.lbl_change.pack(side="left", padx=10)


        self.btn_analyze = tk.Button(
            main_action_buttons_frame, 
            text="開始分析", 
            command=self._start_analysis,
            state=tk.DISABLED
        )
        self.btn_analyze.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(
            main_action_buttons_frame, 
            text="停止分析", 
            command=self._stop_analysis,
            state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.btn_save = tk.Button(
            main_action_buttons_frame, 
            text="儲存標註", 
            command=lambda: self._save_annotations(self.region_name)
        )
        self.btn_save.pack(side=tk.LEFT, padx=5)

        main_area = tk.Frame(self)
        main_area.pack(fill="both", expand=True, padx=10, pady=5)

        video_frame = tk.Frame(main_area, width=self.VID_W, height=self.VID_H, bd=1, relief="sunken")
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        video_frame.pack_propagate(False) 

        self.lbl_video = tk.Label(video_frame, bg="black")
        self.lbl_video.pack(fill="both", expand=True)
        self.lbl_video.bind("<Button-1>", self._on_roi_start)
        self.lbl_video.bind("<B1-Motion>", self._on_roi_drag)
        self.lbl_video.bind("<ButtonRelease-1>", self._on_roi_end)
        self.roi_rect_id = None 

        self._create_control_hint_widget(video_frame)
        # 標註樹狀視圖  
        tree_frame = tk.Frame(main_area)
        tree_frame.pack(side="right", fill="y")

        tree_yscroll = ttk.Scrollbar(tree_frame, orient="vertical")
        tree_yscroll.pack(side="right", fill="y")
        tree_xscroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        tree_xscroll.pack(side="bottom", fill="x")

        self.tree = ttk.Treeview(tree_frame, columns=("frame", "diff", "content", "current_analysis"),
                                 show="headings", yscrollcommand=tree_yscroll.set,
                                 xscrollcommand=tree_xscroll.set)
        self.tree.pack(side="left", fill="y")
        self.tree.heading("frame", text="幀號")
        self.tree.heading("diff", text="Diff")
        self.tree.heading("content", text="標註內容")  # 原有欄位改名
        self.tree.heading("current_analysis", text="當前分析")  # 新增欄位
        self.tree.column("frame", width=60, anchor="center")
        self.tree.column("diff", width=80, anchor="center")
        self.tree.column("content", width=150, anchor="center")
        self.tree.column("current_analysis", width=150, anchor="center")
        tree_yscroll.config(command=self.tree.yview)
        tree_xscroll.config(command=self.tree.xview)
        self.bold_font = tkFont.Font(weight="bold")
        self.tree.tag_configure("changed", font=self.bold_font)
        self.tree.tag_configure("small_diff", foreground="red")
        # self.tree.bind("<Double-1>", self._on_tree_double_click) # Replaced by _on_edit_annotation
        # 確保TreeView不會攔截我們需要的鍵盤事件

        self.tree.bind("<Double-1>", self._on_edit_annotation)
        self.tree.bind("<Return>", self._on_edit_annotation)
        self.tree.bind('<<TreeviewSelect>>', self._on_treeview_select)
        self._setup_treeview_context_menu()

        bottom_frame_slider_num = tk.Frame(self)
        bottom_frame_slider_num.pack(fill="x", padx=10, pady=(0,5))

        self.slider_var = tk.DoubleVar()
        self.slider = ttk.Scale(bottom_frame_slider_num, from_=0, to=100, orient="horizontal",
                                variable=self.slider_var, command=self._on_slider_move)
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.slider.pack(fill="x", padx=5, side="left", expand=True)
        self.lbl_frame_num = tk.Label(bottom_frame_slider_num, text="幀: 0 / 0")
        self.lbl_frame_num.pack(side="right", padx=5)
        
        # nav_frame for "Go to frame" was originally packed into main_area.
        # If user wants it there, it needs to be:
        # nav_frame_goto = tk.Frame(main_area) # main_area, not self
        # nav_frame_goto.pack(side="left", after=video_frame, fill="x", pady=3) # Or some other packing
        # For simplicity, keeping it below slider for now.
        nav_frame_goto = tk.Frame(self) 
        nav_frame_goto.pack(fill="x", padx=10, pady=3)

        tk.Label(nav_frame_goto, text="跳至幀:").pack(side="left", padx=(0,2))
        self.goto_var = tk.IntVar(value=0)
        self.goto_entry = ttk.Entry(nav_frame_goto, textvariable=self.goto_var, width=7)
        self.goto_entry.pack(side="left")
        self.goto_entry.bind("<Return>", self._on_goto_frame)
        tk.Button(nav_frame_goto, text="Go", command=self._on_goto_frame)\
            .pack(side="left", padx=3)
        
        self.lbl_status = tk.Label(
            self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X, padx=3, pady=3)
        
        prog_frame = tk.Frame(self)
        prog_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0,2))

        self.progress_var = tk.IntVar()
        self.progressbar = ttk.Progressbar(
            prog_frame, length=280, mode="determinate", variable=self.progress_var
        )
        self.progressbar.pack(side="right", padx=6)
        self.lbl_prog = tk.Label(prog_frame, text="進度: 0/0")
        self.lbl_prog.pack(side="right")
        
        self._update_roi_fields()
        self._update_roi_ui()
        
    def _on_binarize_toggle(self):
        self._show_frame(self.current_frame_idx)

    def _on_binarize_method_change(self):
        self._show_frame(self.current_frame_idx)

    def _create_control_hint_widget(self, parent_frame):
        """創建方向鍵操作提示圖示"""
        try:
            # 創建提示框架，使用place()固定在左下角 - 進一步增大尺寸
            self.control_hint_frame = tk.Frame(parent_frame, bg="#2C2C2C", bd=1, relief="solid")
            self.control_hint_frame.place(x=10, y=self.VID_H-200, width=250, height=190)
            
            # 創建Canvas來繪製方向鍵圖示 - 進一步增大Canvas尺寸
            self.control_canvas = tk.Canvas(
                self.control_hint_frame, 
                width=240, 
                height=360, 
                bg="#2C2C2C", 
                highlightthickness=0
            )
            self.control_canvas.pack(fill="both", expand=True, padx=5, pady=5)
            
            # 繪製方向鍵圖示
            self._draw_control_hints()
            
            print("方向鍵操作提示圖示已創建")
            
        except Exception as e:
            print(f"創建操作提示圖示時出錯: {e}")
            traceback.print_exc()

    def _draw_control_hints(self):
        """繪製方向鍵和空白鍵圖示"""
        try:
            canvas = self.control_canvas
            
            # 清空畫布
            canvas.delete("all")
            
            # 定義顏色
            key_color = "#4A4A4A"
            text_color = "#CCCCCC"
            highlight_color = "#6A6A6A"
            desc_color = "#AAAAAA"
            title_color = "#FFFFFF"
            
            # 定義鍵位大小和位置 - 調整以適應更大空間
            key_size = 32
            center_x = 120
            center_y = 70
            key_spacing = 60  # 進一步增加按鍵間距
            
            # 標題 - 調整位置
            canvas.create_text(center_x, 18, text="鍵盤操作", fill=title_color, font=("Arial", 12, "bold"))
            
            # 上方向鍵 (↑)
            up_x = center_x
            up_y = center_y - key_spacing
            canvas.create_rectangle(
                up_x - key_size//2, up_y - key_size//2,
                up_x + key_size//2, up_y + key_size//2,
                fill=key_color, outline=highlight_color, width=2
            )
            canvas.create_text(up_x, up_y, text="↑", fill=text_color, font=("Arial", 18, "bold"))
            
            # 下方向鍵 (↓)
            down_x = center_x
            down_y = center_y + key_spacing
            canvas.create_rectangle(
                down_x - key_size//2, down_y - key_size//2,
                down_x + key_size//2, down_y + key_size//2,
                fill=key_color, outline=highlight_color, width=2
            )
            canvas.create_text(down_x, down_y, text="↓", fill=text_color, font=("Arial", 18, "bold"))
            
            # 左方向鍵 (←)
            left_x = center_x - key_spacing
            left_y = center_y
            canvas.create_rectangle(
                left_x - key_size//2, left_y - key_size//2,
                left_x + key_size//2, left_y + key_size//2,
                fill=key_color, outline=highlight_color, width=2
            )
            canvas.create_text(left_x, left_y, text="←", fill=text_color, font=("Arial", 18, "bold"))
            
            # 右方向鍵 (→)
            right_x = center_x + key_spacing
            right_y = center_y
            canvas.create_rectangle(
                right_x - key_size//2, right_y - key_size//2,
                right_x + key_size//2, right_y + key_size//2,
                fill=key_color, outline=highlight_color, width=2
            )
            canvas.create_text(right_x, right_y, text="→", fill=text_color, font=("Arial", 18, "bold"))
            
            # 空白鍵 (中間位置，較寬)
            space_width = 90
            space_height = 22
            space_x = center_x
            space_y = center_y + 75
            canvas.create_rectangle(
                space_x - space_width//2, space_y - space_height//2,
                space_x + space_width//2, space_y + space_height//2,
                fill=key_color, outline=highlight_color, width=2
            )
            canvas.create_text(space_x, space_y, text="SPACE", fill=text_color, font=("Arial", 11, "bold"))
            
            # 添加功能說明文字 - 重新排版，增加行距
            desc_y_start = 125
            line_height = 30  # 增加行距
            
            # 第一行：上下鍵說明
            canvas.create_text(30, desc_y_start, text="↑↓", fill=text_color, font=("Arial", 10, "bold"), anchor="w")
            canvas.create_text(50, desc_y_start, text="跳到前後變化點", fill=desc_color, font=("Arial", 9), anchor="w")
            
            # 第二行：左右鍵說明
            canvas.create_text(30, desc_y_start + line_height, text="←→", fill=text_color, font=("Arial", 10, "bold"), anchor="w")
            canvas.create_text(50, desc_y_start + line_height, text="逐幀切換", fill=desc_color, font=("Arial", 9), anchor="w")
            
            # 第三行：空白鍵說明
            canvas.create_text(30, desc_y_start + line_height * 2, text="空白", fill=text_color, font=("Arial", 10, "bold"), anchor="w")
            canvas.create_text(65, desc_y_start + line_height * 2, text="OCR測試視窗", fill=desc_color, font=("Arial", 9), anchor="w")
            
            # 添加分隔線美化
            canvas.create_line(20, desc_y_start - 8, 220, desc_y_start - 8, fill="#555555", width=1)
            
        except Exception as e:
            print(f"繪製控制提示時出錯: {e}")
            traceback.print_exc()

    def _load_ocr_models(self):
        """
        載入並設定可用的OCR模型選項，包括不同配置
        """
        try:
            # 定義可用的模型選項，包括不同配置
            model_options = [
                "EasyOCR Default",
                "EasyOCR High Precision",
                "EasyOCR Fast Mode",
                # 未來可以添加其他模型，例如：
                # "PaddleOCR Default",
                # "TrOCR Base",
                # "Custom CRNN Model"
            ]
            
            self.ocr_model_combobox["values"] = model_options
            if model_options:
                self.ocr_model_var.set(model_options[0])  # 預設選擇第一個
            else:
                self.ocr_model_var.set("無可用模型")
                
        except Exception as e:
            print(f"設定 OCR 模型下拉框失敗: {e}")
            if hasattr(self, 'ocr_model_var'): 
                self.ocr_model_var.set("設定失敗")



    def _on_ocr_model_change(self, event=None):
        """處理OCR模型切換"""
        selected_model = self.ocr_model_var.get()
        print(f"OCR 模型更改為: {selected_model}")
        
        try:
            # 根據選擇的模型配置來初始化不同的OCR介面
            if selected_model == "EasyOCR Default":
                self.ocr_iface = get_ocr_model(
                    model_type="easyocr",
                    gpu=torch.cuda.is_available(),
                    lang_list=['en'],
                    confidence_threshold=self.OCR_CONF_TH,
                    debug_output=True  # 啟用調試輸出
                )
            elif selected_model == "EasyOCR High Precision":
                # 高精度模式：使用更嚴格的信心閾值和更完整的字符集
                self.ocr_iface = get_ocr_model(
                    model_type="easyocr",
                    gpu=torch.cuda.is_available(),
                    lang_list=['en'],
                    confidence_threshold=0.7,  # 更高的信心閾值
                    debug_output=True  # 啟用調試輸出
                )
            elif selected_model == "EasyOCR Fast Mode":
                # 快速模式：較低的信心閾值，可能更快但精度稍低
                self.ocr_iface = get_ocr_model(
                    model_type="easyocr",
                    gpu=torch.cuda.is_available(),
                    lang_list=['en'],
                    confidence_threshold=0.3,  # 較低的信心閾值
                    debug_output=True  # 啟用調試輸出
                )
            # 未來可以添加其他模型的初始化邏輯
            # elif selected_model == "PaddleOCR Default":
            #     self.ocr_iface = get_ocr_model(
            #         model_type="paddleocr",
            #         gpu=torch.cuda.is_available(),
            #         lang='en'
            #     )
            else:
                print(f"未知的模型配置: {selected_model}")
                return
                
            self._update_status_bar(f"OCR 模型已切換至: {selected_model}")
            print(f"OCR 模型切換成功: {selected_model}")
            
        except Exception as e:
            messagebox.showerror("OCR 模型切換失敗", f"無法載入模型 {selected_model}: {e}")
            print(f"切換 OCR 模型失敗: {e}")
            traceback.print_exc()
            self._update_status_bar(f"OCR 模型 {selected_model} 載入失敗")

    def _toggle_ocr_test_window(self, event=None):
        """切換OCR測試視窗的顯示/隱藏"""
        if self.ocr_test_active and self.ocr_test_window:
            # 如果視窗已開啟，則關閉它
            self._close_ocr_test_window()
        else:
            # 如果視窗未開啟，則顯示它
            self._show_ocr_test_window()

    def _show_ocr_test_window(self):
        """顯示增強版OCR測試視窗 - 支援精細子區域選擇、像素顏色分析、等比例放大和二值化處理"""
        if not self.video_file_path or not self.roi_coords:
            messagebox.showwarning("提示", "請先載入影片並設定ROI區域")
            return
            
        if self.ocr_test_window:
            # 如果視窗已存在，將其帶到前面
            self.ocr_test_window.lift()
            self.ocr_test_window.focus_set()
            return
            
        try:
            # 獲取當前幀的ROI圖像
            roi_image = self._get_current_frame_roi()
            if roi_image is None:
                messagebox.showerror("錯誤", "無法獲取當前幀的ROI圖像")
                return
                
            # 創建測試視窗 - 增大尺寸以容納新功能
            self.ocr_test_window = tk.Toplevel(self.master)
            self.ocr_test_window.title(f"OCR精細測試 - 幀 {self.current_frame_idx} - {self.ocr_model_var.get()}")
            self.ocr_test_window.geometry("1200x800")
            self.ocr_test_window.resizable(True, True)
            
            # 設置視窗關閉時的處理
            self.ocr_test_window.protocol("WM_DELETE_WINDOW", self._close_ocr_test_window)
            
            # 儲存原始ROI圖像用於像素顏色分析和處理
            self.roi_image_original = roi_image
            self.roi_image_processed = None  # 處理後的圖像
            self.is_processed_mode = False   # 當前是否為處理模式
            
            # 初始化子區域相關屬性
            self.sub_regions = []  # 儲存子區域座標 [(x1,y1,x2,y2), ...]
            self.sub_region_rects = []  # 儲存畫布上的矩形ID
            self.current_sub_rect = None  # 當前拖拽的矩形
            self.drag_start = None  # 拖拽起始點
            
            # 初始化縮放相關屬性
            self.zoom_level = tk.DoubleVar(value=4.0)  # 預設放大4倍
            self.min_zoom = 1.0
            self.max_zoom = 20.0
            
            # 主要布局：左右分割
            main_paned = ttk.PanedWindow(self.ocr_test_window, orient=tk.HORIZONTAL)
            main_paned.pack(fill="both", expand=True, padx=10, pady=10)
            
            # 左側：圖像顯示和控制
            left_frame = tk.Frame(main_paned)
            main_paned.add(left_frame, weight=2)  # 增加左側權重
            
            # 右側：OCR結果顯示
            right_frame = tk.Frame(main_paned)
            main_paned.add(right_frame, weight=1)
            
            # 左側 - 標題和縮放控制
            header_frame = tk.Frame(left_frame)
            header_frame.pack(fill="x", pady=(0, 10))
            
            tk.Label(header_frame, text="ROI圖像分析", 
                    font=("Arial", 14, "bold")).pack(side="left")
            
            # 縮放控制區域
            zoom_frame = tk.Frame(header_frame)
            zoom_frame.pack(side="right")
            
            tk.Label(zoom_frame, text="縮放:", font=("Arial", 10)).pack(side="left", padx=(0, 5))
            zoom_scale = ttk.Scale(zoom_frame, from_=self.min_zoom, to=self.max_zoom, 
                                  variable=self.zoom_level, orient="horizontal", length=150,
                                  command=self._on_zoom_change)
            zoom_scale.pack(side="left", padx=(0, 5))
            
            self.zoom_label = tk.Label(zoom_frame, text="4.0x", font=("Courier", 10), width=6)
            self.zoom_label.pack(side="left", padx=(0, 10))
            
            # 預設縮放按鈕
            btn_zoom_frame = tk.Frame(zoom_frame)
            btn_zoom_frame.pack(side="left")
            
            for zoom_val, text in [(2.0, "2x"), (4.0, "4x"), (8.0, "8x"), (16.0, "16x")]:
                tk.Button(btn_zoom_frame, text=text, width=3,
                         command=lambda z=zoom_val: self._set_zoom_level(z)).pack(side="left", padx=1)
            
            # 說明文字
            instruction_text = ("拖拽滑鼠選擇最多3個子區域 | 滑鼠懸停顯示像素顏色\n"
                              "綠色=已選擇，紅色=當前拖拽 | 使用縮放控制查看細節")
            tk.Label(left_frame, text=instruction_text, 
                    font=("Arial", 9), fg="gray", justify="left").pack(pady=(0, 10))
            
            # 圖像處理控制區域
            processing_frame = tk.LabelFrame(left_frame, text="影像處理")
            processing_frame.pack(fill="x", pady=(0, 10))
            
            # 處理按鈕行
            btn_processing_frame = tk.Frame(processing_frame)
            btn_processing_frame.pack(fill="x", padx=5, pady=5)
            
            # 二值化切換按鈕
            self.btn_binarize = tk.Button(btn_processing_frame, text="二值化處理", 
                                         command=self._toggle_binarization,
                                         bg="#E8F4F8", relief="raised")
            self.btn_binarize.pack(side="left", padx=(0, 5))
            
            # 處理方法選擇
            tk.Label(btn_processing_frame, text="方法:", font=("Arial", 9)).pack(side="left", padx=(10, 2))
            self.binarize_method = tk.StringVar(value="rule")
            method_frame = tk.Frame(btn_processing_frame)
            method_frame.pack(side="left", padx=(0, 10))
            
            tk.Radiobutton(method_frame, text="OTSU", variable=self.binarize_method, 
                          value="otsu", font=("Arial", 8)).pack(side="left")
            tk.Radiobutton(method_frame, text="K-means", variable=self.binarize_method, 
                          value="kmeans", font=("Arial", 8)).pack(side="left", padx=(5, 0))
            tk.Radiobutton(method_frame, text="規則分割", variable=self.binarize_method, 
                          value="rule", font=("Arial", 8)).pack(side="left", padx=(5, 0))
            
            # 狀態指示
            self.processing_status_label = tk.Label(btn_processing_frame, text="原始影像", 
                                                   font=("Arial", 9), fg="blue")
            self.processing_status_label.pack(side="right", padx=(10, 0))
            
            # ✨ 規則分割參數控制區域
            rule_params_frame = tk.Frame(processing_frame)
            rule_params_frame.pack(fill="x", padx=5, pady=(0, 5))
            
            # HSV飽和度閾值控制
            tk.Label(rule_params_frame, text="HSV-S閾值:", font=("Arial", 9)).pack(side="left", padx=(0, 2))
            self.hsv_s_threshold_var = tk.IntVar(value=30)  # 預設30%
            self.hsv_s_spinbox = ttk.Spinbox(rule_params_frame, from_=0, to=100, increment=1, 
                                             width=5, textvariable=self.hsv_s_threshold_var)
            self.hsv_s_spinbox.pack(side="left", padx=(0, 2))
            tk.Label(rule_params_frame, text="%", font=("Arial", 9)).pack(side="left", padx=(0, 10))
            
            # 灰階閾值控制
            tk.Label(rule_params_frame, text="灰階閾值:", font=("Arial", 9)).pack(side="left", padx=(0, 2))
            self.gray_threshold_var = tk.IntVar(value=150)  # 預設150
            self.gray_threshold_spinbox = ttk.Spinbox(rule_params_frame, from_=0, to=255, increment=1, 
                                                     width=5, textvariable=self.gray_threshold_var)
            self.gray_threshold_spinbox.pack(side="left", padx=(0, 2))
            
            # 參數說明
            tk.Label(rule_params_frame, text="(低飽和度且高亮度的像素視為前景)", 
                     font=("Arial", 8), fg="gray").pack(side="left", padx=(10, 0))
            
            # 圖像顯示區域 - 使用捲軸容器
            img_container = tk.LabelFrame(left_frame, text=f"原始ROI: {roi_image.size[0]}x{roi_image.size[1]} 像素")
            img_container.pack(fill="both", expand=True, pady=(0, 10))
            
            # 創建捲軸框架
            canvas_frame = tk.Frame(img_container)
            canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
            
            # 水平和垂直捲軸
            h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal")
            v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical")
            
            # 初始顯示尺寸計算
            self._calculate_display_size()
            
            # 創建可捲動的Canvas
            self.roi_canvas = tk.Canvas(canvas_frame, 
                                       bg="white", relief="sunken", bd=2,
                                       xscrollcommand=h_scrollbar.set,
                                       yscrollcommand=v_scrollbar.set)
            
            # 配置捲軸
            h_scrollbar.config(command=self.roi_canvas.xview)
            v_scrollbar.config(command=self.roi_canvas.yview)
            
            # 佈局捲軸和Canvas
            self.roi_canvas.grid(row=0, column=0, sticky="nsew")
            h_scrollbar.grid(row=1, column=0, sticky="ew")
            v_scrollbar.grid(row=0, column=1, sticky="ns")
            
            canvas_frame.grid_rowconfigure(0, weight=1)
            canvas_frame.grid_columnconfigure(0, weight=1)
            
            # 更新圖像顯示
            self._update_roi_display()
            
            # 綁定滑鼠事件用於選擇子區域和顯示像素顏色
            self.roi_canvas.bind("<Button-1>", self._on_sub_roi_start)
            self.roi_canvas.bind("<B1-Motion>", self._on_sub_roi_drag)
            self.roi_canvas.bind("<ButtonRelease-1>", self._on_sub_roi_end)
            self.roi_canvas.bind("<Motion>", self._on_canvas_mouse_move)
            self.roi_canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # 滾輪縮放
            
            # 像素顏色顯示區域 - 增強版
            self._create_pixel_info_panel(left_frame)
            
            # 控制按鈕
            btn_frame1 = tk.Frame(left_frame)
            btn_frame1.pack(fill="x", pady=5)
            
            tk.Button(btn_frame1, text="清除所有子區域", 
                     command=self._clear_sub_regions).pack(side="left", padx=(0, 5))
            tk.Button(btn_frame1, text="分析所有區域", 
                     command=lambda: self._analyze_all_regions_enhanced(roi_image, right_frame)).pack(side="left", padx=(0, 5))
            tk.Button(btn_frame1, text="重設縮放", 
                     command=lambda: self._set_zoom_level(4.0)).pack(side="left")
            
            # 右側 - OCR結果區域
            tk.Label(right_frame, text="OCR分析結果", 
                    font=("Arial", 12, "bold")).pack(pady=(0, 10))
            
            # 滾動式結果顯示區域
            result_scroll_frame = tk.Frame(right_frame)
            result_scroll_frame.pack(fill="both", expand=True)
            
            result_canvas = tk.Canvas(result_scroll_frame)
            result_scrollbar = ttk.Scrollbar(result_scroll_frame, orient="vertical", command=result_canvas.yview)
            self.result_content_frame = tk.Frame(result_canvas)
            
            result_canvas.create_window((0, 0), window=self.result_content_frame, anchor="nw")
            result_canvas.configure(yscrollcommand=result_scrollbar.set)
            
            result_canvas.pack(side="left", fill="both", expand=True)
            result_scrollbar.pack(side="right", fill="y")
            
            # 綁定滾動更新
            def _on_result_configure(event):
                result_canvas.configure(scrollregion=result_canvas.bbox("all"))
            self.result_content_frame.bind("<Configure>", _on_result_configure)
            
            # 底部按鈕
            bottom_btn_frame = tk.Frame(self.ocr_test_window)
            bottom_btn_frame.pack(fill="x", padx=10, pady=(0, 10))
            
            tk.Button(bottom_btn_frame, text="重新分析", 
                     command=lambda: self._analyze_all_regions_enhanced(roi_image, right_frame)).pack(side="left", padx=(0, 5))
            tk.Button(bottom_btn_frame, text="關閉", 
                     command=self._close_ocr_test_window).pack(side="right")
            
            # 初始分析完整ROI
            self._analyze_all_regions_enhanced(roi_image, right_frame)
            
            self.ocr_test_active = True
            self._update_status_bar(f"OCR精細測試視窗已開啟 (幀 {self.current_frame_idx})")
            
        except Exception as e:
            print(f"顯示OCR精細測試視窗時出錯: {e}")
            traceback.print_exc()
            messagebox.showerror("錯誤", f"無法顯示OCR精細測試視窗: {e}")

    def _toggle_binarization(self):
        """切換二值化處理並自動執行OCR分析"""
        try:
            if self.is_processed_mode:
                # 切換回原始模式
                self.is_processed_mode = False
                self.btn_binarize.config(text="二值化處理", bg="#E8F4F8", relief="raised")
                self.processing_status_label.config(text="原始影像", fg="blue")
            else:
                # 切換到處理模式
                method = self.binarize_method.get()
                self.roi_image_processed = self._apply_binarization(self.roi_image_original, method)
                
                if self.roi_image_processed is not None:
                    self.is_processed_mode = True
                    self.btn_binarize.config(text="還原原圖", bg="#F8E8E8", relief="sunken")
                    self.processing_status_label.config(text=f"二值化 ({method.upper()})", fg="red")
                else:
                    messagebox.showerror("錯誤", "二值化處理失敗")
                    return
            
            # 更新顯示
            self._update_roi_display()
            
            # 自動執行OCR分析
            print(f"二值化狀態改變，自動執行OCR分析...")
            self._analyze_all_regions_enhanced(self.roi_image_original, self.result_content_frame)
            
        except Exception as e:
            print(f"切換二值化處理時出錯: {e}")
            messagebox.showerror("錯誤", f"處理失敗: {e}")

    def _apply_binarization(self, image: Image.Image, method: str) -> Optional[Image.Image]:
        """應用二值化處理"""
        try:
            import cv2
            import numpy as np
            from sklearn.cluster import KMeans
            
            # 轉換為OpenCV格式
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            if method == "otsu":
                # OTSU閾值二值化
                threshold_value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                print(f"🎯 OTSU自動閾值: {threshold_value:.1f}")
                
            elif method == "kmeans":
                # K-means聚類二值化
                # 將圖像重塑為一維數組
                pixels = gray.reshape(-1, 1).astype(np.float32)
                
                # 執行K-means聚類（k=2）
                kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
                kmeans.fit(pixels)
                
                # 獲取聚類中心和標籤
                centers = kmeans.cluster_centers_.flatten()
                labels = kmeans.labels_
                
                # 決定哪個聚類代表前景（較亮的）
                if centers[0] > centers[1]:
                    foreground_label = 0
                    background_label = 1
                else:
                    foreground_label = 1  
                    background_label = 0
                
                # 創建二值圖像
                binary = np.zeros_like(gray)
                binary[labels.reshape(gray.shape) == foreground_label] = 255
                
                print(f"🎯 K-means聚類中心: 暗={centers.min():.1f}, 亮={centers.max():.1f}")
                
                # 計算前景和背景像素數量
                foreground_pixels = np.sum(labels == foreground_label)
                background_pixels = np.sum(labels == background_label)
                total_pixels = foreground_pixels + background_pixels
                
                print(f"📊 像素分布: 前景={foreground_pixels}({foreground_pixels/total_pixels*100:.1f}%), "
                    f"背景={background_pixels}({background_pixels/total_pixels*100:.1f}%)")
                    
            elif method == "rule":
                # ✨ 規則分割二值化：基於HSV飽和度和灰階值
                hsv_s_threshold = self.hsv_s_threshold_var.get()
                gray_threshold = self.gray_threshold_var.get()
                
                # 轉換為HSV色彩空間
                hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv_image)
                
                # 將S值從0-255轉換為0-100百分比
                s_percentage = (s / 255.0) * 100
                
                # 規則：低飽和度(S < threshold%)且高亮度(灰階 > threshold)的像素為前景(白色)
                condition1 = s_percentage < hsv_s_threshold  # 低飽和度
                condition2 = gray > gray_threshold           # 高亮度
                foreground_mask = condition1 & condition2
                
                # 創建二值圖像
                binary = np.zeros_like(gray)
                binary[foreground_mask] = 255
                
                # 統計資訊
                total_pixels = gray.size
                foreground_pixels = np.sum(foreground_mask)
                background_pixels = total_pixels - foreground_pixels
                
                # 統計滿足各條件的像素數量
                low_sat_pixels = np.sum(condition1)
                high_gray_pixels = np.sum(condition2)
                

                # print(f"🎯 規則分割參數: HSV-S < {hsv_s_threshold}%, 灰階 > {gray_threshold}")
                # print(f"📊 條件統計:")
                # print(f"   低飽和度像素: {low_sat_pixels}({low_sat_pixels/total_pixels*100:.1f}%)")
                # print(f"   高亮度像素: {high_gray_pixels}({high_gray_pixels/total_pixels*100:.1f}%)")
                # print(f"   符合規則像素: {foreground_pixels}({foreground_pixels/total_pixels*100:.1f}%)")
                # print(f"   背景像素: {background_pixels}({background_pixels/total_pixels*100:.1f}%)")
                
            else:
                print(f"未知的二值化方法: {method}")
                return None
            
            # 轉換回PIL格式
            result_pil = Image.fromarray(binary)
            
            return result_pil
            
        except Exception as e:
            print(f"應用二值化處理時出錯: {e}")
            traceback.print_exc()
            return None
        
    def _get_current_display_image(self):
        """獲取當前應該顯示的圖像（原始或處理後）"""
        if self.is_processed_mode and self.roi_image_processed is not None:
            return self.roi_image_processed
        else:
            return self.roi_image_original

    def _update_roi_display(self):
        """更新ROI圖像顯示 - 支援原始/處理圖像切換"""
        try:
            # 計算新的顯示尺寸
            self._calculate_display_size()
            
            # 更新Canvas尺寸
            self.roi_canvas.config(scrollregion=(0, 0, self.display_w, self.display_h))
            
            # 獲取當前應該顯示的圖像
            current_image = self._get_current_display_image()
            
            # 創建放大的圖像 - 使用最近鄰插值保持像素邊界清晰
            display_image = current_image.resize(
                (self.display_w, self.display_h), 
                Image.Resampling.NEAREST  # 使用NEAREST保持像素邊界清晰
            )

            border_size = 2
            width, height = display_image.size
            bordered_image = Image.new('RGB', (width + 2*border_size, height + 2*border_size), 'white')
            bordered_image.paste(display_image, (border_size, border_size))
            display_image = bordered_image

            # 更新PhotoImage
            self.roi_photo = ImageTk.PhotoImage(display_image)
            
            # 清除舊圖像和矩形
            self.roi_canvas.delete("image")
            self.roi_canvas.delete("sub_rect")
            
            # 在Canvas上顯示新圖像
            self.roi_canvas.create_image(0, 0, anchor="nw", image=self.roi_photo, tags="image")
            
            # 重新繪製已選擇的子區域
            self._redraw_sub_regions()
            
        except Exception as e:
            print(f"更新ROI顯示時出錯: {e}")

    def _analyze_all_regions_enhanced(self, roi_image: Image.Image, result_parent: tk.Widget):
        """分析所有區域（增強版 - 支援處理後圖像）"""
        # 清空結果顯示區域
        for widget in self.result_content_frame.winfo_children():
            widget.destroy()
        
        # 獲取當前應該分析的圖像
        current_image = self._get_current_display_image()
        
        try:
            # 1. 分析完整ROI區域
            full_roi_coords = (0, 0, current_image.size[0], current_image.size[1])
            self._analyze_single_region_enhanced(current_image, full_roi_coords, "完整ROI", 0)
            
            # 2. 分析選定的子區域
            if self.sub_regions:
                for i, coords in enumerate(self.sub_regions):
                    region_name = f"子區域 {i+1}"
                    self._analyze_single_region_enhanced(current_image, coords, region_name, i+1)
            else:
                # 如果沒有子區域，顯示提示
                info_frame = tk.LabelFrame(self.result_content_frame, text="提示")
                info_frame.pack(fill="x", padx=5, pady=5)
                tk.Label(info_frame, text="拖拽選擇子區域進行精細分析", 
                        font=("Arial", 10), fg="gray").pack(pady=10)
                
        except Exception as e:
            print(f"分析所有區域時出錯: {e}")
            messagebox.showerror("錯誤", f"分析失敗: {e}")

    def _analyze_single_region_enhanced(self, image: Image.Image, coords: tuple, region_name: str, index: int):
        """分析單一區域（增強版 - 顯示處理狀態）- 修正OCR方法名稱"""
        try:
            x1, y1, x2, y2 = coords
            
            # 提取子區域圖像
            sub_image = image.crop((x1, y1, x2, y2))
            
            # 執行OCR - 修正方法名稱
            try:
                if hasattr(self.ocr_iface, 'recognize'):
                    # 使用 recognize 方法（返回 text, confidence）
                    ocr_result, confidence = self.ocr_iface.recognize(sub_image)
                elif hasattr(self.ocr_iface, 'predict'):
                    # 備用：如果有 predict 方法
                    ocr_result = self.ocr_iface.predict(sub_image)
                    confidence = getattr(self.ocr_iface, 'last_confidence', None)
                else:
                    # 如果都沒有，嘗試直接調用
                    ocr_result = str(self.ocr_iface(sub_image))
                    confidence = None
                    
            except Exception as ocr_error:
                print(f"OCR調用失敗: {ocr_error}")
                ocr_result = "〈OCR錯誤〉"
                confidence = None
            
            # 創建結果顯示框架
            result_frame = tk.LabelFrame(self.result_content_frame, text=f"{region_name} ({x2-x1}×{y2-y1})")
            result_frame.pack(fill="x", padx=5, pady=5)
            
            # 顯示子圖像（縮略圖）
            thumbnail_size = (100, 60)
            thumbnail = sub_image.copy()
            thumbnail.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            
            img_frame = tk.Frame(result_frame)
            img_frame.pack(side="left", padx=5, pady=5)
            
            thumbnail_photo = ImageTk.PhotoImage(thumbnail)
            img_label = tk.Label(img_frame, image=thumbnail_photo, relief="sunken", bd=1)
            img_label.image = thumbnail_photo  # 保持引用
            img_label.pack()
            
            # 顯示座標
            coord_label = tk.Label(img_frame, text=f"({x1},{y1})-({x2},{y2})", 
                                font=("Courier", 8), fg="gray")
            coord_label.pack()
            
            # 顯示OCR結果
            text_frame = tk.Frame(result_frame)
            text_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
            
            # 處理狀態指示
            status_text = "處理後" if self.is_processed_mode else "原始"
            status_color = "red" if self.is_processed_mode else "blue"
            method_info = f" ({self.binarize_method.get().upper()})" if self.is_processed_mode else ""
            
            tk.Label(text_frame, text=f"OCR結果 ({status_text}{method_info}):", 
                    font=("Arial", 10, "bold"), fg=status_color).pack(anchor="w")
            
            result_text = ocr_result if ocr_result else "〈未識別〉"
            result_label = tk.Label(text_frame, text=result_text, 
                                font=("Arial", 12), fg="darkgreen" if ocr_result else "red",
                                wraplength=200, justify="left")
            result_label.pack(anchor="w", pady=(2, 5))
            
            # 顯示置信度信息（如果可用）
            if confidence is not None:
                confidence_text = f"置信度: {confidence:.3f}"
                confidence_color = "darkgreen" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                tk.Label(text_frame, text=confidence_text, 
                        font=("Arial", 9), fg=confidence_color).pack(anchor="w")
            
            # 顯示像素統計（如果是處理後的圖像）
            if self.is_processed_mode:
                try:
                    import numpy as np
                    sub_array = np.array(sub_image)
                    if len(sub_array.shape) == 3:
                        # RGB圖像，轉換為灰階來計算統計
                        gray_array = np.mean(sub_array, axis=2)
                    else:
                        gray_array = sub_array
                    
                    white_pixels = np.sum(gray_array > 127)
                    total_pixels = gray_array.size
                    white_ratio = white_pixels / total_pixels * 100
                    
                    stats_text = f"白色像素: {white_ratio:.1f}%"
                    tk.Label(text_frame, text=stats_text, 
                            font=("Arial", 8), fg="gray").pack(anchor="w")
                    
                    # 新增：計算左右各40像素寬的平均值
                    left_right_stats = self._calculate_left_right_pixel_stats(gray_array)
                    if left_right_stats:
                        stats_frame = tk.Frame(text_frame)
                        stats_frame.pack(anchor="w", pady=(2, 0))
                        
                        tk.Label(stats_frame, text="左右區域分析:", 
                                font=("Arial", 8, "bold"), fg="darkblue").pack(anchor="w")
                        
                        for stat_text, color in left_right_stats:
                            tk.Label(stats_frame, text=stat_text, 
                                    font=("Courier", 8), fg=color).pack(anchor="w")
                            
                except Exception as e:
                    print(f"計算像素統計時出錯: {e}")
            
            print(f"{region_name} OCR結果: '{result_text}' (座標: {coords}, 狀態: {status_text}{method_info})")
            if confidence is not None:
                print(f"  置信度: {confidence:.3f}")
            
        except Exception as e:
            print(f"分析區域 {region_name} 時出錯: {e}")
            traceback.print_exc()

    def _calculate_left_right_pixel_stats(self, gray_array: np.ndarray) -> List[Tuple[str, str]]:
        """計算左右各40像素寬區域的平均值統計"""
        try:
            import numpy as np
            
            height, width = gray_array.shape
            
            # 如果圖像寬度小於80像素，無法進行左右40像素的分析
            if width < 80:
                return [("區域太小，無法分析左右40像素", "orange")]
            
            # 提取左側40像素寬的區域
            left_region = gray_array[:, :40]
            left_mean = np.mean(left_region)
            left_white_ratio = np.sum(left_region > 127) / left_region.size * 100
            
            # 提取右側40像素寬的區域
            right_region = gray_array[:, -40:]
            right_mean = np.mean(right_region)
            right_white_ratio = np.sum(right_region > 127) / right_region.size * 100
            
            # 計算中間區域（如果存在）
            middle_stats = []
            if width > 80:
                middle_region = gray_array[:, 40:-40]
                middle_mean = np.mean(middle_region)
                middle_white_ratio = np.sum(middle_region > 127) / middle_region.size * 100
                middle_stats.append((f"中間區域: 均值={middle_mean:.1f}, 白色={middle_white_ratio:.1f}%", "gray"))
            
            # 判斷區域特徵
            def get_region_color(white_ratio):
                if white_ratio > 50:
                    return "red"  # 主要是白色（可能有文字）
                elif white_ratio > 10:
                    return "orange"  # 有一些白色
                else:
                    return "darkgreen"  # 主要是黑色
            
            # 建立統計結果
            stats = []
            stats.append((f"左側40px: 均值={left_mean:.1f}, 白色={left_white_ratio:.1f}%", 
                        get_region_color(left_white_ratio)))
            stats.append((f"右側40px: 均值={right_mean:.1f}, 白色={right_white_ratio:.1f}%", 
                        get_region_color(right_white_ratio)))
            
            # 加入中間區域統計
            stats.extend(middle_stats)
            
            # 分析建議
            max_white_ratio = max(left_white_ratio, right_white_ratio)
            if max_white_ratio > 50:
                suggestion = "🔴 檢測到高白色比例，可能有文字內容"
                suggestion_color = "red"
            elif max_white_ratio > 10:
                suggestion = "🟡 檢測到中等白色比例，可能有部分內容"
                suggestion_color = "orange"
            else:
                suggestion = "🟢 主要為黑色背景，無明顯內容"
                suggestion_color = "darkgreen"
            
            stats.append((suggestion, suggestion_color))
            
            # 門檻值建議
            threshold_suggestion = f"建議門檻值: {max_white_ratio/2:.1f}% (最大白色比例的一半)"
            stats.append((threshold_suggestion, "blue"))
            
            return stats
            
        except Exception as e:
            print(f"計算左右像素統計時出錯: {e}")
            return [("統計計算失敗", "red")]

    def _calculate_display_size(self):
        """計算顯示尺寸"""
        zoom = self.zoom_level.get()
        self.roi_display_scale = zoom
        self.display_w = int(self.roi_image_original.size[0] * zoom)
        self.display_h = int(self.roi_image_original.size[1] * zoom)
        
    # def _update_roi_display(self):
    #     """更新ROI圖像顯示"""
    #     try:
    #         # 計算新的顯示尺寸
    #         self._calculate_display_size()
            
    #         # 更新Canvas尺寸
    #         self.roi_canvas.config(scrollregion=(0, 0, self.display_w, self.display_h))
            
    #         # 創建放大的圖像 - 使用最近鄰插值保持像素清晰
    #         display_image = self.roi_image_original.resize(
    #             (self.display_w, self.display_h), 
    #             Image.Resampling.NEAREST  # 使用NEAREST保持像素邊界清晰
    #         )
            
    #         # 更新PhotoImage
    #         self.roi_photo = ImageTk.PhotoImage(display_image)
            
    #         # 清除舊圖像和矩形
    #         self.roi_canvas.delete("image")
    #         self.roi_canvas.delete("sub_rect")
            
    #         # 在Canvas上顯示新圖像
    #         self.roi_canvas.create_image(0, 0, anchor="nw", image=self.roi_photo, tags="image")
            
    #         # 重新繪製已選擇的子區域
    #         self._redraw_sub_regions()
            
    #     except Exception as e:
    #         print(f"更新ROI顯示時出錯: {e}")
            
    def _on_zoom_change(self, value):
        """縮放改變時的處理"""
        zoom = float(value)
        self.zoom_label.config(text=f"{zoom:.1f}x")
        self._update_roi_display()
        
    def _set_zoom_level(self, zoom_value):
        """設定特定的縮放級別"""
        self.zoom_level.set(zoom_value)
        self.zoom_label.config(text=f"{zoom_value:.1f}x")
        self._update_roi_display()
        
    def _on_mouse_wheel(self, event):
        """滑鼠滾輪縮放"""
        if event.state & 0x4:  # Ctrl鍵被按住
            # Ctrl+滾輪進行縮放
            current_zoom = self.zoom_level.get()
            zoom_delta = 0.5 if event.delta > 0 else -0.5
            new_zoom = max(self.min_zoom, min(self.max_zoom, current_zoom + zoom_delta))
            self._set_zoom_level(new_zoom)
        else:
            # 普通滾輪進行捲動
            self.roi_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
    def _redraw_sub_regions(self):
        """重新繪製已選擇的子區域"""
        self.sub_region_rects.clear()
        
        for i, (x1, y1, x2, y2) in enumerate(self.sub_regions):
            # 轉換到當前縮放的座標
            scaled_x1 = x1 * self.roi_display_scale
            scaled_y1 = y1 * self.roi_display_scale
            scaled_x2 = x2 * self.roi_display_scale
            scaled_y2 = y2 * self.roi_display_scale
            
            # 創建矩形
            rect_id = self.roi_canvas.create_rectangle(
                scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                outline="green", width=2, tags="sub_rect"
            )
            self.sub_region_rects.append(rect_id)
            
    def _create_pixel_info_panel(self, parent):
        """創建增強版像素資訊面板"""
        pixel_info_frame = tk.LabelFrame(parent, text="像素顏色資訊")
        pixel_info_frame.pack(fill="x", pady=5)
        
        # 第一行：座標和RGB
        row1_frame = tk.Frame(pixel_info_frame)
        row1_frame.pack(fill="x", padx=5, pady=2)
        
        tk.Label(row1_frame, text="座標:", font=("Arial", 9, "bold")).pack(side="left")
        self.pixel_coord_label = tk.Label(row1_frame, text="(---, ---)", 
                                         font=("Courier", 10), fg="blue", width=10)
        self.pixel_coord_label.pack(side="left", padx=(5, 15))
        
        tk.Label(row1_frame, text="RGB:", font=("Arial", 9, "bold")).pack(side="left")
        self.pixel_rgb_label = tk.Label(row1_frame, text="(---, ---, ---)", 
                                       font=("Courier", 10), fg="red", width=15)
        self.pixel_rgb_label.pack(side="left", padx=(5, 10))
        
        # 顏色顯示方塊
        self.pixel_color_canvas = tk.Canvas(row1_frame, width=50, height=25, 
                                           relief="sunken", bd=2)
        self.pixel_color_canvas.pack(side="left", padx=(5, 0))
        
        # 第二行：HSV和灰階
        row2_frame = tk.Frame(pixel_info_frame)
        row2_frame.pack(fill="x", padx=5, pady=2)
        
        tk.Label(row2_frame, text="HSV:", font=("Arial", 9, "bold")).pack(side="left")
        self.pixel_hsv_label = tk.Label(row2_frame, text="(---, ---, ---)", 
                                       font=("Courier", 10), fg="purple", width=15)
        self.pixel_hsv_label.pack(side="left", padx=(5, 15))
        
        tk.Label(row2_frame, text="灰階:", font=("Arial", 9, "bold")).pack(side="left")
        self.pixel_gray_label = tk.Label(row2_frame, text="---", 
                                        font=("Courier", 10), fg="gray", width=6)
        self.pixel_gray_label.pack(side="left", padx=(5, 10))
        
        # 第三行：縮放資訊
        row3_frame = tk.Frame(pixel_info_frame)
        row3_frame.pack(fill="x", padx=5, pady=2)
        
        tk.Label(row3_frame, text="提示:", font=("Arial", 9, "bold")).pack(side="left")
        tip_text = "Ctrl+滾輪縮放 | 拖拽選擇子區域 | 滑鼠懸停查看像素"
        tk.Label(row3_frame, text=tip_text, font=("Arial", 8), fg="gray").pack(side="left", padx=(5, 0))

    def _on_canvas_mouse_move(self, event):
        """滑鼠在Canvas上移動時顯示像素顏色資訊"""
        try:
            # 轉換Canvas座標到原始圖像座標
            orig_x = int(event.x / self.roi_display_scale)
            orig_y = int(event.y / self.roi_display_scale)
            
            # 確保座標在圖像範圍內
            if (0 <= orig_x < self.roi_image_original.size[0] and 
                0 <= orig_y < self.roi_image_original.size[1]):
                
                # 獲取像素顏色 (RGB)
                pixel_rgb = self.roi_image_original.getpixel((orig_x, orig_y))
                if isinstance(pixel_rgb, int):  # 灰階圖像
                    pixel_rgb = (pixel_rgb, pixel_rgb, pixel_rgb)
                
                r, g, b = pixel_rgb[:3]  # 取前三個值（防止RGBA）
                
                # 計算HSV
                import colorsys
                h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
                h_deg = int(h * 360)
                s_pct = int(s * 100)
                v_pct = int(v * 100)
                
                # 計算灰階值
                gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                
                # 更新顯示
                self.pixel_coord_label.config(text=f"({orig_x:3d}, {orig_y:3d})")
                self.pixel_rgb_label.config(text=f"({r:3d}, {g:3d}, {b:3d})")
                self.pixel_hsv_label.config(text=f"({h_deg:3d}, {s_pct:2d}%, {v_pct:2d}%)")
                self.pixel_gray_label.config(text=f"{gray_value:3d}")
                
                # 顯示顏色方塊
                color_hex = f"#{r:02x}{g:02x}{b:02x}"
                self.pixel_color_canvas.delete("all")
                self.pixel_color_canvas.create_rectangle(0, 0, 40, 20, 
                                                        fill=color_hex, outline="black")
                
            else:
                # 座標超出範圍，清空顯示
                self.pixel_coord_label.config(text="(---, ---)")
                self.pixel_rgb_label.config(text="(---, ---, ---)")
                self.pixel_hsv_label.config(text="(---, ---, ---)")
                self.pixel_gray_label.config(text="---")
                self.pixel_color_canvas.delete("all")
                
        except Exception as e:
            print(f"顯示像素顏色時出錯: {e}")

    def _on_sub_roi_start(self, event):
        """開始選擇子區域"""
        if len(self.sub_regions) >= 3:
            messagebox.showinfo("提示", "最多只能選擇3個子區域")
            return
            
        self.drag_start = (event.x, event.y)
        
        # 創建拖拽矩形（紅色）
        self.current_sub_rect = self.roi_canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="red", width=2, tags="dragging"
        )

    def _on_sub_roi_drag(self, event):
        """拖拽過程中更新矩形"""
        if self.current_sub_rect and self.drag_start:
            x1, y1 = self.drag_start
            x2, y2 = event.x, event.y
            
            # 確保矩形有效（左上到右下）
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
                
            # 更新矩形
            self.roi_canvas.coords(self.current_sub_rect, x1, y1, x2, y2)

    def _on_sub_roi_end(self, event):
        """完成子區域選擇"""
        if not self.current_sub_rect or not self.drag_start:
            return
            
        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y
        
        # 確保矩形有效且有最小尺寸
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            # 矩形太小，刪除
            self.roi_canvas.delete(self.current_sub_rect)
            self.current_sub_rect = None
            self.drag_start = None
            return
        
        # 標準化座標
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        
        # 限制在Canvas範圍內
        canvas_w = self.roi_canvas.winfo_width()
        canvas_h = self.roi_canvas.winfo_height()
        x1 = max(0, min(x1, canvas_w))
        y1 = max(0, min(y1, canvas_h))
        x2 = max(0, min(x2, canvas_w))
        y2 = max(0, min(y2, canvas_h))
        
        # 轉換為原始ROI圖像座標
        orig_x1 = int(x1 / self.roi_display_scale)
        orig_y1 = int(y1 / self.roi_display_scale)
        orig_x2 = int(x2 / self.roi_display_scale)
        orig_y2 = int(y2 / self.roi_display_scale)
        
        # 添加到子區域列表
        self.sub_regions.append((orig_x1, orig_y1, orig_x2, orig_y2))
        
        # 改變矩形顏色為綠色（已確認）
        self.roi_canvas.itemconfig(self.current_sub_rect, outline="green", width=2)
        self.roi_canvas.dtag(self.current_sub_rect, "dragging")
        self.sub_region_rects.append(self.current_sub_rect)
        
        # 添加標籤
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        text_id = self.roi_canvas.create_text(
            center_x, center_y, text=str(len(self.sub_regions)), 
            fill="green", font=("Arial", 10, "bold")
        )
        self.sub_region_rects.append(text_id)
        
        print(f"新增子區域 {len(self.sub_regions)}: {orig_x1},{orig_y1} -> {orig_x2},{orig_y2}")
        
        # 重置拖拽狀態
        self.current_sub_rect = None
        self.drag_start = None

    def _clear_sub_regions(self):
        """清除所有子區域"""
        # 刪除Canvas上的矩形和標籤
        for rect_id in self.sub_region_rects:
            self.roi_canvas.delete(rect_id)
        
        # 清空列表
        self.sub_regions.clear()
        self.sub_region_rects.clear()
        
        print("已清除所有子區域")

    def _analyze_all_regions(self, roi_image: Image.Image, result_parent: tk.Widget):
        """分析完整ROI和所有子區域"""
        try:
            # 清空結果顯示區域
            for widget in self.result_content_frame.winfo_children():
                widget.destroy()
            
            # 分析完整ROI
            self._analyze_single_region(roi_image, None, "完整ROI", 0)
            
            # 分析各個子區域
            for i, (x1, y1, x2, y2) in enumerate(self.sub_regions, 1):
                try:
                    # 裁切子區域
                    sub_image = roi_image.crop((x1, y1, x2, y2))
                    self._analyze_single_region(sub_image, (x1, y1, x2, y2), f"子區域 {i}", i)
                except Exception as e:
                    print(f"裁切子區域 {i} 時出錯: {e}")
            
            # 更新滾動區域
            self.result_content_frame.update_idletasks()
            
        except Exception as e:
            print(f"分析所有區域時出錯: {e}")
            traceback.print_exc()

    def _analyze_single_region(self, image: Image.Image, coords: tuple, region_name: str, index: int):
        """分析單個區域並顯示結果"""
        try:
            # 創建結果框架
            result_frame = tk.LabelFrame(self.result_content_frame, text=region_name, 
                                        font=("Arial", 10, "bold"))
            result_frame.pack(fill="x", padx=5, pady=5)
            
            # 區域資訊
            info_frame = tk.Frame(result_frame)
            info_frame.pack(fill="x", padx=5, pady=2)
            
            if coords:
                size_info = f"座標: ({coords[0]},{coords[1]}) -> ({coords[2]},{coords[3]})"
                size_info += f" | 尺寸: {coords[2]-coords[0]}x{coords[3]-coords[1]}"
            else:
                size_info = f"完整ROI | 尺寸: {image.size[0]}x{image.size[1]}"
                
            tk.Label(info_frame, text=size_info, font=("Arial", 8), fg="gray").pack(anchor="w")
            
            # 圖像預覽
            preview_frame = tk.Frame(result_frame)
            preview_frame.pack(fill="x", padx=5, pady=2)
            
            # 縮放圖像用於預覽
            preview_size = (60, 40)
            if image.size[0] > 0 and image.size[1] > 0:
                ratio = min(preview_size[0]/image.size[0], preview_size[1]/image.size[1])
                preview_w = int(image.size[0] * ratio)
                preview_h = int(image.size[1] * ratio)
                
                preview_image = image.resize((preview_w, preview_h), Image.Resampling.LANCZOS)
                preview_photo = ImageTk.PhotoImage(preview_image)
                
                preview_label = tk.Label(preview_frame, image=preview_photo, relief="solid", bd=1)
                preview_label.image = preview_photo  # 保持引用
                preview_label.pack(side="left", padx=(0, 10))
            
            # OCR結果
            ocr_frame = tk.Frame(preview_frame)
            ocr_frame.pack(side="left", fill="x", expand=True)
            
            # 執行OCR
            start_time = time.time()
            ocr_result = self.ocr_iface.recognize(image)
            end_time = time.time()
            
            # 處理OCR結果
            if isinstance(ocr_result, tuple) and len(ocr_result) > 0:
                ocr_text = str(ocr_result[0])
            elif isinstance(ocr_result, str):
                ocr_text = ocr_result
            else:
                ocr_text = str(ocr_result) if ocr_result else ""
            
            # 顯示OCR結果
            tk.Label(ocr_frame, text="識別結果:", font=("Arial", 9, "bold")).pack(anchor="w")
            
            result_text = tk.Text(ocr_frame, height=2, width=30, wrap=tk.WORD, 
                                 font=("Arial", 11))
            result_text.pack(fill="x", pady=2)
            result_text.insert("1.0", ocr_text if ocr_text else "（無結果）")
            result_text.config(state=tk.DISABLED)
            
            # 處理時間和信心度
            time_text = f"耗時: {end_time - start_time:.3f}s"
            if hasattr(ocr_result, '__len__') and len(ocr_result) > 1:
                time_text += f" | 可信度: {ocr_result[1]:.2f}" if isinstance(ocr_result[1], (int, float)) else ""
            
            tk.Label(ocr_frame, text=time_text, font=("Arial", 8), fg="gray").pack(anchor="w")
            
            print(f"{region_name} OCR結果: '{ocr_text}' (耗時 {end_time - start_time:.3f}s)")
            
        except Exception as e:
            print(f"分析區域 {region_name} 時出錯: {e}")
            traceback.print_exc()
            
            # 顯示錯誤
            error_frame = tk.LabelFrame(self.result_content_frame, text=f"{region_name} - 錯誤", 
                                       fg="red")
            error_frame.pack(fill="x", padx=5, pady=2)
            tk.Label(error_frame, text=f"分析失敗: {e}", fg="red", wraplength=300).pack(padx=5, pady=2)

    # def _close_ocr_test_window(self):
    #     """關閉OCR測試視窗"""
    #     if self.ocr_test_window:
    #         try:
    #             self.ocr_test_window.destroy()
    #         except:
    #             pass
    #         self.ocr_test_window = None
    #     self.ocr_test_active = False
    #     self._update_status_bar("OCR測試視窗已關閉")
    def _close_ocr_test_window(self):
        """關閉OCR測試視窗"""
        if self.ocr_test_window:
            try:
                # 清理子區域相關屬性
                if hasattr(self, 'sub_regions'):
                    self.sub_regions.clear()
                if hasattr(self, 'sub_region_rects'):
                    self.sub_region_rects.clear()
                self.current_sub_rect = None
                self.drag_start = None
                
                self.ocr_test_window.destroy()
            except:
                pass
            self.ocr_test_window = None
        self.ocr_test_active = False
        self._update_status_bar("OCR精細測試視窗已關閉")

    def _get_current_frame_roi(self) -> Optional[Image.Image]:
        """獲取當前幀的ROI圖像"""
        try:
            if not self.cap_ui or not self.cap_ui.isOpened():
                print("UI VideoCapture 未開啟")
                return None
                
            if not self.roi_coords:
                print("ROI 坐標未設定")
                return None
                
            # 讀取當前幀
            self.cap_ui.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap_ui.read()
            if not ret:
                print(f"無法讀取幀 {self.current_frame_idx}")
                return None
                
            # 轉換為PIL圖像
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 裁切ROI
            roi_image = self._crop_roi(frame_pil)
            return roi_image
            
        except Exception as e:
            print(f"獲取當前幀ROI時出錯: {e}")
            traceback.print_exc()
            return None

    def _perform_ocr_test(self, roi_image: Image.Image, result_frame: tk.Frame):
        """執行OCR測試 - 修正OCR方法名稱"""
        # 清空之前的結果
        for widget in result_frame.winfo_children():
            widget.destroy()
        
        try:
            # 執行OCR - 修正方法名稱
            try:
                if hasattr(self.ocr_iface, 'recognize'):
                    # 使用 recognize 方法（返回 text, confidence）
                    ocr_result, confidence = self.ocr_iface.recognize(roi_image)
                elif hasattr(self.ocr_iface, 'predict'):
                    # 備用：如果有 predict 方法
                    ocr_result = self.ocr_iface.predict(roi_image)
                    confidence = getattr(self.ocr_iface, 'last_confidence', None)
                else:
                    # 如果都沒有，嘗試直接調用
                    ocr_result = str(self.ocr_iface(roi_image))
                    confidence = None
                    
            except Exception as ocr_error:
                print(f"OCR調用失敗: {ocr_error}")
                ocr_result = "〈OCR錯誤〉"
                confidence = None
            
            # 顯示結果
            result_text = ocr_result if ocr_result else "〈未識別〉"
            
            tk.Label(result_frame, text="識別結果:", 
                    font=("Arial", 12, "bold")).pack(anchor="w", pady=(5, 2))
            
            result_label = tk.Label(result_frame, text=result_text,
                                   font=("Arial", 14), fg="darkgreen" if ocr_result else "red")
            result_label.pack(anchor="w", pady=(0, 5))
            
            # 顯示置信度
            if confidence is not None:
                confidence_text = f"置信度: {confidence:.3f}"
                confidence_color = "darkgreen" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                tk.Label(result_frame, text=confidence_text,
                        font=("Arial", 10), fg=confidence_color).pack(anchor="w")
            
            # 顯示圖像信息
            img_info = f"圖像尺寸: {roi_image.size[0]} × {roi_image.size[1]} 像素"
            tk.Label(result_frame, text=img_info,
                    font=("Arial", 9), fg="gray").pack(anchor="w", pady=(5, 0))
            
            print(f"OCR測試結果: '{result_text}'")
            if confidence is not None:
                print(f"置信度: {confidence:.3f}")
                
        except Exception as e:
            print(f"OCR測試時出錯: {e}")
            traceback.print_exc()
            tk.Label(result_frame, text=f"測試失敗: {e}",
                    font=("Arial", 10), fg="red").pack(pady=10)

    def _load_video(self):
        """載入影片檔案"""
        filepath = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=[("影片檔案", "*.mp4 *.avi *.mov *.mkv"), ("所有檔案", "*.*")]
        )
        if not filepath:
            return

        self._clear_previous_video_data() 

        try:
            self.cap_ui = cv2.VideoCapture(filepath)
            if not self.cap_ui.isOpened():
                messagebox.showerror("錯誤", "無法開啟影片檔案 (UI Capture)")
                self.video_file_path = None
                return
            # meta_frames = int(self.cap_ui.get(cv2.CAP_PROP_FRAME_COUNT))
            # real_frames = 0
            # while self.cap_ui.grab():  # 使用 grab() 較快
            #     real_frames += 1
            # self.cap_ui.release()
            # if real_frames != meta_frames:
            #     print(f"⚠️ 幀數校正: {meta_frames} → {real_frames}")
            # self.total_frames = real_frames
            # print(f"_load_video self.total_frames: {self.total_frames}")
            self.total_frames = int(self.cap_ui.get(cv2.CAP_PROP_FRAME_COUNT))
            self.original_vid_w = int(self.cap_ui.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_vid_h = int(self.cap_ui.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap_ui.get(cv2.CAP_PROP_FPS)

            self.video_file_path = Path(filepath) 
            self.video_title = self.video_file_path.stem
            if hasattr(self.lbl_video_path, 'config'):
                 self.lbl_video_path.config(text=str(self.video_file_path.name))
            
            self._load_roi_config() 
            self._load_existing_data() 
            
            if hasattr(self, 'slider'):
                self.slider.config(to=self.total_frames - 1 if self.total_frames > 0 else 0, 
                                   state=tk.NORMAL if self.total_frames > 0 else tk.DISABLED)
            
            self.current_frame_idx = 0 
            if self.total_frames > 0:
                self._show_frame(0) 
            else:
                if hasattr(self, 'lbl_frame_num'): self.lbl_frame_num.config(text="幀: 0 / 0")
                if hasattr(self, 'lbl_video'): self.lbl_video.config(image=None)

            if self.total_frames > 0 and hasattr(self, 'btn_analyze'):
                 self.btn_analyze.config(state=tk.NORMAL)
            else:
                 if hasattr(self, 'btn_analyze'): self.btn_analyze.config(state=tk.DISABLED)

            self._update_status_bar(f"已載入: {self.video_title} ({self.total_frames} 幀, {fps:.1f} FPS)")
            print(f"影片載入成功: {self.total_frames} 幀, 解析度: {self.original_vid_w}x{self.original_vid_h}")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"載入影片失敗: {e}")
            print(f"載入影片失敗: {e}")
            traceback.print_exc()
            self.video_file_path = None 
            if hasattr(self, 'btn_analyze'): self.btn_analyze.config(state=tk.DISABLED)

    def _start_analysis_thread(self, tmad_threshold: float, diff_threshold: int):
        """啟動（單一）分析執行緒，傳入當前閾值"""
        if self.analysis_thread and self.analysis_thread.is_alive():
            print("警告：分析執行緒已在運行中。")
            return

        self.stop_event.clear()
        
        # 清空結果佇列，其他佇列不再使用
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        try:
            self.analysis_thread = threading.Thread(
                target=self._master_analysis_worker,
                args=(tmad_threshold, diff_threshold), # 傳遞閾值
                daemon=True,
                name="MasterAnalysisThread"
            )
            self.analysis_thread.start()
            print(f"主分析執行緒已啟動 (region: {self.region_name})")
            
        except Exception as e:
            print(f"啟動主分析執行緒失敗: {e}")
            self._update_status_bar(f"啟動分析失敗: {e}")
            if hasattr(self, 'btn_analyze'): self.btn_analyze.config(state=tk.NORMAL)
            if hasattr(self, 'btn_stop'): self.btn_stop.config(state=tk.DISABLED)


    def _master_analysis_worker(self, tmad_threshold_val: float, diff_threshold_val: int):
        """
        單一背景執行緒，智能分析模式：
        - 如果已有變化幀資料，只對變化幀進行OCR
        - 如果沒有變化幀資料，執行完整的變化偵測+OCR
        """
        print(f"主分析執行緒開始工作 (region: {self.region_name})")
        print(f"  使用 T-MAD 閾值: {tmad_threshold_val}, 忽略差異閾值: {diff_threshold_val}")

        worker_cap = None
        if not self.video_file_path:
            print("主分析執行緒錯誤：影片路徑未設定。")
            self.result_queue.put_nowait(("progress", 0, 0, "error_no_video"))
            return
        
        try:
            worker_cap = cv2.VideoCapture(str(self.video_file_path))
            if not worker_cap.isOpened():
                print(f"主分析執行緒錯誤：無法開啟影片 {self.video_file_path}")
                self.result_queue.put_nowait(("progress", 0, self.total_frames, "error_open_video"))
                return

            # 檢查是否已有變化幀資料
            existing_change_frames = [f for f, has_change in self.change_cache.items() if has_change]
            
            if existing_change_frames:
                # 智能模式：只對已知變化幀進行OCR
                print(f"🚀 智能分析模式：檢測到 {len(existing_change_frames)} 個既有變化幀，只進行OCR分析")
                self._ocr_only_analysis(worker_cap, existing_change_frames)
            else:
                # 完整模式：變化偵測 + OCR
                print(f"🔍 完整分析模式：未檢測到既有變化幀，執行完整分析 ({self.total_frames} 幀)")
                self._full_analysis(worker_cap, tmad_threshold_val, diff_threshold_val)

        except Exception as e:
            print(f"主分析執行緒發生錯誤: {e}")
            traceback.print_exc()
            try:
                self.result_queue.put_nowait(("progress", 0, self.total_frames, "error"))
            except queue.Full: 
                pass
        finally:
            if worker_cap:
                worker_cap.release()
            print(f"主分析執行緒結束，釋放VideoCapture。")
            self.after(0, self._check_analysis_completion_status)

    def _ocr_only_analysis(self, worker_cap: cv2.VideoCapture, change_frames: List[int]):
        """只對指定的變化幀進行OCR分析"""
        total_frames_to_process = len(change_frames)
        self.result_queue.put_nowait(("progress", 0, total_frames_to_process, "processing"))
        
        frames_processed = 0
        
        for i, frame_idx in enumerate(sorted(change_frames)):
            if self.stop_event.is_set():
                print(f"OCR分析在幀 {frame_idx} 被停止。")
                self.result_queue.put_nowait(("progress", frames_processed, total_frames_to_process, "stopped"))
                break

            try:
                # 執行OCR
                ocr_text = self._perform_ocr(frame_idx, worker_cap)
                self.ocr_cache[frame_idx] = ocr_text
                
                # 將OCR結果儲存到當前分析快取
                self.current_analysis_cache[frame_idx] = ocr_text
                try:
                    self.result_queue.put_nowait(("current_analysis", frame_idx, ocr_text))
                except queue.Full: 
                    pass

                frames_processed += 1
                
                # 更新進度
                try:
                    self.result_queue.put_nowait(("progress", frames_processed, total_frames_to_process, "processing"))
                except queue.Full: 
                    pass
                
                if frames_processed % 50 == 0:
                    print(f"OCR分析進度：已處理 {frames_processed}/{total_frames_to_process} 個變化幀")

            except Exception as e:
                print(f"OCR分析幀 {frame_idx} 時出錯: {e}")
                frames_processed += 1  # 仍然計入進度，避免卡住
        
        if not self.stop_event.is_set():
            print(f"✅ OCR分析完成，共處理 {total_frames_to_process} 個變化幀")
            self.result_queue.put_nowait(("progress", total_frames_to_process, total_frames_to_process, "completed"))

    def _full_analysis(self, worker_cap: cv2.VideoCapture, tmad_threshold_val: float, diff_threshold_val: int):
        """執行完整的變化偵測 + OCR分析"""
        self.result_queue.put_nowait(("progress", 0, self.total_frames, "processing"))
        frames_actually_processed = 0

        for frame_idx in range(self.total_frames):
            if self.stop_event.is_set():
                print(f"完整分析在幀 {frame_idx} 被停止。")
                self.result_queue.put_nowait(("progress", frames_actually_processed, self.total_frames, "stopped"))
                break

            try:
                # 1. 變化偵測
                has_change = self._detect_frame_change2(frame_idx, worker_cap, tmad_threshold_val, diff_threshold_val)
                self.change_cache[frame_idx] = has_change
                try:
                    self.result_queue.put_nowait(("change", frame_idx, has_change))
                except queue.Full: 
                    pass

                # 2. 如果有變化，執行 OCR
                if has_change:
                    ocr_text = self._perform_ocr(frame_idx, worker_cap)
                    self.ocr_cache[frame_idx] = ocr_text
                    
                    # 總是將OCR結果儲存到當前分析快取
                    self.current_analysis_cache[frame_idx] = ocr_text
                    try:
                        self.result_queue.put_nowait(("current_analysis", frame_idx, ocr_text))
                    except queue.Full: 
                        pass

                frames_actually_processed += 1
                
                # 3. 更新進度
                try:
                    self.result_queue.put_nowait(("progress", frames_actually_processed, self.total_frames, "processing"))
                except queue.Full: 
                    pass
                
                if frames_actually_processed % 200 == 0:
                    print(f"完整分析進度：已處理 {frames_actually_processed}/{self.total_frames} 幀")

            except Exception as e:
                print(f"完整分析幀 {frame_idx} 時出錯: {e}")
                frames_actually_processed += 1  # 仍然計入進度，避免卡住

        if not self.stop_event.is_set():
            print(f"✅ 完整分析完成，共處理 {self.total_frames} 幀")
            self.result_queue.put_nowait(("progress", self.total_frames, self.total_frames, "completed"))

    def _check_analysis_completion_status(self):
        """Called after master_analysis_worker finishes or is stopped."""
        if not (self.analysis_thread and self.analysis_thread.is_alive()):
            is_stopped_by_user = self.stop_event.is_set()
            # Check last progress message or rely on flags
            
            final_progress_val = self.progress_var.get()
            
            if not is_stopped_by_user and final_progress_val >= self.total_frames :
                self._on_analysis_complete() # Call if naturally completed
            elif is_stopped_by_user:
                self._update_status_bar("分析已手動停止。")
                if hasattr(self, 'btn_analyze'): self.btn_analyze.config(state=tk.NORMAL)
                if hasattr(self, 'btn_stop'): self.btn_stop.config(state=tk.DISABLED)
            else: # Ended due to error or incompletely
                self._update_status_bar("分析未完成或發生錯誤。")
                if hasattr(self, 'btn_analyze'): self.btn_analyze.config(state=tk.NORMAL)
                if hasattr(self, 'btn_stop'): self.btn_stop.config(state=tk.DISABLED)
    
    # _analysis_worker and _ocr_worker are removed.
    # _enqueue_frames_for_analysis is removed.

    def _save_roi_image(self, frame_idx: int, roi_pil: Image.Image):
        """儲存 ROI 圖像到檔案"""
        try:
            roi_dir = self._get_roi_dir(self.region_name) 
            png_path = roi_dir / f"frame_{frame_idx}.png"
            roi_pil.save(png_path, "PNG")
        except Exception as e:
            print(f"[ERR] 儲存 ROI 圖像 {frame_idx} 失敗: {e}")

    def _load_roi_from_file(self, frame_idx: int) -> Optional[Image.Image]:
        """從檔案載入 ROI 圖像"""
        try:
            roi_dir = self._get_roi_dir(self.region_name)
            png_path = roi_dir / f"frame_{frame_idx}.png"
            if png_path.exists():
                return Image.open(png_path)
            return None
        except Exception as e:
            print(f"[ERR] 讀取 ROI 圖像 {frame_idx} 失敗: {e}")
            return None

    def _calculate_binary_diff(self, img1: np.ndarray, img2: np.ndarray) -> float:
        if img1.shape != img2.shape:
            return 0.0
        b1 = (img1 > 127).astype(np.uint8)
        b2 = (img2 > 127).astype(np.uint8)
        diff = np.logical_xor(b1, b2)
        return float(np.mean(diff))

    def _show_frame(self, frame_idx: int):
        """
        顯示指定幀：
        - 預設模式：顯示整個frame並畫ROI紅框
        - 二值化模式：只顯示ROI區域的二值化圖，寬度與主畫面一致，高度等比例縮放並置中
        """
        if not self.cap_ui or not self.cap_ui.isOpened():
            print(f"警告：UI VideoCapture 未開啟或未設定，無法顯示幀 {frame_idx}")
            return
        if not (0 <= frame_idx < self.total_frames):
            return

        print(f"顯示幀: {frame_idx}")

        if not self.binarize_mode_var.get():
            # === 預設模式 ===
            self.cap_ui.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = self.cap_ui.read()
            if not ret:
                print(f"警告：無法讀取幀 {frame_idx}")
                return
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            disp_pil = frame_pil.resize((self.VID_W, self.VID_H), Image.BILINEAR)
            if self.roi_coords and self.original_vid_w > 0 and self.original_vid_h > 0:
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
        else:
            # === 二值化模式 ===
            roi_img = self._get_roi_image(frame_idx, self.cap_ui)
            if roi_img is None:
                print(f"無法取得 ROI 圖像: 幀 {frame_idx}")
                self.lbl_video.config(image=None)
            else:
                # 應用三種二值化方法之一
                bin_method = self.binarize_method_var.get()
                bin_img = self._apply_binarization(roi_img, bin_method)
                if bin_img is None:
                    print(f"二值化失敗，顯示原始 ROI")
                    bin_img = roi_img

                # --- 等比例放大 ROI 到 self.VID_W 寬 ---
                roi_w, roi_h = bin_img.size
                scale = self.VID_W / roi_w
                new_w = self.VID_W
                new_h = int(roi_h * scale)
                disp_pil = bin_img.resize((new_w, new_h), Image.NEAREST)

                # --- 建立黑底畫布，將 ROI 圖置中 ---
                canvas = Image.new("L" if disp_pil.mode == "L" else "RGB", (self.VID_W, self.VID_H), color=0)
                top = (self.VID_H - new_h) // 2
                canvas.paste(disp_pil, (0, top))
                self.current_display_image = ImageTk.PhotoImage(canvas)
                self.lbl_video.config(image=self.current_display_image)

        # --- 更新 Slider/Label 顯示 ---
        self.slider_var.set(frame_idx)
        self.lbl_frame_num.config(text=f"幀: {frame_idx} / {self.total_frames-1 if self.total_frames > 0 else 0}")
        self.current_frame_idx = frame_idx
        self.goto_var.set(frame_idx)

        # --- 顯示 diff 值與變化判定 ---
        diff_text = "Diff: -"
        if frame_idx > 0:
            roi_img1 = self._get_roi_image(frame_idx - 1, self.cap_ui)
            roi_img2 = self._get_roi_image(frame_idx, self.cap_ui)
            if roi_img1 and roi_img2:
                bin_method = self.binarize_method_var.get()
                bin1 = self._apply_binarization(roi_img1, bin_method)
                bin2 = self._apply_binarization(roi_img2, bin_method)
                if bin1 and bin2:
                    arr1 = np.array(bin1.convert("L"))
                    arr2 = np.array(bin2.convert("L"))
                    diff = self._calculate_binary_diff(arr1, arr2)
                    diff_text = f"Diff: {diff:.4f}"
        self.lbl_diff.config(text=diff_text)

        is_change = self.change_cache.get(frame_idx, False)
        self.lbl_change.config(text=f"變化判定: {'變化' if is_change else '未變化'}")

        # --- 控制提示圖示與焦點 ---
        if hasattr(self, 'control_hint_frame') and self.control_hint_frame:
            try:
                self.control_hint_frame.lift()
            except:
                pass
        self.master.focus_set()

    def _canvas_to_video_coords(self, canvas_x: int, canvas_y: int) -> tuple[int, int]:
        """
        將畫布座標轉換為原始影片座標
        """
        if self.original_vid_w <= 0 or self.original_vid_h <= 0:
            # 如果沒有原始影片尺寸資訊，直接返回畫布座標
            return canvas_x, canvas_y
        
        # 計算縮放比例
        scale_x = self.original_vid_w / self.VID_W
        scale_y = self.original_vid_h / self.VID_H
        
        # 轉換座標
        video_x = int(canvas_x * scale_x)
        video_y = int(canvas_y * scale_y)
        
        # 確保座標在有效範圍內
        video_x = max(0, min(video_x, self.original_vid_w - 1))
        video_y = max(0, min(video_y, self.original_vid_h - 1))
        
        return video_x, video_y
    
    def _on_roi_start(self, event):
        """Records the starting coordinates for ROI selection."""
        video_x, video_y = self._canvas_to_video_coords(event.x, event.y)
        self.roi_start_coords = (video_x, video_y)
        # self.roi_start_coords = (event.x, event.y)
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
        """
        使用者在畫面上拖曳完 ROI 框後呼叫：
        1. 儲存新的 ROI 座標
        2. 清空與 ROI 相關的快取與欄位
        3. 重新啟動背景分析執行緒
        """
        if self.roi_start_coords is None:
            self._show_frame(self.current_frame_idx)
            return

        # 計算並驗證ROI座標
        start_x_orig, start_y_orig = self.roi_start_coords
        end_x_orig, end_y_orig = self._canvas_to_video_coords(event.x, event.y)

        x1 = min(start_x_orig, end_x_orig)
        y1 = min(start_y_orig, end_y_orig)
        x2 = max(start_x_orig, end_x_orig)
        y2 = max(start_y_orig, end_y_orig)

        x1 = max(0, min(x1, self.original_vid_w - 1))
        y1 = max(0, min(y1, self.original_vid_h - 1))
        x2 = max(0, min(x2, self.original_vid_w - 1))
        y2 = max(0, min(y2, self.original_vid_h - 1))

        new_roi = (x1, y1, x2, y2)
        self.roi_start_coords = None

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            print("ROI 太小，已忽略。")
            self._show_frame(self.current_frame_idx)
            return

        # 儲存ROI變更
        if new_roi != self.roi_coords:
            self.roi_coords = new_roi
            self.roi_dict[self.region_name] = list(self.roi_coords)
            
            # 拖曳ROI後，詢問是否要儲存
            result = messagebox.askyesno("儲存設定", f"ROI區域已更新，是否儲存到配置檔案？")
            if result:
                self._save_roi_config()

            # 清空快取
            self.change_cache.clear()
            self.ocr_cache.clear()
            self.roi_image_cache.clear()

            # 停止並重啟分析
            self.stop_event.set()
            for th_name in ["analysis_thread", "ocr_thread"]:
                th = getattr(self, th_name, None)
                if th and th.is_alive():
                    th.join(timeout=1.0)
            self.stop_event.clear()

            # 更新UI
            self._update_roi_fields()
            status_msg = f"{self.region_name} ROI 更新: {self.roi_coords}"
            if result:
                status_msg += " (已儲存)"
            else:
                status_msg += " (未儲存)"
            self._update_status_bar(status_msg)
        
        self._show_frame(self.current_frame_idx)

    def _normalize_roi_coords(self, end_x: int, end_y: int) -> tuple[int, int, int, int]: # 參數改為 canvas 座標
        """
        依照拖曳起點 (self.roi_start_coords，已是原始影片座標) 與結束的 canvas 座標 (end_x,end_y)，
        轉回影片原尺寸座標 (x1,y1,x2,y2)，並裁切在有效範圍內。
        """
        if self.roi_start_coords is None: # 防呆
            # 如果沒有起點，嘗試將結束點作為一個小範圍的中心 (或直接返回錯誤/預設)
            # 這裡假設不應該發生，因為 _on_roi_end 會先檢查
            print("警告: _normalize_roi_coords 被呼叫時 roi_start_coords 為 None")
            # 安全起見，返回一個無效或預設的 ROI
            return 0,0,0,0


        start_x_orig, start_y_orig = self.roi_start_coords
        end_x_orig, end_y_orig = self._canvas_to_video_coords(end_x, end_y) # 將 canvas 座標轉為影片座標

        x1 = min(start_x_orig, end_x_orig)
        y1 = min(start_y_orig, end_y_orig)
        x2 = max(start_x_orig, end_x_orig)
        y2 = max(start_y_orig, end_y_orig)

        # 邊界檢查 (雖然 _canvas_to_video_coords 應該已經處理了)
        x1 = max(0, min(x1, self.original_vid_w - 1))
        y1 = max(0, min(y1, self.original_vid_h - 1))
        x2 = max(0, min(x2, self.original_vid_w - 1))
        y2 = max(0, min(y2, self.original_vid_h - 1))
        return x1, y1, x2, y2

    def _crop_roi(self, frame_pil_full: Image.Image) -> Optional[Image.Image]:
        """從完整幀中裁切 ROI 區域"""
        if not self.roi_coords:
            return None
        try:
            # 使用存儲的原始坐標進行裁剪
            x1, y1, x2, y2 = self.roi_coords
            # 確保坐標是整數且在有效範圍內
            width, height = frame_pil_full.size
            x1 = max(0, min(int(x1), width))
            y1 = max(0, min(int(y1), height))
            x2 = max(x1, min(int(x2), width))
            y2 = max(y1, min(int(y2), height))
            
            roi_pil = frame_pil_full.crop((x1, y1, x2, y2))
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

    def _step_frame(self, delta: int):
        """切換到相對當前幀的指定偏移幀 - 單純的幀切換"""
        if self.total_frames == 0:
            print("無影片載入，無法切換幀")
            return
            
        old_idx = self.current_frame_idx
        new_idx = max(0, min(self.total_frames-1, self.current_frame_idx + delta))
        
        # 添加調試信息
        print(f"單純幀切換: {old_idx} -> {new_idx} (delta: {delta})")
        
        if new_idx != old_idx:
            self._show_frame(new_idx)
            print(f"已切換到幀 {new_idx}")
        else:
            if delta > 0 and new_idx == self.total_frames - 1:
                print("已到達最後一幀")
            elif delta < 0 and new_idx == 0:
                print("已到達第一幀")

    def _jump_to_previous_change(self):
        """跳轉到前一個變化幀"""
        if not self.change_cache:
            print("沒有變化幀資料")
            messagebox.showinfo("提示", "沒有變化幀資料，請先進行分析")
            return
        
        # 獲取所有變化幀，按幀號排序
        change_frames = sorted([frame for frame, has_change in self.change_cache.items() if has_change])
        
        if not change_frames:
            print("沒有檢測到變化幀")
            messagebox.showinfo("提示", "沒有檢測到變化幀")
            return
        
        current_frame = self.current_frame_idx
        
        # 找到當前幀之前的最後一個變化幀
        previous_change = None
        for frame in reversed(change_frames):
            if frame < current_frame:
                previous_change = frame
                break
        
        if previous_change is not None:
            print(f"跳轉到前一個變化幀: {current_frame} -> {previous_change}")
            self._show_frame(previous_change)
            self._highlight_treeview_item(previous_change)
        else:
            # 如果沒有更早的變化幀，跳到最後一個變化幀（循環）
            if change_frames:
                last_change = change_frames[-1]
                print(f"沒有更早的變化幀，跳轉到最後一個變化幀: {current_frame} -> {last_change}")
                self._show_frame(last_change)
                self._highlight_treeview_item(last_change)
            else:
                print("沒有更早的變化幀")

    def _jump_to_next_change(self):
        """跳轉到後一個變化幀"""
        if not self.change_cache:
            print("沒有變化幀資料")
            messagebox.showinfo("提示", "沒有變化幀資料，請先進行分析")
            return
        
        # 獲取所有變化幀，按幀號排序
        change_frames = sorted([frame for frame, has_change in self.change_cache.items() if has_change])
        
        if not change_frames:
            print("沒有檢測到變化幀")
            messagebox.showinfo("提示", "沒有檢測到變化幀")
            return
        
        current_frame = self.current_frame_idx
        
        # 找到當前幀之後的第一個變化幀
        next_change = None
        for frame in change_frames:
            if frame > current_frame:
                next_change = frame
                break
        
        if next_change is not None:
            print(f"跳轉到後一個變化幀: {current_frame} -> {next_change}")
            self._show_frame(next_change)
            self._highlight_treeview_item(next_change)
        else:
            # 如果沒有更晚的變化幀，跳到第一個變化幀（循環）
            if change_frames:
                first_change = change_frames[0]
                print(f"沒有更晚的變化幀，跳轉到第一個變化幀: {current_frame} -> {first_change}")
                self._show_frame(first_change)
                self._highlight_treeview_item(first_change)
            else:
                print("沒有更晚的變化幀")

    def _highlight_treeview_item(self, frame_idx: int):
        """在TreeView中高亮顯示指定幀的項目"""
        try:
            # 查找對應的TreeView項目
            for item in self.tree.get_children():
                item_frame = int(self.tree.set(item, "frame"))
                if item_frame == frame_idx:
                    # 選中並確保可見
                    self.tree.selection_set(item)
                    self.tree.see(item)
                    print(f"在TreeView中高亮顯示幀 {frame_idx}")
                    return
            print(f"在TreeView中找不到幀 {frame_idx}")
        except Exception as e:
            print(f"高亮TreeView項目時出錯: {e}")

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

    def _load_annotations(self, region: str):
        """載入指定 region 的標註到 self.annotations"""
        self.annotations.clear()
        path = self._get_annotations_path(region)
        if path and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                obj = json.loads(line.strip())
                                
                                # 嘗試不同的欄位名稱
                                frame_idx = None
                                text_content = ""
                                
                                # 處理新格式：{"query": "<image>", "response": "28 40", "images": "path/frame_123.png"}
                                if "images" in obj and "response" in obj:
                                    # 從 images 路徑中提取 frame 編號
                                    image_path = obj["images"]
                                    if "frame_" in image_path:
                                        frame_str = image_path.split("frame_")[-1].split(".")[0]
                                        try:
                                            frame_idx = int(frame_str)
                                            text_content = obj.get("response", "")
                                        except ValueError:
                                            print(f"無法解析幀編號: {frame_str}")
                                            continue
                                    else:
                                        print(f"第 {line_num} 行圖片路徑格式不正確: {image_path}")
                                        continue
                                
                                # 處理舊格式：{"frame": 123, "text": "content"} 或其他變體
                                elif "frame" in obj:
                                    frame_idx = int(obj["frame"])
                                    text_content = obj.get("ocr_text", obj.get("text", ""))
                                else:
                                    print(f"警告: 第 {line_num} 行找不到 frame 欄位，跳過")
                                    continue
                                
                                if frame_idx is not None:
                                    self.annotations[frame_idx] = text_content
                                
                            except json.JSONDecodeError as je:
                                print(f"第 {line_num} 行 JSON 解析錯誤: {je}")
                                continue
                            except (KeyError, ValueError) as ke:
                                print(f"第 {line_num} 行資料格式錯誤: {ke}")
                                print(f"行內容: {line.strip()}")
                                continue
                                
                print(f"已載入 {len(self.annotations)} 個標註記錄 (region: {region})")
            except Exception as e:
                print(f"讀取標註檔失敗: {e}")
                traceback.print_exc()
        else:
            print(f"標註檔案不存在: {path}")

    def _refresh_treeview(self):
        """重新載入標註檔案並顯示在 Treeview 中"""
        # 清空現有表格
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 載入標註檔案
        if self.video_file_path and self.region_name:
            self._load_annotations(self.region_name)
            self._load_change_frames(self.region_name)
        
        # 如果變化幀為空但標註存在，從標註推導變化幀
        if not any(self.change_cache.values()) and self.annotations:
            print("變化幀檔案為空，從標註檔案推導變化幀...")
            for frame_idx in self.annotations.keys():
                self.change_cache[frame_idx] = True
        
        # 根據 change_cache 建立表格項目
        change_frames = [idx for idx, has_change in self.change_cache.items() if has_change]
        
        for frame_idx in change_frames:
            # 取得標註內容
            content = self.annotations.get(frame_idx, "")
            
            # 先插入項目，Diff 欄位暫時顯示 "計算中..."
            item_id_str = f"F{frame_idx}"
            self.tree.insert("", "end", iid=item_id_str, 
                            values=(frame_idx, "計算中...", content, ""))
            
            # 應用 "changed" 標籤
            self.tree.item(item_id_str, tags=("changed",))
        
        # # 逐步計算 Diff 值
        # if change_frames:
        #     self._calculate_diffs_gradually(change_frames)

    def _calculate_diffs_gradually(self, change_frames: List[int]):
        """逐步計算 Diff 值，不卡住 UI"""
        def process_one_frame(index):
            if index >= len(change_frames):
                print(f"Diff 計算完成，共處理 {len(change_frames)} 個變化幀")
                return  # 完成
            
            frame_idx = change_frames[index]
            item_id_str = f"F{frame_idx}"
            
            if self.tree.exists(item_id_str):
                # 計算 diff 值
                diff_value = self._calculate_frame_diff(frame_idx)
                
                # 更新 Diff 欄位
                self.tree.set(item_id_str, "diff", diff_value)
                
                # 如果 diff < 0.01，加上紅字 tag
                current_tags = list(self.tree.item(item_id_str, "tags"))
                if diff_value != "-" and float(diff_value) < 0.01:
                    if "small_diff" not in current_tags:
                        current_tags.append("small_diff")
                else:
                    if "small_diff" in current_tags:
                        current_tags.remove("small_diff")
                self.tree.item(item_id_str, tags=tuple(current_tags))
            
            # 繼續處理下一個 frame
            self.after(10, lambda: process_one_frame(index + 1))
        
        # 開始處理
        print(f"開始逐步計算 {len(change_frames)} 個變化幀的 Diff 值...")
        process_one_frame(0)

    def _start_background_threads(self):
        """啟動背景分析執行緒"""
        if not self.video_file_path or not self.roi_coords:
            print("無法啟動背景執行緒：影片或 ROI 未設定")
            return
            
        # 停止現有執行緒
        self.stop_event.set()
        for th_name in ["analysis_thread", "ocr_thread"]:
            thread = getattr(self, th_name, None)
            if thread and thread.is_alive():
                thread.join(timeout=1.0)
        self.stop_event.clear()
        
        # 清空隊列
        for q in [self.detect_queue, self.ocr_queue, self.result_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        
        # 啟動新執行緒
        try:
            self.analysis_thread = threading.Thread(
                target=self._analysis_worker,
                daemon=True,
                name="AnalysisThread"
            )
            self.ocr_thread = threading.Thread(
                target=self._ocr_worker,
                daemon=True,
                name="OCRThread"
            )
            
            self.analysis_thread.start()
            self.ocr_thread.start()
            
            print("背景分析執行緒已啟動")
            self._update_status_bar("背景分析已啟動")
            
        except Exception as e:
            print(f"啟動背景執行緒失敗: {e}")
            self._update_status_bar(f"啟動分析失敗: {e}")


    def _detect_frame_change(self, frame_idx: int, video_capture_for_roi: cv2.VideoCapture, 
                           tmad_threshold: float, diff_threshold: int) -> bool:
        """
        偵測指定幀是否有變化 - 使用 PIL 圖像和 T-MAD。
        tmad_threshold 和 diff_threshold 作為參數傳入。
        """
        try:
            if not video_capture_for_roi or not video_capture_for_roi.isOpened():
                print(f"錯誤: _detect_frame_change 的 video_capture_for_roi 無效 (frame {frame_idx})")
                return False 

            if frame_idx == 0:
                # For the first frame, always consider it changed and cache its ROI
                first_frame_roi = self._get_roi_image(frame_idx, video_capture_for_roi)
                if first_frame_roi is None: 
                    print(f"警告: 無法為第一幀 {frame_idx} 獲取ROI圖像。")
                    # Consider returning True to flag it, or False if ROI is critical for subsequent steps.
                    # If first frame ROI is essential for OCR, returning True allows OCR attempt.
                    return True 
                self.roi_image_cache[frame_idx] = first_frame_roi
                return True
            
            curr_roi = self._get_roi_image(frame_idx, video_capture_for_roi)
            # Try to get previous ROI from cache first
            prev_roi = self.roi_image_cache.get(frame_idx - 1)
            # If not in cache, get it from video capture (this might be redundant if _get_roi_image handles caching well)
            if prev_roi is None:
                prev_roi = self._get_roi_image(frame_idx - 1, video_capture_for_roi)
            
            if curr_roi is None or prev_roi is None:
                print(f"無法取得幀 {frame_idx} 或 {frame_idx-1} 的 ROI 圖像進行變化偵測")
                # Mark as change if ROIs are not available, to allow OCR attempt or manual check
                return True 
            
            # 使用傳入的閾值
            has_change = self.change_iface.is_changed(
                prev_roi, curr_roi, 
                tmad_threshold=tmad_threshold, # 使用傳入的 tmad_threshold
                diff_threshold=diff_threshold  # 使用傳入的 diff_threshold
            )
            return has_change
            
        except Exception as e:
            print(f"變化偵測錯誤 (frame {frame_idx}): {e}")
            traceback.print_exc()
            return False # 保守返回 False

    def _detect_frame_change2(self, frame_idx, video_capture_for_roi):
        """二值化後diff值超過0.01視為有變化"""
        if frame_idx == 0:
            return False
        roi_img1 = self._get_roi_image(frame_idx - 1, video_capture_for_roi)
        roi_img2 = self._get_roi_image(frame_idx, video_capture_for_roi)
        if roi_img1 and roi_img2:
            bin_method = self.binarize_method_var.get()
            bin1 = self._apply_binarization(roi_img1, bin_method)
            bin2 = self._apply_binarization(roi_img2, bin_method)
            if bin1 and bin2:
                arr1 = np.array(bin1.convert("L"))
                arr2 = np.array(bin2.convert("L"))
                diff = self._calculate_binary_diff(arr1, arr2)
                return diff > 0.01
        return False
    
        # --- 顯示 diff 值與變化判定 ---
        # diff_text = "Diff: -"
        # if frame_idx > 0:
        #     roi_img1 = self._get_roi_image(frame_idx - 1, self.cap_ui)
        #     roi_img2 = self._get_roi_image(frame_idx, self.cap_ui)
        #     if roi_img1 and roi_img2:
        #         bin_method = self.binarize_method_var.get()
        #         bin1 = self._apply_binarization(roi_img1, bin_method)
        #         bin2 = self._apply_binarization(roi_img2, bin_method)
        #         if bin1 and bin2:
        #             arr1 = np.array(bin1.convert("L"))
        #             arr2 = np.array(bin2.convert("L"))
        #             diff = self._calculate_binary_diff(arr1, arr2)
        #             diff_text = f"Diff: {diff:.4f}"
        # self.lbl_diff.config(text=diff_text)


    def _get_next_unanalyzed_frame(self) -> Optional[int]:
        """取得下一個未分析的幀"""
        for i in range(self.total_frames):
            if i not in self.change_cache:
                return i
        return None

    def _show_compare_roi_on_canvas(self, canvas, frame_idx, result_dict, show_new=True):
        roi_img = self._get_roi_image(frame_idx, self.cap_ui)
        if roi_img is None:
            canvas.delete("all")
            return
        # 這裡可根據 show_new 決定是否要二值化
        if show_new and result_dict.get(frame_idx, False):
            bin_img = self._apply_binarization(roi_img, self.binarize_method_var.get())
            img = bin_img if bin_img else roi_img
        else:
            img = roi_img
        # 顯示到canvas
        img = img.resize((800, 450))
        self._tkimg = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor="nw", image=self._tkimg)

    def _run_compare_analysis_gen(self):
        """生成器：逐步比較新舊frame change結果"""
        new_result = {}
        total = self.total_frames
        for idx in range(total):
            has_change = self._detect_frame_change2(idx, self.cap_ui)
            new_result[idx] = has_change
            yield idx, has_change, new_result

    def _open_compare_analysis_window(self):
        import tkinter as tk
        from tkinter import ttk

        win = tk.Toplevel(self)
        win.title("比較分析（新舊方法）")
        win.geometry("1150x750")
        win.grab_set()

        # ROI顯示區
        roi_canvas = tk.Canvas(win, width=800, height=450, bg="black")
        roi_canvas.grid(row=0, column=1, rowspan=3, padx=10, pady=10)

        # 當前frame號碼顯示與編輯
        frame_idx_frame = tk.Frame(win)
        frame_idx_frame.grid(row=3, column=1, sticky="n")
        tk.Label(frame_idx_frame, text="當前Frame:").pack(side="left")
        frame_idx_var = tk.StringVar(value="0")
        frame_idx_entry = tk.Entry(frame_idx_frame, width=8, textvariable=frame_idx_var, justify="center")
        frame_idx_entry.pack(side="left")
        tk.Label(frame_idx_frame, text=f"/ {self.total_frames-1}").pack(side="left")

        # 差異frame Listbox
        tk.Label(win, text="差異幀號").grid(row=0, column=0)
        diff_listbox = tk.Listbox(win, width=12, height=25)
        diff_listbox.grid(row=1, column=0, sticky="n")

        # 新舊變化幀 Listbox
        tk.Label(win, text="舊方法變化幀").grid(row=0, column=2)
        old_listbox = tk.Listbox(win, width=12, height=12)
        old_listbox.grid(row=1, column=2, sticky="n")
        tk.Label(win, text="新方法變化幀").grid(row=2, column=2)
        new_listbox = tk.Listbox(win, width=12, height=12)
        new_listbox.grid(row=3, column=2, sticky="n")

        # ROI切換按鈕
        btn_frame = tk.Frame(win)
        btn_frame.grid(row=4, column=1, pady=5)
        btn_prev = tk.Button(btn_frame, text="<< 前一幀")
        btn_prev.pack(side="left", padx=5)
        btn_next = tk.Button(btn_frame, text="下一幀 >>")
        btn_next.pack(side="left", padx=5)
        btn_show_old = tk.Button(btn_frame, text="顯示舊分析ROI")
        btn_show_old.pack(side="left", padx=5)
        btn_show_new = tk.Button(btn_frame, text="顯示新分析ROI")
        btn_show_new.pack(side="left", padx=5)

        # 進度條與數字
        progress_frame = tk.Frame(win)
        progress_frame.grid(row=5, column=1, pady=5)
        progress = ttk.Progressbar(progress_frame, length=300)
        progress.pack(side="left")
        progress_label = tk.Label(progress_frame, text="0/0")
        progress_label.pack(side="left", padx=10)

        # 狀態
        diff_frames = []
        old_change_frames = []
        new_change_frames = []
        new_result = {}
        gen = self._run_compare_analysis_gen()

        # ROI顯示狀態
        current_frame = [0]
        show_new = [True]

        def show_roi(frame_idx, use_new):
            roi_img = self._get_roi_image(frame_idx, self.cap_ui)
            if roi_img is None:
                roi_canvas.delete("all")
                return
            if use_new and new_result.get(frame_idx, False):
                bin_img = self._apply_binarization(roi_img, self.binarize_method_var.get())
                img = bin_img if bin_img else roi_img
            else:
                img = roi_img
            img = img.resize((800, 450))
            self._tkimg = ImageTk.PhotoImage(img)
            roi_canvas.delete("all")
            roi_canvas.create_image(0, 0, anchor="nw", image=self._tkimg)
            current_frame[0] = frame_idx
            show_new[0] = use_new
            frame_idx_var.set(str(frame_idx))

        def goto_frame(frame_idx, use_new):
            if 0 <= frame_idx < self.total_frames:
                show_roi(frame_idx, use_new)
                # 同步Listbox選中
                for lb in [diff_listbox, old_listbox, new_listbox]:
                    try:
                        idx = lb.get(0, "end").index(frame_idx)
                        lb.selection_clear(0, "end")
                        lb.selection_set(idx)
                        lb.see(idx)
                    except ValueError:
                        lb.selection_clear(0, "end")

        def on_listbox_select(lb, use_new):
            sel = lb.curselection()
            if not sel: return
            frame_idx = int(lb.get(sel[0]))
            goto_frame(frame_idx, use_new)

        diff_listbox.bind("<<ListboxSelect>>", lambda e: on_listbox_select(diff_listbox, True))
        old_listbox.bind("<<ListboxSelect>>", lambda e: on_listbox_select(old_listbox, False))
        new_listbox.bind("<<ListboxSelect>>", lambda e: on_listbox_select(new_listbox, True))

        btn_show_old.config(command=lambda: show_roi(current_frame[0], False))
        btn_show_new.config(command=lambda: show_roi(current_frame[0], True))
        btn_prev.config(command=lambda: goto_frame(max(0, current_frame[0] - 1), show_new[0]))
        btn_next.config(command=lambda: goto_frame(min(self.total_frames - 1, current_frame[0] + 1), show_new[0]))

        # 鍵盤左右鍵切換
        def on_key(event):
            if event.keysym == "Left":
                goto_frame(max(0, current_frame[0] - 1), show_new[0])
            elif event.keysym == "Right":
                goto_frame(min(self.total_frames - 1, current_frame[0] + 1), show_new[0])
        win.bind("<Left>", on_key)
        win.bind("<Right>", on_key)

        # 支援直接輸入frame號碼跳轉
        def on_frame_idx_entry(event):
            try:
                idx = int(frame_idx_var.get())
                if 0 <= idx < self.total_frames:
                    goto_frame(idx, show_new[0])
            except Exception:
                pass
        frame_idx_entry.bind("<Return>", on_frame_idx_entry)

        # 分析進度
        def step():
            try:
                idx, has_change, new_result_local = next(gen)
                new_result.update(new_result_local)
                if self.change_cache.get(idx, False):
                    old_change_frames.append(idx)
                if has_change:
                    new_change_frames.append(idx)
                if has_change != self.change_cache.get(idx, False):
                    diff_frames.append(idx)
                    diff_listbox.insert("end", idx)
                progress["maximum"] = self.total_frames
                progress["value"] = idx + 1
                progress_label.config(text=f"{idx+1}/{self.total_frames}")
                win.update()
                win.after(1, step)
            except StopIteration:
                # 填入新舊變化幀
                for idx in old_change_frames:
                    old_listbox.insert("end", idx)
                for idx in new_change_frames:
                    new_listbox.insert("end", idx)
                progress["value"] = self.total_frames
                progress_label.config(text=f"{self.total_frames}/{self.total_frames}")
                # 印出分析資訊
                print("==== 舊方法變化幀 ====")
                print(old_change_frames)
                print("==== 新方法變化幀 ====")
                print(new_change_frames)
                print("==== 差異幀 ====")
                print(diff_frames)
                tk.messagebox.showinfo("比較完成", f"新舊分析結果有 {len(diff_frames)} 個幀不同。")
                # 預設顯示第一個差異幀
                if diff_frames:
                    goto_frame(diff_frames[0], True)
                elif new_change_frames:
                    goto_frame(new_change_frames[0], True)
                else:
                    goto_frame(0, True)

        step()

    def _start_analysis(self):
        if self._has_existing_data():
            self._open_compare_analysis_window()
            return
            
        """開始分析當前區域的變化幀和OCR - 智能版本"""
        if not self.video_file_path or not self.roi_coords:
            messagebox.showwarning("警告", "請先載入影片並設定ROI區域")
            return
        
        # 檢查現有變化幀資料
        existing_change_frames = [f for f, has_change in self.change_cache.items() if has_change]
        
        if existing_change_frames:
            # 智能模式提示
            result = messagebox.askyesno(
                "智能分析模式", 
                f"檢測到 {len(existing_change_frames)} 個既有變化幀。\n\n" +
                "📊 智能模式：只對變化幀重新執行OCR分析\n" +
                "🔄 完整模式：重新執行變化偵測 + OCR分析\n\n" +
                "選擇 [是] 使用智能模式（推薦，速度更快）\n" +
                "選擇 [否] 使用完整模式（重新分析所有幀）"
            )
            
            if not result:
                # 用戶選擇完整模式，清空變化快取
                print("用戶選擇完整分析模式，清空變化快取")
                self.change_cache.clear()
        
        # 清空當前分析結果（無論哪種模式都需要重新生成）
        if self.current_analysis_cache:
            clear_result = messagebox.askyesno(
                "清空當前分析", 
                "檢測到當前分析欄位有資料。\n\n是否清空重新開始？"
            )
            if clear_result:
                self.current_analysis_cache.clear()
                print("已清空當前分析結果")
        
        # 清空OCR快取（需要重新計算）
        self.ocr_cache.clear()
        
        # 更新按鈕狀態
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
        # 重置停止事件
        self.stop_event.clear()
        
        # 啟動分析線程
        self._start_analysis_thread(self.tmad_threshold_var.get(), self.diff_threshold_var.get())
        
        # 根據模式顯示不同的狀態訊息
        if existing_change_frames and self.change_cache:  # 如果change_cache沒被清空，說明是智能模式
            self._update_status_bar(f"開始智能分析 - 區域 {self.region_name} ({len(existing_change_frames)} 個變化幀)")
            print(f"🚀 開始智能分析 - 區域: {self.region_name}, 變化幀數: {len(existing_change_frames)}")
        else:
            self._update_status_bar(f"開始完整分析 - 區域 {self.region_name}")
            print(f"🔍 開始完整分析 - 區域: {self.region_name}, ROI: {self.roi_coords}")

    def _has_existing_data(self) -> bool:
        """檢查是否存在現有的分析資料"""
        if not self.video_file_path:
            return False
        
        ocr_path = Path("data") / self.video_title / f"{self.region_name}_ocr.json"
        
        return ocr_path.exists()

    def _backup_current_data(self):
        """備份當前資料用於比較"""
        # 備份現有的標註和變化快取
        self.old_annotations = self.annotations.copy()
        self.old_change_cache = self.change_cache.copy()
        print(f"已備份現有資料：{len(self.old_annotations)} 個標註，{len(self.old_change_cache)} 個變化記錄")

    def _on_analysis_complete(self):
        """分析自然完成後的處理 - 不自動儲存當前分析結果"""
        print("主分析執行緒回報：分析自然完成。")
        if hasattr(self, 'btn_analyze'): 
            self.btn_analyze.config(state=tk.NORMAL if self.video_file_path else tk.DISABLED)
        if hasattr(self, 'btn_stop'): 
            self.btn_stop.config(state=tk.DISABLED)
        
        # 輸出簡單的分析統計
        total_changes = len([f for f, c in self.change_cache.items() if c])
        total_ocr = len(self.current_analysis_cache)
        
        print(f"\n📊 分析完成統計:")
        print(f"   - 檢測到變化幀: {total_changes}")
        print(f"   - OCR識別結果: {total_ocr}")
        print(f"   - 區域: {self.region_name}")
        print(f"   - OCR模型: {self.ocr_model_var.get()}")
        
        # 比較差異（如果有現有標註的話）
        if self.annotations:
            same_content = 0
            different_content = 0
            new_detections = 0
            
            for frame_idx in self.current_analysis_cache:
                old_content = self.annotations.get(frame_idx, "").strip()
                new_content = self.current_analysis_cache[frame_idx].strip()
                
                if frame_idx in self.annotations:
                    if old_content == new_content:
                        same_content += 1
                    else:
                        different_content += 1
                else:
                    new_detections += 1
            
            print(f"\n🔍 與現有標註比較:")
            print(f"   - 內容相同: {same_content}")
            print(f"   - 內容不同: {different_content}")
            print(f"   - 新檢測到: {new_detections}")
            
            if different_content > 0:
                print(f"\n⚠️  有 {different_content} 個幀的內容與現有標註不同，請檢視後決定是否儲存")
        
        print(f"\n💾 請檢視'當前分析'欄位的結果，確認無誤後按'儲存標註'")
        
        # 只儲存變化幀資料，不儲存 annotations（包含 current_analysis_cache）
        if self.video_file_path and self.region_name and self.change_cache:
            try:
                self._save_change_frames(self.region_name)
                print("變化幀資料已自動儲存")
            except Exception as e:
                print(f"儲存變化幀資料時出錯: {e}")
        
        if self.total_frames > 0:
            if hasattr(self, 'progress_var'): 
                self.progress_var.set(self.total_frames)
            if hasattr(self, 'lbl_prog'): 
                self.lbl_prog.config(text=f"完成: {self.total_frames}/{self.total_frames}")
        
        self._update_status_bar("分析完成 - 請檢視結果後儲存")

    def _compare_ocr_results(self) -> Dict[str, List[int]]:
        """比較新舊OCR結果，返回差異統計"""
        differences = {
            'new_frames': [],      # 新增的幀
            'modified_frames': [], # 內容變更的幀  
            'removed_frames': []   # 移除的幀
        }
        
        # 找出新增的幀
        for frame_idx in self.annotations:
            if frame_idx not in self.old_annotations:
                differences['new_frames'].append(frame_idx)
        
        # 找出移除的幀
        for frame_idx in self.old_annotations:
            if frame_idx not in self.annotations:
                differences['removed_frames'].append(frame_idx)
        
        # 找出內容變更的幀
        for frame_idx in self.annotations:
            if frame_idx in self.old_annotations:
                old_text = self.old_annotations[frame_idx].strip()
                new_text = self.annotations[frame_idx].strip()
                if old_text != new_text:
                    differences['modified_frames'].append(frame_idx)
        
        return differences

    def _compare_change_results(self) -> Dict[str, List[int]]:
        """比較新舊變化偵測結果，返回差異統計"""
        differences = {
            'new_changes': [],     # 新偵測到變化的幀
            'lost_changes': [],    # 不再偵測到變化的幀
        }
        
        # 找出新偵測到變化的幀
        for frame_idx, has_change in self.change_cache.items():
            if has_change and not self.old_change_cache.get(frame_idx, False):
                differences['new_changes'].append(frame_idx)
        
        # 找出不再偵測到變化的幀
        for frame_idx, had_change in self.old_change_cache.items():
            if had_change and not self.change_cache.get(frame_idx, False):
                differences['lost_changes'].append(frame_idx)
        
        return differences

    def _print_comparison_report(self, ocr_diff: Dict[str, List[int]], change_diff: Dict[str, List[int]]):
        """印出詳細的比較報告"""
        
        print(f"區域: {self.region_name}")
        print(f"OCR模型: {self.ocr_model_var.get()}")
        print(f"總幀數: {self.total_frames}")
        print()
        
        # OCR結果比較
        print("【OCR結果比較】")
        if ocr_diff['new_frames']:
            print(f"  新增OCR結果 ({len(ocr_diff['new_frames'])} 個):")
            for frame_idx in sorted(ocr_diff['new_frames'])[:10]:  # 只顯示前10個
                text = self.annotations.get(frame_idx, "")
                print(f"    幀 {frame_idx}: '{text}'")
            if len(ocr_diff['new_frames']) > 10:
                print(f"    ... 還有 {len(ocr_diff['new_frames']) - 10} 個")
        
        if ocr_diff['removed_frames']:
            print(f"  移除OCR結果 ({len(ocr_diff['removed_frames'])} 個):")
            for frame_idx in sorted(ocr_diff['removed_frames'])[:10]:
                old_text = self.old_annotations.get(frame_idx, "")
                print(f"    幀 {frame_idx}: '{old_text}' (已移除)")
            if len(ocr_diff['removed_frames']) > 10:
                print(f"    ... 還有 {len(ocr_diff['removed_frames']) - 10} 個")
        
        if ocr_diff['modified_frames']:
            print(f"  內容變更 ({len(ocr_diff['modified_frames'])} 個):")
            for frame_idx in sorted(ocr_diff['modified_frames'])[:10]:
                old_text = self.old_annotations.get(frame_idx, "")
                new_text = self.annotations.get(frame_idx, "")
                print(f"    幀 {frame_idx}: '{old_text}' -> '{new_text}'")
            if len(ocr_diff['modified_frames']) > 10:
                print(f"    ... 還有 {len(ocr_diff['modified_frames']) - 10} 個")
        
        if not any(ocr_diff.values()):
            print("  無OCR結果差異")
        
        print()
        
        # 變化偵測比較
        print("【變化偵測比較】")
        if change_diff['new_changes']:
            print(f"  新偵測到變化 ({len(change_diff['new_changes'])} 個): {sorted(change_diff['new_changes'])[:20]}")
            if len(change_diff['new_changes']) > 20:
                print(f"    ... 還有 {len(change_diff['new_changes']) - 20} 個")
        
        if change_diff['lost_changes']:
            print(f"  不再偵測到變化 ({len(change_diff['lost_changes'])} 個): {sorted(change_diff['lost_changes'])[:20]}")
            if len(change_diff['lost_changes']) > 20:
                print(f"    ... 還有 {len(change_diff['lost_changes']) - 20} 個")
        
        if not any(change_diff.values()):
            print("  無變化偵測差異")
        
        print()
        
        # 統計摘要
        total_ocr_changes = len(ocr_diff['new_frames']) + len(ocr_diff['modified_frames']) + len(ocr_diff['removed_frames'])
        total_change_changes = len(change_diff['new_changes']) + len(change_diff['lost_changes'])
        
        print("【差異摘要】")
        print(f"  OCR結果差異總數: {total_ocr_changes}")
        print(f"  變化偵測差異總數: {total_change_changes}")
        print(f"  總差異數: {total_ocr_changes + total_change_changes}")

    def _stop_analysis(self):
        """停止分析 - 不自動儲存當前分析結果"""
        self.stop_event.set()
        
        # 更新按鈕狀態
        if hasattr(self, 'btn_analyze'):
            self.btn_analyze.config(state=tk.NORMAL)
        if hasattr(self, 'btn_stop'):
            self.btn_stop.config(state=tk.DISABLED)
        
        # 只儲存變化幀資料，不儲存 annotations（包含 current_analysis_cache）
        if self.video_file_path and self.region_name and self.change_cache:
            try:
                self._save_change_frames(self.region_name)
                print("停止分析：變化幀資料已儲存")
            except Exception as e:
                print(f"停止分析時儲存變化幀資料出錯: {e}")
        
        self._update_status_bar("分析已停止 - 當前分析結果未儲存")
        print("分析已停止 - 當前分析結果未儲存")

    def _on_goto_frame(self, event=None):
        try:
            idx = int(self.goto_var.get())
        except (ValueError, TypeError):
            return
        self._show_frame(idx)

    def _easyocr_predict(self, pil_img: Image.Image) -> str:
        result = self.ocr_iface.predict(pil_img)
        return result if result else "〈未識別〉"

    def _update_status_bar(self, message: str):
        """更新狀態列訊息"""
        if hasattr(self, 'status_var'):
            self.status_var.set(message)
        print(f"狀態: {message}")

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
            
            # 載入全域 ROI 設定
            self._load_roi_config()
            
            # 載入現有標註（如果有）以及變化幀列表
            self._load_existing_data()
            
            # 檢查是否有分析進度，並自動跳轉
            self._check_and_jump_to_analysis_position()
            
            # 顯示第一幀
            self._show_frame(0)
            
            return True
        except Exception as e:
            messagebox.showerror("載入失敗", f"影片載入出錯:\n{e}")
            print(f"載入影片時發生錯誤: {e}")
            traceback.print_exc()
            return False

    # 2. 新增函數用於清理上一個影片的資料和UI
    def _clear_previous_video_data(self):
        """Clears data and UI elements related to the previously loaded video."""
        self.lbl_video_path.config(text="未選擇影片")
        self.status_var.set("就緒")

        # Signal existing threads to stop
        self.stop_event.set()
        for th_name in ["analysis_thread", "ocr_thread"]: # Add any other worker thread names here
            thread = getattr(self, th_name, None)
            if thread and thread.is_alive():
                print(f"等待 {th_name} 結束...")
                thread.join(timeout=1.5) # Wait a bit longer
                if thread.is_alive():
                    print(f"警告: {th_name} 未能在1.5秒內結束。")
        self.stop_event.clear() # Reset event for new threads

        # Release ONLY UI VideoCapture object here
        # Worker threads will release their own VideoCaptures
        if self.cap_ui:
            try:
                self.cap_ui.release()
                print(f"cap_ui 已釋放。")
            except Exception as e:
                print(f"釋放舊 VideoCapture (cap_ui) 時出錯: {e}")
        self.cap_ui = None
        # self.cap_detect and self.cap_ocr are no longer class-level for workers

        # Clear model interfaces if they have explicit stop/release methods
        # (Assuming EasyOCRModel and ChangeDetectorInterface don't need explicit stopping here
        # as they are re-instantiated or managed differently)

        # Clear caches and data structures
        self.change_cache.clear()
        self.annotations.clear() 
        self.ocr_cache.clear()
        self.roi_image_cache.clear()

        # Clear TreeView
        if hasattr(self, 'tree'):
            self.tree.delete(*self.tree.get_children())

        # Clear video display label
        if hasattr(self, 'lbl_video'): 
            self.lbl_video.config(image=None) # Clear image
            self.current_display_image = None 

        # Reset current frame index and other video-specific states
        self.current_frame_idx = 0
        self.total_frames = 0
        self.original_vid_w = 0
        self.original_vid_h = 0
        # self.video_file_path = None # Keep this until a new video is truly loaded
        # self.video_title = "" 

        # Reset slider
        if hasattr(self, 'slider'):
            self.slider.config(to=0, state=tk.DISABLED) 
            if hasattr(self, 'slider_var'): self.slider_var.set(0)
        if hasattr(self, 'lbl_frame_num'):
            self.lbl_frame_num.config(text="幀: - / -")

        # Reset ROI related things - be careful if ROI settings are meant to be global or per-video
        # self.roi_coords = None 
        # self.roi_dict.clear() # This would clear all defined regions.
        # if hasattr(self,'region_combobox'): self.region_combobox["values"] = [] 
        # if hasattr(self,'region_var'): self.region_var.set("")

        # Disable analysis/save buttons
        if hasattr(self, 'btn_analyze'): # Assuming this is the correct button name
             self.btn_analyze.config(state=tk.DISABLED)
        # if hasattr(self, 'btn_save_all'): # Check actual save button name
        #     self.btn_save_all.config(state=tk.DISABLED)
        if hasattr(self, 'btn_stop'):
            self.btn_stop.config(state=tk.DISABLED)
        
        # Clear queues
        for q_name in ["detect_queue", "ocr_queue", "result_queue"]:
            q_obj = getattr(self, q_name, None)
            if q_obj:
                while not q_obj.empty():
                    try:
                        q_obj.get_nowait()
                    except queue.Empty:
                        break
        print("先前影片資料已大部分清除 (佇列已清空)。")

    # 3. 新增函數用於檢查分析進度並自動跳轉
    def _check_and_jump_to_analysis_position(self):
        """檢查分析進度並自動跳轉到適當位置"""
        if not self.annotations and not self.change_cache:
            # 沒有任何分析記錄，顯示第一幀
            self._show_frame(0)
            self._update_status_bar("已載入影片，可開始分析")
            return
        
        # 有分析記錄，跳轉到最後分析的位置
        if self.annotations:
            last_annotated_frame = max(self.annotations.keys())
            self._show_frame(last_annotated_frame)
            self._update_status_bar(f"已跳轉到最後標註幀: {last_annotated_frame}")
        elif self.change_cache:
            last_analyzed_frame = max(self.change_cache.keys())
            self._show_frame(last_analyzed_frame)
            self._update_status_bar(f"已跳轉到最後分析幀: {last_analyzed_frame}")
        else:
            self._show_frame(0)
            self._update_status_bar("已載入影片")

    # 4. 如果需要，添加一個輔助方法用於更新滑塊位置
    def _update_slider_position(self, frame_idx):
        """更新滑塊位置到指定幀"""
        if hasattr(self, 'slider_var'):
            self.slider_var.set(frame_idx)

    # 新增處理 TreeView 選擇事件的方法
    def _on_treeview_select(self, event=None):
        """當用戶點擊 TreeView 中的項目時，跳轉到對應的幀"""
        selected_items = self.tree.selection()
        if not selected_items:
            return
        
        selected_id = selected_items[0]
        
        try:
            # 獲取幀號
            frame_idx = int(self.tree.set(selected_id, "frame"))
            
            # 如果當前已經在該幀，則不需要重新載入
            if frame_idx == self.current_frame_idx:
                return
            
            # 跳轉到該幀
            self._show_frame(frame_idx)
            
            # 更新狀態欄
            content = self.tree.set(selected_id, "content")
            self._update_status_bar(f"已跳轉到幀 {frame_idx}: {content}")
            print(f"TreeView點擊跳轉到幀 {frame_idx}")
            
        except (ValueError, KeyError, TclError) as e:
            print(f"跳轉到所選幀時出錯: {e}")


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
            edit_dialog = tk.Toplevel(self)
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

    def _load_existing_data(self):
        """載入現有的標註和變化幀資料"""
        if not self.video_file_path:
            print("錯誤：影片路徑未設定，無法載入現有數據。")
            return
            
        self.video_title = self.video_file_path.stem
        print(f"正在載入區域 '{self.region_name}' 的現有數據: {self.video_title}")
        
        try:
            # 載入當前 region 的標註
            self._load_annotations(self.region_name)
            
            # 載入變化幀資料
            self._load_change_frames(self.region_name)
            
            # 刷新 TreeView 顯示
            self._refresh_treeview()
            
            # 如果有資料，預設選中第一項
            if self.tree.get_children():
                first_item = self.tree.get_children()[0]
                self.tree.selection_set(first_item)
                self.tree.focus(first_item)
                # 但不自動跳轉，讓用戶可以用鍵盤導航
                print(f"已選中第一個項目")
            
            print(f"已載入 {len(self.annotations)} 個標註記錄")
            print(f"已載入 {len([f for f, c in self.change_cache.items() if c])} 個變化幀記錄")
            
        except Exception as e:
            print(f"載入現有數據時出錯: {e}")
            traceback.print_exc()

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

    def _on_region_select(self, event=None):
        """切換 ROI 區域 - 不自動儲存標註內容"""
        new_region = self.region_var.get()
        if new_region == self.region_name:
            return
        
        # 停止當前分析
        if self.btn_stop.cget('state') == tk.NORMAL:
            self._stop_analysis()
        
        # 檢查是否有未儲存的當前分析結果
        if self.current_analysis_cache:
            result = messagebox.askyesnocancel(
                "切換區域", 
                f"目前區域 '{self.region_name}' 有 {len(self.current_analysis_cache)} 個未儲存的當前分析結果。\n\n" +
                "是否要先儲存這些結果？\n\n" +
                "選擇 [是]：儲存後切換\n" +
                "選擇 [否]：放棄當前分析結果並切換\n" +
                "選擇 [取消]：不切換區域"
            )
            
            if result is None:  # 取消
                # 恢復原來的選擇
                self.region_var.set(self.region_name)
                return
            elif result:  # 是 - 儲存後切換
                try:
                    self._save_annotations(self.region_name)
                    self._save_change_frames(self.region_name)
                except Exception as e:
                    messagebox.showerror("儲存失敗", f"儲存失敗：{e}")
                    self.region_var.set(self.region_name)
                    return
            # else: 否 - 直接切換，不儲存
        else:
            # 沒有當前分析結果，只儲存已確認的標註（如果有的話）
            if self.annotations:
                try:
                    self._save_confirmed_annotations_only(self.region_name)
                    self._save_change_frames(self.region_name)
                    print(f"已自動儲存區域 '{self.region_name}' 的已確認標註")
                except Exception as e:
                    print(f"自動儲存區域 '{self.region_name}' 標註時出錯: {e}")
        
        # 切換到新區域
        old_region = self.region_name
        self.region_name = new_region
        self.roi_coords = tuple(self.roi_dict[new_region])
        
        # 清空快取 (重要：避免新舊 region 資料混合)
        self.change_cache.clear()
        self.ocr_cache.clear()
        self.annotations.clear()
        self.roi_image_cache.clear()
        self.current_analysis_cache.clear()  # 也清空當前分析快取
        
        # 載入新區域的資料
        self._load_existing_data()
        
        # 更新 ROI 顯示
        self._update_roi_fields()
        
        # 重新顯示當前幀（使用新的 ROI）
        self._show_frame(self.current_frame_idx)
        
        print(f"已切換區域: {old_region} -> {new_region}")
        self._update_status_bar(f"已切換到區域: {new_region}")

    def _save_roi_config(self):
        """儲存 ROI 設定到檔案"""
        roi_file = get_roi_config_path()
        if not roi_file:
            return
        
        # 確保目錄存在
        roi_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 直接儲存 roi_dict，不包裝在 "rois" 鍵中
            with open(roi_file, "w", encoding="utf-8") as f:
                json.dump(self.roi_dict, f, indent=2, ensure_ascii=False)
            print(f"ROI 設定已儲存至 {roi_file}")
        except Exception as e:
            print(f"儲存 ROI 設定失敗: {e}")

    def _load_roi_config(self):
        """載入全域 ROI 設定"""
        roi_file = get_roi_config_path()
        
        try:
            if roi_file.exists():
                with open(roi_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # 支援兩種格式
                    if "rois" in data:
                        # 新格式：{"rois": {...}}
                        loaded_rois = data["rois"]
                    else:
                        # 舊格式：直接是 ROI 字典
                        loaded_rois = data
                    
                    # 直接使用載入的ROI，不再與預設合併
                    self.roi_dict = loaded_rois.copy()
                    print(f"已載入全域 ROI 設定: {loaded_rois}")
            else:
                print(f"全域 ROI 設定檔不存在，將建立預設配置")
                # 如果檔案不存在，建立一個預設配置
                self.roi_dict = {
                    "region2": [1640, 445, 1836, 525]
                }
        except Exception as e:
            print(f"載入全域 ROI 設定失敗: {e}")
            # 載入失敗時使用預設配置
            self.roi_dict = {
                "region2": [1640, 445, 1836, 525]
            }
        
        # 更新 UI（如果已建立）
        if hasattr(self, 'region_combobox'):
            self._update_roi_ui()
        
        print(f"最終 ROI 字典: {self.roi_dict}")

    def _on_close(self):
        """處理窗口關閉：停止線程並儲存標註進度。"""
        print("關閉應用程式...")
        self.stop_event.set() # 通知線程停止

        # 停止分析（如果正在進行）
        if hasattr(self, 'btn_stop') and self.btn_stop.cget('state') == tk.NORMAL:
            try:
                self.btn_analyze.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.DISABLED)
            except:
                pass

        # 儲存當前標註進度
        if self.video_file_path and self.annotations:
            print("正在自動儲存標註進度...")
            try:
                self._save_annotations(self.region_name)  # 修正：加上 region_name 參數
                self._save_change_frames(self.region_name)
            except Exception as e:
                print(f"儲存標註時出錯: {e}")
        else:
            print("無需儲存標註 (未載入影片或無標註內容)。")

        # 儲存 ROI 設定
        try:
            self._save_roi_config()
        except Exception as e:
            print(f"儲存 ROI 設定時出錯: {e}")

        # 等待線程結束
        print("等待背景線程結束...")
        self.stop_event.set()
        
        # 等待各種線程結束
        for thread_name in ['analysis_thread', 'ocr_thread', 'detect_thread']:
            thread = getattr(self, thread_name, None)
            if thread and thread.is_alive():
                try:
                    thread.join(timeout=2.0)  # 最多等待2秒
                except:
                    pass

        self._finalize_close()

    def _finalize_close(self):
        """最終清理並關閉應用程式"""
        # 釋放 VideoCapture
        for cap in (self.cap_ui, self.cap_detect, self.cap_ocr):
            try:
                if cap:
                    cap.release()
            except Exception:
                pass
        
        # 銷毀 Tk 主視窗
        try:
            self.master.destroy()
        except TclError:
            pass
        print("應用程式已關閉。")

    def _get_annotations_path(self, region_name: str) -> Path:
        """取得指定 region 的標註檔案路徑"""
        if not self.video_file_path:
            return Path()
        
        # 使用統一的 video_title 變數
        path = Path("data") / self.video_title / f"{region_name}.jsonl"
        print(f"DEBUG: _get_annotations_path 返回: {path}")
        return path

    def _get_roi_dir(self, region_name: str) -> Path:
        """取得指定 region 的 ROI 圖片目錄路徑"""
        if not self.video_file_path:
            return Path()
        
        # ROI 圖片目錄：data/[影片名]/[region]/
        roi_dir = Path("data") / self.video_title / f"{region_name}"
        roi_dir.mkdir(parents=True, exist_ok=True)
        return roi_dir

    def _apply_roi_from_fields(self):
        """把 Spinbox 數值寫回 ROI，並立即生效/儲存"""
        x1 = self.roi_x1_var.get()
        y1 = self.roi_y1_var.get()
        x2 = self.roi_x2_var.get()
        y2 = self.roi_y2_var.get()
        
        if x1 >= x2 or y1 >= y2:
            messagebox.showwarning("座標錯誤", "x1,y1 必須小於 x2,y2")
            return
            
        self.roi_coords = (x1, y1, x2, y2)
        self.roi_dict[self.region_name] = list(self.roi_coords)
        self._save_roi_config()
        self._show_frame(self.current_frame_idx)
        self._update_status_bar(f"ROI 已套用: {self.roi_coords}")

    def _update_roi_fields(self):
        """將 self.roi_coords 反映到 4 個 Spinbox"""
        if not self.roi_coords:
            return
        x1, y1, x2, y2 = map(int, self.roi_coords)
        self.roi_x1_var.set(x1)
        self.roi_y1_var.set(y1)
        self.roi_x2_var.set(x2)
        self.roi_y2_var.set(y2)

    def _on_add_region(self):
        """新增一個 region 名稱，預設沿用目前 ROI 座標"""
        name = simpledialog.askstring("新增區域", "輸入區域名稱:", parent=self)
        if not name:
            return
        if name in self.roi_dict:
            messagebox.showinfo("提示", "區域名稱已存在！")
            return
        self.roi_dict[name] = list(self.roi_coords) if self.roi_coords else [0, 0, 100, 100]
        self.region_combobox["values"] = list(self.roi_dict.keys())
        self.region_var.set(name)
        self.region_name = name
        self._save_roi_config()
        self._update_status_bar(f"已新增區域 {name}")
        # 新增完成後，把 ROI 座標同步到 Spinbox 方便微調
        self._update_roi_fields()

    def _update_roi_ui(self):
        """更新 ROI 相關的 UI 元素"""
        # 更新 Combobox
        self.region_combobox["values"] = list(self.roi_dict.keys())
        
        # 設定當前 region
        if self.region_name in self.roi_dict:
            self.region_var.set(self.region_name)
            self.roi_coords = tuple(self.roi_dict[self.region_name])
        else:
            # 如果預設 region 不存在，取第一個可用的
            if self.roi_dict:
                self.region_name = list(self.roi_dict.keys())[0]
                self.region_var.set(self.region_name)
                self.roi_coords = tuple(self.roi_dict[self.region_name])
        
        self._update_roi_fields()
        print(f"ROI UI 已更新。目前區域: {self.region_name}, ROI: {self.roi_coords}")

    def _tag_tree_item(self, frame_idx: int, tag: str):
        for iid in self.tree.get_children():
            if int(self.tree.set(iid, "frame")) == frame_idx:
                current_tags = set(self.tree.item(iid, "tags"))
                if tag not in current_tags:
                    current_tags.add(tag)
                    self.tree.item(iid, tags=tuple(current_tags))
                break

    def _get_roi_image(self, frame_idx: int, video_capture: cv2.VideoCapture) -> Optional[Image.Image]:
        """取得指定幀的 ROI 圖像，使用傳入的 VideoCapture 實例"""
        try:
            # 先檢查快取
            if frame_idx in self.roi_image_cache:
                return self.roi_image_cache[frame_idx]
            
            # 嘗試從檔案載入ROI (如果存在且需要此優化)
            roi_image_from_file = self._load_roi_from_file(frame_idx)
            if roi_image_from_file:
                self.roi_image_cache[frame_idx] = roi_image_from_file
                return roi_image_from_file
            
            # 如果快取和檔案中都沒有，則從影片中提取
            if not video_capture or not video_capture.isOpened() or not self.roi_coords:
                print(f"警告: _get_roi_image 無法獲取圖像，VideoCapture 無效或 ROI 未設定 (frame {frame_idx})")
                return None
            
            # 讀取幀
            # 為確保執行緒安全，對 video_capture 的操作應謹慎
            # 如果 video_capture 可能被多個地方同時 set/read，需要額外同步機制
            # 但在此設計中，每個 worker 應有獨立的 vc，所以直接操作是OK的
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video_capture.read()
            if not ret:
                print(f"警告: _get_roi_image 無法讀取幀 {frame_idx} from video_capture")
                return None
            
            # 轉換為 PIL 圖像
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 提取 ROI
            roi_pil = self._crop_roi(frame_pil)
            if roi_pil:
                # 儲存到快取
                self.roi_image_cache[frame_idx] = roi_pil 
                # 考慮是否在此處保存 ROI 圖像到文件，或由特定流程統一處理
                self._save_roi_image(frame_idx, roi_pil) 
                
            return roi_pil
            
        except Exception as e:
            print(f"取得 ROI 圖像時出錯 (frame {frame_idx}): {e}")
            traceback.print_exc()
            return None

    def _perform_ocr(self, frame_idx: int, video_capture_for_roi: cv2.VideoCapture) -> str:
        """對指定幀進行 OCR，傳入 VideoCapture 以獲取 ROI"""
        try:
            # 確保 video_capture_for_roi 是打開的
            if not video_capture_for_roi or not video_capture_for_roi.isOpened():
                print(f"錯誤: _perform_ocr 的 video_capture_for_roi 無效 (frame {frame_idx})")
                return ""
                
            # 取得 ROI 圖像
            roi_image = self._get_roi_image(frame_idx, video_capture_for_roi)
            if roi_image is None:
                print(f"OCR 時無法取得幀 {frame_idx} 的 ROI")
                return ""
                
            # 使用 OCR 介面進行辨識
            # 假設 self.ocr_iface.recognize 返回一個元組 (text, confidence)
            # 或至少返回一個可以直接使用的 text 字串
            result = self.ocr_iface.recognize(roi_image)
            
            ocr_text = ""
            if isinstance(result, tuple) and len(result) > 0:
                # 假設第一個元素是文字
                ocr_text = str(result[0]) 
            elif isinstance(result, str):
                ocr_text = result
            else:
                print(f"OCR 結果格式無法識別 (frame {frame_idx}): {type(result)}")

            return ocr_text if ocr_text else ""
                
        except AttributeError as ae:
            print(f"OCR 屬性錯誤 (frame {frame_idx}): {ae} - 請檢查 OCR 模型接口是否正確實現 'recognize' 方法。")
            traceback.print_exc()
            return ""
        except Exception as e:
            print(f"OCR 處理幀 {frame_idx} 時出錯: {e}")
            traceback.print_exc()
            return ""

    def _frame_to_timestamp(self, frame_idx: int) -> str:
        """將幀號轉換為時間戳記"""
        try:
            if not self.cap_ui:
                return "00:00:00"
            
            fps = self.cap_ui.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # 預設 FPS
            
            seconds = frame_idx / fps
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            
        except Exception as e:
            print(f"轉換時間戳記時出錯: {e}")
            return "00:00:00"

    def _calculate_frame_diff(self, frame_idx: int) -> str:
        """計算指定 frame 與前一幀的 diff 值"""
        if frame_idx == 0:
            return "0.000"  # 第一幀顯示 0
        
        try:
            roi_img1 = self._get_roi_image(frame_idx - 1, self.cap_ui)
            roi_img2 = self._get_roi_image(frame_idx, self.cap_ui)
            
            if roi_img1 and roi_img2:
                bin_method = self.binarize_method_var.get()
                bin1 = self._apply_binarization(roi_img1, bin_method)
                bin2 = self._apply_binarization(roi_img2, bin_method)
                
                if bin1 and bin2:
                    arr1 = np.array(bin1.convert("L"))
                    arr2 = np.array(bin2.convert("L"))
                    diff = self._calculate_binary_diff(arr1, arr2)
                    return f"{diff:.4f}"
            
            return "-"  # 無法計算時顯示 "-"
        except Exception as e:
            print(f"計算 frame {frame_idx} diff 時出錯: {e}")
            return "-"

    def _update_treeview_item(self, frame_idx: int, has_change: Optional[bool] = None, 
                            content: Optional[str] = None, current_analysis: Optional[str] = None):
        """更新 TreeView 中的特定項目。如果項目不存在則創建它。"""
        item_id_str = f"F{frame_idx}"

        if not self.tree.exists(item_id_str):
            # 項目不存在，創建它
            existing_content = self.annotations.get(frame_idx, "")
            existing_current = self.current_analysis_cache.get(frame_idx, "")
            
            # 計算 diff 值
            diff_value = self._calculate_frame_diff(frame_idx)
            
            # 正確的 values 順序：(frame, diff, content, current_analysis)
            self.tree.insert("", "end", iid=item_id_str, 
                            values=(frame_idx, diff_value, existing_content, existing_current))
            
            # 如果 diff < 0.01，加上紅字 tag
            if diff_value != "-" and float(diff_value) < 0.01:
                self.tree.item(item_id_str, tags=("small_diff",))
        else:
            # 項目存在，更新 diff 值
            diff_value = self._calculate_frame_diff(frame_idx)
            self.tree.set(item_id_str, "diff", diff_value)
            
            # 更新 diff 相關的 tag
            current_tags = list(self.tree.item(item_id_str, "tags"))
            if diff_value != "-" and float(diff_value) < 0.01:
                if "small_diff" not in current_tags:
                    current_tags.append("small_diff")
            else:
                if "small_diff" in current_tags:
                    current_tags.remove("small_diff")
            self.tree.item(item_id_str, tags=tuple(current_tags))

        # 更新其他內容
        if content is not None:
            self.tree.set(item_id_str, "content", content)  # 標註內容寫到 content 欄位
        
        if current_analysis is not None:
            self.tree.set(item_id_str, "current_analysis", current_analysis)
        
        # 確保"changed"標籤正確應用
        if self.change_cache.get(frame_idx, False):
            current_tags = list(self.tree.item(item_id_str, "tags"))
            if "changed" not in current_tags:
                current_tags.append("changed")
                self.tree.item(item_id_str, tags=tuple(current_tags))

    def _save_annotations(self, region_name: str):
        """儲存標註結果 - 手動儲存時處理當前分析結果"""
        # 如果有當前分析結果，詢問是否要覆寫
        if self.current_analysis_cache:
            result = messagebox.askyesno(
                "確認儲存", 
                f"檢測到 {len(self.current_analysis_cache)} 個當前分析結果。\n\n" +
                "是否要將'當前分析'的結果覆寫到'標註內容'並儲存到檔案？\n\n" +
                "選擇 [是]：將當前分析結果合併到標註中並儲存\n" +
                "選擇 [否]：只儲存現有的標註內容，忽略當前分析結果"
            )
            if result:
                # 將當前分析結果覆寫到正式標註
                for frame_idx, content in self.current_analysis_cache.items():
                    self.annotations[frame_idx] = content
                    # 同時更新TreeView顯示
                    item_id_str = f"F{frame_idx}"  # 修正：使用正確的 item_id 格式
                    if self.tree.exists(item_id_str):
                        self.tree.set(item_id_str, "content", content)
                        self.tree.set(item_id_str, "current_analysis", "")  # 清空當前分析欄位
                
                # 清空暫存快取
                self.current_analysis_cache.clear()
                print(f"✅ 已將 {len(self.current_analysis_cache)} 個當前分析結果合併到標註中")
            else:
                print("❌ 用戶選擇不合併當前分析結果，只儲存現有標註")
        
        # 儲存正式的標註內容
        if not self.annotations:
            print("無標註內容需要儲存。")
            messagebox.showinfo("提示", "沒有標註內容需要儲存")
            return
        
        try:
            if not self.video_file_path:
                messagebox.showerror("錯誤", "無法儲存標註，影片路徑未設定。")
                return

            video_data_dir = Path("data") / self.video_title
            video_data_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = video_data_dir / f"{region_name}.jsonl"
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for frame_idx in sorted(self.annotations.keys()):
                    ocr_text = self.annotations[frame_idx]
                    record = {
                        "frame": frame_idx,
                        "ocr_text": ocr_text
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"✅ 標註已儲存至: {jsonl_path}")
            messagebox.showinfo("儲存成功", f"標註已儲存至:\n{jsonl_path.name}")
            self._update_status_bar(f"標註已儲存: {jsonl_path.name}")
            
        except Exception as e:
            messagebox.showerror("儲存標註失敗", f"儲存標註 (region: {region_name}) 時出錯: {e}")
            print(f"❌ 儲存標註 (region: {region_name}) 時出錯: {e}")
            traceback.print_exc()

    def _save_change_frames(self, region_name: str):
        """儲存變化幀列表為 JSON 格式 - 統一版本"""
        if not self.change_cache:
            print(f"區域 {region_name}: 無變化幀數據需要儲存。")
            # self._update_status_bar(f"區域 {region_name}: 無變化幀可儲存") # 可選的狀態更新
            return
        
        try:
            if not self.video_file_path:
                messagebox.showerror("錯誤", f"無法確定區域 {region_name} 的變化幀儲存路徑。影片是否已載入？")
                print(f"錯誤: 無法獲取有效的 video_file_path for region {region_name}.")
                # self._update_status_bar(f"區域 {region_name}: 變化幀儲存路徑無效") # 可選
                return

            video_data_dir = Path("data") / self.video_title
            video_data_dir.mkdir(parents=True, exist_ok=True) 
            
            # 變化幀檔案路徑 (JSONL，每行為一個物件)
            change_path = video_data_dir / f"{region_name}_ocr.jsonl"
            
            # 只儲存 has_change 為 True 的幀 (即偵測到變化)
            changed_frame_indices = sorted([
                frame_idx for frame_idx, has_change in self.change_cache.items() if has_change
            ])
            
            if not changed_frame_indices:
                print(f"區域 {region_name}: 計算後沒有偵測到任何變化幀可儲存。")
                return

            # 依照新格式寫入：每行 {frame, ocr_text, confidence}
            with self.save_lock:
                with open(change_path, 'w', encoding='utf-8') as f:
                    for frame_idx in changed_frame_indices:
                        record = {
                            "frame": frame_idx,
                            "ocr_text": self.annotations.get(frame_idx, ""),
                            "confidence": 1.0  # 若無信心度資訊，預設 1.0
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"區域 {region_name} 的變化幀列表已儲存至: {change_path} (共 {len(changed_frame_indices)} 個變化幀)")
            
        except Exception as e:
            messagebox.showerror("儲存變化幀失敗", f"儲存區域 {region_name} 變化幀時出錯: {e}")
            print(f"儲存區域 {region_name} 變化幀時出錯: {e}")
            traceback.print_exc()

    def _load_change_frames(self, region_name: str):
        """載入變化幀列表 - 支援 JSONL 格式（單行陣列）"""
        try:
            if not self.video_file_path:
                print(f"錯誤: _load_change_frames 無法獲取有效的 video_file_path for region {region_name}.")
                return

            video_data_dir = Path("data") / self.video_title
            change_frames = None

            # 嘗試 .jsonl 格式
            change_path = video_data_dir / f"{region_name}_ocr.jsonl"
            if change_path.exists():
                print(f"載入變化幀檔案: {change_path}")
                change_frames = []
                with open(change_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict) and "frame" in obj:
                                frame_idx = int(obj["frame"])
                                change_frames.append(frame_idx)
                                # 同步寫入 annotations，供後續顯示文字
                                ocr_text_val = obj.get("ocr_text", obj.get("text", ""))
                                if ocr_text_val is not None:
                                    self.annotations[frame_idx] = ocr_text_val
                        except json.JSONDecodeError as e:
                            print(f"第 {line_num} 行解析失敗: {e}")

                if change_frames:
                    self._rebuild_change_cache(change_frames)
                    print(f"已載入 {len(change_frames)} 個變化幀 (JSONL 格式) for region {region_name}")
                else:
                    print(f"檔案存在但未解析到任何變化幀: {change_path}")
            else:
                print(f"變化幀檔案不存在: {change_path}")
                self._rebuild_change_cache([])

        except Exception as e:
            print(f"載入區域 {region_name} 的變化幀時出錯: {e}")
            traceback.print_exc()
            self._rebuild_change_cache([])
    
    def _rebuild_change_cache(self, change_frames: list):
        """重建變化幀快取"""
        self.change_cache.clear()
        for i in range(self.total_frames):
            self.change_cache[i] = False
        for frame_idx in change_frames:
            if 0 <= frame_idx < self.total_frames:
                self.change_cache[frame_idx] = True
            
    def _on_analysis_complete(self):
        """分析自然完成後的處理"""
        print("主分析執行緒回報：分析自然完成。")
        if hasattr(self, 'btn_analyze'): self.btn_analyze.config(state=tk.NORMAL if self.video_file_path else tk.DISABLED)
        if hasattr(self, 'btn_stop'): self.btn_stop.config(state=tk.DISABLED)
        
        if self.video_file_path and self.region_name: 
            self._save_annotations(self.region_name)
            self._save_change_frames(self.region_name)
        
        if self.total_frames > 0:
            if hasattr(self, 'progress_var'): self.progress_var.set(self.total_frames)
            if hasattr(self, 'lbl_prog'): self.lbl_prog.config(text=f"完成: {self.total_frames}/{self.total_frames}")
        
        self._update_status_bar("分析流程已圓滿完成。")

    def _on_closing(self):
        """應用程式關閉時的處理 - 不自動儲存當前分析結果"""
        print("關閉應用程式...")
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            print("正在停止分析執行緒...")
            self.stop_event.set()
            self.analysis_thread.join(timeout=2.5)
            if self.analysis_thread.is_alive():
                print("警告: 分析執行緒未能優雅停止。")
        
        # 只儲存已確認的標註內容和變化幀資料，不包含當前分析快取
        if self.video_file_path and self.region_name:
            try:
                # 只儲存正式的 annotations（不包含 current_analysis_cache）
                if self.annotations:
                    self._save_confirmed_annotations_only(self.region_name)
                    print("已確認的標註資料已儲存")
                
                # 儲存變化幀資料
                if self.change_cache:
                    self._save_change_frames(self.region_name)
                    print("變化幀資料已儲存")
                    
            except Exception as e:
                print(f"關閉時儲存資料出錯: {e}")
        else:
            print("無需儲存資料 (未載入影片或無已確認的標註內容)")
        
        # 提醒用戶未儲存的當前分析結果
        if self.current_analysis_cache:
            print(f"⚠️ 注意：有 {len(self.current_analysis_cache)} 個當前分析結果未儲存")
        
        print("應用程式已關閉。")
        self.master.destroy()

    def _save_confirmed_annotations_only(self, region_name: str):
        """只儲存已確認的標註內容，不處理當前分析快取"""
        if not self.annotations:
            print("無已確認的標註內容需要儲存。")
            return
        
        try:
            if not self.video_file_path:
                print("錯誤：無法儲存標註，影片路徑未設定。")
                return

            video_data_dir = Path("data") / self.video_title
            video_data_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = video_data_dir / f"{region_name}.jsonl"
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for frame_idx in sorted(self.annotations.keys()):
                    ocr_text = self.annotations[frame_idx]
                    record = {
                        "frame": frame_idx,
                        "ocr_text": ocr_text
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"已確認的標註已儲存至: {jsonl_path}")
            
        except Exception as e:
            print(f"儲存已確認標註 (region: {region_name}) 時出錯: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotator(root)
    root.mainloop()
    