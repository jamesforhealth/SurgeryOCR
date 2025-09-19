#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import queue
from typing import Optional, Any
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
import numpy as np
from dataclasses import dataclass # 新增

import cv2
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib
import tkinter.font as tkFont # For bold font in Treeview
from tkinter import TclError
import torch
import traceback
from models.OCR_interface import get_ocr_model
from typing import List, Tuple
import colorsys
from sklearn.cluster import KMeans
from PIL import ImageColor

# 新增：從 utils 導入 diff rule 載入器
from utils.get_configs import load_diff_rules, load_roi_config, load_roi_header_config


"""回傳 config/rois.json 路徑"""
get_roi_config_path = lambda : Path("config") / "rois.json"

"""回傳 config/surgery_stage_rois.json 路徑"""
get_surgery_stage_roi_config_path = lambda : Path("config") / "surgery_stage_rois.json"

# 新增：定義 RegionPattern 資料結構，與分析腳本一致
@dataclass
class RegionPattern:
    pattern_id: int
    array: np.ndarray


frame_width = 80
content_width = 120
end_frame_width = 80
iop_width = 80
asp_width = 120
vac_width = 120

# -------------------- 主GUI --------------------
class VideoAnnotator(tk.Frame):
    VID_W, VID_H = 800, 450
    # ROI = (1640, 445, 1836, 525) # Wird aus roi_dict geladen
    OCR_CONF_TH = 0.5 

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        
        self.master.title("Frame Annotation Tool")
        self.master.geometry("1350x750")
        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.pack(fill="both", expand=True)
        
        self.video_file_path: Optional[Path] = None
        self.video_title = ""
        self.cap_ui: Optional[cv2.VideoCapture] = None
        self.cap_worker: Optional[cv2.VideoCapture] = None
        
        self.total_frames = 0
        self.fps = 30
        self.current_frame_idx = 0
        self.playback_active = False
        
        # 資料模型
        self.annotations = {}
        self.change_cache = {}  # Stores {frame_idx: bool}
        self.roi_coords = None
        
        # OCR數據緩存，用於性能優化
        self.ocr_cache = {}  # {region_name: {frame: ocr_text}}
        
        # 控制表格同步的標誌
        self._user_clicked_treeview = False  # 用戶是否手動點擊了表格
        self.sub_roi_coords = None
        self.region_name = ""
        self.roi_dict = {}

        # 手術階段ROI相關變量
        self.surgery_stage_roi_dict = {}  # 手術階段ROI配置
        self.surgery_stage_mode = False   # 是否在手術階段ROI模式
        self.current_surgery_stage_region = ""  # 當前選中的手術階段區域

        # 手術階段ROI座標變量
        self.surgery_stage_x1_var = tk.IntVar(value=0)
        self.surgery_stage_y1_var = tk.IntVar(value=0)
        self.surgery_stage_x2_var = tk.IntVar(value=0)
        self.surgery_stage_y2_var = tk.IntVar(value=0)

        # 手術階段ROI預覽相關變量
        self.stage_roi_preview_label = None
        self.stage_roi_preview_image = None
        self.stage_roi_preview_size = (150, 50)  # 預覽區域大小
        self.roi_diff_label = None
        self.sub_roi_rect_id = None # 用於在canvas上繪製sub ROI框
        self.diff_table = None # 用於顯示差異表格
        # 新增：用Canvas顯示RMSE矩陣
        self.diff_canvas = None
        
        # 階段分析與時間軸標籤（Stage Tags）
        self.stage_analysis = None  # 讀取 data/<video>/stage_analysis.json 後的資料
        
        # 同步控制變數
        self._sync_in_progress = False
        self._is_arrow_key_navigation = False
        self.stage_tag_panel = None
        self.stage_tag_canvas = None  # 將被替換為多軌道系統
        self.stage_tag_controls_frame = None
        self.stage_tag_meta = {}  # canvas_item_id -> {region, pattern, start, end, avg_rmse}
        self.stage_tag_roi_vars = {}  # region -> tk.BooleanVar
        self.stage_tag_visible = tk.BooleanVar(value=True)
        self.stage_tracks_mode = 'tags'  # 保留擴展
        
        # 多軌道時間軸系統
        self.timeline_tracks = {}  # region_name -> {'canvas': canvas, 'meta': {}, 'frame': frame}
        self.timeline_container = None
        self.timeline_main_canvas = None
        self.timeline_tracks_frame = None
        self.timeline_v_scrollbar = None
        self.track_height = 22  # 每個軌道的高度
        self.track_spacing = 4   # 軌道間距
        self.show_alignment_grid = False  # 是否顯示對齊網格（測試用）
        self.roi_color_map = {
            'PEDAL': '#007bff', # A modern, vibrant blue
            'STAGE': '#28a745', # A clear, distinct green
        }
        
        # 色彩生成方案 (HSV)
        self.color_variation_params = {
            'PEDAL': {'h_shift': 0.02, 's_factor': 0.9, 'v_factor': 0.85},
            'STAGE': {'h_shift': 0.03, 's_factor': 0.95, 'v_factor': 0.9},
        }
        
        # ROI圖像快取系統
        self.roi_image_cache = {}  # {region_name: cached_image_array}
        diff_rules = load_diff_rules()
        region_config = diff_rules.get(self.region_name, {})         
        self.cache_hit_threshold = region_config.get("diff_threshold", 60.0)  # RMSE閾值，低於此值認為是cache hit
        # 各階段區域上一幀 ROI 快取
        self.previous_stage_roi_images = {}

        self.ocr_iface = None
        self.ocr_test_window = None
        self.ocr_test_active = False

        # 手術階段ROI進階分析窗口
        self.surgery_stage_roi_test_window = None
        self.surgery_stage_roi_test_active = False

        # 狀態標誌
        self.status_var = tk.StringVar(value="就緒")

        self.hsv_s_threshold_var = tk.IntVar(value=30)
        self.gray_threshold_var = tk.IntVar(value=150)

        self.ocr_iface = get_ocr_model(
            model_type="easyocr",
            gpu=torch.cuda.is_available(),
            lang_list=['en'],
            confidence_threshold=self.OCR_CONF_TH,
            debug_output=True  # 啟用詳細調試輸出
        )


        self.roi_x1_var = tk.IntVar(value=0)
        self.roi_y1_var = tk.IntVar(value=0)
        self.roi_x2_var = tk.IntVar(value=0)
        self.roi_y2_var = tk.IntVar(value=0)
        
        self._create_widgets()

        # 預加載手術階段ROI配置
        self._load_surgery_stage_roi_config()

        # 預加載OCR ROI配置
        self._load_roi_config()

        master.bind("<Left>", self._on_left_key)
        master.bind("<Right>", self._on_right_key)
        master.bind("<Up>", self._on_up_key)
        master.bind("<Down>", self._on_down_key)
        master.bind("<space>", self._toggle_ocr_test_window)

    def _on_left_key(self, event=None):
        """處理左鍵事件 - 前一幀"""
        print("左鍵按下 - 前一幀")
        self._is_arrow_key_navigation = True
        self._step_frame(-1)
        return "break"  # 阻止事件繼續傳播

    def _on_right_key(self, event=None):
        """處理右鍵事件 - 後一幀"""
        print("右鍵按下 - 後一幀")
        self._is_arrow_key_navigation = True
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
        
        # 初始化變數（需要在UI創建前）
        self.binarize_mode_var = tk.BooleanVar(value=False)
        self.binarize_method_var = tk.StringVar(value="rule")
        
        # ========================= HEADER 區域 =========================
        header_frame = tk.Frame(self, relief="groove", bd=2)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        # Header 左半邊：設定和控制 (固定寬度50%)
        header_left = tk.Frame(header_frame)
        header_left.pack(side="left", fill="both", expand=True, padx=(5, 2), pady=5)
        
        # 影片路徑和基礎控制
        video_control_frame = tk.Frame(header_left)
        video_control_frame.pack(fill="x", pady=(0, 5))
        
        self.btn_load = tk.Button(video_control_frame, text="載入影片", command=self._load_video)
        self.btn_load.pack(side="left", padx=(0, 10))
        
        self.lbl_video_path = tk.Label(video_control_frame, text="未選擇影片")
        self.lbl_video_path.pack(side="left", padx=5)
        
        # OCR模式專用控制項（會根據模式動態顯示/隱藏）
        self.binarize_checkbox = tk.Checkbutton(video_control_frame, text="二值化顯示", variable=self.binarize_mode_var, command=self._on_binarize_toggle)
        self.binarize_checkbox.pack(side="right", padx=5)
        
        self.btn_save = tk.Button(video_control_frame, text="儲存標註", command=lambda: self._save_annotations(self.region_name))
        self.btn_save.pack(side="right", padx=(0, 10))
        
        # 模式選擇標籤頁
        self.mode_notebook = ttk.Notebook(header_left)
        self.mode_notebook.pack(fill="x", pady=(0, 3))
        
        # OCR 模式標籤頁
        self.ocr_mode_frame = ttk.Frame(self.mode_notebook)
        self.mode_notebook.add(self.ocr_mode_frame, text="OCR 標註模式")
        
        # OCR ROI 設定
        ocr_roi_frame = tk.LabelFrame(self.ocr_mode_frame, text="OCR ROI 設定", relief="flat", bd=1)
        ocr_roi_frame.pack(fill="x", pady=(5, 3), padx=5)
        
        ocr_row1 = tk.Frame(ocr_roi_frame)
        ocr_row1.pack(fill="x", padx=5, pady=2)
        
        tk.Label(ocr_row1, text="區域:").pack(side="left")
        self.region_var = tk.StringVar()
        self.region_combobox = ttk.Combobox(ocr_row1, textvariable=self.region_var, state="readonly", width=8)
        self.region_combobox.pack(side="left", padx=2)
        self.region_combobox.bind("<<ComboboxSelected>>", self._on_region_select)
        
        tk.Button(ocr_row1, text="新增", command=self._on_add_region).pack(side="left", padx=2)
        tk.Button(ocr_row1, text="儲存組態", command=self._save_roi_config).pack(side="left", padx=2)
        
        ocr_row2 = tk.Frame(ocr_roi_frame)
        ocr_row2.pack(fill="x", padx=5, pady=2)
        
        tk.Label(ocr_row2, text="座標:").pack(side="left")
        for text, var_tuple in [("x1", self.roi_x1_var), ("y1", self.roi_y1_var), ("x2", self.roi_x2_var), ("y2", self.roi_y2_var)]:
            tk.Label(ocr_row2, text=f"{text}:").pack(side="left")
            ttk.Spinbox(ocr_row2, from_=0, to=99999, width=5, textvariable=var_tuple).pack(side="left", padx=(0,3))
        tk.Button(ocr_row2, text="套用", command=self._apply_roi_from_fields).pack(side="left", padx=3)
        
        # 手術階段分析模式標籤頁
        self.surgery_mode_frame = ttk.Frame(self.mode_notebook)
        self.mode_notebook.add(self.surgery_mode_frame, text="手術階段分析模式")
        
        # 手術階段ROI 設定
        surgery_roi_frame = tk.LabelFrame(self.surgery_mode_frame, text="手術階段ROI 設定", relief="flat", bd=1)
        surgery_roi_frame.pack(fill="x", pady=(5, 3), padx=5)
        
        surgery_row1 = tk.Frame(surgery_roi_frame)
        surgery_row1.pack(fill="x", padx=5, pady=2)
        
        tk.Label(surgery_row1, text="區域:").pack(side="left")
        self.surgery_stage_region_var = tk.StringVar()
        self.surgery_stage_combobox = ttk.Combobox(surgery_row1, textvariable=self.surgery_stage_region_var, state="readonly", width=10)
        self.surgery_stage_combobox.pack(side="left", padx=2)
        self.surgery_stage_combobox.bind("<<ComboboxSelected>>", self._on_surgery_stage_region_select)
        
        tk.Button(surgery_row1, text="新增", command=self._on_add_surgery_stage_region).pack(side="left", padx=2)
        
        surgery_row2 = tk.Frame(surgery_roi_frame)
        surgery_row2.pack(fill="x", padx=5, pady=2)
        
        tk.Label(surgery_row2, text="座標:").pack(side="left")
        for text, var_tuple in [("x1", self.surgery_stage_x1_var), ("y1", self.surgery_stage_y1_var), ("x2", self.surgery_stage_x2_var), ("y2", self.surgery_stage_y2_var)]:
            tk.Label(surgery_row2, text=f"{text}:").pack(side="left")
            ttk.Spinbox(surgery_row2, from_=0, to=99999, width=5, textvariable=var_tuple).pack(side="left", padx=(0,3))
        tk.Button(surgery_row2, text="套用", command=self._apply_surgery_stage_roi_from_fields).pack(side="left", padx=3)
        tk.Button(surgery_row2, text="存入快取", command=self._save_roi_to_cache, bg="lightgreen").pack(side="left", padx=2)
        tk.Button(surgery_row2, text="查看快取", command=self._show_cache_info, bg="lightblue").pack(side="left", padx=2)
        
        # 綁定標籤頁切換事件
        self.mode_notebook.bind("<<NotebookTabChanged>>", self._on_mode_tab_changed)
        
        # 默認選擇OCR模式標籤頁
        self.mode_notebook.select(self.ocr_mode_frame)
        
        # 初始化模式相關控制項的顯示狀態
        self._update_mode_specific_controls()
        
        # Header 右半邊：ROI 預覽對比 (固定寬度50%)
        header_right = tk.Frame(header_frame, relief="sunken", bd=1)
        header_right.pack(side="right", fill="both", expand=True, padx=(2, 5), pady=5)
        
        roi_compare_frame = tk.LabelFrame(header_right, text="ROI 對比")
        roi_compare_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 預覽圖像區域
        preview_area = tk.Frame(roi_compare_frame)
        preview_area.pack(fill="both", expand=True, padx=5, pady=5)
        
        last_frame = tk.LabelFrame(preview_area, text="Last")
        last_frame.pack(side="left", fill="both", expand=True, padx=(0, 2))
        self.stage_roi_preview_label = tk.Label(last_frame, bg="black")
        self.stage_roi_preview_label.pack(fill="both", expand=True, padx=2, pady=2)
        
        current_frame = tk.LabelFrame(preview_area, text="Current")
        current_frame.pack(side="left", fill="both", expand=True, padx=(2, 0))
        self.current_roi_preview_label = tk.Label(current_frame, bg="black")
        self.current_roi_preview_label.pack(fill="both", expand=True, padx=2, pady=2)
        
        # 資訊顯示
        info_area = tk.Frame(roi_compare_frame)
        info_area.pack(fill="x", padx=5, pady=(0, 5))
        
        self.roi_diff_label = tk.Label(info_area, text="Diff: -", font=("Courier", 9))
        self.roi_diff_label.pack(fill="x")

        # ========================= BODY 區域 =========================
        body_frame = tk.Frame(self)
        body_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # Body 左側：影片顯示
        video_frame = tk.Frame(body_frame, width=self.VID_W, height=self.VID_H, bd=1, relief="sunken")
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        video_frame.pack_propagate(False)

        self.lbl_video = tk.Label(video_frame, bg="black")
        self.lbl_video.pack(fill="both", expand=True)
        self.lbl_video.bind("<Button-1>", self._on_roi_start)
        self.lbl_video.bind("<B1-Motion>", self._on_roi_drag)
        self.lbl_video.bind("<ButtonRelease-1>", self._on_roi_end)
        self.roi_rect_id = None

        self._create_control_hint_widget(video_frame)

        # Body 右側：標註樹（設置固定寬度）
        annotation_frame = tk.Frame(body_frame, width=400)
        annotation_frame.pack(side="right", fill="y")
        annotation_frame.pack_propagate(False)  # 防止子元件改變frame大小

        tree_yscroll = ttk.Scrollbar(annotation_frame, orient="vertical")
        tree_yscroll.pack(side="right", fill="y")
        tree_xscroll = ttk.Scrollbar(annotation_frame, orient="horizontal")
        tree_xscroll.pack(side="bottom", fill="x")

        # 初始化時使用基本欄位，後續動態調整
        self.tree = ttk.Treeview(annotation_frame, columns=("frame", "content"),
                                 show="headings", yscrollcommand=tree_yscroll.set,
                                 xscrollcommand=tree_xscroll.set)
        self.tree.pack(side="left", fill="both")
        # 初始化基本欄位設置
        self.tree.heading("frame", text="幀號")
        self.tree.heading("content", text="標註內容")
        self.tree.column("frame", width=frame_width, anchor="center")
        self.tree.column("content", width=content_width, anchor="center")
        tree_yscroll.config(command=self.tree.yview)
        tree_xscroll.config(command=self.tree.xview)

        self.tree.bind("<Double-1>", self._on_edit_annotation)
        self.tree.bind("<Return>", self._on_edit_annotation)
        self.tree.bind('<<TreeviewSelect>>', self._on_treeview_select)
        self._setup_treeview_context_menu()

        # ========================= TIMELINE 區域 =========================
        timeline_frame = tk.Frame(self, relief="groove", bd=1)
        timeline_frame.pack(side="bottom", fill="x", padx=5, pady=5)

        # --- Use a single Grid to manage all timeline components for perfect alignment ---
        timeline_frame.grid_columnconfigure(1, weight=1)

        # -- Row 0: Main Slider --
        # Column 0: A placeholder for the main frame counter
        slider_label_placeholder = tk.Frame(timeline_frame, width=90)
        slider_label_placeholder.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        slider_label_placeholder.pack_propagate(False)

        # The frame counter label will be created further down, and then moved here.
        self.slider_label_placeholder = slider_label_placeholder 

        slider_frame = tk.Frame(timeline_frame)
        slider_frame.grid(row=0, column=1, sticky="ew", pady=(2, 3))
        
        self.slider_var = tk.DoubleVar()
        self.slider = ttk.Scale(slider_frame, from_=0, to=100, orient="horizontal",
                                variable=self.slider_var, command=self._on_slider_move)
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.slider.pack(fill="x", expand=True)

        # -- Row 1: Multi-Track Timelines --
        # Column 0: A container for all track labels
        self.timeline_labels_frame = tk.Frame(timeline_frame)
        self.timeline_labels_frame.grid(row=1, column=0, sticky="ns", padx=(0, 2))

        # Column 1: A container for all track canvases
        stage_tag_parent_frame = tk.Frame(timeline_frame)
        stage_tag_parent_frame.grid(row=1, column=1, sticky="nsew")
        
        # This frame will now directly hold the tracks, replacing the scrollable canvas
        self.timeline_tracks_frame = tk.Frame(stage_tag_parent_frame)
        self.timeline_tracks_frame.pack(fill="both", expand=True)

        # -- Row 2: Navigation Controls --
        nav_frame = tk.Frame(timeline_frame)
        nav_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=3, pady=3)

        # Left side: Go to frame controls
        tk.Label(nav_frame, text="跳至幀:").pack(side="left")
        self.goto_var = tk.IntVar(value=0)
        self.goto_entry = ttk.Entry(nav_frame, textvariable=self.goto_var, width=7)
        self.goto_entry.pack(side="left", padx=2)
        self.goto_entry.bind("<Return>", self._on_goto_frame)
        tk.Button(nav_frame, text="Go", command=self._on_goto_frame).pack(side="left", padx=3)
        
        # Create the frame counter label directly in its final parent container
        self.lbl_frame_num = tk.Label(self.slider_label_placeholder, text="幀: 0 / 0", font=("Arial", 9))
        self.lbl_frame_num.pack(expand=True, fill="both")

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

    def _toggle_ocr_test_window(self, event=None):
        """根據當前模式，切換對應的進階分析視窗"""
        if self.surgery_stage_mode:
            # --- 手術階段ROI模式 ---
            if self.surgery_stage_roi_test_active and self.surgery_stage_roi_test_window:
                self._close_surgery_stage_roi_test_window()
            else:
                self._show_surgery_stage_roi_test_window()
        else:
            # --- OCR ROI模式 (原始功能) ---
            if self.ocr_test_active and self.ocr_test_window:
                self._close_ocr_test_window()
            else:
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
            self.ocr_test_window.title(f"OCR精細測試 - 幀 {self.current_frame_idx}")
            self.ocr_test_window.geometry("1200x800")
            self.ocr_test_window.resizable(True, True)
            
            # 設置視窗關閉時的處理
            self.ocr_test_window.protocol("WM_DELETE_WINDOW", self._close_ocr_test_window)
            
            # 儲存原始ROI圖像用於像素顏色分析和處理
            self.roi_image_original = roi_image
            self.roi_image_processed = self._load_processed_roi_from_disk() # 嘗試從磁碟載入二值化圖
            self.is_processed_mode = self.roi_image_processed is not None # 當前是否為處理模式
            
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
            self.btn_binarize = tk.Button(btn_processing_frame, text="二值化處理" if not self.is_processed_mode else "還原原圖", 
                                         command=self._toggle_binarization,
                                         bg="#E8F4F8" if not self.is_processed_mode else "#F8E8E8", relief="raised" if not self.is_processed_mode else "sunken")
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
            self.processing_status_label = tk.Label(btn_processing_frame, 
                                                   text="原始影像" if not self.is_processed_mode else f"二值化 ({self.binarize_method.get().upper()})",
                                                   font=("Arial", 9), 
                                                   fg="blue" if not self.is_processed_mode else "red")
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
            
            # 重新繪製已選擇的子區域 (OCR模式)
            if hasattr(self, 'sub_regions') and self.sub_regions:
                self._redraw_sub_regions()
            
            # 在手術階段分析模式下，繪製精細分析框
            if hasattr(self, 'surgery_stage_roi_test_active') and self.surgery_stage_roi_test_active:
                self._draw_sub_roi_rect_on_canvas()
            
        except Exception as e:
            print(f"更新ROI顯示時出錯: {e}")

    def _draw_sub_roi_rect_on_canvas(self):
        """在進階分析視窗的Canvas上繪製紅色精細分析框"""
        if not self.surgery_stage_roi_test_active or not hasattr(self, 'roi_canvas'):
            return

        # 清除舊的矩形框
        if hasattr(self, 'sub_roi_rect_id') and self.sub_roi_rect_id:
            self.roi_canvas.delete(self.sub_roi_rect_id)
            self.sub_roi_rect_id = None

        try:
            # 精細區域的相對座標
            sub_roi_coords = (37, 35, 42, 55) # x=37~41, y=35~54
            x1, y1, x2, y2 = sub_roi_coords

            # 根據當前縮放比例計算Canvas上的座標
            scale = self.zoom_level.get()
            border_size = 2  # 與_update_roi_display中的border_size保持一致
            
            canvas_x1 = x1 * scale + border_size
            canvas_y1 = y1 * scale + border_size
            canvas_x2 = x2 * scale + border_size
            canvas_y2 = y2 * scale + border_size
            
            print(f"繪製紅色框: 原始座標({x1},{y1},{x2},{y2}), 縮放({scale}), Canvas座標({canvas_x1},{canvas_y1},{canvas_x2},{canvas_y2})")
            
            # 繪製紅色矩形框
            self.sub_roi_rect_id = self.roi_canvas.create_rectangle(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                outline="red", width=3, tags="sub_roi_rect"
            )
            print(f"紅色框ID: {self.sub_roi_rect_id}")
            
        except Exception as e:
            print(f"繪製 sub ROI 框時出錯: {e}")
            import traceback
            traceback.print_exc()

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
            
    def _on_zoom_change(self, value):
        """縮放改變時的處理"""
        zoom = float(value)
        self.zoom_label.config(text=f"{zoom:.1f}x")
        self._update_roi_display()
        
        # 縮放時也要重繪紅框
        if self.surgery_stage_roi_test_active:
            self._draw_sub_roi_rect_on_canvas()
        
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
        """創建顯示像素信息的面板"""
        info_frame = tk.LabelFrame(parent, text="像素信息")
        info_frame.pack(fill="x", pady=10)

        # 顏色預覽塊
        self.color_preview_label = tk.Label(info_frame, text="", bg="black", width=10, height=3, relief="sunken")
        self.color_preview_label.pack(pady=5, padx=5, fill="x")

        # 像素座標
        self.coord_label = tk.Label(info_frame, text="座標: -, -", font=("Courier", 10))
        self.coord_label.pack(anchor="w", padx=5)
        
        # 顏色值
        self.hex_label = tk.Label(info_frame, text="HEX : #", font=("Courier", 10))
        self.hex_label.pack(anchor="w", padx=5)
        self.rgb_label = tk.Label(info_frame, text="RGB : -, -, -", font=("Courier", 10))
        self.rgb_label.pack(anchor="w", padx=5)
        self.hsv_label = tk.Label(info_frame, text="HSV : -, -, -", font=("Courier", 10))
        self.hsv_label.pack(anchor="w", padx=5)

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
                h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
                h_deg = int(h * 360)
                s_pct = int(s * 100)
                v_pct = int(v * 100)
                
                # 更新顯示
                self.coord_label.config(text=f"座標: {orig_x}, {orig_y}")
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                self.hex_label.config(text=f"HEX : {hex_color.upper()}")
                self.rgb_label.config(text=f"RGB : {r}, {g}, {b}")
                self.hsv_label.config(text=f"HSV : {h_deg}, {s_pct}%, {v_pct}%")
                
                # 更新顏色預覽塊
                self.color_preview_label.config(bg=hex_color)
                
            else:
                # 座標超出範圍，清空顯示
                self.coord_label.config(text="座標: -, -")
                self.hex_label.config(text="HEX : #")
                self.rgb_label.config(text="RGB : -, -, -")
                self.hsv_label.config(text="HSV : -, -, -")
                self.color_preview_label.config(bg="black")
                
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

    def _show_surgery_stage_roi_test_window(self):
        """顯示手術階段ROI的進階分析視窗"""
        if not self.video_file_path or not self.current_surgery_stage_region:
            messagebox.showwarning("提示", "請先載入影片並選擇一個手術階段ROI區域")
            return
            
        if self.surgery_stage_roi_test_window:
            self.surgery_stage_roi_test_window.lift()
            self.surgery_stage_roi_test_window.focus_set()
            return

        try:
            # 獲取當前幀的完整圖像
            full_frame_pil = self._get_full_frame_image_with_cache(self.current_frame_idx)
            if full_frame_pil is None:
                messagebox.showerror("錯誤", f"無法讀取幀 {self.current_frame_idx} 的圖像")
                return

            # 獲取ROI座標並裁剪
            coords = self.surgery_stage_roi_dict.get(self.current_surgery_stage_region)
            if not coords or len(coords) < 4:
                messagebox.showerror("錯誤", f"找不到區域 '{self.current_surgery_stage_region}' 的有效ROI座標")
                return
            
            roi_image = full_frame_pil.crop(tuple(coords))

            # --- 創建視窗 ---
            self.surgery_stage_roi_test_window = tk.Toplevel(self.master)
            self.surgery_stage_roi_test_window.title(f"手術階段ROI分析 - {self.current_surgery_stage_region} (幀 {self.current_frame_idx})")
            self.surgery_stage_roi_test_window.geometry("1000x700")
            self.surgery_stage_roi_test_window.protocol("WM_DELETE_WINDOW", self._close_surgery_stage_roi_test_window)

            # --- 初始化屬性 (與OCR視窗類似) ---
            self.roi_image_original = roi_image
            self.zoom_level = tk.DoubleVar(value=8.0)  # 預設放大8倍
            self.min_zoom = 1.0
            self.max_zoom = 30.0
            self.is_processed_mode = False # 初始化缺失的屬性
            self.sub_region_rects = [] # 初始化缺失的屬性

            # --- 創建UI組件 ---
            main_frame = tk.Frame(self.surgery_stage_roi_test_window)
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # 左側：圖像顯示和控制
            left_frame = tk.Frame(main_frame)
            left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
            
            # 右側：像素資訊面板
            right_frame = tk.Frame(main_frame, width=250)
            right_frame.pack(side="right", fill="y")
            right_frame.pack_propagate(False)

            # 圖像顯示區域
            img_container = tk.LabelFrame(left_frame, text=f"ROI放大圖: {roi_image.size[0]}x{roi_image.size[1]} 像素")
            img_container.pack(fill="both", expand=True)

            canvas_frame = tk.Frame(img_container)
            canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
            h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal")
            v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical")
            
            self.roi_canvas = tk.Canvas(canvas_frame, bg="white", xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
            h_scrollbar.config(command=self.roi_canvas.xview)
            v_scrollbar.config(command=self.roi_canvas.yview)
            
            self.roi_canvas.grid(row=0, column=0, sticky="nsew")
            h_scrollbar.grid(row=1, column=0, sticky="ew")
            v_scrollbar.grid(row=0, column=1, sticky="ns")
            canvas_frame.grid_rowconfigure(0, weight=1)
            canvas_frame.grid_columnconfigure(0, weight=1)

            # 像素資訊面板
            self._create_pixel_info_panel(right_frame)

            # 縮放控制
            zoom_frame = tk.LabelFrame(right_frame, text="縮放控制")
            zoom_frame.pack(fill="x", pady=10)
            zoom_scale = ttk.Scale(zoom_frame, from_=self.min_zoom, to=self.max_zoom, variable=self.zoom_level, orient="horizontal", command=self._on_zoom_change)
            zoom_scale.pack(fill="x", padx=5, pady=5)
            self.zoom_label = tk.Label(zoom_frame, text="8.0x", font=("Courier", 10))
            self.zoom_label.pack()

            # 差異值視覺化 (以矩形陣列顯示)
            diff_frame = tk.LabelFrame(right_frame, text="顏色差異視覺化 (RMSE)")
            diff_frame.pack(fill="x", pady=10)
            # 5x20 的矩形，每格預設 20px
            self.diff_canvas = tk.Canvas(diff_frame, width=120, height=420, bg="white", highlightthickness=1, highlightbackground="gray")
            self.diff_canvas.pack(fill="both", expand=True, padx=5, pady=5)

            # --- 綁定事件並更新顯示 ---
            self._update_roi_display() # 複用此方法來更新canvas
            self.roi_canvas.bind("<Motion>", self._on_canvas_mouse_move) # 複用此方法
            self.roi_canvas.bind("<MouseWheel>", self._on_mouse_wheel) # 複用此方法

            self.surgery_stage_roi_test_active = True
            self._update_status_bar(f"手術階段ROI分析視窗已開啟 ({self.current_surgery_stage_region})")

        except Exception as e:
            messagebox.showerror("錯誤", f"無法顯示手術階段ROI分析視窗: {e}")
            traceback.print_exc()

    def _close_surgery_stage_roi_test_window(self):
        """關閉手術階段ROI分析視窗"""
        if self.surgery_stage_roi_test_window:
            try:
                self.surgery_stage_roi_test_window.destroy()
            except:
                pass
            self.surgery_stage_roi_test_window = None
        self.surgery_stage_roi_test_active = False
        self._update_status_bar("手術階段ROI分析視窗已關閉")

    def _refresh_surgery_stage_test_window(self, full_frame_pil: Image.Image):
        """刷新手術階段ROI分析視窗的內容"""
        if not self.surgery_stage_roi_test_active or not self.current_surgery_stage_region:
            return

        try:
            # 更新視窗標題
            self.surgery_stage_roi_test_window.title(f"手術階段ROI分析 - {self.current_surgery_stage_region} (幀 {self.current_frame_idx})")
            
            # 重新裁剪ROI
            coords = self.surgery_stage_roi_dict.get(self.current_surgery_stage_region)
            if not coords or len(coords) < 4: return
            self.roi_image_original = full_frame_pil.crop(tuple(coords))

            # 刷新顯示
            self._update_roi_display() # 這會重繪放大圖和紅色框

        except Exception as e:
            print(f"刷新手術階段分析視窗時出錯: {e}")
            traceback.print_exc()

    def _get_current_frame_roi(self) -> Optional[Image.Image]:
        """獲取當前幀的ROI圖像 - 優先從磁碟讀取"""
        try:
            # 1. 優先從 data/<video>/<region>/frame_xxx.png 讀取
            roi_image = self._load_roi_from_file(self.current_frame_idx)
            if roi_image:
                print(f"從磁碟快取成功載入 ROI: frame_{self.current_frame_idx}.png")
                return roi_image

            # 2. 如果磁碟快取不存在，再從影片即時裁切
            print(f"ROI快取不存在，嘗試從影片即時裁切 frame {self.current_frame_idx}")
            if not self.cap_ui or not self.cap_ui.isOpened():
                print("UI VideoCapture 未開啟")
                return None
            if not self.roi_coords:
                print("ROI 坐標未設定")
                return None
            
            self.cap_ui.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap_ui.read()
            if not ret:
                print(f"無法讀取幀 {self.current_frame_idx}")
                return None
            
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            roi_image = self._crop_roi(frame_pil)
            return roi_image
            
        except Exception as e:
            print(f"獲取當前幀ROI時出錯: {e}")
            traceback.print_exc()
            return None

    def _load_processed_roi_from_disk(self) -> Optional[Image.Image]:
        """從磁碟載入預處理好的二值化ROI圖像"""
        try:
            if not self.video_file_path or not self.region_name:
                return None
            
            video_name = self.video_file_path.stem
            # 預處理好的二值化圖路徑
            binary_path = Path("data") / video_name / self.region_name / f"frame_{self.current_frame_idx}_binary.png"

            if binary_path.exists():
                print(f"找到預處理的二值化ROI: {binary_path.name}")
                return Image.open(binary_path)
            else:
                print(f"未找到預處理的二值化ROI: {binary_path.name}")
                return None
        except Exception as e:
            print(f"從磁碟載入二值化ROI時出錯: {e}")
            return None

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

            self.total_frames = int(self.cap_ui.get(cv2.CAP_PROP_FRAME_COUNT))
            self.original_vid_w = int(self.cap_ui.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_vid_h = int(self.cap_ui.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap_ui.get(cv2.CAP_PROP_FPS)

            self.video_file_path = Path(filepath) 
            self.video_title = self.video_file_path.stem
            if hasattr(self.lbl_video_path, 'config'):
                 self.lbl_video_path.config(text=str(self.video_file_path.name))
            
            self._load_roi_config() 
            self._load_surgery_stage_roi_config()  # 加載手術階段ROI配置
            self._load_existing_data()
            
            # 確保UI已正確更新
            if hasattr(self, 'region_combobox'):
                self._update_roi_ui()
            if hasattr(self, 'surgery_stage_combobox'):
                self._update_surgery_stage_roi_ui()
            
            if hasattr(self, 'slider'):
                self.slider.config(to=self.total_frames - 1 if self.total_frames > 0 else 0, 
                                   state=tk.NORMAL if self.total_frames > 0 else tk.DISABLED)
            
            self.current_frame_idx = 0 
            if self.total_frames > 0:
                self._show_frame(0) 
            else:
                if hasattr(self, 'lbl_frame_num'): self.lbl_frame_num.config(text="幀: 0 / 0")
                if hasattr(self, 'lbl_video'): self.lbl_video.config(image=None)

            self._update_status_bar(f"已載入: {self.video_title} ({self.total_frames} 幀, {fps:.1f} FPS)")
            print(f"影片載入成功: {self.total_frames} 幀, 解析度: {self.original_vid_w}x{self.original_vid_h}")
            
            # 設定 FPS 供換算秒數
            self.fps = fps if fps and fps > 0 else 30.0
            
            # 載入階段分析（若存在）並渲染標籤
            self._load_stage_analysis()
            self._refresh_stage_tag_ui()
            
            # 預載入OCR數據以優化性能
            self._preload_ocr_data()
            
        except Exception as e:
            messagebox.showerror("錯誤", f"載入影片失敗: {e}")
            print(f"載入影片失敗: {e}")
            traceback.print_exc()
            self.video_file_path = None 

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

    def _show_frame(self, frame_idx: int):
        """
        顯示指定幀，優先從磁碟快取讀取，若無快取則從影片讀取並生成快取。
        - 預設模式：顯示整個frame並畫ROI紅框
        - 二值化模式：只顯示ROI區域的二值化圖
        """
        if not (0 <= frame_idx < self.total_frames):
            print(f"警告：請求的幀 {frame_idx} 超出範圍 (0-{self.total_frames-1})")
            return

        frame_pil = self._get_full_frame_image_with_cache(frame_idx)
        if frame_pil is None:
            print(f"錯誤：無法為幀 {frame_idx} 獲取圖像。")
            self.lbl_video.config(image=None) # 清空畫面
            return

        # --- 更新 Slider/Label 顯示 (提前更新，確保UI同步) ---
        self.slider_var.set(frame_idx)
        self.lbl_frame_num.config(text=f"幀: {frame_idx} / {self.total_frames-1 if self.total_frames > 0 else 0}")
        self.current_frame_idx = frame_idx
        self.goto_var.set(frame_idx)
        
        # --- 如果手術階段分析視窗已開啟，則更新它 ---
        if self.surgery_stage_roi_test_active and self.surgery_stage_roi_test_window:
            self._refresh_surgery_stage_test_window(frame_pil)
        
        # --- 手術階段模式下進行快取比對 ---
        if self.surgery_stage_mode and self.current_surgery_stage_region:
            self._perform_cache_comparison(frame_pil)
        
        display_image = None

        if not self.binarize_mode_var.get():
            # === 預設模式: 顯示完整幀與ROI框 ===
            disp_pil = frame_pil.resize((self.VID_W, self.VID_H), Image.Resampling.BILINEAR)
            
            # 根據當前模式繪製相應的ROI框
            if self.original_vid_w > 0 and self.original_vid_h > 0:
                draw = ImageDraw.Draw(disp_pil)
                scale_x = self.VID_W / self.original_vid_w
                scale_y = self.VID_H / self.original_vid_h
                
                if self.surgery_stage_mode:
                    # 手術階段ROI模式：顯示所有手術階段ROI框
                    for region_name, coords in self.surgery_stage_roi_dict.items():
                        if coords and len(coords) >= 4:
                            x1, y1, x2, y2 = coords
                            # 當前選中的區域用藍色框，其他用綠色框
                            color = "blue" if region_name == self.current_surgery_stage_region else "green"
                            width = 3 if region_name == self.current_surgery_stage_region else 2
                            draw.rectangle(
                                [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y],
                                outline=color, width=width
                            )
                            # 添加區域名稱標籤
                            text_x = x1*scale_x + 5
                            text_y = y1*scale_y - 15 if y1*scale_y > 15 else y1*scale_y + 5
                            draw.text((text_x, text_y), region_name, fill=color)
                else:
                    # OCR ROI模式：顯示OCR ROI框
                    if self.roi_coords:
                        x1, y1, x2, y2 = self.roi_coords
                        draw.rectangle(
                            [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y],
                            outline="red", width=2
                        )
                        # 添加OCR區域標籤
                        if self.region_name:
                            text_x = x1*scale_x + 5
                            text_y = y1*scale_y - 15 if y1*scale_y > 15 else y1*scale_y + 5
                            draw.text((text_x, text_y), f"OCR-{self.region_name}", fill="red")
            
            display_image = disp_pil
        else:
            # === 二值化模式: 只顯示ROI區域的二值化圖 ===
            roi_img = self._crop_roi(frame_pil)
            if roi_img is None:
                print(f"無法取得 ROI 圖像: 幀 {frame_idx}")
                self.lbl_video.config(image=None)
                return
            
            bin_method = self.binarize_method_var.get()
            bin_img = self._apply_binarization(roi_img, bin_method)
            if bin_img is None:
                print(f"二值化失敗，顯示原始 ROI")
                bin_img = roi_img

            roi_w, roi_h = bin_img.size
            if roi_w == 0:
                print(f"警告: 二值化後的ROI寬度為0 (幀 {frame_idx})")
                return 
            scale = self.VID_W / roi_w
            new_w = self.VID_W
            new_h = int(roi_h * scale)
            disp_pil = bin_img.resize((new_w, new_h), Image.Resampling.NEAREST)

            canvas = Image.new("L" if disp_pil.mode == "L" else "RGB", (self.VID_W, self.VID_H), color=0)
            top = (self.VID_H - new_h) // 2
            canvas.paste(disp_pil, (0, top))
            display_image = canvas

        if display_image:
            self.current_display_image = ImageTk.PhotoImage(display_image)
            self.lbl_video.config(image=self.current_display_image)

        # --- 控制提示圖示與焦點 ---
        if hasattr(self, 'control_hint_frame') and self.control_hint_frame:
            try:
                self.control_hint_frame.lift()
            except:
                pass
        self.master.focus_set()

        # 更新 STAGE ROI 預覽
        self._update_stage_roi_preview(frame_pil)
        
        # 更新多軌道時間軸上的位置指示器
        self._update_track_position_indicators()
        
        # 智能同步表格選擇（避免arrow key導航時的干擾）
        self._auto_sync_treeview(frame_idx)

    def _auto_sync_treeview(self, frame_idx: int):
        """智能自動同步表格選擇"""
        # 如果是arrow key導航，跳過同步
        if self._is_arrow_key_navigation:
            self._is_arrow_key_navigation = False  # 重置標記
            return
        
        # 如果用戶手動點擊了表格，跳過同步並重置標誌
        if self._user_clicked_treeview:
            self._user_clicked_treeview = False
            return
            
        # 只在手術階段模式下自動同步
        if not self.surgery_stage_mode:
            return
            
        # 避免在同步進行中觸發
        if self._sync_in_progress:
            return
            
        # 執行同步
        self._sync_treeview_selection_to_frame(frame_idx)

    def _update_stage_roi_preview(self, full_frame_pil: Image.Image):
        """更新右側 Prev/Current ROI 對比視圖。

        - 手術階段模式：使用當前選中的手術階段 ROI
        - OCR 模式：使用當前 OCR ROI (self.roi_coords)
        """
        if not self.surgery_stage_roi_dict or "PEDAL" not in self.surgery_stage_roi_dict:
            # 允許在 OCR 模式下仍然顯示對比
            pass

        try:
            # 如果不是手術階段模式，改為 OCR ROI 對比並提早返回
            if not self.surgery_stage_mode:
                if not self.roi_coords:
                    return
                x1, y1, x2, y2 = self.roi_coords
                curr_roi_image = full_frame_pil.crop((x1, y1, x2, y2))
                prev_image = None
                if self.current_frame_idx > 0:
                    prev_full = self._get_full_frame_image_with_cache(self.current_frame_idx - 1)
                    if prev_full is not None:
                        prev_image = prev_full.crop((x1, y1, x2, y2))

                # 計算 Diff（一般 ROI 使用通用二值化 RMSE）
                diff_val = 0.0
                if prev_image is not None:
                    diff_val = self._calculate_general_roi_diff(prev_image, curr_roi_image)

                # 更新右側對比圖（使用原本的預覽標籤）
                try:
                    if prev_image is not None and hasattr(self, 'stage_roi_preview_label') and self.stage_roi_preview_label:
                        prev_resized = self._resize_roi_for_preview(prev_image, max_size=(200, 150))
                        if prev_resized.mode != "RGB":
                            prev_resized = prev_resized.convert("RGB")
                        prev_photo = ImageTk.PhotoImage(prev_resized)
                        self.stage_roi_preview_label.config(image=prev_photo)
                        self.stage_roi_preview_label.image = prev_photo
                except Exception as e:
                    print(f"更新上一幀 ROI 預覽時出錯: {e}")

                try:
                    if hasattr(self, 'current_roi_preview_label') and self.current_roi_preview_label:
                        curr_resized = self._resize_roi_for_preview(curr_roi_image, max_size=(200, 150))
                        if curr_resized.mode != "RGB":
                            curr_resized = curr_resized.convert("RGB")
                        curr_photo = ImageTk.PhotoImage(curr_resized)
                        self.current_roi_preview_label.config(image=curr_photo)
                        self.current_roi_preview_label.image = curr_photo
                except Exception as e:
                    print(f"更新當前幀 ROI 預覽時出錯: {e}")

                if hasattr(self, 'roi_diff_label') and self.roi_diff_label:
                    self.roi_diff_label.config(text=f"OCR Diff: {diff_val:.4f} ({diff_val*100:.2f}%)")
                return

            # 手術階段模式：決定當前要處理的階段區域 (若未選擇則預設 PEDAL)
            region_name = self.current_surgery_stage_region or "PEDAL"

            if region_name not in self.surgery_stage_roi_dict:
                return  # 該區域尚未設定 ROI

            # 裁剪當前區域 ROI
            region_coords = self.surgery_stage_roi_dict[region_name]
            stage_roi_image = full_frame_pil.crop(tuple(region_coords))

            # 計算roi_diff
            roi_diff_value = 0.0
            diff_matrix = None

            # 取得上一幀同一區域的 ROI
            prev_image = self.previous_stage_roi_images.get(region_name)
            
            # 如果沒有該區域的歷史圖像，嘗試從上一幀生成
            if prev_image is None and hasattr(self, 'current_frame_idx') and self.current_frame_idx > 0:
                prev_image = self._get_previous_frame_roi_for_region(region_name)

            # 計算所有區域的 diff
            if prev_image is not None:
                if region_name == "PEDAL":
                    # PEDAL 區域使用特殊的子區域diff計算
                    roi_diff_value, diff_matrix = self._calculate_pedal_roi_diff(prev_image, stage_roi_image)
                    print(f"PEDAL Diff: {roi_diff_value:.2f}")
                    print(f"PEDAL Diff Matrix:\n{diff_matrix}")
                else:
                    # 其他區域使用通用的RMSE diff計算
                    roi_diff_value = self._calculate_general_roi_diff(prev_image, stage_roi_image)
                    diff_matrix = None

            # 更新diff標籤
            if hasattr(self, 'roi_diff_label') and self.roi_diff_label:
                if prev_image is not None:
                    if region_name == "PEDAL":
                        # PEDAL使用RMSE差異，顯示原始數值
                        self.roi_diff_label.config(text=f"{region_name} Diff: {roi_diff_value:.2f}")
                    else:
                        # 其他region使用像素差異比例，顯示比例和百分比
                        self.roi_diff_label.config(text=f"{region_name} Diff: {roi_diff_value:.4f} ({roi_diff_value*100:.2f}%)")
                else:
                    self.roi_diff_label.config(text=f"{region_name} (No prev frame)")

            # 顯示上一幀的ROI預覽（左側舊視圖仍維持）
            try:
                if prev_image is not None:
                    last_preview_image = self._resize_roi_for_preview(prev_image, max_size=(200, 150))
                    # Tkinter PhotoImage 需要確保 mode 為 RGB
                    if last_preview_image.mode != "RGB":
                        last_preview_image = last_preview_image.convert("RGB")
                    last_preview_photo = ImageTk.PhotoImage(last_preview_image)
                    if hasattr(self, 'stage_roi_preview_label') and self.stage_roi_preview_label:
                        self.stage_roi_preview_label.config(image=last_preview_photo)
                        self.stage_roi_preview_label.image = last_preview_photo
            except Exception as e:
                print(f"更新上一幀ROI預覽時出錯: {e}")
                
            # 顯示當前幀的ROI預覽（左側舊視圖仍維持）
            try:
                current_preview_image = self._resize_roi_for_preview(stage_roi_image, max_size=(200, 150))
                # Tkinter PhotoImage 需要確保 mode 為 RGB
                if current_preview_image.mode != "RGB":
                    current_preview_image = current_preview_image.convert("RGB")
                current_preview_photo = ImageTk.PhotoImage(current_preview_image)
                if hasattr(self, 'current_roi_preview_label') and self.current_roi_preview_label:
                    self.current_roi_preview_label.config(image=current_preview_photo)
                    self.current_roi_preview_label.image = current_preview_photo
            except Exception as e:
                print(f"更新當前幀ROI預覽時出錯: {e}")



            # 保存當前圖像作為下一次比較基準（僅手術階段模式）
            self.previous_stage_roi_images[region_name] = stage_roi_image.copy()

        except Exception as e:
            print(f"更新STAGE ROI預覽時出錯: {e}")
            traceback.print_exc()

    def _resize_roi_for_preview(self, roi_image: Image.Image, max_size: tuple[int, int] = (200, 150)) -> Image.Image:
        """調整ROI圖像大小以適應預覽區域，保持寬高比"""
        try:
            original_width, original_height = roi_image.size
            max_width, max_height = max_size
            
            # 如果原圖已經很小，可以放大顯示
            if original_width <= max_width and original_height <= max_height:
                # 計算放大倍數，但不要過度放大
                scale_x = max_width / original_width
                scale_y = max_height / original_height
                scale = min(scale_x, scale_y, 3.0)  # 最多放大3倍
                
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
            else:
                # 計算縮放比例，保持寬高比
                scale_x = max_width / original_width
                scale_y = max_height / original_height
                scale = min(scale_x, scale_y)
                
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
            
            # 使用高質量重採樣
            resized_image = roi_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return resized_image
            
        except Exception as e:
            print(f"調整ROI預覽大小時發生錯誤: {e}")
            return roi_image  # 返回原圖

    def _get_previous_frame_roi_for_region(self, region_name: str) -> Optional[Image.Image]:
        """獲取指定區域在上一幀的ROI圖像"""
        try:
            if region_name not in self.surgery_stage_roi_dict:
                return None
                
            # 獲取上一幀的完整圖像
            prev_frame_idx = self.current_frame_idx - 1
            if prev_frame_idx < 0:
                return None
                
            prev_full_frame = self._get_full_frame_image_with_cache(prev_frame_idx)
            if prev_full_frame is None:
                return None
                
            # 裁剪該區域的ROI
            region_coords = self.surgery_stage_roi_dict[region_name]
            prev_roi_image = prev_full_frame.crop(tuple(region_coords))
            
            return prev_roi_image
            
        except Exception as e:
            print(f"獲取上一幀ROI時發生錯誤: {e}")
            return None

    def _calculate_general_roi_diff(self, prev_img: Image.Image, curr_img: Image.Image) -> float:
        """計算兩張ROI圖像的二值化像素差異比例（與surgery_analysis_process.py的calculate_binary_diff一致）"""
        try:
            # 對兩張圖像進行二值化處理（使用與core_processing.py一致的方法）
            # 使用固定的rule方法參數，與surgery_analysis_process.py保持一致
            prev_binary = self._apply_core_binarization(prev_img, "rule")
            curr_binary = self._apply_core_binarization(curr_img, "rule")
            
            if prev_binary is None or curr_binary is None:
                return 0.0
            
            # 轉換為numpy數組
            if isinstance(prev_binary, np.ndarray):
                prev_arr = prev_binary
            else:
                prev_arr = np.array(prev_binary)
                
            if isinstance(curr_binary, np.ndarray):
                curr_arr = curr_binary
            else:
                curr_arr = np.array(curr_binary)
            
            # 檢查尺寸是否一致
            if prev_arr.shape != curr_arr.shape:
                return 0.0
            
            # 使用與surgery_analysis_process.py相同的計算方法
            # 轉換為二值（0/1）
            b1 = (prev_arr > 127).astype(np.uint8)
            b2 = (curr_arr > 127).astype(np.uint8)
            
            # 計算XOR差異，返回差異像素的比例
            diff = np.logical_xor(b1, b2)
            return float(np.mean(diff))
            
        except Exception as e:
            print(f"計算通用ROI diff時發生錯誤: {e}")
            return 0.0

    def _apply_core_binarization(self, image: Image.Image, method: str) -> Optional[np.ndarray]:
        """應用與core_processing.py完全一致的二值化處理"""
        try:
            # 轉換為OpenCV格式 (BGR)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            if method == "otsu":
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                return binary
            
            elif method == "rule":
                # 使用與core_processing.py完全相同的參數
                hsv_s_thresh = 30
                gray_thresh = 150
                
                hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                s_pct = (s / 255.0) * 100
                mask = (s_pct < hsv_s_thresh) & (gray > gray_thresh)
                binary = np.zeros_like(gray, dtype=np.uint8)
                binary[mask] = 255
                return binary
            
            else:
                print(f"未支持的二值化方法: {method}")
                return None
                
        except Exception as e:
            print(f"core二值化處理失敗: {e}")
            return None

    def _calculate_pedal_roi_diff(self, prev_img: Image.Image, curr_img: Image.Image) -> tuple[float, np.ndarray | None]:
        """計算兩張 PEDAL ROI 圖像在指定精細區域內的平均RGB顏色差異"""
        try:
            # 更新後的精細區域相對座標
            sub_roi_coords = (20, 13, 26, 19) # x=19~26, y=13~19

            # 從兩張圖像中裁剪出精細區域
            prev_sub_roi = prev_img.crop(sub_roi_coords)
            curr_sub_roi = curr_img.crop(sub_roi_coords)
            
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
            
            # --- 增加調試輸出 ---
            rmse_list = rmse_per_pixel.flatten().tolist()
            rmse_str_list = [f"{val:.2f}" for val in rmse_list]
            print(f"Frame {self.current_frame_idx} - PEDAL Diff Avg: {average_rmse:.2f}, (RMSE per pixel): {rmse_str_list}")
            
            # 返回所有像素顏色距離的平均值和完整的差異矩陣
            return average_rmse, rmse_per_pixel
            
        except Exception as e:
            print(f"計算 roi_diff 時出錯: {e}")
            return 0.0, None

    def _get_full_frame_image_with_cache(self, frame_idx: int) -> Optional[Image.Image]:
        """
        獲取單個完整幀的PIL圖像，實現了磁碟快取機制。
        1. 優先從 `data/<video_name>/frame_cache/frame_{frame_idx}.jpg` 讀取。
        2. 如果快取不存在，則從 `self.cap_ui` 讀取。
        3. 從影片讀取成功後，立刻將其寫入快取資料夾以備後用。
        """
        cache_dir = self._get_frame_cache_dir()
        if not cache_dir:
            print("錯誤: 無法獲取快取目錄")
            return None # 無法獲取快取目錄，直接返回

        cached_frame_path = cache_dir / f"frame_{frame_idx}.jpg"

        # 1. 嘗試從快取讀取
        if cached_frame_path.exists():
            try:
                return Image.open(cached_frame_path)
            except Exception as e:
                print(f"警告: 快取檔案 {cached_frame_path} 損壞，將重新生成: {e}")

        # 2. 快取不存在，從影片讀取 (Fallback)
        if not self.cap_ui or not self.cap_ui.isOpened():
            print(f"錯誤: UI VideoCapture 未開啟，無法讀取幀 {frame_idx}")
            return None
        
        self.cap_ui.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = self.cap_ui.read()

        if not ret:
            print(f"警告：從影片讀取幀 {frame_idx} 失敗")
            return None
        
        # 轉換為PIL圖像
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        # 3. 寫入快取
        try:
            # 使用中等品質(85)的JPEG儲存，以平衡品質和檔案大小
            frame_pil.save(cached_frame_path, "JPEG", quality=85)
        except Exception as e:
            print(f"警告: 無法寫入快取檔案 {cached_frame_path}: {e}")
            
        return frame_pil

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
        if self.surgery_stage_mode:
            self._on_surgery_stage_roi_start(event)
        else:
            self._on_ocr_roi_start(event)

    def _on_roi_drag(self, event):
        """Draws a temporary rectangle while dragging."""
        if self.surgery_stage_mode:
            self._on_surgery_stage_roi_drag(event)
        else:
            self._on_ocr_roi_drag(event)

    def _on_roi_end(self, event):
        """Handles ROI selection completion."""
        if self.surgery_stage_mode:
            self._on_surgery_stage_roi_end(event)
        else:
            self._on_ocr_roi_end(event)

    def _on_ocr_roi_start(self, event):
        """Records the starting coordinates for OCR ROI selection."""
        video_x, video_y = self._canvas_to_video_coords(event.x, event.y)
        self.roi_start_coords = (video_x, video_y)

    def _on_ocr_roi_drag(self, event):
        """Draws a temporary rectangle while dragging for OCR ROI."""
        if not self.roi_start_coords:
            return

        x1, y1 = self.roi_start_coords
        x2, y2 = event.x, event.y

        # --- 在 lbl_video 上繪製拖動矩形 (需要 Canvas) ---
        # If using Canvas:
        # if self.roi_rect_id: self.lbl_video.delete(self.roi_rect_id)
        # self.roi_rect_id = self.lbl_video.create_rectangle(x1, y1, x2, y2, outline="blue", width=1, tags="roi_rect")
        pass # No easy way to draw temporary rect on Label without redrawing image constantly

    def _on_ocr_roi_end(self, event):
        """
        使用者在畫面上拖曳完 OCR ROI 框後呼叫：
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
            print("OCR ROI 太小，已忽略。")
            self._show_frame(self.current_frame_idx)
            return

        # 儲存ROI變更
        if new_roi != self.roi_coords:
            self.roi_coords = new_roi
            self.roi_dict[self.region_name] = list(self.roi_coords)
            
            # 拖曳ROI後，詢問是否要儲存
            result = messagebox.askyesno("儲存設定", f"OCR ROI區域已更新，是否儲存到配置檔案？")
            if result:
                self._save_roi_config()

            # 清空快取
            self.change_cache.clear()

            # 更新UI
            self._update_roi_fields()
            status_msg = f"{self.region_name} OCR ROI 更新: {self.roi_coords}"
            if result:
                status_msg += " (已儲存)"
            else:
                status_msg += " (未儲存)"
            self._update_status_bar(status_msg)
        
        self._show_frame(self.current_frame_idx)

    def _on_surgery_stage_roi_start(self, event):
        """Records the starting coordinates for surgery stage ROI selection."""
        video_x, video_y = self._canvas_to_video_coords(event.x, event.y)
        self.surgery_stage_roi_start_coords = (video_x, video_y)

    def _on_surgery_stage_roi_drag(self, event):
        """Draws a temporary rectangle while dragging for surgery stage ROI."""
        if not hasattr(self, 'surgery_stage_roi_start_coords') or not self.surgery_stage_roi_start_coords:
            return
        pass # No easy way to draw temporary rect on Label without redrawing image constantly

    def _on_surgery_stage_roi_end(self, event):
        """
        使用者在畫面上拖曳完手術階段ROI框後呼叫
        """
        if not hasattr(self, 'surgery_stage_roi_start_coords') or self.surgery_stage_roi_start_coords is None:
            self._show_frame(self.current_frame_idx)
            return

        if not self.current_surgery_stage_region:
            messagebox.showwarning("警告", "請先選擇手術階段區域")
            self.surgery_stage_roi_start_coords = None
            self._show_frame(self.current_frame_idx)
            return

        # 計算並驗證ROI座標
        start_x_orig, start_y_orig = self.surgery_stage_roi_start_coords
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
        self.surgery_stage_roi_start_coords = None

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            print("手術階段ROI 太小，已忽略。")
            self._show_frame(self.current_frame_idx)
            return

        # 儲存手術階段ROI變更
        old_roi = self.surgery_stage_roi_dict.get(self.current_surgery_stage_region)
        if list(new_roi) != old_roi:
            self.surgery_stage_roi_dict[self.current_surgery_stage_region] = list(new_roi)
            
            # 更新手術階段ROI座標顯示
            self._update_surgery_stage_roi_fields()
            
            # 拖曳ROI後，詢問是否要儲存
            result = messagebox.askyesno("儲存設定", f"手術階段ROI區域已更新，是否儲存到配置檔案？")
            if result:
                self._save_surgery_stage_roi_config()

            status_msg = f"{self.current_surgery_stage_region} 手術階段ROI 更新: {new_roi}"
            if result:
                status_msg += " (已儲存)"
            else:
                status_msg += " (未儲存)"
            self._update_status_bar(status_msg)
        
        self._show_frame(self.current_frame_idx)

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
        # slider 更新後，標籤不需要重繪，位置與時間軸一致

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

    def _configure_treeview_columns(self, mode: str, region: str = None):
        """動態配置TreeView的欄位"""
        try:
            if mode == "ocr":
                # OCR模式：只需要幀號和標註內容
                columns = ("frame", "content")
                self.tree.config(columns=columns)
                self.tree.heading("frame", text="幀號")
                self.tree.heading("content", text="標註內容")
                self.tree.column("frame", width=frame_width, anchor="center")
                self.tree.column("content", width=content_width, anchor="center")
                
            elif mode == "surgery_stage":
                if region == "STAGE":
                    # STAGE區域：起始幀、模式類型、結束幀、IOP、Asp、Vac
                    columns = ("frame", "content", "end_frame", "iop", "asp", "vac")
                    self.tree.config(columns=columns)
                    self.tree.heading("frame", text="起始幀")
                    self.tree.heading("content", text="模式類型")
                    self.tree.heading("end_frame", text="結束幀")
                    self.tree.heading("iop", text="IOP")
                    self.tree.heading("asp", text="Asp")
                    self.tree.heading("vac", text="Vac")
                    self.tree.column("frame", width=frame_width, anchor="center")
                    self.tree.column("content", width=content_width, anchor="center")
                    self.tree.column("end_frame", width=end_frame_width, anchor="center")
                    self.tree.column("iop", width=iop_width, anchor="center")
                    self.tree.column("asp", width=asp_width, anchor="center")
                    self.tree.column("vac", width=vac_width, anchor="center")
                else:
                    # PEDAL或其他區域：起始幀、模式類型、結束幀
                    columns = ("frame", "content", "end_frame")
                    self.tree.config(columns=columns)
                    self.tree.heading("frame", text="起始幀")
                    self.tree.heading("content", text="模式類型")
                    self.tree.heading("end_frame", text="結束幀")
                    self.tree.column("frame", width=frame_width, anchor="center")
                    self.tree.column("content", width=content_width, anchor="center")
                    self.tree.column("end_frame", width=end_frame_width, anchor="center")
                    
        except Exception as e:
            print(f"配置TreeView欄位時出錯: {e}")

    def _refresh_treeview(self):
        """根據當前模式重新載入表格內容"""
        # 清空現有表格
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if self.surgery_stage_mode:
            # 手術階段分析模式
            current_region = getattr(self, 'current_surgery_stage_region', None)
            self._configure_treeview_columns("surgery_stage", current_region)
            self._load_stage_analysis_to_treeview()
        else:
            # OCR標註模式
            self._configure_treeview_columns("ocr")
            self._load_ocr_annotations_to_treeview()

    def _load_ocr_annotations_to_treeview(self):
        """載入OCR標註到表格"""
        if not (self.video_file_path and self.region_name):
            return
            
        # 載入標註檔案
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
            ocr_value = self.annotations.get(frame_idx, "")
            
            item_id_str = f"F{frame_idx}"
            self.tree.insert("", "end", iid=item_id_str, 
                            values=(frame_idx, ocr_value))

    def _get_ocr_text_at_frame(self, region_name: str, target_frame: int) -> str:
        """根據frame推算指定region的OCR內容（使用緩存）"""
        try:
            # 優先使用緩存
            if region_name in self.ocr_cache:
                frame_to_ocr = self.ocr_cache[region_name]
                
                # 如果直接找到該frame的OCR
                if target_frame in frame_to_ocr:
                    return frame_to_ocr[target_frame]
                
                # 如果沒有直接找到，找最近的前一個有OCR的frame
                sorted_frames = sorted(frame_to_ocr.keys())
                for i in range(len(sorted_frames) - 1, -1, -1):
                    if sorted_frames[i] <= target_frame:
                        return frame_to_ocr[sorted_frames[i]]
                
                return ""
            
            # 如果緩存中沒有，嘗試即時載入（fallback）
            if not self.video_file_path:
                return ""
            
            ocr_path = Path("data") / self.video_title / f"{region_name}_ocr_testing.jsonl"
            if not ocr_path.exists():
                print(f"OCR檔案不存在: {ocr_path}")
                return ""
            
            print(f"警告：{region_name} 未在緩存中，進行即時載入")
            
            # 讀取OCR數據並建立frame到OCR文本的映射
            frame_to_ocr = {}
            with open(ocr_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            ocr_text = obj.get("ocr_text", obj.get("text", ""))
                            
                            # 處理單個frame的OCR
                            if "frame" in obj:
                                frame_idx = int(obj["frame"])
                                frame_to_ocr[frame_idx] = ocr_text
                            
                            # 處理multi_digit_group的matched_frames
                            if obj.get("type") == "multi_digit_group" and "matched_frames" in obj:
                                for matched_frame in obj["matched_frames"]:
                                    frame_to_ocr[matched_frame] = ocr_text
                    except json.JSONDecodeError:
                        continue
            
            # 存入緩存以備後用
            self.ocr_cache[region_name] = frame_to_ocr
            
            # 如果直接找到該frame的OCR
            if target_frame in frame_to_ocr:
                return frame_to_ocr[target_frame]
            
            # 如果沒有直接找到，找最近的前一個有OCR的frame
            sorted_frames = sorted(frame_to_ocr.keys())
            for i in range(len(sorted_frames) - 1, -1, -1):
                if sorted_frames[i] <= target_frame:
                    return frame_to_ocr[sorted_frames[i]]
            
            return ""
            
        except Exception as e:
            print(f"讀取 {region_name} 在frame {target_frame} 的OCR內容時出錯: {e}")
            return ""

    def _is_stage_start_frame(self, target_frame: int, pedal_segments: list) -> bool:
        """判斷指定frame是否為STAGE開始（PEDAL為pattern 1）"""
        try:
            # 查找包含target_frame的PEDAL段落
            for segment in pedal_segments:
                start_frame = segment.get('start_frame', 0)
                end_frame = segment.get('end_frame', start_frame)
                pattern = segment.get('pattern')
                
                # 如果target_frame在這個PEDAL段落範圍內，且pattern為1
                if start_frame <= target_frame <= end_frame and pattern == 1:
                    # 進一步檢查：target_frame應該正好是這個pattern 1段落的開始
                    # 或者是在pattern 1段落內的STAGE開始點
                    return True
            
            return False
            
        except Exception as e:
            print(f"判斷STAGE開始frame時出錯: {e}")
            return False

    def _preload_ocr_data(self):
        """預載入region1~3的OCR數據到記憶體中"""
        try:
            if not self.video_file_path:
                return
            
            print("開始預載入OCR數據...")
            regions_to_load = ['region1', 'region2', 'region3']
            
            for region_name in regions_to_load:
                ocr_path = Path("data") / self.video_title / f"{region_name}_ocr_testing.jsonl"
                if not ocr_path.exists():
                    print(f"OCR檔案不存在，跳過: {ocr_path}")
                    continue
                
                # 讀取OCR數據並建立frame到OCR文本的映射
                frame_to_ocr = {}
                with open(ocr_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                ocr_text = obj.get("ocr_text", obj.get("text", ""))
                                
                                # 處理單個frame的OCR
                                if "frame" in obj:
                                    frame_idx = int(obj["frame"])
                                    frame_to_ocr[frame_idx] = ocr_text
                                
                                # 處理multi_digit_group的matched_frames
                                if obj.get("type") == "multi_digit_group" and "matched_frames" in obj:
                                    for matched_frame in obj["matched_frames"]:
                                        frame_to_ocr[matched_frame] = ocr_text
                        except json.JSONDecodeError:
                            continue
                
                self.ocr_cache[region_name] = frame_to_ocr
                print(f"已載入 {region_name} 的 {len(frame_to_ocr)} 筆OCR數據")
            
            print("OCR數據預載入完成")
            
        except Exception as e:
            print(f"預載入OCR數據時出錯: {e}")

    def _load_stage_analysis_to_treeview(self):
        """載入手術階段分析結果到表格"""
        if not hasattr(self, 'stage_analysis') or not self.stage_analysis:
            print("沒有手術階段分析數據")
            return
            
        # 檢查數據結構
        regions_data = self.stage_analysis.get('regions', {})
        if not regions_data:
            print("stage_analysis中沒有regions數據")
            return
            
        current_region = getattr(self, 'current_surgery_stage_region', None)
        if not current_region:
            print("沒有選擇當前手術階段區域")
            return
            
        if current_region not in regions_data:
            print(f"當前手術階段區域 '{current_region}' 沒有分析數據")
            print(f"可用區域: {list(regions_data.keys())}")
            return
        
        segments = regions_data[current_region]
        print(f"載入 {current_region} 區域的 {len(segments)} 個段落")
        
        if not segments:
            print(f"警告：{current_region} 區域沒有段落數據")
            return
        
        # 取得PEDAL區域的數據，用於識別STAGE開始
        pedal_segments = regions_data.get('PEDAL', [])
        
        for i, segment in enumerate(segments):
            start_frame = segment.get('start_frame', 0)
            end_frame = segment.get('end_frame', start_frame)
            pattern = segment.get('pattern', f'未知模式_{i}')
            
            # 格式化內容顯示
            if isinstance(pattern, int):
                content = f"模式 {pattern}"
            else:
                content = str(pattern)
            
            # 初始化OCR值
            iop_value = ""
            asp_value = ""
            vac_value = ""
            
            # 只有在當前選擇的是STAGE區域時，才讀取OCR數據
            if current_region == 'STAGE':
                # 檢查該frame是否為STAGE開始（PEDAL為pattern 1）
                is_stage_start = self._is_stage_start_frame(start_frame, pedal_segments)
                
                if is_stage_start:
                    print(f"識別到STAGE開始frame: {start_frame}（PEDAL為pattern 1）")
                    # 讀取region1~3的OCR內容
                    iop_value = self._get_ocr_text_at_frame("region1", start_frame)
                    asp_value = self._get_ocr_text_at_frame("region2", start_frame)
                    vac_value = self._get_ocr_text_at_frame("region3", start_frame)
                    print(f"  OCR讀取結果 - IOP: {iop_value}, Asp: {asp_value}, Vac: {vac_value}")
            
            item_id_str = f"S{start_frame}"
            # 根據當前區域決定插入的欄位數量
            if current_region == 'STAGE':
                # STAGE區域：6個欄位（包含OCR數據）
                self.tree.insert("", "end", iid=item_id_str, 
                                values=(start_frame, content, end_frame, iop_value, asp_value, vac_value))
            else:
                # PEDAL或其他區域：3個欄位
                self.tree.insert("", "end", iid=item_id_str, 
                                values=(start_frame, content, end_frame))
            
            print(f"  添加段落 {i+1}: 幀 {start_frame}-{end_frame}, 模式 {pattern}")
        
        print(f"表格載入完成，共添加 {len(segments)} 個項目")

    def _sync_treeview_selection_to_frame(self, target_frame: int, force: bool = False):
        """將表格選擇同步到指定幀"""
        if self._sync_in_progress and not force:
            return
            
        self._sync_in_progress = True
        try:
            print(f"同步表格選擇到幀 {target_frame}, force={force}")
            
            best_item = None
            best_diff = float('inf')
            
            # 遍歷表格項目找到最接近的幀
            for item in self.tree.get_children():
                values = self.tree.item(item, 'values')
                if not values:
                    continue
                    
                try:
                    item_frame = int(values[0])
                    diff = abs(item_frame - target_frame)
                    
                    if self.surgery_stage_mode:
                        # 手術階段模式：檢查目標幀是否在段落範圍內
                        end_frame = int(values[2]) if len(values) > 2 and values[2] else item_frame
                        if item_frame <= target_frame <= end_frame and diff < best_diff:
                            best_item = item
                            best_diff = diff
                    else:
                        # OCR模式：找最接近的幀
                        if diff < best_diff:
                            best_item = item
                            best_diff = diff
                            
                except (ValueError, IndexError):
                    continue
            
            if best_item:
                # 選中項目
                self.tree.selection_set(best_item)
                self.tree.focus(best_item)
                
                # 滾動到中間
                self._center_treeview_item(best_item)
                print(f"同步完成：選中項目 {best_item}")
            else:
                print("沒有找到匹配的表格項目")
                
        finally:
            self._sync_in_progress = False

    def _center_treeview_item(self, item_id: str):
        """將指定的表格項目滾動到視窗中間"""
        try:
            # 獲取所有項目
            all_items = self.tree.get_children()
            if not all_items or item_id not in all_items:
                return
                
            # 計算項目索引
            item_index = list(all_items).index(item_id)
            total_items = len(all_items)
            
            if total_items <= 1:
                return
                
            # 計算要滾動到的位置（將目標項目置於中間）
            visible_items = 10  # 假設可見項目數量
            scroll_to = max(0, item_index - visible_items // 2)
            scroll_fraction = scroll_to / max(1, total_items - visible_items)
            scroll_fraction = min(1.0, max(0.0, scroll_fraction))
            
            # 執行滾動
            self.tree.yview_moveto(scroll_fraction)
            print(f"表格滾動到位置: {scroll_fraction:.2f}")
            
        except Exception as e:
            print(f"滾動表格失敗: {e}")

    def _on_goto_frame(self, event=None):
        try:
            idx = int(self.goto_var.get())
        except (ValueError, TypeError):
            return
        self._show_frame(idx)

    def _update_status_bar(self, message: str):
        """更新狀態列訊息"""
        if hasattr(self, 'status_var'):
            self.status_var.set(message)
        print(f"狀態: {message}")

    def _clear_previous_video_data(self):
        """清除所有與當前影片相關的數據和UI狀態"""
        self.playback_active = False
        if hasattr(self, 'after_id') and self.after_id:
            self.master.after_cancel(self.after_id)
            self.after_id = None
        
        self.current_frame_idx = 0
        self.annotations.clear()
        self.change_cache.clear()
        self.ocr_cache.clear()  # 清除OCR緩存
        
        self.last_known_change_pos = -1
        self.last_search_direction = "next"
        self.last_loaded_roi_frame = -1
        self.last_loaded_roi_image = None
        self.frame_cache_dir = None
        
        if hasattr(self, 'tree'):
            for item in self.tree.get_children():
                self.tree.delete(item)

        self.video_file_path = None
        self.video_title = ""
        self.total_frames = 0
        self.roi_coords = None
        self.region_name = ""
        self.unsaved_changes = False
        
        # 清除階段標籤
        self.stage_analysis = None
        self.stage_tag_meta.clear()
        if self.stage_tag_canvas:
            try:
                self.stage_tag_canvas.delete("all")
            except:
                pass
        
        if self.cap_ui and self.cap_ui.isOpened():
            self.cap_ui.release()
        self.cap_ui = None

        if hasattr(self, 'status_var'):
            self.status_var.set("就緒")
        if hasattr(self, 'slider'):
            self.slider.set(0)
            self.slider.config(to=0)
        if hasattr(self, 'frame_label'):
            self.frame_label.config(text="幀: 0 / 0")
        if hasattr(self, 'time_label'):
            self.time_label.config(text="00:00.000 / 00:00.000")
        
        # 清除畫布
        if hasattr(self, 'canvas'):
            self.canvas.delete("all")

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
            
            # 設置標誌，表示用戶手動點擊了表格
            self._user_clicked_treeview = True
            
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
        
        # 切換到新區域
        old_region = self.region_name
        self.region_name = new_region
        self.roi_coords = tuple(self.roi_dict[new_region])
        
        # 清空快取 (重要：避免新舊 region 資料混合)
        self.change_cache.clear()
        self.annotations.clear()
        
        # 載入新區域的資料
        self._load_existing_data()
        
        # 更新 ROI 顯示
        self._update_roi_fields()
        
        # 重新顯示當前幀（使用新的 ROI）
        self._show_frame(self.current_frame_idx)
        
        print(f"已切換區域: {old_region} -> {new_region}")
        self._update_status_bar(f"已切換到區域: {new_region}")

    def _save_roi_config(self):
        """儲存 ROI 設定到檔案，支持新的header格式"""
        roi_file = get_roi_config_path()
        if not roi_file:
            return
        
        # 確保目錄存在
        roi_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 構建保存數據：如果有header配置，使用新格式；否則使用舊格式
            save_data = {}
            roi_header_dict = getattr(self, 'roi_header_dict', {})
            
            for region_name, roi_coords in self.roi_dict.items():
                if region_name in roi_header_dict:
                    # 新格式：[[roi_coords], [header_coords]]
                    save_data[region_name] = [roi_coords, roi_header_dict[region_name]]
                else:
                    # 舊格式：直接保存座標
                    save_data[region_name] = roi_coords
            
            with open(roi_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"ROI 設定已儲存至 {roi_file}")
            if roi_header_dict:
                print(f"包含 header 配置的區域: {list(roi_header_dict.keys())}")
        except Exception as e:
            print(f"儲存 ROI 設定失敗: {e}")

    def _load_roi_config(self):
        """載入全域 ROI 設定"""
        roi_file = get_roi_config_path()
        
        try:
            if roi_file.exists():
                # 使用統一的配置加載函數，自動處理新舊格式
                self.roi_dict = load_roi_config(roi_file)
                
                # 同時載入header配置（如果存在）
                try:
                    self.roi_header_dict = load_roi_header_config(roi_file)
                    if self.roi_header_dict:
                        print(f"已載入 ROI header 設定: {self.roi_header_dict}")
                except Exception as e:
                    print(f"載入 ROI header 設定失敗: {e}")
                    self.roi_header_dict = {}
                
                print(f"已載入全域 ROI 設定: {self.roi_dict}")
            else:
                print(f"全域 ROI 設定檔不存在，將建立預設配置")
                # 如果檔案不存在，建立一個預設配置
                self.roi_dict = {
                    "region2": [1640, 445, 1836, 525]
                }
                self.roi_header_dict = {}
        except Exception as e:
            print(f"載入全域 ROI 設定失敗: {e}")
            # 載入失敗時使用預設配置
            self.roi_dict = {
                "region2": [1640, 445, 1836, 525]
            }
            self.roi_header_dict = {}
        
        # 更新 UI（如果已建立）
        if hasattr(self, 'region_combobox'):
            self._update_roi_ui()
        
        print(f"最終 ROI 字典: {self.roi_dict}")
        print(f"最終 ROI header 字典: {getattr(self, 'roi_header_dict', {})}")

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
        self._update_status_bar(f"已新增區域 {name}")
        # 新增完成後，把 ROI 座標同步到 Spinbox 方便微調
        self._update_roi_fields()

    def _update_roi_ui(self):
        """更新OCR ROI UI"""
        if not hasattr(self, 'region_combobox'):
            return
            
        # 更新下拉選單選項
        regions = list(self.roi_dict.keys())
        self.region_combobox['values'] = regions
        
        # 設定當前選中項目
        if self.region_name in regions:
            self.region_combobox.set(self.region_name)
        elif regions:
            # 如果當前區域不在列表中，設定第一個區域為當前區域
            self.region_name = regions[0]
            self.region_combobox.set(regions[0])
            # 設定對應的ROI座標
            if self.region_name in self.roi_dict:
                self.roi_coords = tuple(self.roi_dict[self.region_name])
                self._update_roi_fields()
        
        print(f"OCR ROI UI已更新，當前區域: {self.region_name}")

    def _get_full_frame_image_with_cache(self, frame_idx: int) -> Optional[Image.Image]:
        """
        獲取單個完整幀的PIL圖像，實現了磁碟快取機制。
        1. 優先從 `data/<video_name>/frame_cache/frame_{frame_idx}.jpg` 讀取。
        2. 如果快取不存在，則從 `self.cap_ui` 讀取。
        3. 從影片讀取成功後，立刻將其寫入快取資料夾以備後用。
        """
        cache_dir = self._get_frame_cache_dir()
        if not cache_dir:
            print("錯誤: 無法獲取快取目錄")
            return None # 無法獲取快取目錄，直接返回

        cached_frame_path = cache_dir / f"frame_{frame_idx}.jpg"

        # 1. 嘗試從快取讀取
        if cached_frame_path.exists():
            try:
                return Image.open(cached_frame_path)
            except Exception as e:
                print(f"警告: 快取檔案 {cached_frame_path} 損壞，將重新生成: {e}")

        # 2. 快取不存在，從影片讀取 (Fallback)
        if not self.cap_ui or not self.cap_ui.isOpened():
            print(f"錯誤: UI VideoCapture 未開啟，無法讀取幀 {frame_idx}")
            return None
        
        self.cap_ui.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = self.cap_ui.read()

        if not ret:
            print(f"警告：從影片讀取幀 {frame_idx} 失敗")
            return None
        
        # 轉換為PIL圖像
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        # 3. 寫入快取
        try:
            # 使用中等品質(85)的JPEG儲存，以平衡品質和檔案大小
            frame_pil.save(cached_frame_path, "JPEG", quality=85)
        except Exception as e:
            print(f"警告: 無法寫入快取檔案 {cached_frame_path}: {e}")
            
        return frame_pil

    def _save_annotations(self, region_name: str):
        """儲存標註結果 - 手動儲存時處理當前分析結果"""
        
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

    def _load_change_frames(self, region_name: str):
        """載入變化幀列表 - 支援 JSONL 格式（單行陣列）"""
        try:
            if not self.video_file_path:
                print(f"錯誤: _load_change_frames 無法獲取有效的 video_file_path for region {region_name}.")
                return

            video_data_dir = Path("data") / self.video_title
            change_frames = None

            # 嘗試 .jsonl 格式
            change_path = video_data_dir / f"{region_name}_ocr_testing.jsonl"
            if change_path.exists():
                print(f"載入變化幀檔案: {change_path}")
                change_frames = []
                # 使用buffer來高效處理matched_frames的順序插入
                pending_matched_frames = []  # buffer: [(frame_idx, ocr_text, annotation_text)]
                
                with open(change_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                ocr_text_val = obj.get("ocr_text", obj.get("text", ""))
                                current_frame = None
                                
                                # 確定當前frame（統一處理）
                                if "frame" in obj:
                                    # single_digit 類型
                                    current_frame = int(obj["frame"])
                                elif obj.get("type") == "multi_digit_group" and "source_frame" in obj:
                                    # multi_digit_group 類型
                                    current_frame = int(obj["source_frame"])
                                
                                # 如果有當前frame，進行統一的處理流程
                                if current_frame is not None:
                                    # 先處理buffer中比當前frame更早的matched_frames
                                    self._flush_pending_frames_before(pending_matched_frames, current_frame, change_frames)
                                    
                                    # 添加當前frame到change_frames
                                    change_frames.append(current_frame)
                                    
                                    # 根據類型設置不同的標註
                                    if ocr_text_val is not None:
                                        self.annotations[current_frame] = ocr_text_val
                                        # if "frame" in obj:
                                        #     # single_digit: 直接使用OCR文字
                                        #     self.annotations[current_frame] = ocr_text_val
                                        # else:
                                        #     # multi_digit_group: 標記為群組
                                        #     self.annotations[current_frame] = f"[G] {ocr_text_val}"
                                
                                # multi_digit_group的額外處理：將matched_frames添加到buffer
                                if obj.get("type") == "multi_digit_group" and "matched_frames" in obj and isinstance(obj["matched_frames"], list):
                                    for matched_frame in obj["matched_frames"] :
                                        annotation_text = f"{ocr_text_val}" if ocr_text_val is not None else None
                                        pending_matched_frames.append((matched_frame, ocr_text_val, annotation_text))
                        
                        except json.JSONDecodeError as e:
                            print(f"第 {line_num} 行解析失敗: {e}")
                        except (ValueError, KeyError) as e:
                            print(f"第 {line_num} 行數據格式錯誤: {e}")
                    
                    # 處理剩餘的buffer內容
                    self._flush_all_pending_frames(pending_matched_frames, change_frames)

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

    def _flush_pending_frames_before(self, pending_matched_frames: list, current_frame: int, change_frames: list):
        """
        處理buffer中比current_frame更早的matched_frames，並按順序插入到change_frames中
        """
        # 找出所有比current_frame早的frames
        to_process = []
        remaining = []
        
        for frame_data in pending_matched_frames:
            frame_idx, ocr_text, annotation_text = frame_data
            if frame_idx < current_frame:
                to_process.append(frame_data)
            else:
                remaining.append(frame_data)
        
        # 按frame_idx排序並處理
        to_process.sort(key=lambda x: x[0])
        for frame_idx, ocr_text, annotation_text in to_process:
            change_frames.append(frame_idx)
            if annotation_text is not None:
                self.annotations[frame_idx] = annotation_text
        
        # 更新buffer，只保留未處理的
        pending_matched_frames.clear()
        pending_matched_frames.extend(remaining)

    def _flush_all_pending_frames(self, pending_matched_frames: list, change_frames: list):
        """
        處理buffer中剩餘的所有matched_frames
        """
        # 按frame_idx排序並處理所有剩餘的frames
        pending_matched_frames.sort(key=lambda x: x[0]) #FIXME: 這可能不需要 如果資料源沒問題的話理論上都是照順序的
        for frame_idx, ocr_text, annotation_text in pending_matched_frames:
            change_frames.append(frame_idx)
            if annotation_text is not None:
                self.annotations[frame_idx] = annotation_text
        
        pending_matched_frames.clear()
    
    def _rebuild_change_cache(self, change_frames: list):
        """重建變化幀快取"""
        self.change_cache.clear()
        for i in range(self.total_frames):
            self.change_cache[i] = False
        for frame_idx in change_frames:
            if 0 <= frame_idx < self.total_frames:
                self.change_cache[frame_idx] = True

    def _on_closing(self):
        """應用程式關閉時的處理 - 不自動儲存當前分析結果"""
        print("關閉應用程式...")
        
        # 只儲存已確認的標註內容和變化幀資料，不包含當前分析快取
        if self.video_file_path and self.region_name:
            try:
                if self.annotations:
                    self._save_confirmed_annotations_only(self.region_name)
                    print("已確認的標註資料已儲存")
                    
            except Exception as e:
                print(f"關閉時儲存資料出錯: {e}")
        else:
            print("無需儲存資料 (未載入影片或無已確認的標註內容)")
               
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

    def _get_frame_cache_dir(self) -> Path | None:
        """Helper to get the directory for the full frame cache."""
        if not self.video_file_path:
            return None
        video_name = self.video_file_path.stem
        cache_dir = Path("data") / video_name / "frame_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    # -------------------------
    # Multi-track timeline UI
    # -------------------------
    def _create_stage_tag_area(self, parent_frame):
        try:
            container = tk.Frame(parent_frame)
            container.pack(fill="x", pady=(0, 3))

            # 控制列（ROI 勾選）
            ctrl = tk.Frame(container)
            ctrl.pack(fill="x", pady=(0, 2))
            self.stage_tag_controls_frame = ctrl

            tk.Label(ctrl, text="顯示階段標籤:").pack(side="left")
            chk_master = tk.Checkbutton(ctrl, text="啟用", variable=self.stage_tag_visible, command=self._refresh_stage_tag_ui)
            chk_master.pack(side="left", padx=(4, 8))

            # 多軌道時間軸容器
            timeline_frame = tk.Frame(container, relief="groove", bd=1)
            timeline_frame.pack(fill="x", pady=(2, 0))
            self.timeline_container = timeline_frame

            # 創建捲動區域以容納多個軌道
            self._create_scrollable_timeline_area(timeline_frame)
            
        except Exception as e:
            print(f"建立多軌道時間軸失敗: {e}")

    def _create_scrollable_timeline_area(self, parent):
        """創建可捲動的多軌道時間軸區域"""
        try:
            scroll_frame = tk.Frame(parent)
            scroll_frame.pack(fill="both", expand=True)

            v_scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical")
            
            main_canvas = tk.Canvas(scroll_frame, 
                                  height=120,
                                  yscrollcommand=v_scrollbar.set,
                                  highlightthickness=0)
            
            v_scrollbar.config(command=main_canvas.yview)
            
            tracks_frame = tk.Frame(main_canvas)
            tracks_frame_window = main_canvas.create_window((0, 0), window=tracks_frame, anchor="nw")
            
            def _configure_scroll_region(event):
                main_canvas.configure(scrollregion=main_canvas.bbox("all"))
            tracks_frame.bind("<Configure>", _configure_scroll_region)

            def _configure_frame_width(event):
                canvas_width = event.width
                main_canvas.itemconfig(tracks_frame_window, width=canvas_width)
            main_canvas.bind("<Configure>", _configure_frame_width)

            main_canvas.pack(side="left", fill="both", expand=True)
            v_scrollbar.pack(side="right", fill="y")
            
            self.timeline_main_canvas = main_canvas
            self.timeline_tracks_frame = tracks_frame
            self.timeline_v_scrollbar = v_scrollbar
            
            def _on_mousewheel(event):
                if self.timeline_tracks:
                    main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            main_canvas.bind("<MouseWheel>", _on_mousewheel)
            
        except Exception as e:
            print(f"創建可捲動時間軸區域失敗: {e}")

    def _create_timeline_track(self, region_name: str) -> dict:
        """為指定ROI區域創建一個獨立的時間軸軌道"""
        try:
            if not hasattr(self, 'timeline_tracks_frame'):
                return {}

            timeline_canvas = tk.Canvas(self.timeline_tracks_frame,
                                      height=self.track_height,
                                      bg="#fafafa",
                                      highlightthickness=1, # Add border for visual separation
                                      highlightbackground="#cccccc")
            
            timeline_canvas.pack(fill="x", expand=True, pady=(0, self.track_spacing), padx=0)

            timeline_canvas.bind("<Configure>", 
                                 lambda e, rn=region_name: self._on_track_canvas_resize(e, rn))
            timeline_canvas.bind("<Motion>", lambda e: self._on_track_motion(e, region_name))
            timeline_canvas.bind("<Leave>", lambda e: self._hide_stage_tooltip())
            timeline_canvas.bind("<Button-1>", lambda e: self._on_track_click(e, region_name))

            track_data = {
                'canvas': timeline_canvas,
                'meta': {},
                'region_name': region_name,
                'segments': [] # Initialize segments list
            }

            self.timeline_tracks[region_name] = track_data
            return track_data

        except Exception as e:
            print(f"創建軌道 {region_name} 失敗: {e}")
            return {}

    def _clear_all_tracks(self):
        """清除所有軌道"""
        try:
            for track_data in self.timeline_tracks.values():
                if 'canvas' in track_data and track_data['canvas']:
                    track_data['canvas'].destroy()
            
            self.timeline_tracks.clear()
            
        except Exception as e:
            print(f"清除軌道失敗: {e}")

    def _build_timeline_labels(self, regions: list[str]):
        # 清空舊標籤
        if not self.timeline_labels_frame:
            print("錯誤: timeline_labels_frame 未初始化。")
            return
            
        for widget in self.timeline_labels_frame.winfo_children():
            widget.destroy()

        print(f"--- 正在建立時間軸標籤: {regions} ---")
        try:
            track_height_with_spacing = self.track_height + self.track_spacing
            
            for region in regions:
                print(f"    - 建立 '{region}' 標籤...")
                label_container = tk.Frame(self.timeline_labels_frame, height=track_height_with_spacing)
                label_container.pack(fill="x", expand=True)
                label_container.pack_propagate(False)

                region_color = self.roi_color_map.get(region, "#666666")
                name_label = tk.Label(label_container, text=region, 
                                    bg="#f0f0f0", fg=region_color, 
                                    font=("Arial", 9, "bold"),
                                    anchor="w", padx=5, relief="solid", bd=1)
                name_label.pack(fill="both", expand=True, pady=(0, self.track_spacing))
            print("--- 時間軸標籤建立完畢 ---")

        except Exception as e:
            print(f"建立時間軸標籤失敗: {e}")
            import traceback
            traceback.print_exc()

    def _get_slider_left_padding(self) -> int:
        """獲取主slider的左側padding以對齊軌道"""
        try:
            # 由於slider使用 pack(fill="x", expand=True)，通常沒有額外的左側padding
            # 但我們可以檢查slider的實際位置
            if hasattr(self, 'slider') and self.slider:
                return 0  # 通常slider會填滿整個寬度
            return 0
        except Exception as e:
            print(f"獲取slider左側padding失敗: {e}")
            return 0

    def _get_slider_right_padding(self) -> int:
        """獲取主slider的右側padding"""
        try:
            return 0  # slider通常填滿整個寬度，無右側padding
        except Exception as e:
            print(f"獲取slider右側padding失敗: {e}")
            return 0

    def _sync_track_positions_with_slider(self):
        """同步所有軌道的位置與主slider對齊"""
        try:
            if not hasattr(self, 'slider') or not self.slider:
                return
                
            # 強制更新slider以獲取正確尺寸
            self.slider.update_idletasks()
            
            # 為每個軌道重新計算位置
            for region_name, track_data in self.timeline_tracks.items():
                if 'canvas' in track_data:
                    canvas = track_data['canvas']
                    # 重新渲染以確保與slider對齊
                    self._render_track_tags(region_name, track_data)
                    
        except Exception as e:
            print(f"同步軌道位置失敗: {e}")

    def _force_timeline_alignment(self):
        """強制重新對齊所有軌道與主slider"""
        try:
            # 延遲執行以確保所有UI組件已完全載入
            def do_alignment():
                if hasattr(self, 'slider') and self.slider and self.timeline_tracks:
                    print("強制重新對齊軌道與主slider...")
                    self.slider.update_idletasks()
                    
                    for region_name, track_data in self.timeline_tracks.items():
                        if 'canvas' in track_data:
                            track_data['canvas'].update_idletasks()
                            self._render_track_tags(region_name, track_data)
                    
                    # 更新位置指示器
                    self._update_track_position_indicators()
                    print("軌道對齊完成")
            
            # 延遲200ms執行，確保UI已穩定
            if hasattr(self, 'master'):
                self.master.after(200, do_alignment)
                
        except Exception as e:
            print(f"強制對齊失敗: {e}")

    def _update_timeline_scroll_region(self):
        """更新時間軸捲動區域"""
        try:
            if hasattr(self, 'timeline_main_canvas') and hasattr(self, 'timeline_tracks_frame'):
                self.timeline_tracks_frame.update_idletasks()
                self.timeline_main_canvas.configure(scrollregion=self.timeline_main_canvas.bbox("all"))
        except Exception as e:
            print(f"更新捲動區域失敗: {e}")

    def _generate_pattern_color(self, base_color: str, pattern_id: int, avg_rmse: float, region_name: str) -> str:
        """根據pattern ID和RMSE值生成漸變顏色"""
        try:
            if region_name == "PEDAL":
                # PEDAL區域使用預定義的高對比度顏色
                pedal_colors = [
                    "#FF4444",  # 紅色
                    "#44FF44",  # 綠色
                    "#4444FF",  # 藍色
                    "#FFAA00",  # 橙色
                    "#AA44FF",  # 紫色
                    "#44FFFF",  # 青色
                    "#FF44AA",  # 粉紅色
                    "#AAFF44",  # 黃綠色
                    "#FF8844",  # 橘紅色
                    "#4488FF",  # 天藍色
                ]
                
                base_color = pedal_colors[pattern_id % len(pedal_colors)]
                
                # 根據RMSE調整亮度（RMSE越高越暗）
                r, g, b = ImageColor.getrgb(base_color)
                brightness_factor = max(0.4, 1.0 - min(avg_rmse / 100.0, 0.6))
                
                r = int(r * brightness_factor)
                g = int(g * brightness_factor)
                b = int(b * brightness_factor)
                
                return f"#{r:02x}{g:02x}{b:02x}"
            else:
                # 其他區域使用原有邏輯
                r, g, b = ImageColor.getrgb(base_color)
                h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

                params = self.color_variation_params.get(region_name, {'h_shift': 0.02, 's_factor': 0.9, 'v_factor': 0.9})
                
                # Use pattern_id for a cyclical hue shift
                hue_offset = (pattern_id % 10) * params['h_shift']
                new_h = (h + hue_offset) % 1.0

                # Use RMSE for brightness variation (higher RMSE -> darker)
                brightness_factor = 1.0 - min(avg_rmse / 150.0, 0.5) # Clamp max darkening
                new_v = v * brightness_factor * params['v_factor']
                
                # Ensure saturation and value are within bounds
                new_s = max(0.2, s * params['s_factor'])
                new_v = max(0.1, new_v)

                r_new, g_new, b_new = colorsys.hsv_to_rgb(new_h, new_s, new_v)
                return f"#{int(r_new*255):02x}{int(g_new*255):02x}{int(b_new*255):02x}"
            
        except Exception as e:
            print(f"生成pattern顏色失敗: {e}")
            return base_color

    def _is_dark_color(self, hex_color: str) -> bool:
        """判斷顏色是否為深色"""
        try:
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # 使用亮度公式
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            return brightness < 128
            
        except Exception:
            return False

    def _load_stage_analysis(self):
        try:
            analysis_file_path = None
            if not self.video_file_path:
                print("錯誤：尚未載入影片，無法尋找 stage analysis 檔案。")
                return

            # 1. 優先嘗試自動尋找檔案
            video_name = self.video_file_path.stem
            expected_path = Path("data") / video_name / "stage_analysis.json"

            if expected_path.exists():
                print(f"自動找到 stage analysis 檔案: {expected_path}")
                analysis_file_path = expected_path
            else:
                # 2. 自動尋找失敗，才彈出視窗讓使用者手動選擇
                print(f"在預設路徑 {expected_path} 未找到檔案，請手動選擇。")
                manual_path = filedialog.askopenfilename(
                    title="自動尋找失敗，請手動選擇 stage_analysis.json 檔案",
                    filetypes=[("JSON files", "*.json")],
                    initialdir=self.video_file_path.parent if self.video_file_path else Path.cwd()
                )
                if manual_path:
                    analysis_file_path = Path(manual_path)
                else:
                    print("未選擇檔案，操作取消。")
                    return

            if not analysis_file_path or not analysis_file_path.exists():
                print(f"錯誤：分析檔案不存在或未選擇。")
                return
                
            self.stage_analysis = json.loads(analysis_file_path.read_text(encoding="utf-8"))
            print("階段分析檔載入成功。")
            
            # 調試信息：顯示載入的數據結構
            if "regions" in self.stage_analysis:
                regions = list(self.stage_analysis["regions"].keys())
                print(f"載入的區域: {regions}")
                for region in regions:
                    segments_count = len(self.stage_analysis["regions"][region])
                    print(f"  {region}: {segments_count} 個段落")
            else:
                print("警告：stage_analysis中沒有找到regions數據")
            
            # --- UI 更新 ---
            if self.timeline_labels_frame and self.timeline_tracks_frame:
                self._refresh_stage_tag_ui()
            else:
                print("警告: 時間軸UI元件尚未初始化，無法刷新。")

        except json.JSONDecodeError as e:
            print(f"載入階段分析檔失敗: JSON 格式錯誤 - {e}")
        except Exception as e:
            print(f"載入階段分析檔時發生未預期錯誤: {e}")
            import traceback
            traceback.print_exc()

    def _build_stage_tag_roi_checks(self, regions: list[str]):
        # 清空舊的控制項
        for child in self.stage_tag_controls_frame.winfo_children()[2:]:  # 跳過前兩個（標籤 + 啟用）
            child.destroy()
        # 建立 ROI 勾選
        for region in regions:
            var = self.stage_tag_roi_vars.get(region) or tk.BooleanVar(value=True)
            self.stage_tag_roi_vars[region] = var
            color = self.roi_color_map.get(region, "#444")
            cb = tk.Checkbutton(self.stage_tag_controls_frame, text=region, variable=var, fg=color, command=self._refresh_stage_tag_ui)
            cb.pack(side="left", padx=4)

    def _refresh_stage_tag_ui(self):
        self._clear_all_tracks()
        
        if not self.stage_analysis or "regions" not in self.stage_analysis:
            print("Stage analysis 資料未載入或格式不正確，無法刷新時間軸。")
            return

        print("\n--- 開始刷新多軌道時間軸 UI ---")
        regions_data = self.stage_analysis.get("regions", {})
        
        # 固定只顯示 'PEDAL' 和 'STAGE'
        regions_to_display = ['PEDAL', 'STAGE']
        print(f"計畫顯示的軌道: {regions_to_display}")

        self._build_timeline_labels(regions_to_display)

        # 即使某些 ROI 被取消選取，我們仍然為它們創建軌道以保持垂直對齊
        for region_name in regions_to_display:
            track_data = self._create_timeline_track(region_name)
            
            # 渲染標籤
            if region_name in regions_data:
                region_content = regions_data[region_name]
                segments = []
                # 兼容處理兩種可能的資料結構: dict (新) 或 list (舊)
                if isinstance(region_content, dict):
                    segments = region_content.get("patterns", [])
                elif isinstance(region_content, list):
                    segments = region_content
                
                track_data['segments'] = segments  # 儲存 segment 資料以供後續渲染
                print(f"為 '{region_name}' 找到 {len(segments)} 個區段。")
            else:
                track_data['segments'] = []
                print(f"警告: 在 stage_analysis 中未找到 '{region_name}' 的資料。")

        # 短暫延遲後對齊標籤和軌道的高度
        self.master.after(50, self._align_labels_to_tracks)
        print("--- 多軌道時間軸 UI 刷新結束 ---")

    def _align_labels_to_tracks(self):
        try:
            if hasattr(self, 'timeline_labels_frame') and hasattr(self, 'timeline_tracks_frame'):
                self.timeline_tracks_frame.update_idletasks()
                tracks_height = self.timeline_tracks_frame.winfo_height()
                self.timeline_labels_frame.configure(height=tracks_height)
        except Exception as e:
            print(f"Failed to align label heights: {e}")

    def _on_track_canvas_resize(self, event, region_name):
        """Redraw the track when the canvas is resized to ensure correct scaling."""
        try:
            track_data = self.timeline_tracks.get(region_name)
            if not track_data or 'segments' not in track_data:
                return
            
            # This is the main entry point for rendering.
            self._render_track_tags(region_name, track_data, track_data['segments'])
            self._draw_position_indicator(track_data['canvas'], self.current_frame_idx)
        except Exception as e:
            print(f"Error during track canvas resize for {region_name}: {e}")

    def _render_track_tags(self, region_name: str, track_data: dict, segments: list):
        """Renders the colored segments on a specific track canvas."""
        try:
            if not segments:
                # Still draw the background even if there are no segments
                canvas = track_data['canvas']
                canvas.update_idletasks()
                canvas_width = canvas.winfo_width()
                canvas_height = canvas.winfo_height()
                if canvas_width > 1:
                    canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#f0f0f0", outline="")
                return

            canvas = track_data['canvas']
            meta_dict = track_data['meta']
            
            canvas.update_idletasks()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            canvas.delete("all")
            canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#f0f0f0", outline="")
            
            if canvas_width <= 1 or not segments:
                return

            total_frames = max(1, self.total_frames - 1) if self.total_frames > 0 else 1
            base_color = self.roi_color_map.get(region_name, "#666666")
            
            if self.show_alignment_grid:
                self._draw_alignment_grid(canvas, canvas_width, canvas_height)
            
            self._draw_track_time_marks(canvas, canvas_width, canvas_height, total_frames)
            
            min_dx = 4
            x_positions = []
            prev_x_end = None
            
            for i, seg in enumerate(segments):
                s = int(seg.get("start_frame", 0))
                e = int(seg.get("end_frame", s))
                pid = int(seg.get("pattern", -1))
                rmse = float(seg.get("avg_rmse", 0.0))
                
                # 計算精確的像素位置，確保end_frame被包含
                x_start = (s / total_frames) * canvas_width
                x_end = ((e + 1) / total_frames) * canvas_width
                
                # 檢查是否與前一個段落連續
                is_continuous = False
                if i > 0:
                    prev_seg = segments[i-1]
                    prev_end = int(prev_seg.get("end_frame", 0))
                    if prev_end + 1 == s:  # 連續段落
                        is_continuous = True
                        x_start = prev_x_end  # 精確銜接
                
                # 對於連續段落，不添加間隙；對於非連續段落，添加小間隙
                if not is_continuous:
                    x_start += 0.5
                    x_end -= 0.5
                else:
                    x_end -= 0.5  # 只在右側留小間隙

                if x_start >= x_end:
                    continue
                
                # 確保最小可見寬度
                tag_width = x_end - x_start
                if tag_width < 2:
                    center = (x_start + x_end) / 2
                    x_start = center - 1
                    x_end = center + 1

                x_center = (x_start + x_end) / 2

                # 只對非連續段落應用最小距離檢查
                if not is_continuous and any(abs(x_center - xi) < min_dx for xi in x_positions):
                    continue
                    
                x_positions.append(x_center)
                prev_x_end = x_end
                
                pattern_color = self._generate_pattern_color(base_color, pid, rmse, region_name)
                
                tag_height = canvas_height - 8
                y_top = 4
                y_bottom = y_top + tag_height
                
                item_id = canvas.create_rectangle(
                    x_start, y_top, x_end, y_bottom,
                    fill=pattern_color, 
                    outline="",
                    width=0
                )
                
                meta_dict[item_id] = {
                    "region": region_name,
                    "pattern": pid,
                    "start": s,
                    "end": e,
                    "avg_rmse": rmse,
                    "color": pattern_color
                }
                
                if (x_end - x_start) >= 12:
                    text_x = (x_start + x_end) / 2
                    text_y = (y_top + y_bottom) / 2
                    text_color = "#ffffff" if self._is_dark_color(pattern_color) else "#000000"
                    
                    canvas.create_text(
                        text_x, text_y,
                        text=str(pid),
                        fill=text_color,
                        font=("Arial", 7, "bold"),
                        anchor="center"
                    )
        except Exception as e:
            print(f"渲染軌道 {region_name} 標籤失敗: {e}")

    def _draw_alignment_grid(self, canvas, width: int, height: int):
        """繪製對齊網格（用於測試和調試）"""
        try:
            # 繪製邊框
            canvas.create_rectangle(0, 0, width-1, height-1, outline="blue", width=1)
            
            # 繪製十字線標示中心
            canvas.create_line(width//2, 0, width//2, height, fill="lightblue", width=1)
            canvas.create_line(0, height//2, width, height//2, fill="lightblue", width=1)
            
            # 在四個角落標示尺寸
            canvas.create_text(5, 5, text=f"0,0", fill="blue", font=("Arial", 6), anchor="nw")
            canvas.create_text(width-5, 5, text=f"{width},0", fill="blue", font=("Arial", 6), anchor="ne")
            canvas.create_text(5, height-5, text=f"0,{height}", fill="blue", font=("Arial", 6), anchor="sw")
            canvas.create_text(width-5, height-5, text=f"{width},{height}", fill="blue", font=("Arial", 6), anchor="se")
            
        except Exception as e:
            print(f"繪製對齊網格失敗: {e}")

    def _draw_track_time_marks(self, canvas, width: int, height: int, total_frames: int):
        """在軌道上繪製時間刻度標記"""
        try:
            if total_frames <= 0 or width <= 0:
                return
                
            # 繪製刻度線（與主slider對齊）
            num_marks = min(10, total_frames)  # 最多10個刻度
            if num_marks <= 1:
                return
                
            for i in range(num_marks + 1):
                frame_pos = (i * total_frames) // num_marks
                x = int((frame_pos / total_frames) * (width - 1)) if total_frames > 0 else 0
                
                # 繪製小刻度線
                canvas.create_line(x, height - 2, x, height, fill="#cccccc", width=1)
                
        except Exception as e:
            print(f"繪製時間刻度失敗: {e}")

    def _update_track_position_indicators(self):
        """在所有軌道上更新當前frame位置指示器"""
        try:
            if not self.timeline_tracks or self.total_frames <= 0:
                return
                
            for region_name, track_data in self.timeline_tracks.items():
                canvas = track_data['canvas']
                self._draw_position_indicator(canvas, self.current_frame_idx)
                
        except Exception as e:
            print(f"更新軌道位置指示器失敗: {e}")

    def _draw_position_indicator(self, canvas, current_frame: int):
        """在軌道上繪製當前frame位置指示器，使用Canvas完整寬度"""
        try:
            canvas.delete("position_indicator")

            if self.total_frames <= 0:
                return

            canvas.update_idletasks()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            if canvas_width <= 1:
                return

            total_frames = max(1, self.total_frames - 1)
            indicator_x = int((current_frame / total_frames) * (canvas_width - 1)) if total_frames > 0 else 0
            indicator_x = max(0, min(indicator_x, canvas_width - 1))

            canvas.create_line(
                indicator_x, 2, indicator_x, canvas_height - 2,
                fill="#ff0000", width=3, tags="position_indicator"
            )

            triangle_size = 5
            canvas.create_polygon(
                indicator_x - triangle_size, 2,
                indicator_x + triangle_size, 2,
                indicator_x, 2 + triangle_size,
                fill="#ff0000", outline="", tags="position_indicator"
            )

            canvas.create_polygon(
                indicator_x - triangle_size, canvas_height - 2,
                indicator_x + triangle_size, canvas_height - 2,
                indicator_x, canvas_height - 2 - triangle_size,
                fill="#ff0000", outline="", tags="position_indicator"
            )

        except Exception as e:
            print(f"繪製位置指示器失敗: {e}")

    def _on_track_motion(self, event, region_name: str):
        """處理軌道滑鼠移動事件"""
        try:
            if region_name not in self.timeline_tracks:
                return
                
            track_data = self.timeline_tracks[region_name]
            canvas = track_data['canvas']
            meta_dict = track_data['meta']
            
            # 使用改進的碰撞檢測（3x3像素範圍）
            x, y = event.x, event.y
            overlapping_items = canvas.find_overlapping(x-1, y-1, x+1, y+1)
            
            meta = None
            for item_id in overlapping_items:
                meta = meta_dict.get(item_id)
                if meta:
                    break
                    
            if not meta:
                self._hide_stage_tooltip()
                return
                
            # 構建tooltip文字
            s, e = meta["start"], meta["end"]
            dur_frames = e - s + 1
            fps = self.fps if getattr(self, 'fps', 0) else 30.0
            dur_sec = dur_frames / fps
            
            text = (f"{meta['region']} Pattern #{meta['pattern']}\n"
                   f"幀範圍: {s} → {e} ({dur_frames} 幀)\n"
                   f"時長: {dur_sec:.2f} 秒\n"
                   f"RMSE: {meta['avg_rmse']:.2f}")
            
            self._show_stage_tooltip(event.x_root, event.y_root, text)
            
        except Exception as e:
            print(f"處理軌道 {region_name} 滑鼠移動事件失敗: {e}")

    def _on_track_click(self, event, region_name: str):
        """處理軌道點擊事件"""
        try:
            if region_name not in self.timeline_tracks:
                return
                
            track_data = self.timeline_tracks[region_name]
            canvas = track_data['canvas']
            meta_dict = track_data['meta']
            
            # 使用改進的碰撞檢測
            x, y = event.x, event.y
            overlapping_items = canvas.find_overlapping(x-1, y-1, x+1, y+1)
            
            for item_id in overlapping_items:
                meta = meta_dict.get(item_id)
                if meta:
                    frame_to_show = int(meta["start"])
                    self._show_frame(frame_to_show)
                    print(f"點擊軌道 {region_name} 標籤 #{meta['pattern']}，跳轉到幀 {frame_to_show}")
                    
                    # 延遲同步表格選擇，確保幀已經更新
                    self.master.after(50, lambda: self._sync_treeview_selection_to_frame(frame_to_show, force=True))
                    break
            
        except Exception as e:
            print(f"處理軌道 {region_name} 點擊事件失敗: {e}")

    # 保留舊的方法以維持兼容性（如果還有地方在使用）
    def _on_stage_tag_motion(self, event):
        """舊的單軌道滑鼠移動處理（保持兼容性）"""
        pass

    def _on_stage_tag_click(self, event):
        """舊的單軌道點擊處理（保持兼容性）"""
        pass

    # 簡易 tooltip（用 Toplevel）
    def _show_stage_tooltip(self, x_root: int, y_root: int, text: str):
        try:
            if hasattr(self, '_stage_tip') and self._stage_tip:
                self._stage_tip.destroy()
        except Exception:
            pass
        try:
            tip = tk.Toplevel(self)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x_root+12}+{y_root+12}")
            lbl = tk.Label(tip, text=text, bg="#333", fg="#fff", font=("Arial", 9), padx=6, pady=4, justify="left")
            lbl.pack()
            self._stage_tip = tip
        except Exception:
            pass

    def _hide_stage_tooltip(self):
        try:
            if hasattr(self, '_stage_tip') and self._stage_tip:
                self._stage_tip.destroy()
                self._stage_tip = None
        except Exception:
            pass



    def _on_mode_tab_changed(self, event=None):
        """處理模式標籤頁切換事件"""
        try:
            selected_tab = self.mode_notebook.nametowidget(self.mode_notebook.select())
            
            if selected_tab == self.ocr_mode_frame:
                self.surgery_stage_mode = False
                print("切換到 OCR 標註模式")
                self._update_status_bar("已切換到 OCR 標註模式")
            elif selected_tab == self.surgery_mode_frame:
                self.surgery_stage_mode = True
                print("切換到手術階段分析模式")
                self._update_status_bar("已切換到手術階段分析模式")
                
                # 載入手術階段相關配置
                self._load_surgery_stage_roi_config()
                self._update_surgery_stage_roi_ui()
                
                # 確保有選中的手術階段區域
                if not self.current_surgery_stage_region:
                    regions = list(self.surgery_stage_roi_dict.keys())
                    if regions:
                        self.current_surgery_stage_region = regions[0]
                        self.surgery_stage_combobox.set(regions[0])
                        print(f"自動選擇手術階段區域: {self.current_surgery_stage_region}")
                
                # 重新載入階段分析數據和時間軸標籤
                if self.video_file_path:
                    self._load_stage_analysis()
                    self._refresh_stage_tag_ui()
        
            # 更新模式相關的UI控制項顯示
            self._update_mode_specific_controls()
            
            # 更新相關UI和數據
            self._refresh_treeview()
            
            # 重新顯示當前幀以更新ROI顯示
            if hasattr(self, 'current_frame_idx'):
                self._show_frame(self.current_frame_idx)
                
        except Exception as e:
            print(f"模式切換錯誤: {e}")
            traceback.print_exc()

    def _update_mode_specific_controls(self):
        """根據當前模式顯示/隱藏相關的UI控制項"""
        if not hasattr(self, 'binarize_checkbox') or not hasattr(self, 'btn_save'):
            return
            
        if self.surgery_stage_mode:
            # 手術階段分析模式：隱藏OCR專用功能
            self.binarize_checkbox.pack_forget()
            self.btn_save.pack_forget()
            
            # 自動關閉二值化模式
            if self.binarize_mode_var.get():
                self.binarize_mode_var.set(False)
                print("自動關閉二值化模式（手術階段分析模式不需要）")
            
            print("隱藏OCR專用控制項：二值化顯示、儲存標註")
        else:
            # OCR標註模式：顯示OCR專用功能
            self.btn_save.pack(side="right", padx=(0, 10))
            self.binarize_checkbox.pack(side="right", padx=5)
            print("顯示OCR專用控制項：二值化顯示、儲存標註")

    def _on_surgery_stage_mode_toggle(self):
        """保留舊方法以維持兼容性（已被標籤頁切換取代）"""
        pass

    def _on_surgery_stage_region_select(self, event=None):
        """選擇手術階段區域"""
        new_region = self.surgery_stage_region_var.get()
        if not new_region:
            return
            
        self.current_surgery_stage_region = new_region
        
        # 如果在手術階段模式，重新配置表格欄位並重載數據
        if self.surgery_stage_mode:
            self._refresh_treeview()
        
        # 更新當前手術階段ROI座標
        if new_region in self.surgery_stage_roi_dict:
            coords = self.surgery_stage_roi_dict[new_region]
            print(f"切換到手術階段區域: {new_region}, ROI: {coords}")
        else:
            print(f"手術階段區域 {new_region} 尚未設定ROI")
        
        # 更新手術階段ROI座標顯示
        self._update_surgery_stage_roi_fields()
        
        # 如果在手術階段模式，重新載入表格以顯示新區域的斷點
        if self.surgery_stage_mode:
            self._refresh_treeview()
            
        # 重新顯示當前幀
        if hasattr(self, 'current_frame_idx'):
            self._show_frame(self.current_frame_idx)
            
        self._update_status_bar(f"已切換到手術階段區域: {new_region}")

    def _on_add_surgery_stage_region(self):
        """添加新的手術階段區域"""
        new_region = simpledialog.askstring("新增手術階段區域", "請輸入手術階段區域名稱:")
        if not new_region:
            return
            
        if new_region in self.surgery_stage_roi_dict:
            messagebox.showwarning("重複名稱", f"手術階段區域 '{new_region}' 已存在")
            return
            
        # 添加新區域，使用預設ROI
        self.surgery_stage_roi_dict[new_region] = [100, 100, 300, 200]
        self.current_surgery_stage_region = new_region
        
        # 更新UI
        self._update_surgery_stage_roi_ui()
        self._update_status_bar(f"已新增手術階段區域: {new_region}")

    def _save_surgery_stage_roi_config(self):
        """儲存手術階段ROI設定到檔案"""
        roi_file = get_surgery_stage_roi_config_path()
        if not roi_file:
            return
        
        # 確保目錄存在
        roi_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(roi_file, "w", encoding="utf-8") as f:
                json.dump(self.surgery_stage_roi_dict, f, indent=2, ensure_ascii=False)
            print(f"手術階段ROI設定已儲存至 {roi_file}")
            self._update_status_bar(f"手術階段ROI設定已儲存")
        except Exception as e:
            print(f"儲存手術階段ROI設定失敗: {e}")
            self._update_status_bar(f"儲存手術階段ROI設定失敗: {e}")

    def _load_surgery_stage_roi_config(self):
        """載入手術階段ROI設定"""
        roi_file = get_surgery_stage_roi_config_path()
        
        try:
            if roi_file.exists():
                with open(roi_file, "r", encoding="utf-8") as f:
                    self.surgery_stage_roi_dict = json.load(f)
                print(f"已載入手術階段ROI設定: {self.surgery_stage_roi_dict}")
            else:
                print(f"手術階段ROI設定檔不存在，將建立預設配置")
                # 如果檔案不存在，建立一個預設配置
                self.surgery_stage_roi_dict = {
                    "手術開始": [100, 100, 300, 200],
                    "切口階段": [400, 100, 600, 200],
                    "縫合階段": [700, 100, 900, 200]
                }
        except Exception as e:
            print(f"載入手術階段ROI設定失敗: {e}")
            # 載入失敗時使用預設配置
            self.surgery_stage_roi_dict = {
                "手術開始": [100, 100, 300, 200],
                "切口階段": [400, 100, 600, 200],
                "縫合階段": [700, 100, 900, 200]
            }

        # 更新UI（如果已建立）
        if hasattr(self, 'surgery_stage_combobox'):
            self._update_surgery_stage_roi_ui()
        
        print(f"最終手術階段ROI字典: {self.surgery_stage_roi_dict}")

    def _update_surgery_stage_roi_ui(self):
        """更新手術階段ROI UI"""
        if not hasattr(self, 'surgery_stage_combobox'):
            return
            
        # 更新下拉選單選項
        regions = list(self.surgery_stage_roi_dict.keys())
        self.surgery_stage_combobox['values'] = regions
        
        # 設定當前選中項目
        if self.current_surgery_stage_region in regions:
            self.surgery_stage_combobox.set(self.current_surgery_stage_region)
        elif regions:
            self.current_surgery_stage_region = regions[0]
            self.surgery_stage_combobox.set(regions[0])
        
        # 更新手術階段ROI座標顯示
        self._update_surgery_stage_roi_fields()

    def _apply_surgery_stage_roi_from_fields(self):
        """把手術階段ROI Spinbox 數值寫回，並立即生效/儲存"""
        if not self.current_surgery_stage_region:
            messagebox.showwarning("警告", "請先選擇手術階段區域")
            return
            
        x1 = self.surgery_stage_x1_var.get()
        y1 = self.surgery_stage_y1_var.get()
        x2 = self.surgery_stage_x2_var.get()
        y2 = self.surgery_stage_y2_var.get()
        
        if x1 >= x2 or y1 >= y2:
            messagebox.showwarning("座標錯誤", "x1,y1 必須小於 x2,y2")
            return
            
        # 更新手術階段ROI座標
        self.surgery_stage_roi_dict[self.current_surgery_stage_region] = [x1, y1, x2, y2]
        
        # 詢問是否儲存配置
        result = messagebox.askyesno("儲存設定", f"手術階段ROI座標已更新，是否儲存到配置檔案？")
        if result:
            self._save_surgery_stage_roi_config()
        
        # 重新顯示當前幀
        if hasattr(self, 'current_frame_idx'):
            self._show_frame(self.current_frame_idx)
            
        status_msg = f"{self.current_surgery_stage_region} 手術階段ROI 已套用: ({x1},{y1},{x2},{y2})"
        if result:
            status_msg += " (已儲存)"
        else:
            status_msg += " (未儲存)"
        self._update_status_bar(status_msg)

    def _update_surgery_stage_roi_fields(self):
        """將當前選中的手術階段ROI座標反映到4個Spinbox"""
        if not self.current_surgery_stage_region or self.current_surgery_stage_region not in self.surgery_stage_roi_dict:
            # 如果沒有選中區域或區域不存在，清空座標
            self.surgery_stage_x1_var.set(0)
            self.surgery_stage_y1_var.set(0)
            self.surgery_stage_x2_var.set(0)
            self.surgery_stage_y2_var.set(0)
            return
            
        coords = self.surgery_stage_roi_dict[self.current_surgery_stage_region]
        if coords and len(coords) >= 4:
            x1, y1, x2, y2 = map(int, coords)
            self.surgery_stage_x1_var.set(x1)
            self.surgery_stage_y1_var.set(y1)
            self.surgery_stage_x2_var.set(x2)
            self.surgery_stage_y2_var.set(y2)

    def _save_roi_to_cache(self):
        """將當前選中區域的ROI圖像存入快取"""
        if not self.current_surgery_stage_region:
            messagebox.showwarning("警告", "請先選擇手術階段區域")
            return
            
        if not hasattr(self, 'current_frame_idx') or not hasattr(self, 'video_file_path') or not self.video_file_path:
            messagebox.showwarning("警告", "請先載入影片")
            return
            
        try:
            # 獲取當前幀的完整圖像
            full_frame_image = self._get_full_frame_image_with_cache(self.current_frame_idx)
            if not full_frame_image:
                messagebox.showerror("錯誤", "無法獲取當前幀圖像")
                return
                
            # 獲取當前區域的ROI座標
            region_name = self.current_surgery_stage_region
            if region_name not in self.surgery_stage_roi_dict:
                messagebox.showwarning("警告", f"區域 {region_name} 尚未設定ROI座標")
                return
                
            coords = self.surgery_stage_roi_dict[region_name]
            x1, y1, x2, y2 = coords
            
            # 裁剪ROI圖像
            roi_image = full_frame_image.crop((x1, y1, x2, y2))
            
            # 根據區域名稱決定是否進行二值化處理
            if region_name == "PEDAL":
                # PEDAL區域使用原圖
                processed_image = roi_image
            else:
                # 其他區域進行二值化處理
                processed_image = self._apply_binarization(roi_image, "otsu")
                if processed_image is None:
                    messagebox.showerror("錯誤", "二值化處理失敗")
                    return
            
            # 轉換為numpy數組用於存儲
            if region_name == "PEDAL":
                cache_array = np.array(processed_image)
            else:
                # 二值化圖像轉換為PIL再轉numpy
                if isinstance(processed_image, np.ndarray):
                    cache_array = processed_image
                else:
                    cache_array = np.array(processed_image)
            
            # 創建快取目錄
            cache_dir = Path("data/roi_img_caches") / region_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # 檢查是否有重複的快取（門檻值10）
            duplicate_info = self._check_duplicate_cache(cache_dir, cache_array, threshold=10.0)
            if duplicate_info:
                duplicate_number, duplicate_rmse = duplicate_info
                result = messagebox.askyesno(
                    "發現相似快取", 
                    f"發現相似的快取檔案:\n"
                    f"編號: #{duplicate_number}\n"
                    f"相似度: RMSE = {duplicate_rmse:.2f} (< 10.0)\n\n"
                    f"是否仍要儲存新的快取？"
                )
                if not result:
                    self._update_status_bar(f"取消儲存快取，已存在相似快取 #{duplicate_number}")
                    return
            
            # 生成快取編號（找到下一個可用編號）
            cache_number = self._get_next_cache_number(cache_dir)
            
            # 保存為NPY格式（用於高效計算）
            npy_filename = f"{cache_number}.npy"
            npy_path = cache_dir / npy_filename
            np.save(npy_path, cache_array)
            
            # 保存為PNG格式（用於直觀查看）
            png_filename = f"{cache_number}.png"
            png_path = cache_dir / png_filename
            
            # 根據數據類型保存PNG
            if region_name == "PEDAL":
                # PEDAL區域保存原始RGB圖像
                processed_image.save(png_path)
            else:
                # 其他區域保存二值化圖像
                if isinstance(processed_image, np.ndarray):
                    # numpy數組轉PIL圖像
                    if len(processed_image.shape) == 2:  # 灰度圖
                        pil_image = Image.fromarray(processed_image, mode='L')
                    else:  # RGB圖
                        pil_image = Image.fromarray(processed_image)
                    pil_image.save(png_path)
                else:
                    processed_image.save(png_path)
            
            # 更新內存快取
            self.roi_image_cache[region_name] = cache_array.copy()
            
            # 更新狀態欄
            self._update_status_bar(f"已將 {region_name} ROI圖像存入快取: #{cache_number} (npy+png)")
            
            messagebox.showinfo("成功", f"ROI圖像已存入快取\n區域: {region_name}\n編號: {cache_number}\n格式: {npy_filename} + {png_filename}")
            
        except Exception as e:
            print(f"存入快取時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("錯誤", f"存入快取失敗: {e}")

    def _check_duplicate_cache(self, cache_dir: Path, new_cache_array: np.ndarray, threshold: float = 10.0) -> Optional[tuple[int, float]]:
        """檢查是否有重複的快取，返回 (編號, RMSE) 或 None"""
        try:
            # 找到所有現有的npy檔案
            existing_files = list(cache_dir.glob("*.npy"))
            if not existing_files:
                return None
                
            min_rmse = float('inf')
            duplicate_number = None
            
            for npy_file in existing_files:
                try:
                    # 載入現有快取
                    existing_array = np.load(npy_file)
                    
                    # 計算RMSE差異
                    rmse = self._calculate_roi_diff_rmse(existing_array, new_cache_array)
                    
                    # 如果RMSE小於門檻值，認為是重複
                    if rmse < threshold and rmse < min_rmse:
                        min_rmse = rmse
                        try:
                            duplicate_number = int(npy_file.stem)
                        except ValueError:
                            continue
                            
                except Exception as e:
                    print(f"檢查快取檔案 {npy_file} 時發生錯誤: {e}")
                    continue
                    
            if duplicate_number is not None and min_rmse < threshold:
                return (duplicate_number, min_rmse)
            else:
                return None
                
        except Exception as e:
            print(f"檢查重複快取時發生錯誤: {e}")
            return None

    def _get_next_cache_number(self, cache_dir: Path) -> int:
        """獲取下一個可用的快取編號"""
        try:
            # 找到所有現有的npy檔案
            existing_files = list(cache_dir.glob("*.npy"))
            if not existing_files:
                return 1
                
            # 提取編號並找到最大值
            numbers = []
            for file_path in existing_files:
                try:
                    # 檔案名格式: "數字.npy"
                    number = int(file_path.stem)
                    numbers.append(number)
                except ValueError:
                    # 跳過非數字檔名
                    continue
                    
            if not numbers:
                return 1
                
            return max(numbers) + 1
            
        except Exception as e:
            print(f"獲取快取編號時發生錯誤: {e}")
            return 1

    def _load_roi_cache(self, region_name: str) -> Optional[np.ndarray]:
        """載入指定區域的最新快取圖像"""
        try:
            cache_dir = Path("data/roi_img_caches") / region_name
            if not cache_dir.exists():
                return None
                
            # 找到所有NPY快取檔案
            npy_files = list(cache_dir.glob("*.npy"))
            if not npy_files:
                return None
                
            # 按編號排序，取最新的（最大編號）
            def get_file_number(file_path):
                try:
                    return int(file_path.stem)
                except ValueError:
                    return 0
                    
            latest_npy = max(npy_files, key=get_file_number)
            cache_number = get_file_number(latest_npy)
            
            # 載入快取數組
            cache_array = np.load(latest_npy)
            
            # 更新內存快取
            self.roi_image_cache[region_name] = cache_array.copy()
            
            print(f"已載入 {region_name} 區域快取: #{cache_number} ({latest_npy.name})")
            return cache_array
            
        except Exception as e:
            print(f"載入 {region_name} 區域快取時發生錯誤: {e}")
            return None

    def _calculate_roi_diff_rmse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """計算兩張ROI圖像的RMSE差異"""
        try:
            # 確保兩張圖像尺寸一致
            if img1.shape != img2.shape:
                return float('inf')  # 尺寸不一致，返回無限大差異
            
            # 轉換為float32進行計算
            img1_f = img1.astype(np.float32)
            img2_f = img2.astype(np.float32)
            
            # 計算均方根誤差 (RMSE)
            mse = np.mean((img1_f - img2_f) ** 2)
            rmse = np.sqrt(mse)
            
            return float(rmse)
            
        except Exception as e:
            print(f"計算RMSE時發生錯誤: {e}")
            return float('inf')

    def _check_cache_hit(self, region_name: str, current_roi_image: Image.Image) -> tuple[bool, float]:
        """檢查當前ROI圖像是否與快取匹配"""
        try:
            # 先嘗試從內存快取獲取
            if region_name not in self.roi_image_cache:
                # 如果內存中沒有，嘗試從檔案載入
                cache_array = self._load_roi_cache(region_name)
                if cache_array is None:
                    return False, float('inf')
            else:
                cache_array = self.roi_image_cache[region_name]
            
            # 處理當前ROI圖像
            if region_name == "PEDAL":
                # PEDAL區域使用原圖
                current_array = np.array(current_roi_image)
            else:
                # 其他區域進行二值化處理
                processed_image = self._apply_binarization(current_roi_image, "otsu")
                if processed_image is None:
                    return False, float('inf')
                
                if isinstance(processed_image, np.ndarray):
                    current_array = processed_image
                else:
                    current_array = np.array(processed_image)
            
            # 計算RMSE差異
            rmse = self._calculate_roi_diff_rmse(cache_array, current_array)
            
            # 判斷是否為cache hit
            is_hit = rmse < self.cache_hit_threshold
            
            return is_hit, rmse
            
        except Exception as e:
            print(f"檢查cache hit時發生錯誤: {e}")
            return False, float('inf')

    def _perform_cache_comparison(self, full_frame_image: Image.Image):
        """在切換frame時執行快取比對"""
        try:
            region_name = self.current_surgery_stage_region
            if region_name not in self.surgery_stage_roi_dict:
                return
                
            # 獲取ROI座標並裁剪圖像
            coords = self.surgery_stage_roi_dict[region_name]
            x1, y1, x2, y2 = coords
            roi_image = full_frame_image.crop((x1, y1, x2, y2))
            
            # 檢查快取匹配
            is_hit, rmse = self._check_cache_hit(region_name, roi_image)
            
            # 更新狀態欄顯示比對結果
            if is_hit:
                status_msg = f"🎯 Cache HIT! {region_name} RMSE: {rmse:.2f} (< {self.cache_hit_threshold})"
                print(f"✅ {status_msg}")
            else:
                if rmse == float('inf'):
                    status_msg = f"❌ No cache for {region_name}"
                else:
                    status_msg = f"❌ Cache MISS! {region_name} RMSE: {rmse:.2f} (>= {self.cache_hit_threshold})"
                print(f"⚠️ {status_msg}")
            
            self._update_status_bar(status_msg)
            
        except Exception as e:
            print(f"執行快取比對時發生錯誤: {e}")
            import traceback
            traceback.print_exc()

    def _show_cache_info(self):
        """顯示當前區域的快取資訊"""
        if not self.current_surgery_stage_region:
            messagebox.showwarning("警告", "請先選擇手術階段區域")
            return
            
        try:
            region_name = self.current_surgery_stage_region
            cache_dir = Path("data/roi_img_caches") / region_name
            
            if not cache_dir.exists():
                messagebox.showinfo("快取資訊", f"區域 '{region_name}' 尚無快取檔案")
                return
                
            # 找到所有快取檔案
            npy_files = list(cache_dir.glob("*.npy"))
            png_files = list(cache_dir.glob("*.png"))
            
            if not npy_files and not png_files:
                messagebox.showinfo("快取資訊", f"區域 '{region_name}' 尚無快取檔案")
                return
                
            # 統計資訊
            info_lines = [
                f"區域: {region_name}",
                f"快取目錄: {cache_dir}",
                f"NPY檔案數量: {len(npy_files)}",
                f"PNG檔案數量: {len(png_files)}",
                ""
            ]
            
            # 列出配對的檔案
            if npy_files:
                info_lines.append("現有快取編號:")
                numbers = []
                for npy_file in npy_files:
                    try:
                        number = int(npy_file.stem)
                        numbers.append(number)
                        png_file = cache_dir / f"{number}.png"
                        status = "✓" if png_file.exists() else "✗"
                        info_lines.append(f"  #{number}: {npy_file.name} + {number}.png {status}")
                    except ValueError:
                        info_lines.append(f"  {npy_file.name} (非標準格式)")
                        
                if numbers:
                    info_lines.append(f"\n最新編號: #{max(numbers)}")
                    info_lines.append(f"下一個編號: #{max(numbers) + 1}")
            
            # 顯示資訊對話框
            info_text = "\n".join(info_lines)
            messagebox.showinfo("快取資訊", info_text)
            
        except Exception as e:
            print(f"顯示快取資訊時發生錯誤: {e}")
            messagebox.showerror("錯誤", f"顯示快取資訊失敗: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotator(root)
    root.mainloop()
    