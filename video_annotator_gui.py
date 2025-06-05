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
import time

import cv2
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

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
        
        self.change_cache: Dict[int, bool] = {}
        self.ocr_cache: Dict[int, str] = {}
        self.annotations: Dict[int, str] = {}
        self.roi_image_cache: Dict[int, Image.Image] = {}

        # 新增：用於比較的舊版本資料
        self.old_annotations: Dict[int, str] = {}
        self.old_change_cache: Dict[int, bool] = {}
        self.comparison_mode = False  # 是否為比較模式
        self.has_unsaved_changes = False  # 是否有未儲存的變更

        # result_queue 仍然需要，用於從背景執行緒向UI傳遞結果和進度
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.save_lock = threading.Lock()

        self.ocr_iface = get_ocr_model(
            model_type="easyocr",
            gpu=torch.cuda.is_available(),
            lang_list=['en'],
            confidence_threshold=self.OCR_CONF_TH
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

        self.after(100, self._poll_queue)

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

        self.tree = ttk.Treeview(tree_frame, columns=("frame", "content"),
                                 show="headings", yscrollcommand=tree_yscroll.set,
                                 xscrollcommand=tree_xscroll.set)
        self.tree.pack(side="left", fill="y")
        self.tree.heading("frame", text="幀號")
        self.tree.heading("content", text="內容") 
        self.tree.column("frame", width=60, anchor="center")
        self.tree.column("content", width=200, anchor="w")
        tree_yscroll.config(command=self.tree.yview)
        tree_xscroll.config(command=self.tree.xview)
        self.bold_font = tkFont.Font(weight="bold")
        self.tree.tag_configure("changed", font=self.bold_font)
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
                height=180, 
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
            key_spacing = 40  # 進一步增加按鍵間距
            
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
            line_height = 15  # 增加行距
            
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
                    confidence_threshold=self.OCR_CONF_TH
                )
            elif selected_model == "EasyOCR High Precision":
                # 高精度模式：使用更嚴格的信心閾值和更完整的字符集
                self.ocr_iface = get_ocr_model(
                    model_type="easyocr",
                    gpu=torch.cuda.is_available(),
                    lang_list=['en'],
                    confidence_threshold=0.7,  # 更高的信心閾值
                    # allowlist="0123456789.-+",  # 如果OCR介面支援的話
                )
            elif selected_model == "EasyOCR Fast Mode":
                # 快速模式：較低的信心閾值，可能更快但精度稍低
                self.ocr_iface = get_ocr_model(
                    model_type="easyocr",
                    gpu=torch.cuda.is_available(),
                    lang_list=['en'],
                    confidence_threshold=0.3,  # 較低的信心閾值
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
        """顯示OCR測試視窗"""
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
                
            # 創建測試視窗
            self.ocr_test_window = tk.Toplevel(self.master)
            self.ocr_test_window.title(f"OCR測試 - 幀 {self.current_frame_idx} - {self.ocr_model_var.get()}")
            self.ocr_test_window.geometry("400x300")
            self.ocr_test_window.resizable(True, True)
            
            # 設置視窗關閉時的處理
            self.ocr_test_window.protocol("WM_DELETE_WINDOW", self._close_ocr_test_window)
            
            # 視窗內容框架
            main_frame = tk.Frame(self.ocr_test_window)
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # 顯示ROI圖像
            img_frame = tk.LabelFrame(main_frame, text="ROI 圖像")
            img_frame.pack(fill="x", pady=(0, 10))
            
            # 調整ROI圖像大小以適合顯示
            display_image = roi_image.copy()
            if display_image.size[0] > 200 or display_image.size[1] > 100:
                # 按比例縮放
                ratio = min(200/display_image.size[0], 100/display_image.size[1])
                new_size = (int(display_image.size[0] * ratio), int(display_image.size[1] * ratio))
                display_image = display_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 轉換為tkinter可顯示的圖像
            photo = ImageTk.PhotoImage(display_image)
            img_label = tk.Label(img_frame, image=photo)
            img_label.image = photo  # 保持引用
            img_label.pack()
            
            # OCR結果區域
            result_frame = tk.LabelFrame(main_frame, text="OCR 結果")
            result_frame.pack(fill="both", expand=True, pady=(0, 10))
            
            # 執行OCR並顯示結果
            self._perform_ocr_test(roi_image, result_frame)
            
            # 按鈕區域
            btn_frame = tk.Frame(main_frame)
            btn_frame.pack(fill="x")
            
            tk.Button(btn_frame, text="重新測試", 
                     command=lambda: self._perform_ocr_test(roi_image, result_frame)).pack(side="left", padx=(0, 5))
            tk.Button(btn_frame, text="關閉", 
                     command=self._close_ocr_test_window).pack(side="right")
            
            self.ocr_test_active = True
            self._update_status_bar(f"OCR測試視窗已開啟 (幀 {self.current_frame_idx})")
            
        except Exception as e:
            print(f"顯示OCR測試視窗時出錯: {e}")
            traceback.print_exc()
            messagebox.showerror("錯誤", f"無法顯示OCR測試視窗: {e}")

    def _close_ocr_test_window(self):
        """關閉OCR測試視窗"""
        if self.ocr_test_window:
            try:
                self.ocr_test_window.destroy()
            except:
                pass
            self.ocr_test_window = None
        self.ocr_test_active = False
        self._update_status_bar("OCR測試視窗已關閉")

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
        """執行OCR測試並在指定框架中顯示結果"""
        try:
            # 清空之前的結果
            for widget in result_frame.winfo_children():
                widget.destroy()
                
            # 顯示當前使用的模型
            model_label = tk.Label(result_frame, text=f"模型: {self.ocr_model_var.get()}", 
                                  font=("Arial", 9, "bold"))
            model_label.pack(anchor="w", padx=5, pady=(5, 0))
            
            # 執行OCR
            start_time = time.time()
            ocr_result = self.ocr_iface.recognize(roi_image)
            end_time = time.time()
            
            # 處理OCR結果
            if isinstance(ocr_result, tuple) and len(ocr_result) > 0:
                ocr_text = str(ocr_result[0])
            elif isinstance(ocr_result, str):
                ocr_text = ocr_result
            else:
                ocr_text = str(ocr_result) if ocr_result else ""
                
            # 顯示結果
            result_text = tk.Text(result_frame, height=4, width=40, wrap=tk.WORD)
            result_text.pack(fill="both", expand=True, padx=5, pady=5)
            
            # 插入結果文字
            result_text.insert("1.0", f"識別結果: {ocr_text}\n")
            result_text.insert("end", f"處理時間: {end_time - start_time:.3f} 秒\n")
            result_text.insert("end", f"幀號: {self.current_frame_idx}\n")
            result_text.insert("end", f"ROI: {self.roi_coords}")
            
            # 設為只讀
            result_text.config(state=tk.DISABLED)
            
            print(f"OCR測試完成: '{ocr_text}' (耗時 {end_time - start_time:.3f}s)")
            
        except Exception as e:
            # 顯示錯誤訊息
            error_label = tk.Label(result_frame, text=f"OCR測試失敗: {e}", 
                                  fg="red", wraplength=300)
            error_label.pack(padx=5, pady=5)
            print(f"OCR測試時出錯: {e}")
            traceback.print_exc()




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
        單一背景執行緒，順序處理所有幀的變化偵測和OCR。
        閾值作為參數傳入，避免從非主執行緒訪問tk.Var。
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

            self.result_queue.put_nowait(("progress", 0, self.total_frames, "processing"))
            frames_actually_processed = 0

            for frame_idx in range(self.total_frames):
                if self.stop_event.is_set():
                    print(f"主分析執行緒在幀 {frame_idx} 被停止。")
                    self.result_queue.put_nowait(("progress", frames_actually_processed, self.total_frames, "stopped"))
                    break

                # 1. 變化偵測
                # _detect_frame_change 需要修改以接受閾值
                has_change = self._detect_frame_change(frame_idx, worker_cap, tmad_threshold_val, diff_threshold_val)
                self.change_cache[frame_idx] = has_change
                try:
                    self.result_queue.put_nowait(("change", frame_idx, has_change))
                except queue.Full: pass #盡力而為

                # 2. 如果有變化，執行 OCR
                if has_change:
                    ocr_text = self._perform_ocr(frame_idx, worker_cap) # _perform_ocr 內部處理ROI獲取
                    self.ocr_cache[frame_idx] = ocr_text
                    # 標註也直接使用OCR結果，如果沒有手動編輯的話
                    if frame_idx not in self.annotations or not self.annotations[frame_idx]:
                        if ocr_text and ocr_text.strip(): # 只有非空文字才加入標註
                             self.annotations[frame_idx] = ocr_text

                    try:
                        self.result_queue.put_nowait(("ocr", frame_idx, ocr_text))
                    except queue.Full: pass

                frames_actually_processed += 1
                # 3. 更新進度
                try:
                    self.result_queue.put_nowait(("progress", frames_actually_processed, self.total_frames, "processing"))
                except queue.Full: pass
                
                if frames_actually_processed % 200 == 0 : # 每200幀打印一次日誌
                    print(f"主分析執行緒：已處理 {frames_actually_processed}/{self.total_frames} 幀。")


            if not self.stop_event.is_set():
                print(f"主分析執行緒完成所有 {self.total_frames} 幀的處理。")
                self.result_queue.put_nowait(("progress", self.total_frames, self.total_frames, "completed"))
            
        except Exception as e:
            current_progress = frames_actually_processed if 'frames_actually_processed' in locals() else 0
            print(f"主分析執行緒發生錯誤: {e}")
            traceback.print_exc()
            try:
                self.result_queue.put_nowait(("progress", current_progress, self.total_frames, "error"))
            except queue.Full: pass
        finally:
            if worker_cap:
                worker_cap.release()
            print(f"主分析執行緒結束，釋放VideoCapture。共處理 {frames_actually_processed if 'frames_actually_processed' in locals() else 0} 幀。")
            # Signal to UI that analysis is fully done, successful or not
            self.after(0, self._check_analysis_completion_status)


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

    def _show_frame(self, frame_idx: int):
        """
        1. 讀取 frame_idx 幀並顯示於 self.lbl_video。
        """
        if not self.cap_ui or not self.cap_ui.isOpened():
            print(f"警告：UI VideoCapture 未開啟或未設定，無法顯示幀 {frame_idx}")
            return
        if not (0 <= frame_idx < self.total_frames):
            return
            
        # 添加調試信息
        print(f"顯示幀: {frame_idx}")
        
        # --- 讀幀 ---
        self.cap_ui.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = self.cap_ui.read()
        if not ret:
            print(f"警告：無法讀取幀 {frame_idx}")
            return
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # --- 調整尺寸顯示 ---
        disp_pil = frame_pil.resize((self.VID_W, self.VID_H), Image.BILINEAR)
        if self.roi_coords and self.original_vid_w > 0 and self.original_vid_h > 0:
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

        # --- 更新 Slider/Label 顯示 ---
        self.slider_var.set(frame_idx)
        self.lbl_frame_num.config(text=f"幀: {frame_idx} / {self.total_frames-1 if self.total_frames > 0 else 0}")
        self.current_frame_idx = frame_idx

        # 同步更新 goto_entry
        self.goto_var.set(frame_idx)

        # 確保控制提示圖示保持在最前面
        if hasattr(self, 'control_hint_frame') and self.control_hint_frame:
            try:
                self.control_hint_frame.lift()
            except:
                pass
                    
        # 確保主窗口保持焦點，但不要強制搶奪焦點
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
        if self.roi_start_coords is None: # 沒有開始點，不處理
            self._show_frame(self.current_frame_idx) # 清除可能存在的臨時框
            return

        # 1. 取得並驗證原始座標 ----------------------------
        # roi_start_coords 已經是原始影片座標了 (在 _on_roi_start 中轉換)
        # event.x, event.y 是 canvas 座標，需要轉換
        start_x_orig, start_y_orig = self.roi_start_coords
        end_x_orig, end_y_orig = self._canvas_to_video_coords(event.x, event.y)

        # 確保 x1 < x2 and y1 < y2
        x1 = min(start_x_orig, end_x_orig)
        y1 = min(start_y_orig, end_y_orig)
        x2 = max(start_x_orig, end_x_orig)
        y2 = max(start_y_orig, end_y_orig)

        # 再次確保在邊界內 (理論上 _canvas_to_video_coords 已處理，但多一層保險)
        x1 = max(0, min(x1, self.original_vid_w - 1))
        y1 = max(0, min(y1, self.original_vid_h - 1))
        x2 = max(0, min(x2, self.original_vid_w - 1))
        y2 = max(0, min(y2, self.original_vid_h - 1))

        new_roi = (x1, y1, x2, y2)
        self.roi_start_coords = None # 重置拖曳起點

        if (x2 - x1) < 5 or (y2 - y1) < 5: # 面積太小或寬高太小
            print("ROI 太小，已忽略。")
            self._show_frame(self.current_frame_idx) # 清除臨時框並重繪
            return

        # 2. 儲存至 dict / 檔案 ------------------------------
        if new_roi != self.roi_coords: # 只有 ROI 實際改變時才觸發更新
            self.roi_coords = new_roi
            self.roi_dict[self.region_name] = list(self.roi_coords)
            self._save_roi_config() # 保存到 rois.json

            # 3. 清空快取與 TreeView 兩欄 ------------------------
            self.change_cache.clear()
            self.ocr_cache.clear()
            self.roi_image_cache.clear() # 清除 ROI 圖像快取

            # 清空 TreeView 中的 OCR 與變化欄 (如果需要，或者直接刷新)
            # 這裡選擇在重啟分析後由分析結果更新 TreeView

            # 4. 重新啟動背景線程 -------------------------------
            print(f"ROI 更新為 {self.roi_coords}，重新啟動分析...")
            self.stop_event.set()
            for th_name in ["analysis_thread", "ocr_thread"]:
                th = getattr(self, th_name, None)
                if th and th.is_alive():
                    th.join(timeout=1.0) # 給予停止時間
            self.stop_event.clear()

            # 清空隊列
            for q in [self.detect_queue, self.ocr_queue, self.result_queue]:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            
            self._start_background_threads() # 會重新填充隊列

            # 5. 更新 UI ----------------------------------------
            self._update_roi_fields() # 更新 Spinbox
            self._update_status_bar(f"{self.region_name} ROI 更新: {self.roi_coords}")
        
        self._show_frame(self.current_frame_idx) # 繪製新的 ROI 或清除臨時框

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
        """刷新 TreeView，顯示所有有標註的幀"""
        # 清空現有項目
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 收集所有有資料的幀
        all_frames = set()
        
        # 從標註中獲取幀
        if self.annotations:
            all_frames.update(self.annotations.keys())
        
        # 從變化快取中獲取有變化的幀
        if self.change_cache:
            change_frames = [frame for frame, has_change in self.change_cache.items() if has_change]
            all_frames.update(change_frames)
        
        if not all_frames:
            print("沒有資料需要顯示在TreeView中")
            return
        
        print(f"TreeView 刷新: 共 {len(all_frames)} 個幀有資料")
        
        # 按幀號排序並顯示
        for frame_idx in sorted(all_frames):
            # 獲取標註文字
            annotation_text = self.annotations.get(frame_idx, "")
            
            # 檢查是否為變化幀
            is_change = self.change_cache.get(frame_idx, False)
            
            # 插入到 TreeView
            item_id = self.tree.insert("", "end", values=(frame_idx, annotation_text))
            
            # 如果是變化幀，加上粗體標記
            if is_change:
                self.tree.item(item_id, tags=("changed",))
        
        print(f"TreeView 已更新，顯示 {len(all_frames)} 個幀")

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

    def _get_next_unanalyzed_frame(self) -> Optional[int]:
        """取得下一個未分析的幀"""
        for i in range(self.total_frames):
            if i not in self.change_cache:
                return i
        return None

    def _start_analysis(self):
        """開始分析當前區域的變化幀和OCR"""
        if not self.video_file_path or not self.roi_coords:
            messagebox.showwarning("警告", "請先載入影片並設定ROI區域")
            return
        
        # 詢問用戶是否要進行比較模式
        if self._has_existing_data():
            result = messagebox.askyesnocancel(
                "重新分析", 
                "檢測到現有的分析結果。\n\n" +
                "是 - 比較模式（保留舊結果，分析完成後顯示差異）\n" +
                "否 - 完全重新分析（清除舊結果）\n" +
                "取消 - 取消分析"
            )
            if result is None:  # 取消
                return
            elif result:  # 是 - 比較模式
                self.comparison_mode = True
                self._backup_current_data()
                print("已啟用比較模式，將保留現有結果用於比較")
            else:  # 否 - 完全重新分析
                self.comparison_mode = False
                print("將完全重新分析，清除所有現有結果")
        else:
            self.comparison_mode = False

        # 清空當前區域的快取
        self.change_cache.clear()
        self.ocr_cache.clear()
        self.annotations.clear()
        
        # 清空並刷新 TreeView
        self._refresh_treeview()
        
        # 更新按鈕狀態
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
        # 重置停止事件
        self.stop_event.clear()
        
        # 標記有未儲存的變更
        self.has_unsaved_changes = True

        # 啟動分析線程 (修正方法名稱)
        self._start_analysis_thread(self.tmad_threshold_var.get(), self.diff_threshold_var.get())
        
        mode_text = "比較模式" if self.comparison_mode else "重新分析"
        self._update_status_bar(f"開始{mode_text} - 區域 {self.region_name}...")
        print(f"開始{mode_text} - 區域: {self.region_name}, ROI: {self.roi_coords}")

    def _has_existing_data(self) -> bool:
        """檢查是否存在現有的分析資料"""
        if not self.video_file_path:
            return False
            
        annotations_path = Path("data") / self.video_title / f"{self.region_name}.jsonl"
        change_path = Path("data") / self.video_title / f"{self.region_name}_change.json"
        change_jsonl_path = Path("data") / self.video_title / f"{self.region_name}_change.jsonl"
        
        return annotations_path.exists() or change_path.exists() or change_jsonl_path.exists()

    def _backup_current_data(self):
        """備份當前資料用於比較"""
        # 備份現有的標註和變化快取
        self.old_annotations = self.annotations.copy()
        self.old_change_cache = self.change_cache.copy()
        print(f"已備份現有資料：{len(self.old_annotations)} 個標註，{len(self.old_change_cache)} 個變化記錄")

    def _on_analysis_complete(self):
        """分析自然完成後的處理"""
        print("主分析執行緒回報：分析自然完成。")
        if hasattr(self, 'btn_analyze'): 
            self.btn_analyze.config(state=tk.NORMAL if self.video_file_path else tk.DISABLED)
        if hasattr(self, 'btn_stop'): 
            self.btn_stop.config(state=tk.DISABLED)
        
        # 如果是比較模式，進行比較分析
        if self.comparison_mode:
            self._perform_comparison_analysis()
        
        # 更新進度條
        if self.total_frames > 0:
            if hasattr(self, 'progress_var'): 
                self.progress_var.set(self.total_frames)
            if hasattr(self, 'lbl_prog'): 
                self.lbl_prog.config(text=f"完成: {self.total_frames}/{self.total_frames}")
        
        # 不再自動儲存，等待用戶手動儲存
        status_msg = "分析完成" + ("（比較模式）" if self.comparison_mode else "") + " - 請檢視結果後手動儲存"
        self._update_status_bar(status_msg)

    def _perform_comparison_analysis(self):
        """執行比較分析並輸出差異報告"""
        print("\n" + "="*60)
        print("開始比較分析...")
        print("="*60)
        
        # 比較OCR結果差異
        ocr_differences = self._compare_ocr_results()
        
        # 比較變化幀差異
        change_differences = self._compare_change_results()
        
        # 輸出詳細報告
        self._print_comparison_report(ocr_differences, change_differences)
        
        print("="*60)
        print("比較分析完成。請檢視上述差異報告，確認無誤後再儲存結果。")
        print("="*60 + "\n")

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
        """停止分析"""
        self.stop_event.set()
        
        # 更新按鈕狀態
        if hasattr(self, 'btn_analyze'):
            self.btn_analyze.config(state=tk.NORMAL)
        if hasattr(self, 'btn_stop'):
            self.btn_stop.config(state=tk.DISABLED)
        
        # 儲存當前進度
        self._save_annotations(self.region_name)
        self._save_change_frames(self.region_name)
        
        self._update_status_bar("分析已停止")
        print("分析已停止")

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
        """切換 ROI 區域"""
        new_region = self.region_var.get()
        if new_region == self.region_name:
            return
        
        # 停止當前分析
        if self.btn_stop.cget('state') == tk.NORMAL:
            self._stop_analysis()
        
        # 儲存當前 region 的標註
        if self.annotations:
            self._save_annotations(self.region_name)
            self._save_change_frames(self.region_name)
        
        # 切換到新區域
        old_region = self.region_name
        self.region_name = new_region
        self.roi_coords = tuple(self.roi_dict[new_region])
        
        # 清空快取 (重要：避免新舊 region 資料混合)
        self.change_cache.clear()
        self.ocr_cache.clear()
        self.annotations.clear()
        self.roi_image_cache.clear()
        
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
                    
                    # 合併載入的 ROI 與預設 ROI
                    self.roi_dict.update(loaded_rois)
                    print(f"已載入全域 ROI 設定: {loaded_rois}")
            else:
                print(f"全域 ROI 設定檔不存在，使用預設設定")
        except Exception as e:
            print(f"載入全域 ROI 設定失敗: {e}")
        
        # 確保預設 region 存在
        if self.region_name not in self.roi_dict:
            self.roi_dict[self.region_name] = (1640, 445, 1836, 525)
        
        # 更新 UI
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

    def _get_change_frames_path(self, region_name: str) -> Optional[Path]:
        """取得指定 region 的變化幀檔案路徑"""
        if not self.video_file_path:
            return None
        
        # 使用統一的 video_title 變數
        return Path("data") / self.video_title / f"{region_name}_change.jsonl"

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

    def _poll_queue(self):
        """定期檢查結果隊列並更新 UI"""
        try:
            while True: 
                try:
                    result = self.result_queue.get_nowait()
                    
                    if not isinstance(result, tuple) or len(result) < 2: 
                        print(f"無效的結果格式: {result}")
                        continue
                    
                    result_type = result[0]
                    
                    if result_type == "change":
                        if len(result) < 3:
                            print(f"無效的 'change' 結果格式: {result}")
                            continue
                        frame_idx, has_change = result[1], result[2]
                        # self.change_cache[frame_idx] = has_change # Cache already updated by worker
                        if has_change and not self.tree.exists(f"F{frame_idx}"): # Add to tree only if changed and not exists
                            self._update_treeview_item(frame_idx, has_change=True) # Ensure item is created if it's a change
                        
                    elif result_type == "ocr":
                        if len(result) < 3:
                            print(f"無效的 'ocr' 結果格式: {result}")
                            continue
                        frame_idx, ocr_text = result[1], result[2]
                        # self.ocr_cache[frame_idx] = ocr_text # Cache already updated by worker
                        # self.annotations[frame_idx] = ocr_text # Annotations also updated by worker
                        self._update_treeview_item(frame_idx, ocr_text=ocr_text) 
                    
                    elif result_type == "progress":
                        if len(result) < 3: 
                            print(f"無效的 'progress' 結果格式: {result}")
                            continue
                        current = result[1]
                        total = result[2]
                        status_msg = result[3] if len(result) > 3 else "processing"

                        if total > 0 : 
                            self.progress_var.set(current) 
                            self.lbl_prog.config(text=f"進度: {current}/{total}")
                        else:
                            self.progress_var.set(0)
                            self.lbl_prog.config(text="進度: 0/0")

                        if status_msg == "completed" and current >= total:
                            self._update_status_bar("所有幀分析完成。")
                            # _on_analysis_complete will be called via _check_analysis_completion_status
                        elif status_msg == "error_no_video":
                            self._update_status_bar("錯誤：未選擇影片檔案。")
                        elif status_msg == "error_open_video":
                            self._update_status_bar("錯誤：無法開啟影片檔案進行分析。")
                        elif status_msg == "error":
                             self._update_status_bar("分析過程中發生錯誤。")
                        elif status_msg == "stopped":
                             self._update_status_bar("分析已手動停止。")
                             # Button states handled by _check_analysis_completion_status or _stop_analysis

                except queue.Empty:
                    break 
                    
        except Exception as e:
            print(f"處理結果隊列時出錯: {e}")
            traceback.print_exc()
        
        self.after(100, self._poll_queue)

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

    def _update_treeview_item(self, frame_idx: int, has_change: Optional[bool] = None, ocr_text: Optional[str] = None):
        """更新 TreeView 中的特定項目。如果項目不存在則創建它。"""
        item_id_str = f"F{frame_idx}" # Use a unique string ID for items

        if not self.tree.exists(item_id_str):
            # Item does not exist, insert it, especially if it's a change or has OCR
            # Default content should be from ocr_cache or annotations if available
            initial_content = self.ocr_cache.get(frame_idx, self.annotations.get(frame_idx, ""))
            self.tree.insert("", "end", iid=item_id_str, values=(frame_idx, initial_content))
            if self.change_cache.get(frame_idx, False): # Check actual change_cache for bolding
                 self.tree.item(item_id_str, tags=("changed",))

        # Now update content if ocr_text is provided
        if ocr_text is not None:
            self.tree.set(item_id_str, "content", ocr_text)
        
        # Ensure "changed" tag is correctly applied based on the definitive change_cache
        if self.change_cache.get(frame_idx, False):
            if not "changed" in self.tree.item(item_id_str, "tags"):
                self.tree.item(item_id_str, tags=("changed",))
        else: # If it was previously marked changed but now isn't (e.g. re-analysis)
            current_tags = list(self.tree.item(item_id_str, "tags"))
            if "changed" in current_tags:
                current_tags.remove("changed")
                self.tree.item(item_id_str, tags=tuple(current_tags))

    def _save_annotations(self, region_name: str):
        """儲存標註結果為 JSONL 格式 - 統一版本"""
        if not self.annotations:
            print("無標註內容需要儲存。")
            return
        
        try:
            if not self.video_file_path:
                messagebox.showerror("錯誤", "無法儲存標註，影片路徑未設定。")
                print("錯誤: _save_annotations 無法獲取有效的 video_file_path。")
                return

            video_data_dir = Path("data") / self.video_title
            video_data_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = video_data_dir / f"{region_name}.jsonl"
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for frame_idx in sorted(self.annotations.keys()):
                    ocr_text = self.annotations[frame_idx]
                    # 統一格式：只保留 frame 和 ocr_text
                    record = {
                        "frame": frame_idx,
                        "ocr_text": ocr_text
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"標註已儲存至: {jsonl_path}")
            self._update_status_bar(f"標註已儲存: {jsonl_path.name}") #  更簡潔的狀態更新
            
        except Exception as e:
            messagebox.showerror("儲存標註失敗", f"儲存標註 (region: {region_name}) 時出錯: {e}")
            print(f"儲存標註 (region: {region_name}) 時出錯: {e}")
            traceback.print_exc()
            self._update_status_bar(f"儲存標註 {region_name} 失敗")

    def _save_change_frames(self, region_name: str):
        """儲存變化幀列表為 JSON 格式 - 統一版本"""
        if not self.change_cache:
            print(f"區域 {region_name}: 無變化幀數據需要儲存。")
            # self._update_status_bar(f"區域 {region_name}: 無變化幀可儲存") # 可選的狀態更新
            return
        
        try:
            if not self.video_file_path:
                messagebox.showerror("錯誤", f"無法確定區域 {region_name} 的變化幀儲存路徑。影片是否已載入？")
                print(f"錯誤: _save_change_frames 無法獲取有效的 video_file_path for region {region_name}.")
                # self._update_status_bar(f"區域 {region_name}: 變化幀儲存路徑無效") # 可選
                return

            video_data_dir = Path("data") / self.video_title
            video_data_dir.mkdir(parents=True, exist_ok=True) 
            
            # 變化幀檔案路徑 (使用 .json 儲存幀號列表)
            change_path = video_data_dir / f"{region_name}_change.json"
            
            # 只儲存值為 True 的幀號 (即有變化的幀)
            changed_frame_indices = sorted([
                frame_idx for frame_idx, has_change in self.change_cache.items() if has_change
            ])
            
            if not changed_frame_indices:
                print(f"區域 {region_name}: 計算後沒有偵測到任何變化幀可儲存。")
                # 如果希望在沒有變化幀時刪除舊的變化幀文件，可以取消下面的註解
                # if change_path.exists():
                #     try:
                #         change_path.unlink()
                #         print(f"已刪除舊的空變化幀檔案: {change_path}")
                #     except OSError as e:
                #         print(f"刪除舊變化幀檔案 {change_path} 失敗: {e}")
                return

            with self.save_lock: # 確保檔案寫入的執行緒安全
                with open(change_path, 'w', encoding='utf-8') as f:
                    json.dump(changed_frame_indices, f, ensure_ascii=False, indent=2)
            
            print(f"區域 {region_name} 的變化幀列表已儲存至: {change_path} (共 {len(changed_frame_indices)} 個變化幀)")
            # self._update_status_bar(f"{region_name}: {len(changed_frame_indices)} 個變化幀已儲存") # 可選
            
        except Exception as e:
            messagebox.showerror("儲存變化幀失敗", f"儲存區域 {region_name} 變化幀時出錯: {e}")
            print(f"儲存區域 {region_name} 變化幀時出錯: {e}")
            traceback.print_exc()

    def _load_change_frames(self, region_name: str):
        """載入變化幀列表 - 支援新舊格式"""
        try:
            if not self.video_file_path:
                print(f"錯誤: _load_change_frames 無法獲取有效的 video_file_path for region {region_name}.")
                return

            video_data_dir = Path("data") / self.video_title
            
            # 優先嘗試新格式 (.json)
            change_path = video_data_dir / f"{region_name}_change.json"
            if change_path.exists():
                print(f"載入變化幀檔案: {change_path}")
                with open(change_path, 'r', encoding='utf-8') as f:
                    change_frames = json.load(f)
                
                if isinstance(change_frames, list):
                    self._rebuild_change_cache(change_frames)
                    print(f"已載入 {len(change_frames)} 個變化幀 (新格式) for region {region_name}")
                    return
            
            # 嘗試舊格式 (.jsonl)
            change_jsonl_path = video_data_dir / f"{region_name}_change.jsonl"
            if change_jsonl_path.exists():
                print(f"載入變化幀檔案: {change_jsonl_path}")
                change_frames = []
                with open(change_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            if isinstance(record, dict) and "frame" in record:
                                if record.get("change", True):
                                    change_frames.append(record["frame"])
                        except (json.JSONDecodeError, KeyError):
                            print(f"解析舊格式 change.jsonl 行失敗: {line}")
                            continue
                
                self._rebuild_change_cache(change_frames)
                print(f"已載入 {len(change_frames)} 個變化幀 (舊格式 _change.jsonl) for region {region_name}")
                return
            
            print(f"變化幀檔案不存在 for region {region_name} (已檢查 .json 和 .jsonl)")
            
        except Exception as e:
            print(f"載入區域 {region_name} 的變化幀時出錯: {e}")
            traceback.print_exc()


    def _rebuild_change_cache(self, change_frames: list):
        """重建變化幀快取"""
        self.change_cache.clear()
        
        # 先將所有幀設為無變化
        for i in range(self.total_frames):
            self.change_cache[i] = False
        
        # 設定變化幀
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
        """應用程式關閉時的處理"""
        print("關閉應用程式...")
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            print("正在停止分析執行緒...")
            self.stop_event.set()
            self.analysis_thread.join(timeout=2.5) # Give a bit more time
            if self.analysis_thread.is_alive():
                print("警告: 分析執行緒未能優雅停止。")
        
        if self.video_file_path and self.region_name and (self.annotations or self.change_cache): # Check if there's anything to save
            print(f"儲存區域 {self.region_name} 的標註和變化資料...")
            try:
                self._save_annotations(self.region_name)
                self._save_change_frames(self.region_name) # Also save change frames
                print("資料已儲存。")
            except Exception as e:
                print(f"關閉時儲存資料出錯: {e}")
        else:
            print("無需儲存標註 (未載入影片或無標註/變化內容)。")
        
        try:
            self._save_roi_config()
            print("ROI 設定已儲存。")
        except Exception as e:
            print(f"關閉時儲存 ROI 設定出錯: {e}")
        
        print("應用程式已關閉。")
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotator(root)
    root.mainloop()
    