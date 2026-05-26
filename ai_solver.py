import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import time
import base64
import concurrent.futures
import os
import sys
import shutil
import subprocess
import tempfile
import csv
import sqlite3
import json
from io import BytesIO
from PIL import ImageGrab, ImageTk, Image
from openai import OpenAI
from google import genai as google_genai
import fitz
import numpy as np
from ai_solver_utils import (
    capture_signature,
    chunk_text,
    cosine_similarity,
    embed_text,
    history_path,
    load_json_dict,
    load_history_cache,
    normalize_text_for_cache,
    preprocess_for_ocr,
    save_json_dict,
    save_history_cache,
    score_ocr_text,
    settings_path,
)

try:
    from pypdf import PdfReader
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

try:
    from docx import Document as DocxDocument
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False

# ─── CONFIG ───────────────────────────────────────────────────────────────────
POLL_INTERVAL      = 10.0
CHANGE_THRESHOLD   = 7
OCR_CONF_THRESHOLD = 0.85
OCR_MIN_TEXT_CHARS = 30
OLLAMA_DEFAULT_URL = 'http://localhost:11434/v1'
OLLAMA_DEFAULT_MDL = 'gemma4:e4b'
GEMINI_MODELS      = ['gemini-2.5-flash']
RPD_LIMIT          = 20
LLM_TIMEOUT        = 45.0
HISTORY_LIMIT      = 300
TESSERACT_LANGS    = 'kor+eng'
CHUNK_MAX_CHARS    = 1200
CHUNK_MIN_CHARS    = 300
RETRIEVE_TOP_K     = 6
IS_MACOS          = sys.platform == 'darwin'
IS_WINDOWS        = sys.platform.startswith('win')

# ─── SCREEN REGION SELECTOR ───────────────────────────────────────────────────
class RegionSelector:
    def __init__(self, callback):
        self.callback = callback
        self.root = tk.Toplevel()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)
        self.root.configure(bg='black')
        try:
            self.root.attributes('-topmost', True)
        except tk.TclError:
            pass
        self.canvas = tk.Canvas(self.root, cursor='cross', bg='grey11', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.start = None
        self.rect = None
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        self.root.focus_force()
        label = tk.Label(self.root, text="드래그로 문제 영역 선택  |  ESC 취소",
                         font=('Pretendard', 18, 'bold'), fg='white', bg='grey11')
        label.place(relx=0.5, rely=0.05, anchor='center')

    def on_press(self, e):
        self.start = (e.x, e.y)

    def on_drag(self, e):
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            *self.start, e.x, e.y, outline='#06b6d4', width=2, fill='')

    def on_release(self, e):
        if self.start:
            x1, y1 = self.start
            x2, y2 = e.x, e.y
            region = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
            self.root.destroy()
            self.callback(region)

# ─── REGION OVERLAY ───────────────────────────────────────────────────────────
class RegionOverlay:
    BAR_H  = 22
    HANDLE = 14
    COLOR  = '#06b6d4'
    TRANS  = '#010101'
    MIN_W  = 100
    MIN_H  = 60

    def __init__(self, region, on_change):
        self.on_change = on_change
        self._mode = None
        self._drag_start = (0, 0)
        self._win_start  = (0, 0, 0, 0)

        self.win = tk.Toplevel()
        self.win.overrideredirect(True)
        try:
            self.win.attributes('-topmost', True)
        except tk.TclError:
            pass
        if IS_WINDOWS:
            try:
                self.win.attributes('-transparentcolor', self.TRANS)
                self.win.configure(bg=self.TRANS)
            except tk.TclError:
                self.win.configure(bg='#0a0a12')
                self.win.attributes('-alpha', 0.35)
        else:
            self.win.configure(bg='#0a0a12')
            try:
                self.win.attributes('-alpha', 0.35)
            except tk.TclError:
                pass

        x1, y1, x2, y2 = region
        w = x2 - x1
        h = (y2 - y1) + self.BAR_H
        self.win.geometry(f'{w}x{h}+{x1}+{max(0, y1 - self.BAR_H)}')

        self.canvas = tk.Canvas(self.win, bg=self.TRANS, highlightthickness=0)
        if not IS_WINDOWS:
            self.canvas.configure(bg='#0a0a12')
        self.canvas.pack(fill='both', expand=True)

        self.win.update_idletasks()
        self._draw()
        self._bind()

    def get_region(self):
        x = self.win.winfo_x()
        y = self.win.winfo_y()
        w = self.win.winfo_width()
        h = self.win.winfo_height()
        return (x, y + self.BAR_H, x + w, y + h)

    def _draw(self):
        self.canvas.delete('all')
        w = self.win.winfo_width()
        h = self.win.winfo_height()
        if w < 2 or h < 2:
            return
        HS = self.HANDLE
        BH = self.BAR_H
        self.canvas.create_rectangle(0, 0, w, BH, fill=self.COLOR, outline='')
        self.canvas.create_text(w // 2, BH // 2, text='≡  드래그: 이동  |  코너: 크기 조절  ≡',
                                 fill='#0f0f1a', font=('Pretendard', 8, 'bold'))
        self.canvas.create_rectangle(0, BH, w - 1, h - 1,
                                      outline=self.COLOR, width=2, fill=self.TRANS)
        for cx, cy in [(0, BH), (w - HS, BH), (0, h - HS), (w - HS, h - HS)]:
            self.canvas.create_rectangle(cx, cy, cx + HS, cy + HS,
                                          fill=self.COLOR, outline='#0f0f1a', width=1)

    def _get_mode(self, x, y):
        w  = self.win.winfo_width()
        h  = self.win.winfo_height()
        HS = self.HANDLE
        BH = self.BAR_H
        if y < BH:
            return 'move'
        if x < HS     and y < BH + HS: return 'resize_tl'
        if x > w - HS and y < BH + HS: return 'resize_tr'
        if x < HS     and y > h - HS:  return 'resize_bl'
        if x > w - HS and y > h - HS:  return 'resize_br'
        return None

    def _bind(self):
        self.canvas.bind('<ButtonPress-1>',   self._on_press)
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Motion>',          self._on_motion)
        self.win.bind('<Configure>', lambda e: self.win.after_idle(self._draw))

    def _on_motion(self, e):
        cursors = {'move': 'fleur', 'resize_tl': 'size_nw_se',
                   'resize_tr': 'size_ne_sw', 'resize_bl': 'size_ne_sw',
                   'resize_br': 'size_nw_se'}
        self.canvas.config(cursor=cursors.get(self._get_mode(e.x, e.y), ''))

    def _on_press(self, e):
        self._mode       = self._get_mode(e.x, e.y)
        self._drag_start = (e.x_root, e.y_root)
        self._win_start  = (self.win.winfo_x(), self.win.winfo_y(),
                            self.win.winfo_width(), self.win.winfo_height())

    def _on_drag(self, e):
        if not self._mode:
            return
        dx = e.x_root - self._drag_start[0]
        dy = e.y_root - self._drag_start[1]
        sx, sy, sw, sh = self._win_start
        MW = self.MIN_W
        MH = self.MIN_H + self.BAR_H
        x, y, w, h = sx, sy, sw, sh
        if self._mode == 'move':
            x, y = sx + dx, sy + dy
        elif self._mode == 'resize_br':
            w = max(MW, sw + dx);  h = max(MH, sh + dy)
        elif self._mode == 'resize_bl':
            nw = max(MW, sw - dx); x = sx + sw - nw; w = nw; h = max(MH, sh + dy)
        elif self._mode == 'resize_tr':
            w = max(MW, sw + dx);  nh = max(MH, sh - dy); y = sy + sh - nh; h = nh
        elif self._mode == 'resize_tl':
            nw = max(MW, sw - dx); x = sx + sw - nw; w = nw
            nh = max(MH, sh - dy); y = sy + sh - nh; h = nh
        self.win.geometry(f'{w}x{h}+{x}+{y}')
        self.on_change(self.get_region())

    def _on_release(self, e):
        self._mode = None
        self.on_change(self.get_region())

    def destroy(self):
        try:
            self.win.destroy()
        except Exception:
            pass

# ─── MAIN APP ─────────────────────────────────────────────────────────────────
class AISolverApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("NEURAL_SOLVER")
        self.root.geometry("560x940")
        self.root.configure(bg='#0a0a12')
        self.root.resizable(True, True)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        self.region          = None
        self.overlay         = None
        self.running         = False
        self.last_img        = None
        self.thread          = None
        self.querying        = False
        self.study_materials = []
        self._closing        = False
        self._stop_event     = threading.Event()
        self._query_lock     = threading.Lock()
        self._material_db_lock = threading.Lock()
        self.last_signature  = None
        self.daily_count_day = time.strftime('%Y-%m-%d')
        self.answer_history  = {}
        # Gemini 키 사용량 추적
        self.key_idx         = 0
        self.model_idx       = 0
        self.daily_counts    = {}
        self.settings        = {}
        self.material_db     = None
        self._load_history_cache()
        self._init_material_db()
        self._load_material_index()

        self.tesseract_cmd = self._resolve_tesseract_cmd()
        self.tessdata_dir  = self._resolve_tessdata_dir()
        if self.tesseract_cmd:
            self._set_status_direct(f"내장 OCR 준비 완료 ({os.path.basename(self.tesseract_cmd)})")
        else:
            self._set_status_direct("Tesseract 미감지 - OCR 대신 이미지 모드 사용")

        self._build_ui()
        self._load_settings()
        self.root.mainloop()

    def _on_close(self):
        self._closing = True
        self.running = False
        self._stop_event.set()
        self._save_settings()
        if self.overlay:
            self.overlay.destroy()
        worker = self.thread
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=1.5)
        if self.material_db:
            self.material_db.close()
        self.root.destroy()

    def _set_status_direct(self, msg):
        print(msg)

    def _build_ui(self):
        BG    = '#0a0a12'
        CARD  = '#0d0d1a'
        CARD2 = '#060610'
        ACC   = '#06b6d4'
        ACC2  = '#0891b2'
        FG    = '#e2e8f0'
        FG2   = '#64748b'
        FG3   = '#1e293b'

        FONT_UI = 'Pretendard'
        FONT_FALLBACK = 'Malgun Gothic'
        FMONOB = (FONT_UI, 11, 'bold')
        FBIG   = (FONT_UI, 15, 'bold')
        FSMLB  = (FONT_UI, 9, 'bold')
        FSMALL = (FONT_FALLBACK, 9)

        self.root.configure(bg=BG)

        # ── HEADER BAR ──────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=ACC, height=44)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)

        icon_box = tk.Frame(hdr, bg=ACC2, width=30, height=30)
        icon_box.place(x=10, y=7)
        icon_box.pack_propagate(False)
        tk.Label(icon_box, text='⚡', font=(FONT_UI, 13, 'bold'), fg=BG, bg=ACC2).pack(expand=True)

        tk.Label(hdr, text='NEURAL_SOLVER  v2.5',
                 font=(FONT_UI, 13, 'bold'), fg=BG, bg=ACC).place(x=48, y=12)
        for i, sym in enumerate(['✕', '□', '─']):
            tk.Label(hdr, text=sym, font=FMONOB, fg=BG, bg=ACC).place(x=524 - i * 24, y=12)

        # ── STATUS CARD ─────────────────────────────────────────────────
        stat_card = tk.Frame(self.root, bg=CARD)
        stat_card.pack(fill='x', padx=12, pady=(12, 0))

        top = tk.Frame(stat_card, bg=CARD)
        top.pack(fill='x', padx=14, pady=(12, 8))

        lc = tk.Frame(top, bg=CARD)
        lc.pack(side='left')
        tk.Label(lc, text='작업 상태', font=FSMLB, fg=FG2, bg=CARD).pack(anchor='w')
        self.seq_lbl = tk.Label(lc, text='대기 중', font=FBIG, fg=FG, bg=CARD)
        self.seq_lbl.pack(anchor='w')

        rc = tk.Frame(top, bg=CARD)
        rc.pack(side='right')
        self.mode_lbl = tk.Label(rc, text='OCR 사용', font=FSMLB, fg=FG2, bg=CARD)
        self.mode_lbl.pack(anchor='e')

        pb_wrap = tk.Frame(stat_card, bg=CARD)
        pb_wrap.pack(fill='x', padx=14, pady=(0, 12))
        self.pb_canvas = tk.Canvas(pb_wrap, bg=FG3, height=6, highlightthickness=0)
        self.pb_canvas.pack(fill='x')
        self._progress = 0
        self._pb_color = ACC
        self.pb_canvas.bind('<Configure>', lambda e: self._draw_progress())

        # ── ACTION BUTTONS ───────────────────────────────────────────────
        row1 = tk.Frame(self.root, bg=BG)
        row1.pack(fill='x', padx=12, pady=(10, 0))

        self.start_btn = tk.Button(row1, text='▶  start',
            command=self._toggle, font=FMONOB,
            bg='#0c2a33', fg=ACC, activebackground=ACC, activeforeground=BG,
            relief='flat', bd=0, pady=13, cursor='hand2')
        self.start_btn.pack(side='left', fill='x', expand=True, padx=(0, 5))

        self.region_btn = tk.Button(row1, text='⊹  영역 선택',
            command=self._select_region, font=FMONOB,
            bg='#111120', fg=FG2, activebackground='#1e1e30', activeforeground=FG,
            relief='flat', bd=0, pady=13, cursor='hand2')
        self.region_btn.pack(side='left', fill='x', expand=True, padx=(5, 0))

        row2 = tk.Frame(self.root, bg=BG)
        row2.pack(fill='x', padx=12, pady=(5, 0))

        self.solve_btn = tk.Button(row2, text='⚡  즉시 풀이  [F9]',
            command=self._force_solve, font=FMONOB,
            bg='#0d2117', fg='#22c55e', activebackground='#14532d', activeforeground='#4ade80',
            relief='flat', bd=0, pady=11, cursor='hand2')
        self.solve_btn.pack(side='left', fill='x', expand=True, padx=(0, 5))

        self.settings_btn = tk.Button(row2, text='⚙  설정  ▼',
            command=self._toggle_settings, font=FMONOB,
            bg='#111120', fg=FG2, activebackground='#1e1e30', activeforeground=FG,
            relief='flat', bd=0, pady=11, cursor='hand2')
        self.settings_btn.pack(side='left', fill='x', expand=True, padx=(5, 0))

        self.root.bind('<F9>', lambda _: self._force_solve())

        # ── SETTINGS PANEL (hidden by default) ───────────────────────────
        self._settings_visible = False
        self.settings_panel = tk.Frame(self.root, bg=CARD)
        self._build_settings_panel(self.settings_panel, CARD, CARD2, FG, FG2, FG3, ACC)
        self._on_backend_change()

        # ── OUTPUT AREA ──────────────────────────────────────────────────
        self._output_frame = tk.Frame(self.root, bg=BG)
        self._output_frame.pack(fill='both', expand=True, padx=12, pady=(10, 0))

        out_hdr = tk.Frame(self._output_frame, bg='#111120')
        out_hdr.pack(fill='x')
        tk.Label(out_hdr, text='AI 답변', font=FSMLB, fg=FG2, bg='#111120').pack(
            side='left', padx=12, pady=6)
        self.conn_lbl = tk.Label(out_hdr, text='', font=FSMALL, fg=FG2, bg='#111120')
        self.conn_lbl.pack(side='right', padx=12, pady=6)

        self.output_txt = scrolledtext.ScrolledText(self._output_frame, wrap=tk.WORD,
            font=(FONT_FALLBACK, 12), bg=CARD2, fg='#94a3b8',
            insertbackground=ACC, relief='flat', bd=0, padx=14, pady=12, height=10)
        self.output_txt.pack(fill='both', expand=True)
        self.output_txt.config(state='disabled')

        self.preview_lbl = tk.Label(self._output_frame, bg=BG)
        self.preview_lbl.pack(pady=(0, 2))

        # ── REGION INFO ──────────────────────────────────────────────────
        self.region_label = tk.Label(self.root, text='영역: 선택되지 않음',
                                      font=FSMALL, fg=FG3, bg=BG)
        self.region_label.pack(pady=(3, 0))

        # ── LOG BAR ──────────────────────────────────────────────────────
        log_bar = tk.Frame(self.root, bg=CARD2)
        log_bar.pack(fill='x', pady=(5, 0))

        tk.Label(log_bar, text='▶', font=FSMALL, fg=ACC, bg=CARD2).pack(
            side='left', padx=(10, 2), pady=7)
        self.log_lbl = tk.Label(log_bar, text='준비 완료',
                                 font=FSMALL, fg=FG2, bg=CARD2, anchor='w')
        self.log_lbl.pack(side='left', fill='x', expand=True, pady=7)

        self._logs_visible = False
        self._log_lines = ['준비 완료', '영역 선택 필요']

        tk.Button(log_bar, text='로그', font=FSMLB, fg=ACC, bg=CARD2,
            activebackground='#0d0d1a', activeforeground=ACC,
            relief='flat', bd=0, command=self._toggle_logs).pack(side='right', padx=10, pady=4)

        self.logs_panel = tk.Frame(self.root, bg=CARD2)
        self.logs_inner = tk.Frame(self.logs_panel, bg=CARD2)
        self.logs_inner.pack(fill='x', padx=8, pady=4)

        # alias for legacy _set_status compat
        self.status_lbl = self.log_lbl

    def _build_settings_panel(self, parent, CARD, CARD2, FG, FG2, FG3, ACC):
        FONT_UI = 'Pretendard'
        FONT_FALLBACK = 'Malgun Gothic'
        FMONOB = (FONT_UI, 11, 'bold')
        FMONO  = (FONT_UI, 11)
        FSMLB  = (FONT_UI, 9, 'bold')
        FSMALL = (FONT_FALLBACK, 9)

        tk.Frame(parent, bg='#1e2a3a', height=1).pack(fill='x')
        inner = tk.Frame(parent, bg=CARD)
        inner.pack(fill='x', padx=12, pady=8)

        # ── Backend ──
        tk.Label(inner, text='AI 엔진', font=FSMLB, fg=ACC, bg=CARD).pack(anchor='w', pady=(0, 4))
        self.backend_var = tk.StringVar(value='ollama')
        radio_row = tk.Frame(inner, bg=CARD)
        radio_row.pack(fill='x', pady=(0, 6))
        tk.Radiobutton(radio_row, text='▣ Ollama (로컬)',
            variable=self.backend_var, value='ollama', command=self._on_backend_change,
            font=FMONO, bg=CARD, fg=FG, selectcolor='#0a0a12',
            activebackground=CARD).pack(side='left', padx=(0, 16))
        tk.Radiobutton(radio_row, text='◈ Gemini (클라우드)',
            variable=self.backend_var, value='gemini', command=self._on_backend_change,
            font=FMONO, bg=CARD, fg=FG, selectcolor='#0a0a12',
            activebackground=CARD).pack(side='left')

        # ── Ollama frame ──
        self.ollama_frame = tk.Frame(inner, bg=CARD)
        tk.Label(self.ollama_frame, text='URL', font=FSMLB, fg=FG2, bg=CARD).pack(anchor='w')
        self.url_entry = tk.Entry(self.ollama_frame, font=FMONO,
            bg=CARD2, fg=FG, insertbackground=ACC, relief='flat', bd=4)
        self.url_entry.insert(0, OLLAMA_DEFAULT_URL)
        self.url_entry.pack(fill='x', pady=(0, 4))
        tk.Label(self.ollama_frame, text='MODEL', font=FSMLB, fg=FG2, bg=CARD).pack(anchor='w')
        self.model_entry = tk.Entry(self.ollama_frame, font=FMONO,
            bg=CARD2, fg=FG, insertbackground=ACC, relief='flat', bd=4)
        self.model_entry.insert(0, OLLAMA_DEFAULT_MDL)
        self.model_entry.pack(fill='x', pady=(0, 4))

        # ── Gemini frame ──
        self.gemini_frame = tk.Frame(inner, bg=CARD)
        tk.Label(self.gemini_frame, text='API_KEY  (쉼표로 여러 개 입력)',
                 font=FSMLB, fg=FG2, bg=CARD).pack(anchor='w')
        self.api_entry = tk.Entry(self.gemini_frame, show='*', font=FMONO,
            bg=CARD2, fg=FG, insertbackground=ACC, relief='flat', bd=4)
        self.api_entry.pack(fill='x', pady=(0, 2))
        self.key_status_lbl = tk.Label(self.gemini_frame, text='', font=FSMALL, fg=FG2, bg=CARD)
        self.key_status_lbl.pack(anchor='w')
        self.count_lbl = tk.Label(self.gemini_frame, text='', font=FSMALL, fg='#f59e0b', bg=CARD)
        self.count_lbl.pack(anchor='w', pady=(0, 4))

        # anchor for _on_backend_change pack ordering
        self._backend_sep = tk.Frame(inner, bg=CARD)
        self._backend_sep.pack(fill='x')

        # ── Separator ──
        tk.Frame(inner, bg='#1e2a3a', height=1).pack(fill='x', pady=6)

        # ── Prompt ──
        tk.Label(inner, text='풀이 지시문', font=FSMLB, fg=ACC, bg=CARD).pack(anchor='w', pady=(0, 2))
        self.prompt_entry = tk.Entry(inner, font=FMONO,
            bg=CARD2, fg='#94a3b8', insertbackground=ACC, relief='flat', bd=4)
        self.prompt_entry.insert(0, '한국어로 단계별 풀이와 최종 답을 알려줘')
        self.prompt_entry.pack(fill='x', pady=(0, 6))

        # ── Image mode ──
        self.image_mode_var = tk.BooleanVar(value=False)
        tk.Checkbutton(inner,
            text='📐  수식/이미지 모드  (OCR 건너뛰기 — 수학·과학·다이어그램)',
            variable=self.image_mode_var, command=self._on_image_mode_change,
            font=(FONT_UI, 10), bg=CARD, fg='#f59e0b',
            selectcolor='#0a0a12', activebackground=CARD,
            anchor='w', wraplength=480, justify='left'
        ).pack(anchor='w', pady=(0, 6))

        # ── Separator ──
        tk.Frame(inner, bg='#1e2a3a', height=1).pack(fill='x', pady=6)

        # ── Study materials ──
        mat_hdr = tk.Frame(inner, bg=CARD)
        mat_hdr.pack(fill='x', pady=(0, 4))
        tk.Label(mat_hdr, text='학습 자료', font=FSMLB, fg=ACC, bg=CARD).pack(side='left')
        self.mat_count_lbl = tk.Label(mat_hdr, text='0개', font=FSMALL, fg=FG2, bg=CARD)
        self.mat_count_lbl.pack(side='right')

        self.mat_listbox = tk.Listbox(inner, height=3,
            bg=CARD2, fg=FG, selectbackground='#1a2f40',
            font=(FONT_UI, 10), relief='flat', bd=0,
            highlightthickness=1, highlightbackground='#1e2a3a')
        self.mat_listbox.pack(fill='x', pady=(0, 4))

        btn_r = tk.Frame(inner, bg=CARD)
        btn_r.pack(fill='x', pady=(0, 4))
        for txt, cmd, cbg, cfg in [
            ('＋ 추가',  self._add_material,   '#0c2a33', ACC),
            ('－ 삭제',  self._remove_material, '#2d1010', '#ef4444'),
            ('비우기',  self._clear_materials, '#111120', FG2),
        ]:
            tk.Button(btn_r, text=txt, command=cmd, font=(FONT_UI, 10, 'bold'),
                bg=cbg, fg=cfg, relief='flat', bd=0, padx=14, pady=6,
                cursor='hand2').pack(side='left', padx=(0, 5))

    # ── 백엔드 전환 ──────────────────────────────────────────────────────────
    def _on_backend_change(self):
        if self.backend_var.get() == 'ollama':
            self.gemini_frame.pack_forget()
            self.ollama_frame.pack(fill='x', before=self._backend_sep)
        else:
            self.ollama_frame.pack_forget()
            self.gemini_frame.pack(fill='x', before=self._backend_sep)

    def _toggle_settings(self):
        self._settings_visible = not self._settings_visible
        if self._settings_visible:
            self.settings_panel.pack(fill='x', padx=10, pady=(4, 0),
                                     before=self._output_frame)
            self.settings_btn.config(text='⚙  설정  ▲')
        else:
            self.settings_panel.pack_forget()
            self.settings_btn.config(text='⚙  설정  ▼')

    def _toggle_logs(self):
        self._logs_visible = not self._logs_visible
        if self._logs_visible:
            self.logs_panel.pack(fill='x')
        else:
            self.logs_panel.pack_forget()

    def _refresh_logs(self):
        if not hasattr(self, 'logs_inner'):
            return
        for w in self.logs_inner.winfo_children():
            w.destroy()
        for i, line in enumerate(self._log_lines[:8]):
            tk.Label(self.logs_inner, text=f'[{i}] {line}',
            font=('Malgun Gothic', 9), fg='#475569', bg='#060610',
                     anchor='w').pack(fill='x')

    def _set_progress(self, pct, color='#06b6d4'):
        self._progress = max(0, min(100, pct))
        self._pb_color = color
        self._draw_progress()

    def _draw_progress(self):
        if not hasattr(self, 'pb_canvas'):
            return
        c = self.pb_canvas
        w = c.winfo_width()
        if w < 4:
            return
        c.delete('all')
        h = c.winfo_height() or 6
        c.create_rectangle(0, 0, w, h, fill='#1e293b', outline='')
        if self._progress > 0:
            fw = max(6, int(w * self._progress / 100))
            c.create_rectangle(0, 0, fw, h, fill=self._pb_color, outline='')

    def _parse_status(self, msg):
        ACC     = '#06b6d4'
        SUCCESS = '#22c55e'
        ERR     = '#ef4444'
        NEUTRAL = '#64748b'
        m = msg.lower()
        if '완료' in m or 'success' in m:
            return 'SOLVE_COMPLETE', 100, SUCCESS
        if '강제 풀이' in m:
            return 'FORCE_SOLVING', 30, ACC
        if 'ocr' in m and '추출' in m:
            return 'SCANNING...', 25, ACC
        if '전송 중' in m:
            return 'UPLOADING...', 55, ACC
        if '처리 중' in m or '학습 중' in m:
            return 'PROCESSING...', 70, ACC
        if '학습 완료' in m:
            return 'MATERIAL_READY', 100, SUCCESS
        if '오류' in m or '❌' in m or 'error' in m:
            return 'ERROR', 0, ERR
        if '정지' in m:
            return 'STOPPED', 0, NEUTRAL
        if '대기' in m or '변경' in m:
            return 'MONITORING...', 0, NEUTRAL
        if '불러오는 중' in m:
            return 'LOADING...', 20, ACC
        return 'PROCESSING...', 40, ACC

    def _on_image_mode_change(self):
        mode = self.image_mode_var.get()
        self._safe_after(lambda: self.mode_lbl.config(
            text='이미지 모드 사용' if mode else 'OCR 사용',
            fg='#f59e0b' if mode else '#64748b'))

    def _is_ollama(self):
        return self.backend_var.get() == 'ollama'

    # ── Ollama 헬퍼 ──────────────────────────────────────────────────────────
    def _get_ollama_client(self):
        url = self.url_entry.get().strip() or OLLAMA_DEFAULT_URL
        return OpenAI(base_url=url, api_key='ollama', timeout=LLM_TIMEOUT)

    def _get_model(self):
        return self.model_entry.get().strip() or OLLAMA_DEFAULT_MDL

    # ── Gemini 헬퍼 ──────────────────────────────────────────────────────────
    def _get_keys(self):
        raw = self.api_entry.get().strip()
        return [k.strip() for k in raw.split(',') if k.strip()]

    def _inc_count(self, key):
        self._reset_daily_counts_if_needed()
        self.daily_counts[key] = self.daily_counts.get(key, 0) + 1

    def _set_key_status(self, key, model):
        masked = key[:8] + '...' + key[-4:]
        keys   = self._get_keys()
        idx    = self.key_idx % len(keys) + 1
        self._safe_after(lambda: self.key_status_lbl.config(
            text=f"키 {idx}/{len(keys)}: {masked}  |  모델: {model}", fg='#34d399'))

    def _update_count_label(self):
        self._reset_daily_counts_if_needed()
        keys  = self._get_keys()
        if not keys:
            return
        lines = []
        for i, k in enumerate(keys):
            used = self.daily_counts.get(k, 0)
            bar  = '█' * used + '░' * max(0, RPD_LIMIT - used)
            lines.append(f"키{i+1}: {used}/{RPD_LIMIT} {bar}")
        text  = '  |  '.join(lines)
        color = '#ef4444' if any(self.daily_counts.get(k,0) >= RPD_LIMIT for k in keys) else '#f59e0b'
        self._safe_after(lambda: self.count_lbl.config(text=text, fg=color))

    def _safe_after(self, callback, delay=0):
        if self._closing:
            return
        try:
            if self.root.winfo_exists():
                self.root.after(delay, callback)
        except tk.TclError:
            self._closing = True

    def _reset_daily_counts_if_needed(self):
        today = time.strftime('%Y-%m-%d')
        if self.daily_count_day != today:
            self.daily_count_day = today
            self.daily_counts.clear()

    def _app_base_dir(self):
        if getattr(sys, 'frozen', False):
            return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        return os.path.dirname(__file__)

    def _resolve_tesseract_cmd(self):
        candidates = [
            os.environ.get('TESSERACT_CMD'),
            os.path.join(self._app_base_dir(), 'vendor', 'tesseract', 'tesseract.exe'),
            os.path.join(self._app_base_dir(), 'vendor', 'tesseract', 'tesseract'),
            os.path.join(os.path.dirname(__file__), 'vendor', 'tesseract', 'tesseract.exe'),
            os.path.join(os.path.dirname(__file__), 'vendor', 'tesseract', 'tesseract'),
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            '/opt/homebrew/bin/tesseract',
            '/usr/local/bin/tesseract',
            shutil.which('tesseract'),
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def _resolve_tessdata_dir(self):
        candidates = [
            os.environ.get('TESSDATA_PREFIX'),
            os.path.join(self._app_base_dir(), 'vendor', 'tesseract', 'tessdata'),
            os.path.join(os.path.dirname(__file__), 'vendor', 'tesseract', 'tessdata'),
            r'C:\Program Files\Tesseract-OCR\tessdata',
            r'C:\Program Files (x86)\Tesseract-OCR\tessdata',
            '/opt/homebrew/share/tessdata',
            '/usr/local/share/tessdata',
        ]
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return candidate
        return None

    def _schedule_preview(self, img):
        preview = img.copy()
        self._safe_after(lambda i=preview: self._update_preview(i))

    def _capture_signature(self, img, text=''):
        return capture_signature(img, text)

    def _is_duplicate_capture(self, signature):
        return bool(signature) and signature == self.last_signature

    def _call_with_timeout(self, func, *args, timeout=LLM_TIMEOUT, **kwargs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as exc:
                raise TimeoutError(f"AI 응답이 {timeout:.0f}초 안에 오지 않았습니다.") from exc

    def _history_path(self):
        return history_path(os.path.dirname(__file__))

    def _settings_path(self):
        return settings_path(os.path.dirname(__file__))

    def _load_history_cache(self):
        self.answer_history = load_history_cache(self._history_path())

    def _save_history_cache(self):
        try:
            save_history_cache(self._history_path(), self.answer_history)
        except Exception as e:
            self._set_status(f"⚠️ 히스토리 저장 실패: {e}")

    def _normalize_text_for_cache(self, text):
        return normalize_text_for_cache(text)

    def _collect_settings(self):
        return {
            'backend': self.backend_var.get(),
            'ollama_url': self.url_entry.get().strip(),
            'ollama_model': self.model_entry.get().strip(),
            'gemini_keys': self.api_entry.get().strip(),
            'prompt': self.prompt_entry.get().strip(),
            'image_mode': bool(self.image_mode_var.get()),
            'settings_visible': bool(self._settings_visible),
        }

    def _apply_settings(self, settings):
        self.settings = settings or {}
        backend = self.settings.get('backend', 'ollama')
        self.backend_var.set(backend if backend in ('ollama', 'gemini') else 'ollama')

        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, self.settings.get('ollama_url') or OLLAMA_DEFAULT_URL)

        self.model_entry.delete(0, tk.END)
        self.model_entry.insert(0, self.settings.get('ollama_model') or OLLAMA_DEFAULT_MDL)

        self.api_entry.delete(0, tk.END)
        self.api_entry.insert(0, self.settings.get('gemini_keys', ''))

        self.prompt_entry.delete(0, tk.END)
        self.prompt_entry.insert(0, self.settings.get('prompt') or '한국어로 단계별 풀이와 최종 답을 알려줘')

        self.image_mode_var.set(bool(self.settings.get('image_mode', False)))
        self._on_backend_change()
        self._on_image_mode_change()
        if self.settings.get('settings_visible') and not self._settings_visible:
            self._toggle_settings()

    def _load_settings(self):
        self._apply_settings(load_json_dict(self._settings_path()))

    def _save_settings(self):
        try:
            save_json_dict(self._settings_path(), self._collect_settings())
        except Exception as e:
            self._set_status_direct(f"설정 저장 실패: {e}")

    def _material_db_path(self):
        return os.path.join(os.path.dirname(__file__), 'ai_solver_materials.db')

    def _init_material_db(self):
        self.material_db = sqlite3.connect(self._material_db_path(), check_same_thread=False)
        with self._material_db_lock:
            self.material_db.execute("""
                CREATE TABLE IF NOT EXISTS materials (
                    name TEXT PRIMARY KEY,
                    source_path TEXT,
                    status TEXT,
                    total_pages INTEGER DEFAULT 0,
                    failed_pages TEXT DEFAULT '[]',
                    chunk_count INTEGER DEFAULT 0,
                    updated_at TEXT
                )
            """)
            self.material_db.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material_name TEXT NOT NULL,
                    page_no INTEGER NOT NULL,
                    chunk_no INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    FOREIGN KEY(material_name) REFERENCES materials(name)
                )
            """)
            self.material_db.commit()

    def _load_material_index(self):
        if not self.material_db:
            return
        with self._material_db_lock:
            rows = self.material_db.execute("""
                SELECT name, status, total_pages, failed_pages, chunk_count
                FROM materials
                ORDER BY name
            """).fetchall()
        self.study_materials = []
        for name, status, total_pages, failed_pages, chunk_count in rows:
            try:
                failed = json.loads(failed_pages or '[]')
            except json.JSONDecodeError:
                failed = []
            detail = f"{chunk_count}개 chunk / {total_pages}페이지"
            if failed:
                detail += f" / 실패 {len(failed)}페이지"
            self.study_materials.append({
                'name': name,
                'status': status,
                'chunk_count': chunk_count,
                'total_pages': total_pages,
                'failed_pages': failed,
                'detail': detail,
            })

    def _upsert_material_meta(self, name, source_path, status, total_pages=0, failed_pages=None, chunk_count=0):
        failed_json = json.dumps(failed_pages or [], ensure_ascii=False)
        with self._material_db_lock:
            self.material_db.execute("""
                INSERT INTO materials(name, source_path, status, total_pages, failed_pages, chunk_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    source_path=excluded.source_path,
                    status=excluded.status,
                    total_pages=excluded.total_pages,
                    failed_pages=excluded.failed_pages,
                    chunk_count=excluded.chunk_count,
                    updated_at=excluded.updated_at
            """, (name, source_path, status, total_pages, failed_json, chunk_count, time.strftime('%Y-%m-%d %H:%M:%S')))
            self.material_db.commit()
        self._load_material_index()
        self._safe_after(self._refresh_mat_ui)

    def _replace_material_chunks(self, name, chunks):
        with self._material_db_lock:
            self.material_db.execute("DELETE FROM chunks WHERE material_name = ?", (name,))
            self.material_db.executemany("""
                INSERT INTO chunks(material_name, page_no, chunk_no, text, embedding)
                VALUES (?, ?, ?, ?, ?)
            """, chunks)
            self.material_db.commit()

    def _delete_material(self, name):
        with self._material_db_lock:
            self.material_db.execute("DELETE FROM chunks WHERE material_name = ?", (name,))
            self.material_db.execute("DELETE FROM materials WHERE name = ?", (name,))
            self.material_db.commit()
        self._load_material_index()

    def _clear_all_materials(self):
        with self._material_db_lock:
            self.material_db.execute("DELETE FROM chunks")
            self.material_db.execute("DELETE FROM materials")
            self.material_db.commit()
        self._load_material_index()

    def _history_lookup(self, cache_key):
        if not cache_key:
            return None
        return self.answer_history.get(cache_key)

    def _store_history_answer(self, cache_key, answer, meta=None):
        if not cache_key or not answer:
            return
        self.answer_history[cache_key] = {
            'answer': answer,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'meta': meta or {},
        }
        if len(self.answer_history) > HISTORY_LIMIT:
            overflow = len(self.answer_history) - HISTORY_LIMIT
            for old_key in list(self.answer_history.keys())[:overflow]:
                self.answer_history.pop(old_key, None)
        self._save_history_cache()

    # ── 학습자료 ─────────────────────────────────────────────────────────────
    def _add_material(self):
        paths = filedialog.askopenfilenames(
            title="학습자료 선택",
            filetypes=[
                ("PDF",      "*.pdf"),
                ("이미지",   "*.png *.jpg *.jpeg *.bmp"),
                ("전체",     "*.*"),
            ]
        )
        if not paths:
            return
        for path in paths:
            name = os.path.basename(path)
            self._upsert_material_meta(name, path, 'learning', 0, [], 0)
            self._set_status(f"📂 '{name}' 불러오는 중...")
            threading.Thread(target=self._load_material, args=(path, name), daemon=True).start()

    def _load_material(self, path, name):
        try:
            page_images = self._load_material_pages(path)
        except Exception as e:
            self._safe_after(lambda: self._set_status(f"❌ '{name}' 추출 오류: {e}"))
            self._upsert_material_meta(name, path, 'error', 0, [], 0)
            return

        try:
            total_pages = len(page_images)
            failed_pages = []
            stored_chunks = []
            total_chunks = 0
            for idx, image in enumerate(page_images, start=1):
                self._set_status(f"📄 '{name}' {idx}/{total_pages}페이지 처리 중...")
                page_text, ok = self._extract_material_page_text(image)
                if not ok:
                    failed_pages.append(idx)
                    continue
                for chunk_no, chunk in enumerate(chunk_text(page_text, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS), start=1):
                    embedding = embed_text(chunk).tolist()
                    stored_chunks.append((name, idx, chunk_no, chunk, json.dumps(embedding)))
                    total_chunks += 1

            if not stored_chunks:
                raise RuntimeError("유효한 페이지를 읽지 못함")

            self._replace_material_chunks(name, stored_chunks)
            self._upsert_material_meta(name, path, 'ready', total_pages, failed_pages, total_chunks)
            if failed_pages:
                self._safe_after(lambda fp=failed_pages, n=name, tp=total_pages:
                    self._set_status(f"✅ '{n}' 저장 완료 ({tp}페이지 중 실패 {len(fp)}페이지: {', '.join(map(str, fp))})"))
            else:
                self._safe_after(lambda n=name, tc=total_chunks: self._set_status(f"✅ '{n}' 저장 완료 ({tc}개 chunk)"))
        except Exception as e:
            self._safe_after(lambda: self._set_status(f"❌ '{name}' 학습 실패: {e}"))
            self._upsert_material_meta(name, path, 'error', len(page_images), [], 0)

    def _load_material_pages(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pdf':
            doc = fitz.open(path)
            pages = []
            try:
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                    pages.append(img)
            finally:
                doc.close()
            return pages
        if ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            return [Image.open(path).convert('RGB')]
        raise RuntimeError(f"지원하지 않는 형식: {ext}")

    def _extract_material_page_text(self, img):
        ocr_text, _ = self._ocr_extract(img)
        normalized = self._normalize_text_for_cache(ocr_text)
        if len(normalized) >= OCR_MIN_TEXT_CHARS:
            return ocr_text.strip(), True
        try:
            vision_text = self._query_material_vision(img)
        except Exception:
            vision_text = ''
        vision_text = (vision_text or '').strip()
        return vision_text, bool(vision_text)

    def _query_material_vision(self, img):
        prompt = (
            "이 페이지의 텍스트와 수식 내용을 최대한 정확하게 추출해줘. "
            "설명이나 요약 없이 원문에 가깝게 정리해줘."
        )
        if self._is_ollama():
            model  = self._get_model()
            client = self._get_ollama_client()
            b64 = self._img_to_b64(img)
            resp = self._call_with_timeout(
                client.chat.completions.create,
                model=model,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                        {'type': 'text', 'text': prompt},
                    ],
                }],
            )
            return resp.choices[0].message.content

        keys = self._get_keys()
        if not keys:
            raise RuntimeError("Gemini API Key가 없습니다.")
        client = google_genai.Client(api_key=keys[self.key_idx % len(keys)])
        img_bytes = base64.b64decode(self._img_to_b64(img))
        resp = self._call_with_timeout(
            client.models.generate_content,
            model=GEMINI_MODELS[0],
            contents=[
                google_genai.types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                google_genai.types.Part.from_text(text=prompt),
            ],
        )
        return resp.text

    def _remove_material(self):
        sel = self.mat_listbox.curselection()
        if not sel:
            return
        material = self.study_materials[sel[0]]
        self._delete_material(material['name'])
        self._refresh_mat_ui()

    def _clear_materials(self):
        self._clear_all_materials()
        self._refresh_mat_ui()

    def _refresh_mat_ui(self):
        self.mat_listbox.delete(0, tk.END)
        icon = {'learning': '⏳', 'ready': '✅', 'error': '❌'}
        ready_count = 0
        for m in self.study_materials:
            st     = m.get('status', 'learning')
            mark   = icon.get(st, '⏳')
            detail = m.get('detail', f"{m.get('chunk_count', 0)}개 chunk")
            self.mat_listbox.insert(tk.END, f"  {mark}  {m['name']}  ({detail})")
            if st == 'ready':
                ready_count += 1
        total = len(self.study_materials)
        self.mat_count_lbl.config(
            text=f"{total}개 (학습완료 {ready_count}개)",
            fg='#34d399' if ready_count > 0 else '#f59e0b' if total > 0 else '#6b7280')

    def _build_context(self, query_text=''):
        query = self._normalize_text_for_cache(query_text)
        if not query or not self.material_db:
            return ''
        query_vec = embed_text(query)
        with self._material_db_lock:
            rows = self.material_db.execute("""
                SELECT c.material_name, c.page_no, c.chunk_no, c.text, c.embedding
                FROM chunks c
                JOIN materials m ON m.name = c.material_name
                WHERE m.status = 'ready'
            """).fetchall()
        scored = []
        query_tokens = set(query.split())
        for material_name, page_no, chunk_no, text, embedding_json in rows:
            try:
                emb = json.loads(embedding_json)
            except json.JSONDecodeError:
                continue
            base_score = cosine_similarity(query_vec, np.array(emb, dtype=np.float32))
            text_tokens = set(self._normalize_text_for_cache(text).split())
            overlap = (len(query_tokens & text_tokens) / len(query_tokens)) if query_tokens else 0.0
            score = base_score + overlap
            if score > 0:
                scored.append((score, material_name, page_no, chunk_no, text))
        if not scored:
            return ''
        scored.sort(key=lambda item: item[0], reverse=True)
        parts = [
            f"[{name} / {page_no}페이지 / chunk {chunk_no}]\n{text}"
            for _, name, page_no, chunk_no, text in scored[:RETRIEVE_TOP_K]
        ]
        return "아래 학습자료 조각을 참고해서 답해.\n\n" + "\n\n──────────\n\n".join(parts) + "\n\n"

    # ── 영역 선택 ────────────────────────────────────────────────────────────
    def _select_region(self):
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
        self.root.withdraw()
        time.sleep(0.3)
        sel = RegionSelector(self._on_region_selected)
        self.root.wait_window(sel.root)   # ESC든 선택이든 sel.root가 닫히면 반환
        self.root.deiconify()             # 어떤 경우든 메인 창 복원

    def _on_region_selected(self, region):
        self.region = region
        self.last_img = None
        self.last_signature = None
        self.overlay = RegionOverlay(region, self._on_overlay_change)
        self._update_region_label(region)

    def _on_overlay_change(self, region):
        self.region   = region
        self.last_img = None
        self.last_signature = None
        self._update_region_label(region)

    def _update_region_label(self, region):
        x1, y1, x2, y2 = region
        w, h = x2 - x1, y2 - y1
        self.region_label.config(
            text=f"영역: ({x1},{y1})→({x2},{y2})  {w}×{h}px", fg='#22c55e')

    # ── 시작/정지 ────────────────────────────────────────────────────────────
    def _toggle(self):
        if not self.running:
            if not self.region:
                messagebox.showwarning("경고", "화면 영역을 먼저 선택하세요.")
                return
            if not self._is_ollama() and not self._get_keys():
                messagebox.showwarning("경고", "Gemini API Key를 입력하세요.")
                return
            self.running = True
            self._stop_event.clear()
            self.key_idx = 0; self.model_idx = 0
            self.start_btn.config(text='⏹  stop', bg='#2d1010', fg='#ef4444')
            self.last_img = None
            self.last_signature = None
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
        else:
            self.running = False
            self._stop_event.set()
            self.start_btn.config(text='▶  start', bg='#0c2a33', fg='#06b6d4')
            self._set_status("정지됨.")

    # ── 강제 풀기 ────────────────────────────────────────────────────────────
    def _force_solve(self):
        if not self.region:
            messagebox.showwarning("경고", "화면 영역을 먼저 선택하세요.")
            return
        if not self._is_ollama() and not self._get_keys():
            messagebox.showwarning("경고", "Gemini API Key를 입력하세요.")
            return
        if self.querying:
            self._set_status("⚠️ 이전 요청 처리 중입니다. 잠시 후 다시 시도하세요.")
            return
        def run():
            if not self._query_lock.acquire(blocking=False):
                self._set_status("⚠️ 이전 요청 처리 중입니다. 잠시 후 다시 시도하세요.")
                return
            self.querying = True
            try:
                img = ImageGrab.grab(bbox=self.region)
                self._schedule_preview(img)
                self._set_status("⚡ 강제 풀이 중...")
                answer, signature = self._query_ai(img)
                if not answer:
                    return
                if self._is_duplicate_capture(signature):
                    self._set_status("같은 문제라서 재전송하지 않음")
                    return
                self.last_signature = signature
                self._safe_after(lambda a=answer: self._show_answer(a))
                self.last_img = ImageGrab.grab(bbox=self.region)
                self._set_status("✅ 풀이 완료. 다음 문제 대기 중...")
            except Exception as e:
                self._set_status(f"❌ 오류: {e}")
            finally:
                self.querying = False
                self._query_lock.release()
        threading.Thread(target=run, daemon=True).start()

    # ── 모니터 루프 ──────────────────────────────────────────────────────────
    def _monitor_loop(self):
        while self.running and not self._stop_event.is_set():
            try:
                img  = ImageGrab.grab(bbox=self.region)
                diff = self._img_diff(img)
                if diff >= CHANGE_THRESHOLD:
                    if self.querying:
                        self._set_status("🔍 변경 감지됐지만 이전 요청 처리 중... 대기")
                    else:
                        self.last_img = img
                        self._set_status(f"🔍 변경 {diff:.1f}% 감지 → AI에 전송 중...")
                        self._schedule_preview(img)
                        if self._query_lock.acquire(blocking=False):
                            self.querying = True
                            try:
                                answer, signature = self._query_ai(img)
                                if answer and not self._is_duplicate_capture(signature):
                                    self.last_signature = signature
                                    self._safe_after(lambda a=answer: self._show_answer(a))
                                    self.last_img = ImageGrab.grab(bbox=self.region)
                                    self._set_status("✅ 풀이 완료. 다음 문제 대기 중...")
                                elif answer:
                                    self._set_status("같은 문제라서 재전송하지 않음")
                            finally:
                                self.querying = False
                                self._query_lock.release()
                else:
                    self._set_status(f"대기 중... (변경률 {diff:.1f}% / 기준 {CHANGE_THRESHOLD}%)")
            except Exception as e:
                self.querying = False
                self._set_status(f"❌ 오류: {e}")
            if self._stop_event.wait(POLL_INTERVAL):
                break

    def _img_diff(self, img):
        small = img.resize((64, 64)).convert('L')
        if self.last_img is None:
            self.last_img = img
            return 100.0
        prev    = self.last_img.resize((64, 64)).convert('L')
        total   = 64 * 64
        changed = sum(1 for a, b in zip(small.tobytes(), prev.tobytes()) if abs(a - b) > 10)
        return (changed / total) * 100

    def _update_preview(self, img):
        if self._closing:
            return
        thumb = img.copy()
        thumb.thumbnail((200, 120))
        tk_img = ImageTk.PhotoImage(thumb)
        self.preview_lbl.config(image=tk_img)
        self.preview_lbl.image = tk_img

    def _img_to_b64(self, img):
        img = img.copy()
        img.thumbnail((512, 512))
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def _preprocess_for_ocr(self, img):
        return preprocess_for_ocr(img)

    def _score_ocr_text(self, text):
        return score_ocr_text(text)

    def _run_tesseract_ocr(self, img):
        if not self.tesseract_cmd:
            return '', 0.0

        best_text = ''
        best_conf = 0.0
        env = os.environ.copy()
        if self.tessdata_dir:
            env['TESSDATA_PREFIX'] = self.tessdata_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            for name, candidate in self._preprocess_for_ocr(img):
                image_path = os.path.join(tmpdir, f'{name}.png')
                output_base = os.path.join(tmpdir, f'{name}_ocr')
                candidate.save(image_path)
                cmd = [
                    self.tesseract_cmd,
                    image_path,
                    output_base,
                    '-l', TESSERACT_LANGS,
                    '--oem', '1',
                    '--psm', '6',
                    'tsv',
                ]
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=env,
                    )
                except Exception:
                    continue

                tsv_path = output_base + '.tsv'
                if not os.path.exists(tsv_path):
                    continue

                texts = []
                confs = []
                try:
                    with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        for row in reader:
                            text = (row.get('text') or '').strip()
                            conf_raw = (row.get('conf') or '').strip()
                            if text:
                                texts.append(text)
                            try:
                                conf_value = float(conf_raw)
                            except ValueError:
                                conf_value = -1.0
                            if conf_value >= 0:
                                confs.append(conf_value)
                except Exception:
                    continue

                text = ' '.join(texts).strip()
                if not text:
                    continue
                conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
                quality_bonus = min(0.2, self._score_ocr_text(text) / 200.0)
                conf = min(1.0, max(conf, quality_bonus))
                if self._score_ocr_text(text) > self._score_ocr_text(best_text):
                    best_text = text
                    best_conf = conf

        return best_text, best_conf

    # ── OCR ──────────────────────────────────────────────────────────────────
    def _ocr_extract(self, img):
        text, conf = self._run_tesseract_ocr(img)
        if text and text.strip():
            return text.strip(), conf
        return '', 0.0

    # ── AI 쿼리 ──────────────────────────────────────────────────────────────
    def _query_ai(self, img):
        self._reset_daily_counts_if_needed()
        prompt_suffix = self.prompt_entry.get().strip()
        ready_count   = sum(1 for m in self.study_materials if m.get('status') == 'ready')
        mat_note      = f" (학습자료 {ready_count}개)" if ready_count else ""
        image_mode    = self.image_mode_var.get()

        # OCR (이미지 모드 OFF일 때만)
        ocr_text, ocr_conf = '', 0.0
        if not image_mode:
            self._set_status("🔎 OCR 텍스트 추출 중...")
            try:
                ocr_text, ocr_conf = self._ocr_extract(img)
            except Exception as ocr_err:
                self._set_status(f"⚠️ OCR 오류 → 이미지로 전환 ({type(ocr_err).__name__})")

        ocr_len = len(self._normalize_text_for_cache(ocr_text))
        use_image = image_mode or not (
            ocr_conf >= OCR_CONF_THRESHOLD and
            ocr_len >= OCR_MIN_TEXT_CHARS
        )
        system_ctx = self._build_context(ocr_text if ocr_text else prompt_suffix)
        signature = self._capture_signature(img, ocr_text if not use_image else '')
        cache_basis = self._normalize_text_for_cache(ocr_text) if not use_image else signature
        cache_key = f"{'ocr' if not use_image else 'img'}:{cache_basis}" if cache_basis else None
        cached = self._history_lookup(cache_key)
        if cached:
            self._set_status("📚 저장된 답변 재사용 중...")
            return cached.get('answer'), signature

        if self._is_ollama():
            answer = self._query_ollama(img, ocr_text, use_image, system_ctx, prompt_suffix, mat_note)
        else:
            answer = self._query_gemini(img, ocr_text, use_image, system_ctx, prompt_suffix, mat_note)
        self._store_history_answer(cache_key, answer, {
            'mode': 'image' if use_image else 'ocr',
            'signature': signature,
        })
        return answer, signature

    def _query_ollama(self, img, ocr_text, use_image, system_ctx, prompt_suffix, mat_note):
        model  = self._get_model()
        client = self._get_ollama_client()
        system_msg = system_ctx if system_ctx else "너는 문제 풀이 도우미야."

        if not use_image:
            self._set_status(f"📝 OCR 성공 → [{model}] 전송 중{mat_note}...")
            messages = [
                {'role': 'system', 'content': system_msg},
                {'role': 'user',   'content': f"다음 문제를 풀어줘.\n\n{ocr_text}\n\n{prompt_suffix}"}
            ]
        else:
            mode_str = "수식 모드" if self.image_mode_var.get() else "OCR 실패"
            self._set_status(f"🖼️ {mode_str} → [{model}] 이미지 전송 중{mat_note}...")
            b64 = self._img_to_b64(img)
            messages = [
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                    {'type': 'text', 'text': f"이 이미지에 있는 문제를 풀어줘. {prompt_suffix}"}
                ]}
            ]

        self._safe_after(lambda: self.conn_lbl.config(
            text=f"Ollama  |  {model}{mat_note}", fg='#34d399'))
        resp = self._call_with_timeout(
            client.chat.completions.create,
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content

    def _query_gemini(self, img, ocr_text, use_image, system_ctx, prompt_suffix, mat_note):
        keys = self._get_keys()
        if not keys:
            return None

        if not use_image:
            base_prompt = f"{system_ctx}다음 문제를 풀어줘.\n\n{ocr_text}\n\n{prompt_suffix}"
            contents    = [google_genai.types.Part.from_text(text=base_prompt)]
        else:
            mode_str = "수식 모드" if self.image_mode_var.get() else "OCR 실패"
            self._set_status(f"🖼️ {mode_str} → Gemini 이미지 전송 중{mat_note}...")
            b64       = self._img_to_b64(img)
            img_bytes = base64.b64decode(b64)
            prompt    = f"{system_ctx}이 이미지에 있는 문제를 풀어줘. {prompt_suffix}"
            contents  = [
                google_genai.types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                google_genai.types.Part.from_text(text=prompt)
            ]

        total_combos = len(keys) * len(GEMINI_MODELS)
        for _ in range(total_combos):
            key   = keys[self.key_idx % len(keys)]
            model = GEMINI_MODELS[self.model_idx % len(GEMINI_MODELS)]

            if self.daily_counts.get(key, 0) >= RPD_LIMIT:
                self._set_status(f"⚠️ 키 한도 소진 → 다음 키...")
                self.key_idx += 1; self.model_idx = 0
                continue

            self._set_key_status(key, model)
            mode_label = "수식" if self.image_mode_var.get() else ("OCR" if not use_image else "이미지")
            self._set_status(f"{'📝' if not use_image else '🖼️'} [{model}] 전송 중{mat_note} ({mode_label})...")
            try:
                client = google_genai.Client(api_key=key)
                cfg = google_genai.types.GenerateContentConfig(max_output_tokens=8192)
                resp = self._call_with_timeout(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=cfg,
                )
                self._inc_count(key)
                self._update_count_label()
                self._safe_after(lambda: self.conn_lbl.config(
                    text=f"Gemini  |  {model}{mat_note}", fg='#34d399'))
                return resp.text

            except Exception as e:
                err  = str(e)
                errl = err.lower()
                idx  = self.key_idx % len(keys) + 1
                # ── 잘못된 API 키 (가장 먼저 체크) ──────────────────────────
                _bad_key = (
                    'API_KEY_INVALID' in err or
                    'api key not valid' in errl or
                    'invalid api key' in errl or
                    'permission_denied' in errl or
                    '401' in err or
                    ('400' in err and 'key' in errl)
                )
                if _bad_key:
                    self._set_status(f"❌ {idx}번 API 키가 잘못되었습니다. 키를 확인하세요.")
                    self._safe_after(lambda i=idx: self.conn_lbl.config(
                        text=f"❌ {i}번 키 오류 — 키를 확인하세요", fg='#ef4444'))
                    self.key_idx += 1; self.model_idx = 0
                # ── 서버 과부하 ────────────────────────────────────────────
                elif '503' in err or 'UNAVAILABLE' in err:
                    self._set_status("⏳ 서버 과부하 → 15초 후 재시도...")
                    time.sleep(15)
                # ── 할당량 초과 ────────────────────────────────────────────
                elif '429' in err or 'RESOURCE_EXHAUSTED' in err:
                    if 'quota' in errl or 'daily' in errl or 'limit: 0' in errl:
                        self.daily_counts[key] = RPD_LIMIT
                        self._update_count_label()
                        self.key_idx += 1; self.model_idx = 0
                        self._set_status(f"⚠️ {idx}번 키 일일 한도 소진 → 다음 키...")
                        time.sleep(3)
                    else:
                        self._set_status("⚠️ RPM 초과 → 60초 대기...")
                        time.sleep(60)
                else:
                    self._safe_after(lambda: self.conn_lbl.config(
                        text=f"오류: {err[:50]}", fg='#ef4444'))
                    raise

        self._set_status("❌ 모든 Gemini 키 소진.")
        return None

    # ── 답변 창 ──────────────────────────────────────────────────────────────
    def _show_answer(self, text):
        # Update inline output area
        self.output_txt.config(state='normal')
        self.output_txt.delete('1.0', tk.END)
        self.output_txt.insert(tk.END, text)
        self.output_txt.config(state='disabled', fg='#e2e8f0')

        # Popup window (positioned next to selected region)
        if hasattr(self, '_answer_win') and self._answer_win.winfo_exists():
            self._answer_win.destroy()

        win = tk.Toplevel(self.root)
        win.title("NEURAL_SOLVER — SOLUTION")
        win.configure(bg='#0a0a12')
        win.attributes('-topmost', True)
        self._answer_win = win

        win_w, win_h = 520, 600
        if self.region:
            x1, y1, x2, y2 = self.region
            screen_w = self.root.winfo_screenwidth()
            px = x2 + 10 if x2 + win_w + 10 <= screen_w else max(0, x1 - win_w - 10)
            py = max(0, y1)
        else:
            px, py = 900, 100
        win.geometry(f"{win_w}x{win_h}+{px}+{py}")

        # Cyberpunk header
        pop_hdr = tk.Frame(win, bg='#06b6d4', height=30)
        pop_hdr.pack(fill='x')
        pop_hdr.pack_propagate(False)
        tk.Label(pop_hdr, text='⚡  SOLUTION_OUTPUT',
                 font=('Pretendard', 9, 'bold'), fg='#0a0a12', bg='#06b6d4').pack(
                 side='left', padx=10, pady=6)

        txt = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=('Malgun Gothic', 11),
            bg='#0d0d1a', fg='#e2e8f0', insertbackground='#06b6d4',
            relief='flat', bd=0, padx=14, pady=12)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, text)
        txt.config(state='disabled')

        btn_bar = tk.Frame(win, bg='#0a0a12')
        btn_bar.pack(fill='x', padx=14, pady=8)
        tk.Button(btn_bar, text='답변 복사', font=('Pretendard', 8, 'bold'),
            fg='#06b6d4', bg='#0c2a33', relief='flat', bd=0, padx=10, pady=4, cursor='hand2',
            command=lambda: (win.clipboard_clear(), win.clipboard_append(text))
        ).pack(side='right')

    def _set_status(self, msg):
        def _do():
            self.log_lbl.config(text=msg[:80])
            self._log_lines.insert(0, msg)
            self._log_lines = self._log_lines[:10]
            self._refresh_logs()
            seq, pct, color = self._parse_status(msg)
            self.seq_lbl.config(text=seq, fg=color)
            self._set_progress(pct, color)
        self._safe_after(_do)

# ─── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    AISolverApp()
