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

# ═══ CONFIG ═══════════════════════════════════════════════════════════════════
POLL_INTERVAL      = 10.0
CHANGE_THRESHOLD   = 7
OCR_CONF_THRESHOLD = 0.85
OCR_MIN_TEXT_CHARS = 30
OLLAMA_DEFAULT_URL = 'http://localhost:11434/v1'
OLLAMA_DEFAULT_MDL = 'gemma4:e4b'
GEMINI_MODELS      = ['gemini-3-flash-preview']
RPD_LIMIT          = 20
LLM_TIMEOUT        = 45.0
HISTORY_LIMIT      = 300
TESSERACT_LANGS    = 'kor+eng'
CHUNK_MAX_CHARS    = 1200
CHUNK_MIN_CHARS    = 300
RETRIEVE_TOP_K     = 6
APP_START_WIDTH    = 1120
APP_START_HEIGHT   = 760
APP_MIN_WIDTH      = 1040
APP_MIN_HEIGHT     = 700

# ═══ DESIGN TOKENS ════════════════════════════════════════════════════════════
# Surfaces
BG          = '#0a0a0f'   # page background
SURFACE     = '#13131a'   # main card / panel
SURFACE_2   = '#1a1a24'   # nested / elevated
SURFACE_3   = '#070709'   # deepest (sidebar)
BORDER      = '#24242e'
BORDER_SOFT = '#1f1f2a'

# Text
TEXT_PRI    = '#e4e4e7'
TEXT_SEC    = '#a1a1aa'
TEXT_MUTE   = '#71717a'
TEXT_DIM    = '#52525b'

# Accent (single primary)
ACCENT      = '#06b6d4'   # cyan
ACCENT_DARK = '#0891b2'
ACCENT_BG   = '#0c2a33'   # dark tint used as button bg

# Semantic (sparingly)
SUCCESS     = '#22c55e'
WARNING     = '#f59e0b'
DANGER      = '#ef4444'

# Typography
FONT_UI     = 'Pretendard'
FONT_FB     = 'Malgun Gothic'
F_DISPLAY   = (FONT_UI, 18, 'bold')
F_HEADING   = (FONT_UI, 13, 'bold')
F_BODY_B    = (FONT_UI, 11, 'bold')
F_BODY      = (FONT_UI, 11)
F_SMALL_B   = (FONT_UI, 9, 'bold')
F_SMALL     = (FONT_FB, 9)
F_OUTPUT    = (FONT_FB, 12)

# ═══ SCREEN REGION SELECTOR ═══════════════════════════════════════════════════
class RegionSelector:
    def __init__(self, callback):
        self.callback = callback
        self.root = tk.Toplevel()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.22)
        self.root.configure(bg='#050816')
        self.root.attributes('-topmost', True)
        self.canvas = tk.Canvas(self.root, cursor='cross', bg='#050816', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.start = None
        self.rect = None
        self.inner_rect = None
        self.guide_box = None
        self.guide_text = None
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        self.root.focus_force()
        label = tk.Label(self.root, text='드래그해서 문제 영역 선택  |  ESC 취소',
                         font=(FONT_UI, 18, 'bold'), fg='#f8fafc', bg='#050816')
        label.place(relx=0.5, rely=0.05, anchor='center')

    def on_press(self, e):
        self.start = (e.x, e.y)

    def on_drag(self, e):
        if self.rect:
            self.canvas.delete(self.rect)
        if self.inner_rect:
            self.canvas.delete(self.inner_rect)
        if self.guide_box:
            self.canvas.delete(self.guide_box)
        if self.guide_text:
            self.canvas.delete(self.guide_text)
        x1, y1 = self.start
        x2, y2 = e.x, e.y
        left, top = min(x1, x2), min(y1, y2)
        right, bottom = max(x1, x2), max(y1, y2)
        self.rect = self.canvas.create_rectangle(
            left, top, right, bottom,
            outline='#38bdf8', width=3, fill='#38bdf8', stipple='gray12')
        self.inner_rect = self.canvas.create_rectangle(
            left + 3, top + 3, right - 3, bottom - 3,
            outline='#e0f2fe', width=1, dash=(6, 4), fill='')
        guide_y = max(28, top - 28)
        self.guide_box = self.canvas.create_rectangle(
            left, guide_y, left + 170, guide_y + 24,
            fill='#082f49', outline='#38bdf8', width=1)
        self.guide_text = self.canvas.create_text(
            left + 10, guide_y + 12,
            text=f"{right - left} × {bottom - top}",
            anchor='w', fill='#e0f2fe', font=(FONT_UI, 10, 'bold'))

    def on_release(self, e):
        if self.start:
            x1, y1 = self.start
            x2, y2 = e.x, e.y
            region = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
            self.root.destroy()
            self.callback(region)

# ═══ REGION OVERLAY ═══════════════════════════════════════════════════════════
class RegionOverlay:
    BAR_H  = 28
    HANDLE = 16
    COLOR  = '#38bdf8'
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
        self.win.attributes('-topmost', True)
        self.win.attributes('-transparentcolor', self.TRANS)
        self.win.configure(bg=self.TRANS)

        x1, y1, x2, y2 = region
        w = x2 - x1
        h = (y2 - y1) + self.BAR_H
        self.win.geometry(f'{w}x{h}+{x1}+{max(0, y1 - self.BAR_H)}')

        self.canvas = tk.Canvas(self.win, bg=self.TRANS, highlightthickness=0)
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
        self.canvas.create_rectangle(0, 0, w, BH, fill='#082f49', outline=self.COLOR, width=2)
        self.canvas.create_text(
            w // 2, BH // 2,
            text='드래그로 이동  |  모서리로 크기 조절',
            fill='#e0f2fe',
            font=(FONT_UI, 9, 'bold'),
        )
        self.canvas.create_rectangle(0, BH, w - 1, h - 1,
                                     outline=self.COLOR, width=3, fill=self.TRANS)
        self.canvas.create_rectangle(4, BH + 4, w - 5, h - 5,
                                     outline='#bae6fd', width=1, dash=(6, 4), fill='')
        for cx, cy in [(0, BH), (w - HS, BH), (0, h - HS), (w - HS, h - HS)]:
            self.canvas.create_rectangle(cx, cy, cx + HS, cy + HS,
                                         fill='#e0f2fe', outline=self.COLOR, width=2)

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

# ═══ UI HELPERS ═══════════════════════════════════════════════════════════════
class HoverButton(tk.Button):
    """Button with hover / active state transitions."""
    def __init__(self, parent, hover_bg=None, hover_fg=None, **kwargs):
        self._base_bg = kwargs.get('bg', SURFACE)
        self._base_fg = kwargs.get('fg', TEXT_PRI)
        self._hover_bg = hover_bg or self._base_bg
        self._hover_fg = hover_fg or self._base_fg
        kwargs.setdefault('relief', 'flat')
        kwargs.setdefault('bd', 0)
        kwargs.setdefault('cursor', 'hand2')
        kwargs.setdefault('activebackground', self._hover_bg)
        kwargs.setdefault('activeforeground', self._hover_fg)
        super().__init__(parent, **kwargs)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)

    def _on_enter(self, _):
        if self['state'] == 'disabled':
            return
        self.config(bg=self._hover_bg, fg=self._hover_fg)

    def _on_leave(self, _):
        self.config(bg=self._base_bg, fg=self._base_fg)

    def set_colors(self, bg, fg, hover_bg=None, hover_fg=None):
        self._base_bg = bg
        self._base_fg = fg
        self._hover_bg = hover_bg or bg
        self._hover_fg = hover_fg or fg
        self.config(bg=bg, fg=fg,
                    activebackground=self._hover_bg,
                    activeforeground=self._hover_fg)


# ═══ MAIN APP ═════════════════════════════════════════════════════════════════
class AISolverApp:
    SECTIONS = [('dashboard', '대시보드'),
                ('region',    '영역'),
                ('engine',    'AI 엔진'),
                ('material',  '학습 자료'),
                ('logs',      '로그')]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("NEURAL_SOLVER")
        self._set_initial_window_size()
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        self.region          = None
        self.overlay         = None
        self.overlay_visible = True
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
        self.key_idx         = 0
        self.model_idx       = 0
        self.daily_counts    = {}
        self.settings        = {}
        self.material_db     = None
        self.active_section  = 'dashboard'
        self._nav_buttons    = {}
        self._section_frames = {}
        self._log_lines      = ['준비 완료', '영역 선택 필요']

        self._load_history_cache()
        self._init_material_db()
        self._load_material_index()

        self.tesseract_cmd = self._resolve_tesseract_cmd()
        self.tessdata_dir  = self._resolve_tessdata_dir()
        if self.tesseract_cmd:
            self._set_status_direct(f"내장 OCR 준비 완료 ({os.path.basename(self.tesseract_cmd)})")
        else:
            self._set_status_direct("Tesseract 미감지 — OCR 실패 시 이미지 모드 사용")

        self._build_ui()
        self._load_settings()
        self.root.mainloop()

    def _set_initial_window_size(self):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(APP_START_WIDTH, max(APP_MIN_WIDTH, screen_w - 80))
        height = min(APP_START_HEIGHT, max(APP_MIN_HEIGHT, screen_h - 100))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(min(APP_MIN_WIDTH, width), min(APP_MIN_HEIGHT, height))

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

    # ── UI BUILD ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # top status bar
        topbar = tk.Frame(self.root, bg=SURFACE, height=40)
        topbar.pack(fill='x', side='top')
        topbar.pack_propagate(False)

        tk.Label(topbar, text='NEURAL_SOLVER',
                 font=(FONT_UI, 11, 'bold'), fg=TEXT_PRI, bg=SURFACE
                 ).pack(side='left', padx=(18, 8), pady=11)
        tk.Label(topbar, text='v2.5',
                 font=(FONT_UI, 9), fg=TEXT_DIM, bg=SURFACE
                 ).pack(side='left', pady=11)

        # status pills in top bar (right side)
        self.tb_state_lbl = tk.Label(topbar, text='● 대기 중',
            font=(FONT_UI, 9, 'bold'), fg=TEXT_MUTE, bg=SURFACE)
        self.tb_state_lbl.pack(side='right', padx=(0, 18), pady=11)

        self.tb_engine_lbl = tk.Label(topbar, text='엔진 미설정',
            font=(FONT_UI, 9), fg=TEXT_MUTE, bg=SURFACE)
        self.tb_engine_lbl.pack(side='right', padx=(0, 16), pady=11)

        self.tb_region_lbl = tk.Label(topbar, text='영역 없음',
            font=(FONT_UI, 9), fg=TEXT_MUTE, bg=SURFACE)
        self.tb_region_lbl.pack(side='right', padx=(0, 16), pady=11)

        # thin divider
        tk.Frame(self.root, bg=BORDER_SOFT, height=1).pack(fill='x')

        # body = sidebar + main
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True)

        self._build_sidebar(body)

        main = tk.Frame(body, bg=BG)
        main.pack(side='left', fill='both', expand=True, padx=16, pady=16)

        # build each section into its own frame, hidden by default
        for key, _ in self.SECTIONS:
            frm = tk.Frame(main, bg=BG)
            self._section_frames[key] = frm
        self._build_dashboard(self._section_frames['dashboard'])
        self._build_region(self._section_frames['region'])
        self._build_engine(self._section_frames['engine'])
        self._build_material(self._section_frames['material'])
        self._build_logs(self._section_frames['logs'])

        self._switch_section('dashboard')
        self.root.bind('<F9>', lambda _: self._force_solve())

        # progress tracker (for parser)
        self._progress = 0
        self._pb_color = ACCENT

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=SURFACE_3, width=60)
        side.pack(side='left', fill='y')
        side.pack_propagate(False)

        # logo badge
        badge = tk.Frame(side, bg=ACCENT, width=32, height=32)
        badge.pack(pady=(16, 20))
        badge.pack_propagate(False)
        tk.Label(badge, text='N', font=(FONT_UI, 14, 'bold'),
                 fg=BG, bg=ACCENT).pack(expand=True)

        # nav icons
        icons = {
            'dashboard': '▦',
            'region':    '◰',
            'engine':    '◉',
            'material':  '≡',
            'logs':      '◊',
        }
        for key, label in self.SECTIONS:
            btn_wrap = tk.Frame(side, bg=SURFACE_3)
            btn_wrap.pack(pady=2)

            indicator = tk.Frame(btn_wrap, bg=SURFACE_3, width=3, height=36)
            indicator.pack(side='left')
            indicator.pack_propagate(False)

            btn = HoverButton(btn_wrap, text=icons[key],
                font=(FONT_UI, 16),
                bg=SURFACE_3, fg=TEXT_DIM,
                hover_bg=SURFACE_2, hover_fg=TEXT_SEC,
                width=3, pady=6,
                command=lambda k=key: self._switch_section(k))
            btn.pack(side='left')

            self._nav_buttons[key] = (btn, indicator, label)

        # bottom status dot
        tk.Frame(side, bg=SURFACE_3).pack(fill='both', expand=True)
        self.side_status_dot = tk.Label(side, text='●',
            font=(FONT_UI, 10), fg=TEXT_DIM, bg=SURFACE_3)
        self.side_status_dot.pack(pady=(0, 14))

    def _switch_section(self, key):
        self.active_section = key
        for k, (btn, indicator, label) in self._nav_buttons.items():
            if k == key:
                btn.set_colors(SURFACE_2, ACCENT, SURFACE_2, ACCENT)
                indicator.config(bg=ACCENT)
            else:
                btn.set_colors(SURFACE_3, TEXT_DIM, SURFACE_2, TEXT_SEC)
                indicator.config(bg=SURFACE_3)
        for k, frm in self._section_frames.items():
            if k == key:
                frm.pack(fill='both', expand=True)
            else:
                frm.pack_forget()

    # ── section: dashboard ───────────────────────────────────────────────────
    def _build_dashboard(self, parent):
        # header
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill='x', pady=(0, 14))
        tk.Label(hdr, text='대시보드', font=F_DISPLAY,
                 fg=TEXT_PRI, bg=BG).pack(side='left')
        tk.Label(hdr, text='작업 상태와 AI 답변을 확인하세요',
                 font=F_SMALL, fg=TEXT_MUTE, bg=BG).pack(side='left', padx=(12, 0), pady=(8, 0))

        # content: left (answer) + right (controls)
        content = tk.Frame(parent, bg=BG)
        content.pack(fill='both', expand=True)

        left = tk.Frame(content, bg=BG)
        left.pack(side='left', fill='both', expand=True)

        right = tk.Frame(content, bg=BG, width=260)
        right.pack(side='right', fill='y', padx=(14, 0))
        right.pack_propagate(False)

        # status card (left top)
        stat_card = tk.Frame(left, bg=SURFACE)
        stat_card.pack(fill='x', pady=(0, 12))

        stat_inner = tk.Frame(stat_card, bg=SURFACE)
        stat_inner.pack(fill='x', padx=18, pady=14)

        top_row = tk.Frame(stat_inner, bg=SURFACE)
        top_row.pack(fill='x')

        left_col = tk.Frame(top_row, bg=SURFACE)
        left_col.pack(side='left')
        tk.Label(left_col, text='작업 상태', font=F_SMALL_B,
                 fg=TEXT_MUTE, bg=SURFACE).pack(anchor='w')
        self.seq_lbl = tk.Label(left_col, text='대기 중',
            font=(FONT_UI, 16, 'bold'), fg=TEXT_PRI, bg=SURFACE)
        self.seq_lbl.pack(anchor='w', pady=(2, 0))

        right_col = tk.Frame(top_row, bg=SURFACE)
        right_col.pack(side='right')
        tk.Label(right_col, text='입력 방식', font=F_SMALL_B,
                 fg=TEXT_MUTE, bg=SURFACE).pack(anchor='e')
        self.mode_lbl = tk.Label(right_col, text='OCR 사용',
            font=F_BODY_B, fg=TEXT_SEC, bg=SURFACE)
        self.mode_lbl.pack(anchor='e', pady=(2, 0))

        # progress bar
        self.pb_canvas = tk.Canvas(stat_inner, bg='#1e293b', height=4, highlightthickness=0)
        self.pb_canvas.pack(fill='x', pady=(14, 0))
        self.pb_canvas.bind('<Configure>', lambda e: self._draw_progress())

        # answer card
        ans_card = tk.Frame(left, bg=SURFACE)
        ans_card.pack(fill='both', expand=True)

        ans_hdr = tk.Frame(ans_card, bg=SURFACE, height=36)
        ans_hdr.pack(fill='x')
        ans_hdr.pack_propagate(False)
        tk.Label(ans_hdr, text='AI 답변', font=F_SMALL_B,
                 fg=TEXT_MUTE, bg=SURFACE).pack(side='left', padx=16)
        self.conn_lbl = tk.Label(ans_hdr, text='',
            font=F_SMALL, fg=TEXT_MUTE, bg=SURFACE)
        self.conn_lbl.pack(side='right', padx=16)
        tk.Frame(ans_card, bg=BORDER_SOFT, height=1).pack(fill='x')

        self.output_txt = scrolledtext.ScrolledText(ans_card, wrap=tk.WORD,
            font=F_OUTPUT, bg=SURFACE, fg=TEXT_SEC,
            insertbackground=ACCENT, relief='flat', bd=0,
            padx=18, pady=14)
        self.output_txt.pack(fill='both', expand=True)
        self.output_txt.config(state='disabled')

        # right side: primary actions
        act_title = tk.Label(right, text='액션', font=F_SMALL_B,
                             fg=TEXT_MUTE, bg=BG)
        act_title.pack(anchor='w', pady=(0, 8))

        self.start_btn = HoverButton(right, text='자동 풀이 시작',
            command=self._toggle, font=F_BODY_B,
            bg=ACCENT, fg=BG,
            hover_bg=ACCENT_DARK, hover_fg=BG,
            pady=14)
        self.start_btn.pack(fill='x', pady=(0, 6))

        self.solve_btn = HoverButton(right, text='수동 풀이 (F9)',
            command=self._force_solve, font=F_BODY_B,
            bg=SURFACE, fg=TEXT_PRI,
            hover_bg=SURFACE_2, hover_fg=TEXT_PRI,
            pady=11)
        self.solve_btn.pack(fill='x', pady=3)

        # mini preview
        tk.Label(right, text='최근 캡처', font=F_SMALL_B,
                 fg=TEXT_MUTE, bg=BG).pack(anchor='w', pady=(20, 8))

        self.preview_frame = tk.Frame(right, bg=SURFACE, height=120)
        self.preview_frame.pack(fill='x')
        self.preview_frame.pack_propagate(False)
        self.preview_lbl = tk.Label(self.preview_frame,
            bg=SURFACE, fg=TEXT_DIM,
            text='(영역을 선택하면\n캡처가 여기에 표시됩니다)',
            font=F_SMALL, justify='center')
        self.preview_lbl.pack(expand=True)

        # quick info stack
        info_wrap = tk.Frame(right, bg=BG)
        info_wrap.pack(fill='x', pady=(16, 0))

        self.db_region_info = self._kv_row(info_wrap, '영역', '선택되지 않음')
        self.db_engine_info = self._kv_row(info_wrap, '엔진', 'Ollama')
        self.db_mat_info    = self._kv_row(info_wrap, '학습자료', '0개')
        self.db_usage_info  = self._kv_row(info_wrap, '오늘 사용', '—')

    def _kv_row(self, parent, k, v):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill='x', pady=3)
        tk.Label(row, text=k, font=F_SMALL, fg=TEXT_MUTE, bg=BG
                 ).pack(side='left')
        val = tk.Label(row, text=v, font=F_SMALL_B, fg=TEXT_SEC, bg=BG)
        val.pack(side='right')
        return val

    # ── section: region ──────────────────────────────────────────────────────
    def _build_region(self, parent):
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill='x', pady=(0, 14))
        tk.Label(hdr, text='영역', font=F_DISPLAY,
                 fg=TEXT_PRI, bg=BG).pack(side='left')
        tk.Label(hdr, text='AI가 모니터링할 화면 영역을 지정하세요',
                 font=F_SMALL, fg=TEXT_MUTE, bg=BG).pack(side='left', padx=(12, 0), pady=(8, 0))

        card = tk.Frame(parent, bg=SURFACE)
        card.pack(fill='x')

        inner = tk.Frame(card, bg=SURFACE)
        inner.pack(fill='x', padx=20, pady=18)

        tk.Label(inner, text='현재 영역', font=F_SMALL_B,
                 fg=TEXT_MUTE, bg=SURFACE).pack(anchor='w')
        self.region_label = tk.Label(inner,
            text='선택되지 않음',
            font=(FONT_UI, 13), fg=TEXT_DIM, bg=SURFACE)
        self.region_label.pack(anchor='w', pady=(4, 16))

        btn_row = tk.Frame(inner, bg=SURFACE)
        btn_row.pack(fill='x')

        self.region_btn = HoverButton(btn_row, text='영역 선택',
            command=self._select_region, font=F_BODY_B,
            bg=ACCENT_BG, fg=ACCENT,
            hover_bg=ACCENT, hover_fg=BG,
            padx=20, pady=10)
        self.region_btn.pack(side='left', padx=(0, 8))

        self.overlay_btn = HoverButton(btn_row, text='영역 숨기기',
            command=self._toggle_overlay_visibility, font=F_BODY_B,
            bg=SURFACE_2, fg=TEXT_SEC,
            hover_bg=BORDER, hover_fg=TEXT_PRI,
            padx=20, pady=10, state='disabled')
        self.overlay_btn.pack(side='left')

        # hint card
        hint = tk.Frame(parent, bg=SURFACE_2)
        hint.pack(fill='x', pady=(16, 0))
        tk.Label(hint,
            text='TIP  드래그로 화면 영역 선택 후, 오버레이 모서리를 잡고 크기 조절할 수 있습니다.',
            font=F_SMALL, fg=TEXT_MUTE, bg=SURFACE_2,
            anchor='w', padx=18, pady=14).pack(fill='x')

    # ── section: engine ──────────────────────────────────────────────────────
    def _build_engine(self, parent):
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill='x', pady=(0, 14))
        tk.Label(hdr, text='AI 엔진', font=F_DISPLAY,
                 fg=TEXT_PRI, bg=BG).pack(side='left')
        tk.Label(hdr, text='로컬(Ollama) 또는 클라우드(Gemini) 백엔드를 선택하세요',
                 font=F_SMALL, fg=TEXT_MUTE, bg=BG).pack(side='left', padx=(12, 0), pady=(8, 0))

        scroll_container = tk.Frame(parent, bg=BG)
        scroll_container.pack(fill='both', expand=True)

        # backend selector
        sel_card = tk.Frame(scroll_container, bg=SURFACE)
        sel_card.pack(fill='x', pady=(0, 12))
        sel_inner = tk.Frame(sel_card, bg=SURFACE)
        sel_inner.pack(fill='x', padx=20, pady=16)

        tk.Label(sel_inner, text='백엔드', font=F_SMALL_B,
                 fg=TEXT_MUTE, bg=SURFACE).pack(anchor='w', pady=(0, 8))

        self.backend_var = tk.StringVar(value='ollama')
        radio_row = tk.Frame(sel_inner, bg=SURFACE)
        radio_row.pack(fill='x')

        for val, label in [('ollama', 'Ollama  (로컬)'),
                           ('gemini', 'Gemini  (클라우드)')]:
            tk.Radiobutton(radio_row, text=label,
                variable=self.backend_var, value=val,
                command=self._on_backend_change,
                font=F_BODY, bg=SURFACE, fg=TEXT_PRI,
                selectcolor=BG, activebackground=SURFACE,
                activeforeground=TEXT_PRI
            ).pack(side='left', padx=(0, 24))

        # ollama config
        self.ollama_frame = tk.Frame(scroll_container, bg=SURFACE)
        of_inner = tk.Frame(self.ollama_frame, bg=SURFACE)
        of_inner.pack(fill='x', padx=20, pady=16)

        tk.Label(of_inner, text='Ollama URL', font=F_SMALL_B,
                 fg=TEXT_MUTE, bg=SURFACE).pack(anchor='w')
        self.url_entry = tk.Entry(of_inner, font=F_BODY,
            bg=SURFACE_3, fg=TEXT_PRI, insertbackground=ACCENT,
            relief='flat', bd=0, highlightthickness=1,
            highlightbackground=BORDER_SOFT, highlightcolor=ACCENT)
        self.url_entry.insert(0, OLLAMA_DEFAULT_URL)
        self.url_entry.pack(fill='x', pady=(6, 14), ipady=8)

        tk.Label(of_inner, text='모델', font=F_SMALL_B,
                 fg=TEXT_MUTE, bg=SURFACE).pack(anchor='w')
        self.model_entry = tk.Entry(of_inner, font=F_BODY,
            bg=SURFACE_3, fg=TEXT_PRI, insertbackground=ACCENT,
            relief='flat', bd=0, highlightthickness=1,
            highlightbackground=BORDER_SOFT, highlightcolor=ACCENT)
        self.model_entry.insert(0, OLLAMA_DEFAULT_MDL)
        self.model_entry.pack(fill='x', pady=(6, 0), ipady=8)

        # gemini config
        self.gemini_frame = tk.Frame(scroll_container, bg=SURFACE)
        gf_inner = tk.Frame(self.gemini_frame, bg=SURFACE)
        gf_inner.pack(fill='x', padx=20, pady=16)

        tk.Label(gf_inner, text='API Key  (쉼표로 여러 개 입력)',
                 font=F_SMALL_B, fg=TEXT_MUTE, bg=SURFACE).pack(anchor='w')
        self.api_entry = tk.Entry(gf_inner, show='*', font=F_BODY,
            bg=SURFACE_3, fg=TEXT_PRI, insertbackground=ACCENT,
            relief='flat', bd=0, highlightthickness=1,
            highlightbackground=BORDER_SOFT, highlightcolor=ACCENT)
        self.api_entry.pack(fill='x', pady=(6, 8), ipady=8)

        self.key_status_lbl = tk.Label(gf_inner, text='',
            font=F_SMALL, fg=TEXT_MUTE, bg=SURFACE)
        self.key_status_lbl.pack(anchor='w')
        self.count_lbl = tk.Label(gf_inner, text='',
            font=F_SMALL, fg=WARNING, bg=SURFACE)
        self.count_lbl.pack(anchor='w', pady=(4, 0))

        # prompt + options card
        opt_card = tk.Frame(scroll_container, bg=SURFACE)
        opt_card.pack(fill='x', pady=(12, 0))
        opt_inner = tk.Frame(opt_card, bg=SURFACE)
        opt_inner.pack(fill='x', padx=20, pady=16)

        tk.Label(opt_inner, text='기본 지시문', font=F_SMALL_B,
                 fg=TEXT_MUTE, bg=SURFACE).pack(anchor='w')
        self.prompt_entry = tk.Entry(opt_inner, font=F_BODY,
            bg=SURFACE_3, fg=TEXT_SEC, insertbackground=ACCENT,
            relief='flat', bd=0, highlightthickness=1,
            highlightbackground=BORDER_SOFT, highlightcolor=ACCENT)
        self.prompt_entry.insert(0, '한국어로 단계별 풀이와 최종 답을 알려줘')
        self.prompt_entry.pack(fill='x', pady=(6, 14), ipady=8)

        self.image_mode_var = tk.BooleanVar(value=False)
        self.popup_mode_var = tk.BooleanVar(value=True)

        tk.Checkbutton(opt_inner,
            text='수식 / 이미지 모드 — OCR 대신 이미지 자체를 AI에 전달',
            variable=self.image_mode_var, command=self._on_image_mode_change,
            font=F_BODY, bg=SURFACE, fg=WARNING,
            selectcolor=BG, activebackground=SURFACE,
            activeforeground=WARNING, anchor='w'
        ).pack(anchor='w', pady=(0, 4))

        tk.Checkbutton(opt_inner,
            text='추가 결과 팝업 표시',
            variable=self.popup_mode_var,
            font=F_BODY, bg=SURFACE, fg='#93c5fd',
            selectcolor=BG, activebackground=SURFACE,
            activeforeground='#93c5fd', anchor='w'
        ).pack(anchor='w')

        self._on_backend_change()

    # ── section: material ────────────────────────────────────────────────────
    def _build_material(self, parent):
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill='x', pady=(0, 14))
        tk.Label(hdr, text='학습 자료', font=F_DISPLAY,
                 fg=TEXT_PRI, bg=BG).pack(side='left')
        self.mat_count_lbl = tk.Label(hdr, text='0개',
            font=F_SMALL_B, fg=TEXT_MUTE, bg=BG)
        self.mat_count_lbl.pack(side='right', pady=(6, 0))

        tk.Label(parent, text='PDF 또는 이미지 자료를 추가하면 답변에 참고됩니다',
                 font=F_SMALL, fg=TEXT_MUTE, bg=BG
                 ).pack(anchor='w', pady=(0, 14))

        # actions
        btn_row = tk.Frame(parent, bg=BG)
        btn_row.pack(fill='x', pady=(0, 12))

        HoverButton(btn_row, text='＋  자료 추가',
            command=self._add_material, font=F_BODY_B,
            bg=ACCENT_BG, fg=ACCENT,
            hover_bg=ACCENT, hover_fg=BG,
            padx=16, pady=10).pack(side='left', padx=(0, 6))

        HoverButton(btn_row, text='선택 삭제',
            command=self._remove_material, font=F_BODY_B,
            bg='#2d1010', fg=DANGER,
            hover_bg='#7f1d1d', hover_fg='#fff5f5',
            padx=16, pady=10).pack(side='left', padx=(0, 6))

        HoverButton(btn_row, text='전체 비우기',
            command=self._clear_materials, font=F_BODY_B,
            bg=SURFACE, fg=TEXT_MUTE,
            hover_bg=SURFACE_2, hover_fg=TEXT_SEC,
            padx=16, pady=10).pack(side='left')

        # list
        list_card = tk.Frame(parent, bg=SURFACE)
        list_card.pack(fill='both', expand=True)

        self.mat_listbox = tk.Listbox(list_card,
            bg=SURFACE, fg=TEXT_PRI, selectbackground=SURFACE_2,
            selectforeground=ACCENT,
            font=(FONT_UI, 11), relief='flat', bd=0,
            highlightthickness=0,
            activestyle='none')
        self.mat_listbox.pack(fill='both', expand=True, padx=14, pady=14)

    # ── section: logs ────────────────────────────────────────────────────────
    def _build_logs(self, parent):
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill='x', pady=(0, 14))
        tk.Label(hdr, text='로그', font=F_DISPLAY,
                 fg=TEXT_PRI, bg=BG).pack(side='left')
        tk.Label(hdr, text='최근 이벤트와 시스템 메시지',
                 font=F_SMALL, fg=TEXT_MUTE, bg=BG
                 ).pack(side='left', padx=(12, 0), pady=(8, 0))

        card = tk.Frame(parent, bg=SURFACE)
        card.pack(fill='both', expand=True)

        self.logs_text = scrolledtext.ScrolledText(card, wrap=tk.WORD,
            font=(FONT_FB, 10), bg=SURFACE, fg=TEXT_SEC,
            insertbackground=ACCENT, relief='flat', bd=0,
            padx=16, pady=14)
        self.logs_text.pack(fill='both', expand=True)
        self.logs_text.config(state='disabled')
        self._refresh_logs()

        # alias for legacy _set_status compat
        self.status_lbl = None  # not used anymore

    # ── UI STATE / HELPERS ────────────────────────────────────────────────────
    def _refresh_logs(self):
        if not hasattr(self, 'logs_text'):
            return
        self.logs_text.config(state='normal')
        self.logs_text.delete('1.0', tk.END)
        now = time.strftime('%H:%M:%S')
        for i, line in enumerate(self._log_lines[:50]):
            prefix = '›' if i == 0 else '·'
            self.logs_text.insert(tk.END, f' {prefix}  {line}\n')
        self.logs_text.config(state='disabled')

    def _set_progress(self, pct, color=ACCENT):
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
        h = c.winfo_height() or 4
        c.create_rectangle(0, 0, w, h, fill='#1e293b', outline='')
        if self._progress > 0:
            fw = max(6, int(w * self._progress / 100))
            c.create_rectangle(0, 0, fw, h, fill=self._pb_color, outline='')

    def _parse_status(self, msg):
        m = msg.lower()
        if '완료' in m or 'success' in m:
            return '완료', 100, SUCCESS
        if '강제' in m or '수동' in m:
            return '수동 풀이', 30, ACCENT
        if 'ocr' in m and '추출' in m:
            return '화면 스캔', 25, ACCENT
        if '전송 중' in m:
            return '업로드', 55, ACCENT
        if '처리 중' in m or '학습 중' in m:
            return '처리 중', 70, ACCENT
        if '학습 완료' in m:
            return '자료 준비됨', 100, SUCCESS
        if '오류' in m or 'error' in m:
            return '오류', 0, DANGER
        if '중지' in m:
            return '중지됨', 0, TEXT_MUTE
        if '대기' in m or '변경' in m or '변화' in m:
            return '모니터링', 0, TEXT_MUTE
        if '불러오는 중' in m:
            return '로딩 중', 20, ACCENT
        return '처리 중', 40, ACCENT

    def _on_image_mode_change(self):
        mode = self.image_mode_var.get()
        def _apply():
            if mode:
                self.mode_lbl.config(text='이미지 모드', fg=WARNING)
            else:
                self.mode_lbl.config(text='OCR 사용', fg=TEXT_SEC)
        self._safe_after(_apply)

    def _is_ollama(self):
        return self.backend_var.get() == 'ollama'

    # ── Ollama helpers ────────────────────────────────────────────────────────
    def _get_ollama_client(self):
        url = self.url_entry.get().strip() or OLLAMA_DEFAULT_URL
        return OpenAI(base_url=url, api_key='ollama', timeout=LLM_TIMEOUT)

    def _get_model(self):
        return self.model_entry.get().strip() or OLLAMA_DEFAULT_MDL

    # ── Gemini helpers ────────────────────────────────────────────────────────
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
            text=f"키 {idx}/{len(keys)}: {masked}  |  모델: {model}",
            fg=SUCCESS))

    def _update_count_label(self):
        self._reset_daily_counts_if_needed()
        keys  = self._get_keys()
        if not keys:
            return
        lines = []
        for i, k in enumerate(keys):
            used = self.daily_counts.get(k, 0)
            bar  = '■' * used + '□' * max(0, RPD_LIMIT - used)
            lines.append(f"키{i+1}: {used}/{RPD_LIMIT} {bar}")
        text  = '  |  '.join(lines)
        color = DANGER if any(self.daily_counts.get(k,0) >= RPD_LIMIT for k in keys) else WARNING
        self._safe_after(lambda: self.count_lbl.config(text=text, fg=color))
        # also update dashboard quick info
        total_used = sum(self.daily_counts.get(k, 0) for k in keys)
        total_cap  = RPD_LIMIT * len(keys)
        self._safe_after(lambda: self.db_usage_info.config(text=f'{total_used}/{total_cap}'))

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
            os.path.join(os.path.dirname(__file__), 'vendor', 'tesseract', 'tesseract.exe'),
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
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
                raise TimeoutError(f"AI 응답이 {timeout:.0f}초 내에 오지 않았습니다") from exc

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
            self._set_status(f"히스토리 저장 실패: {e}")

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
            'show_answer_popup': bool(self.popup_mode_var.get()),
            'show_region_overlay': bool(self.overlay_visible),
            'active_section': self.active_section,
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
        self.popup_mode_var.set(bool(self.settings.get('show_answer_popup', True)))
        self.overlay_visible = bool(self.settings.get('show_region_overlay', True))
        self._on_backend_change()
        self._on_image_mode_change()
        self._apply_overlay_visibility()

        sec = self.settings.get('active_section')
        if sec in self._section_frames:
            self._switch_section(sec)

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
            detail = f"{chunk_count} chunk  /  {total_pages}페이지"
            if failed:
                detail += f"  /  실패 {len(failed)}페이지"
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

    # ── Material management ──────────────────────────────────────────────────
    def _add_material(self):
        paths = filedialog.askopenfilenames(
            title="학습 자료 선택",
            filetypes=[
                ("PDF",    "*.pdf"),
                ("이미지", "*.png *.jpg *.jpeg *.bmp"),
                ("전체",   "*.*"),
            ]
        )
        if not paths:
            return
        for path in paths:
            name = os.path.basename(path)
            self._upsert_material_meta(name, path, 'learning', 0, [], 0)
            self._set_status(f"'{name}' 불러오는 중...")
            threading.Thread(target=self._load_material, args=(path, name), daemon=True).start()

    def _load_material(self, path, name):
        try:
            page_images = self._load_material_pages(path)
        except Exception as e:
            self._safe_after(lambda: self._set_status(f"'{name}' 추출 오류: {e}"))
            self._upsert_material_meta(name, path, 'error', 0, [], 0)
            return

        try:
            total_pages = len(page_images)
            failed_pages = []
            stored_chunks = []
            total_chunks = 0
            for idx, image in enumerate(page_images, start=1):
                self._set_status(f"'{name}' {idx}/{total_pages}페이지 처리 중...")
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
                    self._set_status(f"'{n}' 학습 완료 ({tp}페이지 중 실패 {len(fp)}페이지: {', '.join(map(str, fp))})"))
            else:
                self._safe_after(lambda n=name, tc=total_chunks: self._set_status(f"'{n}' 학습 완료 ({tc} chunk)"))
        except Exception as e:
            self._safe_after(lambda: self._set_status(f"'{name}' 학습 실패: {e}"))
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
            "설명이나 요약 없이 원문을 가깝게 정리해줘."
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
        icon = {'learning': '…', 'ready': '●', 'error': '×'}
        ready_count = 0
        for m in self.study_materials:
            st     = m.get('status', 'learning')
            mark   = icon.get(st, '?')
            detail = m.get('detail', f"{m.get('chunk_count', 0)} chunk")
            self.mat_listbox.insert(tk.END, f"   {mark}    {m['name']}       {detail}")
            if st == 'ready':
                ready_count += 1
        total = len(self.study_materials)
        count_text = f"{total}개 · 준비됨 {ready_count}"
        count_color = SUCCESS if ready_count > 0 else (WARNING if total > 0 else TEXT_MUTE)
        self.mat_count_lbl.config(text=count_text, fg=count_color)

        # dashboard info update
        if hasattr(self, 'db_mat_info'):
            self.db_mat_info.config(text=f'{total}개 ({ready_count} 준비)')

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
        return "아래 학습자료 조각을 참고해서 답해줘.\n\n" + "\n\n──────────\n\n".join(parts) + "\n\n"

    # ── Region selection ─────────────────────────────────────────────────────
    def _select_region(self):
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
        self.root.withdraw()
        time.sleep(0.3)
        sel = RegionSelector(self._on_region_selected)
        self.root.wait_window(sel.root)
        self.root.deiconify()

    def _on_region_selected(self, region):
        self.region = region
        self.last_img = None
        self.last_signature = None
        self._apply_overlay_visibility()
        self._update_region_label(region)

    def _on_overlay_change(self, region):
        self.region   = region
        self.last_img = None
        self.last_signature = None
        self._update_region_label(region)

    def _apply_overlay_visibility(self):
        if hasattr(self, 'overlay_btn'):
            self.overlay_btn.config(
                text='영역 숨기기' if self.overlay_visible else '영역 보이기',
                state='normal' if self.region else 'disabled',
            )
        if self.overlay and not self.overlay_visible:
            self.overlay.destroy()
            self.overlay = None
        if self.overlay_visible and self.region and self.overlay is None:
            self.overlay = RegionOverlay(self.region, self._on_overlay_change)

    def _toggle_overlay_visibility(self):
        if not self.region:
            return
        self.overlay_visible = not self.overlay_visible
        self._apply_overlay_visibility()
        self._save_settings()

    def _update_region_label(self, region):
        x1, y1, x2, y2 = region
        w, h = x2 - x1, y2 - y1
        info = f"({x1}, {y1}) → ({x2}, {y2})    {w} × {h}px"
        if hasattr(self, 'region_label'):
            self.region_label.config(text=info, fg=SUCCESS)
        if hasattr(self, 'tb_region_lbl'):
            self.tb_region_lbl.config(text=f"{w}×{h}", fg=TEXT_SEC)
        if hasattr(self, 'db_region_info'):
            self.db_region_info.config(text=f'{w}×{h}', fg=SUCCESS)

    # ── Toggle / stop ────────────────────────────────────────────────────────
    def _toggle(self):
        if not self.running:
            if not self.region:
                messagebox.showwarning("알림", "화면 영역을 먼저 선택하세요.")
                return
            if not self._is_ollama() and not self._get_keys():
                messagebox.showwarning("알림", "Gemini API Key를 입력하세요.")
                return
            self.running = True
            self._stop_event.clear()
            self.key_idx = 0; self.model_idx = 0
            self.start_btn.set_colors(DANGER, '#fff5f5', '#b91c1c', '#fff5f5')
            self.start_btn.config(text='자동 풀이 중지')
            if hasattr(self, 'side_status_dot'):
                self.side_status_dot.config(fg=SUCCESS)
            if hasattr(self, 'tb_state_lbl'):
                self.tb_state_lbl.config(text='● 자동 풀이 중', fg=SUCCESS)
            self.last_img = None
            self.last_signature = None
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
        else:
            self.running = False
            self._stop_event.set()
            self.start_btn.set_colors(ACCENT, BG, ACCENT_DARK, BG)
            self.start_btn.config(text='자동 풀이 시작')
            if hasattr(self, 'side_status_dot'):
                self.side_status_dot.config(fg=TEXT_DIM)
            if hasattr(self, 'tb_state_lbl'):
                self.tb_state_lbl.config(text='● 대기 중', fg=TEXT_MUTE)
            self._set_status("중지됨")

    # ── Force solve ──────────────────────────────────────────────────────────
    def _force_solve(self):
        if not self.region:
            messagebox.showwarning("알림", "화면 영역을 먼저 선택하세요.")
            return
        if not self._is_ollama() and not self._get_keys():
            messagebox.showwarning("알림", "Gemini API Key를 입력하세요.")
            return
        if self.querying:
            self._set_status("이전 요청 처리 중입니다. 잠시 후 다시 시도하세요.")
            return
        def run():
            if not self._query_lock.acquire(blocking=False):
                self._set_status("이전 요청 처리 중입니다. 잠시 후 다시 시도하세요.")
                return
            self.querying = True
            try:
                img = ImageGrab.grab(bbox=self.region)
                self._schedule_preview(img)
                self._set_status("강제 풀이 요청 중...")
                answer, signature = self._query_ai(img)
                if not answer:
                    return
                if self._is_duplicate_capture(signature):
                    self._set_status("같은 문제라서 재전송하지 않음")
                    return
                self.last_signature = signature
                self._safe_after(lambda a=answer: self._show_answer(a))
                self.last_img = ImageGrab.grab(bbox=self.region)
                self._set_status("답변 완료. 다음 문제 대기 중...")
            except Exception as e:
                self._set_status(f"오류: {e}")
            finally:
                self.querying = False
                self._query_lock.release()
        threading.Thread(target=run, daemon=True).start()

    # ── Monitor loop ─────────────────────────────────────────────────────────
    def _monitor_loop(self):
        while self.running and not self._stop_event.is_set():
            try:
                img  = ImageGrab.grab(bbox=self.region)
                diff = self._img_diff(img)
                if diff >= CHANGE_THRESHOLD:
                    if self.querying:
                        self._set_status("변화 감지됨, 이전 요청 처리 중이라 대기합니다.")
                    else:
                        self.last_img = img
                        self._set_status(f"변화 {diff:.1f}% 감지, AI에 전송 중...")
                        self._schedule_preview(img)
                        if self._query_lock.acquire(blocking=False):
                            self.querying = True
                            try:
                                answer, signature = self._query_ai(img)
                                if answer and not self._is_duplicate_capture(signature):
                                    self.last_signature = signature
                                    self._safe_after(lambda a=answer: self._show_answer(a))
                                    self.last_img = ImageGrab.grab(bbox=self.region)
                                    self._set_status("답변 완료. 다음 문제 대기 중...")
                                elif answer:
                                    self._set_status("같은 문제라서 재전송하지 않음")
                            finally:
                                self.querying = False
                                self._query_lock.release()
                else:
                    self._set_status(f"대기 중... (변경률 {diff:.1f}% / 기준 {CHANGE_THRESHOLD}%)")
            except Exception as e:
                self.querying = False
                self._set_status(f"오류: {e}")
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
        thumb.thumbnail((240, 110))
        tk_img = ImageTk.PhotoImage(thumb)
        self.preview_lbl.config(image=tk_img, text='')
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

    # ── AI query ─────────────────────────────────────────────────────────────
    def _query_ai(self, img):
        self._reset_daily_counts_if_needed()
        prompt_suffix = self.prompt_entry.get().strip()
        ready_count   = sum(1 for m in self.study_materials if m.get('status') == 'ready')
        mat_note      = f" (학습자료 {ready_count}개)" if ready_count else ""
        image_mode    = self.image_mode_var.get()

        ocr_text, ocr_conf = '', 0.0
        if not image_mode:
            self._set_status("OCR 텍스트 추출 중...")
            try:
                ocr_text, ocr_conf = self._ocr_extract(img)
            except Exception as ocr_err:
                self._set_status(f"OCR 오류 — 이미지로 전환 ({type(ocr_err).__name__})")

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
            self._set_status("저장된 답변 사용 중...")
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
        system_msg = system_ctx if system_ctx else "너는 문제 풀이를 도와주는 학습 도우미야."

        if not use_image:
            self._set_status(f"OCR 결과를 [{model}]에 전송 중{mat_note}...")
            messages = [
                {'role': 'system', 'content': system_msg},
                {'role': 'user',   'content': f"다음 문제를 풀어줘.\n\n{ocr_text}\n\n{prompt_suffix}"}
            ]
        else:
            mode_str = "수식/이미지 모드" if self.image_mode_var.get() else "OCR 실패"
            self._set_status(f"{mode_str}로 [{model}]에 이미지 전송 중{mat_note}...")
            b64 = self._img_to_b64(img)
            messages = [
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                    {'type': 'text', 'text': f"이 이미지에 있는 문제를 풀어줘. {prompt_suffix}"}
                ]}
            ]

        self._safe_after(lambda: self.conn_lbl.config(
            text=f"Ollama  ·  {model}{mat_note}", fg=SUCCESS))
        self._safe_after(lambda: self.tb_engine_lbl.config(
            text=f"Ollama · {model}", fg=TEXT_SEC))
        if hasattr(self, 'db_engine_info'):
            self._safe_after(lambda: self.db_engine_info.config(text=f'Ollama · {model}'))
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
            mode_str = "수식/이미지 모드" if self.image_mode_var.get() else "OCR 실패"
            self._set_status(f"{mode_str}로 Gemini 이미지 전송 중{mat_note}...")
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
                self._set_status("이 키의 일일 한도를 넘어 다음 키로 전환합니다.")
                self.key_idx += 1; self.model_idx = 0
                continue

            self._set_key_status(key, model)
            mode_label = "수식" if self.image_mode_var.get() else ("OCR" if not use_image else "이미지")
            self._set_status(f"[{model}] 요청 전송 중{mat_note} ({mode_label})...")
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
                    text=f"Gemini  ·  {model}{mat_note}", fg=SUCCESS))
                self._safe_after(lambda: self.tb_engine_lbl.config(
                    text=f"Gemini · {model}", fg=TEXT_SEC))
                if hasattr(self, 'db_engine_info'):
                    self._safe_after(lambda: self.db_engine_info.config(text=f'Gemini · {model}'))
                return resp.text

            except Exception as e:
                err  = str(e)
                errl = err.lower()
                idx  = self.key_idx % len(keys) + 1
                _bad_key = (
                    'API_KEY_INVALID' in err or
                    'api key not valid' in errl or
                    'invalid api key' in errl or
                    'permission_denied' in errl or
                    '401' in err or
                    ('400' in err and 'key' in errl)
                )
                if _bad_key:
                    self._set_status(f"{idx}번 API 키 오류입니다. 값을 확인해 주세요.")
                    self._safe_after(lambda i=idx: self.conn_lbl.config(
                        text=f"{i}번 키 오류", fg=DANGER))
                    self.key_idx += 1; self.model_idx = 0
                elif '503' in err or 'UNAVAILABLE' in err:
                    self._set_status("서버 과부하, 15초 후 재시도합니다.")
                    time.sleep(15)
                elif '429' in err or 'RESOURCE_EXHAUSTED' in err:
                    if 'quota' in errl or 'daily' in errl or 'limit: 0' in errl:
                        self.daily_counts[key] = RPD_LIMIT
                        self._update_count_label()
                        self.key_idx += 1; self.model_idx = 0
                        self._set_status(f"{idx}번 키 한도 도달, 다음 키로 전환합니다.")
                        time.sleep(3)
                    else:
                        self._set_status("RPM 초과, 60초 대기합니다.")
                        time.sleep(60)
                else:
                    self._safe_after(lambda: self.conn_lbl.config(
                        text=f"오류: {err[:50]}", fg=DANGER))
                    raise

        self._set_status("모든 Gemini 키가 한도 또는 오류 상태입니다.")
        return None

    # ── Backend toggle ───────────────────────────────────────────────────────
    def _on_backend_change(self):
        if not hasattr(self, 'ollama_frame'):
            return
        if self.backend_var.get() == 'ollama':
            self.gemini_frame.pack_forget()
            self.ollama_frame.pack(fill='x', pady=(0, 0))
            if hasattr(self, 'tb_engine_lbl'):
                self.tb_engine_lbl.config(
                    text=f'Ollama · {self._get_model()}', fg=TEXT_SEC)
            if hasattr(self, 'db_engine_info'):
                self.db_engine_info.config(text=f'Ollama · {self._get_model()}')
        else:
            self.ollama_frame.pack_forget()
            self.gemini_frame.pack(fill='x', pady=(0, 0))
            if hasattr(self, 'tb_engine_lbl'):
                self.tb_engine_lbl.config(text='Gemini', fg=TEXT_SEC)
            if hasattr(self, 'db_engine_info'):
                self.db_engine_info.config(text='Gemini')

    # ── Answer display ───────────────────────────────────────────────────────
    def _show_answer(self, text):
        self.output_txt.config(state='normal')
        self.output_txt.delete('1.0', tk.END)
        self.output_txt.insert(tk.END, text)
        self.output_txt.config(state='disabled', fg=TEXT_PRI)

        if not self.popup_mode_var.get():
            return

        if hasattr(self, '_answer_win') and self._answer_win.winfo_exists():
            self._answer_win.destroy()

        win = tk.Toplevel(self.root)
        win.title('NEURAL_SOLVER — Solution')
        win.configure(bg=BG)
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
        win.geometry(f'{win_w}x{win_h}+{px}+{py}')

        pop_hdr = tk.Frame(win, bg=ACCENT, height=34)
        pop_hdr.pack(fill='x')
        pop_hdr.pack_propagate(False)
        tk.Label(pop_hdr, text='SOLUTION',
                 font=(FONT_UI, 10, 'bold'), fg=BG, bg=ACCENT
                 ).pack(side='left', padx=14, pady=8)
        tk.Label(pop_hdr, text=time.strftime('%H:%M:%S'),
                 font=(FONT_UI, 9), fg=BG, bg=ACCENT
                 ).pack(side='right', padx=14, pady=8)

        txt = scrolledtext.ScrolledText(win, wrap=tk.WORD,
            font=F_OUTPUT, bg=SURFACE, fg=TEXT_PRI,
            insertbackground=ACCENT, relief='flat', bd=0,
            padx=18, pady=14)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, text)
        txt.config(state='disabled')

        btn_bar = tk.Frame(win, bg=BG)
        btn_bar.pack(fill='x', padx=14, pady=10)
        HoverButton(btn_bar, text='답변 복사', font=F_SMALL_B,
            bg=ACCENT_BG, fg=ACCENT,
            hover_bg=ACCENT, hover_fg=BG,
            padx=14, pady=6,
            command=lambda: (win.clipboard_clear(), win.clipboard_append(text))
        ).pack(side='right')

    def _set_status(self, msg):
        def _do():
            self._log_lines.insert(0, f'{time.strftime("%H:%M:%S")}   {msg}')
            self._log_lines = self._log_lines[:50]
            self._refresh_logs()
            seq, pct, color = self._parse_status(msg)
            if hasattr(self, 'seq_lbl'):
                self.seq_lbl.config(text=seq, fg=color)
            self._set_progress(pct, color)
        self._safe_after(_do)


# ═══ ENTRY ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    AISolverApp()
