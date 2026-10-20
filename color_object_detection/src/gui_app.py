"""
gui_app.py
==========
Modern desktop GUI for the Color-Based Object Detection System.

Architecture
------------
Built with Tkinter + CustomTkinter (falls back to standard Tkinter if ctk is
unavailable) for a clean, dark-themed UI.  Pillow converts OpenCV BGR frames
into Tkinter-compatible PhotoImage objects for display.

Layout (left to right)
-----------------------
 [ Live Camera Feed + Overlays ]  [ Binary Mask View ]
 Status Dashboard  |  Colour Preset + Feature Toggles  |  HSV Sliders
 [Start] [Pause] [Reset Calibration] [Snapshot]

New features
------------
- 9-colour + Multi-Color preset selector
- CLAHE / Bilateral / EMA / Trail / Rotated-BBox feature toggles
- Click-to-Pick (Eyedropper): click the camera canvas to sample a 5x5
  pixel region, convert to HSV, and auto-set the sliders to ± tolerances.

Threading model
---------------
- The Tkinter event loop runs on the main thread.
- A separate threading.Thread runs the OpenCV capture loop, posting frames
  to a queue.Queue.
- The Tkinter after() callback dequeues frames and refreshes the canvas.

Usage
-----
    python -m src.gui_app                      # webcam index 0
    python -m src.gui_app --source 1           # webcam index 1
    python -m src.gui_app --source video.mp4   # video file
"""

import argparse
import datetime
import os
import queue
import threading
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image, ImageTk  # type: ignore

try:
    import customtkinter as ctk  # type: ignore
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    _USE_CTK = True
except Exception:
    # customtkinter may be installed but require Python 3.7+ (uses
    # `from __future__ import annotations`).  Fall back to plain Tkinter.
    _USE_CTK = False

import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as msgbox

from . import config
from .tracker import ColorTracker, FrameResult, open_capture
from .utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISPLAY_W = 480
_DISPLAY_H = 360
_POLL_MS   = 16

# All 9 chromatic presets + multi-colour + custom
_COLOR_NAMES = list(config.COLOR_PRESETS.keys()) + ["Multi-Color", "Custom"]

# Eyedropper tolerance (HSV ± values applied around sampled pixel)
_EYE_H_TOL = 10
_EYE_S_TOL = 50
_EYE_V_TOL = 50


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _resize_for_display(frame, w, h):
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def _np_to_photoimage(rgb_array):
    img = Image.fromarray(rgb_array)
    return ImageTk.PhotoImage(image=img)


# ---------------------------------------------------------------------------
# Capture worker thread
# ---------------------------------------------------------------------------

class _CaptureWorker(threading.Thread):
    """Background thread that reads frames from the video source.

    Parameters
    ----------
    source : int or str
        Camera index or file path.
    frame_queue : queue.Queue
        Thread-safe queue shared with the UI thread (maxsize=2).
    stop_event : threading.Event
        Set this to signal the worker to exit cleanly.
    """

    def __init__(self, source, frame_queue, stop_event, loop_video=True):
        super(_CaptureWorker, self).__init__(daemon=True, name="CaptureWorker")
        self._source     = source
        self._queue      = frame_queue
        self._stop_event = stop_event
        self._loop_video = loop_video
        self._paused     = threading.Event()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def run(self):
        try:
            cap = open_capture(
                self._source, fallback_to_synthetic=True, loop_video=self._loop_video
            )
        except Exception as exc:
            logger.error("CaptureWorker failed to initialize video source: %s", exc)
            return

        if cap is None:
            logger.error("CaptureWorker failed to initialize video source: open_capture returned None")
            return

        # Frame pacing for video file playback
        import time as _time
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1.0 / fps_val if (fps_val and fps_val > 0) else 0.016
        is_video_file = isinstance(self._source, (str, Path))
        last_frame_time = _time.time()

        while not self._stop_event.is_set():
            if self._paused.is_set():
                _time.sleep(0.05)
                continue

            if is_video_file:
                now = _time.time()
                elapsed = now - last_frame_time
                if elapsed < frame_interval:
                    _time.sleep(max(0.001, frame_interval - elapsed))
                last_frame_time = _time.time()

            try:
                ok, frame = cap.read()
            except Exception as exc:
                logger.error("CaptureWorker error reading frame: %s", exc)
                break

            if not ok or frame is None:
                if self._loop_video and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        logger.warning("CaptureWorker: end of stream after rewind.")
                        break
                else:
                    logger.warning("CaptureWorker: cap.read() returned False or empty frame.")
                    break

            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass  # drop frame if UI is slow

        try:
            cap.release()
        except Exception:
            pass
        logger.info("CaptureWorker: exited cleanly.")


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class ColorDetectionApp(object):
    """Top-level Tkinter application for the colour detection system.

    Parameters
    ----------
    source : int or str
        Camera index or video file path to open.
    """

    def __init__(self, source=None):
        if source is None:
            source = config.DEFAULT_CAMERA_INDEX
        self._source = source

        # ---- Root window -----------------------------------------------
        if _USE_CTK:
            self._root = ctk.CTk()
        else:
            self._root = tk.Tk()
            self._root.configure(bg="#1e1e2e")

        self._root.title("Color-Based Object Detection and Tracking System")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---- Tkinter variables -----------------------------------------
        self._selected_color   = tk.StringVar(value="Red")
        self._h_min_var        = tk.IntVar(value=0)
        self._h_max_var        = tk.IntVar(value=10)
        self._s_min_var        = tk.IntVar(value=110)
        self._s_max_var        = tk.IntVar(value=255)
        self._v_min_var        = tk.IntVar(value=80)
        self._v_max_var        = tk.IntVar(value=255)
        self._min_area_var     = tk.IntVar(value=config.MIN_CONTOUR_AREA)
        initial_source_text = "Source: Live Webcam"
        if isinstance(self._source, (str, Path)):
            initial_source_text = "Source: {} (Looping)".format(Path(str(self._source)).name)
        self._source_label_var = tk.StringVar(value=initial_source_text)
        self._fps_label_var    = tk.StringVar(value="FPS: --")
        self._status_label_var = tk.StringVar(value="Status: IDLE")
        self._centroid_var     = tk.StringVar(value="Centroid: --, --")
        self._area_var         = tk.StringVar(value="Area: --")
        self._solidity_var     = tk.StringVar(value="Solidity: --")
        self._circ_var         = tk.StringVar(value="Circularity: --")
        self._is_running       = False
        self._is_paused        = False

        # ---- Feature toggle BooleanVars --------------------------------
        self._var_clahe    = tk.BooleanVar(value=True)
        self._var_ema      = tk.BooleanVar(value=True)
        self._var_trail    = tk.BooleanVar(value=True)
        self._var_rotbbox  = tk.BooleanVar(value=False)

        # ---- Tracker ---------------------------------------------------
        self._tracker = ColorTracker(
            use_clahe=True,
            use_bilateral=False,
            use_ema_smoothing=True,
            show_trail=True,
            use_rotated_bbox=False,
        )

        # ---- Capture threading -----------------------------------------
        self._frame_queue = queue.Queue(maxsize=2)
        self._stop_event  = threading.Event()
        self._worker      = None

        # ---- Cached PhotoImages (must hold references to prevent GC) ---
        self._photo_main = None
        self._photo_mask = None
        self._last_annotated     = None
        self._last_raw_frame     = None  # kept for eyedropper sampling

        # ---- Active mask / mode ----------------------------------------
        self._active_mask_name = "Red"
        self._multi_color_mode = False

        # ---- Build UI --------------------------------------------------
        self._build_ui()
        self._load_preset("Red")

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        PAD = 8

        # === Top: dual video panels ====================================
        video_frame = self._make_frame(self._root)
        video_frame.pack(side="top", padx=PAD, pady=PAD)

        self._canvas_main = tk.Canvas(
            video_frame, width=_DISPLAY_W, height=_DISPLAY_H,
            bg="#1a1a2e", highlightthickness=2, highlightbackground="#4a90d9",
            cursor="crosshair",
        )
        self._canvas_main.pack(side="left", padx=PAD)
        # Bind eyedropper click
        self._canvas_main.bind("<Button-1>", self._on_canvas_click)

        self._canvas_mask = tk.Canvas(
            video_frame, width=_DISPLAY_W, height=_DISPLAY_H,
            bg="#1a1a2e", highlightthickness=2, highlightbackground="#e94560",
        )
        self._canvas_mask.pack(side="left", padx=PAD)

        self._canvas_main.create_text(
            _DISPLAY_W // 2, _DISPLAY_H // 2,
            text="Camera Feed\n(press Start)\n\nClick to pick colour",
            fill="#555577",
            font=("Consolas", 13), tags="placeholder",
        )
        self._canvas_mask.create_text(
            _DISPLAY_W // 2, _DISPLAY_H // 2,
            text="Mask Preview\n(press Start)", fill="#554444",
            font=("Consolas", 14), tags="placeholder",
        )

        # === Middle: controls ==========================================
        ctrl_frame = self._make_frame(self._root)
        ctrl_frame.pack(side="top", fill="x", padx=PAD, pady=(0, PAD))

        # --- Status dashboard ---
        dash_frame = self._make_labelframe(ctrl_frame, "Dashboard")
        dash_frame.pack(side="left", fill="y", padx=(0, PAD), pady=PAD)

        for var in [self._source_label_var, self._fps_label_var, self._status_label_var,
                    self._centroid_var, self._area_var,
                    self._solidity_var, self._circ_var]:
            lbl = self._make_label(dash_frame, textvariable=var)
            lbl.pack(anchor="w", padx=6, pady=2)

        # --- Colour selector + feature toggles ---
        color_frame = self._make_labelframe(ctrl_frame, "Colour Preset")
        color_frame.pack(side="left", fill="y", padx=(0, PAD), pady=PAD)

        if _USE_CTK:
            self._color_menu = ctk.CTkOptionMenu(
                color_frame,
                variable=self._selected_color,
                values=_COLOR_NAMES,
                command=self._on_color_select,
                width=180,
            )
        else:
            self._color_menu = tk.OptionMenu(
                color_frame,
                self._selected_color,
                *_COLOR_NAMES,
                command=self._on_color_select,
            )
        self._color_menu.pack(padx=8, pady=8)

        # Feature checkboxes
        toggle_defs = [
            ("Enable CLAHE Lighting Stabilization", self._var_clahe,   self._on_toggle_clahe),
            ("Enable Kalman/EMA Centroid Smoothing", self._var_ema,    self._on_toggle_ema),
            ("Show Motion Trajectory Trail",         self._var_trail,  self._on_toggle_trail),
            ("Use Rotated Bounding Box",             self._var_rotbbox, self._on_toggle_rotbbox),
        ]
        for label_text, bvar, cmd in toggle_defs:
            if _USE_CTK:
                cb = ctk.CTkCheckBox(
                    color_frame, text=label_text, variable=bvar, command=cmd,
                )
            else:
                cb = tk.Checkbutton(
                    color_frame, text=label_text, variable=bvar, command=cmd,
                    bg="#1e1e2e", fg="#ccccff", selectcolor="#333355",
                    font=("Consolas", 8), activebackground="#1e1e2e",
                    activeforeground="#ffffff",
                )
            cb.pack(anchor="w", padx=8, pady=2)

        # --- HSV sliders ---
        slider_frame = self._make_labelframe(ctrl_frame, "HSV Calibration")
        slider_frame.pack(side="left", fill="both", expand=True, pady=PAD)

        sliders_cfg = [
            ("H Min",    self._h_min_var,    0, 180),
            ("H Max",    self._h_max_var,    0, 180),
            ("S Min",    self._s_min_var,    0, 255),
            ("S Max",    self._s_max_var,    0, 255),
            ("V Min",    self._v_min_var,    0, 255),
            ("V Max",    self._v_max_var,    0, 255),
            ("Min Area", self._min_area_var, 50, 10000),
        ]

        for row_idx, (name, var, lo, hi) in enumerate(sliders_cfg):
            lbl = self._make_label(slider_frame, text="{}:".format(name))
            lbl.grid(row=row_idx, column=0, sticky="w", padx=6, pady=2)

            if _USE_CTK:
                slider = ctk.CTkSlider(
                    slider_frame, from_=lo, to=hi,
                    variable=var, number_of_steps=hi - lo,
                    command=lambda _val: self._on_slider_change(),
                    width=200,
                )
            else:
                slider = tk.Scale(
                    slider_frame, from_=lo, to=hi,
                    orient="horizontal", variable=var, length=200,
                    command=lambda _val: self._on_slider_change(),
                    bg="#1e1e2e", fg="white", troughcolor="#333355",
                    highlightthickness=0,
                )
            slider.grid(row=row_idx, column=1, padx=6, pady=2)

            val_lbl = self._make_label(slider_frame, textvariable=var, width=5)
            val_lbl.grid(row=row_idx, column=2, padx=4)

        # === Bottom: action buttons ====================================
        btn_frame = self._make_frame(self._root)
        btn_frame.pack(side="top", pady=(0, PAD))

        btn_defs = [
            ("📹 Live Webcam",       "#2e7d32", self._on_live_webcam),
            ("📂 Load Video (.mp4)", "#00838f", self._on_load_video),
            ("⏸ Pause / Resume",     "#1565c0", self._on_pause),
            ("↺ Reset Calibration",   "#6a1b9a", self._on_reset),
            ("📸 Snapshot",           "#bf360c", self._on_snapshot),
        ]
        for text, color, cmd in btn_defs:
            if _USE_CTK:
                btn = ctk.CTkButton(
                    btn_frame, text=text, command=cmd,
                    fg_color=color, hover_color=color,
                    width=150, corner_radius=8,
                )
            else:
                btn = tk.Button(
                    btn_frame, text=text, command=cmd,
                    bg=color, fg="white", relief="flat",
                    padx=10, pady=6,
                )
            btn.pack(side="left", padx=5)

    # ------------------------------------------------------------------
    # Widget factory helpers
    # ------------------------------------------------------------------

    def _make_frame(self, parent):
        if _USE_CTK:
            return ctk.CTkFrame(parent)
        return tk.Frame(parent, bg="#1e1e2e")

    def _make_labelframe(self, parent, text):
        if _USE_CTK:
            return ctk.CTkFrame(parent)
        return tk.LabelFrame(
            parent, text=text, bg="#1e1e2e", fg="#aaaacc",
            font=("Consolas", 9, "bold"), padx=4, pady=4,
        )

    def _make_label(self, parent, **kwargs):
        if _USE_CTK:
            return ctk.CTkLabel(parent, **kwargs)
        return tk.Label(
            parent, bg="#1e1e2e", fg="#ccccff",
            font=("Consolas", 9), **kwargs
        )

    # ------------------------------------------------------------------
    # Feature toggle handlers
    # ------------------------------------------------------------------

    def _on_toggle_clahe(self):
        self._tracker.use_clahe = self._var_clahe.get()

    def _on_toggle_ema(self):
        self._tracker.use_ema_smoothing = self._var_ema.get()
        if not self._tracker.use_ema_smoothing:
            for state in self._tracker._color_states.values():
                state.reset()

    def _on_toggle_trail(self):
        self._tracker.show_trail = self._var_trail.get()

    def _on_toggle_rotbbox(self):
        self._tracker.use_rotated_bbox = self._var_rotbbox.get()

    # ------------------------------------------------------------------
    # Eyedropper: click-to-pick colour from camera canvas
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event):
        """Sample a 5x5 pixel region at the click position and auto-set sliders."""
        if self._last_raw_frame is None:
            return

        frame = self._last_raw_frame
        fh, fw = frame.shape[:2]

        # Map canvas coordinates to frame coordinates
        cx = int(event.x * fw / _DISPLAY_W)
        cy = int(event.y * fh / _DISPLAY_H)

        # 5x5 neighbourhood (clamped to frame boundaries)
        x1 = max(0, cx - 2)
        y1 = max(0, cy - 2)
        x2 = min(fw, cx + 3)
        y2 = min(fh, cy + 3)

        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return

        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h_val = int(np.median(hsv_region[:, :, 0]))
        s_val = int(np.median(hsv_region[:, :, 1]))
        v_val = int(np.median(hsv_region[:, :, 2]))

        h_lo = max(0,   h_val - _EYE_H_TOL)
        h_hi = min(180, h_val + _EYE_H_TOL)
        s_lo = max(0,   s_val - _EYE_S_TOL)
        s_hi = min(255, s_val + _EYE_S_TOL)
        v_lo = max(0,   v_val - _EYE_V_TOL)
        v_hi = min(255, v_val + _EYE_V_TOL)

        self._h_min_var.set(h_lo)
        self._h_max_var.set(h_hi)
        self._s_min_var.set(s_lo)
        self._s_max_var.set(s_hi)
        self._v_min_var.set(v_lo)
        self._v_max_var.set(v_hi)

        # Force Custom mode so the picked range is applied
        self._selected_color.set("Custom")
        self._on_slider_change()

        logger.info(
            "Eyedropper: pixel=(%d,%d) HSV=(%d,%d,%d) -> H[%d-%d] S[%d-%d] V[%d-%d]",
            cx, cy, h_val, s_val, v_val, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _switch_source(self, new_source, is_video=False):
        """Safely stop existing capture worker, switch source, and start worker."""
        if self._worker is not None and self._worker.is_alive():
            self._stop_event.set()
            self._worker.join(timeout=1.0)

        # Clear queued frames
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._source = new_source
        self._stop_event.clear()

        if is_video:
            filename = Path(str(new_source)).name
            self._source_label_var.set("Source: {} (Looping)".format(filename))
        else:
            self._source_label_var.set("Source: Live Webcam")

        self._tracker.reset()

        self._worker = _CaptureWorker(
            self._source, self._frame_queue, self._stop_event, loop_video=True
        )
        self._worker.start()

        was_running = self._is_running
        self._is_running = True
        self._is_paused  = False
        self._status_label_var.set("Status: RUNNING")

        if not was_running:
            self._root.after(_POLL_MS, self._poll_frame)

        logger.info("Source switched to: %s (is_video=%s)", new_source, is_video)

    def _on_live_webcam(self):
        """Switch video source to the live webcam."""
        self._switch_source(config.DEFAULT_CAMERA_INDEX, is_video=False)

    def _on_load_video(self):
        """Prompt user for a video file (defaulting to benchmark video) and loop playback."""
        possible_dirs = [
            config.BASE_DIR / "assets",
            config.BASE_DIR.parent / "assets",
            Path.cwd() / "assets",
        ]
        initial_dir = str(config.BASE_DIR / "assets")
        initial_file = "color_test_benchmark.mp4"
        for d in possible_dirs:
            if (d / initial_file).exists():
                initial_dir = str(d)
                break

        file_path = filedialog.askopenfilename(
            title="Select Video File (.mp4)",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[
                ("MP4 Video", "*.mp4"),
                ("All Video Files", "*.mp4;*.avi;*.mov;*.mkv"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        self._switch_source(file_path, is_video=True)

    def _on_start(self):
        if self._is_running:
            return
        is_video = isinstance(self._source, (str, Path)) and not str(self._source).isdigit()
        self._switch_source(self._source, is_video=is_video)

    def _on_pause(self):
        if not self._is_running or self._worker is None:
            return
        if self._is_paused:
            self._worker.resume()
            self._is_paused = False
            self._status_label_var.set("Status: RUNNING")
        else:
            self._worker.pause()
            self._is_paused = True
            self._status_label_var.set("Status: PAUSED")

    def _on_reset(self):
        self._tracker.reset()
        self._load_preset(self._selected_color.get())

    def _on_snapshot(self):
        if self._last_annotated is None:
            msgbox.showinfo("Snapshot", "No frame available. Start the feed first.")
            return

        frame  = self._last_annotated.copy()
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = config.SNAPSHOT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path   = out_dir / "snapshot_{}.png".format(ts)
        cv2.imwrite(str(path), frame)
        msgbox.showinfo("Snapshot saved", "Frame saved successfully to:\n{}".format(path.name))

    def _on_color_select(self, selection):
        if selection == "Multi-Color":
            self._multi_color_mode = True
            self._active_mask_name = list(config.COLOR_PRESETS.keys())[0]
        elif selection == "Custom":
            self._multi_color_mode = False
            # Keep active mask name; slider values apply to it
        else:
            self._multi_color_mode = False
            self._load_preset(selection)
            self._active_mask_name = selection
            self._on_slider_change()

    def _on_slider_change(self):
        selected = self._selected_color.get()

        lower1 = np.array([
            self._h_min_var.get(),
            self._s_min_var.get(),
            self._v_min_var.get(),
        ], dtype=np.uint8)
        upper1 = np.array([
            self._h_max_var.get(),
            self._s_max_var.get(),
            self._v_max_var.get(),
        ], dtype=np.uint8)

        target = self._active_mask_name if selected == "Custom" else selected
        if selected not in ("Custom", "Multi-Color"):
            self._active_mask_name = selected

        if target in config.COLOR_PRESETS:
            if target == "Red":
                self._tracker.update_hsv_range(
                    target, lower1, upper1,
                    lower2=config.COLOR_PRESETS["Red"]["lower2"],
                    upper2=config.COLOR_PRESETS["Red"]["upper2"],
                )
            else:
                self._tracker.update_hsv_range(target, lower1, upper1)

        self._tracker.min_contour_area = self._min_area_var.get()

    def _on_close(self):
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._root.destroy()

    # ------------------------------------------------------------------
    # Frame polling callback
    # ------------------------------------------------------------------

    def _poll_frame(self):
        if not self._is_running:
            return

        try:
            frame = self._frame_queue.get_nowait()
        except queue.Empty:
            self._root.after(_POLL_MS, self._poll_frame)
            return

        self._last_raw_frame = frame  # store for eyedropper

        result = self._tracker.process_frame(frame)
        self._last_annotated = result.annotated_frame

        self._update_dashboard(result)

        # Main feed canvas
        display_main = _resize_for_display(result.annotated_frame, _DISPLAY_W, _DISPLAY_H)
        rgb_main     = cv2.cvtColor(display_main, cv2.COLOR_BGR2RGB)
        self._photo_main = _np_to_photoimage(rgb_main)
        self._canvas_main.delete("placeholder")
        self._canvas_main.create_image(0, 0, anchor="nw", image=self._photo_main)

        # Mask canvas
        if self._multi_color_mode:
            # Composite all colour masks side-by-side on the mask canvas
            composite = self._build_composite_mask(result)
            mask_disp    = _resize_for_display(composite, _DISPLAY_W, _DISPLAY_H)
            mask_rgb     = cv2.cvtColor(mask_disp, cv2.COLOR_GRAY2RGB)
        else:
            mask = result.masks.get(self._active_mask_name)
            if mask is not None:
                mask_disp = _resize_for_display(mask, _DISPLAY_W, _DISPLAY_H)
                mask_rgb  = cv2.cvtColor(mask_disp, cv2.COLOR_GRAY2RGB)
            else:
                mask_rgb = None

        if mask_rgb is not None:
            self._photo_mask = _np_to_photoimage(mask_rgb)
            self._canvas_mask.delete("placeholder")
            self._canvas_mask.create_image(0, 0, anchor="nw", image=self._photo_mask)

        self._root.after(_POLL_MS, self._poll_frame)

    def _build_composite_mask(self, result):
        # type: (FrameResult) -> np.ndarray
        """Overlay all active colour masks into a single greyscale image."""
        masks = list(result.masks.values())
        if not masks:
            return np.zeros((480, 640), dtype=np.uint8)
        composite = masks[0].copy()
        for m in masks[1:]:
            composite = cv2.bitwise_or(composite, m)
        return composite

    # ------------------------------------------------------------------
    # Dashboard update
    # ------------------------------------------------------------------

    def _update_dashboard(self, result):
        self._fps_label_var.set("FPS: {:.1f}".format(result.fps))

        active_blobs = [
            b for b in result.blobs
            if b.color_name == self._active_mask_name
        ]

        if active_blobs:
            blob = active_blobs[0]
            self._status_label_var.set("Status: TRACKING")
            cx, cy = blob.centroid if blob.centroid else (0, 0)
            self._centroid_var.set("Centroid: ({}, {})".format(cx, cy))
            self._area_var.set("Area: {} px^2".format(int(blob.area)))
            self._solidity_var.set("Solidity: {:.2f}".format(blob.solidity))
            self._circ_var.set("Circularity: {:.2f}".format(blob.circularity))
        else:
            self._status_label_var.set("Status: LOST")
            self._centroid_var.set("Centroid: --, --")
            self._area_var.set("Area: --")
            self._solidity_var.set("Solidity: --")
            self._circ_var.set("Circularity: --")

    # ------------------------------------------------------------------
    # Preset loader
    # ------------------------------------------------------------------

    def _load_preset(self, color_name):
        if color_name in ("Custom", "Multi-Color"):
            return
        preset = config.COLOR_PRESETS.get(color_name)
        if preset is None:
            return

        lower = preset.get("lower1")
        upper = preset.get("upper1")
        if lower is None or upper is None:
            return

        self._h_min_var.set(int(lower[0]))
        self._h_max_var.set(int(upper[0]))
        self._s_min_var.set(int(lower[1]))
        self._s_max_var.set(int(upper[1]))
        self._v_min_var.set(int(lower[2]))
        self._v_max_var.set(int(upper[2]))

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        logger.info("GUI application starting.")
        self._on_start()
        self._root.mainloop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        prog="python -m src.gui_app",
        description="Color-Based Object Detection -- GUI Application",
    )
    parser.add_argument(
        "--source",
        default=str(config.DEFAULT_CAMERA_INDEX),
        help="Video source: camera index (e.g. 0) or path to a video file.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    app = ColorDetectionApp(source=source)
    app.run()


if __name__ == "__main__":
    main()
