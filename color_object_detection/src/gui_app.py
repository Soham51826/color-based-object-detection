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
 Status Dashboard  |  Colour Preset  |  HSV Sliders
 [Start] [Pause] [Reset Calibration] [Snapshot]

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

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    _USE_CTK = True
except Exception:
    # customtkinter may be installed but require Python 3.7+ (uses
    # `from __future__ import annotations`).  Fall back to plain Tkinter.
    _USE_CTK = False

import tkinter as tk
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

_COLOR_NAMES = list(config.COLOR_PRESETS.keys()) + ["Custom"]


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

    def __init__(self, source, frame_queue, stop_event):
        super(_CaptureWorker, self).__init__(daemon=True, name="CaptureWorker")
        self._source     = source
        self._queue      = frame_queue
        self._stop_event = stop_event
        self._paused     = threading.Event()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def run(self):
        try:
            cap = open_capture(self._source)
        except RuntimeError as exc:
            logger.error("CaptureWorker: %s", exc)
            return

        while not self._stop_event.is_set():
            if self._paused.is_set():
                threading.Event().wait(0.05)
                continue

            ok, frame = cap.read()
            if not ok:
                logger.warning("CaptureWorker: cap.read() returned False.")
                break

            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass  # drop frame if UI is slow

        cap.release()
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

        self._root.title("Color-Based Object Detection -- MPSTME IVP")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---- Tkinter variables -----------------------------------------
        self._selected_color   = tk.StringVar(value="Red")
        self._h_min_var        = tk.IntVar(value=0)
        self._h_max_var        = tk.IntVar(value=10)
        self._s_min_var        = tk.IntVar(value=100)
        self._s_max_var        = tk.IntVar(value=255)
        self._v_min_var        = tk.IntVar(value=100)
        self._v_max_var        = tk.IntVar(value=255)
        self._min_area_var     = tk.IntVar(value=config.MIN_CONTOUR_AREA)
        self._fps_label_var    = tk.StringVar(value="FPS: --")
        self._status_label_var = tk.StringVar(value="Status: IDLE")
        self._centroid_var     = tk.StringVar(value="Centroid: --, --")
        self._area_var         = tk.StringVar(value="Area: --")
        self._is_running       = False
        self._is_paused        = False

        # ---- Tracker ---------------------------------------------------
        self._tracker = ColorTracker()

        # ---- Capture threading -----------------------------------------
        self._frame_queue = queue.Queue(maxsize=2)
        self._stop_event  = threading.Event()
        self._worker      = None

        # ---- Cached PhotoImages (must hold references to prevent GC) ---
        self._photo_main = None
        self._photo_mask = None
        self._last_annotated = None

        # ---- Active mask -----------------------------------------------
        self._active_mask_name = "Red"

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
        )
        self._canvas_main.pack(side="left", padx=PAD)

        self._canvas_mask = tk.Canvas(
            video_frame, width=_DISPLAY_W, height=_DISPLAY_H,
            bg="#1a1a2e", highlightthickness=2, highlightbackground="#e94560",
        )
        self._canvas_mask.pack(side="left", padx=PAD)

        self._canvas_main.create_text(
            _DISPLAY_W // 2, _DISPLAY_H // 2,
            text="Camera Feed\n(press Start)", fill="#555577",
            font=("Consolas", 14), tags="placeholder",
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

        for var in [self._fps_label_var, self._status_label_var,
                    self._centroid_var, self._area_var]:
            lbl = self._make_label(dash_frame, textvariable=var)
            lbl.pack(anchor="w", padx=6, pady=2)

        # --- Colour selector ---
        color_frame = self._make_labelframe(ctrl_frame, "Colour Preset")
        color_frame.pack(side="left", fill="y", padx=(0, PAD), pady=PAD)

        if _USE_CTK:
            self._color_menu = ctk.CTkOptionMenu(
                color_frame,
                variable=self._selected_color,
                values=_COLOR_NAMES,
                command=self._on_color_select,
                width=160,
            )
        else:
            self._color_menu = tk.OptionMenu(
                color_frame,
                self._selected_color,
                *_COLOR_NAMES,
                command=self._on_color_select,
            )
        self._color_menu.pack(padx=8, pady=8)

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
            ("Start Feed",      "#2e7d32", self._on_start),
            ("Pause / Resume",  "#1565c0", self._on_pause),
            ("Reset Calibration", "#6a1b9a", self._on_reset),
            ("Snapshot",        "#bf360c", self._on_snapshot),
        ]
        for text, color, cmd in btn_defs:
            if _USE_CTK:
                btn = ctk.CTkButton(
                    btn_frame, text=text, command=cmd,
                    fg_color=color, hover_color=color,
                    width=160, corner_radius=8,
                )
            else:
                btn = tk.Button(
                    btn_frame, text=text, command=cmd,
                    bg=color, fg="white", relief="flat",
                    padx=12, pady=6,
                )
            btn.pack(side="left", padx=6)

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
    # Event handlers
    # ------------------------------------------------------------------

    def _on_start(self):
        if self._is_running:
            return

        self._stop_event.clear()
        self._worker = _CaptureWorker(
            self._source, self._frame_queue, self._stop_event
        )
        self._worker.start()
        self._is_running = True
        self._is_paused  = False
        self._status_label_var.set("Status: RUNNING")
        self._root.after(_POLL_MS, self._poll_frame)
        logger.info("Capture started.")

    def _on_pause(self):
        if not self._is_running:
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
        out_dir = Path("snapshots")
        out_dir.mkdir(exist_ok=True)
        path   = out_dir / "snapshot_{}.png".format(ts)
        cv2.imwrite(str(path), frame)
        msgbox.showinfo("Snapshot saved", "Frame saved to:\n{}".format(path.resolve()))

    def _on_color_select(self, selection):
        if selection != "Custom":
            self._load_preset(selection)
            self._active_mask_name = selection

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
        if selected != "Custom":
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
        mask = result.masks.get(self._active_mask_name)
        if mask is not None:
            mask_disp    = _resize_for_display(mask, _DISPLAY_W, _DISPLAY_H)
            mask_rgb     = cv2.cvtColor(mask_disp, cv2.COLOR_GRAY2RGB)
            self._photo_mask = _np_to_photoimage(mask_rgb)
            self._canvas_mask.delete("placeholder")
            self._canvas_mask.create_image(0, 0, anchor="nw", image=self._photo_mask)

        self._root.after(_POLL_MS, self._poll_frame)

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
        else:
            self._status_label_var.set("Status: LOST")
            self._centroid_var.set("Centroid: --, --")
            self._area_var.set("Area: --")

    # ------------------------------------------------------------------
    # Preset loader
    # ------------------------------------------------------------------

    def _load_preset(self, color_name):
        if color_name == "Custom":
            return
        preset = config.COLOR_PRESETS.get(color_name)
        if preset is None:
            return

        lower = preset["lower1"]
        upper = preset["upper1"]

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
