"""
utils.py
========
Shared helper functions for the Color-Based Object Detection System.

Responsibilities
----------------
- Frame annotation (bounding boxes, crosshairs, text labels, rotated bboxes)
- Rolling-window FPS counter
- Image-moment centroid computation
- Contour geometry validators (solidity, circularity, aspect ratio)
- BGR <-> RGB conversion helpers
- Logging configuration factory
- Pipeline timing context manager

All functions are pure (no global state) unless explicitly documented.
"""

import logging
import math
import time
from collections import deque
from typing import Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from . import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name):
    # type: (str) -> logging.Logger
    """Return a consistently configured logger.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class FPSCounter(object):
    """Rolling-average frames-per-second counter.

    Parameters
    ----------
    window : int
        Number of recent frame timestamps to retain.

    Example
    -------
    >>> fps_counter = FPSCounter(window=30)
    >>> for frame in video_stream:
    ...     fps_counter.tick()
    ...     print(fps_counter.fps)
    """

    def __init__(self, window=None):
        if window is None:
            window = config.FPS_SMOOTHING_WINDOW
        self._window = window
        self._timestamps = deque(maxlen=window)

    def tick(self):
        """Record the arrival of one new frame."""
        self._timestamps.append(time.time())

    @property
    def fps(self):
        # type: () -> float
        """Current rolling-average FPS. Returns 0.0 if fewer than 2 frames."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self):
        """Clear all recorded timestamps."""
        self._timestamps.clear()


# ---------------------------------------------------------------------------
# Centroid calculation
# ---------------------------------------------------------------------------

def compute_centroid(contour):
    # type: (np.ndarray) -> Optional[Tuple[int, int]]
    """Compute the spatial centroid of a contour via first-order moments.

    The centroid coordinates are derived from raw image moments:
        x_bar = M10 / M00
        y_bar = M01 / M00

    Parameters
    ----------
    contour : np.ndarray
        Single contour array as returned by cv2.findContours.

    Returns
    -------
    (cx, cy) integer pixel coordinates, or None if M00 == 0.
    """
    moments = cv2.moments(contour)
    m00 = moments["m00"]
    if m00 == 0.0:
        return None
    cx = int(moments["m10"] / m00)
    cy = int(moments["m01"] / m00)
    return (cx, cy)


# ---------------------------------------------------------------------------
# Contour geometry descriptors
# ---------------------------------------------------------------------------

def compute_solidity(contour):
    # type: (np.ndarray) -> float
    """Return the solidity of a contour: area / convex_hull_area.

    Solidity near 1.0 indicates a compact, filled blob.
    Low solidity (e.g. < 0.65) indicates a streak, arc, or fragmented noise.

    Parameters
    ----------
    contour : np.ndarray
        Single contour array.

    Returns
    -------
    float in [0, 1].  Returns 0.0 if hull area is zero.
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0.0:
        return 0.0
    return float(area) / float(hull_area)


def compute_circularity(contour):
    # type: (np.ndarray) -> float
    """Return the circularity of a contour: 4*pi*area / perimeter^2.

    A perfect circle gives circularity = 1.0.
    A very elongated or irregular shape gives a value near 0.

    Parameters
    ----------
    contour : np.ndarray
        Single contour array.

    Returns
    -------
    float in [0, 1].  Returns 0.0 if perimeter is zero.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter <= 0.0:
        return 0.0
    return float(4.0 * math.pi * area) / float(perimeter * perimeter)


def compute_aspect_ratio(contour):
    # type: (np.ndarray) -> float
    """Return the aspect ratio (w/h) of the upright bounding rectangle.

    Returns 0.0 if height is zero (degenerate contour).
    """
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return 0.0
    return float(w) / float(h)


# ---------------------------------------------------------------------------
# Frame annotation helpers
# ---------------------------------------------------------------------------

def draw_bounding_box(frame, x, y, w, h, color_bgr, thickness=None):
    # type: (np.ndarray, int, int, int, int, Tuple[int,int,int], Optional[int]) -> None
    """Draw a rectangle on *frame* in-place."""
    if thickness is None:
        thickness = config.BBOX_THICKNESS
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, thickness)


def draw_rotated_bounding_box(frame, contour, color_bgr, thickness=None):
    # type: (np.ndarray, np.ndarray, Tuple[int,int,int], Optional[int]) -> None
    """Draw a minimum-area rotated bounding box around *contour* in-place.

    Uses cv2.minAreaRect + cv2.boxPoints for angled objects.
    """
    if thickness is None:
        thickness = config.BBOX_THICKNESS
    rect = cv2.minAreaRect(contour)
    box  = cv2.boxPoints(rect)
    box  = np.intp(box)
    cv2.drawContours(frame, [box], 0, color_bgr, thickness)


def draw_crosshair(frame, cx, cy, color_bgr,
                   arm_length=None, thickness=None, dot_radius=None):
    # type: (np.ndarray, int, int, Tuple[int,int,int], Optional[int], Optional[int], Optional[int]) -> None
    """Draw a crosshair marker centred on (cx, cy) in-place.

    The marker consists of:
    - A horizontal line segment of length 2 * arm_length.
    - A vertical line segment of length 2 * arm_length.
    - A filled circle of radius dot_radius at the centroid.
    """
    if arm_length is None:
        arm_length = config.CROSSHAIR_ARM_LENGTH
    if thickness is None:
        thickness = config.CROSSHAIR_THICKNESS
    if dot_radius is None:
        dot_radius = config.CENTROID_DOT_RADIUS

    cv2.line(frame, (cx - arm_length, cy), (cx + arm_length, cy), color_bgr, thickness)
    cv2.line(frame, (cx, cy - arm_length), (cx, cy + arm_length), color_bgr, thickness)
    cv2.circle(frame, (cx, cy), dot_radius, color_bgr, -1)


def draw_label(frame, text, x, y,
               color_bgr=(255, 255, 255),
               bg_color_bgr=(0, 0, 0),
               font=None, font_scale=None, thickness=None):
    # type: (np.ndarray, str, int, int, Tuple[int,int,int], Optional[Tuple[int,int,int]], Optional[int], Optional[float], Optional[int]) -> None
    """Render text on frame with an optional contrasting background."""
    if font is None:
        font = config.FONT
    if font_scale is None:
        font_scale = config.FONT_SCALE
    if thickness is None:
        thickness = config.FONT_THICKNESS

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    if bg_color_bgr is not None:
        pad = 2
        cv2.rectangle(
            frame,
            (x - pad, y - text_h - pad),
            (x + text_w + pad, y + baseline + pad),
            bg_color_bgr,
            -1,
        )
    cv2.putText(frame, text, (x, y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)


def draw_fps_overlay(frame, fps, position=(10, 30)):
    # type: (np.ndarray, float, Tuple[int,int]) -> None
    """Render a FPS counter in the top-left corner of frame."""
    text = "FPS: {:.1f}".format(fps)
    draw_label(frame, text, position[0], position[1], color_bgr=(0, 255, 0))


def draw_motion_trail(frame, trail, color_bgr, min_alpha=0.3):
    # type: (np.ndarray, deque, Tuple[int,int,int], float) -> None
    """Draw a fading motion-trail polyline on *frame* in-place.

    Older positions are rendered with lower opacity by blending onto an
    overlay copy.  The newest point is drawn at full opacity.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (modified in-place).
    trail : collections.deque
        Deque of (x, y) int centroid coordinates, oldest first.
    color_bgr : tuple
        BGR colour for the trail.
    min_alpha : float
        Minimum blend alpha for the oldest segment.
    """
    n = len(trail)
    if n < 2:
        return

    trail_list = list(trail)
    for i in range(1, n):
        alpha = min_alpha + (1.0 - min_alpha) * (i / float(n))
        thickness = max(1, int(3 * (i / float(n))))

        overlay = frame.copy()
        cv2.line(overlay, trail_list[i - 1], trail_list[i], color_bgr, thickness)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def annotate_detection(
    frame,
    contour,
    color_name,
    color_bgr,
    use_rotated_bbox=False,
    smoothed_centroid=None,
):
    # type: (np.ndarray, np.ndarray, str, Tuple[int,int,int], bool, Optional[Tuple[int,int]]) -> Optional[Tuple[int,int]]
    """Annotate a single detected contour on frame in-place.

    Draws the bounding box (upright or rotated), centroid crosshair, and
    an information label that includes area, circularity, and solidity.

    Parameters
    ----------
    frame : np.ndarray
        BGR image array (modified in-place).
    contour : np.ndarray
        The contour to annotate.
    color_name : str
        Human-readable colour label.
    color_bgr : tuple
        BGR colour used for all overlay graphics.
    use_rotated_bbox : bool
        If True, draw the minimum-area rotated bounding box instead of the
        standard axis-aligned rectangle.
    smoothed_centroid : tuple or None
        Pre-computed EMA-smoothed centroid (cx, cy).  If None, the raw
        moment centroid is used for drawing.

    Returns
    -------
    (cx, cy) centroid coordinates or None.
    """
    raw_centroid = compute_centroid(contour)
    if raw_centroid is None:
        return None

    cx, cy = smoothed_centroid if smoothed_centroid is not None else raw_centroid
    area        = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    circularity = compute_circularity(contour)
    solidity    = compute_solidity(contour)

    if use_rotated_bbox:
        draw_rotated_bounding_box(frame, contour, color_bgr)
    else:
        draw_bounding_box(frame, x, y, w, h, color_bgr)

    draw_crosshair(frame, cx, cy, color_bgr)

    label = "{} ({},{}) A={} C={:.2f}".format(
        color_name, cx, cy, int(area), circularity
    )
    draw_label(frame, label, x, y - 8, color_bgr=color_bgr)

    return (cx, cy)


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------

def bgr_to_rgb(frame):
    # type: (np.ndarray) -> np.ndarray
    """Convert a BGR OpenCV frame to RGB for Tkinter / Pillow."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def gray_to_rgb(mask):
    # type: (np.ndarray) -> np.ndarray
    """Convert a single-channel binary mask to a 3-channel RGB array."""
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


# ---------------------------------------------------------------------------
# Pipeline timing context manager
# ---------------------------------------------------------------------------

class StageTimer(object):
    """Context manager that measures the wall-clock duration of a code block.

    Example
    -------
    >>> timer = StageTimer()
    >>> with timer:
    ...     result = expensive_operation()
    >>> print(timer.elapsed_ms)
    """

    def __init__(self):
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self._start) * 1000.0
