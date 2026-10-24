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


def compute_contour_angle(contour):
    # type: (np.ndarray) -> Tuple[float, Tuple[int, int], Tuple[int, int]]
    """Compute the principal orientation angle of a contour in degrees.

    Uses second-order central image moments (PCA axis of inertia) combined with
    cv2.minAreaRect to calculate the object's orientation angle in [-90.0, +90.0]
    degrees relative to the horizontal x-axis.

    Parameters
    ----------
    contour : np.ndarray
        Single contour array.

    Returns
    -------
    (angle_deg, (cx, cy), (major_len, minor_len))
        angle_deg : float in [-90.0, +90.0]
        (cx, cy) : int centroid coordinates
        (major_len, minor_len) : int dimensions along major/minor axes
    """
    moments = cv2.moments(contour)
    m00 = moments["m00"]
    if m00 == 0.0:
        return 0.0, (0, 0), (0, 0)

    cx = int(moments["m10"] / m00)
    cy = int(moments["m01"] / m00)

    # Orientation via second-order central moments (principal inertia tensor)
    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    mu11 = moments["mu11"]

    if mu20 != mu02 or mu11 != 0:
        pca_angle_rad = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
        angle_deg = math.degrees(pca_angle_rad)
    else:
        angle_deg = 0.0

    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    major_len = int(max(rw, rh))
    minor_len = int(min(rw, rh))

    # Keep angle bounded in [-90, +90]
    while angle_deg > 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0

    return round(angle_deg, 1), (cx, cy), (major_len, minor_len)


def draw_rotated_bounding_box(frame, contour, color_bgr, thickness=None,
                              show_angle=True, show_axes=True, show_corners=True):
    # type: (np.ndarray, np.ndarray, Tuple[int,int,int], Optional[int], bool, bool, bool) -> float
    """Draw a minimum-area rotated bounding box around *contour* with clear angle metrics.

    Visualizations include:
    - Anti-aliased minimum-area rotated bounding box outline
    - Corner accent markers at each of the 4 bounding box vertices
    - Principal major axis vector through the centroid with directional arrow
    - Orthogonal minor axis guideline
    - Protractor angle arc indicating orientation relative to horizontal
    - High-contrast angle & dimension badge (e.g. 'θ: +34.5° [120x65]')

    Parameters
    ----------
    frame : np.ndarray
        BGR image (modified in-place).
    contour : np.ndarray
        Contour to enclose.
    color_bgr : tuple
        Overlay color.
    thickness : int, optional
        Border thickness.
    show_angle : bool
        If True, render the numerical angle badge and arc.
    show_axes : bool
        If True, draw principal major and minor symmetry axes.
    show_corners : bool
        If True, draw highlighted corner dots at box vertices.

    Returns
    -------
    float : Principal angle in degrees in range [-90.0, +90.0].
    """
    if thickness is None:
        thickness = config.BBOX_THICKNESS

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box_int = np.intp(box)

    # 1. Draw the anti-aliased rotated rectangle
    cv2.drawContours(frame, [box_int], 0, color_bgr, thickness, lineType=cv2.LINE_AA)

    # 2. Corner markers for visual clarity
    if show_corners:
        for pt in box_int:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), max(2, thickness + 1), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(pt[0]), int(pt[1])), max(1, thickness), color_bgr, 1, cv2.LINE_AA)

    # 3. Accurate angle and axis calculation
    angle_deg, (cx, cy), (major_len, minor_len) = compute_contour_angle(contour)
    rad = math.radians(angle_deg)

    if show_axes and (major_len > 0 or minor_len > 0):
        # Major axis endpoints
        half_major = max(15, major_len // 2)
        dx_maj = int(half_major * math.cos(rad))
        dy_maj = int(half_major * math.sin(rad))

        # Minor axis endpoints (perpendicular: rad + pi/2)
        half_minor = max(8, minor_len // 2)
        dx_min = int(half_minor * math.cos(rad + math.pi / 2.0))
        dy_min = int(half_minor * math.sin(rad + math.pi / 2.0))

        # Minor axis (soft grey cross-line)
        cv2.line(
            frame,
            (cx - dx_min, cy - dy_min),
            (cx + dx_min, cy + dy_min),
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # Major axis line with directional arrowhead pointing along orientation
        cv2.line(
            frame,
            (cx - dx_maj, cy - dy_maj),
            (cx, cy),
            (0, 255, 255),
            max(1, thickness),
            cv2.LINE_AA,
        )
        cv2.arrowedLine(
            frame,
            (cx, cy),
            (cx + dx_maj, cy + dy_maj),
            (0, 255, 255),
            max(1, thickness),
            tipLength=0.3,
            line_type=cv2.LINE_AA,
        )

    # 4. Protractor angle arc & Badge
    if show_angle:
        # Horizontal reference guideline
        ref_len = 28
        cv2.line(frame, (cx, cy), (cx + ref_len, cy), (120, 120, 150), 1, cv2.LINE_AA)

        # Angle arc
        arc_radius = 22
        start_angle = min(0.0, angle_deg)
        end_angle = max(0.0, angle_deg)
        if abs(end_angle - start_angle) > 1.0:
            cv2.ellipse(
                frame,
                (cx, cy),
                (arc_radius, arc_radius),
                0,
                start_angle,
                end_angle,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Top-most vertex of the box for badge placement
        top_y = int(np.min(box[:, 1]))
        badge_x = int(np.mean(box[:, 0])) - 45
        badge_y = max(18, top_y - 8)

        angle_text = "θ: {:+.1f}° ({}x{})".format(angle_deg, major_len, minor_len)
        draw_label(
            frame,
            angle_text,
            badge_x,
            badge_y,
            color_bgr=(0, 255, 255),
            bg_color_bgr=(20, 20, 30),
            font_scale=0.45,
            thickness=1,
        )

    return angle_deg


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

    cv2.line(frame, (cx - arm_length, cy), (cx + arm_length, cy), color_bgr, thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm_length), (cx, cy + arm_length), color_bgr, thickness, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), dot_radius, color_bgr, -1, lineType=cv2.LINE_AA)


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


def draw_motion_trail(frame, trail, color_bgr, min_alpha=0.3,
                      speed=None, draw_velocity_vector=True):
    # type: (np.ndarray, deque, Tuple[int,int,int], float, Optional[float], bool) -> None
    """Draw a high-performance, anti-aliased fading motion-trail polyline on *frame* in-place.

    Optimized with a single overlay buffer to eliminate frame copy overheads that cause
    webcam latency drops.

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
    speed : float, optional
        Current instantaneous/smoothed speed in px/s to display at the trail head.
    draw_velocity_vector : bool
        If True and moving, draws a directional velocity arrow at the trail head.
    """
    n = len(trail)
    if n < 2:
        return

    trail_list = list(trail)

    # Perform all trail drawing on a single overlay copy to ensure 60+ FPS performance
    overlay = frame.copy()

    for i in range(1, n):
        frac = i / float(n)
        thickness = max(1, int(3.5 * frac))
        cv2.line(
            overlay,
            trail_list[i - 1],
            trail_list[i],
            color_bgr,
            thickness,
            lineType=cv2.LINE_AA,
        )
        # Render trajectory nodes along the path
        if i % 3 == 0 or i == n - 1:
            cv2.circle(
                overlay,
                trail_list[i],
                max(2, int(4 * frac)),
                color_bgr,
                -1,
                lineType=cv2.LINE_AA,
            )

    # Blend the complete trail into the frame
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Head of trail markers (current position)
    head_x, head_y = trail_list[-1]

    # Draw velocity vector arrow at trail head if moving
    if draw_velocity_vector and speed is not None and speed >= config.SPEED_MIN_DISPLAY_THRESH:
        # Calculate heading vector from recent trajectory points
        ref_idx = max(0, n - 4)
        dx = head_x - trail_list[ref_idx][0]
        dy = head_y - trail_list[ref_idx][1]
        dist = math.hypot(dx, dy)
        if dist > 2.0:
            arrow_scale = min(40.0, max(15.0, speed * 0.15))
            arrow_dx = int((dx / dist) * arrow_scale)
            arrow_dy = int((dy / dist) * arrow_scale)
            arrow_end = (head_x + arrow_dx, head_y + arrow_dy)
            cv2.arrowedLine(
                frame,
                (head_x, head_y),
                arrow_end,
                (0, 255, 255),
                2,
                tipLength=0.35,
                line_type=cv2.LINE_AA,
            )

    # Draw speed badge near trail head
    if speed is not None and speed >= 1.0:
        speed_text = "{:.0f} px/s".format(speed)
        draw_label(
            frame,
            speed_text,
            head_x + 12,
            head_y - 12,
            color_bgr=(0, 255, 255),
            bg_color_bgr=(20, 20, 35),
            font_scale=0.45,
            thickness=1,
        )


def annotate_detection(
    frame,
    contour,
    color_name,
    color_bgr,
    use_rotated_bbox=False,
    smoothed_centroid=None,
    speed=None,
    angle=None,
):
    # type: (np.ndarray, np.ndarray, str, Tuple[int,int,int], bool, Optional[Tuple[int,int]], Optional[float], Optional[float]) -> Optional[Tuple[int,int]]
    """Annotate a single detected contour on frame in-place.

    Draws the bounding box (upright or rotated with angle indicators), centroid
    crosshair, motion speed metric, and an information label.

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
        If True, draw the minimum-area rotated bounding box with orientation angle.
    smoothed_centroid : tuple or None
        Pre-computed smoothed centroid (cx, cy). If None, raw centroid is used.
    speed : float or None
        Current movement speed in px/s.
    angle : float or None
        Principal orientation angle in degrees.

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

    detected_angle = angle
    if use_rotated_bbox:
        box_angle = draw_rotated_bounding_box(frame, contour, color_bgr, show_angle=True)
        if detected_angle is None:
            detected_angle = box_angle
    else:
        draw_bounding_box(frame, x, y, w, h, color_bgr)

    draw_crosshair(frame, cx, cy, color_bgr)

    # Build comprehensive label with area, circularity, speed, and angle
    label_parts = ["{} ({},{}) A={}".format(color_name, cx, cy, int(area))]
    if speed is not None and speed >= 1.0:
        label_parts.append("Spd={:.0f}px/s".format(speed))
    if use_rotated_bbox and detected_angle is not None:
        label_parts.append("θ={:+.1f}°".format(detected_angle))
    else:
        label_parts.append("C={:.2f}".format(circularity))

    label = " ".join(label_parts)
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
