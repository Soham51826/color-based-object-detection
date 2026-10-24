"""
tracker.py
==========
Core colour-based object detection and tracking pipeline.

Pipeline stages (in order)
---------------------------
1. Frame ingestion  - cv2.VideoCapture (webcam or file)
2. Pre-filter       - Gaussian blur (default) OR bilateral filter (edge-preserving)
3. BGR -> HSV       - cv2.cvtColor(BGR2HSV)
4. CLAHE            - (optional) Adaptive histogram eq. on V channel only
5. In-range mask    - cv2.inRange for each target colour
6. Red wrap-around  - cv2.bitwise_or of two hue sub-ranges
7. Morphological    - opening (remove noise) then closing (fill voids)
8. Contour find     - RETR_EXTERNAL + CHAIN_APPROX_SIMPLE
9. Geometry filter  - min_area + solidity check (convex hull ratio)
10. EMA smoothing   - exponential moving average on centroid + bbox
11. Moment centroid - M10/M00, M01/M00
12. Trail update    - deque of historical centroids per colour
13. Annotation      - BBox (upright or rotated), crosshair, trail, label

The ColorTracker class is designed to be instantiated once per run
and called in a tight loop (tracker.process_frame(frame)).
"""

import copy
import json
import logging
import math
import os
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from . import config
from .utils import (
    FPSCounter,
    StageTimer,
    annotate_detection,
    compute_centroid,
    compute_contour_angle,
    compute_solidity,
    draw_fps_overlay,
    draw_motion_trail,
    get_logger,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes (using plain classes for Python 3.6 compatibility)
# ---------------------------------------------------------------------------

class BlobInfo(object):
    """Describes a single detected colour blob.

    Attributes
    ----------
    color_name : str
        Human-readable colour identifier, e.g. "Red".
    centroid : tuple or None
        (x, y) pixel coordinates of the blob's spatial centroid.
        When EMA smoothing is enabled this is the smoothed position.
    raw_centroid : tuple or None
        The unsmoothed moment centroid.
    area : float
        Contour area in pixels squared.
    bbox : tuple
        Bounding rectangle as (x, y, w, h).
    contour : np.ndarray
        Raw contour array returned by cv2.findContours.
    solidity : float
        Ratio of contour area to its convex hull area.
    circularity : float
        4 * pi * area / perimeter^2.
    speed : float
        Current instantaneous/smoothed motion speed in pixels/second.
    angle : float
        Principal orientation angle in degrees [-90.0, +90.0].
    dimensions : tuple
        (major_axis_length, minor_axis_length) in pixels.
    """

    def __init__(self, color_name, centroid, area, bbox, contour,
                 raw_centroid=None, solidity=1.0, circularity=1.0,
                 speed=0.0, angle=0.0, dimensions=None):
        # type: (str, Optional[Tuple[int,int]], float, Tuple[int,int,int,int], np.ndarray, Optional[Tuple[int,int]], float, float, float, float, Optional[Tuple[int,int]]) -> None
        self.color_name   = color_name
        self.centroid     = centroid
        self.raw_centroid = raw_centroid if raw_centroid is not None else centroid
        self.area         = area
        self.bbox         = bbox
        self.contour      = contour
        self.solidity     = solidity
        self.circularity  = circularity
        self.speed        = float(speed)
        self.angle        = float(angle)
        self.dimensions   = dimensions if dimensions is not None else (bbox[2], bbox[3])

    def __repr__(self):
        return "BlobInfo(color={}, centroid={}, area={:.0f}, speed={:.1f}px/s, angle={:+.1f}°)".format(
            self.color_name, self.centroid, self.area, self.speed, self.angle
        )


class FrameResult(object):
    """Aggregated result for one processed video frame.

    Attributes
    ----------
    annotated_frame : np.ndarray
        A copy of the original frame with all detection overlays applied.
    masks : dict
        Mapping from colour name to its binary morphological mask.
    blobs : list
        Flat list of all BlobInfo objects detected in this frame.
    fps : float
        Rolling-average frames-per-second at the time of processing.
    stage_times_ms : dict
        Per-stage wall-clock durations in milliseconds.
    """

    def __init__(self, annotated_frame, masks, blobs, fps, stage_times_ms=None):
        # type: (np.ndarray, Dict[str, np.ndarray], List[BlobInfo], float, Optional[Dict[str,float]]) -> None
        self.annotated_frame = annotated_frame
        self.masks           = masks
        self.blobs           = blobs
        self.fps             = fps
        self.stage_times_ms  = stage_times_ms if stage_times_ms is not None else {}


# ---------------------------------------------------------------------------
# Kalman Filter and Temporal Tracking State
# ---------------------------------------------------------------------------

class KalmanFilterTracker(object):
    """Linear Kalman Filter for continuous tracking of 2D centroid position & velocity.

    State: [x, y, vx, vy]^T
    Measurement: [x, y]^T
    """

    def __init__(self, process_noise=None, measurement_noise=None):
        if process_noise is None:
            process_noise = getattr(config, "KALMAN_PROCESS_NOISE", 1e-2)
        if measurement_noise is None:
            measurement_noise = getattr(config, "KALMAN_MEASUREMENT_NOISE", 1e-1)

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(measurement_noise)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self._is_initialized = False

    def initialize(self, x, y):
        self.kf.statePost = np.array([[float(x)], [float(y)], [0.0], [0.0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        self._is_initialized = True

    def predict(self, dt=1.0):
        if not self._is_initialized:
            return None, (0.0, 0.0)
        self.kf.transitionMatrix[0, 2] = float(dt)
        self.kf.transitionMatrix[1, 3] = float(dt)
        pred = self.kf.predict()
        pos = (float(pred[0, 0]), float(pred[1, 0]))
        vel = (float(pred[2, 0]), float(pred[3, 0]))
        return pos, vel

    def correct(self, x, y):
        if not self._is_initialized:
            self.initialize(x, y)
            return (float(x), float(y)), (0.0, 0.0)
        meas = np.array([[float(x)], [float(y)]], dtype=np.float32)
        corrected = self.kf.correct(meas)
        pos = (float(corrected[0, 0]), float(corrected[1, 0]))
        vel = (float(corrected[2, 0]), float(corrected[3, 0]))
        return pos, vel

    def reset(self):
        self._is_initialized = False


class _ColorState(object):
    """Holds per-colour Kalman filter, EMA state, motion trail, and accurate speed."""

    def __init__(self, trail_maxlen=None, ema_alpha=None):
        if trail_maxlen is None:
            trail_maxlen = config.TRAIL_MAX_LEN
        if ema_alpha is None:
            ema_alpha = config.EMA_ALPHA

        self.alpha = ema_alpha
        # Smoothed centroid (floats)
        self._cx   = None  # type: Optional[float]
        self._cy   = None  # type: Optional[float]
        # Motion trail
        self.trail = deque(maxlen=trail_maxlen)  # type: deque
        # Kalman filter
        self.kalman = KalmanFilterTracker()
        # Timing and speed tracking
        self._last_time = None
        self.speed = 0.0           # Current smoothed speed in px/s
        self.velocity = (0.0, 0.0) # Current (vx, vy) in px/s
        self.missed_frames = 0

    def update(self, raw_cx, raw_cy, timestamp=None):
        # type: (int, int, Optional[float]) -> Tuple[int, int]
        """Apply Kalman filtering + EMA to the raw centroid, compute speed, and append to trail.

        Returns the smoothed integer centroid.
        """
        now = time.time() if timestamp is None else timestamp
        dt = 0.033  # default ~30fps
        if self._last_time is not None:
            dt = max(0.001, min(0.5, now - self._last_time))
        self._last_time = now

        prev_cx = self._cx
        prev_cy = self._cy

        # 1. Kalman prediction & correction
        self.kalman.predict(dt=dt)
        self.kalman.correct(raw_cx, raw_cy)

        # 2. Centroid smoothing with EMA
        if self._cx is None or self._cy is None:
            self._cx = float(raw_cx)
            self._cy = float(raw_cy)
            self.speed = 0.0
            self.velocity = (0.0, 0.0)
        else:
            self._cx = self.alpha * float(raw_cx) + (1.0 - self.alpha) * self._cx
            self._cy = self.alpha * float(raw_cy) + (1.0 - self.alpha) * self._cy

            dx = self._cx - prev_cx
            dy = self._cy - prev_cy
            inst_speed = math.hypot(dx, dy) / dt
            alpha_spd = getattr(config, "SPEED_SMOOTHING_ALPHA", 0.6)
            self.speed = alpha_spd * inst_speed + (1.0 - alpha_spd) * self.speed
            self.velocity = (dx / dt, dy / dt)

        self.missed_frames = 0
        cx_int = round(self._cx)
        cy_int = round(self._cy)
        self.trail.append((cx_int, cy_int))
        return (cx_int, cy_int)

    def on_missed_frame(self):
        """Handle frames where this color was not detected to decay speed smoothly."""
        self.missed_frames += 1
        self.speed *= 0.6
        if self.missed_frames > getattr(config, "MAX_MISSED_TRACK_FRAMES", 8):
            self.trail.clear()
            self._cx = None
            self._cy = None
            self.kalman.reset()

    def reset(self):
        """Clear EMA state, Kalman filter, speed, and trail history."""
        self._cx  = None
        self._cy  = None
        self.trail.clear()
        self.kalman.reset()
        self.speed = 0.0
        self.velocity = (0.0, 0.0)
        self._last_time = None
        self.missed_frames = 0

    @property
    def smoothed_centroid(self):
        # type: () -> Optional[Tuple[int, int]]
        if self._cx is None or self._cy is None:
            return None
        return (round(self._cx), round(self._cy))


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------

class ColorTracker(object):
    """Stateful colour-based object detection and tracking pipeline.

    Parameters
    ----------
    color_names : list, optional
        List of colour keys to track.  Each must exist in
        config.COLOR_PRESETS.  Defaults to all presets.
    min_contour_area : int
        Area threshold in pixels squared.
    gaussian_kernel : tuple
        Kernel size for the Gaussian blur pre-processing step.
    morph_kernel_size : tuple
        Size of the elliptical structuring element for morphological ops.
    use_clahe : bool
        If True, apply CLAHE to the V channel before thresholding.
    use_bilateral : bool
        If True, use edge-preserving bilateral filter instead of Gaussian.
    use_ema_smoothing : bool
        If True, apply EMA smoothing to detected centroids.
    show_trail : bool
        If True, draw motion-trail lines on the annotated frame.
    use_rotated_bbox : bool
        If True, draw minimum-area rotated bounding boxes.
    min_solidity : float
        Solidity threshold for the convex-hull false-positive filter.

    Example
    -------
    >>> tracker = ColorTracker(color_names=["Red", "Blue"])
    >>> cap = cv2.VideoCapture(0)
    >>> while True:
    ...     ok, frame = cap.read()
    ...     if not ok:
    ...         break
    ...     result = tracker.process_frame(frame)
    ...     cv2.imshow("Detection", result.annotated_frame)
    ...     if cv2.waitKey(1) & 0xFF == ord('q'):
    ...         break
    >>> cap.release()
    """

    def __init__(
        self,
        color_names=None,
        min_contour_area=None,
        gaussian_kernel=None,
        morph_kernel_size=None,
        use_clahe=True,
        use_bilateral=False,
        use_ema_smoothing=True,
        show_trail=True,
        use_rotated_bbox=False,
        min_solidity=None,
    ):
        if color_names is None:
            color_names = list(config.COLOR_PRESETS.keys())
        if min_contour_area is None:
            min_contour_area = config.MIN_CONTOUR_AREA
        if gaussian_kernel is None:
            gaussian_kernel = config.GAUSSIAN_KERNEL_SIZE
        if morph_kernel_size is None:
            morph_kernel_size = config.MORPH_KERNEL_SIZE
        if min_solidity is None:
            min_solidity = config.MIN_SOLIDITY

        # Validate requested colours
        for name in color_names:
            if name not in config.COLOR_PRESETS:
                raise ValueError(
                    "Unknown colour '{}'. Available: {}".format(
                        name, list(config.COLOR_PRESETS.keys())
                    )
                )

        self.color_names      = color_names
        self.min_contour_area = min_contour_area
        self.gaussian_kernel  = gaussian_kernel
        self.min_solidity     = min_solidity

        # Feature flags (can be toggled at runtime by the GUI)
        self.use_clahe        = use_clahe
        self.use_bilateral    = use_bilateral
        self.use_ema_smoothing = use_ema_smoothing
        self.show_trail       = show_trail
        self.use_rotated_bbox = use_rotated_bbox

        # Build the elliptical morphological structuring element once
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, morph_kernel_size
        )

        # CLAHE object (applied to V channel only)
        self._clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID_SIZE,
        )

        self._fps_counter = FPSCounter()

        # Per-colour HSV bounds cache (deep copy to avoid mutating config)
        self._presets = {}
        for name in color_names:
            self._presets[name] = dict(config.COLOR_PRESETS[name])

        # Per-colour temporal smoothing state
        self._color_states = {name: _ColorState() for name in color_names}

        logger.info(
            "ColorTracker initialised | colours=%s | min_area=%d px^2 "
            "| clahe=%s | bilateral=%s | ema=%s | trail=%s | rotated_bbox=%s",
            color_names,
            min_contour_area,
            use_clahe,
            use_bilateral,
            use_ema_smoothing,
            show_trail,
            use_rotated_bbox,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_frame(self, frame):
        # type: (np.ndarray) -> FrameResult
        """Run the full detection pipeline on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            H x W x 3 BGR uint8 numpy array.

        Returns
        -------
        FrameResult
        """
        self._fps_counter.tick()
        stage_times = {}

        # ------ Stage 1: Spatial pre-filter ---------------------------
        with StageTimer() as t_gauss:
            if self.use_bilateral:
                blurred = cv2.bilateralFilter(
                    frame,
                    d=config.BILATERAL_D,
                    sigmaColor=config.BILATERAL_SIGMA_COLOR,
                    sigmaSpace=config.BILATERAL_SIGMA_SPACE,
                )
            else:
                blurred = cv2.GaussianBlur(
                    frame,
                    self.gaussian_kernel,
                    config.GAUSSIAN_SIGMA,
                )
        stage_times["gaussian_ms"] = t_gauss.elapsed_ms

        # ------ Stage 2: Lighting stabilization (LAB CLAHE) -----------
        with StageTimer() as t_clahe:
            if self.use_clahe:
                # Perceptual CIE L*a*b* Illumination Stabilization
                # Equalizing Lightness (L) decouples shadows/glare from chromaticity
                lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                l_eq = self._clahe.apply(l_ch)
                lab_eq = cv2.merge([l_eq, a_ch, b_ch])
                stabilized_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            else:
                stabilized_bgr = blurred
        stage_times["clahe_ms"] = t_clahe.elapsed_ms

        # ------ Stage 3: BGR -> HSV -----------------------------------
        with StageTimer() as t_hsv:
            hsv = cv2.cvtColor(stabilized_bgr, cv2.COLOR_BGR2HSV)
        stage_times["hsv_convert_ms"] = t_hsv.elapsed_ms

        # ------ Stage 4-6: Per-colour mask + morphology ---------------
        masks = {}
        with StageTimer() as t_mask:
            for name in self.color_names:
                masks[name] = self._build_mask(hsv, name)
        stage_times["inrange_morph_ms"] = t_mask.elapsed_ms

        # ------ Stage 7-9: Contour extraction + geometry filter -------
        annotated  = frame.copy()
        all_blobs  = []

        with StageTimer() as t_contour:
            for name in self.color_names:
                blobs = self._extract_blobs(masks[name], name)
                all_blobs.extend(blobs)

                preset = self._presets[name]
                state  = self._color_states[name]

                if blobs:
                    # Select primary tracked blob (closest to previous centroid, or largest)
                    primary_blob = blobs[0]
                    if len(blobs) > 1 and state.smoothed_centroid is not None:
                        pcx, pcy = state.smoothed_centroid
                        best_dist = float("inf")
                        for b in blobs:
                            if b.raw_centroid is not None:
                                dist = math.hypot(b.raw_centroid[0] - pcx, b.raw_centroid[1] - pcy)
                                if dist < best_dist:
                                    best_dist = dist
                                    primary_blob = b

                    for blob in blobs:
                        is_primary = (blob is primary_blob)
                        if is_primary:
                            # Primary target gets Kalman + EMA tracking and trail update
                            if self.use_ema_smoothing and blob.raw_centroid is not None:
                                smooth_cx, smooth_cy = state.update(
                                    blob.raw_centroid[0], blob.raw_centroid[1]
                                )
                                blob.centroid = (smooth_cx, smooth_cy)
                            else:
                                blob.centroid = blob.raw_centroid
                            blob.speed = state.speed
                        else:
                            # Secondary blobs display without scrambling primary motion trail
                            blob.centroid = blob.raw_centroid
                            blob.speed = 0.0

                        annotate_detection(
                            annotated,
                            blob.contour,
                            name,
                            preset["bgr_display"],
                            use_rotated_bbox=self.use_rotated_bbox,
                            smoothed_centroid=blob.centroid,
                            speed=blob.speed,
                            angle=blob.angle,
                        )

                    # Draw motion trail for this colour with speed indicator
                    if self.show_trail and len(state.trail) >= 2:
                        draw_motion_trail(
                            annotated,
                            state.trail,
                            preset["bgr_display"],
                            speed=state.speed,
                        )
                else:
                    state.on_missed_frame()

        stage_times["contour_moments_ms"] = t_contour.elapsed_ms

        # ------ Stage 10: HUD overlays --------------------------------
        fps = self._fps_counter.fps
        draw_fps_overlay(annotated, fps)
        self._draw_status_hud(annotated, all_blobs)

        return FrameResult(
            annotated_frame=annotated,
            masks=masks,
            blobs=all_blobs,
            fps=fps,
            stage_times_ms=stage_times,
        )

    def update_hsv_range(self, color_name, lower1, upper1, lower2=None, upper2=None):
        # type: (str, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]) -> None
        """Dynamically update the HSV bounds for a tracked colour.

        Parameters
        ----------
        color_name : str
            Key of the colour preset to update.
        lower1, upper1 : np.ndarray
            Primary HSV lower/upper bounds as uint8 numpy arrays of shape (3,).
        lower2, upper2 : np.ndarray or None
            Secondary HSV bounds (for red hue wrap-around only).
        """
        if color_name not in self._presets:
            raise KeyError("Colour '{}' is not being tracked.".format(color_name))

        self._presets[color_name]["lower1"] = lower1.astype(np.uint8)
        self._presets[color_name]["upper1"] = upper1.astype(np.uint8)
        self._presets[color_name]["lower2"] = (
            lower2.astype(np.uint8) if lower2 is not None else None
        )
        self._presets[color_name]["upper2"] = (
            upper2.astype(np.uint8) if upper2 is not None else None
        )
        logger.debug(
            "HSV updated | %s -> L1=%s U1=%s", color_name, lower1, upper1
        )

    def reset(self):
        """Reset per-colour HSV bounds to factory defaults, clear FPS and EMA."""
        for name in self.color_names:
            self._presets[name] = dict(config.COLOR_PRESETS[name])
            self._color_states[name].reset()
        self._fps_counter.reset()
        logger.info("ColorTracker reset to factory defaults.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_mask(self, hsv, color_name):
        # type: (np.ndarray, str) -> np.ndarray
        """Construct and morphologically clean the binary mask for one colour.

        For Red, which straddles the 0/180 hue discontinuity in OpenCV's HSV
        representation, two separate inRange results are combined with
        bitwise OR before morphological cleaning is applied.
        """
        preset = self._presets[color_name]

        # Primary mask
        mask = cv2.inRange(hsv, preset["lower1"], preset["upper1"])

        # Secondary mask for red hue wrap-around
        if preset["lower2"] is not None and preset["upper2"] is not None:
            mask2 = cv2.inRange(hsv, preset["lower2"], preset["upper2"])
            mask  = cv2.bitwise_or(mask, mask2)

        # Morphological opening: removes small specks (salt noise)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self._morph_kernel,
            iterations=config.MORPH_OPEN_ITERATIONS,
        )

        # Morphological closing: fills small holes (specular reflections)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self._morph_kernel,
            iterations=config.MORPH_CLOSE_ITERATIONS,
        )

        return mask

    def _extract_blobs(self, mask, color_name):
        # type: (np.ndarray, str) -> List[BlobInfo]
        """Find contours in mask and return qualifying blobs.

        Filtering criteria applied in order:
        1. Contour area >= self.min_contour_area.
        2. Solidity (area/hull_area) >= self.min_solidity — rejects streaks.
        """
        from .utils import compute_circularity, compute_solidity  # local to avoid circular at module level

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            solidity = compute_solidity(cnt)
            if solidity < self.min_solidity:
                continue

            circularity = compute_circularity(cnt)
            raw_centroid = compute_centroid(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            angle, _, dimensions = compute_contour_angle(cnt)

            blobs.append(
                BlobInfo(
                    color_name=color_name,
                    centroid=raw_centroid,        # will be updated by EMA in process_frame
                    raw_centroid=raw_centroid,
                    area=area,
                    bbox=(x, y, w, h),
                    contour=cnt,
                    solidity=solidity,
                    circularity=circularity,
                    speed=0.0,
                    angle=angle,
                    dimensions=dimensions,
                )
            )

        # Return largest blobs first; cap at MAX_BLOBS_PER_COLOR
        blobs.sort(key=lambda b: b.area, reverse=True)
        return blobs[:config.MAX_BLOBS_PER_COLOR]

    def _draw_status_hud(self, frame, blobs):
        # type: (np.ndarray, List[BlobInfo]) -> None
        """Draw a compact status HUD in the top-right corner of frame."""
        detected_colors = set(b.color_name for b in blobs)
        h, w = frame.shape[:2]
        x_start    = w - 180
        y_start    = 20
        line_height = 22

        for idx, name in enumerate(self.color_names):
            y      = y_start + idx * line_height
            preset = self._presets[name]

            if name in detected_colors:
                status     = "TRACKING"
                text_color = (0, 255, 0)
            else:
                status     = "LOST"
                text_color = (0, 0, 255)

            label = "{}: {}".format(name, status)
            cv2.putText(
                frame,
                label,
                (x_start, y),
                config.FONT,
                0.45,
                text_color,
                1,
                cv2.LINE_AA,
            )


# ---------------------------------------------------------------------------
# Synthetic Capture Fallback
# ---------------------------------------------------------------------------

class _SyntheticTarget(object):
    def __init__(self, name, color, pos, vel, size):
        self.name = name
        self.color = color
        self.pos = pos
        self.vel = vel
        self.size = size

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        setattr(self, item, value)


class SyntheticCapture(object):
    """Fallback video capture that generates synthetic frames with moving colored targets.

    Provides the same interface as cv2.VideoCapture (isOpened, read, get, set, release)
    so that downstream pipelines, GUI threads, and test scripts function seamlessly
    even when no physical webcam or video file is available.
    """

    def __init__(self, width=640, height=480, fps=30):
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._is_opened = True
        self._frame_idx = 0

        # Moving targets in BGR space for all 9 supported colour presets
        self._targets = [
            _SyntheticTarget("Red",    (48,  68,  167), [120.0, 150.0], [3.0,  2.0],  32),
            _SyntheticTarget("Green",  (52,  157, 56),  [320.0, 240.0], [-2.5, 2.5],  36),
            _SyntheticTarget("Blue",   (157, 67,  45),  [460.0, 180.0], [2.2, -2.8],  30),
            _SyntheticTarget("Yellow", (54,  169, 177), [220.0, 340.0], [-2.4, -1.8], 34),
            _SyntheticTarget("Orange", (47,  116, 177), [180.0, 120.0], [2.6,  1.8],  30),
            _SyntheticTarget("Cyan",   (162, 155, 50),  [380.0, 120.0], [-1.8, 2.6],  32),
            _SyntheticTarget("Purple", (157, 54,  116), [520.0, 300.0], [1.9, -2.1],  32),
            _SyntheticTarget("Pink",   (154, 61,  177), [280.0, 200.0], [-2.1, -2.4], 30),
            _SyntheticTarget("White",  (227, 227, 212), [400.0, 380.0], [2.4,  2.0],  30),
        ]

    def isOpened(self):
        return self._is_opened

    def read(self):
        if not self._is_opened:
            return False, None

        self._frame_idx += 1
        # Create dark canvas
        frame = np.full((self._height, self._width, 3), 35, dtype=np.uint8)

        # Update and render targets
        for t in self._targets:
            t.pos[0] += t.vel[0]
            t.pos[1] += t.vel[1]

            r = t.size
            if t.pos[0] - r < 10:
                t.pos[0] = float(10 + r)
                t.vel[0] = abs(t.vel[0])
            elif t.pos[0] + r > self._width - 10:
                t.pos[0] = float(self._width - 10 - r)
                t.vel[0] = -abs(t.vel[0])

            if t.pos[1] - r < 10:
                t.pos[1] = float(10 + r)
                t.vel[1] = abs(t.vel[1])
            elif t.pos[1] + r > self._height - 40:
                t.pos[1] = float(self._height - 40 - r)
                t.vel[1] = -abs(t.vel[1])

            cx = int(t.pos[0])
            cy = int(t.pos[1])
            cv2.circle(frame, (cx, cy), r, t.color, -1)

        # Subtle banner indicator
        banner = "SYNTHETIC TEST FEED (NO PHYSICAL CAMERA)"
        cv2.putText(
            frame,
            banner,
            (15, self._height - 15),
            config.FONT,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return True, frame

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        elif prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            self._width = int(value)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self._height = int(value)
        elif prop_id == cv2.CAP_PROP_FPS:
            self._fps = float(value)
        return True

    def release(self):
        self._is_opened = False


class LoopingVideoCapture(object):
    """Wraps cv2.VideoCapture for video files, seamlessly looping back to frame 0 on EOF.

    Enables continuous calibration of HSV sliders, CLAHE stabilization, and motion
    trails without manually reloading the video file.
    """

    def __init__(self, cap, source_path=None):
        self._cap = cap
        self._source_path = str(source_path) if source_path is not None else None
        self._is_opened = cap.isOpened()
        # Rewind to first frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)

    def isOpened(self):
        return self._cap.isOpened() if self._is_opened else False

    def read(self):
        if not self._is_opened:
            return False, None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            # Reached end of video file -> loop seamlessly back to frame 0
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return False, None

        return True, frame

    def get(self, prop_id):
        return self._cap.get(prop_id)

    def set(self, prop_id, value):
        return self._cap.set(prop_id, value)

    def release(self):
        self._is_opened = False
        return self._cap.release()


# ---------------------------------------------------------------------------
# Camera device enumeration & Iriun Webcam auto-detection
# ---------------------------------------------------------------------------

def list_available_cameras(max_probe=6):
    """Enumerate connected physical and virtual camera devices.

    Returns
    -------
    list of dict
        Each dict contains:
            'index': int
            'name': str
            'is_iriun': bool
            'opened': bool
    """
    cameras = []
    pnp_names = {}
    if os.name == "nt":
        try:
            cmd = "Get-PnpDevice -Class Camera,Image | Select-Object FriendlyName, InstanceId, Status | ConvertTo-Json"
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                if isinstance(data, dict):
                    data = [data]
                for idx, dev in enumerate(data):
                    fname = dev.get("FriendlyName", "")
                    pnp_names[idx] = fname
        except Exception:
            pass

    for idx in range(max_probe):
        cap = None
        try:
            if os.name == "nt":
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx)

            if cap.isOpened():
                ok, _ = cap.read()
                name = pnp_names.get(idx, f"Camera {idx}")
                is_iriun = any(k in name.lower() for k in getattr(config, "IRIUN_DEVICE_KEYWORDS", ["iriun", "phone"]))
                cameras.append({
                    "index": idx,
                    "name": name,
                    "is_iriun": is_iriun,
                    "opened": ok,
                })
        except Exception:
            pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return cameras


def find_iriun_camera_index(max_probe=6):
    """Attempt auto-detection of Iriun Webcam (or phone webcam) device index.

    Returns
    -------
    int or None : Candidate camera index for Iriun, or None.
    """
    # 1. Search Windows PnP device list for explicit Iriun device
    if os.name == "nt":
        try:
            cmd = "Get-PnpDevice -Class Camera,Image | Select-Object FriendlyName, InstanceId, Status | ConvertTo-Json"
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                if isinstance(data, dict):
                    data = [data]
                for dev in data:
                    name = dev.get("FriendlyName", "")
                    if any(k in name.lower() for k in getattr(config, "IRIUN_DEVICE_KEYWORDS", ["iriun", "phone"])):
                        logger.info("Identified Iriun/Phone webcam in Windows PnP: %s", name)
        except Exception:
            pass

    # 2. Probe candidate indices [1, 2, 3, 4, 0] for an active live feed
    for idx in getattr(config, "IRIUN_SEARCH_INDICES", [1, 2, 3, 4, 0]):
        if idx >= max_probe:
            continue
        cap = None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    logger.info("Found active live stream on camera index %d (Iriun candidate).", idx)
                    return idx
        except Exception:
            pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    return 1


# ---------------------------------------------------------------------------
# Convenience: open a VideoCapture with preferred settings and safe fallback
# ---------------------------------------------------------------------------

def open_capture(
    source=None,
    width=None,
    height=None,
    fps=None,
    fallback_to_synthetic=True,
    loop_video=True,
):
    """Open a cv2.VideoCapture with safe exception handling and synthetic fallback.

    Parameters
    ----------
    source : int or str, optional
        Camera index, 'iriun' alias, or path to a video file.
    width, height : int, optional
        Requested capture dimensions in pixels.
    fps : int, optional
        Requested capture frame rate.
    fallback_to_synthetic : bool, optional
        If True, returns a SyntheticCapture instance instead of raising RuntimeError.
    loop_video : bool, optional
        If True and source is a video file, returns a LoopingVideoCapture.

    Returns
    -------
    cv2.VideoCapture, LoopingVideoCapture, or SyntheticCapture
    """
    if source is None:
        source = config.DEFAULT_CAMERA_INDEX
    if width is None:
        width = config.CAPTURE_WIDTH
    if height is None:
        height = config.CAPTURE_HEIGHT
    if fps is None:
        fps = config.CAPTURE_FPS

    # Resolve 'iriun' or 'phone' string keyword to camera index
    if isinstance(source, str) and source.lower() in ("iriun", "phone", "iriun webcam"):
        resolved_idx = find_iriun_camera_index()
        source = resolved_idx if resolved_idx is not None else 1
        logger.info("Resolved Iriun source keyword to camera index %s", source)

    cap = None
    is_valid = False

    try:
        # On Windows, try DirectShow for near-zero buffer latency on physical/virtual webcams
        if isinstance(source, int) and os.name == "nt":
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        if cap.isOpened():
            # Set buffer size to 1 to eliminate frame queuing latency
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, test_frame = cap.read()
            if ok and test_frame is not None:
                is_valid = True
            else:
                logger.warning(
                    "Capture source '%s' opened but failed to produce a frame.", source
                )
        else:
            logger.warning("Capture source '%s' could not be opened.", source)
    except Exception as exc:
        logger.warning(
            "Exception while opening capture source '%s': %s", source, exc
        )

    if not is_valid:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        if fallback_to_synthetic:
            logger.warning(
                "Falling back to synthetic animated test feed (width=%d, height=%d, fps=%d).",
                width, height, fps,
            )
            return SyntheticCapture(width=width, height=height, fps=fps)
        else:
            raise RuntimeError(
                "Cannot open video source '{}' and synthetic fallback is disabled.".format(
                    source
                )
            )

    if cap is None:
        raise RuntimeError("VideoCapture failed to initialize.")

    # Determine if this is a video file rather than a live camera
    is_video_file = False
    if isinstance(source, Path):
        is_video_file = True
    elif isinstance(source, str):
        try:
            int(source)
        except ValueError:
            is_video_file = True
    elif cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
        is_video_file = True

    # Rewind to frame 0 (since test read consumed frame 0)
    if is_video_file:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if loop_video:
            logger.info(
                "VideoCapture opened in seamless looping mode | source=%s", source
            )
            return LoopingVideoCapture(cap, source_path=source)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(
        "VideoCapture opened | source=%s | resolution=%dx%d | fps=%.1f",
        source, actual_w, actual_h, actual_fps,
    )
    return cap


# ---------------------------------------------------------------------------
# CLI entry point  (python -m src.tracker)
# ---------------------------------------------------------------------------

def _parse_cli_args():
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m src.tracker",
        description=(
            "Headless Color-Based Object Detection — runs tracking in a plain "
            "OpenCV window without the Tkinter GUI.  Press 'q' or Esc to quit."
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Camera index (e.g. 0, 1) or path to a video file. Default: 0.",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        default=None,
        metavar="COLOR",
        help=(
            "One or more colour names to track, e.g. --colors Red Blue Green. "
            "Available: " + ", ".join(config.COLOR_PRESETS.keys()) + ". "
            "Default: all colours."
        ),
    )
    return parser.parse_args()


def _run_headless():
    """Run the full detection pipeline in a plain OpenCV window."""
    args = _parse_cli_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass  # keep as file-path string

    color_names = args.colors
    if color_names:
        for name in color_names:
            if name not in config.COLOR_PRESETS:
                logger.error(
                    "Unknown colour '%s'. Available: %s",
                    name, list(config.COLOR_PRESETS.keys()),
                )
                import sys
                sys.exit(1)

    tracker = ColorTracker(color_names=color_names)
    cap = open_capture(source, fallback_to_synthetic=True)

    window_name = "Color Detection — press 'q' or Esc to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[Color Tracker] Running. Press 'q' or Esc in the window to stop.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("End of stream or camera lost.")
                break

            result = tracker.process_frame(frame)
            cv2.imshow(window_name, result.annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 = Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Color Tracker] Stopped.")


if __name__ == "__main__":
    _run_headless()
