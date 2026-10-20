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
import logging
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
    """

    def __init__(self, color_name, centroid, area, bbox, contour,
                 raw_centroid=None, solidity=1.0, circularity=1.0):
        # type: (str, Optional[Tuple[int,int]], float, Tuple[int,int,int,int], np.ndarray, Optional[Tuple[int,int]], float, float) -> None
        self.color_name   = color_name
        self.centroid     = centroid
        self.raw_centroid = raw_centroid if raw_centroid is not None else centroid
        self.area         = area
        self.bbox         = bbox
        self.contour      = contour
        self.solidity     = solidity
        self.circularity  = circularity

    def __repr__(self):
        return "BlobInfo(color={}, centroid={}, area={:.0f}, solidity={:.2f})".format(
            self.color_name, self.centroid, self.area, self.solidity
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
# Per-colour state for temporal smoothing
# ---------------------------------------------------------------------------

class _ColorState(object):
    """Holds per-colour EMA state and motion trail."""

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

    def update(self, raw_cx, raw_cy):
        # type: (int, int) -> Tuple[int, int]
        """Apply EMA to the raw centroid and append to trail.

        Returns the smoothed integer centroid.
        """
        if self._cx is None or self._cy is None:
            self._cx = float(raw_cx)
            self._cy = float(raw_cy)
        else:
            self._cx = self.alpha * raw_cx + (1.0 - self.alpha) * self._cx
            self._cy = self.alpha * raw_cy + (1.0 - self.alpha) * self._cy

        cx_int = round(self._cx)
        cy_int = round(self._cy)
        self.trail.append((cx_int, cy_int))
        return (cx_int, cy_int)

    def reset(self):
        """Clear EMA state and trail history."""
        self._cx  = None
        self._cy  = None
        self.trail.clear()

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

        # ------ Stage 2: BGR -> HSV -----------------------------------
        with StageTimer() as t_hsv:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        stage_times["hsv_convert_ms"] = t_hsv.elapsed_ms

        # ------ Stage 3 (optional): CLAHE on V channel ----------------
        with StageTimer() as t_clahe:
            if self.use_clahe:
                h_ch, s_ch, v_ch = cv2.split(hsv)
                v_eq = self._clahe.apply(v_ch)
                hsv  = cv2.merge([h_ch, s_ch, v_eq])
        stage_times["clahe_ms"] = t_clahe.elapsed_ms

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

                for blob in blobs:
                    # EMA centroid smoothing
                    if self.use_ema_smoothing and blob.raw_centroid is not None:
                        smooth_cx, smooth_cy = state.update(
                            blob.raw_centroid[0], blob.raw_centroid[1]
                        )
                        blob.centroid = (smooth_cx, smooth_cy)
                    else:
                        blob.centroid = blob.raw_centroid

                    annotate_detection(
                        annotated,
                        blob.contour,
                        name,
                        preset["bgr_display"],
                        use_rotated_bbox=self.use_rotated_bbox,
                        smoothed_centroid=blob.centroid,
                    )

                # Draw motion trail for this colour
                if self.show_trail and len(state.trail) >= 2:
                    draw_motion_trail(annotated, state.trail, preset["bgr_display"])

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
        Camera index or path to a video file. Defaults to config.DEFAULT_CAMERA_INDEX.
    width, height : int, optional
        Requested capture dimensions in pixels.
    fps : int, optional
        Requested capture frame rate.
    fallback_to_synthetic : bool, optional
        If True, returns a SyntheticCapture instance instead of raising RuntimeError
        when the physical camera or video file is inaccessible.
    loop_video : bool, optional
        If True and source is a video file, returns a LoopingVideoCapture that
        automatically rewinds to frame 0 on EOF for continuous playback.

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

    cap = None
    is_valid = False

    try:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ok, test_frame = cap.read()
            if ok and test_frame is not None:
                is_valid = True
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
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
    if isinstance(source, (str, Path)):
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
