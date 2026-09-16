"""
tracker.py
==========
Core colour-based object detection and tracking pipeline.

Pipeline stages (in order)
---------------------------
1. Frame ingestion  - cv2.VideoCapture (webcam or file)
2. Gaussian blur    - 5x5 kernel, sigma=0 (auto)
3. BGR -> HSV       - cv2.cvtColor(BGR2HSV)
4. In-range mask    - cv2.inRange for each target colour
5. Red wrap-around  - cv2.bitwise_or of two hue sub-ranges
6. Morphological    - opening (remove noise) then closing (fill voids)
7. Contour find     - RETR_EXTERNAL + CHAIN_APPROX_SIMPLE
8. Area filter      - discard blobs < MIN_CONTOUR_AREA px^2
9. Moment centroid  - M10/M00, M01/M00
10. Annotation      - BBox, crosshair, label drawn onto frame copy

The ColorTracker class is designed to be instantiated once per run
and called in a tight loop (tracker.process_frame(frame)).
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .utils import (
    FPSCounter,
    StageTimer,
    annotate_detection,
    compute_centroid,
    draw_fps_overlay,
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
    area : float
        Contour area in pixels squared.
    bbox : tuple
        Bounding rectangle as (x, y, w, h).
    contour : np.ndarray
        Raw contour array returned by cv2.findContours.
    """

    def __init__(self, color_name, centroid, area, bbox, contour):
        # type: (str, Optional[Tuple[int,int]], float, Tuple[int,int,int,int], np.ndarray) -> None
        self.color_name = color_name
        self.centroid   = centroid
        self.area       = area
        self.bbox       = bbox
        self.contour    = contour

    def __repr__(self):
        return "BlobInfo(color={}, centroid={}, area={:.0f})".format(
            self.color_name, self.centroid, self.area
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
# Main tracker class
# ---------------------------------------------------------------------------

class ColorTracker(object):
    """Stateful colour-based object detection and tracking pipeline.

    Parameters
    ----------
    color_names : list, optional
        List of colour keys to track.  Each must exist in
        config.COLOR_PRESETS.  Defaults to all four presets.
    min_contour_area : int
        Area threshold in pixels squared.
    gaussian_kernel : tuple
        Kernel size for the Gaussian blur pre-processing step.
    morph_kernel_size : tuple
        Size of the elliptical structuring element for morphological ops.

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
    ):
        if color_names is None:
            color_names = list(config.COLOR_PRESETS.keys())
        if min_contour_area is None:
            min_contour_area = config.MIN_CONTOUR_AREA
        if gaussian_kernel is None:
            gaussian_kernel = config.GAUSSIAN_KERNEL_SIZE
        if morph_kernel_size is None:
            morph_kernel_size = config.MORPH_KERNEL_SIZE

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

        # Build the elliptical morphological structuring element once
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, morph_kernel_size
        )

        self._fps_counter = FPSCounter()

        # Per-colour HSV bounds cache (deep copy to avoid mutating config)
        self._presets = {}
        for name in color_names:
            self._presets[name] = dict(config.COLOR_PRESETS[name])

        logger.info(
            "ColorTracker initialised | colours=%s | min_area=%d px^2",
            color_names,
            min_contour_area,
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

        # ------ Stage 1: Gaussian blur --------------------------------
        with StageTimer() as t_gauss:
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

        # ------ Stage 3-6: Per-colour mask + morphology ---------------
        masks = {}
        with StageTimer() as t_mask:
            for name in self.color_names:
                masks[name] = self._build_mask(hsv, name)
        stage_times["inrange_morph_ms"] = t_mask.elapsed_ms

        # ------ Stage 7-9: Contour extraction + centroid --------------
        annotated  = frame.copy()
        all_blobs  = []

        with StageTimer() as t_contour:
            for name in self.color_names:
                blobs = self._extract_blobs(masks[name], name)
                all_blobs.extend(blobs)

                preset = self._presets[name]
                for blob in blobs:
                    annotate_detection(
                        annotated, blob.contour, name, preset["bgr_display"]
                    )
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
        """Reset per-colour HSV bounds to factory defaults and clear FPS."""
        for name in self.color_names:
            self._presets[name] = dict(config.COLOR_PRESETS[name])
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

        Only contours whose area exceeds self.min_contour_area are returned.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            centroid  = compute_centroid(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            blobs.append(
                BlobInfo(
                    color_name=color_name,
                    centroid=centroid,
                    area=area,
                    bbox=(x, y, w, h),
                    contour=cnt,
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
# Convenience: open a VideoCapture with preferred settings
# ---------------------------------------------------------------------------

def open_capture(source=None, width=None, height=None, fps=None):
    """Open a cv2.VideoCapture and configure its resolution and FPS.

    Parameters
    ----------
    source : int or str
        Camera index or path to a video file.
    width, height : int
        Requested capture dimensions in pixels.
    fps : int
        Requested capture frame rate.

    Returns
    -------
    cv2.VideoCapture

    Raises
    ------
    RuntimeError
        If the capture device cannot be opened.
    """
    if source is None:
        source = config.DEFAULT_CAMERA_INDEX
    if width is None:
        width = config.CAPTURE_WIDTH
    if height is None:
        height = config.CAPTURE_HEIGHT
    if fps is None:
        fps = config.CAPTURE_FPS

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open video source '{}'. "
            "Check camera index or file path.".format(source)
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(
        "VideoCapture opened | source=%s | resolution=%dx%d | fps=%.1f",
        source, actual_w, actual_h, actual_fps,
    )
    return cap
