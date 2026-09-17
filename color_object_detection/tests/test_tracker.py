"""
test_tracker.py
===============
Unit tests for the core tracking pipeline: mask generation, morphological
cleaning, contour extraction, moment calculation, and the red hue wrap-around
logic.

Run with:
    pytest tests/test_tracker.py -v

All tests are designed to be deterministic and require no physical camera.
They work entirely with synthetic NumPy arrays.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COLOR_PRESETS, MIN_CONTOUR_AREA, MORPH_KERNEL_SIZE
from src.tracker import BlobInfo, ColorTracker, FrameResult, SyntheticCapture, open_capture
from src.utils import FPSCounter, StageTimer, compute_centroid


# ---------------------------------------------------------------------------
# Helpers: synthetic frame factories
# ---------------------------------------------------------------------------

def _solid_bgr_frame(h, w, bgr):
    """Create a solid-colour H x W x 3 BGR uint8 frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


def _make_hsv_patch(h, w, hue, sat, val,
                    patch_x, patch_y, patch_w, patch_h):
    """
    Return a BGR frame that is dark everywhere except a solid HSV-coloured
    rectangle located at (patch_x, patch_y) with size (patch_w x patch_h).
    """
    bg_hsv   = np.zeros((h, w, 3), dtype=np.uint8)
    bg_frame = cv2.cvtColor(bg_hsv, cv2.COLOR_HSV2BGR)

    patch_hsv = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
    patch_hsv[:] = (hue, sat, val)
    patch_bgr = cv2.cvtColor(patch_hsv, cv2.COLOR_HSV2BGR)

    bg_frame[patch_y:patch_y + patch_h,
             patch_x:patch_x + patch_w] = patch_bgr
    return bg_frame


# ---------------------------------------------------------------------------
# Tracker instantiation
# ---------------------------------------------------------------------------

class TestColorTrackerInit(object):
    """Tests for ColorTracker construction."""

    def test_default_init_uses_all_presets(self):
        tracker = ColorTracker()
        assert set(tracker.color_names) == set(COLOR_PRESETS.keys())

    def test_single_colour_init(self):
        tracker = ColorTracker(color_names=["Green"])
        assert tracker.color_names == ["Green"]

    def test_invalid_colour_raises(self):
        with pytest.raises(ValueError):
            ColorTracker(color_names=["Magenta"])

    def test_custom_min_area(self):
        tracker = ColorTracker(min_contour_area=1000)
        assert tracker.min_contour_area == 1000

    def test_morph_kernel_is_built(self):
        """Ensure the structuring element is built without errors."""
        tracker = ColorTracker()
        kernel  = tracker._morph_kernel
        assert kernel.dtype == np.uint8
        assert kernel.shape == MORPH_KERNEL_SIZE


# ---------------------------------------------------------------------------
# Mask generation - Green (single range)
# ---------------------------------------------------------------------------

class TestGreenMask(object):
    """Tests for the single-range HSV mask (Green)."""

    def _get_mask(self, frame):
        tracker = ColorTracker(color_names=["Green"])
        result  = tracker.process_frame(frame)
        return result.masks["Green"]

    def test_green_patch_produces_nonzero_mask(self):
        """A bright green patch should yield a non-zero mask."""
        frame       = _make_hsv_patch(480, 640, 60, 255, 255, 200, 150, 180, 140)
        mask        = self._get_mask(frame)
        assert mask.max() == 255, "Expected non-zero mask for green patch"
        white_ratio = np.count_nonzero(mask) / float(mask.size)
        assert white_ratio > 0.02, "Mask is mostly zero: {:.4f}".format(white_ratio)

    def test_blue_frame_gives_zero_green_mask(self):
        """A pure blue frame should not trigger the green mask."""
        frame = _solid_bgr_frame(480, 640, (200, 50, 30))
        mask  = self._get_mask(frame)
        assert mask.max() == 0, "Green mask should be zero for a blue frame"

    def test_mask_shape_matches_frame(self):
        """Mask dimensions must match input frame (height x width)."""
        frame = _solid_bgr_frame(360, 480, (0, 0, 0))
        mask  = self._get_mask(frame)
        assert mask.shape == (360, 480)

    def test_mask_dtype_is_uint8(self):
        frame = _solid_bgr_frame(240, 320, (0, 0, 0))
        mask  = self._get_mask(frame)
        assert mask.dtype == np.uint8


# ---------------------------------------------------------------------------
# Red hue wrap-around (CRITICAL TEST)
# ---------------------------------------------------------------------------

class TestRedHueWrapAround(object):
    """
    Verify that the red mask correctly handles the hue discontinuity
    in OpenCV's HSV representation.

    OpenCV represents Hue in [0, 180] (degrees/2).
    Red occupies two disjoint sub-ranges:
      - [  0,  10] -- low-hue red
      - [160, 180] -- high-hue red (wraps from 360 degrees)

    These must be combined with cv2.bitwise_or.
    """

    def _red_mask(self, frame):
        tracker = ColorTracker(color_names=["Red"])
        result  = tracker.process_frame(frame)
        return result.masks["Red"]

    def test_low_hue_red_detected(self):
        """Pure red at H=5 (low hue) must be detected."""
        frame       = _make_hsv_patch(480, 640, 5, 230, 230, 100, 100, 200, 200)
        mask        = self._red_mask(frame)
        white_pixels = np.count_nonzero(mask)
        assert white_pixels > 500, (
            "Low-hue red (H=5) not detected. Non-zero pixels: {}".format(white_pixels)
        )

    def test_high_hue_red_detected(self):
        """Wrap-around red at H=170 must also be detected via the second mask."""
        frame        = _make_hsv_patch(480, 640, 170, 230, 230, 100, 100, 200, 200)
        mask         = self._red_mask(frame)
        white_pixels = np.count_nonzero(mask)
        assert white_pixels > 500, (
            "High-hue red (H=170) not detected. Non-zero pixels: {}".format(white_pixels)
        )

    def test_mid_hue_not_detected_as_red(self):
        """Green (H~60) must NOT be detected by the red mask."""
        frame        = _make_hsv_patch(480, 640, 60, 255, 255, 100, 100, 200, 200)
        mask         = self._red_mask(frame)
        white_pixels = np.count_nonzero(mask)
        assert white_pixels < 200, (
            "Green patch wrongly detected as red: {} white pixels".format(white_pixels)
        )

    def test_blue_hue_not_detected_as_red(self):
        """Blue (H~110) must NOT trigger the red mask."""
        frame = _make_hsv_patch(480, 640, 110, 200, 200, 100, 100, 200, 200)
        mask  = self._red_mask(frame)
        assert np.count_nonzero(mask) < 200

    def test_yellow_hue_not_detected_as_red(self):
        """Yellow (H~30) must NOT trigger the red mask."""
        frame = _make_hsv_patch(480, 640, 30, 220, 220, 100, 100, 200, 200)
        mask  = self._red_mask(frame)
        assert np.count_nonzero(mask) < 200

    def test_both_red_sub_ranges_unioned(self):
        """
        Frame containing one low-hue red patch AND one high-hue red patch
        must produce a mask with two distinct clusters of white pixels.
        """
        h, w  = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Left patch: H=5 (low-hue red)
        left_hsv = np.zeros((200, 150, 3), dtype=np.uint8)
        left_hsv[:] = (5, 220, 220)
        left_bgr = cv2.cvtColor(left_hsv, cv2.COLOR_HSV2BGR)
        frame[100:300, 50:200] = left_bgr

        # Right patch: H=173 (high-hue red)
        right_hsv = np.zeros((200, 150, 3), dtype=np.uint8)
        right_hsv[:] = (173, 220, 220)
        right_bgr = cv2.cvtColor(right_hsv, cv2.COLOR_HSV2BGR)
        frame[100:300, 420:570] = right_bgr

        mask = self._red_mask(frame)

        left_mask_region  = mask[100:300, 50:200]
        right_mask_region = mask[100:300, 420:570]

        assert np.count_nonzero(left_mask_region)  > 300, "Left red patch missed"
        assert np.count_nonzero(right_mask_region) > 300, "Right red patch missed"


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------

class TestCentroidComputation(object):
    """Tests for the spatial moment centroid calculation."""

    def test_circular_contour_centroid(self):
        """Centroid of a perfect circle must be at its centre."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 50, 255, -1)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        assert contours, "No contours found"
        centroid = compute_centroid(contours[0])
        assert centroid is not None
        cx, cy = centroid
        assert abs(cx - 150) <= 3, "cx={} expected ~150".format(cx)
        assert abs(cy - 150) <= 3, "cy={} expected ~150".format(cy)

    def test_rectangular_contour_centroid(self):
        """Centroid of a rectangle must be at its geometric centre."""
        mask = np.zeros((200, 400), dtype=np.uint8)
        cv2.rectangle(mask, (100, 50), (300, 150), 255, -1)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        centroid = compute_centroid(contours[0])
        assert centroid is not None
        cx, cy = centroid
        assert abs(cx - 200) <= 3
        assert abs(cy - 100) <= 3

    def test_degenerate_contour_returns_none(self):
        """A zero-area contour (single point) should return None."""
        degenerate = np.array([[[5, 5]]], dtype=np.int32)
        result = compute_centroid(degenerate)
        assert result is None


# ---------------------------------------------------------------------------
# FrameResult structure
# ---------------------------------------------------------------------------

class TestFrameResult(object):
    """Verify the shape and types of FrameResult returned by process_frame."""

    def test_process_frame_returns_frame_result(self):
        tracker = ColorTracker(color_names=["Blue"])
        frame   = _solid_bgr_frame(240, 320, (0, 0, 0))
        result  = tracker.process_frame(frame)
        assert isinstance(result, FrameResult)

    def test_annotated_frame_same_shape(self):
        tracker = ColorTracker(color_names=["Blue"])
        frame   = _solid_bgr_frame(480, 640, (100, 0, 0))
        result  = tracker.process_frame(frame)
        assert result.annotated_frame.shape == frame.shape

    def test_masks_dict_keys_match_colour_names(self):
        tracker = ColorTracker(color_names=["Red", "Yellow"])
        frame   = _solid_bgr_frame(240, 320, (0, 0, 0))
        result  = tracker.process_frame(frame)
        assert set(result.masks.keys()) == {"Red", "Yellow"}

    def test_fps_is_non_negative(self):
        tracker = ColorTracker(color_names=["Green"])
        frame   = _solid_bgr_frame(240, 320, (0, 0, 0))
        result  = tracker.process_frame(frame)
        assert result.fps >= 0.0

    def test_stage_times_keys_present(self):
        tracker       = ColorTracker(color_names=["Blue"])
        frame         = _solid_bgr_frame(240, 320, (0, 0, 0))
        result        = tracker.process_frame(frame)
        expected_keys = {
            "gaussian_ms", "hsv_convert_ms",
            "inrange_morph_ms", "contour_moments_ms",
        }
        assert expected_keys.issubset(set(result.stage_times_ms.keys()))

    def test_stage_times_are_non_negative(self):
        tracker = ColorTracker(color_names=["Green"])
        frame   = _solid_bgr_frame(480, 640, (0, 0, 0))
        result  = tracker.process_frame(frame)
        for k, v in result.stage_times_ms.items():
            assert v >= 0.0, "Stage '{}' reported negative time: {}".format(k, v)


# ---------------------------------------------------------------------------
# Blob detection
# ---------------------------------------------------------------------------

class TestBlobDetection(object):
    """End-to-end blob detection tests."""

    def test_large_green_patch_yields_blob(self):
        """A large green rectangle must produce at least one green blob."""
        frame   = _make_hsv_patch(480, 640, 60, 200, 200, 50, 50, 400, 300)
        tracker = ColorTracker(color_names=["Green"])
        result  = tracker.process_frame(frame)
        green_blobs = [b for b in result.blobs if b.color_name == "Green"]
        assert green_blobs, "No green blobs detected for a large green patch"

    def test_blob_has_valid_bbox(self):
        """Each blob must have a non-degenerate bounding box."""
        frame   = _make_hsv_patch(480, 640, 60, 200, 200, 50, 50, 400, 300)
        tracker = ColorTracker(color_names=["Green"])
        result  = tracker.process_frame(frame)
        for blob in result.blobs:
            x, y, w, h = blob.bbox
            assert w > 0 and h > 0, "Degenerate bbox: {}".format(blob.bbox)

    def test_min_area_filter_rejects_tiny_blobs(self):
        """Patches smaller than MIN_CONTOUR_AREA must be filtered out."""
        frame   = _make_hsv_patch(480, 640, 110, 200, 200, 300, 200, 5, 5)
        tracker = ColorTracker(color_names=["Blue"], min_contour_area=500)
        result  = tracker.process_frame(frame)
        blue_blobs = [b for b in result.blobs if b.color_name == "Blue"]
        assert len(blue_blobs) == 0, "Tiny blob should have been filtered"


# ---------------------------------------------------------------------------
# Dynamic HSV update
# ---------------------------------------------------------------------------

class TestDynamicHSVUpdate(object):
    """Test runtime HSV range updates (slider simulation)."""

    def test_update_green_range(self):
        tracker   = ColorTracker(color_names=["Green"])
        new_lower = np.array([40, 50, 50], dtype=np.uint8)
        new_upper = np.array([80, 255, 255], dtype=np.uint8)
        tracker.update_hsv_range("Green", new_lower, new_upper)
        assert np.array_equal(tracker._presets["Green"]["lower1"], new_lower)
        assert np.array_equal(tracker._presets["Green"]["upper1"], new_upper)

    def test_update_invalid_colour_raises(self):
        tracker = ColorTracker(color_names=["Green"])
        with pytest.raises(KeyError):
            tracker.update_hsv_range(
                "Magenta",
                np.array([0, 0, 0], dtype=np.uint8),
                np.array([180, 255, 255], dtype=np.uint8),
            )

    def test_reset_restores_defaults(self):
        tracker = ColorTracker(color_names=["Red"])
        # Clobber the presets
        tracker.update_hsv_range(
            "Red",
            np.array([0, 0, 0], dtype=np.uint8),
            np.array([1, 1, 1], dtype=np.uint8),
        )
        tracker.reset()
        original = COLOR_PRESETS["Red"]["lower1"]
        restored = tracker._presets["Red"]["lower1"]
        assert np.array_equal(original, restored)


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class TestFPSCounter(object):
    """Unit tests for the rolling-average FPS counter."""

    def test_zero_fps_before_ticks(self):
        counter = FPSCounter(window=10)
        assert counter.fps == 0.0

    def test_fps_after_single_tick(self):
        counter = FPSCounter(window=10)
        counter.tick()
        assert counter.fps == 0.0  # need at least 2 ticks

    def test_fps_is_positive_after_multiple_ticks(self):
        counter = FPSCounter(window=10)
        for _ in range(5):
            counter.tick()
            time.sleep(0.005)
        assert counter.fps > 0.0

    def test_reset_clears_history(self):
        counter = FPSCounter(window=10)
        for _ in range(5):
            counter.tick()
            time.sleep(0.001)
        counter.reset()
        assert counter.fps == 0.0


# ---------------------------------------------------------------------------
# Stage timer
# ---------------------------------------------------------------------------

class TestStageTimer(object):
    """Unit tests for the StageTimer context manager."""

    def test_elapsed_ms_positive(self):
        timer = StageTimer()
        with timer:
            time.sleep(0.01)
        assert timer.elapsed_ms >= 8.0, \
            "Expected >= 8 ms, got {:.3f}".format(timer.elapsed_ms)

    def test_elapsed_ms_is_float(self):
        timer = StageTimer()
        with timer:
            pass
        assert isinstance(timer.elapsed_ms, float)


# ---------------------------------------------------------------------------
# Synthetic capture and open_capture fallback
# ---------------------------------------------------------------------------

class TestSyntheticCapture(object):
    """Unit tests for the fallback SyntheticCapture device."""

    def test_synthetic_capture_lifecycle(self):
        cap = SyntheticCapture(width=320, height=240, fps=15)
        assert cap.isOpened() is True
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 320.0
        assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 240.0
        assert cap.get(cv2.CAP_PROP_FPS) == 15.0

        ok, frame = cap.read()
        assert ok is True
        assert frame is not None
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

        # Test set properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 640.0

        cap.release()
        assert cap.isOpened() is False
        ok_after, frame_after = cap.read()
        assert ok_after is False
        assert frame_after is None

    def test_synthetic_capture_runs_with_color_tracker(self):
        cap = SyntheticCapture(width=640, height=480, fps=30)
        tracker = ColorTracker()

        # Run 5 frames through tracker
        for _ in range(5):
            ok, frame = cap.read()
            assert ok is True
            result = tracker.process_frame(frame)
            assert isinstance(result, FrameResult)
            assert result.annotated_frame.shape == (480, 640, 3)
        cap.release()


class TestOpenCaptureFallback(object):
    """Unit tests for open_capture error handling and fallback behavior."""

    def test_open_capture_fallback_when_invalid_device(self):
        # Index -999 is invalid on all standard systems
        cap = open_capture(source=-999, fallback_to_synthetic=True)
        assert isinstance(cap, SyntheticCapture)
        assert cap.isOpened() is True
        ok, frame = cap.read()
        assert ok is True
        assert frame is not None
        cap.release()

    def test_open_capture_raises_when_fallback_disabled(self):
        with pytest.raises(RuntimeError):
            open_capture(source=-999, fallback_to_synthetic=False)

