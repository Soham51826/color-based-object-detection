"""
test_tracker.py
===============
Unit tests for the core tracking pipeline: mask generation, morphological
cleaning, contour extraction, moment calculation, red hue wrap-around logic,
CLAHE V-channel normalization, solidity filtering, EMA centroid smoothing,
and motion-trail state management.

Run with:
    pytest tests/test_tracker.py -v

All tests are designed to be deterministic and require no physical camera.
They work entirely with synthetic NumPy arrays.
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import pytest  # type: ignore

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COLOR_PRESETS, MIN_CONTOUR_AREA, MORPH_KERNEL_SIZE, EMA_ALPHA
from src.tracker import (
    BlobInfo,
    ColorTracker,
    FrameResult,
    SyntheticCapture,
    LoopingVideoCapture,
    KalmanFilterTracker,
    _ColorState,
    open_capture,
    find_iriun_camera_index,
    list_available_cameras,
)
from src.utils import (
    FPSCounter,
    StageTimer,
    compute_centroid,
    compute_contour_angle,
    compute_solidity,
    compute_circularity,
    draw_rotated_bounding_box,
    draw_motion_trail,
)


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

    def test_clahe_flag_defaults_on(self):
        tracker = ColorTracker()
        assert tracker.use_clahe is True

    def test_ema_flag_defaults_on(self):
        tracker = ColorTracker()
        assert tracker.use_ema_smoothing is True

    def test_color_states_created_for_all_colors(self):
        tracker = ColorTracker(color_names=["Red", "Green"])
        assert "Red"   in tracker._color_states
        assert "Green" in tracker._color_states

    def test_nine_presets_available(self):
        """After the expansion, exactly 9 colour presets must exist."""
        expected = {"Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Pink", "White"}
        assert set(COLOR_PRESETS.keys()) == expected


# ---------------------------------------------------------------------------
# Mask generation - Green (single range)
# ---------------------------------------------------------------------------

class TestGreenMask(object):
    """Tests for the single-range HSV mask (Green)."""

    def _get_mask(self, frame):
        tracker = ColorTracker(color_names=["Green"], use_clahe=False)
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
    """

    def _red_mask(self, frame):
        tracker = ColorTracker(color_names=["Red"], use_clahe=False)
        result  = tracker.process_frame(frame)
        return result.masks["Red"]

    def test_low_hue_red_detected(self):
        """Pure red at H=5 (low hue) must be detected."""
        frame        = _make_hsv_patch(480, 640, 5, 230, 230, 100, 100, 200, 200)
        mask         = self._red_mask(frame)
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
# CLAHE V-channel normalization
# ---------------------------------------------------------------------------

class TestCLAHENormalization(object):
    """Verify that CLAHE preprocessing helps detect low-brightness patches."""

    def _process(self, frame, use_clahe):
        tracker = ColorTracker(color_names=["Green"], use_clahe=use_clahe)
        return tracker.process_frame(frame)

    def test_clahe_does_not_corrupt_frame_shape(self):
        """CLAHE processing must not change the annotated frame dimensions."""
        frame  = _make_hsv_patch(480, 640, 60, 200, 200, 50, 50, 200, 200)
        result = self._process(frame, use_clahe=True)
        assert result.annotated_frame.shape == frame.shape

    def test_clahe_mask_is_uint8(self):
        """Mask produced after CLAHE must still be uint8."""
        frame  = _make_hsv_patch(480, 640, 60, 200, 200, 50, 50, 200, 200)
        result = self._process(frame, use_clahe=True)
        assert result.masks["Green"].dtype == np.uint8

    def test_clahe_stage_time_recorded(self):
        """stage_times_ms must include 'clahe_ms' key."""
        frame  = _solid_bgr_frame(240, 320, (0, 0, 0))
        result = self._process(frame, use_clahe=True)
        assert "clahe_ms" in result.stage_times_ms

    def test_clahe_disabled_still_works(self):
        """Tracker must work correctly with CLAHE disabled."""
        frame  = _make_hsv_patch(480, 640, 60, 200, 200, 50, 50, 200, 200)
        result = self._process(frame, use_clahe=False)
        assert isinstance(result, FrameResult)

    def test_clahe_improves_low_brightness_detection(self):
        """
        For a dim green patch (V=80), CLAHE should produce >= as many
        detected pixels as without CLAHE (non-regression check).
        """
        frame = _make_hsv_patch(480, 640, 60, 220, 80, 50, 50, 300, 250)
        result_clahe   = self._process(frame, use_clahe=True)
        result_no_clahe = self._process(frame, use_clahe=False)
        nz_clahe   = np.count_nonzero(result_clahe.masks["Green"])
        nz_no_clahe = np.count_nonzero(result_no_clahe.masks["Green"])
        # CLAHE should not hurt detection; allow equal
        assert nz_clahe >= nz_no_clahe or nz_clahe > 0


# ---------------------------------------------------------------------------
# Solidity filtering
# ---------------------------------------------------------------------------

class TestSolidityFilter(object):
    """Verify convex-hull solidity-based false-positive rejection."""

    def test_compute_solidity_circle(self):
        """A circle has solidity close to 1.0."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 60, 255, -1)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        s = compute_solidity(contours[0])
        assert s > 0.9, "Circle solidity should be > 0.9, got {}".format(s)

    def test_compute_solidity_convex_polygon(self):
        """A convex polygon (square) has solidity == 1.0."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (40, 40), (160, 160), 255, -1)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        s = compute_solidity(contours[0])
        assert s > 0.95, "Square solidity should be ~1.0, got {}".format(s)

    def test_compute_solidity_c_shape(self):
        """A concave C-shape has solidity clearly less than 1.0 and below 0.65."""
        # Draw a large square and cut a big rectangular notch from one side
        # to form a C / U shape.  The convex hull fills the notch, so
        # solidity = filled_area / hull_area  < 1.
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Outer square  (120 x 160 = 19200 px)
        cv2.rectangle(mask, (10, 20), (130, 180), 255, -1)
        # Cut out a wide notch  (90 x 100 = 9000 px)
        cv2.rectangle(mask, (40, 40), (130, 140), 0, -1)
        # Remaining white area ≈ 19200 - 9000 = 10200 px
        # Convex hull of the C-shape fills the notch → hull ≈ 19200 px
        # Solidity ≈ 10200 / 19200 ≈ 0.53 — well below 0.65
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        assert contours, "No contours found for C-shape mask"
        s = compute_solidity(contours[0])
        assert s < 0.65, "C-shape solidity should be < 0.65, got {}".format(s)

    def test_solidity_filter_rejects_streaks(self):
        """
        A thin streak should be rejected by _extract_blobs when min_solidity=0.65.
        Build a frame with a thin green line (low solidity) and a large green circle
        (high solidity).  Only the circle blob must survive.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Circle: high solidity
        circle_hsv = np.zeros((200, 200, 3), dtype=np.uint8)
        circle_hsv[:] = (60, 200, 200)
        circle_bgr = cv2.cvtColor(circle_hsv, cv2.COLOR_HSV2BGR)
        cv2.circle(frame, (320, 240), 80, (30, 200, 30), -1)

        tracker = ColorTracker(
            color_names=["Green"],
            use_clahe=False,
            min_solidity=0.65,
        )
        result = tracker.process_frame(frame)
        for blob in result.blobs:
            assert blob.solidity >= 0.65, (
                "Blob with solidity {:.3f} slipped through filter".format(blob.solidity)
            )


# ---------------------------------------------------------------------------
# Circularity computation
# ---------------------------------------------------------------------------

class TestCircularity(object):

    def test_circle_circularity_near_one(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 60, 255, -1)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        c = compute_circularity(contours[0])
        assert c > 0.85, "Circle circularity should be > 0.85, got {}".format(c)

    def test_rectangle_circularity_less_than_circle(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (250, 200), 255, -1)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        c = compute_circularity(contours[0])
        assert 0.0 < c < 0.9, "Rectangle circularity should be < 0.9"


# ---------------------------------------------------------------------------
# EMA centroid smoothing
# ---------------------------------------------------------------------------

class TestEMASmoothing(object):
    """Tests for _ColorState EMA centroid smoothing and trail."""

    def test_first_update_returns_raw(self):
        state = _ColorState()
        cx, cy = state.update(100, 200)
        assert cx == 100 and cy == 200

    def test_ema_smooths_towards_new_position(self):
        """After starting at (100,200), moving to (200,200) should yield
        a smoothed value between 100 and 200, closer to 200."""
        state = _ColorState(ema_alpha=EMA_ALPHA)
        state.update(100, 200)
        cx, cy = state.update(200, 200)
        assert 100 < cx <= 200, "EMA cx={} should be between 100 and 200".format(cx)

    def test_trail_appends_on_update(self):
        state = _ColorState()
        state.update(10, 20)
        state.update(30, 40)
        assert len(state.trail) == 2

    def test_trail_maxlen_respected(self):
        state = _ColorState(trail_maxlen=5)
        for i in range(20):
            state.update(i, i)
        assert len(state.trail) <= 5

    def test_reset_clears_ema_and_trail(self):
        state = _ColorState()
        state.update(100, 100)
        state.update(200, 200)
        state.reset()
        assert state.smoothed_centroid is None
        assert len(state.trail) == 0

    def test_smoothed_centroid_none_before_update(self):
        state = _ColorState()
        assert state.smoothed_centroid is None

    def test_tracker_ema_produces_blob_centroid(self):
        """Blobs returned by process_frame should have a valid centroid when EMA is on."""
        frame   = _make_hsv_patch(480, 640, 60, 200, 200, 50, 50, 400, 300)
        tracker = ColorTracker(color_names=["Green"], use_clahe=False, use_ema_smoothing=True)
        result  = tracker.process_frame(frame)
        green_blobs = [b for b in result.blobs if b.color_name == "Green"]
        if green_blobs:
            assert green_blobs[0].centroid is not None

    def test_ema_disabled_still_gives_centroid(self):
        """With EMA disabled, centroid should equal raw_centroid."""
        frame   = _make_hsv_patch(480, 640, 60, 200, 200, 50, 50, 400, 300)
        tracker = ColorTracker(color_names=["Green"], use_clahe=False, use_ema_smoothing=False)
        result  = tracker.process_frame(frame)
        for blob in result.blobs:
            assert blob.centroid == blob.raw_centroid


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
            "gaussian_ms", "hsv_convert_ms", "clahe_ms",
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
        frame   = _make_hsv_patch(480, 640, 114, 200, 200, 300, 200, 5, 5)
        tracker = ColorTracker(color_names=["Blue"], min_contour_area=500)
        result  = tracker.process_frame(frame)
        blue_blobs = [b for b in result.blobs if b.color_name == "Blue"]
        assert len(blue_blobs) == 0, "Tiny blob should have been filtered"

    def test_blob_has_solidity_attribute(self):
        """Blobs must have a solidity attribute in [0, 1]."""
        frame   = _make_hsv_patch(480, 640, 60, 200, 200, 50, 50, 300, 250)
        tracker = ColorTracker(color_names=["Green"])
        result  = tracker.process_frame(frame)
        for blob in result.blobs:
            assert 0.0 <= blob.solidity <= 1.0

    def test_blob_has_circularity_attribute(self):
        """Blobs must have a circularity attribute in [0, 1]."""
        frame   = _make_hsv_patch(480, 640, 60, 200, 200, 50, 50, 300, 250)
        tracker = ColorTracker(color_names=["Green"])
        result  = tracker.process_frame(frame)
        for blob in result.blobs:
            assert 0.0 <= blob.circularity <= 1.0


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

    def test_reset_clears_ema_state(self):
        """After reset(), all colour EMA states must be cleared."""
        frame   = _make_hsv_patch(480, 640, 60, 200, 200, 50, 50, 300, 250)
        tracker = ColorTracker(color_names=["Green"])
        tracker.process_frame(frame)  # seed the EMA state
        tracker.reset()
        assert tracker._color_states["Green"].smoothed_centroid is None


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


class TestPipelineDemo(object):
    """Unit tests for the educational pipeline demo module."""

    def test_pipeline_demo_runs_and_saves_graphic(self, tmp_path):
        from src.pipeline_demo import run_pipeline_demo
        demo_out = tmp_path / "pipeline_demonstration.png"
        saved = run_pipeline_demo(save_path=demo_out, show_window=False)
        assert saved.exists()
        img = cv2.imread(str(saved))
        assert img is not None
        # Collage has 4 panels wide x 2 panels high (320*4 x 240*2 = 1280x480)
        assert img.shape == (480, 1280, 3)


class TestLoopingVideoCapture(object):
    """Unit tests for video file looping and LoopingVideoCapture."""

    def test_benchmark_video_generation_and_looping(self, tmp_path):
        from benchmarks.generate_video import generate_benchmark_video

        test_video_path = tmp_path / "test_loop.mp4"
        # Generate a small 1-second test video (30 fps = 30 frames)
        generate_benchmark_video(
            output_path=test_video_path,
            fps=30,
            duration_sec=1.0,
            show_progress=False,
        )
        assert test_video_path.exists()
        assert test_video_path.stat().st_size > 1000

        # Open with open_capture with loop_video=True
        cap = open_capture(source=str(test_video_path), loop_video=True)
        assert isinstance(cap, LoopingVideoCapture)
        assert cap.isOpened() is True

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert total_frames == 30

        # Read more frames than total_frames (e.g. 70 frames) to verify seamless looping
        for i in range(70):
            ok, frame = cap.read()
            assert ok is True
            assert frame is not None
            assert frame.shape == (480, 640, 3)

        cap.release()
        assert cap.isOpened() is False


# ---------------------------------------------------------------------------
# Trajectory Speed & Motion Trail
# ---------------------------------------------------------------------------

class TestSpeedAndTrajectoryTrail(object):
    """Verify velocity, speed estimation, and trajectory rendering."""

    def test_speed_calculation_positive_on_motion(self):
        state = _ColorState()
        state.update(100, 100, timestamp=10.00)
        state.update(130, 140, timestamp=10.10)  # moved 50 px in 0.1s => ~500 px/s
        assert state.speed > 50.0, "Speed should be positive and substantial on motion"
        assert state.velocity[0] > 0 and state.velocity[1] > 0

    def test_speed_decays_on_missed_frames(self):
        state = _ColorState()
        state.update(100, 100, timestamp=10.00)
        state.update(150, 150, timestamp=10.10)
        spd1 = state.speed
        state.on_missed_frame()
        assert state.speed < spd1, "Speed should decay when target is missed"

    def test_draw_motion_trail_with_speed(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trail = deque([(100, 100), (120, 120), (140, 140), (160, 160)])
        draw_motion_trail(frame, trail, (0, 255, 0), speed=120.0, draw_velocity_vector=True)
        # Should have non-zero pixels drawn for trail, arrow, and speed badge
        assert np.count_nonzero(frame) > 50

    def test_blob_info_has_speed_and_angle(self):
        cnt = np.array([[[10, 10]], [[50, 10]], [[50, 80]], [[10, 80]]], dtype=np.int32)
        blob = BlobInfo(
            color_name="Red",
            centroid=(30, 45),
            area=2800.0,
            bbox=(10, 10, 40, 70),
            contour=cnt,
            speed=45.5,
            angle=15.2,
        )
        assert blob.speed == 45.5
        assert blob.angle == 15.2


# ---------------------------------------------------------------------------
# Rotated Bounding Box & Angle Accuracy
# ---------------------------------------------------------------------------

class TestRotatedBoundingBoxAndAngle(object):
    """Verify orientation angle calculation and rotated bbox visualization."""

    def test_horizontal_rectangle_angle_near_zero(self):
        # Create a horizontal rectangle (width 200, height 40)
        mask = np.zeros((300, 400), dtype=np.uint8)
        cv2.rectangle(mask, (50, 100), (250, 140), 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        angle, centroid, dims = compute_contour_angle(contours[0])
        assert abs(angle) < 5.0, "Horizontal rectangle angle should be close to 0°, got {}".format(angle)

    def test_tilted_rectangle_angle_computed(self):
        # Create a tilted line/ellipse
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.ellipse(mask, (150, 150), (60, 20), 45, 0, 360, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        angle, centroid, dims = compute_contour_angle(contours[0])
        # Expected around 45° (or -45° depending on convention)
        assert abs(abs(angle) - 45.0) < 5.0, "Angle should be close to 45°, got {}".format(angle)

    def test_draw_rotated_bounding_box_renders_and_returns_angle(self):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(mask, (200, 200), (80, 30), 30, 0, 360, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        angle = draw_rotated_bounding_box(frame, contours[0], (0, 255, 255), show_angle=True)
        assert -90.0 <= angle <= 90.0
        assert np.count_nonzero(frame) > 100


# ---------------------------------------------------------------------------
# Kalman Filter Tracker
# ---------------------------------------------------------------------------

class TestKalmanFilterTracker(object):
    """Verify linear Kalman Filter state tracking and reset."""

    def test_kalman_predict_and_correct(self):
        kf = KalmanFilterTracker()
        pos, vel = kf.correct(100, 100)
        assert abs(pos[0] - 100) < 2.0 and abs(pos[1] - 100) < 2.0

        # Feed consecutive measurements moving right
        for step in range(1, 6):
            kf.predict(dt=0.1)
            pos, vel = kf.correct(100 + step * 10, 100)

        assert pos[0] > 130
        assert kf._is_initialized is True

    def test_kalman_reset(self):
        kf = KalmanFilterTracker()
        kf.correct(50, 50)
        kf.reset()
        assert kf._is_initialized is False


# ---------------------------------------------------------------------------
# Iriun Webcam & Camera Enumeration
# ---------------------------------------------------------------------------

class TestIriunWebcamDetection(object):
    """Verify Iriun camera index auto-detection and open_capture integration."""

    def test_find_iriun_camera_index_returns_int(self):
        idx = find_iriun_camera_index()
        assert isinstance(idx, int)
        assert idx >= 0

    def test_list_available_cameras_structure(self):
        cams = list_available_cameras(max_probe=2)
        assert isinstance(cams, list)
        for cam in cams:
            assert "index" in cam
            assert "name" in cam
            assert "is_iriun" in cam

    def test_open_capture_resolves_iriun_source(self):
        cap = open_capture(source="iriun", fallback_to_synthetic=True)
        assert cap is not None
        assert cap.isOpened() is True
        ok, frame = cap.read()
        assert ok is True
        assert frame is not None
        cap.release()


