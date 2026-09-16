"""
test_bounds.py
==============
Unit tests for HSV colour boundary verification.

These tests validate the correctness of the HSV range constants defined in
src/config.py, and verify that cv2.inRange calls produce the expected binary
masks for canonical colour samples drawn from each preset's specified range.

Run with:
    pytest tests/test_bounds.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COLOR_PRESETS


# ---------------------------------------------------------------------------
# Helper: create a 1x1 HSV image and mask it
# ---------------------------------------------------------------------------

def _single_pixel_mask(hue, sat, val, preset_name):
    """
    Create a 1x1 pixel HSV image and apply the named preset's inRange.

    Returns the sum of the mask (0 = not detected, 255 = detected).
    For presets with a secondary range (Red), both sub-ranges are checked
    and their results are OR-ed.
    """
    pixel  = np.array([[[hue, sat, val]]], dtype=np.uint8)
    preset = COLOR_PRESETS[preset_name]

    mask = cv2.inRange(pixel, preset["lower1"], preset["upper1"])

    if preset["lower2"] is not None and preset["upper2"] is not None:
        mask2 = cv2.inRange(pixel, preset["lower2"], preset["upper2"])
        mask  = cv2.bitwise_or(mask, mask2)

    return int(mask.sum())


# ---------------------------------------------------------------------------
# Structural bound consistency
# ---------------------------------------------------------------------------

class TestBoundConsistency(object):
    """Verify that every preset's lower bounds <= upper bounds component-wise."""

    @pytest.mark.parametrize("color_name", list(COLOR_PRESETS.keys()))
    def test_lower1_le_upper1(self, color_name):
        preset = COLOR_PRESETS[color_name]
        lower  = preset["lower1"]
        upper  = preset["upper1"]
        assert np.all(lower <= upper), (
            "{}: lower1={} > upper1={} in some dimension".format(
                color_name, lower, upper
            )
        )

    @pytest.mark.parametrize("color_name", ["Red"])
    def test_lower2_le_upper2_for_red(self, color_name):
        preset = COLOR_PRESETS[color_name]
        lower2 = preset["lower2"]
        upper2 = preset["upper2"]
        assert lower2 is not None and upper2 is not None, (
            "Red preset must define lower2 and upper2"
        )
        assert np.all(lower2 <= upper2), (
            "Red: lower2={} > upper2={}".format(lower2, upper2)
        )

    @pytest.mark.parametrize("color_name", ["Green", "Blue", "Yellow"])
    def test_non_red_has_no_secondary_range(self, color_name):
        preset = COLOR_PRESETS[color_name]
        assert preset["lower2"] is None, (
            "{} should not have a secondary range".format(color_name)
        )
        assert preset["upper2"] is None

    @pytest.mark.parametrize("color_name", list(COLOR_PRESETS.keys()))
    def test_hue_bounds_within_opencv_range(self, color_name):
        """OpenCV Hue must be in [0, 180]."""
        preset = COLOR_PRESETS[color_name]
        assert 0 <= int(preset["lower1"][0]) <= 180
        assert 0 <= int(preset["upper1"][0]) <= 180
        if preset["lower2"] is not None:
            assert 0 <= int(preset["lower2"][0]) <= 180
            assert 0 <= int(preset["upper2"][0]) <= 180

    @pytest.mark.parametrize("color_name", list(COLOR_PRESETS.keys()))
    def test_saturation_bounds_in_range(self, color_name):
        """Saturation bounds must be in [0, 255]."""
        preset = COLOR_PRESETS[color_name]
        assert 0 <= int(preset["lower1"][1]) <= 255
        assert 0 <= int(preset["upper1"][1]) <= 255

    @pytest.mark.parametrize("color_name", list(COLOR_PRESETS.keys()))
    def test_value_bounds_in_range(self, color_name):
        """Value bounds must be in [0, 255]."""
        preset = COLOR_PRESETS[color_name]
        assert 0 <= int(preset["lower1"][2]) <= 255
        assert 0 <= int(preset["upper1"][2]) <= 255

    @pytest.mark.parametrize("color_name", list(COLOR_PRESETS.keys()))
    def test_bgr_display_is_valid_3tuple(self, color_name):
        """bgr_display must be a 3-element tuple of ints in [0, 255]."""
        bgr = COLOR_PRESETS[color_name]["bgr_display"]
        assert len(bgr) == 3, \
            "{}: bgr_display must have 3 components".format(color_name)
        for c in bgr:
            assert 0 <= c <= 255, \
                "{}: bgr component {} out of range".format(color_name, c)


# ---------------------------------------------------------------------------
# Canonical in-range pixel detection
# ---------------------------------------------------------------------------

class TestCanonicalPixelDetection(object):
    """
    Verify that hand-picked canonical HSV samples for each colour are
    correctly detected (mask = 255) by the preset bounds.
    """

    def test_red_low_hue_canonical_detected(self):
        """H=5, S=200, V=200 should be detected as Red (low sub-range)."""
        result = _single_pixel_mask(5, 200, 200, "Red")
        assert result == 255, "Expected 255, got {}".format(result)

    def test_red_high_hue_canonical_detected(self):
        """H=170, S=200, V=200 should be detected as Red (high sub-range)."""
        result = _single_pixel_mask(170, 200, 200, "Red")
        assert result == 255, "Expected 255, got {}".format(result)

    def test_green_canonical_detected(self):
        """H=60, S=200, V=200 should be detected as Green."""
        result = _single_pixel_mask(60, 200, 200, "Green")
        assert result == 255, "Expected 255, got {}".format(result)

    def test_blue_canonical_detected(self):
        """H=110, S=200, V=200 should be detected as Blue."""
        result = _single_pixel_mask(110, 200, 200, "Blue")
        assert result == 255, "Expected 255, got {}".format(result)

    def test_yellow_canonical_detected(self):
        """H=25, S=200, V=200 should be detected as Yellow."""
        result = _single_pixel_mask(25, 200, 200, "Yellow")
        assert result == 255, "Expected 255, got {}".format(result)


# ---------------------------------------------------------------------------
# Canonical out-of-range pixel rejection
# ---------------------------------------------------------------------------

class TestOutOfRangeRejection(object):
    """
    Verify that HSV values outside each preset's range are NOT detected.
    """

    def test_green_not_detected_as_red(self):
        """Pure green (H=60) must not trigger the red mask."""
        result = _single_pixel_mask(60, 200, 200, "Red")
        assert result == 0, "Green falsely detected as Red: {}".format(result)

    def test_blue_not_detected_as_red(self):
        """Blue (H=110) must not trigger the red mask."""
        result = _single_pixel_mask(110, 200, 200, "Red")
        assert result == 0

    def test_yellow_not_detected_as_blue(self):
        """Yellow (H=25) must not trigger the blue mask."""
        result = _single_pixel_mask(25, 200, 200, "Blue")
        assert result == 0

    def test_red_low_not_detected_as_green(self):
        """Low-hue red (H=5) must not trigger the green mask."""
        result = _single_pixel_mask(5, 200, 200, "Green")
        assert result == 0

    def test_red_high_not_detected_as_green(self):
        """High-hue red (H=170) must not trigger the green mask."""
        result = _single_pixel_mask(170, 200, 200, "Green")
        assert result == 0

    def test_blue_not_detected_as_green(self):
        """Blue (H=110) must not trigger the green mask."""
        result = _single_pixel_mask(110, 200, 200, "Green")
        assert result == 0

    def test_green_not_detected_as_yellow(self):
        """Green (H=60) should not be detected as Yellow (max hue 35)."""
        result = _single_pixel_mask(60, 200, 200, "Yellow")
        assert result == 0

    def test_achromatic_grey_rejected_by_saturation(self):
        """A grey pixel (S=10) must not be detected by any colour preset."""
        for color_name in COLOR_PRESETS:
            result = _single_pixel_mask(60, 10, 200, color_name)
            assert result == 0, (
                "Grey pixel detected as {}: {}".format(color_name, result)
            )

    def test_near_black_rejected_by_value(self):
        """A near-black pixel (V=20) must not be detected by any preset."""
        for color_name in COLOR_PRESETS:
            result = _single_pixel_mask(60, 200, 20, color_name)
            assert result == 0, (
                "Near-black pixel detected as {}: {}".format(color_name, result)
            )


# ---------------------------------------------------------------------------
# Red hue sub-range properties
# ---------------------------------------------------------------------------

class TestRedHueBandProperties(object):
    """
    Structural assertions about the two red hue sub-ranges:
    - Sub-range 1: low-hue band near 0 degrees (H = [0, ~10])
    - Sub-range 2: high-hue band near 180 degrees (H = [~160, 180])
    """

    def test_red_has_two_distinct_hue_sub_ranges(self):
        """The two red hue bands must be non-overlapping."""
        red      = COLOR_PRESETS["Red"]
        upper_h1 = int(red["upper1"][0])
        lower_h2 = int(red["lower2"][0])
        assert upper_h1 < lower_h2, (
            "Red hue sub-ranges overlap: upper1_H={}, lower2_H={}".format(
                upper_h1, lower_h2
            )
        )

    def test_red_sub_range1_starts_at_zero(self):
        """The low-hue red sub-range should start at or near 0."""
        red = COLOR_PRESETS["Red"]
        assert int(red["lower1"][0]) <= 5

    def test_red_sub_range2_ends_at_180(self):
        """The high-hue red sub-range should end at 180 (max OpenCV hue)."""
        red = COLOR_PRESETS["Red"]
        assert int(red["upper2"][0]) == 180

    def test_mid_hue_gap_between_red_ranges(self):
        """Confirm a significant gap (>= 100 hue units) between the two red bands."""
        red = COLOR_PRESETS["Red"]
        gap = int(red["lower2"][0]) - int(red["upper1"][0])
        assert gap >= 100, "Gap between red sub-ranges is too small: {}".format(gap)

    def test_hue_140_not_detected_as_red(self):
        """H=140 falls in the gap between red sub-ranges and must be rejected."""
        result = _single_pixel_mask(140, 200, 200, "Red")
        assert result == 0, "H=140 falsely detected as Red: {}".format(result)


# ---------------------------------------------------------------------------
# Non-overlap between distinct colour presets
# ---------------------------------------------------------------------------

class TestPresetHueNonOverlap(object):
    """Verify that no two non-Red presets share the same canonical hue."""

    _CANONICAL_HUES = {
        "Green":  60,
        "Blue":   110,
        "Yellow": 25,
    }

    @pytest.mark.parametrize("detecting_color,target_hue_color", [
        ("Green",  "Blue"),
        ("Green",  "Yellow"),
        ("Blue",   "Green"),
        ("Blue",   "Yellow"),
        ("Yellow", "Green"),
        ("Yellow", "Blue"),
    ])
    def test_cross_detection_absent(self, detecting_color, target_hue_color):
        """A canonical pixel for target_hue_color must NOT trigger detecting_color."""
        hue    = self._CANONICAL_HUES[target_hue_color]
        result = _single_pixel_mask(hue, 200, 200, detecting_color)
        assert result == 0, (
            "{} falsely detected H={} ({}): {}".format(
                detecting_color, hue, target_hue_color, result
            )
        )
