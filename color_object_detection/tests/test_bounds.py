"""
test_bounds.py
==============
Unit tests for HSV colour boundary verification.

These tests validate the correctness of the HSV range constants defined in
src/config.py, and verify that cv2.inRange calls produce the expected binary
masks for canonical colour samples drawn from each preset's specified range.

Now covers all 9 colour presets (Red, Orange, Yellow, Green, Cyan, Blue,
Purple, Pink, White) plus secondary-range validation and hue non-overlap tests.

Run with:
    pytest tests/test_bounds.py -v
"""

import sys
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import pytest  # type: ignore

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
# Structural bound consistency  (all 9 presets)
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

    @pytest.mark.parametrize("color_name", [
        name for name in COLOR_PRESETS if name != "Red"
    ])
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
# Canonical in-range pixel detection  (all 9 colours)
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

    def test_orange_canonical_detected(self):
        """H=16, S=200, V=200 should be detected as Orange."""
        result = _single_pixel_mask(16, 200, 200, "Orange")
        assert result == 255, "Expected 255 for Orange, got {}".format(result)

    def test_yellow_canonical_detected(self):
        """H=28, S=200, V=200 should be detected as Yellow."""
        result = _single_pixel_mask(28, 200, 200, "Yellow")
        assert result == 255, "Expected 255, got {}".format(result)

    def test_green_canonical_detected(self):
        """H=60, S=200, V=200 should be detected as Green."""
        result = _single_pixel_mask(60, 200, 200, "Green")
        assert result == 255, "Expected 255, got {}".format(result)

    def test_cyan_canonical_detected(self):
        """H=92, S=200, V=200 should be detected as Cyan."""
        result = _single_pixel_mask(92, 200, 200, "Cyan")
        assert result == 255, "Expected 255 for Cyan, got {}".format(result)

    def test_blue_canonical_detected(self):
        """H=114, S=200, V=200 should be detected as Blue."""
        result = _single_pixel_mask(114, 200, 200, "Blue")
        assert result == 255, "Expected 255, got {}".format(result)

    def test_purple_canonical_detected(self):
        """H=138, S=150, V=150 should be detected as Purple."""
        result = _single_pixel_mask(138, 150, 150, "Purple")
        assert result == 255, "Expected 255 for Purple, got {}".format(result)

    def test_pink_canonical_detected(self):
        """H=155, S=150, V=200 should be detected as Pink."""
        result = _single_pixel_mask(155, 150, 200, "Pink")
        assert result == 255, "Expected 255 for Pink, got {}".format(result)

    def test_white_canonical_detected(self):
        """H=0, S=10, V=230 should be detected as White (achromatic)."""
        result = _single_pixel_mask(0, 10, 230, "White")
        assert result == 255, "Expected 255 for White, got {}".format(result)


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
        """Blue (H=114) must not trigger the red mask."""
        result = _single_pixel_mask(114, 200, 200, "Red")
        assert result == 0

    def test_yellow_not_detected_as_blue(self):
        """Yellow (H=28) must not trigger the blue mask."""
        result = _single_pixel_mask(28, 200, 200, "Blue")
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
        """Blue (H=114) must not trigger the green mask."""
        result = _single_pixel_mask(114, 200, 200, "Green")
        assert result == 0

    def test_green_not_detected_as_yellow(self):
        """Green (H=60) should not be detected as Yellow (upper hue 34)."""
        result = _single_pixel_mask(60, 200, 200, "Yellow")
        assert result == 0

    def test_orange_not_detected_as_red(self):
        """Orange (H=16) must not trigger the red mask (H max = 10)."""
        result = _single_pixel_mask(16, 200, 200, "Red")
        assert result == 0

    def test_blue_not_detected_as_cyan(self):
        """Blue (H=114) is above Cyan upper bound (99); must not be detected."""
        result = _single_pixel_mask(114, 200, 200, "Cyan")
        assert result == 0

    def test_achromatic_grey_rejected_by_saturation(self):
        """A grey pixel (S=10) must not be detected by any chromatic preset."""
        chromatic = [k for k in COLOR_PRESETS if k != "White"]
        for color_name in chromatic:
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

    def test_white_not_detected_as_chromatic(self):
        """A white pixel (S=5, V=240) must not trigger chromatic presets."""
        chromatic = [k for k in COLOR_PRESETS if k != "White"]
        for color_name in chromatic:
            result = _single_pixel_mask(0, 5, 240, color_name)
            assert result == 0, (
                "White pixel detected as {}: {}".format(color_name, result)
            )


# ---------------------------------------------------------------------------
# Red hue sub-range properties
# ---------------------------------------------------------------------------

class TestRedHueBandProperties(object):
    """
    Structural assertions about the two red hue sub-ranges:
    - Sub-range 1: low-hue band near 0 degrees (H = [0, ~10])
    - Sub-range 2: high-hue band near 180 degrees (H = [~165, 180])
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
# Non-overlap between distinct colour presets  (expanded to 9 colours)
# ---------------------------------------------------------------------------

# Canonical (H, S, V) for each non-White chromatic colour (module-level for
# Python 3.6 compatibility – class-scope variables are not accessible inside
# list comprehensions used in @pytest.mark.parametrize).
_CANONICAL_HSV = {
    "Red":    (5,   200, 200),
    "Orange": (16,  200, 200),
    "Yellow": (28,  200, 200),
    "Green":  (60,  200, 200),
    "Cyan":   (92,  200, 200),
    "Blue":   (114, 200, 200),
    "Purple": (138, 150, 150),
    "Pink":   (155, 150, 200),
}

_CROSS_PAIRS = [
    (t, d)
    for t in _CANONICAL_HSV
    for d in _CANONICAL_HSV
    if t != d
]


class TestPresetHueNonOverlap(object):
    """Verify that canonical hue samples for each colour don't fire wrong masks."""

    @pytest.mark.parametrize("target,detector", _CROSS_PAIRS)
    def test_cross_detection_absent(self, target, detector):
        """Canonical pixel for 'target' colour must NOT fire the 'detector' mask."""
        h, s, v = _CANONICAL_HSV[target]
        result = _single_pixel_mask(h, s, v, detector)
        assert result == 0, (
            "{} falsely detected H={} ({}) with result: {}".format(
                detector, h, target, result
            )
        )
