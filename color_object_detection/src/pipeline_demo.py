"""
pipeline_demo.py
================
Educational pipeline demonstration and visual step-by-step walkthrough.

This module demonstrates exactly how the Color-Based Object Detection
system processes an image frame through each of its 10 computer vision stages.
It outputs:
1. A detailed terminal summary explaining each stage's mathematical purpose.
2. A composite visual comparison image saved to `benchmark_results/pipeline_demonstration.png`
   showing the frame at each intermediate transformation.

Usage
-----
    python run.py --demo
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config
from .tracker import ColorTracker
from .utils import (
    annotate_detection,
    compute_centroid,
    compute_circularity,
    compute_solidity,
    get_logger,
)

logger = get_logger(__name__)


def run_pipeline_demo(save_path: Optional[Path] = None, show_window: bool = False) -> Path:
    """Execute the full 10-stage detection pipeline on a synthetic test scene.

    Parameters
    ----------
    save_path : Path, optional
        Where to save the composite explanation image.
        Defaults to `config.BENCHMARK_RESULTS_DIR / "pipeline_demonstration.png"`.
    show_window : bool, optional
        If True, displays the composite image in an OpenCV window until a key is pressed.

    Returns
    -------
    Path : Location of the saved demonstration graphic.
    """
    if save_path is None:
        save_path = config.BENCHMARK_RESULTS_DIR / "pipeline_demonstration.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 68)
    print("  COLOR-BASED OBJECT DETECTION — 10-STAGE PIPELINE WALKTHROUGH")
    print("=" * 68 + "\n")

    w, h = 640, 480

    # ------------------------------------------------------------------
    # Stage 1: Frame Acquisition
    # ------------------------------------------------------------------
    print("[Stage 1: Frame Acquisition]")
    print("  Input: 640x480 BGR image array (uint8).")
    print("  In OpenCV, images are represented as 3-channel Blue-Green-Red matrices.")

    # Create dark textured canvas
    frame = np.full((h, w, 3), 40, dtype=np.uint8)

    # Draw colored objects
    # 1. Red circle
    cv2.circle(frame, (160, 160), 45, (48, 68, 167), -1)
    # 2. Green rectangle
    cv2.rectangle(frame, (380, 110), (480, 210), (52, 157, 56), -1)
    # 3. Blue triangle
    pts_blue = np.array([[160, 400], [110, 310], [210, 310]], np.int32)
    cv2.fillPoly(frame, [pts_blue], (157, 67, 45))
    # 4. Yellow circle
    cv2.circle(frame, (430, 350), 40, (54, 169, 177), -1)

    # Add realistic sensor noise & fine speckles
    np.random.seed(42)
    noise = np.random.normal(0, 8.0, frame.shape).astype(np.int16)
    noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Stage 2: Gaussian Blur Pre-processing
    # ------------------------------------------------------------------
    print("\n[Stage 2: Gaussian Blur]")
    print("  Formula: G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))")
    print(f"  Kernel: {config.GAUSSIAN_KERNEL_SIZE}, Sigma: Auto-computed.")
    print("  Purpose: Suppresses high-frequency sensor noise and sensor readout artifacts.")
    blurred = cv2.GaussianBlur(noisy_frame, config.GAUSSIAN_KERNEL_SIZE, config.GAUSSIAN_SIGMA)

    # ------------------------------------------------------------------
    # Stage 3: CLAHE Lighting Normalization
    # ------------------------------------------------------------------
    print("\n[Stage 3: CLAHE Lighting Normalization]")
    print("  Contrast-Limited Adaptive Histogram Equalization operates on the L channel")
    print(f"  Clip Limit: {config.CLAHE_CLIP_LIMIT}, Grid Size: {config.CLAHE_TILE_GRID_SIZE}.")
    print("  Purpose: Stabilizes detection against uneven ambient shadows and glare.")
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    l_eq = clahe.apply(l_chan)
    clahe_bgr = cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Stage 4: Color Space Conversion (BGR -> HSV)
    # ------------------------------------------------------------------
    print("\n[Stage 4: BGR -> HSV Color Space Conversion]")
    print("  Why HSV? In standard RGB/BGR, chromaticity is entangled with brightness.")
    print("  HSV separates color into:")
    print("    - Hue (H): Chromatic wavelength angle [0..180 in OpenCV, representing 0..360°]")
    print("    - Saturation (S): Color purity/vibrancy [0..255, 0=gray, 255=pure color]")
    print("    - Value (V): Perceived brightness/illuminance [0..255, 0=black, 255=full bright]")
    hsv = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2HSV)

    # ------------------------------------------------------------------
    # Stage 5: InRange Masking & Red Hue Wrap-Around
    # ------------------------------------------------------------------
    print("\n[Stage 5: InRange Color Thresholding]")
    print("  Thresholds pixels where: lower <= HSV <= upper")
    print("  Special Red Wrap-Around:")
    print("    Because hue is a circular angle, red wraps around 0° / 360°.")
    print("    OpenCV evaluates: (0 <= H <= 10) OR (165 <= H <= 180)")
    red_preset = config.COLOR_PRESETS["Red"]
    m1 = cv2.inRange(hsv, red_preset["lower1"], red_preset["upper1"])
    m2 = cv2.inRange(hsv, red_preset["lower2"], red_preset["upper2"])
    raw_red_mask = cv2.bitwise_or(m1, m2)

    # ------------------------------------------------------------------
    # Stage 6: Morphological Cleaning (Opening + Closing)
    # ------------------------------------------------------------------
    print("\n[Stage 6: Morphological Filtering]")
    print(f"  Structuring Element: Ellipse {config.MORPH_KERNEL_SIZE}")
    print("  1. Opening (Erosion followed by Dilation): Removes isolated noise pixels.")
    print("  2. Closing (Dilation followed by Erosion): Bridges voids and holes inside the shape.")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MORPH_KERNEL_SIZE)
    opened = cv2.morphologyEx(raw_red_mask, cv2.MORPH_OPEN, kernel, iterations=config.MORPH_OPEN_ITERATIONS)
    cleaned_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=config.MORPH_CLOSE_ITERATIONS)

    # ------------------------------------------------------------------
    # Stage 7: Contour Extraction
    # ------------------------------------------------------------------
    print("\n[Stage 7: Contour Extraction]")
    print("  Algorithm: Suzuki85 Border Following (cv2.findContours, RETR_EXTERNAL)")
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Found {len(contours)} external contour(s) for Red target.")

    # ------------------------------------------------------------------
    # Stage 8: Spatial Image Moments & Centroid Calculation
    # ------------------------------------------------------------------
    print("\n[Stage 8: Spatial Moments & Centroids]")
    print("  Formulas:")
    print("    Area: M00 = sum(I(x, y))")
    print("    Centroid X: x_bar = M10 / M00")
    print("    Centroid Y: y_bar = M01 / M00")

    filtered_blobs = []
    contour_vis = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        centroid = compute_centroid(cnt)
        solidity = compute_solidity(cnt)
        circularity = compute_circularity(cnt)

        # --------------------------------------------------------------
        # Stage 9: Geometric Shape Descriptors & Rejection
        # --------------------------------------------------------------
        passed_area = area >= config.MIN_CONTOUR_AREA
        passed_solidity = solidity >= config.MIN_SOLIDITY

        print(f"  Candidate Contour:")
        print(f"    - Area: {area:.1f} px^2 (Min threshold: {config.MIN_CONTOUR_AREA}) -> {'PASS' if passed_area else 'REJECT'}")
        print(f"    - Centroid: {centroid}")
        print(f"    - Solidity: {solidity:.2f} (Min threshold: {config.MIN_SOLIDITY}) -> {'PASS' if passed_solidity else 'REJECT'}")
        print(f"    - Circularity: {circularity:.2f} (Circle=1.0, Square=0.78, Triangle=0.60)")

        # Draw contour on contour_vis
        cv2.drawContours(contour_vis, [cnt], -1, (0, 255, 0), 2)
        hull = cv2.convexHull(cnt)
        cv2.drawContours(contour_vis, [hull], -1, (255, 128, 0), 1)

        if passed_area and passed_solidity:
            filtered_blobs.append((cnt, centroid, area, solidity, circularity))

    # ------------------------------------------------------------------
    # Stage 10: Full Multi-Color Tracking & HUD Overlays
    # ------------------------------------------------------------------
    print("\n[Stage 10: Full Multi-Color Tracking & HUD Output]")
    print("  Applies Exponential Moving Average (EMA) centroid smoothing:")
    print(f"    smoothed_c = alpha * raw_c + (1 - alpha) * prev_c  (alpha={config.EMA_ALPHA})")
    print("  Annotates bounding boxes, centroids, color labels, and motion trails.")

    tracker = ColorTracker()
    result = tracker.process_frame(noisy_frame)
    annotated_hud = result.annotated_frame

    # ------------------------------------------------------------------
    # Build 8-Panel Educational Collage Graphic
    # ------------------------------------------------------------------
    panel_w, panel_h = 320, 240

    def prepare_panel(img, label):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        # Add top banner
        cv2.rectangle(resized, (0, 0), (panel_w, 28), (20, 20, 30), -1)
        cv2.putText(
            resized,
            label,
            (8, 20),
            config.FONT,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return resized

    p1 = prepare_panel(noisy_frame, "1. Input Frame (Sensor Noise)")
    p2 = prepare_panel(blurred, "2. Gaussian Blur (5x5, Noise Suppressed)")
    p3 = prepare_panel(clahe_bgr, "3. CLAHE (Luminance Stabilized)")
    p4 = prepare_panel(hsv, "4. HSV Color Space Representation")
    p5 = prepare_panel(raw_red_mask, "5. Raw Red Mask (Hue Wrap-Around)")
    p6 = prepare_panel(cleaned_mask, "6. Morphologically Cleaned Mask")
    p7 = prepare_panel(contour_vis, "7. Contours & Convex Hulls")
    p8 = prepare_panel(annotated_hud, "8. Final Output (HUD + Multi-Tracking)")

    row1 = np.hstack([p1, p2, p3, p4])
    row2 = np.hstack([p5, p6, p7, p8])
    collage = np.vstack([row1, row2])

    cv2.imwrite(str(save_path), collage)
    print(f"\n[Visual Artifact Saved] -> {save_path.resolve()}")
    print("=" * 68 + "\n")

    if show_window:
        win_title = "Color-Based Detection — 10 Pipeline Stages (Press any key)"
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.imshow(win_title, collage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return save_path


if __name__ == "__main__":
    run_pipeline_demo(show_window="--show" in sys.argv)
