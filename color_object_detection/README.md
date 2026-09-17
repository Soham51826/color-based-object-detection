# Color-Based Object Detection and Real-Time Tracking System

> **Production-Ready, Modular Computer Vision Pipeline for Real-Time Color Detection and Tracking**

A production-ready, modular Python system that uses the HSV colour space to
detect and track coloured objects in real time via a webcam or pre-recorded
video file.  The system ships with a modern desktop GUI, a quantitative
benchmark harness, and a comprehensive unit-test suite.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Step-by-Step: How to Run](#step-by-step-how-to-run)
   - [Step 1 — Open Terminal & Navigate to Project](#step-1--open-terminal--navigate-to-project)
   - [Step 2 — Launch the GUI Desktop App (Recommended)](#step-2--launch-the-gui-desktop-app-recommended)
   - [Step 3 — Run Live Webcam Tracking via CLI](#step-3--run-live-webcam-tracking-via-cli)
   - [Step 4 — Run Automated Unit Tests (85 Tests)](#step-4--run-automated-unit-tests-85-tests)
   - [Step 5 — Run Benchmark Suite & Generate Plots](#step-5--run-benchmark-suite--generate-plots)
5. [Quick Start (Python API)](#quick-start-python-api)
6. [GUI Controls & Dual-Canvas Interface](#gui-controls--dual-canvas-interface)
7. [Running the Benchmark Suite](#running-the-benchmark-suite)
8. [Running the Unit Tests](#running-the-unit-tests)
9. [Algorithm Explanation](#algorithm-explanation)
10. [Configuration Reference](#configuration-reference)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

---

## Features

| Feature | Details |
|---------|---------|
| **Live webcam and file input** | `cv2.VideoCapture` with configurable index or path |
| **Gaussian pre-processing** | 5×5 kernel, σ = auto, suppresses sensor noise |
| **HSV colour space** | Cylindrical representation decouples chromatic and achromatic information |
| **4 target colours** | Red, Green, Blue, Yellow |
| **Red hue wrap-around** | Bitwise-OR of two sub-ranges [0–10] and [160–180] |
| **Morphological cleaning** | Opening (remove noise) + Closing (fill voids) with 7×7 ellipse |
| **Spatial moment centroids** | `x̄ = M10/M00`, `ȳ = M01/M00` |
| **Desktop GUI** | Tkinter + CustomTkinter, dual video canvas, live sliders |
| **Benchmark harness** | IoU, Precision, Recall, F1, per-stage latency, FPS across 4 resolutions |
| **Synthetic data generator** | Gaussian noise, lux-scaled brightness, glare, occlusion |
| **Unit tests** | 40+ tests covering mask generation, red wrap-around, centroids |

---

## Project Structure

```
color_object_detection/
├── README.md                   ← This file
├── requirements.txt            ← Python dependencies
├── setup.py                    ← Package configuration
│
├── src/
│   ├── __init__.py
│   ├── config.py               ← HSV presets, kernel sizes, constants
│   ├── tracker.py              ← Core CV pipeline (ColorTracker class)
│   ├── utils.py                ← FPS counter, drawing helpers, StageTimer
│   └── gui_app.py              ← Desktop GUI application
│
├── benchmarks/
│   ├── __init__.py
│   ├── synthetic_data.py       ← Synthetic scene generator
│   └── evaluate.py             ← Benchmark suite + matplotlib plots
│
└── tests/
    ├── __init__.py
    ├── test_tracker.py         ← Tracker, mask, blob, centroid tests
    └── test_bounds.py          ← HSV boundary verification tests
```

---

## Prerequisites & Installation

### System Requirements
- **Python**: Version **3.6.8 or newer** (tested on Python 3.6, 3.8, 3.10, and 3.12).
- **Webcam**: Built-in or external USB webcam (optional; synthetic benchmarks and pre-recorded videos work without a camera).
- **OS**: Windows 10/11, macOS, or Linux.

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Soham51826/color-based-object-detection.git
cd color-based-object-detection
```

### Step 2 — Create and Activate a Virtual Environment (Optional)
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
cd color_object_detection
pip install -r requirements.txt
```

---

## Step-by-Step: How to Run

### Step 1 — Open Terminal & Navigate to Project
Make sure your terminal is inside the `color_object_detection` directory where `src/` is located:
```bash
# Navigate into the package directory:
cd color_object_detection

# (Optional) Set up local environment configuration
cp .env.example .env
```

---

### Step 2 — Launch the GUI Desktop App (Recommended)
The full-featured graphical application provides real-time dual-canvas visualization, calibration sliders, and snapshot tools:

```bash
# Launch with default webcam (index 0)
python -m src.gui_app

# Launch with a specific external webcam index (e.g. index 1)
python -m src.gui_app --source 1

# Launch with a pre-recorded video file
python -m src.gui_app --source path/to/sample_video.mp4
```

> **What happens**:
> - A desktop window titled **"Color-Based Object Detection & Tracking System"** will appear.
> - Click **▶ Start Feed** to start streaming from your camera.
> - Select any color (**Red**, **Green**, **Blue**, **Yellow**, or **Custom**) to isolate its binary mask on the right canvas.
> - Adjust the **H / S / V sliders** in real time to fine-tune detection for your lighting conditions.
> - Click **📸 Snapshot** to save timestamped detection results to the `snapshots/` folder.

---

### Step 3 — Run Live Webcam Tracking via CLI
If you prefer a lightweight OpenCV window without the Tkinter GUI:

```bash
# Run tracker on default camera (index 0) tracking all 4 colors
python -m src.tracker --source 0

# Track specific colors only (e.g., Red and Blue)
python -m src.tracker --source 0 --colors Red Blue

# Run against a video file
python -m src.tracker --source path/to/video.mp4
```

> **Controls**: Press <kbd>q</kbd> or <kbd>Esc</kbd> while focused on the OpenCV window to exit.

---

### Step 4 — Run Automated Unit Tests (85 Tests)
Run the test suite to verify HSV boundary math, red wrap-around union, centroid calculations, and noise filtering:

```bash
# Run all tests with verbose output
python -m pytest tests -v

# Run with test coverage report
python -m pytest tests -v --cov=src
```

> **Expected output**: All **85 passed** in ~1.3 seconds.

---

### Step 5 — Run Benchmark Suite & Generate Plots
Run the quantitative benchmark harness to evaluate detection accuracy (Precision, Recall, F1, IoU) and latency across resolutions (QVGA to 1080p):

```bash
# Run default benchmark (generates synthetic scenes & evaluation charts)
python -m benchmarks.evaluate

# Custom output directory
python -m benchmarks.evaluate --out ./benchmark_results/

# High-precision benchmark (more frames, slower)
python -m benchmarks.evaluate --frames 500
```

> **Generated Plots (saved to `benchmark_results/`)**:
> - `benchmark_curves.png`: End-to-end FPS vs. resolution + F1-score/IoU vs. ambient lux.
> - `confusion_matrix.png`: Per-class Precision, Recall, and F1 performance heatmap.
> - `pipeline_stages.png`: Detailed microsecond latency breakdown per pipeline stage.

---

## Quick Start (Python API)

### Headless test (no GUI, no camera needed)

```python
import cv2
from src.tracker import ColorTracker

tracker = ColorTracker(color_names=["Red", "Green"])
cap = cv2.VideoCapture(0)          # webcam index 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = tracker.process_frame(frame)
    cv2.imshow("Detection", result.annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Launching the GUI Application

```bash
# From the project root (color_object_detection/)

# Default webcam (index 0)
python -m src.gui_app

# Specific webcam index
python -m src.gui_app --source 1

# Pre-recorded video file
python -m src.gui_app --source path/to/video.mp4
```

If the package is installed in editable mode, you can also use:

```bash
color-detect --source 0
```

### GUI Controls

| Control | Description |
|---------|-------------|
| **▶ Start Feed** | Opens the video source and begins real-time detection |
| **⏸ Pause / Resume** | Freezes or resumes the capture worker thread |
| **🔄 Reset Calibration** | Restores HSV sliders to the selected colour preset's defaults |
| **📸 Snapshot** | Saves the current annotated frame as `snapshots/snapshot_<timestamp>.png` |
| **Colour Preset** | Dropdown to select Red / Green / Blue / Yellow / Custom |
| **H Min / H Max** | Hue threshold sliders (0 – 180 OpenCV units) |
| **S Min / S Max** | Saturation threshold sliders (0 – 255) |
| **V Min / V Max** | Value (brightness) threshold sliders (0 – 255) |
| **Min Area** | Minimum contour area (px²) — increase to suppress small false positives |

### Dual Canvas Layout

```
┌─────────────────────┬─────────────────────┐
│  Live Camera Feed   │   Binary Mask View  │
│  (BGR + overlays)   │  (selected colour)  │
└─────────────────────┴─────────────────────┘
```

The left canvas shows the raw camera image with bounding boxes, centroid
crosshairs, colour labels, and coordinate annotations.

The right canvas shows the binary morphological mask for the currently
selected colour — white pixels are candidate detections, black pixels are
background.

---

## Running the Benchmark Suite

```bash
# From the project root
python -m benchmarks.evaluate

# Save plots to a custom directory
python -m benchmarks.evaluate --out ./results/

# Use more frames for higher statistical accuracy (slower)
python -m benchmarks.evaluate --frames 500
```

The benchmark will:

1. **Stage latency** — measure mean wall-clock time (ms) for each pipeline
   stage (Gaussian, HSV, inRange+Morph, Contours+Moments) at QVGA, VGA,
   HD 720p, and FHD 1080p using synthetic frames.

2. **Throughput** — measure end-to-end FPS across the same four resolutions.

3. **Accuracy vs. illuminance** — compute IoU, Precision, Recall, and F1
   over 100 synthetic frames per illuminance level (50–1600 Lux).

4. **Per-class accuracy** — compute P/R/F1 separately for Red, Green, Blue,
   and Yellow.

Output files saved to `--out` directory:

| File | Contents |
|------|---------|
| `benchmark_curves.png` | FPS vs. resolution + F1/IoU vs. Lux |
| `confusion_matrix.png` | Per-class P/R/F1 heatmap |
| `pipeline_stages.png` | Per-stage latency grouped bar chart |

---

## Running the Unit Tests

```bash
# From the project root
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run only HSV boundary tests
pytest tests/test_bounds.py -v

# Run only tracker / centroid tests
pytest tests/test_tracker.py -v
```

### Test Coverage Summary

| Module | Tests |
|--------|-------|
| `test_tracker.py` | Tracker init, mask shape, green mask, red wrap-around (6 sub-tests), centroid, FrameResult structure, blob detection, dynamic HSV update, FPS counter, StageTimer |
| `test_bounds.py` | Bound consistency, canonical pixel detection, out-of-range rejection, red sub-range structure, cross-colour non-overlap |

---

## Algorithm Explanation

### 1. Frame Ingestion

Video frames are captured using `cv2.VideoCapture`. For webcams the preferred
resolution and FPS are requested via `CAP_PROP_*` properties; the driver
chooses the closest supported mode.

### 2. Gaussian Pre-Processing

```
I_smooth = GaussianBlur(I_raw, kernel=(5,5), sigma=0)
```

A 5×5 Gaussian kernel (σ computed automatically from kernel size) is convolved
with the input frame to attenuate high-frequency sensor noise (photon shot noise,
readout noise, JPEG compression ringing) before the colour analysis step.
The kernel size is a trade-off: too small → noise survives; too large → blurs
colour boundaries and reduces centroid precision.

### 3. BGR → HSV Colour Space Conversion

```
I_hsv = cvtColor(I_smooth, BGR2HSV)
```

OpenCV's HSV representation decouples **chromatic** information (Hue) from
**achromatic** information (Saturation, Value).  This makes colour segmentation
robust to illumination changes: a colour that changes brightness stays in the
same hue band, whereas in RGB it would drift across multiple channels.

OpenCV conventions:
- **H ∈ [0, 180]** (degrees / 2, to fit in uint8)
- **S ∈ [0, 255]** (0 = grey, 255 = fully saturated)
- **V ∈ [0, 255]** (0 = black, 255 = maximum brightness)

### 4. In-Range Masking

For each target colour a binary mask is computed:

```
mask = inRange(I_hsv, lower_hsv, upper_hsv)
```

Pixels within the HSV bounding box become 255 (white); all others become 0.

#### Special Case: Red Hue Wrap-Around

Red occupies **two** disjoint sub-ranges in OpenCV HSV because the physical
hue angle (0° for pure red on the standard colour wheel) maps to both
H ≈ 0 and H ≈ 180 in the [0, 180] representation:

```
mask_low  = inRange(I_hsv, [0,  100, 100], [10,  255, 255])
mask_high = inRange(I_hsv, [160, 100, 100], [180, 255, 255])
mask_red  = bitwise_or(mask_low, mask_high)
```

Failing to union these two sub-ranges causes partial red objects (those with
hue near 180°) to be silently missed — a common bug in student implementations.

### 5. Morphological Cleaning

A two-stage morphological filter is applied with a 7×7 elliptical structuring
element `K`:

**Opening** (erosion → dilation): removes isolated noise pixels (salt noise,
background flicker specks) smaller than the structuring element.

```
mask = dilate(erode(mask, K), K)
```

**Closing** (dilation → erosion): fills small holes inside detected blobs
caused by specular reflections (bright spots on shiny surfaces that wash out
the colour channel).

```
mask = erode(dilate(mask, K), K)
```

### 6. Contour Extraction

```
contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
```

`RETR_EXTERNAL` returns only the outermost contours, ignoring nested holes.
`CHAIN_APPROX_SIMPLE` compresses straight segments (horizontal, vertical,
diagonal) into two endpoints, reducing memory usage.

### 7. Area Filtering

Contours with area < `MIN_CONTOUR_AREA` (default 500 px²) are discarded.
This threshold should be tuned based on the expected object size at the
working distance.

### 8. Spatial Moment Centroid

The centroid **(x̄, ȳ)** of each detected blob is computed using first-order
image moments:

```
M = moments(contour)
x̄ = M['m10'] / M['m00']
ȳ = M['m01'] / M['m00']
```

- **M₀₀** = zeroth moment = total area (same as `contourArea`)
- **M₁₀** = Σ(x · pixel) = first horizontal moment
- **M₀₁** = Σ(y · pixel) = first vertical moment

This gives the *centre of mass* of the blob, which is more stable and
accurate than the bounding-box centre, especially for non-convex or
irregularly shaped objects.

### 9. Annotation

For each detected blob, the pipeline draws:
- **Bounding rectangle** using `cv2.rectangle`
- **Centroid crosshair** (horizontal line + vertical line + filled dot)
- **Text label**: `<ColourName> (cx,cy) A=<area>`

A per-colour status HUD in the top-right corner shows TRACKING (green) or
LOST (red) for each target colour.

---

## Configuration Reference

All tunable parameters are in [`src/config.py`](src/config.py):

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_CAMERA_INDEX` | 0 | `cv2.VideoCapture` source index |
| `CAPTURE_WIDTH` | 640 | Preferred capture width (px) |
| `CAPTURE_HEIGHT` | 480 | Preferred capture height (px) |
| `CAPTURE_FPS` | 30 | Requested FPS from capture device |
| `GAUSSIAN_KERNEL_SIZE` | (5, 5) | Gaussian blur kernel |
| `MORPH_KERNEL_SIZE` | (7, 7) | Morphological structuring element size |
| `MORPH_OPEN_ITERATIONS` | 2 | Opening iterations |
| `MORPH_CLOSE_ITERATIONS` | 2 | Closing iterations |
| `MIN_CONTOUR_AREA` | 500 | Minimum blob area (px²) |
| `MAX_BLOBS_PER_COLOR` | 5 | Max simultaneous tracked blobs per colour |
| `FPS_SMOOTHING_WINDOW` | 30 | Rolling FPS counter window size |
| `IOU_TP_THRESHOLD` | 0.5 | Minimum IoU to count as a True Positive |

### Custom HSV Presets

To add or modify colour presets, edit `COLOR_PRESETS` in `config.py`:

```python
COLOR_PRESETS["Orange"] = {
    "lower1": np.array([10, 120, 120], dtype=np.uint8),
    "upper1": np.array([22, 255, 255], dtype=np.uint8),
    "lower2": None,
    "upper2": None,
    "bgr_display": (0, 128, 255),
}
```

---

## Troubleshooting

### Camera not detected / Synthetic Fallback

If no physical camera is detected, the application automatically falls back to an animated synthetic test feed with moving colored targets. This enables full UI testing, threshold tuning, and pipeline evaluation on headless or camera-less machines without crashes.

To connect a physical camera:
- Try `--source 1`, `--source 2`, etc. to select alternative camera indices.
- On Linux, ensure camera permissions: `sudo chmod a+rw /dev/video0`
- On Windows, ensure camera access is allowed under *Settings → Privacy & Security → Camera*.

### CustomTkinter not installed

The GUI falls back to standard Tkinter automatically.  For the modern themed
UI, install CustomTkinter:

```bash
pip install customtkinter
```

### Low FPS

- Reduce `CAPTURE_WIDTH` and `CAPTURE_HEIGHT` in `config.py`.
- Reduce `MORPH_OPEN_ITERATIONS` and `MORPH_CLOSE_ITERATIONS` to 1.
- Increase `MIN_CONTOUR_AREA` to skip small blobs faster.
- Disable unused colour channels (pass only needed names to `ColorTracker`).

### HSV calibration tips

1. Use the **🔄 Reset Calibration** button to start from the preset defaults.
2. Hold the target object under consistent lighting in front of the camera.
3. Adjust **V Min** upward to reject shadows; adjust **V Max** downward if
   bright specular highlights cause false negatives.
4. Adjust **S Min** upward to reject whites and greys.
5. Narrow the Hue range to reduce false positives from similarly-coloured
   background objects.

---

## License

MIT License.  See `LICENSE` for details.

---

*Maintained by open-source contributors. Released under the MIT License.*
