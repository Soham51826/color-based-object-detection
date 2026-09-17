"""
config.py
=========
Central configuration store for the Color-Based Object Detection System.

All HSV boundary ranges, morphological kernel parameters, display constants,
and camera settings are defined here so that every other module can import
a single source of truth instead of hard-coding magic numbers.

HSV Color Space reference (OpenCV conventions)
-----------------------------------------------
  Hue        : 0 - 180  (degrees / 2)
  Saturation : 0 - 255
  Value      : 0 - 255
"""

import os
from pathlib import Path
import numpy as np

# Load environment variables if python-dotenv is installed
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------

#: Root directory of the package/project
BASE_DIR = Path(__file__).resolve().parent.parent

#: Directory for saving snapshot frames
SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", str(BASE_DIR / "snapshots")))

#: Directory for benchmark plots and outputs
BENCHMARK_RESULTS_DIR = Path(
    os.getenv("BENCHMARK_RESULTS_DIR", str(BASE_DIR / "benchmark_results"))
)

# ---------------------------------------------------------------------------
# Camera / capture settings
# ---------------------------------------------------------------------------

def _get_camera_index(default=0):
    val = os.getenv("DEFAULT_CAMERA_INDEX", str(default))
    try:
        return int(val)
    except ValueError:
        return val

#: Default camera index passed to cv2.VideoCapture(). 0 = first USB/built-in
#: webcam. Override via CLI, GUI, or DEFAULT_CAMERA_INDEX env var.
DEFAULT_CAMERA_INDEX = _get_camera_index(0)

#: Preferred capture width in pixels.
CAPTURE_WIDTH = int(os.getenv("CAPTURE_WIDTH", "640"))

#: Preferred capture height in pixels.
CAPTURE_HEIGHT = int(os.getenv("CAPTURE_HEIGHT", "480"))

#: Requested frames per second from the capture device.
CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", "30"))

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

#: Gaussian blur kernel size (must be odd).  A 5x5 kernel provides effective
#: high-frequency noise suppression while preserving coarse edges.
GAUSSIAN_KERNEL_SIZE = (5, 5)

#: Standard deviation for Gaussian blur. 0 = auto-computed from kernel size.
GAUSSIAN_SIGMA = 0.0

# ---------------------------------------------------------------------------
# Morphological operations
# ---------------------------------------------------------------------------

#: Size of the elliptical structuring element used for opening and closing.
MORPH_KERNEL_SIZE = (7, 7)

#: Number of erosion/dilation iterations for morphological opening.
MORPH_OPEN_ITERATIONS = 2

#: Number of erosion/dilation iterations for morphological closing.
MORPH_CLOSE_ITERATIONS = 2

# ---------------------------------------------------------------------------
# Contour filtering
# ---------------------------------------------------------------------------

#: Contours with a pixel area smaller than this threshold are discarded.
MIN_CONTOUR_AREA = int(os.getenv("MIN_CONTOUR_AREA", "500"))

#: Maximum number of simultaneously tracked blobs per color class.
MAX_BLOBS_PER_COLOR = 5

# ---------------------------------------------------------------------------
# HSV colour presets
# ---------------------------------------------------------------------------
# Each entry is a dict with:
#   "lower1" / "upper1" : primary HSV range as numpy uint8 arrays
#   "lower2" / "upper2" : secondary HSV range (only used for red wrap-around)
#   "bgr_display"       : BGR tuple used when drawing overlays on the frame

COLOR_PRESETS = {
    "Red": {
        # Red hue wraps around 0/180 in OpenCV HSV space.
        # We union two sub-ranges to capture the full red hue band.
        "lower1": np.array([0,   100, 100], dtype=np.uint8),
        "upper1": np.array([10,  255, 255], dtype=np.uint8),
        "lower2": np.array([160, 100, 100], dtype=np.uint8),
        "upper2": np.array([180, 255, 255], dtype=np.uint8),
        "bgr_display": (0, 0, 255),
    },
    "Green": {
        "lower1": np.array([35,  60, 60],  dtype=np.uint8),
        "upper1": np.array([85,  255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "bgr_display": (0, 255, 0),
    },
    "Blue": {
        "lower1": np.array([90,  60, 60],  dtype=np.uint8),
        "upper1": np.array([130, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "bgr_display": (255, 0, 0),
    },
    "Yellow": {
        "lower1": np.array([18,  80,  80],  dtype=np.uint8),
        "upper1": np.array([35,  255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "bgr_display": (0, 255, 255),
    },
}

# ---------------------------------------------------------------------------
# Overlay / drawing settings
# ---------------------------------------------------------------------------

BBOX_THICKNESS = 2
CENTROID_DOT_RADIUS = 5
CROSSHAIR_ARM_LENGTH = 15
CROSSHAIR_THICKNESS = 2

import cv2 as _cv2
FONT = _cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1

# ---------------------------------------------------------------------------
# Performance / FPS smoothing
# ---------------------------------------------------------------------------

FPS_SMOOTHING_WINDOW = int(os.getenv("FPS_SMOOTHING_WINDOW", "30"))

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------

BENCHMARK_RESOLUTIONS = [
    (320,  240),
    (640,  480),
    (1280, 720),
    (1920, 1080),
]

BENCHMARK_FRAMES_PER_RESOLUTION = 200

BENCHMARK_LUX_LEVELS = [50, 200, 400, 800, 1200, 1600]

IOU_TP_THRESHOLD = 0.5
