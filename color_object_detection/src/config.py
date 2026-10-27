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

import cv2 as _cv2  # type: ignore
import numpy as np  # type: ignore

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

#: Bilateral filter diameter (pixel neighbourhood diameter).
#: Larger d => slower but more smoothing while preserving edges.
BILATERAL_D = 5

#: Bilateral filter sigma for color space smoothing.
BILATERAL_SIGMA_COLOR = 50

#: Bilateral filter sigma for coordinate space smoothing.
BILATERAL_SIGMA_SPACE = 50

# ---------------------------------------------------------------------------
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ---------------------------------------------------------------------------

#: CLAHE clip limit — controls contrast amplification ceiling.
CLAHE_CLIP_LIMIT = 2.0

#: Tile grid size for CLAHE.  8x8 is a balanced default.
CLAHE_TILE_GRID_SIZE = (8, 8)

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

#: Minimum solidity (area / convex_hull_area) to accept a blob.
#: Rejects elongated background streaks and fragmented blobs.
MIN_SOLIDITY = 0.65

# ---------------------------------------------------------------------------
# Temporal smoothing (EMA centroid / bounding-box stabiliser)
# ---------------------------------------------------------------------------

#: EMA alpha for centroid smoothing.  Higher = less smoothing / more responsive.
EMA_ALPHA = 0.7

#: Length of the motion-trail history deque (frames).
TRAIL_MAX_LEN = 32

# ---------------------------------------------------------------------------
# HSV colour presets  (9 chromatic classes)
# ---------------------------------------------------------------------------
# Each entry is a dict with:
#   "lower1" / "upper1" : primary HSV range as numpy uint8 arrays
#   "lower2" / "upper2" : secondary HSV range (only for red hue wrap-around)
#   "bgr_display"       : BGR tuple used when drawing overlays on the frame

COLOR_PRESETS = {
    "Red": {
        # Red hue wraps around 0/180 in OpenCV HSV space.
        # Finer bounds tighten rejection of orange (S >= 110, V >= 80).
        "lower1": np.array([0,   110, 80],  dtype=np.uint8),
        "upper1": np.array([10,  255, 255], dtype=np.uint8),
        "lower2": np.array([165, 110, 80],  dtype=np.uint8),
        "upper2": np.array([180, 255, 255], dtype=np.uint8),
        "bgr_display": (0, 0, 255),
    },
    "Orange": {
        "lower1": np.array([11, 120, 100], dtype=np.uint8),
        "upper1": np.array([21, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "bgr_display": (0, 128, 255),
    },
    "Yellow": {
        "lower1": np.array([22, 100, 100], dtype=np.uint8),
        "upper1": np.array([34, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "bgr_display": (0, 255, 255),
    },
    "Green": {
        "lower1": np.array([35,  85, 60],  dtype=np.uint8),
        "upper1": np.array([84,  255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "bgr_display": (0, 255, 0),
    },
    "Cyan": {
        "lower1": np.array([85,  100, 70],  dtype=np.uint8),
        "upper1": np.array([99,  255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "bgr_display": (255, 255, 0),
    },
    "Blue": {
        "lower1": np.array([100, 110, 60],  dtype=np.uint8),
        "upper1": np.array([128, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "bgr_display": (255, 0, 0),
    },
    "Purple": {
        "lower1": np.array([129, 80, 60],   dtype=np.uint8),
        "upper1": np.array([148, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "bgr_display": (128, 0, 128),
    },
    "Pink": {
        "lower1": np.array([149, 80, 100],  dtype=np.uint8),
        "upper1": np.array([164, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "bgr_display": (147, 20, 255),
    },
    "White": {
        # Achromatic / high-brightness targets: very low saturation, high value.
        "lower1": np.array([0,   0,   200], dtype=np.uint8),
        "upper1": np.array([180, 35,  255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "bgr_display": (220, 220, 220),
    },
}

# ---------------------------------------------------------------------------
# Overlay / drawing settings
# ---------------------------------------------------------------------------

BBOX_THICKNESS = 2
CENTROID_DOT_RADIUS = 5
CROSSHAIR_ARM_LENGTH = 15
CROSSHAIR_THICKNESS = 2

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

# ---------------------------------------------------------------------------
# Kalman Filter & Speed Estimation settings
# ---------------------------------------------------------------------------

#: Process noise covariance parameter for 4-state Kalman Filter (position + velocity)
KALMAN_PROCESS_NOISE = float(os.getenv("KALMAN_PROCESS_NOISE", "1e-2"))

#: Measurement noise covariance parameter for Kalman Filter
KALMAN_MEASUREMENT_NOISE = float(os.getenv("KALMAN_MEASUREMENT_NOISE", "1e-1"))

#: EMA smoothing factor for instantaneous speed calculation (0 < alpha <= 1)
SPEED_SMOOTHING_ALPHA = 0.6

#: Minimum speed in px/s to display velocity arrow and movement badge
SPEED_MIN_DISPLAY_THRESH = 3.0

#: Maximum number of missed frames to sustain trajectory before reset
MAX_MISSED_TRACK_FRAMES = 8

# ---------------------------------------------------------------------------
# Iriun Webcam & Live Streaming settings
# ---------------------------------------------------------------------------

#: Candidate camera indices searched when attempting auto-detection of Iriun Webcam
IRIUN_SEARCH_INDICES = [1, 2, 3, 4, 0]

#: Keywords searched in Windows PnP camera device names to identify Iriun / phone webcams
IRIUN_DEVICE_KEYWORDS = ["iriun", "phone", "droidcam", "android", "epoccam", "ivcam"]

# ---------------------------------------------------------------------------
# Metric Distance & Camera Pinhole Model settings
# ---------------------------------------------------------------------------

#: Estimated camera focal length in pixels (standard for ~60° FOV at 640x480)
PINHOLE_FOCAL_LENGTH_PX = float(os.getenv("PINHOLE_FOCAL_LENGTH_PX", "700.0"))

#: Assumed real-world physical diameter/width of tracked target in centimeters
#: (e.g. 6.5cm standard tennis ball, 5.0cm bottle cap/mug)
REAL_OBJECT_WIDTH_CM = float(os.getenv("REAL_OBJECT_WIDTH_CM", "6.5"))

# ---------------------------------------------------------------------------
# Illumination Robustness & Color Constancy
# ---------------------------------------------------------------------------

#: Enable Gray World color constancy algorithm to eliminate warm/cool lighting casts
USE_GRAY_WORLD = False

# ---------------------------------------------------------------------------
# Analytics & Recording settings
# ---------------------------------------------------------------------------

#: Default FPS for live video session recording
RECORDING_FPS = 30.0

#: Decay factor for 2D trajectory density heatmap
HEATMAP_DECAY = 0.999

