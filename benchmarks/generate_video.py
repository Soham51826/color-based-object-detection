"""
benchmarks/generate_video.py
============================
Root-level entry point forwarding to color_object_detection.benchmarks.generate_video.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PKG_PATH = _ROOT / "color_object_detection"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from color_object_detection.benchmarks.generate_video import (
    generate_benchmark_video,
    main,
)

if __name__ == "__main__":
    main()
