"""
run.py
======
Unified entry point for the Color-Based Object Detection System.

Run this script directly from the repository root:
    python run.py                   # Launch the Desktop GUI application
    python run.py --source 1        # Launch GUI with specific camera index
    python run.py --benchmark       # Run the quantitative benchmark suite
    python run.py --test            # Run the automated unit test suite
"""

import argparse
import sys
from pathlib import Path

# Auto-switch to local .venv if dependencies (e.g. cv2) are not in the active Python interpreter
ROOT_DIR = Path(__file__).resolve().parent
_VENV_PY = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
if not _VENV_PY.exists():
    _VENV_PY = ROOT_DIR.parent / ".venv" / "Scripts" / "python.exe"

if _VENV_PY.exists() and Path(sys.executable).resolve() != _VENV_PY.resolve():
    try:
        import cv2  # noqa: F401
    except ImportError:
        import subprocess
        sys.exit(subprocess.call([str(_VENV_PY), *sys.argv]))

# Add package directory to sys.path so src and benchmarks can be imported from root
PACKAGE_DIR = ROOT_DIR / "color_object_detection"
if not PACKAGE_DIR.exists():
    PACKAGE_DIR = ROOT_DIR
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Color-Based Object Detection and Real-Time Tracking System"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Launch the Desktop GUI application (default mode).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video source: camera index (e.g. 0, 1) or path to video file.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the quantitative benchmark suite instead of the GUI.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the automated pytest test suite instead of the GUI.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames for benchmark evaluation (when --benchmark is set).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for benchmark plots (when --benchmark is set).",
    )

    args, unknown = parser.parse_known_args()

    if args.test:
        import pytest
        test_dir = PACKAGE_DIR / "tests"
        sys.exit(pytest.main([str(test_dir), "-v"] + unknown))

    if args.benchmark:
        try:
            from color_object_detection.benchmarks.evaluate import run_full_benchmark
            from color_object_detection.src import config
        except ImportError:
            from benchmarks.evaluate import run_full_benchmark  # type: ignore
            from src import config  # type: ignore

        out_dir = args.out or config.BENCHMARK_RESULTS_DIR
        frames = args.frames or config.BENCHMARK_FRAMES_PER_RESOLUTION
        run_full_benchmark(out_dir=out_dir, frames_per_res=frames)
        return

    # Default action: launch GUI application
    try:
        from color_object_detection.src.gui_app import ColorDetectionApp
    except ImportError:
        from src.gui_app import ColorDetectionApp  # type: ignore

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    app = ColorDetectionApp(source=source)
    app.run()


if __name__ == "__main__":
    main()
