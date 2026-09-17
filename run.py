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

# Add package directory to sys.path so src and benchmarks can be imported from root
PACKAGE_DIR = Path(__file__).resolve().parent / "color_object_detection"
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
        from benchmarks.evaluate import run_full_benchmark
        from src import config
        out_dir = args.out or config.BENCHMARK_RESULTS_DIR
        frames = args.frames or config.BENCHMARK_FRAMES_PER_RESOLUTION
        run_full_benchmark(out_dir=out_dir, frames_per_res=frames)
        return

    # Default action: launch GUI application
    from src.gui_app import ColorDetectionApp
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
