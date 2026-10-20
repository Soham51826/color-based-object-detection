"""
run.py
======
Unified entry point for the Color-Based Object Detection System.

Run this script directly:
    python run.py                   # Launch the Desktop GUI application
    python run.py --source 1        # Launch GUI with specific camera index
    python run.py --source assets/color_test_benchmark.mp4 # Launch GUI with benchmark video
    python run.py --cli             # Run live tracking in OpenCV window
    python run.py --demo            # Run 10-stage pipeline walkthrough
    python run.py --benchmark       # Run the quantitative benchmark suite
    python run.py --test            # Run the automated unit test suite
"""

import argparse
import sys
from pathlib import Path

# Auto-switch to local .venv if dependencies (e.g. cv2) are not in the active Python interpreter
ROOT_DIR = Path(__file__).resolve().parent
_CANDIDATE_VENVS = [
    ROOT_DIR / ".venv" / "Scripts" / "python.exe",
    ROOT_DIR / ".venv" / "bin" / "python",
    ROOT_DIR.parent / ".venv" / "Scripts" / "python.exe",
    ROOT_DIR.parent / ".venv" / "bin" / "python",
]
_VENV_PY = next((p for p in _CANDIDATE_VENVS if p.exists()), None)

if _VENV_PY and Path(sys.executable).resolve() != _VENV_PY.resolve():
    try:
        import cv2  # noqa: F401
    except ImportError:
        import subprocess
        sys.exit(subprocess.call([str(_VENV_PY), *sys.argv]))

# Add package directory and root directory to sys.path so modules resolve cleanly from anywhere
PACKAGE_DIR = ROOT_DIR / "color_object_detection"
if not PACKAGE_DIR.exists():
    PACKAGE_DIR = ROOT_DIR
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _resolve_and_prepare_source(source):
    """Normalize video source and auto-generate the benchmark video if missing."""
    if source is None:
        return None
    try:
        return int(source)
    except ValueError:
        src_path = Path(source)
        if not src_path.exists() and src_path.name == "color_test_benchmark.mp4":
            print(f"\n[run.py] Benchmark video not found at '{source}'. Synthesizing automatically...")
            try:
                from color_object_detection.benchmarks.generate_video import generate_benchmark_video
            except ImportError:
                from benchmarks.generate_video import generate_benchmark_video  # type: ignore
            generate_benchmark_video(output_path=src_path, show_progress=True)
            print("[run.py] Benchmark video generated successfully!\n")
        return str(src_path)


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
        "--cli",
        action="store_true",
        help="Run live tracking in a lightweight OpenCV window without the GUI.",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        default=None,
        metavar="COLOR",
        help="One or more color names to track (e.g. --colors Red Blue). Default: all colors.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run an interactive educational walkthrough of all 10 computer vision pipeline stages.",
    )
    parser.add_argument(
        "--generate-video",
        action="store_true",
        help="Generate synthetic multi-target benchmark video (assets/color_test_benchmark.mp4).",
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

    if args.generate_video:
        try:
            from color_object_detection.benchmarks.generate_video import generate_benchmark_video
        except ImportError:
            from benchmarks.generate_video import generate_benchmark_video  # type: ignore
        generate_benchmark_video(show_progress=True)
        return

    if args.demo:
        try:
            from color_object_detection.src.pipeline_demo import run_pipeline_demo
        except ImportError:
            from src.pipeline_demo import run_pipeline_demo  # type: ignore
        run_pipeline_demo(show_window=False)
        return

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

    if args.cli:
        try:
            import cv2
            from color_object_detection.src.tracker import ColorTracker, open_capture
            from color_object_detection.src import config
        except ImportError:
            import cv2
            from src.tracker import ColorTracker, open_capture  # type: ignore
            from src import config  # type: ignore

        source = _resolve_and_prepare_source(args.source)

        color_names = args.colors
        if color_names:
            for name in color_names:
                if name not in config.COLOR_PRESETS:
                    print(f"Unknown colour '{name}'. Available: {list(config.COLOR_PRESETS.keys())}")
                    sys.exit(1)

        tracker = ColorTracker(color_names=color_names)
        cap = open_capture(source, fallback_to_synthetic=True)
        window_name = "Color Detection — press 'q' or Esc to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("\n[Color Tracker] Running in OpenCV window. Press 'q' or Esc to exit.\n")
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                result = tracker.process_frame(frame)
                cv2.imshow(window_name, result.annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[Color Tracker] Stopped.")
        return

    # Default action: launch GUI application
    try:
        from color_object_detection.src.gui_app import ColorDetectionApp
    except ImportError:
        from src.gui_app import ColorDetectionApp  # type: ignore

    source = _resolve_and_prepare_source(args.source)

    app = ColorDetectionApp(source=source)
    app.run()


if __name__ == "__main__":
    main()
