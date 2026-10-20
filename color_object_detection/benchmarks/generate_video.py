"""
generate_video.py
=================
Automated benchmark video generator for the Color-Based Object Detection System.

Generates an annotated MP4 benchmark video (`assets/color_test_benchmark.mp4`)
using OpenCV VideoWriter (mp4v codec, 640x480 resolution, 60 FPS, 10 seconds = 600 frames).

Features:
- Moving spherical and rectangular targets across the supported color palette:
  Red, Orange, Yellow, Green, Cyan, Blue, Purple, Pink.
- Realistic 3D specular highlights on spherical targets and edge gloss on rectangles.
- Physics-based wall bounces with boundary margin containment.
- Dynamic lighting shifts: ambient illuminance oscillations, moving room shadow sweeps,
  and specular glare patches.
- Mild Gaussian sensor noise simulating real-world webcam conditions.

Usage:
    python -m benchmarks.generate_video
    python benchmarks/generate_video.py
    python run.py --generate-video
"""

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is on sys.path
_FILE_PATH = Path(__file__).resolve()
_PROJECT_ROOT = _FILE_PATH.parents[2] if _FILE_PATH.parents[1].name == "color_object_detection" else _FILE_PATH.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_PKG_DIR = _PROJECT_ROOT / "color_object_detection"
if _PKG_DIR.exists() and str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))


class TargetObject:
    """Physics-enabled target object for benchmark video synthesis."""

    def __init__(
        self,
        name: str,
        bgr: Tuple[int, int, int],
        shape: str,
        pos: Tuple[float, float],
        vel: Tuple[float, float],
        size: int,
    ):
        self.name = name
        self.bgr = bgr
        self.shape = shape  # "sphere" or "rect"
        self.pos = list(pos)
        self.vel = list(vel)
        self.size = size

    def update_physics(self, width: int, height: int, margin: int = 15):
        """Update object position with boundary wall collisions."""
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        r = self.size
        # Left and Right walls
        if self.pos[0] - r < margin:
            self.pos[0] = float(margin + r)
            self.vel[0] = abs(self.vel[0])
        elif self.pos[0] + r > width - margin:
            self.pos[0] = float(width - margin - r)
            self.vel[0] = -abs(self.vel[0])

        # Top and Bottom walls
        if self.pos[1] - r < margin:
            self.pos[1] = float(margin + r)
            self.vel[1] = abs(self.vel[1])
        elif self.pos[1] + r > height - margin:
            self.pos[1] = float(height - margin - r)
            self.vel[1] = -abs(self.vel[1])


def generate_benchmark_video(
    output_path: Optional[Path] = None,
    width: int = 640,
    height: int = 480,
    fps: int = 60,
    duration_sec: float = 10.0,
    show_progress: bool = True,
) -> Path:
    """Generate the benchmark MP4 video file.

    Parameters
    ----------
    output_path : Path, optional
        Target .mp4 file path. Defaults to `assets/color_test_benchmark.mp4`.
    width, height : int
        Frame dimensions (default: 640x480).
    fps : int
        Frame rate (default: 60).
    duration_sec : float
        Video length in seconds (default: 10.0 -> 600 frames).
    show_progress : bool
        Whether to print generation progress to stdout.

    Returns
    -------
    Path : The generated MP4 file path.
    """
    if output_path is None:
        output_path = _PROJECT_ROOT / "assets" / "color_test_benchmark.mp4"

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to initialize cv2.VideoWriter with fourcc='mp4v' at {output_path}."
        )

    # 8 targets spanning the supported chromatic palette
    # Coordinates and velocities chosen for lively, non-degenerate motion
    targets: List[TargetObject] = [
        TargetObject("Red",    (48,  68,  167), "sphere", (100.0, 120.0), ( 2.8,  2.1), 32),
        TargetObject("Orange", (47,  116, 177), "rect",   (220.0, 160.0), (-2.4,  2.6), 30),
        TargetObject("Yellow", (54,  169, 177), "sphere", (340.0, 240.0), ( 2.1, -2.7), 32),
        TargetObject("Green",  (52,  157, 56),  "rect",   (460.0, 140.0), (-2.7, -1.9), 32),
        TargetObject("Cyan",   (162, 155, 50),  "sphere", (120.0, 340.0), (-2.0, -2.5), 30),
        TargetObject("Blue",   (157, 67,  45),  "rect",   (260.0, 360.0), ( 2.5, -2.2), 32),
        TargetObject("Purple", (157, 54,  116), "sphere", (420.0, 360.0), (-2.2,  2.4), 30),
        TargetObject("Pink",   (154, 61,  177), "sphere", (520.0, 240.0), ( 2.6,  1.8), 30),
    ]

    # Pre-compute coordinate grids for smooth shadow and glare gradients
    y_coords, x_coords = np.ogrid[:height, :width]

    if show_progress:
        print(f"[Video Generator] Creating {output_path.name} ({width}x{height} @ {fps}fps, {duration_sec}s)...")

    for f_idx in range(total_frames):
        t = f_idx / float(fps)  # current time in seconds

        # 1. Base dark background canvas with subtle contrast texture
        frame = np.full((height, width, 3), 36, dtype=np.uint8)

        # Draw grid lines for subtle visual depth
        for gx in range(0, width, 80):
            cv2.line(frame, (gx, 0), (gx, height), (42, 42, 48), 1)
        for gy in range(0, height, 80):
            cv2.line(frame, (0, gy), (width, gy), (42, 42, 48), 1)

        # 2. Render all targets with realistic specular highlights
        for obj in targets:
            cx, cy = int(obj.pos[0]), int(obj.pos[1])
            r = obj.size

            if obj.shape == "sphere":
                # Base colored sphere
                cv2.circle(frame, (cx, cy), r, obj.bgr, -1)
                # Outer shading ring for 3D sphere curvature
                cv2.circle(frame, (cx, cy), r, (int(obj.bgr[0] * 0.75), int(obj.bgr[1] * 0.75), int(obj.bgr[2] * 0.75)), 2)
                # Realistic specular highlight (offset toward top-left light source)
                hx = cx - int(0.35 * r)
                hy = cy - int(0.35 * r)
                hr = max(3, int(0.24 * r))
                cv2.circle(frame, (hx, hy), hr, (240, 240, 252), -1)
            else:  # "rect"
                # Base colored rectangle
                x1, y1 = cx - r, cy - r
                x2, y2 = cx + r, cy + r
                cv2.rectangle(frame, (x1, y1), (x2, y2), obj.bgr, -1)
                # Bevel border
                cv2.rectangle(frame, (x1, y1), (x2, y2), (int(obj.bgr[0] * 0.75), int(obj.bgr[1] * 0.75), int(obj.bgr[2] * 0.75)), 2)
                # Glossy specular line along top-left inner border
                cv2.line(frame, (x1 + 4, y1 + 4), (x2 - 4, y1 + 4), (235, 235, 250), 2)
                cv2.line(frame, (x1 + 4, y1 + 4), (x1 + 4, y2 - 4), (235, 235, 250), 2)

            # Advance physics to next frame
            obj.update_physics(width, height)

        # 3. Dynamic ambient lighting shifts (simulates fluctuating room lighting)
        ambient_factor = 1.0 + 0.12 * math.sin(2.0 * math.pi * t / 3.0) + 0.05 * math.cos(2.0 * math.pi * t / 1.5)
        frame_float = frame.astype(np.float32) * ambient_factor

        # 4. Moving dynamic shadow sweep (simulates a moving soft shadow across the room)
        shadow_cx = width * (0.5 + 0.45 * math.sin(2.0 * math.pi * t / 5.0))
        shadow_cy = height * (0.5 + 0.35 * math.cos(2.0 * math.pi * t / 4.0))
        dist_sq_shadow = (x_coords - shadow_cx) ** 2 + (y_coords - shadow_cy) ** 2
        # Soft shadow dip (15-20% attenuation in shadow core)
        shadow_mask = 1.0 - 0.20 * np.exp(-dist_sq_shadow / (2.0 * (180.0 ** 2)))
        frame_float = frame_float * shadow_mask[:, :, np.newaxis]

        # 5. Moving specular glare patch (simulates direct overhead light reflection)
        glare_cx = width * (0.5 + 0.4 * math.cos(2.0 * math.pi * t / 6.0))
        glare_cy = height * (0.3 + 0.25 * math.sin(2.0 * math.pi * t / 4.5))
        dist_sq_glare = (x_coords - glare_cx) ** 2 + (y_coords - glare_cy) ** 2
        glare_intensity = 35.0 * np.exp(-dist_sq_glare / (2.0 * (90.0 ** 2)))
        frame_float = frame_float + glare_intensity[:, :, np.newaxis]

        # 6. Mild Gaussian sensor noise (realistic camera photon shot noise)
        np.random.seed(f_idx)
        sensor_noise = np.random.normal(0, 3.2, frame.shape)
        final_frame = np.clip(frame_float + sensor_noise, 0, 255).astype(np.uint8)

        # Subtle benchmark overlay footer
        footer_text = f"COLOR BENCHMARK TEST FEED | 60 FPS | T={t:04.1f}s | F={f_idx+1}/{total_frames}"
        cv2.putText(
            final_frame,
            footer_text,
            (16, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(final_frame)

        if show_progress and (f_idx + 1) % 120 == 0:
            print(f"  Rendered {f_idx + 1}/{total_frames} frames ({(f_idx + 1)/total_frames*100:.0f}%)...")

    writer.release()

    # Mirror to color_object_detection/assets if distinct
    mirror_path = _PKG_DIR / "assets" / output_path.name
    if mirror_path.resolve() != output_path.resolve():
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(output_path, mirror_path)

    if show_progress:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[Video Generator] Success! Saved {output_path} ({file_size_mb:.2f} MB)")

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate MP4 benchmark test video.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Target output video path (default: assets/color_test_benchmark.mp4).",
    )
    parser.add_argument("--fps", type=int, default=60, help="Frames per second (default: 60).")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds (default: 10.0).")
    args = parser.parse_args()

    generate_benchmark_video(
        output_path=args.out,
        fps=args.fps,
        duration_sec=args.duration,
        show_progress=True,
    )


if __name__ == "__main__":
    main()
