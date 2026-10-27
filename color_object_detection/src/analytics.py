"""
analytics.py
============
Advanced Computer Vision Analytics Suite:
1. Virtual Tripwire (Directional In/Out Line-Crossing Counter)
2. ROI Zone Analytics (Occupancy & Dwell-Time Tracking)
3. 2D Motion Trajectory Heatmap Generator (Jet Colormap)
4. Telemetry Kinematics Logger (CSV & JSON Exporter)
5. Live Video Session Recorder (Safe cv2.VideoWriter Wrapper)
"""

import csv
import datetime
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from . import config
from .utils import draw_label, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Virtual Tripwire (Directional In/Out Counter)
# ---------------------------------------------------------------------------

class Tripwire(object):
    """Directional line-crossing counter for traffic, people, and object analytics."""

    def __init__(
        self,
        pt1: Tuple[int, int] = (100, 240),
        pt2: Tuple[int, int] = (540, 240),
        name: str = "Tripwire-1",
        line_color: Tuple[int, int, int] = (0, 220, 255),
    ):
        self.pt1 = pt1
        self.pt2 = pt2
        self.name = name
        self.line_color = line_color

        self.count_in = 0
        self.count_out = 0
        self._prev_sides: Dict[int, float] = {}  # track_id -> side value
        self._flash_frames = 0
        self.is_enabled = True

    def reset_counts(self):
        self.count_in = 0
        self.count_out = 0
        self._prev_sides.clear()
        self._flash_frames = 0

    def _side(self, pt: Tuple[int, int]) -> float:
        """Compute the signed cross product relative to the line (pt1 -> pt2)."""
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        px, py = pt
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def _segments_intersect(self, p1, p2, p3, p4) -> bool:
        """Check if segment (p1-p2) intersects segment (p3-p4)."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def update(self, tracks) -> Tuple[int, int]:
        """Update crossing events given active tracks."""
        if not self.is_enabled:
            return self.count_in, self.count_out

        if self._flash_frames > 0:
            self._flash_frames -= 1

        for trk in tracks:
            tid = getattr(trk, "track_id", id(trk))
            curr_pt = getattr(trk, "centroid", None)
            if curr_pt is None:
                continue

            curr_side = self._side(curr_pt)
            prev_side = self._prev_sides.get(tid)

            if prev_side is not None:
                # Check if signs differ and distance crossed the line
                if (prev_side * curr_side) < 0:
                    prev_pt = trk.trail[-2] if len(trk.trail) >= 2 else None
                    if prev_pt is not None and self._segments_intersect(prev_pt, curr_pt, self.pt1, self.pt2):
                        if prev_side < 0 and curr_side > 0:
                            self.count_in += 1
                            self._flash_frames = 8
                            logger.info("Tripwire '%s': Object #%d crossed IN (Total IN: %d)", self.name, tid, self.count_in)
                        elif prev_side > 0 and curr_side < 0:
                            self.count_out += 1
                            self._flash_frames = 8
                            logger.info("Tripwire '%s': Object #%d crossed OUT (Total OUT: %d)", self.name, tid, self.count_out)

            self._prev_sides[tid] = curr_side

        return self.count_in, self.count_out

    def render(self, frame: np.ndarray):
        """Draw the tripwire line and In/Out tally badge."""
        if not self.is_enabled:
            return

        col = (0, 255, 0) if self._flash_frames > 0 else self.line_color
        thickness = 3 if self._flash_frames > 0 else 2

        # Draw tripwire guideline
        cv2.line(frame, self.pt1, self.pt2, col, thickness, lineType=cv2.LINE_AA)
        cv2.circle(frame, self.pt1, 5, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, self.pt2, 5, col, -1, lineType=cv2.LINE_AA)

        # Midpoint indicator
        mid_x = (self.pt1[0] + self.pt2[0]) // 2
        mid_y = (self.pt1[1] + self.pt2[1]) // 2

        badge_text = f"TRIPWIRE | IN: {self.count_in}  OUT: {self.count_out}  NET: {self.count_in - self.count_out}"
        draw_label(
            frame,
            badge_text,
            mid_x - 120,
            mid_y - 12,
            color_bgr=(0, 255, 255),
            bg_color_bgr=(20, 25, 40),
            font_scale=0.5,
            thickness=1,
        )


# ---------------------------------------------------------------------------
# 2. ROI Zone Analytics (Occupancy & Dwell-Time Tracking)
# ---------------------------------------------------------------------------

class ROIZone(object):
    """Monitors occupancy count and dwell times inside a designated region."""

    def __init__(
        self,
        zone_rect: Tuple[int, int, int, int] = (160, 100, 320, 280),
        name: str = "Monitored Zone",
        color_bgr: Tuple[int, int, int] = (255, 140, 0),
    ):
        self.zone_rect = zone_rect  # (x, y, w, h)
        self.name = name
        self.color_bgr = color_bgr
        self.occupants = 0
        self.dwell_times: Dict[int, float] = {}  # track_id -> total seconds in zone
        self._entry_times: Dict[int, float] = {}
        self.is_enabled = True

    def reset(self):
        self.occupants = 0
        self.dwell_times.clear()
        self._entry_times.clear()

    def contains(self, pt: Tuple[int, int]) -> bool:
        x, y, w, h = self.zone_rect
        px, py = pt
        return x <= px <= (x + w) and y <= py <= (y + h)

    def update(self, tracks, dt: float = 0.033) -> int:
        """Update occupancy and dwell times for all tracked objects."""
        if not self.is_enabled:
            return 0

        current_occupants = 0
        now = time.time()

        for trk in tracks:
            tid = getattr(trk, "track_id", id(trk))
            pt = getattr(trk, "centroid", None)
            if pt is None:
                continue

            if self.contains(pt):
                current_occupants += 1
                if tid not in self._entry_times:
                    self._entry_times[tid] = now
                self.dwell_times[tid] = self.dwell_times.get(tid, 0.0) + dt
            else:
                if tid in self._entry_times:
                    del self._entry_times[tid]

        self.occupants = current_occupants
        return self.occupants

    def render(self, frame: np.ndarray):
        """Draw translucent highlighted boundary with live occupancy badge."""
        if not self.is_enabled:
            return

        x, y, w, h = self.zone_rect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.color_bgr, -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.color_bgr, 2)

        avg_dwell = 0.0
        if self.dwell_times:
            avg_dwell = sum(self.dwell_times.values()) / max(1, len(self.dwell_times))

        zone_badge = f"{self.name} | Occupants: {self.occupants} | Avg Dwell: {avg_dwell:.1f}s"
        draw_label(frame, zone_badge, x + 8, y + 20, color_bgr=(255, 255, 255), bg_color_bgr=(30, 20, 20))


# ---------------------------------------------------------------------------
# 3. 2D Motion Trajectory Heatmap Generator
# ---------------------------------------------------------------------------

class TrajectoryHeatmap(object):
    """Accumulates centroid locations over time to render a JET thermal heatmap."""

    def __init__(self, width: int = 640, height: int = 480, decay: float = 0.999):
        self.width = width
        self.height = height
        self.decay = decay
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.is_enabled = False

    def reset(self):
        self.grid.fill(0.0)

    def add_point(self, cx: int, cy: int, radius: int = 16, intensity: float = 1.0):
        """Add a smoothed Gaussian density splat centered at (cx, cy)."""
        if not (0 <= cx < self.width and 0 <= cy < self.height):
            return

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(self.width, cx + radius + 1)
        y2 = min(self.height, cy + radius + 1)

        for y in range(y1, y2):
            for x in range(x1, x2):
                dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                if dist_sq <= radius ** 2:
                    falloff = math.exp(-dist_sq / (2.0 * (radius / 2.0) ** 2))
                    self.grid[y, x] += float(intensity * falloff)

    def update_tracks(self, tracks):
        """Update heatmap with all active tracked centroids."""
        if not self.is_enabled:
            return
        for trk in tracks:
            pt = getattr(trk, "centroid", None)
            if pt is not None:
                self.add_point(pt[0], pt[1])

    def render_overlay(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend the thermal heatmap directly onto the output frame."""
        if not self.is_enabled:
            return frame

        max_val = np.max(self.grid)
        if max_val <= 0.0:
            return frame

        # Normalize to 0-255
        norm_grid = np.clip((self.grid / max_val) * 255.0, 0, 255).astype(np.uint8)
        colored_map = cv2.applyColorMap(norm_grid, cv2.COLORMAP_JET)

        # Apply mask so low-activity areas remain clear
        mask = norm_grid > 20
        blended = frame.copy()
        blended[mask] = cv2.addWeighted(frame, 1.0 - alpha, colored_map, alpha, 0)[mask]
        return blended

    def export_image(self, filepath: Path) -> Path:
        """Export current accumulated heatmap to an image file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        max_val = np.max(self.grid)
        if max_val > 0.0:
            norm_grid = np.clip((self.grid / max_val) * 255.0, 0, 255).astype(np.uint8)
            heatmap_img = cv2.applyColorMap(norm_grid, cv2.COLORMAP_JET)
        else:
            heatmap_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.imwrite(str(filepath), heatmap_img)
        return filepath


# ---------------------------------------------------------------------------
# 4. Telemetry Logger (CSV & JSON Kinematics Exporter)
# ---------------------------------------------------------------------------

class TelemetryLogger(object):
    """Logs kinematics telemetry (x, y, speed, distance, bbox) for physics/data analysis."""

    def __init__(self):
        self.rows: List[dict] = []
        self.is_active = False
        self._frame_counter = 0

    def start(self):
        self.rows.clear()
        self.is_active = True
        self._frame_counter = 0

    def stop(self):
        self.is_active = False

    def log_frame(self, tracks, timestamp: Optional[float] = None):
        if not self.is_active:
            return

        now = time.time() if timestamp is None else timestamp
        self._frame_counter += 1

        for trk in tracks:
            cx, cy = getattr(trk, "centroid", (0, 0))
            bbox = getattr(trk, "bbox", (0, 0, 0, 0))
            row = {
                "frame_id": self._frame_counter,
                "timestamp_sec": round(now, 4),
                "track_id": getattr(trk, "track_id", 1),
                "color_name": getattr(trk, "color_name", "Unknown"),
                "centroid_x": cx,
                "centroid_y": cy,
                "area_px2": getattr(trk, "area", 0.0),
                "speed_px_s": getattr(trk, "speed", 0.0),
                "distance_cm": getattr(trk, "distance_cm", 0.0),
                "metric_speed_mps": getattr(trk, "metric_speed_mps", 0.0),
                "is_occluded": getattr(trk, "is_occluded", False),
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2],
                "bbox_h": bbox[3],
            }
            self.rows.append(row)

    def export_csv(self, filepath: Path) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            return filepath

        keys = list(self.rows[0].keys())
        with open(str(filepath), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
        return filepath

    def export_json(self, filepath: Path) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(str(filepath), "w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=2)
        return filepath

    @property
    def total_rows(self) -> int:
        return len(self.rows)


# ---------------------------------------------------------------------------
# 5. Live Video Session Recorder
# ---------------------------------------------------------------------------

class VideoRecorder(object):
    """Robust OpenCV VideoWriter wrapper with automatic codec negotiation."""

    def __init__(self):
        self.writer: Optional[cv2.VideoWriter] = None
        self.filepath: Optional[Path] = None
        self.is_recording = False
        self.frame_count = 0
        self.start_time: Optional[float] = None

    def start(self, filepath: Path, width: int = 640, height: int = 480, fps: float = 30.0) -> bool:
        self.stop()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        candidate_fourccs = ["mp4v", "avc1", "XVID", "MJPG"]
        writer = None

        for code in candidate_fourccs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*code)
                w = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
                if w.isOpened():
                    writer = w
                    logger.info("VideoRecorder initialized with codec '%s' -> %s", code, filepath.name)
                    break
            except Exception:
                continue

        if writer is None:
            logger.error("Failed to initialize VideoWriter with candidate codecs.")
            return False

        self.writer = writer
        self.filepath = filepath
        self.is_recording = True
        self.frame_count = 0
        self.start_time = time.time()
        return True

    def write_frame(self, frame: np.ndarray):
        if self.is_recording and self.writer is not None:
            self.writer.write(frame)
            self.frame_count += 1

    def stop(self) -> Optional[Path]:
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None

        self.is_recording = False
        saved_path = self.filepath
        self.filepath = None
        self.start_time = None
        return saved_path

    @property
    def duration_sec(self) -> float:
        if self.start_time is None:
            return 0.0
        return round(time.time() - self.start_time, 1)
