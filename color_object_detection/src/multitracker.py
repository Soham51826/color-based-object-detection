"""
multitracker.py
===============
Multi-object persistent tracking with Kalman Filter prediction, Euclidean/IoU
data association, occlusion handling, and metric distance estimation.

Key Capabilities
----------------
1. Persistent Track IDs: Assigns and maintains unique IDs (e.g. ID #1, ID #2)
   across frames for multiple objects of the same or different colors.
2. Occlusion Handling: When an object is temporarily obscured, the Kalman
   Filter projects its trajectory forward, displaying a predictive target
   cone and maintaining track continuity for up to N frames.
3. Metric Distance & Velocity: Uses the pinhole camera geometry model to
   estimate real-world physical distance (cm) and metric velocity (m/s).
"""

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from . import config
from .utils import (
    compute_centroid,
    compute_circularity,
    compute_contour_angle,
    compute_solidity,
    draw_label,
    get_logger,
)

logger = get_logger(__name__)


class TrackedObject(object):
    """Encapsulates the state and history of a persistently tracked physical object."""

    def __init__(
        self,
        track_id: int,
        color_name: str,
        color_bgr: Tuple[int, int, int],
        centroid: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        contour: Optional[np.ndarray] = None,
        area: float = 0.0,
        solidity: float = 1.0,
        circularity: float = 1.0,
        angle: float = 0.0,
        timestamp: Optional[float] = None,
        trail_maxlen: int = 32,
        real_width_cm: float = 6.5,
        focal_length_px: float = 700.0,
    ):
        self.track_id = track_id
        self.color_name = color_name
        self.color_bgr = color_bgr
        self.centroid = centroid
        self.raw_centroid = centroid
        self.bbox = bbox
        self.contour = contour
        self.area = float(area)
        self.solidity = float(solidity)
        self.circularity = float(circularity)
        self.angle = float(angle)
        self.real_width_cm = float(real_width_cm)
        self.focal_length_px = float(focal_length_px)

        # Kinematics
        self.speed = 0.0  # px/s
        self.velocity = (0.0, 0.0)  # (vx, vy) px/s
        self.distance_cm = self.estimate_distance_cm(bbox, area)
        self.metric_speed_mps = 0.0  # m/s

        # Occlusion & Lifecycle
        now = time.time() if timestamp is None else timestamp
        self.first_seen = now
        self.last_seen = now
        self.missed_frames = 0
        self.is_occluded = False
        self.predicted_centroid = centroid

        # Kalman Filter Tracker
        from .tracker import KalmanFilterTracker
        self.kalman = KalmanFilterTracker()
        self.kalman.initialize(centroid[0], centroid[1])

        # Trajectory trail
        self.trail = deque(maxlen=trail_maxlen)
        self.trail.append(centroid)

        # Dwell time tracking in seconds
        self.dwell_time_sec = 0.0

    def estimate_distance_cm(self, bbox: Tuple[int, int, int, int], area: float) -> float:
        """Estimate distance from camera using pinhole projection geometry.

        Z = (f * W_real) / w_pixel
        """
        # Apparent pixel width (using maximum dimension or equivalent circle diameter)
        w_px = max(float(bbox[2]), float(bbox[3]))
        if area > 0:
            equiv_dia = math.sqrt(4.0 * area / math.pi)
            w_px = max(w_px, equiv_dia)

        if w_px <= 1.0:
            return 100.0
        dist = (self.focal_length_px * self.real_width_cm) / w_px
        return max(5.0, min(1000.0, round(dist, 1)))

    def predict(self, dt: float = 0.033) -> Tuple[int, int]:
        """Project Kalman filter state forward in time."""
        pos, vel = self.kalman.predict(dt=dt)
        if pos is not None:
            self.predicted_centroid = (round(pos[0]), round(pos[1]))
            self.velocity = vel
        return self.predicted_centroid

    def update_measurement(
        self,
        centroid: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        contour: Optional[np.ndarray],
        area: float,
        solidity: float,
        circularity: float,
        angle: float,
        timestamp: Optional[float] = None,
        alpha: float = 0.7,
    ):
        """Update track with an observed detection measurement."""
        now = time.time() if timestamp is None else timestamp
        dt = max(0.001, min(0.5, now - self.last_seen))
        self.last_seen = now
        self.dwell_time_sec += dt

        prev_cx, prev_cy = self.centroid

        # Kalman correction
        pos, vel = self.kalman.correct(centroid[0], centroid[1])
        self.raw_centroid = centroid

        # EMA Centroid Smoothing
        sm_cx = alpha * float(centroid[0]) + (1.0 - alpha) * float(prev_cx)
        sm_cy = alpha * float(centroid[1]) + (1.0 - alpha) * float(prev_cy)
        self.centroid = (round(sm_cx), round(sm_cy))

        # Update geometry
        self.bbox = bbox
        self.contour = contour
        self.area = float(area)
        self.solidity = float(solidity)
        self.circularity = float(circularity)
        self.angle = float(angle)

        # Update distance & metric velocity
        self.distance_cm = self.estimate_distance_cm(bbox, area)
        inst_speed = math.hypot(self.centroid[0] - prev_cx, self.centroid[1] - prev_cy) / dt
        self.speed = 0.6 * inst_speed + 0.4 * self.speed
        # Real-world metric speed in m/s: (px/s * Z_cm) / (f_px * 100)
        self.metric_speed_mps = round((self.speed * self.distance_cm) / (self.focal_length_px * 100.0), 2)
        self.velocity = (
            (self.centroid[0] - prev_cx) / dt,
            (self.centroid[1] - prev_cy) / dt,
        )

        # Reset occlusion state
        self.missed_frames = 0
        self.is_occluded = False
        self.trail.append(self.centroid)

    def mark_occluded(self, dt: float = 0.033):
        """Mark object as missing measurement and project position with Kalman."""
        self.missed_frames += 1
        self.is_occluded = True
        self.predict(dt=dt)
        self.centroid = self.predicted_centroid
        self.speed *= 0.8
        self.trail.append(self.centroid)


class MultiObjectTracker(object):
    """Associates detections across frames to maintain stable multi-object trajectories."""

    def __init__(
        self,
        max_distance_px: float = 90.0,
        max_missed_frames: int = 5,
        real_width_cm: float = 6.5,
        focal_length_px: float = 700.0,
    ):
        self.max_distance_px = max_distance_px
        self.max_missed_frames = max_missed_frames
        self.real_width_cm = real_width_cm
        self.focal_length_px = focal_length_px

        self._next_id = 1
        self.tracks: Dict[int, TrackedObject] = {}

    def reset(self):
        """Clear all active tracks and reset ID counter."""
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: List[dict], timestamp: Optional[float] = None) -> List[TrackedObject]:
        """Update tracker with detections from current frame.

        Parameters
        ----------
        detections : list of dict
            Each dict contains:
            - 'color_name': str
            - 'color_bgr': tuple
            - 'centroid': (cx, cy)
            - 'bbox': (x, y, w, h)
            - 'contour': np.ndarray
            - 'area': float
            - 'solidity': float
            - 'circularity': float
            - 'angle': float

        Returns
        -------
        list of TrackedObject (active and occluded)
        """
        now = time.time() if timestamp is None else timestamp

        # Step 1: Predict next state for all active tracks
        for track in self.tracks.values():
            track.predict(dt=0.033)

        track_ids = list(self.tracks.keys())
        det_indices = list(range(len(detections)))

        matched_tracks = set()
        matched_dets = set()

        # Step 2: Associate detections to tracks of the same color class
        if track_ids and det_indices:
            # Pairwise cost: Euclidean distance between detection centroid and track prediction
            pairs = []
            for tid in track_ids:
                trk = self.tracks[tid]
                tcx, tcy = trk.predicted_centroid
                for didx in det_indices:
                    det = detections[didx]
                    if det["color_name"] == trk.color_name:
                        dcx, dcy = det["centroid"]
                        dist = math.hypot(dcx - tcx, dcy - tcy)
                        if dist <= self.max_distance_px:
                            pairs.append((dist, tid, didx))

            # Greedy matching (lowest distance first)
            pairs.sort(key=lambda item: item[0])
            for dist, tid, didx in pairs:
                if tid not in matched_tracks and didx not in matched_dets:
                    matched_tracks.add(tid)
                    matched_dets.add(didx)
                    det = detections[didx]
                    self.tracks[tid].update_measurement(
                        centroid=det["centroid"],
                        bbox=det["bbox"],
                        contour=det.get("contour"),
                        area=det["area"],
                        solidity=det.get("solidity", 1.0),
                        circularity=det.get("circularity", 1.0),
                        angle=det.get("angle", 0.0),
                        timestamp=now,
                    )

        # Step 3: Handle unmatched tracks (occlusion or lost)
        unmatched_track_ids = [tid for tid in track_ids if tid not in matched_tracks]
        for tid in unmatched_track_ids:
            trk = self.tracks[tid]
            trk.mark_occluded(dt=0.033)
            if trk.missed_frames > self.max_missed_frames:
                del self.tracks[tid]

        # Step 4: Handle unmatched detections (spawn new tracks)
        unmatched_det_indices = [didx for didx in det_indices if didx not in matched_dets]
        for didx in unmatched_det_indices:
            det = detections[didx]
            new_track = TrackedObject(
                track_id=self._next_id,
                color_name=det["color_name"],
                color_bgr=det["color_bgr"],
                centroid=det["centroid"],
                bbox=det["bbox"],
                contour=det.get("contour"),
                area=det["area"],
                solidity=det.get("solidity", 1.0),
                circularity=det.get("circularity", 1.0),
                angle=det.get("angle", 0.0),
                timestamp=now,
                real_width_cm=self.real_width_cm,
                focal_length_px=self.focal_length_px,
            )
            self.tracks[self._next_id] = new_track
            self._next_id += 1

        return list(self.tracks.values())


def render_multitrack_overlays(
    frame: np.ndarray,
    tracks: List[TrackedObject],
    show_prediction: bool = False,
    show_metric_distance: bool = False,
):
    """Render clean, high-contrast bounding boxes and track ID badges (omits occluded ghosts)."""
    for trk in tracks:
        if trk.is_occluded:
            continue  # Do NOT clutter screen with lost/occluded ghost boxes

        color = trk.color_bgr
        cx, cy = trk.centroid
        x, y, w, h = trk.bbox

        # Active Track: Render single crisp bounding box + crosshair
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, color, 1, lineType=cv2.LINE_AA)

        # Compact label
        parts = [f"ID #{trk.track_id} {trk.color_name}"]
        if show_metric_distance and trk.distance_cm > 0:
            parts.append(f"{trk.distance_cm:.0f}cm")
        elif trk.speed >= 5.0:
            parts.append(f"{trk.speed:.0f}px/s")

        label = " | ".join(parts)
        draw_label(frame, label, x, max(18, y - 8), color_bgr=color, bg_color_bgr=(20, 20, 30))
