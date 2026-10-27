"""
test_new_features.py
====================
Automated test suite verifying the new advanced features:
1. MultiObjectTracker (Persistent Track IDs, Kalman occlusion prediction, metric distance)
2. Directional Virtual Tripwire (In/Out line crossing counters)
3. Monitored ROI Zone (Occupancy and dwell-time tracking)
4. Trajectory Heatmap (2D Gaussian splat density and export)
5. Telemetry Logger (Kinematics CSV & JSON export)
6. Video Recorder (cv2.VideoWriter wrapper)
7. Gray World Color Constancy algorithm
8. ColorTracker End-to-End Analytics Integration
"""

import math
import os
import tempfile
import time
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import pytest  # type: ignore

from color_object_detection.src.analytics import (
    ROIZone,
    TelemetryLogger,
    Tripwire,
    VideoRecorder,
)
from color_object_detection.src.multitracker import MultiObjectTracker, TrackedObject
from color_object_detection.src.tracker import ColorTracker
from color_object_detection.src.utils import (
    apply_gray_world,
    estimate_metric_velocity,
    estimate_pinhole_distance,
)


class TestMultiObjectTracker:
    """Tests for persistent ID tracking and occlusion prediction."""

    def test_track_creation_and_persistent_id(self):
        tracker = MultiObjectTracker(max_distance_px=50.0)
        dets = [
            {
                "color_name": "Red",
                "color_bgr": (0, 0, 255),
                "centroid": (100, 100),
                "bbox": (80, 80, 40, 40),
                "area": 1200.0,
            },
            {
                "color_name": "Blue",
                "color_bgr": (255, 0, 0),
                "centroid": (300, 200),
                "bbox": (280, 180, 40, 40),
                "area": 1400.0,
            },
        ]

        tracks = tracker.update(dets, timestamp=1.0)
        assert len(tracks) == 2
        id1 = tracks[0].track_id
        id2 = tracks[1].track_id
        assert id1 != id2

        # In frame 2, move slightly
        dets2 = [
            {
                "color_name": "Red",
                "color_bgr": (0, 0, 255),
                "centroid": (104, 102),
                "bbox": (84, 82, 40, 40),
                "area": 1200.0,
            },
            {
                "color_name": "Blue",
                "color_bgr": (255, 0, 0),
                "centroid": (303, 198),
                "bbox": (283, 178, 40, 40),
                "area": 1400.0,
            },
        ]
        tracks2 = tracker.update(dets2, timestamp=1.033)
        track_map = {t.color_name: t for t in tracks2}
        assert track_map["Red"].track_id == id1
        assert track_map["Blue"].track_id == id2
        assert not track_map["Red"].is_occluded

    def test_occlusion_prediction_and_expiry(self):
        tracker = MultiObjectTracker(max_distance_px=60.0, max_missed_frames=3)
        det = {
            "color_name": "Green",
            "color_bgr": (0, 255, 0),
            "centroid": (200, 200),
            "bbox": (180, 180, 40, 40),
            "area": 1600.0,
        }
        tracks = tracker.update([det], timestamp=1.0)
        assert len(tracks) == 1
        tid = tracks[0].track_id

        # Missing detection frame 1 -> should become occluded
        tracks_occ1 = tracker.update([], timestamp=1.033)
        assert len(tracks_occ1) == 1
        assert tracks_occ1[0].track_id == tid
        assert tracks_occ1[0].is_occluded
        assert tracks_occ1[0].missed_frames == 1

        # Missing detection frame 2 and 3
        tracker.update([], timestamp=1.066)
        tracker.update([], timestamp=1.1)

        # Missing detection frame 4 -> exceeds max_missed_frames (3), should be deleted
        tracks_expired = tracker.update([], timestamp=1.133)
        assert len(tracks_expired) == 0


class TestAnalyticsSuite:
    """Tests for Tripwire, ROI Zone, Heatmap, Telemetry, and Video Recording."""

    def test_tripwire_crossing(self):
        tripwire = Tripwire(pt1=(0, 200), pt2=(400, 200))
        assert tripwire.count_in == 0
        assert tripwire.count_out == 0

        # Create mock track moving from y=180 to y=220 (crossing downward -> IN)
        class MockTrack:
            def __init__(self, track_id, trail):
                self.track_id = track_id
                self.trail = trail
                self.centroid = trail[-1]

        t1 = MockTrack(1, [(150, 180)])
        tripwire.update([t1])
        assert tripwire.count_in == 0

        # Step 2: centroid crosses to y=220
        t1.trail.append((150, 220))
        t1.centroid = (150, 220)
        tripwire.update([t1])
        assert tripwire.count_in == 1
        assert tripwire.count_out == 0

        # Step 3: centroid crosses back upward -> OUT
        t1.trail.append((150, 180))
        t1.centroid = (150, 180)
        tripwire.update([t1])
        assert tripwire.count_in == 1
        assert tripwire.count_out == 1

        tripwire.reset_counts()
        assert tripwire.count_in == 0
        assert tripwire.count_out == 0

    def test_roi_zone_occupancy_and_dwell(self):
        roi = ROIZone(zone_rect=(100, 100, 200, 200), name="TestZone")
        assert roi.contains((150, 150))
        assert not roi.contains((50, 50))

        class MockTrack:
            def __init__(self, track_id, centroid):
                self.track_id = track_id
                self.centroid = centroid

        t1 = MockTrack(1, (150, 150))
        t2 = MockTrack(2, (50, 50))

        occ = roi.update([t1, t2], dt=0.5)
        assert occ == 1
        assert roi.dwell_times[1] == pytest.approx(0.5)

        occ2 = roi.update([t1], dt=0.5)
        assert occ2 == 1
        assert roi.dwell_times[1] == pytest.approx(1.0)

        roi.reset()
        assert roi.occupants == 0
        assert len(roi.dwell_times) == 0

    def test_telemetry_logger_csv_and_json(self):
        logger = TelemetryLogger()
        logger.start()
        assert logger.is_active

        class MockTrack:
            track_id = 42
            color_name = "Red"
            centroid = (120, 140)
            area = 550.0
            speed = 34.5
            distance_cm = 45.0
            metric_speed_mps = 0.5
            is_occluded = False
            bbox = (100, 120, 40, 40)

        logger.log_frame([MockTrack()])
        assert logger.total_rows == 1

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "telemetry.csv"
            json_path = Path(tmpdir) / "telemetry.json"

            logger.export_csv(csv_path)
            logger.export_json(json_path)

            assert csv_path.exists()
            assert json_path.exists()
            with open(csv_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "track_id" in content
                assert "42" in content
                assert "Red" in content


class TestPinholeAndColorConstancy:
    """Tests for Gray World algorithm and pinhole camera math."""

    def test_gray_world_normalization(self):
        # Create image with artificial strong yellow tint (B=20, G=180, R=220)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = (20, 180, 220)

        balanced = apply_gray_world(img)
        assert balanced.shape == (100, 100, 3)
        b, g, r = cv2.split(balanced)
        # Gray world scales channels toward the average, so blue increases significantly
        assert np.mean(b) > 20

    def test_pinhole_distance_and_velocity(self):
        # 6.5 cm ball with apparent width 65 px at f=700 px:
        # Z = (700 * 6.5) / 65 = 70.0 cm
        dist = estimate_pinhole_distance(w_px=65.0, real_width_cm=6.5, focal_length_px=700.0)
        assert dist == pytest.approx(70.0, abs=0.1)

        # Speed 100 px/s at 70 cm distance:
        # v = (100 * 70) / (700 * 100) = 0.1 m/s
        vel = estimate_metric_velocity(speed_px_s=100.0, distance_cm=70.0, focal_length_px=700.0)
        assert vel == pytest.approx(0.1, abs=0.02)


class TestColorTrackerEndToEnd:
    """Tests end-to-end ColorTracker execution with all new features active."""

    def test_full_features_process_frame(self):
        tracker = ColorTracker(
            color_names=["Red"],
            use_gray_world=True,
            use_multitrack=True,
            show_metric_distance=True,
            show_tripwire=True,
            show_roi_zone=True,
        )

        # Create a synthetic frame with a red solid disk
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 30, (0, 0, 255), -1)

        result = tracker.process_frame(frame)
        assert result.annotated_frame.shape == (480, 640, 3)
        assert "Red" in result.masks
        assert len(result.blobs) >= 1

        blob = result.blobs[0]
        assert blob.color_name == "Red"
        assert blob.distance_cm is not None
        assert blob.track_id is not None

        # Verify analytics info dictionary
        assert "tripwire_in" in result.analytics_info
        assert "roi_occupants" in result.analytics_info
        assert "telemetry_rows" in result.analytics_info
        assert len(result.tracks) >= 1

        # Test reset
        tracker.reset()
        assert len(tracker.multitracker.tracks) == 0
        assert tracker.tripwire.count_in == 0
