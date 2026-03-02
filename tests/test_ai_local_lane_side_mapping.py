import time
import unittest

import numpy as np

from src.hardware.camera.threads.threadLineFollowing import threadLineFollowing


class AILocalLaneSideMappingTests(unittest.TestCase):
    def setUp(self):
        self.detector = threadLineFollowing.__new__(threadLineFollowing)
        self.detector.roi_height_start = 0.0
        self.detector.roi_height_end = 1.0
        self.detector.local_ai_max_result_age = 1.0
        self.detector._smooth_detected_line = lambda line, side: line

    def test_lane_side_points_preserve_both_lanes_even_on_same_half(self):
        lane_side_points = {
            "left": [[58, 95], [56, 75], [54, 55]],
            "right": [[82, 95], [80, 75], [78, 55]],
        }

        avg_left, avg_right = self.detector._lane_side_points_to_lines(lane_side_points, 100, 100)

        self.assertIsNotNone(avg_left)
        self.assertIsNotNone(avg_right)

        legacy_left, legacy_right = self.detector._lane_points_to_lines(
            [lane_side_points["left"], lane_side_points["right"]], 100, 100
        )
        self.assertIsNone(legacy_left)
        self.assertIsNotNone(legacy_right)

    def test_lane_side_points_preserve_single_left_lane_even_on_right_half(self):
        lane_side_points = {
            "left": [[72, 95], [70, 75], [68, 55]],
            "right": [],
        }

        avg_left, avg_right = self.detector._lane_side_points_to_lines(lane_side_points, 100, 100)

        self.assertIsNotNone(avg_left)
        self.assertIsNone(avg_right)

    def test_detect_with_local_ai_prefers_explicit_lane_side_points(self):
        self.detector._last_local_lane_payload = {
            "lane_points": [
                [[58, 95], [56, 75], [54, 55]],
                [[82, 95], [80, 75], [78, 55]],
            ],
            "lane_side_points": {
                "left": [[58, 95], [56, 75], [54, 55]],
                "right": [[82, 95], [80, 75], [78, 55]],
            },
            "inference_time_ms": 12.0,
            "frame_id": 7,
            "timestamp": time.time(),
            "model_ready": True,
        }

        avg_left, avg_right, _, _, _, debug_info = self.detector._detect_with_local_ai(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )

        self.assertIsNotNone(avg_left)
        self.assertIsNotNone(avg_right)
        self.assertEqual(len(debug_info["left_lines"]), 1)
        self.assertEqual(len(debug_info["right_lines"]), 1)
        self.assertEqual(debug_info["remote_lane_count"], 2)

    def test_detect_with_local_ai_prefers_explicit_lane_side_lines(self):
        self.detector._last_local_lane_payload = {
            "lane_points": [],
            "lane_side_points": {"left": [], "right": []},
            "lane_side_lines": {
                "left": [58, 95, 54, 55],
                "right": [82, 95, 78, 55],
            },
            "inference_time_ms": 12.0,
            "frame_id": 8,
            "timestamp": time.time(),
            "model_ready": True,
        }

        avg_left, avg_right, _, _, _, debug_info = self.detector._detect_with_local_ai(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )

        self.assertIsNotNone(avg_left)
        self.assertIsNotNone(avg_right)
        self.assertEqual(avg_left[0].tolist(), [58, 95, 54, 55])
        self.assertEqual(avg_right[0].tolist(), [82, 95, 78, 55])
        self.assertEqual(debug_info["remote_lane_count"], 2)

    def test_lane_points_fallback_still_works_without_side_metadata(self):
        lane_points = [
            [[22, 95], [26, 75], [30, 55]],
            [[78, 95], [74, 75], [70, 55]],
        ]

        avg_left, avg_right = self.detector._lane_points_to_lines(lane_points, 100, 100)

        self.assertIsNotNone(avg_left)
        self.assertIsNotNone(avg_right)

    def test_smooth_detected_line_holds_previous_line_for_brief_miss(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.line_visual_smoothing_alpha = 1.0
        detector.line_visual_missing_reset_frames = 2
        detector._smoothed_left_line = None
        detector._left_line_missing_frames = 0
        detector._smoothed_right_line = None
        detector._right_line_missing_frames = 0

        line = np.array([[10, 90, 20, 50]], dtype=np.int32)
        first = detector._smooth_detected_line(line, 'left')
        held = detector._smooth_detected_line(None, 'left')
        dropped = detector._smooth_detected_line(None, 'left')

        self.assertEqual(first[0].tolist(), [10, 90, 20, 50])
        self.assertEqual(held[0].tolist(), [10, 90, 20, 50])
        self.assertIsNone(dropped)

    def test_single_line_error_uses_visible_side_geometry(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector._last_px_per_cm = 1.0
        detector.lane_width_cm = 35.0
        detector.line_width_cm = 2.0
        detector.car_width = 19.0
        detector.single_line_offset_factor = 0.5
        detector._single_line_heading_ref_left = 0.0
        detector._single_line_heading_ref_right = 0.0

        left_line = np.array([[30, 95, 30, 55]], dtype=np.int32)
        right_line = np.array([[70, 95, 70, 55]], dtype=np.int32)

        left_error, left_heading = detector._compute_single_line_error(
            left_line, "left", 100, 100, prefer_center=True
        )
        right_error, right_heading = detector._compute_single_line_error(
            right_line, "right", 100, 100, prefer_center=True
        )

        self.assertAlmostEqual(left_error, -1.5, places=1)
        self.assertAlmostEqual(right_error, 1.5, places=1)
        self.assertEqual(left_heading, 0.0)
        self.assertEqual(right_heading, 0.0)


if __name__ == "__main__":
    unittest.main()
