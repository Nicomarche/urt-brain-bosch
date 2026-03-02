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

    def test_lane_points_fallback_still_works_without_side_metadata(self):
        lane_points = [
            [[22, 95], [26, 75], [30, 55]],
            [[78, 95], [74, 75], [70, 55]],
        ]

        avg_left, avg_right = self.detector._lane_points_to_lines(lane_points, 100, 100)

        self.assertIsNotNone(avg_left)
        self.assertIsNotNone(avg_right)


if __name__ == "__main__":
    unittest.main()
