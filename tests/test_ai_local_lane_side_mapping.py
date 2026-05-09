import math
import os
import tempfile
import time
import unittest

import numpy as np

from src.hardware.camera.threads.localPerceptionEngine import LocalPerceptionEngine
from src.hardware.camera.threads.threadLineFollowing import threadLineFollowing


class _CaptureBuffer:
    def __init__(self):
        self.value = None
        self.timestamp = None

    def write(self, value, timestamp=None):
        self.value = value
        self.timestamp = timestamp


class AILocalLaneSideMappingTests(unittest.TestCase):
    @staticmethod
    def _mask_from_points(points, shape=(100, 100)):
        mask = np.zeros(shape, dtype=np.uint8)
        height, width = shape
        for x, y in points:
            x = int(x)
            y = int(y)
            x1 = max(0, x - 1)
            x2 = min(width, x + 2)
            y1 = max(0, y - 1)
            y2 = min(height, y + 2)
            mask[y1:y2, x1:x2] = 255
        return mask

    def setUp(self):
        self.detector = threadLineFollowing.__new__(threadLineFollowing)
        self.detector.roi_height_start = 0.0
        self.detector.roi_height_end = 1.0
        self.detector.local_ai_max_result_age = 1.0
        self.detector.ai_local_use_mask_geometry = True
        self.detector.lookahead = 0.4
        self.detector.lane_width_cm = 35.0
        self.detector.line_width_cm = 2.0
        self.detector.car_width = 19.0
        self.detector.lane_safety_margin_cm = 5.0
        self.detector.camera_to_front_axle = 0.0
        self.detector._last_local_ai_lane_width_px = None
        self.detector._last_px_per_cm = None
        self.detector._last_good_steering = 0.0
        self.detector._curve_state = "STRAIGHT"
        self.detector._last_processed_sequence = 0
        self.detector._last_two_line_reference_y = 0
        self.detector._last_two_line_ref_px_per_cm = 0.0
        self.detector._last_two_line_reference_monotonic = 0.0
        self.detector._last_two_line_reference_sequence = 0
        self.detector.show_debug = False
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

    def test_build_local_mask_guidance_uses_raw_masks(self):
        left_mask = self._mask_from_points([(30, 95), (29, 85), (28, 75), (26, 65), (24, 55)])
        right_mask = self._mask_from_points([(70, 95), (71, 85), (72, 75), (74, 65), (76, 55)])

        guidance = self.detector._build_local_mask_guidance(
            {"left": left_mask, "right": right_mask},
            100,
            100,
        )

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance["detected_sides"], ("left", "right"))
        self.assertFalse(guidance["used_virtual_boundary"])
        self.assertGreater(guidance["lane_width_px"], 0.0)
        self.assertAlmostEqual(guidance["midpoint_x"], 50.0, delta=5.0)

    def test_build_local_mask_guidance_estimates_virtual_boundary_for_single_side(self):
        self.detector._last_local_ai_lane_width_px = 36.0
        left_mask = self._mask_from_points([(32, 95), (31, 85), (30, 75), (29, 65), (28, 55)])

        guidance = self.detector._build_local_mask_guidance(
            {"left": left_mask, "right": None},
            100,
            100,
        )

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance["detected_sides"], ("left",))
        self.assertTrue(guidance["used_virtual_boundary"])
        self.assertAlmostEqual(guidance["right_x"] - guidance["left_x"], 36.0, delta=1.5)

    def test_build_local_mask_guidance_clamps_single_side_error_to_lane_window(self):
        self.detector._last_local_ai_lane_width_px = 36.0
        left_mask = self._mask_from_points([(4, 95), (4, 85), (4, 75), (4, 65), (4, 55)])

        guidance = self.detector._build_local_mask_guidance(
            {"left": left_mask, "right": None},
            100,
            100,
        )

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance["guidance_mode"], "midpoint")
        self.assertIsNotNone(guidance["single_side_error_limit_px"])
        self.assertLess(guidance["raw_error_px"], -guidance["single_side_error_limit_px"])
        self.assertAlmostEqual(
            abs(guidance["error_px"]),
            guidance["single_side_error_limit_px"],
            delta=0.1,
        )

    def test_build_local_mask_guidance_collapses_implausibly_narrow_two_line_width(self):
        self.detector._last_local_ai_lane_width_px = 90.0
        left_mask = self._mask_from_points([(40, 95), (40, 85), (40, 75), (40, 65), (40, 55)])
        right_mask = self._mask_from_points([(60, 95), (60, 85), (60, 75), (60, 65), (60, 55)])

        guidance = self.detector._build_local_mask_guidance(
            {"left": left_mask, "right": right_mask},
            100,
            100,
        )

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance["detected_sides"], ("right",))
        self.assertTrue(guidance["used_virtual_boundary"])
        self.assertIn("duplicate_line_collapse", guidance)
        self.assertEqual(guidance["duplicate_line_collapse"]["trigger"], "lane_width_too_small")

    def test_collapse_overlapping_two_lines_treats_detection_as_single_line(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.lookahead = 0.4
        detector.lane_width_cm = 35.0
        detector._last_local_ai_lane_width_px = 80.0
        detector._last_px_per_cm = None
        detector.last_seen_side = "both"
        detector._line_x_at_y = threadLineFollowing._line_x_at_y.__get__(detector, threadLineFollowing)
        detector._resolve_nominal_lane_width_px = threadLineFollowing._resolve_nominal_lane_width_px.__get__(
            detector, threadLineFollowing
        )
        detector._resolve_ambiguous_local_lane = (
            lambda line, img_h, img_w, prev_seen_side=None: ("right", "history")
        )

        avg_left = np.array([[50, 95, 50, 55]], dtype=np.int32)
        avg_right = np.array([[53, 95, 53, 55]], dtype=np.int32)

        collapsed_left, collapsed_right, info = detector._collapse_overlapping_two_lines(
            avg_left, avg_right, 100, 100, prev_seen_side="both"
        )

        self.assertIsNone(collapsed_left)
        self.assertIsNotNone(collapsed_right)
        self.assertIsNotNone(info)
        self.assertEqual(info["kept_side"], "right")
        self.assertLess(info["line_gap_px"], info["line_gap_limit_px"])

    def test_single_line_transition_blend_softens_side_switch_spike(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.single_line_transition_blend_frames = 2
        detector.max_steering = 25.0
        detector.last_steering = 20.0
        detector._last_good_steering = 20.0

        blended, info = detector._blend_single_line_transition_steering(
            steering_angle=-20.0,
            side="left",
            prev_seen_side="right",
            streak=1,
        )

        self.assertIsNotNone(info)
        self.assertTrue(info["applied"])
        self.assertAlmostEqual(blended, 6.6667, places=3)

    def test_noise_filter_allows_large_error_jump_during_curve_entering(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.use_noise_filter = True
        detector.noise_max_hough_lines = 40
        detector.noise_max_error_jump_px = 80
        detector.noise_max_steer_jump_deg = 15
        detector.noise_max_reject_frames = 3
        detector._noise_reject_count = 0
        detector._last_good_error = -59.0
        detector._last_good_steering = -2.76
        detector._last_good_num_lines = 2
        detector._curve_state = "ENTERING"

        is_noisy, reason = detector._is_frame_noisy({}, 30.0, -17.5, 2)

        self.assertFalse(is_noisy)
        self.assertEqual(reason, "")

    def test_noise_filter_still_rejects_large_error_jump_on_straight(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.use_noise_filter = True
        detector.noise_max_hough_lines = 40
        detector.noise_max_error_jump_px = 80
        detector.noise_max_steer_jump_deg = 15
        detector.noise_max_reject_frames = 3
        detector._noise_reject_count = 0
        detector._last_good_error = -59.0
        detector._last_good_steering = -2.76
        detector._last_good_num_lines = 2
        detector._curve_state = "STRAIGHT"

        is_noisy, reason = detector._is_frame_noisy({}, 30.0, -17.5, 2)

        self.assertTrue(is_noisy)
        self.assertTrue(reason.startswith("error_jump("))

    def test_compute_single_line_physical_error_rejects_impossible_lane_geometry(self):
        debug_info = {}
        px_per_cm = 309.75 / self.detector.lane_width_cm

        direct_error_m = self.detector._compute_sl_physical_error(
            "right",
            243.56,
            640,
            px_per_cm,
            debug_info,
        )

        self.assertIsNone(direct_error_m)
        self.assertFalse(debug_info["direct_error_valid"])
        self.assertEqual(debug_info["direct_error_reason"], "impossible_lane_geometry")
        self.assertAlmostEqual(debug_info["sl_D_left_cm"], 43.64, places=2)
        self.assertAlmostEqual(debug_info["sl_D_right_cm"], -8.64, places=2)

    def test_single_line_physical_error_rejects_stale_two_line_reference(self):
        self.detector._last_two_line_reference_y = 75
        self.detector._last_two_line_ref_px_per_cm = 8.0
        self.detector._last_two_line_reference_monotonic = time.monotonic() - 0.50
        self.detector._last_two_line_reference_sequence = 3
        self.detector._last_processed_sequence = 10
        debug_info = {}
        mask_guidance_line = np.array([[250, 95, 230, 55]], dtype=np.int32)

        direct_error_m = self.detector._get_single_line_physical_direct_error(
            "left",
            mask_guidance_line,
            {},
            640,
            debug_info=debug_info,
        )

        self.assertIsNone(direct_error_m)
        self.assertFalse(debug_info["direct_error_valid"])
        self.assertEqual(debug_info["direct_error_reason"], "stale_two_line_reference")

    def test_prepare_local_ai_side_masks_reassigns_wrong_explicit_single_side_using_history(self):
        self.detector._last_two_line_left = np.array([[28, 95, 26, 55]], dtype=np.int32)
        self.detector._last_two_line_right = np.array([[72, 95, 70, 55]], dtype=np.int32)
        self.detector._last_two_line_ts = time.time()
        self.detector.last_seen_side = "both"

        left_mask = self._mask_from_points([(72, 95), (71, 85), (70, 75), (69, 65), (68, 55)])
        side_masks = {"left": left_mask, "right": None}
        lane_side_sources = {"left": "explicit_class", "right": "none"}
        lane_side_lines = {"left": [72, 95, 68, 55], "right": []}

        prepared_masks, prepared_lines, resolution = self.detector._prepare_local_ai_side_masks(
            side_masks,
            lane_side_sources,
            lane_side_lines,
            100,
            100,
        )

        self.assertIsNone(prepared_masks["left"])
        self.assertIsNotNone(prepared_masks["right"])
        self.assertIsNone(prepared_lines["left"])
        self.assertIsNotNone(prepared_lines["right"])
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution["detected_side"], "left")
        self.assertEqual(resolution["resolved_side"], "right")
        self.assertEqual(resolution["source"], "explicit_class")
        self.assertEqual(resolution["resolution_source"], "history")

    def test_publish_visual_pipeline_state_keeps_valid_single_line_measurement_in_route_tracking(self):
        capture = _CaptureBuffer()
        self.detector.visual_candidate_buffer = None
        self.detector.visual_state_buffer = capture
        self.detector._control_policy_mode = "ROUTE_TRACKING"
        self.detector._planner_priority_active = True
        self.detector.is_line_following_active = True
        self.detector.detection_mode = "ai_local"
        self.detector._curve_state = "IN_CURVE"
        self.detector._heading_error = 0.0
        self.detector._last_camera_yaw_hint_rad = None
        self.detector._last_camera_yaw_hint_confidence = 0.0
        self.detector._build_local_lane_payload_log = lambda: {}
        self.detector._last_stopline_visual_debug = {}
        self.detector._last_frame_trace = {
            "debug": {
                "measurement_mode": "single_line",
                "sl_direct_error_m": -0.08,
                "direct_error_valid": True,
                "direct_error_reason": "single_line_physical",
            }
        }

        self.detector._publish_visual_pipeline_state(None, frame_sequence=17)

        self.assertIsNotNone(capture.value)
        debug_info = capture.value.frame_trace["debug"]
        self.assertEqual(debug_info["control_policy_mode"], "ROUTE_TRACKING")
        self.assertTrue(debug_info["planner_priority_active"])
        self.assertTrue(debug_info["direct_error_valid"])
        self.assertEqual(debug_info["direct_error_reason"], "single_line_physical")

    def test_build_local_mask_guidance_single_line_uses_conservative_center_after_streak(self):
        self.detector.consecutive_single_right = 4
        self.detector.single_line_prefer_center_frames = 2
        self.detector.single_line_offset_factor = 0.2
        right_mask = self._mask_from_points([(70, 95), (70, 85), (70, 75), (70, 65), (70, 55)])

        guidance = self.detector._build_local_mask_guidance(
            {"left": None, "right": right_mask},
            100,
            100,
            side_lines={"right": np.array([[70, 95, 70, 55]], dtype=np.int32)},
        )

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance["guidance_mode"], "single_line_physical")
        self.assertFalse(guidance["single_line_prefer_center"])
        self.assertEqual(guidance["single_line_streak"], 5)

    def test_detect_with_local_ai_prefers_raw_masks_when_enabled(self):
        left_mask = self._mask_from_points([(30, 95), (29, 85), (28, 75), (26, 65), (24, 55)])
        right_mask = self._mask_from_points([(70, 95), (71, 85), (72, 75), (74, 65), (76, 55)])
        lane_mask = np.bitwise_or(left_mask, right_mask)
        self.detector._last_local_lane_payload = {
            "lane_points": [],
            "lane_side_points": {
                "left": [[58, 95], [56, 75], [54, 55]],
                "right": [[82, 95], [80, 75], [78, 55]],
            },
            "lane_side_lines": {
                "left": [58, 95, 54, 55],
                "right": [82, 95, 78, 55],
            },
            "side_masks": {"left": left_mask, "right": right_mask},
            "lane_mask": lane_mask,
            "inference_time_ms": 12.0,
            "frame_id": 17,
            "timestamp": time.time(),
            "model_ready": True,
        }

        avg_left, avg_right, _, _, returned_mask, debug_info = self.detector._detect_with_local_ai(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )

        self.assertIsNone(avg_left)
        self.assertIsNone(avg_right)
        self.assertEqual(debug_info["control_mode"], "mask_guidance")
        self.assertEqual(debug_info["remote_lane_count"], 2)
        self.assertEqual(debug_info["left_lines"], [])
        self.assertEqual(debug_info["right_lines"], [])
        self.assertIn("local_mask_guidance", debug_info)
        self.assertTrue(np.array_equal(returned_mask, lane_mask))

    def test_detect_with_local_ai_collapses_overlapping_mask_sides_to_one_lane(self):
        duplicate_mask = self._mask_from_points([(62, 95), (60, 85), (58, 75), (56, 65), (54, 55)])
        self.detector._last_local_lane_payload = {
            "lane_points": [],
            "lane_side_points": {"left": [], "right": []},
            "lane_side_lines": {
                "left": [62, 95, 54, 55],
                "right": [61, 95, 53, 55],
            },
            "lane_side_sources": {"left": "explicit_class", "right": "explicit_class"},
            "side_masks": {"left": duplicate_mask, "right": duplicate_mask.copy()},
            "lane_mask": duplicate_mask.copy(),
            "inference_time_ms": 12.0,
            "frame_id": 19,
            "timestamp": time.time(),
            "model_ready": True,
        }

        avg_left, avg_right, _, _, _, debug_info = self.detector._detect_with_local_ai(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )

        self.assertIsNone(avg_left)
        self.assertIsNone(avg_right)
        self.assertEqual(debug_info["control_mode"], "mask_guidance")
        self.assertEqual(debug_info["remote_lane_count"], 1)
        self.assertEqual(debug_info["local_mask_guidance"]["detected_sides"], ("right",))
        self.assertEqual(debug_info["single_line_side_source"], "guessed_single:frame_center")
        self.assertEqual(debug_info["single_line_resolved_side"], "right")
        self.assertIsNone(debug_info["render_side_masks"]["left"])
        self.assertIsNotNone(debug_info["render_side_masks"]["right"])

    def test_collapse_duplicate_local_ai_sides_uses_line_gap_when_masks_are_sparse(self):
        left_mask = self._mask_from_points([(62, 95), (61, 94), (60, 93)])
        right_mask = self._mask_from_points([(63, 40), (62, 39), (61, 38)])
        side_masks = {"left": left_mask.copy(), "right": right_mask.copy()}
        side_lines = {
            "left": np.array([[62, 95, 54, 55]], dtype=np.int32),
            "right": np.array([[64, 95, 56, 55]], dtype=np.int32),
        }
        lane_side_sources = {"left": "explicit_class", "right": "explicit_class"}

        collapse = self.detector._collapse_duplicate_local_ai_sides(
            side_masks,
            side_lines,
            lane_side_sources,
            100,
            100,
        )

        self.assertIsNotNone(collapse)
        self.assertIsNotNone(collapse["line_gap_px"])
        self.assertLess(collapse["line_gap_px"], 6.0)
        kept_side = collapse["kept_side"]
        dropped_side = "right" if kept_side == "left" else "left"
        self.assertIsNotNone(side_masks[kept_side])
        self.assertIsNone(side_masks[dropped_side])

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

    def test_detect_with_local_ai_flag_disabled_falls_back_to_legacy_lines(self):
        self.detector.ai_local_use_mask_geometry = False
        left_mask = self._mask_from_points([(30, 95), (29, 85), (28, 75), (26, 65), (24, 55)])
        right_mask = self._mask_from_points([(70, 95), (71, 85), (72, 75), (74, 65), (76, 55)])
        self.detector._last_local_lane_payload = {
            "lane_points": [],
            "lane_side_points": {"left": [], "right": []},
            "lane_side_lines": {
                "left": [58, 95, 54, 55],
                "right": [82, 95, 78, 55],
            },
            "side_masks": {"left": left_mask, "right": right_mask},
            "lane_mask": np.bitwise_or(left_mask, right_mask),
            "inference_time_ms": 12.0,
            "frame_id": 18,
            "timestamp": time.time(),
            "model_ready": True,
        }

        avg_left, avg_right, _, _, _, debug_info = self.detector._detect_with_local_ai(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )

        self.assertIsNotNone(avg_left)
        self.assertIsNotNone(avg_right)
        self.assertEqual(debug_info["control_mode"], "legacy_line_fit")
        self.assertNotIn("local_mask_guidance", debug_info)
        self.assertEqual(len(debug_info["left_lines"]), 1)
        self.assertEqual(len(debug_info["right_lines"]), 1)

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

    def test_local_ai_ignores_guessed_single_side_and_re_resolves_from_history(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.roi_height_start = 0.0
        detector.roi_height_end = 1.0
        detector.local_ai_max_result_age = 1.0
        detector.last_seen_side = "both"
        detector._smooth_detected_line = lambda line, side: line
        detector._last_two_line_left = np.array([[18, 95, 18, 55]], dtype=np.int32)
        detector._last_two_line_right = np.array([[74, 95, 74, 55]], dtype=np.int32)
        detector._last_two_line_ts = time.time()
        detector._last_local_lane_payload = {
            "lane_points": [],
            "lane_side_points": {"left": [[72, 95], [70, 75], [68, 55]], "right": []},
            "lane_side_lines": {"left": [72, 95, 68, 55], "right": []},
            "lane_side_sources": {"left": "guessed_single", "right": "none"},
            "inference_time_ms": 12.0,
            "frame_id": 9,
            "timestamp": time.time(),
            "model_ready": True,
        }

        avg_left, avg_right, _, _, _, debug_info = detector._detect_with_local_ai(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )

        self.assertIsNone(avg_left)
        self.assertIsNotNone(avg_right)
        self.assertEqual(debug_info["single_line_side_source"], "guessed_single:history")

    def test_ambiguous_single_lane_prefers_last_two_line_boundary_over_frame_center(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.lookahead = 0.4
        detector.local_ai_max_result_age = 1.0
        detector._last_two_line_left = np.array([[10, 95, 10, 55]], dtype=np.int32)
        detector._last_two_line_right = np.array([[48, 95, 48, 55]], dtype=np.int32)
        detector._last_two_line_ts = time.time()

        side, source = detector._resolve_ambiguous_local_lane(
            np.array([[46, 95, 46, 55]], dtype=np.int32),
            100,
            100,
            prev_seen_side="both",
        )

        self.assertEqual(side, "right")
        self.assertEqual(source, "history")

    def test_physical_single_line_error_uses_lookahead_row_not_image_bottom(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector._last_px_per_cm = 1.0
        detector.lane_width_cm = 35.0
        detector.line_width_cm = 2.0
        detector.car_width = 19.0
        detector.single_line_offset_factor = 0.5
        detector.lookahead = 0.4
        detector.camera_to_front_axle = 0.0
        detector._single_line_heading_ref_left = 0.0
        detector._single_line_heading_ref_right = 0.0

        line = np.array([[30, 95, 10, 55]], dtype=np.int32)

        physical_error, _ = detector._compute_single_line_error(
            line, "left", 100, 100, prefer_center=True, physical_projection=True
        )
        self.assertEqual(detector._last_single_line_projection_debug["single_line_reference_y"], 60)
        self.assertEqual(detector._last_single_line_projection_debug["single_line_mode"], "physical")
        legacy_error, _ = detector._compute_single_line_error(
            line, "left", 100, 100, prefer_center=True, physical_projection=False
        )

        self.assertAlmostEqual(physical_error, -19.0, places=1)
        self.assertAlmostEqual(legacy_error, -1.5, places=1)
        self.assertEqual(detector._last_single_line_projection_debug["single_line_mode"], "legacy")

    def test_physical_single_line_error_applies_camera_to_front_axle_compensation(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector._last_px_per_cm = 1.0
        detector.lane_width_cm = 35.0
        detector.line_width_cm = 2.0
        detector.car_width = 19.0
        detector.single_line_offset_factor = 0.5
        detector.lookahead = 0.4
        detector._single_line_heading_ref_left = 0.0
        detector._single_line_heading_ref_right = 0.0

        line = np.array([[30, 95, 10, 55]], dtype=np.int32)

        detector.camera_to_front_axle = 0.0
        no_shift_error, _ = detector._compute_single_line_error(
            line, "left", 100, 100, prefer_center=True, physical_projection=True
        )
        detector.camera_to_front_axle = 10.0
        shifted_error, _ = detector._compute_single_line_error(
            line, "left", 100, 100, prefer_center=True, physical_projection=True
        )

        self.assertAlmostEqual(shifted_error - no_shift_error, 5.0, places=1)
        self.assertAlmostEqual(
            detector._last_single_line_projection_debug["single_line_camera_shift_px"],
            -5.0,
            places=1,
        )

    def test_ai_local_single_line_uses_geometric_center_not_single_line_offset_factor(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.detection_mode = "ai_local"
        detector._last_px_per_cm = 1.0
        detector.lane_width_cm = 35.0
        detector.line_width_cm = 2.0
        detector.car_width = 19.0
        detector.lookahead = 0.4
        detector.camera_to_front_axle = 0.0
        detector._single_line_heading_ref_left = 0.0
        detector._single_line_heading_ref_right = 0.0

        line = np.array([[30, 95, 30, 55]], dtype=np.int32)

        detector.single_line_offset_factor = 0.1
        error_low_bias, _ = detector._compute_single_line_error(
            line,
            "left",
            100,
            100,
            prefer_center=(detector.detection_mode == "ai_local"),
            physical_projection=(detector.detection_mode == "ai_local"),
        )
        detector.single_line_offset_factor = 0.9
        error_high_bias, _ = detector._compute_single_line_error(
            line,
            "left",
            100,
            100,
            prefer_center=(detector.detection_mode == "ai_local"),
            physical_projection=(detector.detection_mode == "ai_local"),
        )

        self.assertAlmostEqual(error_low_bias, -1.5, places=1)
        self.assertAlmostEqual(error_high_bias, -1.5, places=1)

    def test_single_line_error_adds_curve_heading_bias_for_shallow_lane(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector._last_px_per_cm = 1.0
        detector.lane_width_cm = 35.0
        detector.line_width_cm = 2.0
        detector.car_width = 19.0
        detector.single_line_offset_factor = 0.5
        detector.lookahead = 0.4
        detector.camera_to_front_axle = 0.0
        detector._single_line_heading_ref_left = 0.0
        detector._single_line_heading_ref_right = 0.0
        detector._curve_state = "STRAIGHT"
        detector._curve_direction = 0

        curve_like_left = np.array([[60, 95, 20, 75]], dtype=np.int32)

        _, heading = detector._compute_single_line_error(
            curve_like_left, "left", 100, 100, prefer_center=True, physical_projection=True
        )

        self.assertGreater(heading, 0.1)
        self.assertTrue(detector._last_single_line_projection_debug["single_line_curve_like"])
        self.assertGreater(detector._last_single_line_projection_debug["single_line_curve_strength"], 0.0)

    def test_single_line_error_forces_negative_when_outer_curve_is_confirmed_by_history(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector._last_px_per_cm = 1.0
        detector.lane_width_cm = 35.0
        detector.line_width_cm = 2.0
        detector.car_width = 19.0
        detector.single_line_offset_factor = 0.5
        detector.single_line_outer_bias_gain = 0.6
        detector.lookahead = 0.4
        detector.camera_to_front_axle = 0.0
        detector.local_ai_max_result_age = 1.0
        detector._single_line_heading_ref_left = 0.0
        detector._single_line_heading_ref_right = 0.0
        detector._curve_state = "STRAIGHT"
        detector._curve_direction = 0
        detector._last_two_line_left = np.array([[50, 95, 50, 55]], dtype=np.int32)
        detector._last_two_line_right = np.array([[80, 95, 80, 55]], dtype=np.int32)
        detector._last_two_line_ts = time.time()
        detector.consecutive_single_left = 0
        detector.consecutive_single_right = 0

        curve_like_left = np.array([[120, 95, 40, 55]], dtype=np.int32)

        error, _ = detector._compute_single_line_error(
            curve_like_left, "left", 100, 100, prefer_center=True, physical_projection=True
        )

        self.assertLess(error, 0.0)
        self.assertGreater(error, -7.5)
        self.assertTrue(detector._last_single_line_projection_debug["single_line_outer_confirmed"])
        self.assertEqual(
            detector._last_single_line_projection_debug["single_line_outer_confirm_sources"],
            "history",
        )

    def test_lane_observation_history_tracks_visible_and_lost_side(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.local_ai_max_result_age = 1.0
        detector._last_two_line_left = np.array([[20, 95, 20, 55]], dtype=np.int32)
        detector._last_two_line_right = np.array([[80, 95, 80, 55]], dtype=np.int32)
        detector._last_two_line_ts = time.time()

        detector._record_lane_observation(("left", "right"))
        entry = detector._record_lane_observation(("left",))
        debug_fields = detector._lane_observation_debug_fields(entry)

        self.assertEqual(entry["visible_side"], "left")
        self.assertEqual(entry["inferred_lost_side"], "right")
        self.assertEqual(debug_fields["visible_lane_side"], "left")
        self.assertEqual(debug_fields["lost_lane_side"], "right")
        self.assertIn("transition_lost_right", debug_fields["lane_context_sources"])

    def test_auto_run_log_resets_when_auto_is_reentered(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.auto_run_log_path = os.path.join(
            tempfile.gettempdir(),
            "line_following_auto_last_run_test.txt",
        )
        detector.detection_mode = "ai_local"
        detector._auto_run_log_enabled = False
        detector._auto_run_log_frame_idx = 0
        detector._last_frame_trace = {}

        detector._reset_auto_run_log("AUTO")
        detector._append_auto_run_log_entry("FRAME 0001", {"frame": 1})

        with open(detector.auto_run_log_path, "r", encoding="utf-8") as log_file:
            first_log = log_file.read()

        detector._reset_auto_run_log("AUTO")

        with open(detector.auto_run_log_path, "r", encoding="utf-8") as log_file:
            second_log = log_file.read()

        self.assertIn("AUTO RUN RESET", first_log)
        self.assertIn("FRAME 0001", first_log)
        self.assertIn("AUTO RUN RESET", second_log)
        self.assertNotIn("FRAME 0001", second_log)

    def test_curve_recovery_can_trigger_in_ai_local_single_side_context(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.use_curve_recovery = True
        detector.detection_mode = "ai_local"
        detector.last_seen_side = "left"
        detector._curve_state = "STRAIGHT"
        detector._curve_direction = 0
        detector._noise_reject_count = 0
        detector._recovery_state = "NONE"
        detector._recovery_start_time = 0.0
        detector._recovery_actual_rev_time = 0.0
        detector._recovery_reverse_steer_angle = 0.0
        detector._recovery_curve_sign = 0
        detector._max_steer_consecutive = 2
        detector._error_at_max_steer_start = 100.0
        detector._last_good_error = 100.0
        detector.max_steering = 25.0
        detector.max_error_px = 40.0
        detector.recovery_max_steer_frames = 3
        detector.recovery_reverse_time_min = 0.3
        detector.recovery_reverse_time_max = 1.5
        detector.recovery_reverse_steer_scale = 1.5
        detector.recovery_error_shrink_ratio = 0.85

        steer, speed, active = detector._check_curve_recovery(-25.0, 10.0, None)

        self.assertTrue(active)
        self.assertEqual(steer, 0)
        self.assertEqual(speed, 0)
        self.assertEqual(detector._recovery_state, "STOPPING")
        self.assertEqual(detector._recovery_curve_sign, -1)

    def test_single_line_protective_stop_triggers_after_sustained_saturation(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.detection_mode = "ai_local"
        detector.max_steering = 25.0
        detector.max_error_px = 40.0
        detector.single_line_stop_error_scale = 1.5
        detector.single_line_stop_max_steer_frames = 3
        detector._single_line_stop_consecutive = 0

        local_mask_guidance = {
            "guidance_mode": "single_line_physical",
            "detected_sides": ("right",),
            "lane_width_px": 100.0,
            "raw_error_px": -80.0,
        }

        self.assertFalse(
            detector._should_apply_single_line_protective_stop(local_mask_guidance, -25.0, -80.0)
        )
        self.assertFalse(
            detector._should_apply_single_line_protective_stop(local_mask_guidance, -25.0, -80.0)
        )
        self.assertTrue(
            detector._should_apply_single_line_protective_stop(local_mask_guidance, -25.0, -80.0)
        )

    def test_progressive_speed_respects_safety_cap(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.highway_mode_event = None
        detector.max_speed = 20.0
        detector.min_speed = 10.0
        detector.max_steering = 25.0
        detector.speed_ramp_step = 0.5
        detector._current_speed = 18.0

        updated_speed = detector._update_progressive_speed(steering_angle=0.0, speed_cap=12.0)

        self.assertEqual(updated_speed, 12.0)
        self.assertEqual(detector._current_speed, 12.0)

    def test_no_lane_hold_uses_last_good_steering_for_initial_blind_frames(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.max_steering = 25.0
        detector.frames_without_line = 2
        detector._last_good_steering = -18.0

        self.assertEqual(detector._get_no_lane_hold_steering(), -18.0)

        detector.frames_without_line = 5
        self.assertIsNone(detector._get_no_lane_hold_steering())

    def test_two_line_curve_priority_holds_previous_steer_through_deadband(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector._curve_state = "STRAIGHT"
        detector._curve_enter_origin = "none"
        detector._curve_direction = 0
        detector._local_ai_heading_hint_source = "two_line"
        detector._local_ai_heading_hint_confidence = 0.68
        detector._local_ai_heading_hint_rad = math.radians(19.0)
        detector._last_good_steering = -1.0
        detector.last_steering = -1.0
        detector.mpc_output_deadband_deg = 0.5
        detector.max_steering = 25.0

        debug_info = {}
        guarded = detector._enforce_two_line_curve_priority(0.0, debug_info=debug_info)

        self.assertLess(guarded, -0.7)
        self.assertTrue(debug_info["two_line_curve_priority"])
        self.assertEqual(debug_info["two_line_curve_priority_mode"], "continuity_hold")

    def test_two_line_curve_priority_allows_real_reversal_command(self):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector._curve_state = "STRAIGHT"
        detector._curve_enter_origin = "none"
        detector._curve_direction = 0
        detector._local_ai_heading_hint_source = "two_line"
        detector._local_ai_heading_hint_confidence = 0.68
        detector._local_ai_heading_hint_rad = math.radians(19.0)
        detector._last_good_steering = -1.0
        detector.last_steering = -1.0
        detector.mpc_output_deadband_deg = 0.5
        detector.max_steering = 25.0

        guarded = detector._enforce_two_line_curve_priority(3.0, debug_info={})

        self.assertEqual(guarded, 3.0)

class LocalPerceptionEngineLaneGeometryTests(unittest.TestCase):
    def test_build_lane_geometry_collapses_overlapping_sides_to_single_lane(self):
        engine = LocalPerceptionEngine.__new__(LocalPerceptionEngine)
        duplicate_mask = AILocalLaneSideMappingTests._mask_from_points(
            [(62, 95), (60, 85), (58, 75), (56, 65), (54, 55)]
        )
        side_candidates = {
            "left": [
                {
                    "mask": duplicate_mask,
                    "x_center": 58.0,
                    "confidence": 0.95,
                    "side_source": "explicit_class",
                }
            ],
            "right": [
                {
                    "mask": duplicate_mask.copy(),
                    "x_center": 58.0,
                    "confidence": 0.94,
                    "side_source": "explicit_class",
                }
            ],
        }

        side_masks, lane_points, lane_side_points, _, lane_side_sources, lane_mask = engine._build_lane_geometry(
            side_candidates
        )

        self.assertIsNone(side_masks["left"])
        self.assertIsNotNone(side_masks["right"])
        self.assertEqual(len(lane_points), 1)
        self.assertEqual(lane_side_points["left"], [])
        self.assertTrue(lane_side_points["right"])
        self.assertEqual(lane_side_sources["left"], "none")
        self.assertEqual(lane_side_sources["right"], "guessed_single")
        self.assertTrue(np.array_equal(lane_mask, side_masks["right"]))


if __name__ == "__main__":
    unittest.main()
