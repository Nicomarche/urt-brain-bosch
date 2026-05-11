import unittest

from src.perception.lane.lane_observer_thread import threadLaneObserver
from src.core.types import VisualControlCandidate, VisualStateSnapshot


class LaneObserverTests(unittest.TestCase):
    def test_build_lane_observation_preserves_detected_sides_and_direct_error(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=10.0,
            detection_mode="ai_local",
            curve_state="IN_CURVE",
            heading_error_rad=0.12,
            camera_yaw_hint_rad=1.5,
            camera_yaw_hint_confidence=0.85,
            local_lane_payload={
                "lane_side_point_counts": {"left": 3, "right": 4},
            },
            frame_trace={
                "debug": {
                    "two_line_direct_error_m": 0.04,
                    "two_line_D_left_cm": 13.5,
                    "two_line_D_right_cm": 21.5,
                }
            },
            candidate=VisualControlCandidate(
                timestamp=10.0,
                steering_deg=6.0,
                speed_cmd=5.0,
                confidence=1.0,
                source="ai_local",
            ),
        )

        lane_observation = observer._build_lane_observation(snapshot)

        self.assertEqual(lane_observation.detected_sides, ("left", "right"))
        self.assertEqual(lane_observation.measurement_mode, "two_line")
        self.assertTrue(lane_observation.direct_error_valid)
        self.assertAlmostEqual(lane_observation.direct_error_m, 0.04, places=4)
        self.assertAlmostEqual(lane_observation.heading_error_rad, 0.12, places=4)
        self.assertAlmostEqual(lane_observation.left_line_distance_m, 0.135, places=4)
        self.assertAlmostEqual(lane_observation.right_line_distance_m, 0.215, places=4)
        self.assertAlmostEqual(lane_observation.line_center_offset_m, 0.04, places=4)
        self.assertEqual(lane_observation.curve_hint, "IN_CURVE")
        self.assertAlmostEqual(lane_observation.camera_yaw_hint_confidence, 0.85, places=4)
        self.assertGreater(lane_observation.quality, 0.9)

    def test_two_line_observation_uses_geometric_center_not_safety_direct_error(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=10.5,
            detection_mode="ai_local",
            local_lane_payload={
                "lane_side_point_counts": {"left": 8, "right": 8},
            },
            frame_trace={
                "debug": {
                    "measurement_mode": "two_line",
                    "two_line_direct_error_m": -0.1242,
                    "two_line_D_left_cm": 25.21,
                    "two_line_D_right_cm": 9.79,
                }
            },
        )

        lane_observation = observer._build_lane_observation(snapshot)

        self.assertEqual(lane_observation.measurement_mode, "two_line")
        self.assertTrue(lane_observation.direct_error_valid)
        self.assertAlmostEqual(lane_observation.line_center_offset_m, -0.0771, places=4)
        self.assertAlmostEqual(lane_observation.direct_error_m, -0.0771, places=4)
        self.assertAlmostEqual(lane_observation.debug["raw_direct_error_m"], -0.1242, places=4)
        self.assertAlmostEqual(
            lane_observation.debug["two_line_direct_error_for_localization_m"],
            -0.0771,
            places=4,
        )

    def test_build_lane_observation_preserves_single_line_physical_direct_error(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=12.0,
            detection_mode="ai_local",
            curve_state="STRAIGHT",
            heading_error_rad=-0.08,
            local_lane_payload={
                "lane_side_point_counts": {"left": 4, "right": 0},
            },
            frame_trace={
                "debug": {
                    "measurement_mode": "single_line",
                    "sl_direct_error_m": -0.0825,
                    "sl_D_left_cm": 25.75,
                    "sl_D_right_cm": 9.25,
                    "control_policy_mode": "ROUTE_TRACKING",
                    "planner_priority_active": True,
                }
            },
        )

        lane_observation = observer._build_lane_observation(snapshot)

        self.assertEqual(lane_observation.detected_sides, ("left",))
        self.assertEqual(lane_observation.measurement_mode, "single_line")
        self.assertTrue(lane_observation.direct_error_valid)
        self.assertAlmostEqual(lane_observation.direct_error_m, -0.0825, places=4)
        self.assertAlmostEqual(lane_observation.left_line_distance_m, 0.2575, places=4)
        self.assertAlmostEqual(lane_observation.right_line_distance_m, 0.0925, places=4)
        self.assertAlmostEqual(lane_observation.line_center_offset_m, -0.0825, places=4)
        self.assertEqual(lane_observation.control_policy_mode, "ROUTE_TRACKING")
        self.assertTrue(lane_observation.planner_priority_active)
        self.assertAlmostEqual(lane_observation.debug["raw_direct_error_m"], -0.0825, places=4)

    def test_single_line_transition_prefers_error_close_to_last_two_line(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        two_line_snapshot = VisualStateSnapshot(
            timestamp=30.0,
            detection_mode="ai_local",
            local_lane_payload={
                "lane_side_point_counts": {"left": 8, "right": 8},
            },
            frame_trace={
                "debug": {
                    "measurement_mode": "two_line",
                    "two_line_direct_error_m": -0.04,
                    "two_line_D_left_cm": 21.5,
                    "two_line_D_right_cm": 13.5,
                }
            },
        )
        single_line_snapshot = VisualStateSnapshot(
            timestamp=30.05,
            detection_mode="ai_local",
            local_lane_payload={
                "lane_side_point_counts": {"left": 8, "right": 0},
            },
            frame_trace={
                "debug": {
                    "measurement_mode": "single_line",
                    "sl_direct_error_m": -0.18,
                    "sl_D_left_cm": 22.0,
                    "sl_D_right_cm": 13.0,
                    "control_policy_mode": "ROUTE_TRACKING",
                    "planner_priority_active": True,
                }
            },
        )

        observer._build_lane_observation(two_line_snapshot)
        lane_observation = observer._build_lane_observation(single_line_snapshot)

        self.assertEqual(lane_observation.previous_measurement_mode, "two_line")
        self.assertTrue(lane_observation.transition_error_coherent)
        self.assertTrue(lane_observation.direct_error_valid)
        self.assertAlmostEqual(lane_observation.transition_reference_error_m, -0.04, places=4)
        self.assertAlmostEqual(lane_observation.direct_error_m, -0.045, places=4)
        self.assertEqual(lane_observation.debug["transition_selected_error_source"], "line_center_offset_m")

    def test_single_line_transition_rejects_error_jump_from_last_two_line(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        two_line_snapshot = VisualStateSnapshot(
            timestamp=40.0,
            detection_mode="ai_local",
            local_lane_payload={
                "lane_side_point_counts": {"left": 8, "right": 8},
            },
            frame_trace={
                "debug": {
                    "measurement_mode": "two_line",
                    "two_line_direct_error_m": -0.04,
                    "two_line_D_left_cm": 21.5,
                    "two_line_D_right_cm": 13.5,
                }
            },
        )
        single_line_snapshot = VisualStateSnapshot(
            timestamp=40.05,
            detection_mode="ai_local",
            local_lane_payload={
                "lane_side_point_counts": {"left": 8, "right": 0},
            },
            frame_trace={
                "debug": {
                    "measurement_mode": "single_line",
                    "sl_direct_error_m": -0.18,
                    "sl_D_left_cm": 33.5,
                    "sl_D_right_cm": 1.5,
                    "control_policy_mode": "ROUTE_TRACKING",
                    "planner_priority_active": True,
                }
            },
        )

        observer._build_lane_observation(two_line_snapshot)
        lane_observation = observer._build_lane_observation(single_line_snapshot)

        self.assertFalse(lane_observation.transition_error_coherent)
        self.assertFalse(lane_observation.direct_error_valid)
        self.assertIsNone(lane_observation.direct_error_m)
        self.assertLessEqual(lane_observation.quality, 0.45)
        self.assertEqual(
            lane_observation.debug["single_line_transition_reject_reason"],
            "error_jump_from_last_two_line",
        )

    def test_build_lane_observation_rejects_impossible_line_geometry(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=12.5,
            detection_mode="ai_local",
            curve_state="IN_CURVE",
            local_lane_payload={
                "lane_side_point_counts": {"left": 8, "right": 8},
            },
            frame_trace={
                "debug": {
                    "measurement_mode": "two_line",
                    "two_line_direct_error_m": -1.62,
                    "two_line_D_left_cm": 100.0,
                    "two_line_D_right_cm": -65.0,
                },
                "visual_lane_waypoints": {
                    "center_waypoints_body": tuple((0.05 * idx, 0.10, 0.0) for idx in range(20)),
                    "lane_width_m": 0.35,
                },
            },
        )

        lane_observation = observer._build_lane_observation(snapshot)

        self.assertEqual(lane_observation.measurement_mode, "two_line")
        self.assertFalse(lane_observation.direct_error_valid)
        self.assertIsNone(lane_observation.direct_error_m)
        self.assertIsNone(lane_observation.left_line_distance_m)
        self.assertIsNone(lane_observation.right_line_distance_m)
        self.assertIsNone(lane_observation.line_center_offset_m)
        self.assertLessEqual(lane_observation.quality, 0.2)
        self.assertFalse(lane_observation.debug["line_geometry_valid"])
        self.assertAlmostEqual(lane_observation.debug["raw_left_line_distance_m"], 1.0, places=4)

    def test_build_lane_observation_rejects_two_line_boundary_crossing(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=12.55,
            detection_mode="ai_local",
            curve_state="IN_CURVE",
            local_lane_payload={
                "lane_side_point_counts": {"left": 12, "right": 12},
            },
            frame_trace={
                "debug": {
                    "measurement_mode": "two_line",
                    "two_line_direct_error_m": -0.237,
                    "two_line_D_left_cm": 41.2,
                    "two_line_D_right_cm": -6.2,
                },
                "visual_lane_waypoints": {
                    "center_waypoints_body": tuple((0.05 * idx, -0.18, 0.0) for idx in range(20)),
                    "lane_width_m": 0.35,
                },
            },
        )

        lane_observation = observer._build_lane_observation(snapshot)

        self.assertEqual(lane_observation.measurement_mode, "two_line")
        self.assertFalse(lane_observation.direct_error_valid)
        self.assertIsNone(lane_observation.direct_error_m)
        self.assertIsNone(lane_observation.line_center_offset_m)
        self.assertLessEqual(lane_observation.quality, 0.2)
        self.assertFalse(lane_observation.debug["line_geometry_valid"])
        self.assertAlmostEqual(lane_observation.debug["raw_right_line_distance_m"], -0.062, places=4)

    def test_build_lane_observation_respects_explicit_invalid_direct_error(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=12.6,
            detection_mode="ai_local",
            local_lane_payload={
                "lane_side_point_counts": {"left": 8, "right": 8},
            },
            frame_trace={
                "debug": {
                    "measurement_mode": "two_line",
                    "direct_error_valid": False,
                },
                "visual_lane_waypoints": {
                    "center_waypoints_body": tuple((0.05 * idx, 0.04, 0.0) for idx in range(20)),
                    "lane_width_m": 0.35,
                },
            },
        )

        lane_observation = observer._build_lane_observation(snapshot)

        self.assertFalse(lane_observation.direct_error_valid)
        self.assertIsNone(lane_observation.direct_error_m)
        self.assertGreaterEqual(lane_observation.quality, 0.85)

    def test_build_lane_observation_prefers_resolved_visible_side_over_raw_payload_counts(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=13.0,
            detection_mode="ai_local",
            curve_state="IN_CURVE",
            local_lane_payload={
                "lane_side_point_counts": {"left": 13, "right": 0},
            },
            frame_trace={
                "lane_observation": {
                    "visible_side": "right",
                    "sides": ["right"],
                },
                "debug": {
                    "measurement_mode": "single_line",
                    "sl_direct_error_m": -0.095,
                    "control_policy_mode": "ROUTE_TRACKING",
                    "planner_priority_active": True,
                },
            },
        )

        lane_observation = observer._build_lane_observation(snapshot)

        self.assertEqual(lane_observation.detected_sides, ("right",))
        self.assertEqual(lane_observation.measurement_mode, "single_line")
        self.assertTrue(lane_observation.direct_error_valid)
        self.assertAlmostEqual(lane_observation.direct_error_m, -0.095, places=4)

    def test_build_lane_observation_falls_back_to_screen_position_not_raw_side_label(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=14.0,
            detection_mode="ai_local",
            curve_state="IN_CURVE",
            local_lane_payload={
                "frame_width": 640,
                "lane_side_point_counts": {"left": 13, "right": 0},
                "lane_side_lines": {
                    "left": [432, 354, 262, 171],
                    "right": [],
                },
            },
            frame_trace={
                "debug": {
                    "measurement_mode": "single_line",
                    "sl_direct_error_m": -0.095,
                    "control_policy_mode": "ROUTE_TRACKING",
                },
            },
        )

        lane_observation = observer._build_lane_observation(snapshot)

        self.assertEqual(lane_observation.detected_sides, ("right",))
        self.assertTrue(lane_observation.direct_error_valid)
        self.assertAlmostEqual(lane_observation.direct_error_m, -0.095, places=4)

    def test_build_lane_observation_falls_back_to_single_line_physical_mask_error(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=15.0,
            detection_mode="ai_local",
            curve_state="IN_CURVE",
            local_lane_payload={
                "frame_width": 640,
                "lane_side_point_counts": {"left": 0, "right": 14},
                "lane_side_lines": {
                    "left": [],
                    "right": [406, 360, 326, 158],
                },
            },
            frame_trace={
                "lane_observation": {
                    "visible_side": "right",
                    "sides": ["right"],
                },
                "debug": {
                    "measurement_mode": "single_line",
                    "control_policy_mode": "ROUTE_TRACKING",
                    "local_mask_guidance": {
                        "guidance_mode": "single_line_physical",
                        "error_cm": -10.3,
                    },
                },
            },
        )

        lane_observation = observer._build_lane_observation(snapshot)

        self.assertEqual(lane_observation.detected_sides, ("right",))
        self.assertTrue(lane_observation.direct_error_valid)
        self.assertAlmostEqual(lane_observation.direct_error_m, -0.103, places=4)

    def test_single_line_side_flip_without_two_line_confirmation_drops_visual_path(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        left_snapshot = VisualStateSnapshot(
            timestamp=16.0,
            detection_mode="ai_local",
            local_lane_payload={"lane_side_point_counts": {"left": 12, "right": 0}},
            frame_trace={
                "debug": {
                    "measurement_mode": "single_line",
                    "single_line_resolved_side": "left",
                    "sl_direct_error_m": 0.02,
                },
                "visual_lane_waypoints": {
                    "center_waypoints_body": tuple((0.05 * idx, 0.01, 0.0) for idx in range(20)),
                    "lane_width_m": 0.35,
                    "extrapolated_side": "right",
                },
            },
        )
        right_snapshot = VisualStateSnapshot(
            timestamp=16.05,
            detection_mode="ai_local",
            local_lane_payload={"lane_side_point_counts": {"left": 0, "right": 12}},
            frame_trace={
                "debug": {
                    "measurement_mode": "single_line",
                    "single_line_resolved_side": "right",
                    "sl_direct_error_m": 0.015,
                },
                "visual_lane_waypoints": {
                    "center_waypoints_body": tuple((0.05 * idx, -0.01, 0.0) for idx in range(20)),
                    "lane_width_m": 0.35,
                    "extrapolated_side": "left",
                },
            },
        )

        observer._build_lane_observation(left_snapshot)
        lane_observation = observer._build_lane_observation(right_snapshot)

        self.assertEqual(lane_observation.measurement_mode, "single_line")
        self.assertEqual(lane_observation.detected_sides, ("right",))
        self.assertFalse(lane_observation.direct_error_valid)
        self.assertEqual(len(lane_observation.center_waypoints_body), 0)
        self.assertLessEqual(lane_observation.quality, 0.35)
        self.assertEqual(
            lane_observation.debug["single_line_side_reject_reason"],
            "side_flip_without_two_line_confirmation",
        )

    def test_single_line_waypoint_geometry_outlier_drops_visual_path(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=17.0,
            detection_mode="ai_local",
            local_lane_payload={"lane_side_point_counts": {"left": 0, "right": 12}},
            frame_trace={
                "debug": {
                    "measurement_mode": "single_line",
                    "single_line_resolved_side": "right",
                    "sl_direct_error_m": 0.02,
                },
                "visual_lane_waypoints": {
                    "center_waypoints_body": tuple((0.05 * idx, 0.34, 0.0) for idx in range(20)),
                    "lane_width_m": 0.35,
                    "extrapolated_side": "left",
                },
            },
        )

        lane_observation = observer._build_lane_observation(snapshot)

        self.assertEqual(lane_observation.measurement_mode, "single_line")
        self.assertEqual(len(lane_observation.center_waypoints_body), 0)
        self.assertLessEqual(lane_observation.quality, 0.35)
        self.assertEqual(
            lane_observation.debug["visual_waypoint_reject_reason"],
            "single_line_waypoint_lateral_out_of_lane",
        )

    def test_build_stopline_observation_keeps_pass_event_payload(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=20.0,
            stopline_debug={
                "stopline_visible_candidate": True,
                "stopline_stable_visible": True,
                "stopline_distance_m": 0.32,
                "stopline_confidence": 0.77,
                "stopline_expected_node_id": "node-17",
                "stopline_expected_node_attr": 7,
                "stopline_source": "opencv_bev",
                "stopline_pass_event_payload": {
                    "distance_m": 0.28,
                    "triggered_by": "missing_streak",
                },
            },
        )

        stopline_observation = observer._build_stopline_observation(snapshot)

        self.assertTrue(stopline_observation.visible)
        self.assertTrue(stopline_observation.stable)
        self.assertAlmostEqual(stopline_observation.distance_m, 0.32, places=4)
        self.assertEqual(stopline_observation.expected_node_id, "node-17")
        self.assertEqual(stopline_observation.expected_node_attr, 7)
        self.assertEqual(stopline_observation.pass_event["triggered_by"], "missing_streak")


if __name__ == "__main__":
    unittest.main()
