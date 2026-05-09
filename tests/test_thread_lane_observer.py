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
        self.assertEqual(lane_observation.curve_hint, "IN_CURVE")
        self.assertAlmostEqual(lane_observation.camera_yaw_hint_confidence, 0.85, places=4)
        self.assertGreater(lane_observation.quality, 0.9)

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
        self.assertEqual(lane_observation.control_policy_mode, "ROUTE_TRACKING")
        self.assertTrue(lane_observation.planner_priority_active)
        self.assertAlmostEqual(lane_observation.debug["raw_direct_error_m"], -0.0825, places=4)

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
