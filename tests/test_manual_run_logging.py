import json
import os
import tempfile
import unittest
from collections import deque

from src.hardware.camera.threads.threadLineFollowing import threadLineFollowing
from src.localization.relocalization_thread import TrackingState


class ManualRunLoggingTests(unittest.TestCase):
    def _make_tracking_state(self):
        state = TrackingState()
        state.update(
            x=1.23,
            y=2.34,
            yaw=0.45,
            error_m=0.08,
            heading_rad=0.12,
            path_psi=0.40,
            path_kappa=0.018,
            speed_mps=0.56,
            wp_idx=6,
            waypoint_mode=True,
            node_attr=7,
            imu_received=True,
            steer_rad=0.07,
            target_idx=8,
            raw_x=1.19,
            raw_y=2.29,
            raw_yaw=0.43,
            matched_x=1.24,
            matched_y=2.35,
            matched_yaw=0.46,
            map_match_error_m=0.03,
            route_active=True,
            route_id="reference-1",
            current_node_id="46",
            current_node_attr=7,
            upcoming_node_id="47",
            upcoming_node_attr=2,
            maneuver_type="stopline",
            destination_node_id="96",
            route_progress=0.12345,
            route_points=[{"x": 0.5, "y": 2.3}, {"x": 0.7, "y": 2.6}],
            route_completed=False,
            route_replans=1,
            route_source="reference",
            destination_point={"x": 3.2, "y": 4.1},
            destination_label="finish",
            route_queue=[{"node_id": "96", "label": "finish"}],
            next_semantic_id="intersection_1",
            next_semantic_type="intersection",
            next_semantic_label="Cruce principal",
            next_semantic_distance_m=0.42,
            expected_control_type="stop",
            current_zone_ids=["zone_1"],
            current_zone_types=["parking"],
            map_metadata={"meters_per_pixel": 0.01, "pixels_per_meter": 100.0},
            available_destinations=[{"id": "finish", "node_id": "96"}],
            relocalization_mode="semantic",
            last_relocalization_source="lane_relocalization",
            last_relocalization_error_m=0.012,
            raw_lateral_error_m=0.071,
        )
        state.last_yaw_correction_deg = 1.25
        state.last_cam_yaw_hint_deg = 2.5
        state.last_cam_yaw_hint_conf = 0.91
        return state

    def _make_controller(self, log_path):
        controller = threadLineFollowing.__new__(threadLineFollowing)
        controller._sanitize_log_value = threadLineFollowing._sanitize_log_value.__get__(
            controller, threadLineFollowing
        )
        controller._build_local_lane_payload_log = threadLineFollowing._build_local_lane_payload_log.__get__(
            controller, threadLineFollowing
        )
        controller._build_tracking_log_snapshot = threadLineFollowing._build_tracking_log_snapshot.__get__(
            controller, threadLineFollowing
        )
        controller._reset_manual_run_log = threadLineFollowing._reset_manual_run_log.__get__(
            controller, threadLineFollowing
        )
        controller._log_manual_run_frame = threadLineFollowing._log_manual_run_frame.__get__(
            controller, threadLineFollowing
        )

        controller.manual_run_log_path = log_path
        controller._manual_run_log_enabled = False
        controller._manual_run_log_frame_idx = 0
        controller._tracking_state = self._make_tracking_state()
        controller.detection_mode = "AI_LOCAL"
        controller.is_line_following_active = False
        controller._last_state_change_message = "MANUAL"
        controller._last_yaw = 90.0
        controller._last_imu_roll = 1.2
        controller._last_imu_pitch = -0.4
        controller._yaw_rate = 0.05
        controller._measured_speed = 123.0
        controller._measured_steer = -45.0
        controller._last_cmd_speed = 110.0
        controller._last_cmd_steer = -30.0
        controller._mission_state_name = "LANE_FOLLOW"
        controller._control_policy_mode = "manual"
        controller._control_authority = "dashboard"
        controller._active_maneuver_name = "none"
        controller._planner_priority_active = True
        controller._safety_stop_reason = None
        controller._last_maneuver_notes = "manual replay"
        controller._last_requested_motor_command = {"speed_x10": 110, "steer_x10": -30}
        controller._last_actuator_status = {"serial_connected": True, "engine_enabled": True}
        controller._last_local_perception_status = {"model_ready": True, "inference_time_ms": 88.0}
        controller._last_local_lane_payload = {
            "frame_id": 5,
            "timestamp": 100.0,
            "lane_count": 2,
            "model_ready": True,
            "heading_hint_rad": 0.21,
            "heading_hint_confidence": 0.8,
            "heading_hint_source": "both_lines",
            "road_type_class": "straight",
            "road_type_confidence": 0.9,
            "lead_distance_class": "near",
            "lead_distance_confidence": 0.7,
            "lead_distance_area": 1234.0,
            "lane_side_sources": {"left": "mask", "right": "mask"},
            "lane_side_points": {"left": [(1, 2)], "right": [(3, 4), (5, 6)]},
            "lane_side_lines": {"left": [0, 0, 1, 1], "right": [2, 2, 3, 3]},
        }
        controller._lane_observation_history = deque(
            [{"visible_sides": ["left", "right"], "confidence": 0.88}],
            maxlen=12,
        )
        controller._last_frame_trace = {"mode": "manual", "reason": "test"}
        controller._heading_error = 0.11
        controller.stanley_measured_speed_to_mps = 0.001
        controller._measured_steer_delta = 2.0
        controller._stanley_last_speed_mps = 0.123
        controller._stanley_last_speed_source = "measured"
        controller._stanley_last_error_m = 0.07
        controller._stanley_last_px_per_cm = 1.5
        controller._theta_dmae_deg = 1.25
        controller._offset_dma_m = 0.02
        controller._local_ai_heading_hint_rad = 0.2
        controller._local_ai_heading_hint_confidence = 0.85
        controller._local_ai_heading_hint_source = "both_lines"
        controller._local_ai_road_type_class = "straight"
        controller._local_ai_road_type_confidence = 0.92
        controller._local_ai_lead_distance_class = "near"
        controller._local_ai_lead_distance_confidence = 0.61
        controller._curve_state = "ENTERING"
        controller._curve_direction = 1
        controller.frames_without_line = 0
        controller.last_seen_side = "left"
        return controller

    def test_manual_run_log_captures_tracking_route_and_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "line_following_manual_last_run.txt")
            controller = self._make_controller(log_path)

            controller._reset_manual_run_log("MANUAL")
            controller._log_manual_run_frame()

            with open(log_path, "r", encoding="utf-8") as log_file:
                content = log_file.read()

        header_json = content.split("=== MANUAL RUN RESET ===\n", 1)[1].split("\n\n", 1)[0]
        frame_json = content.split("=== FRAME 0001 ===\n", 1)[1].split("\n\n", 1)[0]
        header = json.loads(header_json)
        frame = json.loads(frame_json)

        self.assertEqual(header["tracking_reference"]["route_reference"]["map_metadata"]["meters_per_pixel"], 0.01)
        self.assertEqual(header["tracking_reference"]["route_reference"]["route_points_count"], 2)
        self.assertEqual(frame["tracking"]["navigation"]["current_node_id"], "46")
        self.assertEqual(frame["tracking"]["navigation"]["upcoming_node_id"], "47")
        self.assertEqual(frame["tracking"]["semantics"]["next_semantic_id"], "intersection_1")
        self.assertEqual(frame["tracking"]["semantics"]["expected_control_type"], "stop")
        self.assertEqual(frame["tracking"]["relocalization"]["mode"], "semantic")
        self.assertAlmostEqual(frame["tracking"]["pose"]["raw_x_m"], 1.19, places=4)
        self.assertAlmostEqual(frame["tracking"]["dead_reckoning"]["dr_yaw_correction_deg"], 1.25, places=3)
        self.assertEqual(frame["local_lane_payload"]["lane_side_point_counts"]["right"], 2)


if __name__ == "__main__":
    unittest.main()
