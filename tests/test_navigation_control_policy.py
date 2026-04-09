import types
import unittest

from src.hardware.camera.threads.maneuverManager import ManeuverManager
from src.hardware.camera.threads.threadLineFollowing import threadLineFollowing


class NavigationControlPolicyTests(unittest.TestCase):
    def test_blind_control_prefers_route_tracking_when_route_active(self):
        controller = threadLineFollowing.__new__(threadLineFollowing)
        controller._tracking_state = types.SimpleNamespace(
            initialized=True,
            route_active=True,
            waypoint_mode_active=True,
            maneuver_type="turn_left",
            current_node_attr=2,
            upcoming_node_attr=2,
            heading_rad=0.12,
            error_m=0.03,
        )
        controller._current_speed = 0.0
        controller.min_speed = 5.0
        controller.max_steering = 25.0
        controller.frames_without_line = 1
        controller.no_lane_hold_steering_frames = 4
        controller._compute_lateral_control = lambda *args, **kwargs: 7.5
        controller._update_progressive_speed = lambda steering_angle, speed_cap=None: 4.5
        controller._get_navigation_context = threadLineFollowing._get_navigation_context.__get__(
            controller, threadLineFollowing
        )

        steer, speed, mode = threadLineFollowing._resolve_ai_local_blind_control(
            controller,
            img_w=640,
            speed_cap=6.0,
        )

        self.assertEqual(steer, 7.5)
        self.assertEqual(speed, 4.5)
        self.assertEqual(mode, "route_tracking")

    def test_blind_control_uses_visual_hold_when_no_route_is_active(self):
        controller = threadLineFollowing.__new__(threadLineFollowing)
        controller._tracking_state = types.SimpleNamespace(
            initialized=False,
            route_active=False,
            waypoint_mode_active=False,
            maneuver_type="none",
            current_node_attr=0,
            upcoming_node_attr=0,
        )
        controller.min_speed = 5.0
        controller.max_steering = 25.0
        controller.frames_without_line = 1
        controller.no_lane_hold_steering_frames = 4
        controller._last_good_steering = 3.0
        controller.last_steering = 3.0
        controller._get_navigation_context = threadLineFollowing._get_navigation_context.__get__(
            controller, threadLineFollowing
        )

        steer, speed, mode = threadLineFollowing._resolve_ai_local_blind_control(
            controller,
            img_w=640,
            speed_cap=4.0,
        )

        self.assertEqual(steer, 3.0)
        self.assertEqual(speed, 4.0)
        self.assertEqual(mode, "visual_fallback")

    def test_maneuver_manager_filters_out_of_context_traffic_signs(self):
        manager = ManeuverManager()
        tracking_state = types.SimpleNamespace(
            route_active=True,
            next_semantic_type="intersection",
            expected_control_type="traffic_light",
            next_semantic_distance_m=0.3,
            current_zone_types=[],
        )

        manager.observe_sign({"sign": "stop", "timestamp": 1.0}, tracking_state=tracking_state)
        decision = manager.decide(1.1, tracking_state=tracking_state)

        self.assertNotEqual(decision.active_maneuver, "stop")
        self.assertEqual(decision.mission_state, "APPROACH_INTERSECTION")


if __name__ == "__main__":
    unittest.main()
