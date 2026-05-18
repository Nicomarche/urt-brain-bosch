import math
import time
import types
import unittest
from collections import deque

from src.control.motor_command_dispatcher import threadMotorCommandDispatcher
from src.localization.dead_reckoning import DeadReckoning
from src.localization.relocalization_thread import threadTracking


class _FakeDR:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)

    def correct_lateral(self, lateral_error_m: float, path_psi: float) -> None:
        dx = float(lateral_error_m) * math.cos(float(path_psi) + math.pi / 2.0)
        dy = float(lateral_error_m) * math.sin(float(path_psi) + math.pi / 2.0)
        self.x -= dx
        self.y -= dy

    def reset(self, x: float, y: float, yaw: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)

    def get_state(self):
        return self.x, self.y, self.yaw


class _FakeSub:
    def __init__(self, values):
        self._values = list(values)

    def receive(self):
        if not self._values:
            return None
        return self._values.pop(0)


class TrackingRelocalizationTests(unittest.TestCase):
    def test_speed_command_fallback_is_used_when_encoder_feedback_is_missing(self):
        tracker = threadTracking.__new__(threadTracking)
        tracker._speed_sub = _FakeSub([None])
        tracker._speed_cmd_sub = _FakeSub(["108"])
        tracker._last_raw_speed = None
        tracker._last_speed_t = None
        tracker._last_speed = 0.0
        tracker._last_speed_source = "none"
        tracker._last_cmd_speed_raw = 0.0
        tracker._last_cmd_speed_t = None

        speed_mps = tracker._resolve_speed_mps(now=10.0)

        self.assertAlmostEqual(speed_mps, 0.108, places=4)
        self.assertEqual(tracker._last_speed_source, "command")

    def test_encoder_outlier_is_replaced_by_recent_command(self):
        tracker = threadTracking.__new__(threadTracking)
        tracker._speed_sub = _FakeSub(["500"])  # 0.50 m/s encoder spike
        tracker._speed_cmd_sub = _FakeSub(["200"])  # 0.20 m/s command
        tracker._last_raw_speed = None
        tracker._last_speed_t = None
        tracker._last_speed = 0.0
        tracker._last_speed_source = "none"
        tracker._last_cmd_speed_raw = 200.0
        tracker._last_cmd_speed_t = 9.95
        tracker._cmd_speed_history = deque([0.20] * 5, maxlen=5)
        tracker._current_state_message = "AUTO"

        speed_mps = tracker._resolve_speed_mps(now=10.0)

        self.assertAlmostEqual(speed_mps, 0.20, places=4)
        self.assertEqual(tracker._last_speed_source, "encoder_filtered_command")

    def test_encoder_near_zero_is_clamped_to_zero(self):
        tracker = threadTracking.__new__(threadTracking)
        tracker._speed_sub = _FakeSub(["10"])  # 0.01 m/s noise
        tracker._speed_cmd_sub = _FakeSub([None])
        tracker._last_raw_speed = None
        tracker._last_speed_t = None
        tracker._last_speed = 0.0
        tracker._last_speed_source = "none"
        tracker._last_cmd_speed_raw = 0.0
        tracker._last_cmd_speed_t = None
        tracker._cmd_speed_history = deque([0.0] * 5, maxlen=5)
        tracker._current_state_message = "AUTO"

        speed_mps = tracker._resolve_speed_mps(now=10.0)

        self.assertAlmostEqual(speed_mps, 0.0, places=4)
        self.assertEqual(tracker._last_speed_source, "encoder_zero")

    def test_auto_uses_command_when_encoder_zero_but_cmd_moving(self):
        """Post-mortem run_20260518_030858: en sim el encoder de Gazebo
        reporta ~0 el 87% del tiempo aunque el robot se mueve. El DR
        quedaba estancado. En AUTO con encoder_zero + cmd>=0.05 m/s, el
        comando del MPC debe sustituir al encoder.
        """
        tracker = threadTracking.__new__(threadTracking)
        tracker._speed_sub = _FakeSub(["5"])  # 0.005 m/s — encoder zero clamp
        tracker._speed_cmd_sub = _FakeSub(["300"])  # 0.30 m/s comando MPC
        tracker._last_raw_speed = None
        tracker._last_speed_t = None
        tracker._last_speed = 0.0
        tracker._last_speed_source = "none"
        tracker._last_cmd_speed_raw = 0.0
        tracker._last_cmd_speed_t = None
        tracker._cmd_speed_history = deque([0.30] * 5, maxlen=5)
        tracker._current_state_message = "AUTO"

        speed_mps = tracker._resolve_speed_mps(now=10.0)

        # El encoder clampea a 0 (encoder_zero), pero el AUTO command
        # fallback toma el valor del cmd (0.30 m/s).
        self.assertAlmostEqual(speed_mps, 0.30, places=4)
        self.assertEqual(tracker._last_speed_source, "auto_command_hold")

    def test_auto_keeps_encoder_zero_when_cmd_is_below_threshold(self):
        """Si el cmd del MPC es bajo (stop progresivo), no sobrescribir el
        encoder_zero — el robot realmente se está deteniendo.
        """
        tracker = threadTracking.__new__(threadTracking)
        tracker._speed_sub = _FakeSub(["5"])  # encoder zero
        tracker._speed_cmd_sub = _FakeSub(["30"])  # 0.03 m/s — bajo umbral
        tracker._last_raw_speed = None
        tracker._last_speed_t = None
        tracker._last_speed = 0.0
        tracker._last_speed_source = "none"
        tracker._last_cmd_speed_raw = 0.0
        tracker._last_cmd_speed_t = None
        tracker._cmd_speed_history = deque([0.03] * 5, maxlen=5)
        tracker._current_state_message = "AUTO"

        speed_mps = tracker._resolve_speed_mps(now=10.0)

        # cmd 0.03 < _CMD_FALLBACK_AUTO_MIN_MPS (0.05) → mantiene encoder_zero.
        self.assertAlmostEqual(speed_mps, 0.0, places=4)
        self.assertEqual(tracker._last_speed_source, "encoder_zero")

    def test_manual_path_still_uses_manual_command_hold(self):
        """El fallback AUTO no debe robar el caso MANUAL. En MANUAL con
        encoder zero, debe seguir saliendo manual_command_hold (no
        auto_command_hold).
        """
        tracker = threadTracking.__new__(threadTracking)
        tracker._speed_sub = _FakeSub(["5"])
        tracker._speed_cmd_sub = _FakeSub(["300"])
        tracker._last_raw_speed = None
        tracker._last_speed_t = None
        tracker._last_speed = 0.0
        tracker._last_speed_source = "none"
        tracker._last_cmd_speed_raw = 0.0
        tracker._last_cmd_speed_t = None
        tracker._cmd_speed_history = deque([0.30] * 5, maxlen=5)
        tracker._current_state_message = "MANUAL"

        tracker._resolve_speed_mps(now=10.0)
        self.assertEqual(tracker._last_speed_source, "manual_command_hold")

    def test_current_steer_feedback_is_preferred_over_command(self):
        tracker = threadTracking.__new__(threadTracking)
        tracker._steer_feedback_sub = _FakeSub(["250"])
        tracker._steer_sub = _FakeSub(["100"])
        tracker._last_steer_feedback_rad = 0.0
        tracker._last_steer_feedback_t = None
        tracker._last_steer_rad = 0.0
        tracker._last_steer_feedback_raw = None
        tracker._last_steer_cmd_raw = None
        tracker._last_logged_steer_snapshot = None

        steer_rad = tracker._resolve_steer_rad(now=5.0)

        # CurrentSteer raw sigue el wire legacy (+ = derecha); internamente
        # localization debe usar convención matemática (+ = izquierda).
        self.assertAlmostEqual(steer_rad, -math.radians(25.0), places=6)

    def test_steering_round_trip_matches_mathematical_convention(self):
        for steering_deg in (10.0, -10.0):
            with self.subTest(steering_deg=steering_deg):
                wire_x10 = threadMotorCommandDispatcher._steer_deg_to_wire_x10(steering_deg)
                parsed = threadTracking._parse_steer_rad(str(wire_x10))
                self.assertIsNotNone(parsed)
                self.assertAlmostEqual(parsed, math.radians(steering_deg), places=6)

    def test_dead_reckoning_positive_left_steer_decreases_osm_yaw(self):
        dr = DeadReckoning(0.0, 0.0, 0.0)

        dr.update(
            speed_mps=0.20,
            yaw_rad=0.0,
            dt=0.10,
            steer_rad=math.radians(12.0),
            wheelbase_m=0.26,
        )

        x, y, yaw = dr.get_state()
        self.assertGreater(x, 0.0)
        self.assertLess(y, 0.0)
        self.assertLess(yaw, 0.0)

    def test_dead_reckoning_measured_right_yaw_increases_osm_yaw(self):
        dr = DeadReckoning(0.0, 0.0, 0.0)

        dr.update_from_yaw(
            speed_mps=0.20,
            yaw_start_rad=0.0,
            yaw_end_rad=math.radians(12.0),
            dt=0.10,
        )

        x, y, yaw = dr.get_state()
        self.assertGreater(x, 0.0)
        self.assertGreater(y, 0.0)
        self.assertGreater(yaw, 0.0)

    def test_dead_reckoning_beta_moves_cg_before_body_yaw_changes(self):
        dr = DeadReckoning(0.0, 0.0, 0.0)

        dr.update_from_yaw(
            speed_mps=0.20,
            yaw_start_rad=0.0,
            yaw_end_rad=0.0,
            dt=0.10,
            steer_rad=math.radians(25.0),
            wheelbase_m=0.258,
            rear_axle_to_cg_m=0.103,
        )

        x, y, yaw = dr.get_state()
        self.assertGreater(x, 0.0)
        self.assertLess(y, 0.0)
        self.assertAlmostEqual(yaw, 0.0, places=9)

    def _make_lane_reloc_tracker(self, dr_y=0.10, speed=0.20):
        tracker = threadTracking.__new__(threadTracking)
        tracker._dr = _FakeDR(x=0.0, y=dr_y, yaw=0.0)
        tracker._last_speed = speed
        tracker._last_lane_visual_reloc_t = 0.0
        tracker.tracking_state = types.SimpleNamespace(
            lane_measurement_reliable=True,
            raw_lateral_error_m=0.0,
            set_lane_measurement_state=lambda reliable, applied_correction_m=0.0: None,
        )
        return tracker

    def _make_lane_reloc_path_update(self, waypoint_mode_active=False):
        return types.SimpleNamespace(
            matched_x=0.0,
            matched_y=0.0,
            matched_yaw=0.0,
            waypoint_mode_active=waypoint_mode_active,
        )

    def test_lane_visual_relocalization_recenters_dead_reckoning(self):
        tracker = self._make_lane_reloc_tracker(dr_y=0.10)
        path_update = self._make_lane_reloc_path_update()

        raw_x, raw_y, raw_yaw, correction_m, raw_lat_err = tracker._apply_lane_visual_relocalization(
            0.0, 0.10, 0.0, path_update, now=1000.0,
        )

        self.assertGreater(correction_m, 0.0)
        self.assertAlmostEqual(raw_lat_err, 0.10, places=4)
        self.assertAlmostEqual(raw_x, 0.0, places=4)
        self.assertLess(raw_y, 0.10)
        self.assertAlmostEqual(raw_yaw, 0.0, places=4)

    def test_lane_visual_relocalization_skips_large_raw_error(self):
        tracker = self._make_lane_reloc_tracker(dr_y=0.40)
        path_update = self._make_lane_reloc_path_update()

        raw_x, raw_y, raw_yaw, correction_m, raw_lat_err = tracker._apply_lane_visual_relocalization(
            0.0, 0.40, 0.0, path_update, now=1000.0,
        )

        self.assertEqual(correction_m, 0.0)
        self.assertAlmostEqual(raw_lat_err, 0.40, places=4)
        self.assertAlmostEqual(raw_x, 0.0, places=4)
        self.assertAlmostEqual(raw_y, 0.40, places=4)
        self.assertAlmostEqual(raw_yaw, 0.0, places=4)

    def test_lane_visual_relocalization_skips_in_precision_zone(self):
        tracker = self._make_lane_reloc_tracker(dr_y=0.10)
        path_update = self._make_lane_reloc_path_update(waypoint_mode_active=True)

        _, _, _, correction_m, _ = tracker._apply_lane_visual_relocalization(
            0.0, 0.10, 0.0, path_update, now=1000.0,
        )

        self.assertEqual(correction_m, 0.0)

    def test_lane_visual_relocalization_skips_when_stopped(self):
        tracker = self._make_lane_reloc_tracker(dr_y=0.10, speed=0.01)
        path_update = self._make_lane_reloc_path_update()

        _, _, _, correction_m, _ = tracker._apply_lane_visual_relocalization(
            0.0, 0.10, 0.0, path_update, now=1000.0,
        )

        self.assertEqual(correction_m, 0.0)

    def test_lane_visual_relocalization_respects_cooldown(self):
        tracker = self._make_lane_reloc_tracker(dr_y=0.10)
        tracker._last_lane_visual_reloc_t = 999.95  # 50 ms ago
        path_update = self._make_lane_reloc_path_update()

        _, _, _, correction_m, _ = tracker._apply_lane_visual_relocalization(
            0.0, 0.10, 0.0, path_update, now=1000.0,
        )

        self.assertEqual(correction_m, 0.0)

    def test_semantic_relocalization_resets_pose_when_event_matches(self):
        tracker = threadTracking.__new__(threadTracking)
        tracker._dr = _FakeDR(x=0.2, y=0.2, yaw=0.0)
        tracker._last_semantic_relocalization_t = 0.0
        now = time.monotonic()
        tracker._last_sign_observation = {
            "sign": "stop",
            "observed_at_monotonic": now,
            "distance_m": 0.22,
        }

        path_update = types.SimpleNamespace(
            expected_control_type="stop",
            next_semantic_type="intersection",
            next_semantic_distance_m=0.20,
            map_match_error_m=0.08,
            matched_x=1.2,
            matched_y=0.3,
            matched_yaw=0.4,
        )

        applied, semantic_match = tracker._apply_semantic_relocalization(path_update, now=now + 0.2)

        self.assertTrue(applied)
        self.assertEqual(semantic_match[0], "sign:stop")
        self.assertAlmostEqual(tracker._dr.x, 1.2, places=4)
        self.assertAlmostEqual(tracker._dr.y, 0.3, places=4)
        self.assertAlmostEqual(tracker._dr.yaw, 0.4, places=4)


if __name__ == "__main__":
    unittest.main()
