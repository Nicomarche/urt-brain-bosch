import math
import time
import types
import unittest

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

    def test_current_steer_feedback_is_preferred_over_command(self):
        tracker = threadTracking.__new__(threadTracking)
        tracker._steer_feedback_sub = _FakeSub(["250"])
        tracker._steer_sub = _FakeSub(["100"])
        tracker._last_steer_feedback_rad = 0.0
        tracker._last_steer_feedback_t = None
        tracker._last_steer_rad = 0.0

        steer_rad = tracker._resolve_steer_rad(now=5.0)

        self.assertAlmostEqual(steer_rad, math.radians(25.0), places=6)

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
