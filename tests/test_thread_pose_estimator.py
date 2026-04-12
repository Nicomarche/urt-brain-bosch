import math
import time
import types
import unittest

from src.hardware.pipeline.sharedTypes import LaneObservation, Pose2D, RouteContext
from src.hardware.tracking.threadPoseEstimator import threadPoseEstimator


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

    def correct_yaw(self, yaw_correction_rad: float) -> None:
        self.yaw += float(yaw_correction_rad)

    def reset(self, x: float, y: float, yaw: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)

    def get_state(self):
        return self.x, self.y, self.yaw


class PoseEstimatorTests(unittest.TestCase):
    def test_apply_lane_observation_corrects_dead_reckoning(self):
        estimator = threadPoseEstimator.__new__(threadPoseEstimator)
        estimator._dr = _FakeDR(x=0.0, y=0.08, yaw=0.0)
        estimator._last_speed = 0.20
        estimator._last_camera_lateral_correction_monotonic = 0.0

        route_context = RouteContext(
            route_active=True,
            matched_pose=Pose2D(0.0, 0.0, 0.0),
            waypoint_mode_active=False,
        )
        lane_observation = LaneObservation(
            direct_error_m=0.08,
            quality=1.0,
            detected_sides=("left", "right"),
        )

        raw_x, raw_y, _, correction_m, reliable = estimator._apply_lane_observation(
            route_context,
            lane_observation,
            now=1000.0,
            raw_x=0.0,
            raw_y=0.08,
            raw_yaw=0.0,
        )

        self.assertTrue(reliable)
        self.assertGreater(correction_m, 0.0)
        self.assertAlmostEqual(raw_x, 0.0, places=4)
        self.assertLess(raw_y, 0.08)

    def test_apply_semantic_reset_reanchors_to_matched_pose(self):
        estimator = threadPoseEstimator.__new__(threadPoseEstimator)
        estimator._dr = _FakeDR(x=0.2, y=0.2, yaw=0.0)
        estimator._last_semantic_relocalization_t = 0.0
        estimator._last_sign_observation = {
            "sign": "stop",
            "observed_at_monotonic": time.monotonic(),
            "distance_m": 0.20,
        }
        estimator._last_yaw_rad = 0.0
        estimator._sign_matches_expected_semantic = threadPoseEstimator._sign_matches_expected_semantic.__get__(
            estimator,
            threadPoseEstimator,
        )

        route_context = RouteContext(
            route_active=True,
            matched_pose=Pose2D(1.2, 0.3, 0.4),
            expected_control_type="stop",
            next_semantic_type="intersection",
            next_semantic_distance_m=0.20,
            map_match_error_m=0.08,
        )

        applied, semantic_match = estimator._apply_semantic_reset(route_context, now=time.monotonic() + 0.2)

        self.assertTrue(applied)
        self.assertEqual(semantic_match[0], "sign:stop")
        self.assertAlmostEqual(estimator._dr.x, 1.2, places=4)
        self.assertAlmostEqual(estimator._dr.y, 0.3, places=4)
        self.assertAlmostEqual(estimator._dr.yaw, 0.4, places=4)


if __name__ == "__main__":
    unittest.main()
