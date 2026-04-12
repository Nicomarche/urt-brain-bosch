import math
import types
import unittest

from src.hardware.camera.threads.threadLaneObserver import threadLaneObserver
from src.hardware.control.navigationControlPolicy import NavigationControlPolicy
from src.hardware.pipeline.sharedBuffers import LatestValueBuffer
from src.hardware.pipeline.sharedTypes import Pose2D, RouteContext, VisualControlCandidate, VisualStateSnapshot
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

    def get_state(self):
        return self.x, self.y, self.yaw


class PipelineIntegrationTests(unittest.TestCase):
    def test_visual_snapshot_flows_into_pose_estimate_and_control_decision(self):
        visual_state_buffer = LatestValueBuffer()
        lane_observation_buffer = LatestValueBuffer()
        stopline_observation_buffer = LatestValueBuffer()

        candidate = VisualControlCandidate(
            timestamp=10.0,
            steering_deg=4.5,
            speed_cmd=3.5,
            confidence=1.0,
            source="ai_local",
            active=True,
        )
        snapshot = VisualStateSnapshot(
            timestamp=10.0,
            detection_mode="ai_local",
            curve_state="STRAIGHT",
            heading_error_rad=0.08,
            camera_yaw_hint_rad=0.04,
            camera_yaw_hint_confidence=0.8,
            local_lane_payload={"lane_side_point_counts": {"left": 4, "right": 4}},
            frame_trace={"debug": {"two_line_direct_error_m": 0.05}},
            candidate=candidate,
        )
        visual_state_buffer.write(snapshot, timestamp=snapshot.timestamp)

        observer = threadLaneObserver(
            queuesList={},
            visual_state_buffer=visual_state_buffer,
            lane_observation_buffer=lane_observation_buffer,
            stopline_observation_buffer=stopline_observation_buffer,
        )
        observer.thread_work()

        lane_observation = lane_observation_buffer.read_latest()
        self.assertAlmostEqual(lane_observation.direct_error_m, 0.05, places=4)
        self.assertEqual(lane_observation.detected_sides, ("left", "right"))

        estimator = threadPoseEstimator.__new__(threadPoseEstimator)
        estimator._dr = _FakeDR(x=0.0, y=0.05, yaw=0.0)
        estimator._last_speed = 0.20
        estimator._last_camera_lateral_correction_monotonic = 0.0
        estimator._last_speed_t = None
        estimator._last_cmd_speed_t = None
        estimator._last_steer_rad = 0.0
        estimator._last_speed_source = "encoder"
        estimator._imu_received = True

        route_context = RouteContext(
            route_active=True,
            waypoint_mode_active=False,
            matched_idx=12,
            matched_pose=Pose2D(0.0, 0.0, 0.0),
            map_match_error_m=0.04,
            maneuver_type="turn_left",
        )

        raw_x, raw_y, raw_yaw, correction_m, reliable = estimator._apply_lane_observation(
            route_context,
            lane_observation,
            now=1000.0,
            raw_x=0.0,
            raw_y=0.05,
            raw_yaw=0.0,
        )
        pose_estimate = estimator._build_pose_estimate(
            now=1000.0,
            raw_pose=Pose2D(0.0, 0.05, 0.0),
            fused_pose=Pose2D(raw_x, raw_y, raw_yaw),
            raw_lateral_error_m=0.05,
            lane_measurement_reliable=reliable,
            camera_lateral_correction_m=correction_m,
            relocalization_mode="lane_visual",
            relocalization_source="lane_observation",
            relocalization_error_m=abs(correction_m),
            route_context=route_context,
        )

        policy = NavigationControlPolicy()
        decision = policy.decide(
            candidate,
            tracking_state=types.SimpleNamespace(
                route_active=route_context.route_active,
                waypoint_mode_active=route_context.waypoint_mode_active,
                maneuver_type=route_context.maneuver_type,
            ),
        )

        self.assertTrue(pose_estimate.localization_confidence > 0.9)
        self.assertTrue(pose_estimate.lane_measurement_reliable)
        self.assertEqual(decision.authority, "planner")
        self.assertEqual(decision.speed_cmd, 3.5)


if __name__ == "__main__":
    unittest.main()
