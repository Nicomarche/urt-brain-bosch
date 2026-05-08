from __future__ import annotations

import queue
from unittest.mock import MagicMock

from src.behavior.planner_thread import threadBehaviorPlanner
from src.core.types.pose import Pose2D, PoseEstimate
from src.core.types.routing import RouteContext


class _Buffer:
    def __init__(self, value=None):
        self._value = value

    def read_latest(self):
        return self._value

    def write(self, value, timestamp=None):
        self._value = value


def test_behavior_planner_stops_in_auto_when_route_is_inactive() -> None:
    queues = {
        "Config": queue.Queue(),
        "General": queue.Queue(),
    }
    planner = MagicMock()
    pose_buf = _Buffer(
        PoseEstimate(
            timestamp=0.0,
            fused_pose=Pose2D(x=1.0, y=2.0, yaw=0.0),
            raw_pose=Pose2D(x=1.0, y=2.0, yaw=0.0),
            speed_mps=0.0,
            yaw_rad=0.0,
            localization_mode="ekf7",
        )
    )
    route_buf = _Buffer(RouteContext(route_active=False))
    output_buf = _Buffer()
    thread = threadBehaviorPlanner(
        queues,
        planner=planner,
        lanelet_map=None,
        pose_estimate_buffer=pose_buf,
        route_context_buffer=route_buf,
        lane_observation_buffer=_Buffer(),
        stopline_observation_buffer=_Buffer(),
        tracked_objects_buffer=None,
        behavior_output_buffer=output_buf,
        dt_s=0.05,
        horizon_n=10,
        nominal_speed_mps=0.5,
        max_speed_mps=1.0,
        pause_s=0.05,
    )
    thread._current_mode = "AUTO"

    thread.thread_work()

    planner.plan.assert_not_called()
    assert output_buf.read_latest() is not None
    assert output_buf.read_latest().valid is False
    assert output_buf.read_latest().stop_required is True
    assert output_buf.read_latest().notes["reason"] == "route_inactive"
