from __future__ import annotations

import queue
from unittest.mock import MagicMock

import numpy as np
import pytest
import src.behavior.planner_thread as planner_thread_module
from src.behavior.planner_thread import threadBehaviorPlanner
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.core.types.pose import Pose2D, PoseEstimate
from src.core.types.routing import RouteContext


class _Buffer:
    def __init__(self, value=None):
        self._value = value

    def read_latest(self):
        return self._value

    def write(self, value, timestamp=None):
        self._value = value


class _LaneletMapStub:
    def __init__(self, lanelet_ids=()):
        self._lanelet_ids = {str(item) for item in lanelet_ids}

    def get_lanelet(self, lanelet_id):
        if str(lanelet_id) in self._lanelet_ids:
            return object()
        return None

    def __len__(self):
        return len(self._lanelet_ids)


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


def test_behavior_planner_holds_stop_while_auto_gps_entry_is_pending() -> None:
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
            relocalization_mode="auto_gps_entry_pending",
        )
    )
    route_buf = _Buffer(RouteContext(route_active=True))
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
    assert output_buf.read_latest().valid is False
    assert output_buf.read_latest().notes["reason"] == "auto_gps_entry_pending"


def test_behavior_planner_ramp_starts_from_zero_without_min_speed_jump() -> None:
    queues = {
        "Config": queue.Queue(),
        "General": queue.Queue(),
    }

    def _plan(ctx):
        return BehaviorOutput(
            timestamp=0.0,
            dt=ctx.dt,
            target_path=np.zeros((ctx.horizon_n + 1, 3), dtype=float),
            speed_profile=np.full(ctx.horizon_n, ctx.nominal_speed_mps, dtype=float),
            scenario_name=ScenarioName.LANE_KEEP.value,
            valid=True,
        )

    planner = MagicMock()
    planner.plan.side_effect = _plan
    pose_buf = _Buffer(
        PoseEstimate(
            timestamp=0.0,
            fused_pose=Pose2D(x=1.0, y=2.0, yaw=0.0),
            raw_pose=Pose2D(x=1.0, y=2.0, yaw=0.0),
        )
    )
    route_buf = _Buffer(RouteContext(route_active=True, route_id="route-test"))
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
    thread._accel_mps2 = 0.25
    thread._min_moving_speed_mps = 0.20

    thread.thread_work()

    assert planner.plan.call_args.args[0].nominal_speed_mps == pytest.approx(0.0125)
    assert output_buf.read_latest().speed_profile[0] == pytest.approx(0.0125)


def test_behavior_planner_recovers_lanelet_map_from_tracking_config(monkeypatch) -> None:
    sentinel_lanelet_map = object()
    monkeypatch.setattr(
        planner_thread_module,
        "_load_lanelet_map_from_tracking_config",
        lambda: sentinel_lanelet_map,
    )

    queues = {
        "Config": queue.Queue(),
        "General": queue.Queue(),
    }
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
    route_buf = _Buffer(RouteContext(route_active=True, route_id="route-test"))
    thread = threadBehaviorPlanner(
        queues,
        planner=MagicMock(),
        lanelet_map=None,
        pose_estimate_buffer=pose_buf,
        route_context_buffer=route_buf,
        lane_observation_buffer=_Buffer(),
        stopline_observation_buffer=_Buffer(),
        tracked_objects_buffer=None,
        behavior_output_buffer=_Buffer(),
        dt_s=0.05,
        horizon_n=10,
        nominal_speed_mps=0.5,
        max_speed_mps=1.0,
        pause_s=0.05,
    )

    ctx = thread._build_context(now_s=0.0)

    assert ctx is not None
    assert ctx.lanelet_map is sentinel_lanelet_map
    assert thread._lanelet_map is sentinel_lanelet_map


def test_behavior_planner_reloads_lanelet_map_when_route_lanelet_is_missing(monkeypatch) -> None:
    recovered_lanelet_map = _LaneletMapStub(("154", "159"))
    monkeypatch.setattr(
        planner_thread_module,
        "_load_lanelet_map_from_tracking_config",
        lambda force_reload=False: recovered_lanelet_map,
    )

    queues = {
        "Config": queue.Queue(),
        "General": queue.Queue(),
    }
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
    route_buf = _Buffer(
        RouteContext(
            route_active=True,
            route_id="route-test",
            current_lanelet_id="154",
            next_lanelet_ids=("159",),
        )
    )
    thread = threadBehaviorPlanner(
        queues,
        planner=MagicMock(),
        lanelet_map=_LaneletMapStub(()),
        pose_estimate_buffer=pose_buf,
        route_context_buffer=route_buf,
        lane_observation_buffer=_Buffer(),
        stopline_observation_buffer=_Buffer(),
        tracked_objects_buffer=None,
        behavior_output_buffer=_Buffer(),
        dt_s=0.05,
        horizon_n=10,
        nominal_speed_mps=0.5,
        max_speed_mps=1.0,
        pause_s=0.05,
    )

    ctx = thread._build_context(now_s=0.0)

    assert ctx is not None
    assert ctx.lanelet_map is recovered_lanelet_map
    assert thread._lanelet_map is recovered_lanelet_map
