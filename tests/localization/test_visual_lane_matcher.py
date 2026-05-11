from __future__ import annotations

import math

import pytest

from src.core.types.perception import LaneObservation
from src.core.types.pose import Pose2D
from src.core.types.routing import RouteContext
from src.localization.visual_lane_matcher import match_visual_lane_to_route


def _route() -> RouteContext:
    return RouteContext(
        route_active=True,
        current_lanelet_id="n1->n2",
        matched_idx=0,
        matched_pose=Pose2D(0.0, 0.0, 0.0),
        route_waypoints=[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )


def _lane(y_left: float = 0.0) -> LaneObservation:
    return LaneObservation(
        detected_sides=("left", "right"),
        quality=0.95,
        measurement_mode="two_line",
        direct_error_valid=True,
        center_waypoints_body=tuple((0.05 * i, y_left, 0.0) for i in range(12)),
        lane_width_m=0.35,
    )


def _lane_with_yaw(yaw_deg: float) -> LaneObservation:
    slope = math.tan(math.radians(float(yaw_deg)))
    return LaneObservation(
        detected_sides=("left", "right"),
        quality=0.95,
        measurement_mode="two_line",
        direct_error_valid=True,
        center_waypoints_body=tuple((0.05 * i, slope * 0.05 * i, 0.0) for i in range(12)),
        lane_width_m=0.35,
    )


def test_visual_lane_match_accepts_lane_curve_consistent_with_route() -> None:
    match = match_visual_lane_to_route(
        _route(),
        _lane(0.0),
        Pose2D(0.0, 0.0, 0.0),
    )

    assert match is not None
    assert match.accepted is True
    assert match.lanelet_id == "n1->n2"
    assert match.lateral_error_m == pytest.approx(0.0, abs=1e-9)
    assert match.yaw_error_rad == pytest.approx(0.0, abs=1e-9)
    assert match.confidence > 0.9


def test_visual_lane_match_rejects_large_lateral_map_disagreement() -> None:
    match = match_visual_lane_to_route(
        _route(),
        _lane(0.30),
        Pose2D(0.0, 0.0, 0.0),
    )

    assert match is not None
    assert match.accepted is False
    assert abs(match.lateral_error_m) > 0.175


def test_visual_lane_match_rejects_near_field_yaw_disagreement() -> None:
    match = match_visual_lane_to_route(
        _route(),
        _lane_with_yaw(9.0),
        Pose2D(0.0, 0.0, 0.0),
    )

    assert match is not None
    assert match.accepted is False
    assert abs(match.near_yaw_error_rad) > math.radians(6.0)
    assert match.reason in {"median_yaw_error", "near_yaw_error", "sample_yaw_error"}
