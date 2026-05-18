from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.localization.chamfer_map_matcher import ChamferMapMatcher


def _metadata() -> dict:
    return {
        "world_bounds": {"x_min": 0.0, "y_min": 0.0},
        "meters_per_pixel_x": 0.01,
        "meters_per_pixel_y": 0.01,
        "image_width_px": 220,
        "image_height_px": 220,
        "y_axis_inverted": False,
    }


def test_chamfer_match_recovers_sparse_visual_lane_offset() -> None:
    meta = _metadata()
    binary = np.zeros((220, 220), dtype=np.uint8)
    # Map lane boundaries in world coordinates: y=0.825 and y=1.175.
    cv2.line(binary, (40, 82), (170, 82), 255, 2)
    cv2.line(binary, (40, 118), (170, 118), 255, 2)
    matcher = ChamferMapMatcher.from_binary_map(binary, meta)

    body_points = []
    for x_fwd in np.linspace(0.0, 1.2, 50):
        body_points.append((float(x_fwd), 0.175))
        body_points.append((float(x_fwd), -0.175))

    result = matcher.match_body_points(
        body_points,
        pose_x=0.40,
        pose_y=1.12,
        pose_yaw=0.0,
        search_radius_m=0.20,
        search_step_m=0.01,
        min_points=24,
        max_cost_px=10.0,
        min_improvement_px=1.0,
    )

    assert result.applied is True
    assert result.dy_m == pytest.approx(-0.12, abs=0.015)
    assert abs(result.dx_m) <= 0.02
    assert result.best_cost_px < result.baseline_cost_px


def test_chamfer_match_accepts_filtered_sparse_points_seen_in_runs() -> None:
    meta = _metadata()
    binary = np.zeros((220, 220), dtype=np.uint8)
    cv2.line(binary, (40, 82), (170, 82), 255, 2)
    cv2.line(binary, (40, 118), (170, 118), 255, 2)
    matcher = ChamferMapMatcher.from_binary_map(binary, meta)

    body_points = []
    for x_fwd in np.linspace(0.03, 1.16, 9):
        body_points.append((float(x_fwd), 0.175))
        body_points.append((float(x_fwd), -0.175))

    result = matcher.match_body_points(
        body_points,
        pose_x=0.40,
        pose_y=1.12,
        pose_yaw=0.0,
        search_radius_m=0.20,
        search_step_m=0.01,
        min_points=12,
        max_cost_px=10.0,
        min_improvement_px=1.0,
    )

    assert result.input_point_count == 18
    assert result.point_count == 18
    assert result.applied is True
    assert result.dy_m == pytest.approx(-0.12, abs=0.02)


def test_chamfer_match_rejects_when_current_pose_is_already_aligned() -> None:
    meta = _metadata()
    binary = np.zeros((220, 220), dtype=np.uint8)
    cv2.line(binary, (40, 82), (170, 82), 255, 2)
    cv2.line(binary, (40, 118), (170, 118), 255, 2)
    matcher = ChamferMapMatcher.from_binary_map(binary, meta)

    body_points = []
    for x_fwd in np.linspace(0.0, 1.2, 50):
        body_points.append((float(x_fwd), 0.175))
        body_points.append((float(x_fwd), -0.175))

    result = matcher.match_body_points(
        body_points,
        pose_x=0.40,
        pose_y=1.00,
        pose_yaw=0.0,
        search_radius_m=0.20,
        search_step_m=0.01,
        min_points=24,
    )

    assert result.applied is False
    assert result.reason in {"already_aligned", "insufficient_improvement"}
