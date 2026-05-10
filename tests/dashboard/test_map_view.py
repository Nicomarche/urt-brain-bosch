from __future__ import annotations

from pathlib import Path

import numpy as np

from src.dashboard.gui.config import settings
from src.dashboard.gui.widgets._map_alignment import (
    adjusted_map_transform,
    map_to_texture_point,
    relative_layer_adjustment_px,
    relative_offset_px,
    scale_is_effectively_one,
    shifted_map_transform_translation,
    shifted_world_bounds,
    texture_offset_m,
    texture_to_map_point,
    world_bounds_delta_for_visual_offset,
)
from src.dashboard.gui.widgets._map_path_overlay import (
    extract_control_path_points,
    extract_nav_route_preview_points,
)
from src.dashboard.gui.widgets._map_click_routing import resolve_click_destination_lanelet
from src.routing.lanelet.lanelet_map import Lanelet, LaneletMap


def _lanelet(lanelet_id: str, centerline_pts: list[tuple[float, float]]) -> Lanelet:
    centerline = np.asarray(centerline_pts, dtype=float)
    if centerline.shape[0] >= 2:
        length_m = float(np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1)))
    else:
        length_m = 0.0
    return Lanelet(
        lanelet_id=lanelet_id,
        source_node_id=f"{lanelet_id}:src",
        target_node_id=f"{lanelet_id}:dst",
        centerline=centerline,
        length_m=length_m,
        attribute=0,
    )


def test_resolve_click_destination_lanelet_prefers_polygon_cover_over_nearest_centerline() -> None:
    lanelet_map = LaneletMap(
        lanelets={
            "covered": _lanelet("covered", [(0.0, 0.0), (1.0, 0.0)]),
            "nearby": _lanelet("nearby", [(0.0, 0.42), (1.0, 0.42)]),
        },
        regulators={},
        kdtree_index=None,
    )
    lanelet_polygons = {
        "covered": ((0.0, -0.50), (1.0, -0.50), (1.0, 0.80), (0.0, 0.80)),
        "nearby": ((0.0, 0.90), (1.0, 0.90), (1.0, 1.10), (0.0, 1.10)),
    }

    resolved = resolve_click_destination_lanelet(
        x_m=0.5,
        y_m=0.35,
        lanelet_map=lanelet_map,
        lanelet_polygons=lanelet_polygons,
    )

    assert resolved == "covered"


def test_resolve_click_destination_lanelet_prefers_active_corridor_on_distance_tie() -> None:
    lanelet_map = LaneletMap(
        lanelets={
            "current": _lanelet("current", [(0.0, 0.10), (1.0, 0.10)]),
            "other": _lanelet("other", [(0.0, -0.10), (1.0, -0.10)]),
        },
        regulators={},
        kdtree_index=None,
    )
    overlapping_polygon = ((0.0, -0.25), (1.0, -0.25), (1.0, 0.25), (0.0, 0.25))

    resolved = resolve_click_destination_lanelet(
        x_m=0.5,
        y_m=0.0,
        lanelet_map=lanelet_map,
        lanelet_polygons={
            "current": overlapping_polygon,
            "other": overlapping_polygon,
        },
        current_lanelet_id="current",
        next_lanelet_ids=("other",),
    )

    assert resolved == "current"


def test_default_track_map_dir_uses_backend_track_map_dir(monkeypatch) -> None:
    expected = Path("/tmp/custom-track-map")
    monkeypatch.setattr(settings, "TRACK_MAP_DIR", expected)

    assert settings.default_track_map_dir() == expected


def test_extract_nav_route_preview_points_reads_route_preview_payload() -> None:
    payload = {
        "route_points": [
            {"x": 1.0, "y": 2.0},
            {"x": 3.0, "y": 4.0},
        ]
    }

    assert extract_nav_route_preview_points(payload) == [(1.0, 2.0), (3.0, 4.0)]


def test_extract_control_path_points_reads_behavior_output_target_path() -> None:
    payload = {
        "valid": True,
        "target_path": [
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.2],
            [5.0, 6.0, 0.4],
        ],
    }

    assert extract_control_path_points(payload) == [
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
    ]


def test_extract_control_path_points_hides_invalid_behavior_path() -> None:
    payload = {
        "valid": False,
        "target_path": [
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.2],
        ],
    }

    assert extract_control_path_points(payload) == []


def test_alignment_relative_offset_can_drag_either_layer() -> None:
    assert relative_offset_px(
        lanelet_offset_px=(12.0, -3.0),
        background_offset_px=(0.0, 0.0),
    ) == (12.0, -3.0)
    assert relative_offset_px(
        lanelet_offset_px=(0.0, 0.0),
        background_offset_px=(-12.0, 3.0),
    ) == (12.0, -3.0)


def test_alignment_relative_adjustment_handles_scaling_either_layer() -> None:
    assert relative_layer_adjustment_px(
        lanelet_offset_px=(12.0, -3.0),
        background_offset_px=(0.0, 0.0),
        lanelet_scale_xy=(1.02, 0.98),
        background_scale_xy=(1.0, 1.0),
    ) == ((12.0, -3.0), (1.02, 0.98))

    rel_offset, rel_scale = relative_layer_adjustment_px(
        lanelet_offset_px=(0.0, 0.0),
        background_offset_px=(-12.0, 3.0),
        lanelet_scale_xy=(1.0, 1.0),
        background_scale_xy=(1.02, 0.98),
    )
    assert tuple(round(value, 6) for value in rel_offset) == (11.764706, -3.061224)
    assert tuple(round(value, 6) for value in rel_scale) == (0.980392, 1.020408)


def test_alignment_world_bounds_delta_matches_non_inverted_dashboard_frame() -> None:
    visual_px = (10.0, -5.0)

    assert texture_offset_m(visual_px, meters_per_pixel=0.01) == (0.1, -0.05)
    assert world_bounds_delta_for_visual_offset(
        visual_px,
        meters_per_pixel=0.01,
        y_axis_inverted=False,
    ) == (-0.1, 0.05)
    assert shifted_world_bounds(
        {"x_min": -10.0, "x_max": 10.0, "y_min": -7.0, "y_max": 7.0},
        dx_m=-0.1,
        dy_m=0.05,
    ) == {
        "x_min": -10.1,
        "x_max": 9.9,
        "y_min": -6.95,
        "y_max": 7.05,
    }


def test_alignment_world_bounds_delta_flips_y_when_axis_is_inverted() -> None:
    assert world_bounds_delta_for_visual_offset(
        (0.0, 5.0),
        meters_per_pixel=0.01,
        y_axis_inverted=True,
    ) == (-0.0, 0.05)


def test_alignment_offsets_support_distinct_x_y_scale() -> None:
    visual_px = (10.0, -5.0)

    assert texture_offset_m(
        visual_px,
        meters_per_pixel_x=0.01,
        meters_per_pixel_y=0.02,
    ) == (0.1, -0.1)
    assert world_bounds_delta_for_visual_offset(
        visual_px,
        meters_per_pixel_x=0.01,
        meters_per_pixel_y=0.02,
        y_axis_inverted=False,
    ) == (-0.1, 0.1)


def test_map_to_texture_transform_scales_and_round_trips_points() -> None:
    transform = {
        "scale_xy": [1.015, 1.027],
        "translation_xy": [-0.089, 0.119],
        "yaw_deg": 0.0,
    }
    point = (4.0, -2.0)

    texture_point = map_to_texture_point(point, transform)

    assert tuple(round(value, 6) for value in texture_point) == (3.971, -1.935)
    assert tuple(round(value, 6) for value in texture_to_map_point(texture_point, transform)) == point


def test_shifted_map_transform_translation_preserves_scale() -> None:
    transform = {
        "scale_xy": [1.015, 1.027],
        "translation_xy": [-0.089, 0.119],
    }

    assert shifted_map_transform_translation(
        transform,
        dx_m=0.1,
        dy_m=-0.2,
    ) == {
        "scale_xy": [1.015, 1.027],
        "translation_xy": [0.011, -0.081],
    }


def test_adjusted_map_transform_scales_around_anchor() -> None:
    transform = {
        "scale_xy": [1.015, 1.027],
        "translation_xy": [-0.089, 0.119],
    }

    assert adjusted_map_transform(
        transform,
        scale_xy=(1.1, 0.9),
        offset_m=(0.2, -0.1),
        anchor_m=(0.0, 0.0),
    ) == {
        "scale_xy": [1.1165, 0.9243],
        "translation_xy": [0.1021, 0.0071],
    }


def test_scale_is_effectively_one() -> None:
    assert scale_is_effectively_one((1.0, 1.0))
    assert not scale_is_effectively_one((1.001, 1.0))
