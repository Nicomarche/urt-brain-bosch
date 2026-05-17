from __future__ import annotations

from pathlib import Path

import numpy as np

from src.gui.config import settings
from src.gui.widgets._map_path_overlay import (
    extract_control_path_points,
    extract_nav_route_preview_points,
)
from src.gui.widgets._map_click_routing import resolve_click_destination_lanelet
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
