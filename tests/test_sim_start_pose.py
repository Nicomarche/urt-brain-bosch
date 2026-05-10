from __future__ import annotations

import math
from pathlib import Path

from src.utils import sim_start_pose as start_pose_module


class _FakeGraph:
    def __init__(self, osm_path: Path):
        self._osm_path = str(osm_path)
        self.calls: list[tuple[dict, float]] = []

    def localisation_to_world_pose(self, payload, *, default_yaw: float | None = None):
        self.calls.append((dict(payload), float(default_yaw if default_yaw is not None else 0.0)))
        yaw_rad = (
            math.radians(float(payload["yaw_deg"]))
            if payload.get("yaw_deg") is not None
            else float(default_yaw if default_yaw is not None else 0.0)
        )
        return (float(payload["world_x"]), float(payload["world_y"]), yaw_rad)


def test_save_and_load_saved_start_pose_roundtrip(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config" / "sim_start_pose.json"
    map_dir = tmp_path / "maps" / "sim"
    map_dir.mkdir(parents=True)
    monkeypatch.setattr(start_pose_module, "SIM_START_POSE_PATH", path)

    saved = start_pose_module.save_saved_start_pose(
        {
            "world_x": 1.25,
            "world_y": -0.75,
            "yaw_deg": 92.5,
            "lanelet_id": "42",
            "map_dir": map_dir,
            "saved_at": 123.4,
        }
    )

    loaded = start_pose_module.load_saved_start_pose(map_dir=map_dir)

    assert saved == loaded
    assert loaded is not None
    assert loaded["map_dir"] == str(map_dir.resolve())


def test_load_saved_start_pose_ignores_other_map(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config" / "sim_start_pose.json"
    map_dir = tmp_path / "maps" / "sim"
    other_map_dir = tmp_path / "maps" / "other"
    map_dir.mkdir(parents=True)
    other_map_dir.mkdir(parents=True)
    monkeypatch.setattr(start_pose_module, "SIM_START_POSE_PATH", path)

    start_pose_module.save_saved_start_pose(
        {
            "world_x": 2.0,
            "world_y": 3.0,
            "yaw_deg": 180.0,
            "map_dir": map_dir,
        }
    )

    assert start_pose_module.load_saved_start_pose(map_dir=other_map_dir) is None


def test_resolve_saved_start_pose_uses_graph_localisation(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config" / "sim_start_pose.json"
    map_dir = tmp_path / "maps" / "sim"
    osm_path = map_dir / "lanelet2_map.osm"
    map_dir.mkdir(parents=True)
    osm_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(start_pose_module, "SIM_START_POSE_PATH", path)

    start_pose_module.save_saved_start_pose(
        {
            "world_x": 4.5,
            "world_y": -1.25,
            "yaw_deg": 45.0,
            "map_dir": map_dir,
        }
    )
    graph = _FakeGraph(osm_path)

    pose = start_pose_module.resolve_saved_start_pose(
        graph,
        default=(0.0, 0.0, 0.0),
    )

    assert pose == (4.5, -1.25, math.radians(45.0))
    assert graph.calls == [
        (
            {
                "map_dir": str(map_dir.resolve()),
                "world_x": 4.5,
                "world_y": -1.25,
                "yaw_deg": 45.0,
            },
            0.0,
        )
    ]
