from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any


_logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_START_POSE_PATH = REPO_ROOT / "config" / "sim_start_pose.json"


def saved_start_pose_path() -> Path:
    return Path(SIM_START_POSE_PATH)


def _resolve_repo_path(path_like: str | os.PathLike[str] | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _normalize_saved_pose(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("start_pose"), dict):
        payload = payload["start_pose"]
    try:
        world_x = float(payload["world_x"])
        world_y = float(payload["world_y"])
    except (KeyError, TypeError, ValueError):
        return None

    normalized: dict[str, Any] = {
        "world_x": float(world_x),
        "world_y": float(world_y),
    }

    try:
        if payload.get("yaw_deg") is not None:
            normalized["yaw_deg"] = float(payload["yaw_deg"])
        elif payload.get("yaw_rad") is not None:
            normalized["yaw_rad"] = float(payload["yaw_rad"])
    except (TypeError, ValueError):
        pass

    lanelet_id = payload.get("lanelet_id")
    if lanelet_id is not None and str(lanelet_id).strip():
        normalized["lanelet_id"] = str(lanelet_id).strip()

    try:
        if payload.get("saved_at") is not None:
            normalized["saved_at"] = float(payload["saved_at"])
    except (TypeError, ValueError):
        pass

    resolved_map_dir = _resolve_repo_path(payload.get("map_dir"))
    if resolved_map_dir is not None:
        normalized["map_dir"] = str(resolved_map_dir)

    return normalized


def _save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def save_saved_start_pose(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_saved_pose(payload)
    if normalized is None:
        raise ValueError("Saved start pose requires numeric world_x/world_y.")
    _save_json_atomic(saved_start_pose_path(), normalized)
    return dict(normalized)


def load_saved_start_pose(*, map_dir: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    path = saved_start_pose_path()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _logger.warning("Failed to load saved sim start pose %s: %s", path, exc)
        return None

    normalized = _normalize_saved_pose(payload)
    if normalized is None:
        return None

    current_map_dir = _resolve_repo_path(map_dir)
    saved_map_dir = _resolve_repo_path(normalized.get("map_dir"))
    if current_map_dir is not None and saved_map_dir is not None and current_map_dir != saved_map_dir:
        return None
    return normalized


def resolve_saved_start_pose(graph, default: tuple[float, float, float] | None = None):
    if graph is None:
        return default
    osm_path = getattr(graph, "_osm_path", None)
    map_dir = Path(osm_path).resolve().parent if osm_path else None
    payload = load_saved_start_pose(map_dir=map_dir)
    if payload is None:
        return default
    try:
        default_yaw = float(default[2]) if default is not None else 0.0
        pose = graph.localisation_to_world_pose(payload, default_yaw=default_yaw)
    except Exception as exc:
        _logger.warning("Failed to resolve saved sim start pose against map: %s", exc)
        return default
    return pose if pose is not None else default
