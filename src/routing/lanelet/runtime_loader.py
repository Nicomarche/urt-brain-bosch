from __future__ import annotations

import os
import threading

from src.routing.lanelet.lanelet_map import LaneletMap

_CACHE_LOCK = threading.Lock()
_CACHED_LANELET_MAP: LaneletMap | None = None
_CACHED_KEY: tuple[str, float] | None = None


def _resolve_tracking_lanelet_config() -> tuple[str, float] | None:
    try:
        import config as _cfg
    except Exception:
        return None

    osm_path = str(getattr(_cfg, "TRACKING_LANELET2_OSM", "") or "")
    if not osm_path:
        return None
    if not os.path.isabs(osm_path):
        repo_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        )
        osm_path = os.path.join(repo_root, osm_path)
    if not os.path.exists(osm_path):
        return None

    step_m = float(getattr(_cfg, "TRACKING_WAYPOINT_STEP_M", 0.05) or 0.05)
    return osm_path, step_m


def load_tracking_lanelet_map(*, force_reload: bool = False) -> LaneletMap | None:
    global _CACHED_KEY, _CACHED_LANELET_MAP

    resolved = _resolve_tracking_lanelet_config()
    if resolved is None:
        return None
    osm_path, step_m = resolved
    cache_key = (str(osm_path), float(step_m))

    with _CACHE_LOCK:
        if not force_reload and _CACHED_LANELET_MAP is not None and _CACHED_KEY == cache_key:
            return _CACHED_LANELET_MAP

    try:
        from src.routing.lanelet.from_osm import load_lanelet2_osm

        lanelet_map = load_lanelet2_osm(osm_path, step_m=step_m)
    except Exception:
        return None

    with _CACHE_LOCK:
        _CACHED_LANELET_MAP = lanelet_map
        _CACHED_KEY = cache_key
    return lanelet_map


def lanelet_map_covers_ids(
    lanelet_map: LaneletMap | None,
    lanelet_ids: tuple[str, ...] | list[str],
) -> bool:
    if lanelet_map is None:
        return False
    clean_ids = tuple(str(item) for item in lanelet_ids if str(item))
    if not clean_ids:
        return True
    for lanelet_id in clean_ids:
        if lanelet_map.get_lanelet(str(lanelet_id)) is None:
            return False
    return True
