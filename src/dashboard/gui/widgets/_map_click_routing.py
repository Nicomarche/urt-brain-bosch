from __future__ import annotations

import math


_CLICK_LANELET_COVER_MARGIN_M = 0.12
_CLICK_LANELET_DISTANCE_TIE_M = 0.02


def _distance_point_to_segment(
    x_m: float,
    y_m: float,
    ax_m: float,
    ay_m: float,
    bx_m: float,
    by_m: float,
) -> float:
    seg_dx = float(bx_m) - float(ax_m)
    seg_dy = float(by_m) - float(ay_m)
    seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
    if seg_len_sq <= 1e-12:
        return math.hypot(float(x_m) - float(ax_m), float(y_m) - float(ay_m))
    rel_x = float(x_m) - float(ax_m)
    rel_y = float(y_m) - float(ay_m)
    proj_t = max(0.0, min(1.0, (rel_x * seg_dx + rel_y * seg_dy) / seg_len_sq))
    proj_x = float(ax_m) + proj_t * seg_dx
    proj_y = float(ay_m) + proj_t * seg_dy
    return math.hypot(float(x_m) - proj_x, float(y_m) - proj_y)


def _distance_point_to_polyline(
    x_m: float,
    y_m: float,
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> float:
    if len(points) < 2:
        if not points:
            return float("inf")
        return math.hypot(float(x_m) - float(points[0][0]), float(y_m) - float(points[0][1]))
    best = float("inf")
    for idx in range(len(points) - 1):
        ax_m, ay_m = points[idx]
        bx_m, by_m = points[idx + 1]
        best = min(best, _distance_point_to_segment(x_m, y_m, ax_m, ay_m, bx_m, by_m))
    return best


def _distance_point_to_polygon_boundary(
    x_m: float,
    y_m: float,
    polygon: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> float:
    if len(polygon) < 2:
        return float("inf")
    best = float("inf")
    for idx in range(len(polygon)):
        ax_m, ay_m = polygon[idx]
        bx_m, by_m = polygon[(idx + 1) % len(polygon)]
        best = min(best, _distance_point_to_segment(x_m, y_m, ax_m, ay_m, bx_m, by_m))
    return best


def _point_in_polygon(
    x_m: float,
    y_m: float,
    polygon: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    px = float(x_m)
    py = float(y_m)
    for idx in range(len(polygon)):
        x0, y0 = polygon[idx]
        x1, y1 = polygon[(idx + 1) % len(polygon)]
        intersects = ((float(y0) > py) != (float(y1) > py))
        if not intersects:
            continue
        x_cross = float(x0) + (py - float(y0)) * (float(x1) - float(x0)) / (float(y1) - float(y0))
        if x_cross >= px:
            inside = not inside
    return inside


def _corridor_rank(
    lanelet_id: str,
    *,
    current_lanelet_id: str | None,
    next_lanelet_ids: tuple[str, ...],
) -> tuple[int, int]:
    candidate = str(lanelet_id)
    if current_lanelet_id and candidate == str(current_lanelet_id):
        return (0, 0)
    for idx, item in enumerate(next_lanelet_ids):
        if candidate == str(item):
            return (1, idx)
    return (2, 0)


def resolve_click_destination_lanelet(
    *,
    x_m: float,
    y_m: float,
    lanelet_map,
    lanelet_polygons,
    current_lanelet_id: str | None = None,
    next_lanelet_ids: tuple[str, ...] = (),
) -> str | None:
    """Resolve the lanelet the operator most likely intended to click."""
    if lanelet_map is None:
        return None
    polygon_by_lanelet_id = dict(lanelet_polygons or {})
    covered: list[tuple[float, tuple[int, int], str]] = []
    fallback: list[tuple[float, tuple[int, int], str]] = []

    for lanelet_id in getattr(lanelet_map, "lanelet_ids", ()):
        candidate_id = str(lanelet_id)
        lanelet = lanelet_map.get_lanelet(candidate_id)
        if lanelet is None or getattr(lanelet.centerline, "shape", (0,))[0] < 1:
            continue
        centerline_pts = [
            (float(pt[0]), float(pt[1]))
            for pt in lanelet.centerline.tolist()
        ]
        centerline_distance_m = _distance_point_to_polyline(x_m, y_m, centerline_pts)
        rank = _corridor_rank(
            candidate_id,
            current_lanelet_id=current_lanelet_id,
            next_lanelet_ids=next_lanelet_ids,
        )
        fallback.append((centerline_distance_m, rank, candidate_id))

        polygon = polygon_by_lanelet_id.get(candidate_id)
        if not polygon:
            continue
        if (
            _point_in_polygon(x_m, y_m, polygon)
            or _distance_point_to_polygon_boundary(x_m, y_m, polygon) <= _CLICK_LANELET_COVER_MARGIN_M
        ):
            covered.append((centerline_distance_m, rank, candidate_id))

    if covered:
        min_distance_m = min(item[0] for item in covered)
        contenders = [
            item
            for item in covered
            if item[0] <= (min_distance_m + _CLICK_LANELET_DISTANCE_TIE_M)
        ]
        contenders.sort(key=lambda item: (item[1], item[0], item[2]))
        return contenders[0][2]

    if not fallback:
        return None
    fallback.sort(key=lambda item: (item[0], item[1], item[2]))
    return fallback[0][2]
