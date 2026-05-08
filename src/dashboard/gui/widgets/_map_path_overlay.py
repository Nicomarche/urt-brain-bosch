from __future__ import annotations


def extract_path_points(path_pts) -> list[tuple[float, float]]:
    if not isinstance(path_pts, list):
        return []
    points: list[tuple[float, float]] = []
    for point in path_pts:
        try:
            if isinstance(point, dict):
                x_m = float(point["x"])
                y_m = float(point["y"])
            else:
                x_m = float(point[0])
                y_m = float(point[1])
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        points.append((x_m, y_m))
    return points


def extract_nav_route_preview_points(payload) -> list[tuple[float, float]]:
    if not isinstance(payload, dict):
        return []
    path_pts = payload.get("path") or payload.get("waypoints") or payload.get("route_points")
    return extract_path_points(path_pts)


def extract_control_path_points(payload) -> list[tuple[float, float]]:
    if not isinstance(payload, dict):
        return []
    if not bool(payload.get("valid", False)):
        return []
    return extract_path_points(payload.get("target_path"))
