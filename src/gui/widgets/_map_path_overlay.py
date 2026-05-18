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


def extract_control_path_points_with_speed(
    payload,
) -> list[tuple[float, float, float]]:
    """Plan F1: extrae el target_path emparejado con speed_profile.

    Devuelve tuplas ``(x, y, speed_mps)`` para que el map_view pueda
    colorear cada segmento del path por velocidad. Si el payload no
    tiene speed_profile o las longitudes no coinciden, se pone 0.0.
    """
    if not isinstance(payload, dict):
        return []
    if not bool(payload.get("valid", False)):
        return []
    xy_points = extract_path_points(payload.get("target_path"))
    if not xy_points:
        return []
    speed_profile = payload.get("speed_profile") or []
    out: list[tuple[float, float, float]] = []
    for idx, (x, y) in enumerate(xy_points):
        speed = 0.0
        if idx < len(speed_profile):
            try:
                speed = float(speed_profile[idx])
            except (TypeError, ValueError):
                speed = 0.0
        out.append((float(x), float(y), float(speed)))
    return out


def speed_to_color_hsl(speed_mps: float, *, max_speed_mps: float = 0.5) -> str:
    """Plan F1: mapea velocidad [0, max] → color HSL hue 0..120 (rojo→verde).

    Returns: string ``"#RRGGBB"`` para usar directo en QPen.
    """
    speed_mps = max(0.0, float(speed_mps))
    max_speed = max(1e-3, float(max_speed_mps))
    ratio = min(1.0, speed_mps / max_speed)
    hue = ratio * 120.0  # 0=rojo, 120=verde
    # HSL → RGB con S=0.85, L=0.5
    h = hue / 360.0
    s = 0.85
    l = 0.50
    if s == 0:
        r = g = b = l
    else:
        q = l + s - l * s
        p_ = 2.0 * l - q
        def _hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1.0 / 6:
                return p + (q - p) * 6.0 * t
            if t < 1.0 / 2:
                return q
            if t < 2.0 / 3:
                return p + (q - p) * (2.0 / 3 - t) * 6.0
            return p
        r = _hue_to_rgb(p_, q, h + 1.0 / 3)
        g = _hue_to_rgb(p_, q, h)
        b = _hue_to_rgb(p_, q, h - 1.0 / 3)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
