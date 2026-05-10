from __future__ import annotations

import math
from typing import Mapping


OffsetPx = tuple[float, float]
Point = tuple[float, float]
ScaleXY = tuple[float, float]


def relative_offset_px(
    *,
    lanelet_offset_px: OffsetPx,
    background_offset_px: OffsetPx,
) -> OffsetPx:
    """Return lanelet-layer displacement relative to the SVG/background."""
    return (
        float(lanelet_offset_px[0]) - float(background_offset_px[0]),
        float(lanelet_offset_px[1]) - float(background_offset_px[1]),
    )


def relative_layer_adjustment_px(
    *,
    lanelet_offset_px: OffsetPx,
    background_offset_px: OffsetPx,
    lanelet_scale_xy: ScaleXY,
    background_scale_xy: ScaleXY,
) -> tuple[OffsetPx, ScaleXY]:
    """Return lanelet layer transform relative to the background layer.

    This lets the user drag/scale either layer while the persisted alignment
    always describes how lanelets should map into the SVG/simulator frame.
    """
    bg_sx = float(background_scale_xy[0])
    bg_sy = float(background_scale_xy[1])
    if abs(bg_sx) <= 1e-12 or abs(bg_sy) <= 1e-12:
        raise ValueError("background alignment scale cannot be zero")
    lane_sx = float(lanelet_scale_xy[0])
    lane_sy = float(lanelet_scale_xy[1])
    return (
        (
            (float(lanelet_offset_px[0]) - float(background_offset_px[0])) / bg_sx,
            (float(lanelet_offset_px[1]) - float(background_offset_px[1])) / bg_sy,
        ),
        (lane_sx / bg_sx, lane_sy / bg_sy),
    )


def texture_offset_m(
    offset_px: OffsetPx,
    *,
    meters_per_pixel: float | None = None,
    meters_per_pixel_x: float | None = None,
    meters_per_pixel_y: float | None = None,
) -> tuple[float, float]:
    """Convert a visual pixel shift to texture meters, x-right/y-down."""
    if meters_per_pixel_x is None:
        if meters_per_pixel is None:
            raise TypeError("meters_per_pixel or meters_per_pixel_x is required")
        meters_per_pixel_x = meters_per_pixel
    if meters_per_pixel_y is None:
        meters_per_pixel_y = meters_per_pixel_x
    return (
        float(offset_px[0]) * float(meters_per_pixel_x),
        float(offset_px[1]) * float(meters_per_pixel_y),
    )


def world_bounds_delta_for_visual_offset(
    offset_px: OffsetPx,
    *,
    meters_per_pixel: float | None = None,
    meters_per_pixel_x: float | None = None,
    meters_per_pixel_y: float | None = None,
    y_axis_inverted: bool,
) -> tuple[float, float]:
    """World-bounds delta that makes map geometry render at ``offset_px``.

    ``world_to_pixel`` computes x from ``(x - x_min) / meters_per_pixel``.
    Therefore moving map geometry visually right by +dx px means decreasing
    x_min by dx meters. Y depends on whether map Y is inverted into pixels.
    """
    dx_texture_m, dy_texture_m = texture_offset_m(
        offset_px,
        meters_per_pixel=meters_per_pixel,
        meters_per_pixel_x=meters_per_pixel_x,
        meters_per_pixel_y=meters_per_pixel_y,
    )
    bounds_dx_m = -dx_texture_m
    bounds_dy_m = dy_texture_m if bool(y_axis_inverted) else -dy_texture_m
    return bounds_dx_m, bounds_dy_m


def shifted_world_bounds(
    bounds: Mapping[str, object],
    *,
    dx_m: float,
    dy_m: float,
) -> dict[str, float]:
    """Shift x/y world-bounds while preserving map width and height."""
    x_min = float(bounds.get("x_min", 0.0) or 0.0)
    x_max = float(bounds.get("x_max", x_min) or x_min)
    y_min = float(bounds.get("y_min", 0.0) or 0.0)
    y_max = float(bounds.get("y_max", y_min) or y_min)
    return {
        "x_min": round(x_min + float(dx_m), 6),
        "x_max": round(x_max + float(dx_m), 6),
        "y_min": round(y_min + float(dy_m), 6),
        "y_max": round(y_max + float(dy_m), 6),
    }


def _pair(
    value: object,
    *,
    default: tuple[float, float],
) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return default


def map_to_texture_point(
    point: Point,
    transform: Mapping[str, object] | None,
) -> Point:
    """Map lanelet/world coordinates into the SVG/simulator texture frame."""
    if not transform:
        return float(point[0]), float(point[1])
    sx, sy = _pair(transform.get("scale_xy"), default=(1.0, 1.0))
    tx, ty = _pair(transform.get("translation_xy"), default=(0.0, 0.0))
    yaw_rad = math.radians(float(transform.get("yaw_deg", 0.0) or 0.0))
    x = float(point[0]) * sx
    y = float(point[1]) * sy
    if abs(yaw_rad) > 1e-12:
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        x, y = c * x - s * y, s * x + c * y
    return x + tx, y + ty


def texture_to_map_point(
    point: Point,
    transform: Mapping[str, object] | None,
) -> Point:
    """Inverse of :func:`map_to_texture_point`."""
    if not transform:
        return float(point[0]), float(point[1])
    sx, sy = _pair(transform.get("scale_xy"), default=(1.0, 1.0))
    if abs(sx) <= 1e-12 or abs(sy) <= 1e-12:
        raise ValueError("map_to_texture scale cannot be zero")
    tx, ty = _pair(transform.get("translation_xy"), default=(0.0, 0.0))
    x = float(point[0]) - tx
    y = float(point[1]) - ty
    yaw_rad = math.radians(float(transform.get("yaw_deg", 0.0) or 0.0))
    if abs(yaw_rad) > 1e-12:
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        x, y = c * x + s * y, -s * x + c * y
    return x / sx, y / sy


def shifted_map_transform_translation(
    transform: Mapping[str, object] | None,
    *,
    dx_m: float,
    dy_m: float,
) -> dict[str, object]:
    """Shift a map-to-texture transform translation without changing scale."""
    updated = dict(transform or {})
    tx, ty = _pair(updated.get("translation_xy"), default=(0.0, 0.0))
    updated["translation_xy"] = [
        round(tx + float(dx_m), 9),
        round(ty + float(dy_m), 9),
    ]
    return updated


def adjusted_map_transform(
    transform: Mapping[str, object] | None,
    *,
    scale_xy: ScaleXY,
    offset_m: Point,
    anchor_m: Point,
) -> dict[str, object]:
    """Compose a preview layer adjustment into a map-to-texture transform."""
    updated = dict(transform or {})
    old_sx, old_sy = _pair(updated.get("scale_xy"), default=(1.0, 1.0))
    old_tx, old_ty = _pair(updated.get("translation_xy"), default=(0.0, 0.0))
    sx, sy = float(scale_xy[0]), float(scale_xy[1])
    off_x, off_y = float(offset_m[0]), float(offset_m[1])
    anchor_x, anchor_y = float(anchor_m[0]), float(anchor_m[1])
    updated["scale_xy"] = [
        round(old_sx * sx, 12),
        round(old_sy * sy, 12),
    ]
    updated["translation_xy"] = [
        round(anchor_x + sx * (old_tx - anchor_x) + off_x, 9),
        round(anchor_y + sy * (old_ty - anchor_y) + off_y, 9),
    ]
    return updated


def scale_is_effectively_one(scale_xy: ScaleXY, *, eps: float = 1e-6) -> bool:
    return abs(float(scale_xy[0]) - 1.0) <= eps and abs(float(scale_xy[1]) - 1.0) <= eps


def offset_is_effectively_zero(offset_px: OffsetPx, *, eps_px: float = 0.01) -> bool:
    return abs(float(offset_px[0])) <= eps_px and abs(float(offset_px[1])) <= eps_px
