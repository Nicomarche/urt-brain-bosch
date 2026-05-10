from __future__ import annotations

from typing import Mapping


OffsetPx = tuple[float, float]


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


def texture_offset_m(
    offset_px: OffsetPx,
    *,
    meters_per_pixel: float,
) -> tuple[float, float]:
    """Convert a visual pixel shift to texture meters, x-right/y-down."""
    mpp = float(meters_per_pixel)
    return float(offset_px[0]) * mpp, float(offset_px[1]) * mpp


def world_bounds_delta_for_visual_offset(
    offset_px: OffsetPx,
    *,
    meters_per_pixel: float,
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


def offset_is_effectively_zero(offset_px: OffsetPx, *, eps_px: float = 0.01) -> bool:
    return abs(float(offset_px[0])) <= eps_px and abs(float(offset_px[1])) <= eps_px
