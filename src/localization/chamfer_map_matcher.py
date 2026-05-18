"""Sparse visual-map chamfer matching.

This is the production-shaped version of the experiment in
``urt-ref/.../template_maching/chamfer.py``.  Instead of sliding a whole BEV
image over a raster track, we slide sparse BEV lane-line points over a distance
transform of the lanelet boundaries.  That keeps the runtime cheap enough for
the control process and avoids matching against text/checkers in ``track.png``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.core.frames import body_to_world_yflip


@dataclass(frozen=True)
class ChamferMatchResult:
    applied: bool
    reason: str
    dx_m: float = 0.0
    dy_m: float = 0.0
    best_cost_px: float = math.inf
    baseline_cost_px: float = math.inf
    improvement_px: float = 0.0
    input_point_count: int = 0
    point_count: int = 0
    valid_fraction: float = 0.0
    best_offset_px: tuple[int, int] = (0, 0)

    @property
    def correction_norm_m(self) -> float:
        return math.hypot(float(self.dx_m), float(self.dy_m))


class ChamferMapMatcher:
    """Distance-transform matcher over map lane markings/boundaries."""

    def __init__(self, distance_field_px: np.ndarray, map_metadata: dict) -> None:
        field = np.asarray(distance_field_px, dtype=np.float32)
        if field.ndim != 2 or field.size == 0:
            raise ValueError("distance_field_px must be a non-empty 2D array")
        self.distance_field_px = field
        self.metadata = _normalize_metadata(map_metadata, image_shape=field.shape)
        self.height_px = int(field.shape[0])
        self.width_px = int(field.shape[1])

    @classmethod
    def from_binary_map(cls, binary_map: np.ndarray, map_metadata: dict) -> "ChamferMapMatcher":
        """Build from a binary map image where non-zero pixels are map lines."""
        cv2 = _require_cv2()
        binary = np.asarray(binary_map)
        if binary.ndim == 3:
            binary = binary[..., 0]
        binary = (binary > 0).astype(np.uint8) * 255
        inv = np.where(binary > 0, 0, 255).astype(np.uint8)
        distance = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        return cls(distance, map_metadata)

    @classmethod
    def from_lanelet_map(
        cls,
        lanelet_map,
        *,
        map_metadata: dict | None = None,
        line_width_px: int = 3,
    ) -> "ChamferMapMatcher":
        """Rasterize lanelet left/right boundaries and build a matcher."""
        cv2 = _require_cv2()
        metadata = _normalize_metadata(map_metadata or lanelet_map.get_map_metadata())
        width = int(metadata["image_width_px"])
        height = int(metadata["image_height_px"])
        canvas = np.zeros((height, width), dtype=np.uint8)
        line_width = max(1, int(line_width_px))

        for lanelet_id in tuple(getattr(lanelet_map, "lanelet_ids", ()) or ()):
            lanelet = lanelet_map.get_lanelet(str(lanelet_id))
            if lanelet is None:
                continue
            for boundary in (getattr(lanelet, "left_boundary", None), getattr(lanelet, "right_boundary", None)):
                pts = _world_points_to_pixels(boundary, metadata)
                if pts.shape[0] >= 2:
                    cv2.polylines(canvas, [pts.astype(np.int32)], False, 255, line_width)

        return cls.from_binary_map(canvas, metadata)

    @classmethod
    def from_raster_path(
        cls,
        raster_path: str,
        *,
        map_metadata: dict,
        threshold: int = 127,
    ) -> "ChamferMapMatcher":
        """Build from ``track.png``/``track.jpg`` as a fallback."""
        cv2 = _require_cv2()
        image = cv2.imread(str(raster_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(str(raster_path))
        metadata = _normalize_metadata(map_metadata, image_shape=image.shape)
        if not bool(metadata.get("y_axis_inverted", False)):
            image = cv2.flip(image, 0)
        _, binary = cv2.threshold(image, int(threshold), 255, cv2.THRESH_BINARY)
        return cls.from_binary_map(binary, metadata)

    def match_body_points(
        self,
        body_points: Iterable[Iterable[float]],
        *,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
        search_radius_m: float = 0.28,
        search_step_m: float = 0.02,
        max_points: int = 180,
        min_points: int = 24,
        min_forward_m: float = 0.03,
        max_forward_m: float = 1.20,
        max_lateral_abs_m: float = 0.65,
        max_cost_px: float = 9.0,
        min_improvement_px: float = 1.25,
        min_correction_m: float = 0.015,
    ) -> ChamferMatchResult:
        raw_points = () if body_points is None else tuple(body_points)
        body = _clean_body_points(
            raw_points,
            max_points=max_points,
            min_forward_m=min_forward_m,
            max_forward_m=max_forward_m,
            max_lateral_abs_m=max_lateral_abs_m,
        )
        if body.shape[0] < int(min_points):
            return ChamferMatchResult(
                applied=False,
                reason="not_enough_visual_points",
                input_point_count=int(len(raw_points)),
                point_count=int(body.shape[0]),
            )

        pixels = self._body_points_to_pixels(
            body,
            pose_x=float(pose_x),
            pose_y=float(pose_y),
            pose_yaw=float(pose_yaw),
        )
        baseline_cost, baseline_fraction = self._cost_at_offset(pixels, 0, 0)
        if not math.isfinite(baseline_cost):
            return ChamferMatchResult(
                applied=False,
                reason="baseline_out_of_map",
                baseline_cost_px=float(baseline_cost),
                input_point_count=int(len(raw_points)),
                point_count=int(body.shape[0]),
                valid_fraction=float(baseline_fraction),
            )

        radius_px_x = max(1, int(round(float(search_radius_m) / float(self.metadata["meters_per_pixel_x"]))))
        radius_px_y = max(1, int(round(float(search_radius_m) / float(self.metadata["meters_per_pixel_y"]))))
        step_px_x = max(1, int(round(float(search_step_m) / float(self.metadata["meters_per_pixel_x"]))))
        step_px_y = max(1, int(round(float(search_step_m) / float(self.metadata["meters_per_pixel_y"]))))

        offsets = np.asarray(
            [
                (int(dx), int(dy))
                for dy in _offset_range(radius_px_y, step_px_y)
                for dx in _offset_range(radius_px_x, step_px_x)
            ],
            dtype=np.int32,
        )
        costs, fractions = self._costs_at_offsets(pixels, offsets)
        best_cost = float(baseline_cost)
        best_fraction = float(baseline_fraction)
        best_dx = 0
        best_dy = 0
        finite_better = np.isfinite(costs) & (costs < (float(baseline_cost) - 1e-6))
        if np.any(finite_better):
            candidate_cost = float(np.min(costs[finite_better]))
            tied = finite_better & (np.abs(costs - candidate_cost) <= 1e-6)
            tied_indices = np.flatnonzero(tied)
            offset_norm_sq = (
                offsets[tied_indices, 0].astype(np.int64) * offsets[tied_indices, 0].astype(np.int64)
                + offsets[tied_indices, 1].astype(np.int64) * offsets[tied_indices, 1].astype(np.int64)
            )
            best_index = int(tied_indices[int(np.argmin(offset_norm_sq))])
            best_cost = float(costs[best_index])
            best_fraction = float(fractions[best_index])
            best_dx = int(offsets[best_index, 0])
            best_dy = int(offsets[best_index, 1])

        dx_m = float(best_dx) * float(self.metadata["meters_per_pixel_x"])
        dy_sign = -1.0 if bool(self.metadata.get("y_axis_inverted", False)) else 1.0
        dy_m = dy_sign * float(best_dy) * float(self.metadata["meters_per_pixel_y"])
        improvement = float(baseline_cost) - float(best_cost)
        correction_norm = math.hypot(dx_m, dy_m)

        common = dict(
            dx_m=float(dx_m),
            dy_m=float(dy_m),
            best_cost_px=float(best_cost),
            baseline_cost_px=float(baseline_cost),
            improvement_px=float(improvement),
            input_point_count=int(len(raw_points)),
            point_count=int(body.shape[0]),
            valid_fraction=float(best_fraction),
            best_offset_px=(int(best_dx), int(best_dy)),
        )
        if abs(int(best_dx)) >= int(radius_px_x) or abs(int(best_dy)) >= int(radius_px_y):
            return ChamferMatchResult(applied=False, reason="edge_match", **common)
        if best_cost > float(max_cost_px):
            return ChamferMatchResult(applied=False, reason="cost_too_high", **common)
        if improvement < float(min_improvement_px):
            return ChamferMatchResult(applied=False, reason="insufficient_improvement", **common)
        if correction_norm < float(min_correction_m):
            return ChamferMatchResult(applied=False, reason="already_aligned", **common)
        return ChamferMatchResult(applied=True, reason="matched", **common)

    def _body_points_to_pixels(
        self,
        body_points: np.ndarray,
        *,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
    ) -> np.ndarray:
        out = np.empty((body_points.shape[0], 2), dtype=float)
        for idx, (x_fwd, y_left) in enumerate(body_points):
            world = body_to_world_yflip(
                x_fwd_m=float(x_fwd),
                y_left_m=float(y_left),
                pose_x=float(pose_x),
                pose_y=float(pose_y),
                pose_yaw=float(pose_yaw),
            )
            out[idx, :] = _world_to_pixel(float(world[0]), float(world[1]), self.metadata)
        return out

    def _cost_at_offset(self, pixels: np.ndarray, dx_px: int, dy_px: int) -> tuple[float, float]:
        offsets = np.asarray([[int(dx_px), int(dy_px)]], dtype=np.int32)
        costs, fractions = self._costs_at_offsets(pixels, offsets)
        return float(costs[0]), float(fractions[0])

    def _costs_at_offsets(
        self,
        pixels: np.ndarray,
        offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        offset_arr = np.asarray(offsets, dtype=np.int32)
        if offset_arr.ndim != 2 or offset_arr.shape[0] <= 0 or offset_arr.shape[1] < 2:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        base_x = np.rint(pixels[:, 0]).astype(np.int32)
        base_y = np.rint(pixels[:, 1]).astype(np.int32)
        shifted_x = base_x[None, :] + offset_arr[:, 0:1]
        shifted_y = base_y[None, :] + offset_arr[:, 1:2]
        valid = (
            (shifted_x >= 0)
            & (shifted_x < self.width_px)
            & (shifted_y >= 0)
            & (shifted_y < self.height_px)
        )
        total = int(pixels.shape[0])
        if total <= 0:
            return (
                np.full((offset_arr.shape[0],), math.inf, dtype=np.float32),
                np.zeros((offset_arr.shape[0],), dtype=np.float32),
            )
        valid_count = np.count_nonzero(valid, axis=1).astype(np.float32)
        valid_fraction = valid_count / float(total)
        clipped_x = np.clip(shifted_x, 0, self.width_px - 1)
        clipped_y = np.clip(shifted_y, 0, self.height_px - 1)
        sampled_costs = self.distance_field_px[clipped_y, clipped_x]
        cost_sum = np.sum(np.where(valid, sampled_costs, 0.0), axis=1, dtype=np.float32)
        missing_penalty = float(max(self.width_px, self.height_px))
        mean_cost = np.full((offset_arr.shape[0],), math.inf, dtype=np.float32)
        usable = (valid_count > 0.0) & (valid_fraction >= 0.70)
        mean_cost[usable] = (
            cost_sum[usable] / valid_count[usable]
            + ((float(total) - valid_count[usable]) / float(total)) * missing_penalty
        )
        return mean_cost.astype(np.float32, copy=False), valid_fraction.astype(np.float32, copy=False)


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on deployment image
        raise RuntimeError("opencv-python/cv2 is required for chamfer matching") from exc
    return cv2


def _normalize_metadata(map_metadata: dict | None, *, image_shape: tuple[int, int] | None = None) -> dict:
    meta = dict(map_metadata or {})
    bounds = dict(meta.get("world_bounds") or {})
    x_min = float(bounds.get("x_min", 0.0))
    y_min = float(bounds.get("y_min", 0.0))
    mpp = float(meta.get("meters_per_pixel", meta.get("metersPerPixel", 0.01)) or 0.01)
    mpp_x = float(meta.get("meters_per_pixel_x", meta.get("metersPerPixelX", mpp)) or mpp)
    mpp_y = float(meta.get("meters_per_pixel_y", meta.get("metersPerPixelY", mpp)) or mpp)
    if image_shape is not None:
        height_px, width_px = int(image_shape[0]), int(image_shape[1])
    else:
        width_px = int(meta.get("image_width_px", meta.get("imgW", 0)) or 0)
        height_px = int(meta.get("image_height_px", meta.get("imgH", 0)) or 0)
        if width_px <= 0 or height_px <= 0:
            x_max = float(bounds.get("x_max", x_min + 1.0))
            y_max = float(bounds.get("y_max", y_min + 1.0))
            width_px = max(1, int(math.ceil((x_max - x_min) / max(mpp_x, 1e-6))))
            height_px = max(1, int(math.ceil((y_max - y_min) / max(mpp_y, 1e-6))))
    return {
        **meta,
        "world_bounds": {
            **bounds,
            "x_min": x_min,
            "y_min": y_min,
        },
        "meters_per_pixel_x": max(mpp_x, 1e-6),
        "meters_per_pixel_y": max(mpp_y, 1e-6),
        "image_width_px": width_px,
        "image_height_px": height_px,
        "y_axis_inverted": bool(meta.get("y_axis_inverted", False)),
    }


def _world_to_pixel(x_m: float, y_m: float, metadata: dict) -> tuple[float, float]:
    bounds = dict(metadata.get("world_bounds") or {})
    x_min = float(bounds.get("x_min", 0.0))
    y_min = float(bounds.get("y_min", 0.0))
    px = (float(x_m) - x_min) / float(metadata["meters_per_pixel_x"])
    local_y = (float(y_m) - y_min) / float(metadata["meters_per_pixel_y"])
    if bool(metadata.get("y_axis_inverted", False)):
        py = float(metadata["image_height_px"]) - local_y
    else:
        py = local_y
    return float(px), float(py)


def _world_points_to_pixels(points, metadata: dict) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.empty((0, 2), dtype=np.int32)
    out = []
    width = int(metadata["image_width_px"])
    height = int(metadata["image_height_px"])
    for x_m, y_m in arr[:, :2]:
        if not (math.isfinite(float(x_m)) and math.isfinite(float(y_m))):
            continue
        px, py = _world_to_pixel(float(x_m), float(y_m), metadata)
        if -width <= px <= width * 2 and -height <= py <= height * 2:
            out.append((int(round(px)), int(round(py))))
    return np.asarray(out, dtype=np.int32)


def _clean_body_points(
    points: Iterable[Iterable[float]],
    *,
    max_points: int,
    min_forward_m: float,
    max_forward_m: float,
    max_lateral_abs_m: float,
) -> np.ndarray:
    cleaned: list[tuple[float, float]] = []
    for item in points or ():
        try:
            x_fwd = float(item[0])
            y_left = float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (math.isfinite(x_fwd) and math.isfinite(y_left)):
            continue
        if x_fwd < float(min_forward_m) or x_fwd > float(max_forward_m):
            continue
        if abs(y_left) > float(max_lateral_abs_m):
            continue
        cleaned.append((x_fwd, y_left))
    if not cleaned:
        return np.empty((0, 2), dtype=float)
    if len(cleaned) > int(max_points):
        idx = np.linspace(0, len(cleaned) - 1, int(max_points), dtype=int)
        cleaned = [cleaned[int(i)] for i in idx]
    return np.asarray(cleaned, dtype=float)


def _offset_range(radius_px: int, step_px: int) -> tuple[int, ...]:
    radius = max(0, int(radius_px))
    step = max(1, int(step_px))
    values = list(range(-radius, radius + 1, step))
    if 0 not in values:
        values.append(0)
    if radius not in values:
        values.append(radius)
    if -radius not in values:
        values.append(-radius)
    return tuple(sorted(set(int(v) for v in values)))


__all__ = ["ChamferMapMatcher", "ChamferMatchResult"]
