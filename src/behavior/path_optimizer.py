from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.behavior.context import PlanningContext
from src.core.types.behavior import BehaviorPathPlan
from src.utils.live_log import live_log

try:
    from config import LANE_WIDTH_CM as _LANE_WIDTH_CM
except Exception:
    _LANE_WIDTH_CM = 35.0

_MIN_STEP_M = 0.05
_SMOOTH_WINDOW = 5
_PREVIOUS_BLEND_POINTS = 6
_DRIVABLE_HALF_WIDTH_M = max(0.10, float(_LANE_WIDTH_CM) / 200.0)
_BLEND_SAMPLE1_MAX_GAP_M = 0.10
_BLEND_SAMPLE1_MAX_HEADING_DELTA_DEG = 12.0


@dataclass(frozen=True)
class OptimizedPathResult:
    target_path: np.ndarray
    drivable_left_bound: np.ndarray
    drivable_right_bound: np.ndarray


@dataclass(frozen=True)
class _BlendSignature:
    scenario_name: str
    route_id: str | None
    current_lanelet_id: str | None
    first_next_lanelet_id: str | None


class PathOptimizer:
    """Convierte un path crudo en una trayectoria más estable para el MPC.

    Inspiración Autoware:
      - parte de un root/candidate path generado por el behavior path planner;
      - re-muestrea a una discretización temporal usable por el controlador;
      - suaviza la geometría para evitar cambios bruscos entre ciclos;
      - emite bounds laterales del corredor manejable.

    No hace evasión compleja ni optimización convexa todavía, pero deja la
    arquitectura lista para eso.
    """

    def __init__(self) -> None:
        self._prev_target_path: np.ndarray | None = None
        self._prev_blend_signature: _BlendSignature | None = None

    def optimize(
        self,
        path_plan: BehaviorPathPlan,
        ctx: PlanningContext,
    ) -> OptimizedPathResult:
        blend_signature = _build_blend_signature(path_plan, ctx)
        raw_path = _sanitize_raw_path(path_plan.raw_path, ctx)
        step_arc = _infer_step_arc(
            base_speed_profile=path_plan.base_speed_profile,
            horizon_n=ctx.horizon_n,
            dt=ctx.dt,
            fallback_speed_mps=ctx.nominal_speed_mps,
        )

        polyline = raw_path[:, :2]
        polyline = _smooth_polyline(polyline)
        sampled_xy = _resample_polyline(polyline, step_arc=step_arc, n_samples=ctx.horizon_n + 1)
        pose_xy = np.array([ctx.pose.fused_pose.x, ctx.pose.fused_pose.y], dtype=float)
        sampled_xy[0] = pose_xy
        blend_applied = False
        blend_reason = "no_previous_path"
        sample1_gap_m: float | None = None
        sample1_heading_delta_deg: float | None = None

        if self._prev_target_path is not None and self._prev_blend_signature is not None:
            if self._prev_blend_signature != blend_signature:
                blend_reason = "signature_changed"
            else:
                previous_xy = self._prev_target_path[:, :2]
                sample1_gap_m = _sample_gap_m(sampled_xy, previous_xy, idx=1)
                if sample1_gap_m is None:
                    blend_reason = "insufficient_samples"
                elif sample1_gap_m > _BLEND_SAMPLE1_MAX_GAP_M:
                    blend_reason = "sample1_gap"
                else:
                    sample1_heading_delta_deg = _sample_heading_delta_deg(sampled_xy, previous_xy)
                    if (
                        sample1_heading_delta_deg is not None
                        and sample1_heading_delta_deg > _BLEND_SAMPLE1_MAX_HEADING_DELTA_DEG
                    ):
                        blend_reason = "sample1_heading_delta"
                    else:
                        blend_reason = "applied"
                        blend_applied = True
        if blend_applied and self._prev_target_path is not None:
            sampled_xy = _blend_prefix_with_previous(
                current_xy=sampled_xy,
                previous_xy=self._prev_target_path[:, :2],
                blend_points=_PREVIOUS_BLEND_POINTS,
            )
            sampled_xy[0] = pose_xy

        live_log(
            "path_optimizer",
            event="blend_decision",
            applied=bool(blend_applied),
            reason=str(blend_reason),
            prev_scenario_name=(
                self._prev_blend_signature.scenario_name
                if self._prev_blend_signature is not None
                else None
            ),
            current_scenario_name=blend_signature.scenario_name,
            prev_route_id=(
                self._prev_blend_signature.route_id
                if self._prev_blend_signature is not None
                else None
            ),
            current_route_id=blend_signature.route_id,
            prev_lanelet_id=(
                self._prev_blend_signature.current_lanelet_id
                if self._prev_blend_signature is not None
                else None
            ),
            current_lanelet_id=blend_signature.current_lanelet_id,
            prev_next_lanelet_id=(
                self._prev_blend_signature.first_next_lanelet_id
                if self._prev_blend_signature is not None
                else None
            ),
            current_next_lanelet_id=blend_signature.first_next_lanelet_id,
            sample1_gap_m=(float(sample1_gap_m) if sample1_gap_m is not None else None),
            sample1_heading_delta_deg=(
                float(sample1_heading_delta_deg)
                if sample1_heading_delta_deg is not None
                else None
            ),
            path_points=int(sampled_xy.shape[0]),
        )

        headings = _compute_headings(sampled_xy, initial_yaw=float(ctx.pose.fused_pose.yaw))
        target_path = np.column_stack([sampled_xy, headings])
        left_bound, right_bound = _build_drivable_bounds(target_path, half_width_m=_DRIVABLE_HALF_WIDTH_M)

        self._prev_target_path = np.array(target_path, copy=True)
        self._prev_blend_signature = blend_signature
        return OptimizedPathResult(
            target_path=target_path,
            drivable_left_bound=left_bound,
            drivable_right_bound=right_bound,
        )


def _build_blend_signature(
    path_plan: BehaviorPathPlan,
    ctx: PlanningContext,
) -> _BlendSignature:
    next_lanelet_ids = tuple(str(item) for item in (ctx.route.next_lanelet_ids or ()) if str(item))
    return _BlendSignature(
        scenario_name=str(path_plan.scenario_name),
        route_id=str(ctx.route.route_id) if ctx.route.route_id is not None else None,
        current_lanelet_id=(
            str(ctx.route.current_lanelet_id)
            if ctx.route.current_lanelet_id is not None
            else None
        ),
        first_next_lanelet_id=next_lanelet_ids[0] if next_lanelet_ids else None,
    )


def _sanitize_raw_path(raw_path: np.ndarray, ctx: PlanningContext) -> np.ndarray:
    arr = np.asarray(raw_path, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        arr = np.array(
            [[ctx.pose.fused_pose.x, ctx.pose.fused_pose.y, ctx.pose.fused_pose.yaw]],
            dtype=float,
        )
    if arr.shape[1] < 2:
        arr = np.pad(arr, ((0, 0), (0, max(0, 2 - arr.shape[1]))))
    if arr.shape[1] < 3:
        arr = np.column_stack([arr[:, :2], np.zeros(arr.shape[0], dtype=float)])
    if arr.shape[0] == 1:
        arr = np.vstack([arr, arr])
    return arr[:, :3]


def _infer_step_arc(
    *,
    base_speed_profile: np.ndarray,
    horizon_n: int,
    dt: float,
    fallback_speed_mps: float,
) -> float:
    speed = np.asarray(base_speed_profile, dtype=float).reshape(-1)
    if speed.size == 0:
        ref_speed = float(fallback_speed_mps)
    else:
        positive = speed[speed > 1e-6]
        ref_speed = float(np.median(positive)) if positive.size else float(speed[0])
    return max(_MIN_STEP_M, abs(ref_speed) * float(dt), _MIN_STEP_M / max(1, min(horizon_n, 5)))


def _smooth_polyline(polyline: np.ndarray) -> np.ndarray:
    if polyline.shape[0] < 3:
        return np.array(polyline, copy=True)
    window = min(_SMOOTH_WINDOW, polyline.shape[0] if polyline.shape[0] % 2 == 1 else polyline.shape[0] - 1)
    if window < 3:
        return np.array(polyline, copy=True)
    radius = window // 2
    out = np.array(polyline, copy=True)
    for idx in range(radius, polyline.shape[0] - radius):
        segment = polyline[idx - radius : idx + radius + 1]
        out[idx] = np.mean(segment, axis=0)
    out[0] = polyline[0]
    out[-1] = polyline[-1]
    return out


def _resample_polyline(polyline: np.ndarray, *, step_arc: float, n_samples: int) -> np.ndarray:
    if polyline.shape[0] < 2:
        return np.tile(polyline[0], (n_samples, 1)) if polyline.shape[0] == 1 else np.zeros((n_samples, 2))

    seg_lens = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    samples = np.zeros((n_samples, 2), dtype=float)
    for k in range(n_samples):
        target_arc = float(step_arc) * k
        if target_arc >= total:
            samples[k] = polyline[-1]
            continue
        idx = int(np.searchsorted(cum, target_arc, side="right") - 1)
        idx = max(0, min(idx, polyline.shape[0] - 2))
        seg_len = float(seg_lens[idx])
        if seg_len <= 1e-9:
            samples[k] = polyline[idx]
            continue
        t = (target_arc - float(cum[idx])) / seg_len
        samples[k] = polyline[idx] + t * (polyline[idx + 1] - polyline[idx])
    return samples


def _blend_prefix_with_previous(
    *,
    current_xy: np.ndarray,
    previous_xy: np.ndarray,
    blend_points: int,
) -> np.ndarray:
    if current_xy.shape[0] < 2 or previous_xy.shape[0] < 2:
        return current_xy
    out = np.array(current_xy, copy=True)
    count = min(int(blend_points), current_xy.shape[0], previous_xy.shape[0])
    if count <= 1:
        return out
    for idx in range(1, count):
        alpha = float(idx) / float(count - 1)
        out[idx] = (alpha * current_xy[idx]) + ((1.0 - alpha) * previous_xy[idx])
    return out


def _sample_gap_m(current_xy: np.ndarray, previous_xy: np.ndarray, *, idx: int) -> float | None:
    if current_xy.shape[0] <= idx or previous_xy.shape[0] <= idx:
        return None
    return float(np.linalg.norm(current_xy[idx] - previous_xy[idx]))


def _sample_heading_delta_deg(current_xy: np.ndarray, previous_xy: np.ndarray) -> float | None:
    current_heading = _segment_heading(current_xy)
    previous_heading = _segment_heading(previous_xy)
    if current_heading is None or previous_heading is None:
        return None
    return math.degrees(abs(_wrap_angle(current_heading - previous_heading)))


def _segment_heading(samples_xy: np.ndarray) -> float | None:
    if samples_xy.shape[0] < 2:
        return None
    dx = float(samples_xy[1, 0] - samples_xy[0, 0])
    dy = float(samples_xy[1, 1] - samples_xy[0, 1])
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.atan2(dy, dx)


def _wrap_angle(angle_rad: float) -> float:
    out = float(angle_rad)
    while out > math.pi:
        out -= 2.0 * math.pi
    while out < -math.pi:
        out += 2.0 * math.pi
    return out


def _compute_headings(samples_xy: np.ndarray, *, initial_yaw: float) -> np.ndarray:
    n = int(samples_xy.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.array([float(initial_yaw)], dtype=float)

    psi = np.zeros(n, dtype=float)
    diffs = np.diff(samples_xy, axis=0)
    for idx in range(n - 1):
        dx = float(diffs[idx, 0])
        dy = float(diffs[idx, 1])
        if math.hypot(dx, dy) < 1e-6:
            psi[idx] = psi[idx - 1] if idx > 0 else float(initial_yaw)
        else:
            psi[idx] = math.atan2(dy, dx)
    psi[-1] = psi[-2]
    psi = _unwrap_angles(psi)
    # El estado k=0 del MPC coincide con la pose actual del auto, así que la
    # referencia de yaw en tp0 no debe "arrancar doblada". Si hacemos que el
    # primer psi apunte ya al centro del corredor, el controlador mete volante
    # a fondo aunque geométricamente todavía deba seguir derecho.
    psi[0] = float(initial_yaw)
    ramp_count = min(4, n)
    for idx in range(1, ramp_count):
        alpha = float(idx) / float(ramp_count)
        psi[idx] = _blend_angle(float(initial_yaw), float(psi[idx]), alpha=alpha)
    return _unwrap_angles(psi)


def _unwrap_angles(angles: np.ndarray) -> np.ndarray:
    if angles.size <= 1:
        return angles
    out = np.array(angles, copy=True)
    for idx in range(1, out.size):
        delta = float(out[idx] - out[idx - 1])
        while delta > math.pi:
            out[idx] -= 2.0 * math.pi
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            out[idx] += 2.0 * math.pi
            delta += 2.0 * math.pi
    return out


def _blend_angle(a: float, b: float, *, alpha: float) -> float:
    delta = float(b - a)
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return float(a + alpha * delta)


def _build_drivable_bounds(
    target_path: np.ndarray,
    *,
    half_width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if target_path.shape[0] == 0:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty
    left = np.zeros((target_path.shape[0], 2), dtype=float)
    right = np.zeros((target_path.shape[0], 2), dtype=float)
    for idx, (x, y, psi) in enumerate(target_path):
        nx = -math.sin(float(psi))
        ny = math.cos(float(psi))
        offset = float(half_width_m)
        left[idx] = np.array([float(x) + offset * nx, float(y) + offset * ny], dtype=float)
        right[idx] = np.array([float(x) - offset * nx, float(y) - offset * ny], dtype=float)
    return left, right
