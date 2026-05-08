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
_STRAIGHT_HOLD_BRIDGE_MODE = "straight_hold"
_PRECISION_ROUTE_MIN_STEP_M = 0.005
_PRECISION_ROUTE_ARM_DISTANCE_M = 0.60
_PRECISION_ROUTE_MAX_MAP_MATCH_ERROR_M = 0.03
_PRECISION_ROUTE_STRAIGHT_WINDOW_M = 0.35
_PRECISION_ROUTE_STRAIGHT_HEADING_TOL_DEG = 4.0
_PRECISION_ROUTE_SEMANTIC_TYPES = {"intersection", "roundabout", "stopline"}
_PRECISION_ROUTE_MANEUVERS = {
    "turn_left",
    "turn_right",
    "intersection_straight",
    "roundabout",
    "stopline",
}


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
    bridge_mode: str | None


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
        ref_speed_mps = _reference_speed_from_profile(
            base_speed_profile=path_plan.base_speed_profile,
            fallback_speed_mps=ctx.nominal_speed_mps,
        )
        step_arc = _infer_step_arc(
            base_speed_profile=path_plan.base_speed_profile,
            horizon_n=ctx.horizon_n,
            dt=ctx.dt,
            fallback_speed_mps=ctx.nominal_speed_mps,
        )
        step_arc, precision_step_meta = _refine_step_arc_for_precision_route(
            step_arc=step_arc,
            ref_speed_mps=ref_speed_mps,
            raw_path=raw_path,
            ctx=ctx,
        )
        bridge_mode = _path_note_str(path_plan, "bridge_mode")
        protected_prefix_m = _path_note_float(path_plan, "protected_prefix_m")
        protected_prefix_samples = _protected_prefix_sample_count(
            protected_prefix_m=protected_prefix_m if bridge_mode == _STRAIGHT_HOLD_BRIDGE_MODE else 0.0,
            step_arc=step_arc,
            horizon_n=ctx.horizon_n,
        )
        live_log(
            "path_optimizer",
            event="step_arc_decision",
            step_arc_m=float(step_arc),
            base_step_arc_m=float(precision_step_meta["base_step_arc_m"]),
            ref_speed_mps=float(ref_speed_mps),
            precision_applied=bool(precision_step_meta["applied"]),
            precision_reason=str(precision_step_meta["reason"]),
            waypoint_mode_active=bool(precision_step_meta["waypoint_mode_active"]),
            next_semantic_type=precision_step_meta["next_semantic_type"],
            next_semantic_distance_m=precision_step_meta["next_semantic_distance_m"],
            map_match_error_m=precision_step_meta["map_match_error_m"],
            local_straight=precision_step_meta["local_straight"],
            precision_speed_cap_mps=precision_step_meta["precision_speed_cap_mps"],
        )

        polyline = raw_path[:, :2]
        protected_prefix_xy = None
        if protected_prefix_samples > 0:
            protected_prefix_xy = _resample_polyline(
                polyline,
                step_arc=step_arc,
                n_samples=protected_prefix_samples,
            )
        polyline = _smooth_polyline(polyline, preserve_count=protected_prefix_samples)
        sampled_xy = _resample_polyline(polyline, step_arc=step_arc, n_samples=ctx.horizon_n + 1)
        pose_xy = np.array([ctx.pose.fused_pose.x, ctx.pose.fused_pose.y], dtype=float)
        sampled_xy[0] = pose_xy
        if protected_prefix_xy is not None:
            sampled_xy[: protected_prefix_xy.shape[0]] = protected_prefix_xy
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
                preserve_points=protected_prefix_samples,
            )
            sampled_xy[0] = pose_xy
        if protected_prefix_xy is not None:
            sampled_xy[: protected_prefix_xy.shape[0]] = protected_prefix_xy
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
            bridge_mode=bridge_mode,
            protected_prefix_m=float(protected_prefix_m),
            protected_prefix_samples=int(protected_prefix_samples),
            path_points=int(sampled_xy.shape[0]),
        )

        headings = _compute_headings(sampled_xy, initial_yaw=float(ctx.pose.fused_pose.yaw))
        target_path = np.column_stack([sampled_xy, headings])
        heading_consistency = _heading_consistency_metrics(target_path, sample_count=3)
        live_log(
            "path_optimizer",
            event="heading_consistency",
            path_points=int(target_path.shape[0]),
            sample1_heading_vs_tangent_error_deg=heading_consistency["sample1_error_deg"],
            sample2_heading_vs_tangent_error_deg=heading_consistency["sample2_error_deg"],
            sample3_heading_vs_tangent_error_deg=heading_consistency["sample3_error_deg"],
            max_heading_vs_tangent_error_deg=heading_consistency["max_error_deg"],
        )
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
        bridge_mode=_path_note_str(path_plan, "bridge_mode"),
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
    ref_speed = _reference_speed_from_profile(
        base_speed_profile=base_speed_profile,
        fallback_speed_mps=fallback_speed_mps,
    )
    return max(_MIN_STEP_M, abs(ref_speed) * float(dt), _MIN_STEP_M / max(1, min(horizon_n, 5)))


def _reference_speed_from_profile(
    *,
    base_speed_profile: np.ndarray,
    fallback_speed_mps: float,
) -> float:
    speed = np.asarray(base_speed_profile, dtype=float).reshape(-1)
    if speed.size == 0:
        return float(fallback_speed_mps)
    positive = speed[speed > 1e-6]
    return float(np.median(positive)) if positive.size else float(speed[0])


def _refine_step_arc_for_precision_route(
    *,
    step_arc: float,
    ref_speed_mps: float,
    raw_path: np.ndarray,
    ctx: PlanningContext,
) -> tuple[float, dict[str, object]]:
    route = getattr(ctx, "route", None)
    meta: dict[str, object] = {
        "applied": False,
        "reason": "not_precision_context",
        "base_step_arc_m": float(step_arc),
        "waypoint_mode_active": bool(getattr(route, "waypoint_mode_active", False)) if route is not None else False,
        "next_semantic_type": str(getattr(route, "next_semantic_type", "") or "") if route is not None else "",
        "next_semantic_distance_m": (
            float(getattr(route, "next_semantic_distance_m", 0.0))
            if route is not None and getattr(route, "next_semantic_distance_m", None) is not None
            else None
        ),
        "map_match_error_m": float(getattr(route, "map_match_error_m", 0.0) or 0.0) if route is not None else 0.0,
        "local_straight": False,
        "precision_speed_cap_mps": None,
    }
    if route is None or not bool(getattr(route, "route_active", False)):
        meta["reason"] = "route_inactive"
        return float(step_arc), meta

    next_semantic_type = str(getattr(route, "next_semantic_type", "") or "")
    maneuver_type = str(getattr(route, "maneuver_type", "") or "")
    precision_context = bool(getattr(route, "waypoint_mode_active", False)) or (
        next_semantic_type in _PRECISION_ROUTE_SEMANTIC_TYPES
        or maneuver_type in _PRECISION_ROUTE_MANEUVERS
    )
    if not precision_context:
        meta["reason"] = "not_precision_context"
        return float(step_arc), meta

    next_distance = getattr(route, "next_semantic_distance_m", None)
    if next_distance is None:
        meta["reason"] = "no_semantic_distance"
        return float(step_arc), meta
    try:
        next_distance = max(0.0, float(next_distance))
    except (TypeError, ValueError):
        meta["reason"] = "invalid_semantic_distance"
        return float(step_arc), meta
    meta["next_semantic_distance_m"] = float(next_distance)

    map_match_error_m = abs(float(getattr(route, "map_match_error_m", 0.0) or 0.0))
    meta["map_match_error_m"] = float(map_match_error_m)
    if map_match_error_m > _PRECISION_ROUTE_MAX_MAP_MATCH_ERROR_M:
        meta["reason"] = "map_match_unreliable"
        return float(step_arc), meta

    local_straight = _raw_path_is_locally_straight(
        raw_path,
        window_m=min(_PRECISION_ROUTE_STRAIGHT_WINDOW_M, max(float(next_distance), float(step_arc))),
        heading_tol_deg=_PRECISION_ROUTE_STRAIGHT_HEADING_TOL_DEG,
    )
    meta["local_straight"] = bool(local_straight)
    if not local_straight:
        meta["reason"] = "local_curve_present"
        return float(step_arc), meta

    waypoint_mode_active = bool(getattr(route, "waypoint_mode_active", False))
    if not waypoint_mode_active and next_distance > _PRECISION_ROUTE_ARM_DISTANCE_M:
        meta["reason"] = "semantic_far"
        return float(step_arc), meta

    effective_ref_speed_mps = _precision_route_effective_speed_mps(
        ref_speed_mps=ref_speed_mps,
        ctx=ctx,
    )
    meta["precision_speed_cap_mps"] = float(effective_ref_speed_mps)
    desired_step_arc = max(_PRECISION_ROUTE_MIN_STEP_M, abs(float(effective_ref_speed_mps)) * float(ctx.dt))
    if desired_step_arc + 1e-9 >= float(step_arc):
        meta["reason"] = "already_time_scaled"
        return float(step_arc), meta

    meta["applied"] = True
    meta["reason"] = "precision_time_scaled"
    return float(desired_step_arc), meta


def _precision_route_effective_speed_mps(
    *,
    ref_speed_mps: float,
    ctx: PlanningContext,
) -> float:
    ref_speed = abs(float(ref_speed_mps))
    speed_caps: list[float] = []

    try:
        pose_speed = abs(float(getattr(ctx.pose, "speed_mps", 0.0) or 0.0))
    except (TypeError, ValueError):
        pose_speed = 0.0
    if pose_speed > 1e-6:
        speed_caps.append(pose_speed)

    try:
        nominal_speed = abs(float(getattr(ctx, "nominal_speed_mps", 0.0) or 0.0))
    except (TypeError, ValueError):
        nominal_speed = 0.0
    if nominal_speed > 1e-6:
        speed_caps.append(nominal_speed)

    if not speed_caps:
        return ref_speed
    return min(ref_speed, max(speed_caps))


def _path_note_str(path_plan: BehaviorPathPlan, key: str) -> str | None:
    notes = getattr(path_plan, "notes", None)
    if not isinstance(notes, dict):
        return None
    value = notes.get(key)
    if value is None:
        return None
    return str(value)


def _path_note_float(path_plan: BehaviorPathPlan, key: str) -> float:
    notes = getattr(path_plan, "notes", None)
    if not isinstance(notes, dict):
        return 0.0
    try:
        return float(notes.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _raw_path_is_locally_straight(
    raw_path: np.ndarray,
    *,
    window_m: float,
    heading_tol_deg: float,
) -> bool:
    arr = np.asarray(raw_path, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return True
    window_m = max(1e-6, float(window_m))
    heading_tol_rad = math.radians(max(0.1, float(heading_tol_deg)))
    samples_xy = np.asarray(arr[:, :2], dtype=float)
    start_heading = _tangent_heading_at(samples_xy, 0)
    if start_heading is None and arr.shape[1] >= 3:
        start_heading = float(arr[0, 2])
    if start_heading is None:
        return True
    max_delta = 0.0
    traveled = 0.0
    for idx in range(1, arr.shape[0]):
        step = float(np.linalg.norm(samples_xy[idx] - samples_xy[idx - 1]))
        traveled += step
        heading = _tangent_heading_at(samples_xy, idx)
        if heading is None and arr.shape[1] >= 3:
            heading = float(arr[idx, 2])
        if heading is None:
            if traveled >= window_m:
                break
            continue
        delta = abs(_wrap_angle(heading - start_heading))
        if delta > max_delta:
            max_delta = delta
        if traveled >= window_m:
            break
    return max_delta <= heading_tol_rad


def _protected_prefix_sample_count(
    *,
    protected_prefix_m: float,
    step_arc: float,
    horizon_n: int,
) -> int:
    if protected_prefix_m <= 1e-6:
        return 0
    count = int(math.ceil(float(protected_prefix_m) / max(float(step_arc), 1e-6)))
    count = max(1, count)
    return min(horizon_n + 1, count)


def _smooth_polyline(polyline: np.ndarray, *, preserve_count: int = 0) -> np.ndarray:
    if polyline.shape[0] < 3:
        return np.array(polyline, copy=True)
    window = min(_SMOOTH_WINDOW, polyline.shape[0] if polyline.shape[0] % 2 == 1 else polyline.shape[0] - 1)
    if window < 3:
        return np.array(polyline, copy=True)
    radius = window // 2
    out = np.array(polyline, copy=True)
    start_idx = max(radius, int(preserve_count))
    for idx in range(start_idx, polyline.shape[0] - radius):
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
    preserve_points: int = 0,
) -> np.ndarray:
    if current_xy.shape[0] < 2 or previous_xy.shape[0] < 2:
        return current_xy
    out = np.array(current_xy, copy=True)
    start_idx = max(1, int(preserve_points))
    end_idx = min(start_idx + int(blend_points), current_xy.shape[0], previous_xy.shape[0])
    if end_idx - start_idx <= 0:
        return out
    for idx in range(start_idx, end_idx):
        alpha = float(idx - start_idx + 1) / float(end_idx - start_idx + 1)
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

def _tangent_heading_at(samples_xy: np.ndarray, idx: int) -> float | None:
    n = int(samples_xy.shape[0])
    if n < 2 or idx < 0 or idx >= n:
        return None
    lo = max(0, int(idx) - 1)
    hi = min(n - 1, int(idx) + 1)
    if lo == hi:
        return None
    dx = float(samples_xy[hi, 0] - samples_xy[lo, 0])
    dy = float(samples_xy[hi, 1] - samples_xy[lo, 1])
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.atan2(dy, dx)


def _heading_consistency_metrics(
    target_path: np.ndarray,
    *,
    sample_count: int = 3,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "sample1_error_deg": None,
        "sample2_error_deg": None,
        "sample3_error_deg": None,
        "max_error_deg": None,
    }
    if target_path.shape[0] < 2:
        return metrics

    errors: list[float] = []
    samples_xy = target_path[:, :2]
    headings = target_path[:, 2]
    for offset in range(1, max(1, int(sample_count)) + 1):
        if offset >= target_path.shape[0]:
            break
        tangent = _tangent_heading_at(samples_xy, offset)
        if tangent is None:
            continue
        error_deg = math.degrees(abs(_wrap_angle(float(headings[offset]) - tangent)))
        metrics[f"sample{offset}_error_deg"] = float(error_deg)
        errors.append(float(error_deg))
    if errors:
        metrics["max_error_deg"] = float(max(errors))
    return metrics


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
