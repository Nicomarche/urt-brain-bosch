from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.behavior.context import PlanningContext
from src.core.types.behavior import ScenarioName
from src.core.types.perception import lane_observation_has_visual_path
from src.routing.lanelet.attributes import (
    ATTR_CROSSWALK,
    ATTR_INTERSECTION,
    ATTR_INTERSECTION_EXIT,
    ATTR_ROUNDABOUT,
    ATTR_STOPLINE,
)


_MAP_AUTHORITY_ATTRS = frozenset({
    ATTR_CROSSWALK,
    ATTR_INTERSECTION,
    ATTR_INTERSECTION_EXIT,
    ATTR_ROUNDABOUT,
    ATTR_STOPLINE,
})
_MAP_AUTHORITY_TYPES = frozenset({
    "crosswalk",
    "intersection",
    "intersection_exit",
    "merge",
    "roundabout",
    "stop",
    "stopline",
    "traffic_light",
    "yield",
})
_MAP_AUTHORITY_MANEUVERS = frozenset({
    "intersection_straight",
    "roundabout",
    "stopline",
    "turn_left",
    "turn_right",
})


@dataclass(frozen=True)
class VisualPrimaryDecision:
    use_visual_path: bool
    reason: str
    map_authority_active: bool = False
    map_authority_reason: str = ""
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _GateConfig:
    enabled: bool
    allowed_scenarios: frozenset[str]
    two_line_min_quality: float
    single_line_min_quality: float
    min_points: int
    map_authority_distance_m: float
    min_forward_span_m: float
    max_lateral_to_forward_ratio: float
    max_heading_error_deg: float
    max_corridor_violation_m: float
    corridor_check_max_map_match_error_m: float
    enter_ticks: int
    exit_ticks: int


class VisualPrimaryGate:
    """Stateful tick gate for camera-primary lane keeping."""

    def __init__(self) -> None:
        self._active = False
        self._enter_ticks = 0
        self._exit_ticks = 0

    def decide(
        self,
        *,
        ctx: PlanningContext,
        lane_observation,
        scenario_name: str,
        route_corridor_available: bool,
        lanelet_corridor_available: bool,
    ) -> VisualPrimaryDecision:
        cfg = _load_config()
        raw_allowed, reason, hard_block, map_active, map_reason, notes = _evaluate_raw(
            cfg=cfg,
            ctx=ctx,
            lane_observation=lane_observation,
            scenario_name=scenario_name,
            route_corridor_available=route_corridor_available,
            lanelet_corridor_available=lanelet_corridor_available,
        )

        if hard_block:
            self._active = False
            self._enter_ticks = 0
            self._exit_ticks = 0
            notes.update({
                "visual_primary_gate_active": False,
                "visual_primary_enter_ticks": 0,
                "visual_primary_exit_ticks": 0,
                "visual_primary_hard_block": True,
            })
            return VisualPrimaryDecision(
                use_visual_path=False,
                reason=reason,
                map_authority_active=map_active,
                map_authority_reason=map_reason,
                notes=notes,
            )

        if raw_allowed:
            self._enter_ticks += 1
            self._exit_ticks = 0
            if self._active or self._enter_ticks >= cfg.enter_ticks:
                self._active = True
                reason = str(reason)
            else:
                reason = f"visual_primary_enter_hysteresis:{self._enter_ticks}/{cfg.enter_ticks}"
        else:
            self._enter_ticks = 0
            if self._active:
                self._exit_ticks += 1
                if self._exit_ticks >= cfg.exit_ticks:
                    self._active = False
                else:
                    reason = f"visual_primary_exit_hysteresis:{self._exit_ticks}/{cfg.exit_ticks}:{reason}"
            else:
                self._exit_ticks = 0

        notes.update({
            "visual_primary_gate_active": bool(self._active),
            "visual_primary_enter_ticks": int(self._enter_ticks),
            "visual_primary_exit_ticks": int(self._exit_ticks),
            "visual_primary_hard_block": False,
        })
        return VisualPrimaryDecision(
            use_visual_path=bool(self._active),
            reason=str(reason),
            map_authority_active=map_active,
            map_authority_reason=map_reason,
            notes=notes,
        )


def _load_config() -> _GateConfig:
    try:
        import config as _cfg
    except Exception:
        _cfg = None

    def _get(name: str, default):
        return getattr(_cfg, name, default) if _cfg is not None else default

    allowed = _get(
        "LANE_VISUAL_PRIMARY_ALLOWED_SCENARIOS",
        {ScenarioName.LANE_KEEP.value},
    )
    return _GateConfig(
        enabled=bool(_get("LANE_VISUAL_PRIMARY_ENABLED", True)),
        allowed_scenarios=frozenset(str(item) for item in (allowed or ())),
        two_line_min_quality=float(_get("LANE_VISUAL_PRIMARY_TWO_LINE_MIN_QUALITY", 0.75)),
        single_line_min_quality=float(_get("LANE_VISUAL_PRIMARY_SINGLE_LINE_MIN_QUALITY", 0.85)),
        min_points=int(_get("LANE_VISUAL_MIN_POLY_POINTS", 8)),
        map_authority_distance_m=float(_get("LANE_VISUAL_PRIMARY_MAP_AUTHORITY_DISTANCE_M", 0.90)),
        min_forward_span_m=float(_get("LANE_VISUAL_PRIMARY_MIN_FORWARD_SPAN_M", 0.25)),
        max_lateral_to_forward_ratio=float(_get("LANE_VISUAL_PRIMARY_MAX_LATERAL_TO_FORWARD_RATIO", 1.2)),
        max_heading_error_deg=float(_get("LANE_VISUAL_PRIMARY_MAX_HEADING_ERROR_DEG", 35.0)),
        max_corridor_violation_m=float(_get("LANE_VISUAL_PRIMARY_MAX_CORRIDOR_VIOLATION_M", 0.12)),
        corridor_check_max_map_match_error_m=float(
            _get("LANE_VISUAL_PRIMARY_CORRIDOR_CHECK_MAX_MAP_MATCH_ERROR_M", 0.12)
        ),
        enter_ticks=max(1, int(_get("LANE_VISUAL_PRIMARY_ENTER_TICKS", 3))),
        exit_ticks=max(1, int(_get("LANE_VISUAL_PRIMARY_EXIT_TICKS", 2))),
    )


def _evaluate_raw(
    *,
    cfg: _GateConfig,
    ctx: PlanningContext,
    lane_observation,
    scenario_name: str,
    route_corridor_available: bool,
    lanelet_corridor_available: bool,
) -> tuple[bool, str, bool, bool, str, dict[str, Any]]:
    notes: dict[str, Any] = {}
    map_active, map_reason = _map_authority_state(ctx, cfg.map_authority_distance_m)
    notes["visual_primary_map_authority_active"] = bool(map_active)
    if map_reason:
        notes["visual_primary_map_authority_reason"] = str(map_reason)

    if not cfg.enabled:
        return False, "visual_primary_disabled", False, map_active, map_reason, notes

    if str(scenario_name) not in cfg.allowed_scenarios:
        return False, f"scenario_not_allowed:{scenario_name}", True, map_active, map_reason, notes

    if map_active:
        return False, f"map_authority:{map_reason}", True, map_active, map_reason, notes

    mode = str(getattr(lane_observation, "measurement_mode", "none") or "none")
    if mode == "two_line":
        min_quality = cfg.two_line_min_quality
    elif mode == "single_line":
        min_quality = cfg.single_line_min_quality
    else:
        return False, f"unsupported_measurement_mode:{mode}", False, map_active, map_reason, notes

    if not lane_observation_has_visual_path(
        lane_observation,
        min_quality=min_quality,
        min_points=cfg.min_points,
    ):
        quality = float(getattr(lane_observation, "quality", 0.0) or 0.0)
        points = len(tuple(getattr(lane_observation, "center_waypoints_body", ()) or ()))
        notes["visual_primary_quality"] = quality
        notes["visual_primary_waypoint_count"] = int(points)
        if quality < min_quality:
            return False, f"low_quality:{mode}", False, map_active, map_reason, notes
        return False, "visual_path_gate_rejected", False, map_active, map_reason, notes

    geometry_ok, geometry_reason, geometry_notes = _visual_geometry_is_usable(
        ctx=ctx,
        lane_observation=lane_observation,
        cfg=cfg,
        route_corridor_available=route_corridor_available,
        lanelet_corridor_available=lanelet_corridor_available,
    )
    notes.update(geometry_notes)
    if not geometry_ok:
        return False, geometry_reason, True, map_active, map_reason, notes

    return True, f"{mode}_primary", False, map_active, map_reason, notes


def _map_authority_state(ctx: PlanningContext, distance_m: float) -> tuple[bool, str]:
    route = getattr(ctx, "route", None)
    if route is None:
        return False, ""

    for field_name in ("current_node_attr", "upcoming_node_attr"):
        try:
            attr = int(getattr(route, field_name, 0) or 0)
        except (TypeError, ValueError):
            attr = 0
        if attr in _MAP_AUTHORITY_ATTRS:
            return True, f"{field_name}:{attr}"

    for zone_type in tuple(getattr(route, "current_zone_types", ()) or ()):
        normalized = str(zone_type or "").strip().lower()
        if normalized in _MAP_AUTHORITY_TYPES:
            return True, f"current_zone:{normalized}"

    expected = str(getattr(route, "expected_control_type", "") or "").strip().lower()
    if expected in _MAP_AUTHORITY_TYPES:
        return True, f"expected_control_type:{expected}"

    maneuver = str(getattr(route, "maneuver_type", "") or "").strip().lower()
    if maneuver in _MAP_AUTHORITY_MANEUVERS:
        return True, f"maneuver_type:{maneuver}"

    next_type = str(getattr(route, "next_semantic_type", "") or "").strip().lower()
    if next_type in _MAP_AUTHORITY_TYPES:
        raw_distance = getattr(route, "next_semantic_distance_m", None)
        try:
            semantic_distance = float(raw_distance)
        except (TypeError, ValueError):
            semantic_distance = math.inf
        if math.isfinite(semantic_distance) and semantic_distance <= float(distance_m):
            return True, f"next_semantic:{next_type}@{semantic_distance:.2f}m"
        if not math.isfinite(semantic_distance):
            return True, f"next_semantic:{next_type}@unknown"

    return False, ""


def _visual_geometry_is_usable(
    *,
    ctx: PlanningContext,
    lane_observation,
    cfg: _GateConfig,
    route_corridor_available: bool,
    lanelet_corridor_available: bool,
) -> tuple[bool, str, dict[str, Any]]:
    waypoints = np.asarray(tuple(lane_observation.center_waypoints_body or ()), dtype=float)
    notes: dict[str, Any] = {}
    if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] < 2:
        return False, "visual_path_geometry_invalid", notes

    xy = waypoints[:, :2]
    forward_span = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
    lateral_span = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    ratio = abs(lateral_span) / max(abs(forward_span), 1e-6)
    notes.update({
        "visual_primary_forward_span_m": forward_span,
        "visual_primary_lateral_span_m": lateral_span,
        "visual_primary_lateral_to_forward_ratio": ratio,
    })

    if forward_span < cfg.min_forward_span_m:
        return False, "visual_path_insufficient_forward_span", notes
    if ratio > cfg.max_lateral_to_forward_ratio:
        return False, "visual_path_transverse", notes

    visual_heading = _heading_from_body_waypoints(xy)
    notes["visual_primary_heading_body_deg"] = math.degrees(visual_heading)

    if route_corridor_available:
        route_heading = _route_heading_in_body(ctx)
        if route_heading is not None:
            heading_error_deg = abs(math.degrees(_wrap_angle(visual_heading - route_heading)))
            notes["visual_primary_route_heading_body_deg"] = math.degrees(route_heading)
            notes["visual_primary_route_heading_error_deg"] = heading_error_deg
            if heading_error_deg > cfg.max_heading_error_deg:
                return False, "visual_path_route_heading_conflict", notes

    corridor_ok, corridor_reason, corridor_notes = _visual_corridor_is_compatible(
        ctx=ctx,
        body_xy=xy,
        cfg=cfg,
        route_corridor_available=route_corridor_available,
        lanelet_corridor_available=lanelet_corridor_available,
    )
    notes.update(corridor_notes)
    if not corridor_ok:
        return False, corridor_reason, notes

    return True, "geometry_ok", notes


def _visual_corridor_is_compatible(
    *,
    ctx: PlanningContext,
    body_xy: np.ndarray,
    cfg: _GateConfig,
    route_corridor_available: bool,
    lanelet_corridor_available: bool,
) -> tuple[bool, str, dict[str, Any]]:
    notes: dict[str, Any] = {
        "visual_primary_corridor_check_enabled": False,
    }
    if not (route_corridor_available or lanelet_corridor_available):
        notes["visual_primary_corridor_check_reason"] = "no_corridor_reference"
        return True, "corridor_check_skipped", notes

    route = getattr(ctx, "route", None)
    route_map_match_error_m = abs(float(getattr(route, "map_match_error_m", 0.0) or 0.0))
    notes["visual_primary_corridor_map_match_error_m"] = route_map_match_error_m
    notes["visual_primary_corridor_map_match_threshold_m"] = float(
        cfg.corridor_check_max_map_match_error_m
    )
    if route_map_match_error_m > cfg.corridor_check_max_map_match_error_m:
        notes["visual_primary_corridor_check_reason"] = "map_match_error_high"
        return True, "corridor_check_skipped", notes

    try:
        from src.behavior.path_optimizer import (
            _build_lanelet_corridor,
            _project_point_to_corridor,
            _safe_half_width_from_span,
        )
    except Exception as exc:
        notes["visual_primary_corridor_check_reason"] = "import_failed"
        notes["visual_primary_corridor_check_error"] = str(exc)
        return True, "corridor_check_skipped", notes

    try:
        corridor, corridor_build_notes = _build_lanelet_corridor(ctx)
    except Exception as exc:
        notes["visual_primary_corridor_check_reason"] = "build_failed"
        notes["visual_primary_corridor_check_error"] = str(exc)
        return True, "corridor_check_skipped", notes

    if corridor is None:
        notes["visual_primary_corridor_check_reason"] = str(
            corridor_build_notes.get("corridor_source", "missing_corridor")
        )
        if "corridor_lanelet_ids" in corridor_build_notes:
            notes["visual_primary_corridor_lanelet_ids"] = list(
                corridor_build_notes.get("corridor_lanelet_ids") or ()
            )
        return True, "corridor_check_skipped", notes

    world_xy = _body_waypoints_to_world_yflip(ctx, body_xy)
    max_abs_offset_m = 0.0
    max_violation_m = 0.0
    projected_points = 0
    missing_projection_points = 0
    first_violation_index: int | None = None
    for idx, point in enumerate(world_xy):
        projection = _project_point_to_corridor(corridor, point)
        if projection is None:
            missing_projection_points += 1
            continue

        projected_points += 1
        abs_offset_m = abs(float(projection.center_offset_m))
        safe_half_width_m = float(_safe_half_width_from_span(projection.width_m))
        violation_m = max(0.0, abs_offset_m - safe_half_width_m)
        max_abs_offset_m = max(max_abs_offset_m, abs_offset_m)
        if violation_m > max_violation_m:
            max_violation_m = violation_m
            first_violation_index = int(idx)

    notes.update({
        "visual_primary_corridor_check_enabled": True,
        "visual_primary_corridor_source": str(getattr(corridor, "source", "unknown")),
        "visual_primary_corridor_lanelet_ids": list(getattr(corridor, "lanelet_ids", ()) or ()),
        "visual_primary_corridor_projected_points": int(projected_points),
        "visual_primary_corridor_missing_projection_points": int(missing_projection_points),
        "visual_primary_corridor_max_abs_offset_m": float(max_abs_offset_m),
        "visual_primary_corridor_max_violation_m": float(max_violation_m),
        "visual_primary_corridor_violation_threshold_m": float(cfg.max_corridor_violation_m),
    })
    if first_violation_index is not None:
        notes["visual_primary_corridor_first_violation_index"] = int(first_violation_index)

    if projected_points <= 0:
        notes["visual_primary_corridor_check_reason"] = "no_projected_points"
        return True, "corridor_check_skipped", notes

    if max_violation_m > cfg.max_corridor_violation_m:
        notes["visual_primary_corridor_check_reason"] = "visual_path_outside_corridor"
        return False, "visual_path_corridor_conflict", notes

    notes["visual_primary_corridor_check_reason"] = "ok"
    return True, "corridor_check_ok", notes


def _body_waypoints_to_world_yflip(ctx: PlanningContext, body_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(body_xy, dtype=float)
    pose = ctx.pose.fused_pose
    pose_x = float(getattr(pose, "x", 0.0) or 0.0)
    pose_y = float(getattr(pose, "y", 0.0) or 0.0)
    pose_yaw = float(getattr(pose, "yaw", 0.0) or 0.0)
    cos_y = math.cos(pose_yaw)
    sin_y = math.sin(pose_yaw)
    world = np.empty((arr.shape[0], 2), dtype=float)
    world[:, 0] = pose_x + (cos_y * arr[:, 0]) + (sin_y * arr[:, 1])
    world[:, 1] = pose_y + (sin_y * arr[:, 0]) - (cos_y * arr[:, 1])
    return world


def _heading_from_body_waypoints(xy: np.ndarray) -> float:
    origin = xy[0]
    for idx in range(1, xy.shape[0]):
        delta = xy[idx] - origin
        if float(np.linalg.norm(delta)) > 1e-6:
            return math.atan2(float(delta[1]), float(delta[0]))
    return 0.0


def _route_heading_in_body(ctx: PlanningContext) -> float | None:
    route = getattr(ctx.route, "route_waypoints", None) or ()
    if len(route) < 2:
        return None

    try:
        idx0 = max(0, min(int(getattr(ctx.route, "matched_idx", 0) or 0), len(route) - 2))
        p0 = np.asarray(route[idx0][:2], dtype=float)
        p1 = None
        for idx in range(idx0 + 1, len(route)):
            candidate = np.asarray(route[idx][:2], dtype=float)
            if float(np.linalg.norm(candidate - p0)) > 1e-6:
                p1 = candidate
                break
        if p1 is None:
            return None
    except Exception:
        return None

    yaw = float(getattr(ctx.pose.fused_pose, "yaw", 0.0) or 0.0)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    x_fwd = (math.cos(yaw) * dx) + (math.sin(yaw) * dy)
    y_left = (math.sin(yaw) * dx) - (math.cos(yaw) * dy)
    if math.hypot(x_fwd, y_left) <= 1e-6:
        return None
    return math.atan2(y_left, x_fwd)


def _wrap_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad
