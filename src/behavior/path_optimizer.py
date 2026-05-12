from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.behavior.context import PlanningContext
from src.core.types.behavior import BehaviorPathPlan
from src.routing.lanelet.runtime_loader import (
    lanelet_map_covers_ids,
    load_tracking_lanelet_map,
)
from src.routing.lanelet.attributes import (
    ATTR_DOTTED,
    ATTR_HIGHWAY_LEFT,
    ATTR_HIGHWAY_RIGHT,
    ATTR_NORMAL,
    ATTR_ONEWAY,
)
from src.utils.live_log import live_log

try:
    from config import LANE_WIDTH_CM as _LANE_WIDTH_CM
except Exception:
    _LANE_WIDTH_CM = 35.0

try:
    from config import (
        BEHAVIOR_CONTAINMENT_CLEARANCE_M as _BEHAVIOR_CONTAINMENT_CLEARANCE_M,
    )
except Exception:
    _BEHAVIOR_CONTAINMENT_CLEARANCE_M = 0.01

try:
    from config import BEHAVIOR_VEHICLE_WIDTH_M as _BEHAVIOR_VEHICLE_WIDTH_M
except Exception:
    _BEHAVIOR_VEHICLE_WIDTH_M = 0.19

try:
    from config import (
        BEHAVIOR_ROUTE_VISUAL_REENTRY_ENABLED as _ROUTE_VISUAL_REENTRY_ENABLED,
    )
except Exception:
    _ROUTE_VISUAL_REENTRY_ENABLED = True

_MIN_STEP_M = 0.05
_SMOOTH_WINDOW = 5
_PREVIOUS_BLEND_POINTS = 6
_DRIVABLE_HALF_WIDTH_M = max(0.10, float(_LANE_WIDTH_CM) / 200.0)
_LANE_HALF_WIDTH_M = float(_LANE_WIDTH_CM) / 200.0
_VEHICLE_HALF_WIDTH_M = max(0.0, float(_BEHAVIOR_VEHICLE_WIDTH_M) * 0.5)
_SAFE_HALF_WIDTH_M = max(
    0.0,
    _LANE_HALF_WIDTH_M - _VEHICLE_HALF_WIDTH_M - float(_BEHAVIOR_CONTAINMENT_CLEARANCE_M),
)
_CONTAINMENT_INFEASIBLE_STOP_TICKS = 4
_BLEND_SAMPLE1_MAX_GAP_M = 0.10
_BLEND_SAMPLE1_MAX_HEADING_DELTA_DEG = 12.0
_VISUAL_BLEND_SAMPLE1_MAX_HEADING_DELTA_DEG = 70.0
_STRAIGHT_HOLD_BRIDGE_MODE = "straight_hold"
_PRECISION_ROUTE_MIN_STEP_M = 0.005
_PRECISION_ROUTE_WAYPOINT_MIN_STEP_M = 0.0125
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
_VISUAL_LANE_REENTRY_MIN_ERROR_M = 0.03
_VISUAL_LANE_REENTRY_TWO_LINE_MIN_QUALITY = 0.8
_VISUAL_LANE_REENTRY_GAIN = 0.60
_VISUAL_LANE_REENTRY_MAX_SHIFT_M = 0.10
_VISUAL_LANE_REENTRY_FADE_DISTANCE_M = 0.80
_VISUAL_LANE_REENTRY_SHIFT_DEADBAND_M = 0.004
_VISUAL_LANE_REENTRY_SHIFT_MAX_STEP_M = 0.020
_VISUAL_LANE_REENTRY_SHIFT_DECAY_STEP_M = 0.008
_VISUAL_LANE_REENTRY_SIGN_FLIP_CONFIRM_M = 0.060
_VISUAL_LANE_REENTRY_SIGN_FLIP_CONFIRM_TICKS = 3
_VISUAL_LANE_REENTRY_SIGN_FLIP_ZERO_HOLD_TICKS = 2
_VISUAL_LANE_REENTRY_DECAY_HOLD_TICKS = 12
_VISUAL_CONTROL_MEMORY_MAX_TICKS = 12
_VISUAL_CONTROL_MEMORY_MAX_SHIFT_M = 0.04
_VISUAL_LANE_REENTRY_SINGLE_LINE_MIN_QUALITY = 0.6
_VISUAL_LANE_REENTRY_SINGLE_LINE_GAIN = 0.40
_VISUAL_LANE_REENTRY_SINGLE_LINE_MAX_SHIFT_M = 0.07
_VISUAL_LANE_REENTRY_SINGLE_LINE_FADE_DISTANCE_M = 0.65
_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_MIN_QUALITY = 0.82
_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_GAIN = 0.38
_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_MAX_SHIFT_M = 0.065
_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_FADE_DISTANCE_M = 0.65
_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_MAX_HEADING_RAD = math.radians(28.0)
_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_MAX_ERROR_M = 0.16
_VISUAL_LANE_REENTRY_SINGLE_LINE_REACQUIRE_MIN_QUALITY = 0.40
_VISUAL_LANE_REENTRY_SINGLE_LINE_REACQUIRE_GAIN = 0.32
_VISUAL_LANE_REENTRY_SINGLE_LINE_REACQUIRE_MAX_SHIFT_M = 0.055
_VISUAL_LANE_REENTRY_SINGLE_LINE_REACQUIRE_FADE_DISTANCE_M = 0.55
_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_ATTRS = frozenset({
    ATTR_NORMAL,
    ATTR_ONEWAY,
    ATTR_HIGHWAY_LEFT,
    ATTR_HIGHWAY_RIGHT,
    ATTR_DOTTED,
})
_VISUAL_ROUTE_BLEND_SINGLE_LINE_WEIGHT = 0.18
_VISUAL_ROUTE_BLEND_SINGLE_LINE_MAX_SHIFT_M = 0.035
_VISUAL_ROUTE_BLEND_SINGLE_LINE_MAX_OBSERVATION_M = 0.16
_VISUAL_ROUTE_BLEND_SINGLE_LINE_MIN_SHIFT_M = 0.005
_VISUAL_LANE_REENTRY_RELAXED_MAP_MATCH_ERROR_M = 0.10
_VISUAL_PATH_PRIORITY_TWO_LINE_BLEND = 0.35
_VISUAL_PATH_PRIORITY_SINGLE_LINE_BLEND = 0.20
_VISUAL_PREFIX_STABILIZE_DISTANCE_M = 0.65
_VISUAL_PREFIX_MIN_FORWARD_M = 0.035
_VISUAL_PREFIX_EGO_START_TOL_M = 0.025
_VISUAL_PREFIX_FORWARD_PROGRESS_RATIO = 0.60
_VISUAL_PREFIX_MAX_LATERAL_RATIO = 0.75
_VISUAL_PREFIX_MIN_LATERAL_LIMIT_M = 0.05
_VISUAL_PREFIX_FORWARD_STEP_EPS_M = 0.015
_VISUAL_PREFIX_LATERAL_RAMP_DISTANCE_M = 0.65
_VISUAL_PREFIX_MAX_ENTRY_HEADING_DEG = 10.0
_VISUAL_PREFIX_TWO_LINE_MAX_HEADING_DEG = 10.0
# 25° en error alto (curva pronunciada). Era 5° (no doblaba) y 45° (oscilaba).
# 25° es el sweet spot: suficiente para doblar curvas BFMC pero sin
# producir saltos de heading que provocaban oscilación lateral.
_VISUAL_PREFIX_TWO_LINE_HIGH_ERROR_MAX_HEADING_DEG = 25.0
_VISUAL_PREFIX_TWO_LINE_RELAX_START_RATIO = 0.45
_VISUAL_PREFIX_TWO_LINE_RELAX_END_RATIO = 0.85
_ROUTE_PREFIX_STABILIZE_DISTANCE_M = 0.45
_ROUTE_PREFIX_MIN_FORWARD_M = 0.025
_ROUTE_PREFIX_FORWARD_STEP_EPS_M = 0.012
_ROUTE_PREFIX_MAX_ENTRY_HEADING_DEG = 25.0
_LINE_LOSS_UNTRUSTED_ROUTE_RELEASE_TICKS = 48
_LINE_LOSS_UNTRUSTED_ROUTE_RELEASE_ENTRY_HEADING_DEG = 8.0
_LINE_LOSS_UNTRUSTED_ROUTE_RELEASE_DISTANCE_M = 0.75
_SINGLE_LINE_VISUAL_PREFIX_MIN_HEADING_DEG = 8.0
# Hasta 25° en single_line curva pronunciada. 45° causaba oscilación.
_SINGLE_LINE_VISUAL_PREFIX_MAX_HEADING_DEG = 25.0
_SINGLE_LINE_VISUAL_ROUTE_OVERRIDE_MAP_MATCH_ERROR_M = max(0.15, _LANE_HALF_WIDTH_M + 0.02)
_SINGLE_LINE_VISUAL_ROUTE_OVERRIDE_EGO_CORRIDOR_ERROR_M = _LANE_HALF_WIDTH_M
_SINGLE_LINE_VISUAL_CORRIDOR_FLIP_MIN_VIOLATION_M = max(0.35, _LANE_HALF_WIDTH_M + 0.15)
_SINGLE_LINE_VISUAL_CORRIDOR_FLIP_MAX_EGO_ERROR_M = 0.10
_SINGLE_LINE_VISUAL_CORRIDOR_FLIP_MIN_HEADING_DEG = 6.0
_SINGLE_LINE_VISUAL_CORRIDOR_FLIP_MIN_DIRECT_ERROR_M = 0.01
_SINGLE_LINE_ROUTE_STRAIGHT_TOL_RAD = math.radians(6.0)
_SINGLE_LINE_ROUTE_DIRECTION_MARGIN_RAD = math.radians(10.0)
_SINGLE_LINE_ROUTE_MAX_YAW_MISMATCH_RAD = math.radians(18.0)
_SINGLE_LINE_PATH_ONLY_YAW_REJECT_MARKERS = (
    "single_line_visual_map_yaw_mismatch",
    "single_line_visual_map_sample_yaw_mismatch",
)
_ROUTE_TRACKING_VISUAL_HOLD_MAX_TICKS = 24
_ROUTE_TRACKING_VISUAL_HOLD_MIN_MAP_MATCH_ERROR_M = 0.18
_ROUTE_TRACKING_VISUAL_HOLD_MIN_FORWARD_M = 0.02
_ROUTE_TRACKING_VISUAL_HOLD_NEAR_FORWARD_M = 0.10
_ROUTE_TRACKING_VISUAL_HOLD_MAX_LATERAL_M = max(0.18, _LANE_HALF_WIDTH_M + 0.08)
_ROUTE_TRACKING_VISUAL_HOLD_HEADING_FLIP_MIN_MAP_MATCH_ERROR_M = 0.05
_ROUTE_TRACKING_VISUAL_HOLD_HEADING_FLIP_MAX_FORWARD_M = 0.08
_ROUTE_TRACKING_VISUAL_HOLD_HEADING_FLIP_MIN_HEADING_DEG = 5.0
_LINE_LOSS_STRAIGHT_HOLD_MAX_TICKS = 80
_LINE_LOSS_STRAIGHT_HOLD_UNTRUSTED_ROUTE_MAX_TICKS = 160
_LINE_LOSS_STRAIGHT_HOLD_MODES = frozenset({"single_line", "route_tracking", "blind", "none"})
_LINE_LOSS_STRAIGHT_HOLD_BLOCKING_SEMANTICS = frozenset({
    "intersection",
    "roundabout",
    "stopline",
    "crosswalk",
    "parking",
    "parking_spot",
    "traffic_light",
    "stop",
})
_LINE_LOSS_STRAIGHT_HOLD_BLOCK_DISTANCE_M = 0.45
try:
    from config import (
        BEHAVIOR_LINE_LOSS_STRAIGHT_HOLD_ROUTE_ESCAPE_MIN_TICKS as _CONFIG_LINE_LOSS_ROUTE_ESCAPE_MIN_TICKS,
    )
except Exception:
    _CONFIG_LINE_LOSS_ROUTE_ESCAPE_MIN_TICKS = 24

try:
    from config import (
        BEHAVIOR_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M as _CONFIG_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M,
    )
except Exception:
    _CONFIG_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M = 0.25

try:
    from config import (
        BEHAVIOR_LINE_LOSS_PREVIOUS_VISUAL_HOLD_ENABLED as _CONFIG_LINE_LOSS_PREVIOUS_VISUAL_HOLD_ENABLED,
    )
except Exception:
    _CONFIG_LINE_LOSS_PREVIOUS_VISUAL_HOLD_ENABLED = True

try:
    from config import (
        BEHAVIOR_VISUAL_TWO_LINE_ABSURD_CORRIDOR_VIOLATION_M as _CONFIG_VISUAL_TWO_LINE_ABSURD_CORRIDOR_VIOLATION_M,
    )
except Exception:
    _CONFIG_VISUAL_TWO_LINE_ABSURD_CORRIDOR_VIOLATION_M = 20.0

try:
    from config import (
        BEHAVIOR_VISUAL_ABSURD_CORRIDOR_VIOLATION_M as _CONFIG_VISUAL_ABSURD_CORRIDOR_VIOLATION_M,
    )
except Exception:
    _CONFIG_VISUAL_ABSURD_CORRIDOR_VIOLATION_M = 0.50

try:
    from config import (
        BEHAVIOR_VISUAL_PREVIOUS_PATH_MAX_SAMPLE1_GAP_M as _CONFIG_VISUAL_PREVIOUS_PATH_MAX_SAMPLE1_GAP_M,
    )
except Exception:
    _CONFIG_VISUAL_PREVIOUS_PATH_MAX_SAMPLE1_GAP_M = 0.08

_LINE_LOSS_STRAIGHT_HOLD_ROUTE_ESCAPE_MIN_TICKS = int(
    max(1, _CONFIG_LINE_LOSS_ROUTE_ESCAPE_MIN_TICKS)
)
_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M = float(
    max(0.0, _CONFIG_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M)
)
_LINE_LOSS_PREVIOUS_VISUAL_HOLD_ENABLED = bool(_CONFIG_LINE_LOSS_PREVIOUS_VISUAL_HOLD_ENABLED)
_VISUAL_TWO_LINE_ABSURD_CORRIDOR_VIOLATION_M = float(
    max(0.0, _CONFIG_VISUAL_TWO_LINE_ABSURD_CORRIDOR_VIOLATION_M)
)
_VISUAL_ABSURD_CORRIDOR_VIOLATION_M = float(
    max(0.0, _CONFIG_VISUAL_ABSURD_CORRIDOR_VIOLATION_M)
)
_VISUAL_PREVIOUS_PATH_MAX_SAMPLE1_GAP_M = float(
    max(0.0, _CONFIG_VISUAL_PREVIOUS_PATH_MAX_SAMPLE1_GAP_M)
)
_VISUAL_HOLD_PATH_SOURCES = frozenset({
    "visual_lane_waypoints",
    "route_tracking_visual_hold",
})
_LINE_LOSS_STRAIGHT_HOLD_ROUTE_HEADING_MAX_RAD = math.radians(10.0)
_LINE_LOSS_STRAIGHT_HOLD_ROUTE_PATH_HEADING_MAX_RAD = math.radians(11.0)
_LINE_LOSS_STRAIGHT_HOLD_ROUTE_KAPPA_MAX = 0.30
_LINE_LOSS_STRAIGHT_HOLD_ROUTE_LOOKAHEAD_MIN_M = 0.08
_SINGLE_LINE_TRANSITION_HOLD_MAX_TICKS = 6
_SINGLE_LINE_TRANSITION_HOLD_MAX_DIRECT_ERROR_M = 0.06
_SINGLE_LINE_TRANSITION_HOLD_MAX_SAMPLE1_GAP_M = 0.06
_SINGLE_LINE_TRANSITION_HOLD_MIN_HEADING_DELTA_DEG = 20.0
_SINGLE_LINE_TRANSITION_HOLD_RELEASE_HEADING_DELTA_DEG = 10.0


@dataclass(frozen=True)
class OptimizedPathResult:
    target_path: np.ndarray
    drivable_left_bound: np.ndarray
    drivable_right_bound: np.ndarray
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _CorridorGeometry:
    axis: np.ndarray
    left: np.ndarray
    right: np.ndarray
    lanelet_ids: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class _CorridorProjection:
    center_point: np.ndarray
    left_point: np.ndarray
    right_point: np.ndarray
    lateral_dir: np.ndarray
    center_offset_m: float
    width_m: float
    arc_m: float


@dataclass(frozen=True)
class _ContainmentResult:
    target_xy: np.ndarray
    left_bound: np.ndarray
    right_bound: np.ndarray
    corridor_available: bool
    infeasible: bool
    used_prev_safe_path: bool
    first_infeasible_index: int | None
    max_violation_m: float
    max_abs_offset_m: float
    ego_corridor_error_m: float
    touches_bound: bool
    mode: str
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _BlendSignature:
    scenario_name: str
    route_id: str | None
    current_lanelet_id: str | None
    first_next_lanelet_id: str | None
    bridge_mode: str | None


@dataclass(frozen=True)
class _RouteTrackingVisualHoldResult:
    target_path: np.ndarray
    left_bound: np.ndarray
    right_bound: np.ndarray
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _LineLossStraightHoldResult:
    target_path: np.ndarray
    left_bound: np.ndarray
    right_bound: np.ndarray
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SingleLineTransitionHoldResult:
    target_xy: np.ndarray
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _VisualLaneReentryState:
    active: bool
    error_m: float = 0.0
    quality: float = 0.0
    reason: str = "inactive"
    measurement_mode: str = "none"
    measurement_source: str = "none"
    control_source: str = "none"
    gain: float = 0.0
    max_shift_m: float = 0.0
    fade_distance_m: float = 0.0


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _visual_lane_reentry_base_shift_m(state: _VisualLaneReentryState) -> float:
    base_shift_m = float(np.clip(
        float(state.error_m) * float(state.gain),
        -float(state.max_shift_m),
        float(state.max_shift_m),
    ))
    quality_scale = max(0.0, min(1.0, float(state.quality)))
    return float(base_shift_m * quality_scale)


def _visual_lane_shift_sign(value_m: float) -> int:
    if float(value_m) > float(_VISUAL_LANE_REENTRY_SHIFT_DEADBAND_M):
        return 1
    if float(value_m) < -float(_VISUAL_LANE_REENTRY_SHIFT_DEADBAND_M):
        return -1
    return 0


def _visual_control_memory_allowed(ctx: PlanningContext, *, path_source: str) -> bool:
    if str(path_source or "") != "route_waypoints":
        return False
    lane_observation = getattr(ctx, "lane_observation", None)
    if lane_observation is None:
        return False
    measurement_mode = str(getattr(lane_observation, "measurement_mode", "none") or "none")
    if measurement_mode not in {"blind", "none", "route_tracking"}:
        return False
    return _route_semantics_allow_line_loss_straight_hold(ctx)


def _route_context_allows_derived_single_line_reentry(ctx: PlanningContext) -> bool:
    route = getattr(ctx, "route", None)
    if route is None or not bool(getattr(route, "route_active", False)):
        return False
    lanelet_map = getattr(ctx, "lanelet_map", None)
    lanelet_id = getattr(route, "current_lanelet_id", None)
    if lanelet_map is not None and lanelet_id:
        try:
            lanelet = lanelet_map.get_lanelet(str(lanelet_id))
        except Exception:
            lanelet = None
        if lanelet is not None:
            return int(getattr(lanelet, "attribute", -1)) in _VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_ATTRS
    try:
        return int(getattr(route, "current_node_attr", -1)) in _VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_ATTRS
    except (TypeError, ValueError):
        return False


def _single_line_visual_waypoint_error_m(ctx: PlanningContext, lane_observation) -> float | None:
    if lane_observation is None:
        return None
    if str(getattr(lane_observation, "measurement_mode", "none") or "none") != "single_line":
        return None
    if float(getattr(lane_observation, "quality", 0.0) or 0.0) < float(
        _VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_MIN_QUALITY
    ):
        return None
    if not bool(getattr(getattr(ctx, "route", None), "route_active", False)):
        return None
    if not _route_context_allows_derived_single_line_reentry(ctx):
        return None
    try:
        heading_error_rad = abs(float(getattr(lane_observation, "heading_error_rad", 0.0) or 0.0))
    except (TypeError, ValueError):
        return None
    if heading_error_rad > float(_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_MAX_HEADING_RAD):
        return None

    waypoints = tuple(getattr(lane_observation, "center_waypoints_body", ()) or ())
    if len(waypoints) < 3:
        return None

    lateral_samples: list[float] = []
    for x_fwd_m, y_left_m, _psi_rad in waypoints[: min(len(waypoints), 10)]:
        try:
            x_fwd = float(x_fwd_m)
            y_left = float(y_left_m)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x_fwd) and math.isfinite(y_left)):
            continue
        if x_fwd < 0.02 or x_fwd > 0.45:
            continue
        lateral_samples.append(y_left)
    if not lateral_samples:
        return None

    lateral_samples.sort()
    mid = len(lateral_samples) // 2
    if len(lateral_samples) % 2:
        center_y_left_m = float(lateral_samples[mid])
    else:
        center_y_left_m = 0.5 * (float(lateral_samples[mid - 1]) + float(lateral_samples[mid]))
    error_m = -center_y_left_m
    if not math.isfinite(error_m):
        return None
    if abs(error_m) > float(_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_MAX_ERROR_M):
        return None
    return float(error_m)


def _single_line_path_only_yaw_rejected(path_plan: BehaviorPathPlan | None) -> bool:
    if path_plan is None:
        return False
    reason = str(_path_note_str(path_plan, "visual_path_primary_rejected_reason") or "")
    if not reason:
        return False
    return any(marker in reason for marker in _SINGLE_LINE_PATH_ONLY_YAW_REJECT_MARKERS)


def _single_line_has_usable_lateral_guidance(
    lane_observation,
    *,
    path_plan: BehaviorPathPlan | None = None,
) -> bool:
    if lane_observation is None:
        return False
    if str(getattr(lane_observation, "measurement_mode", "none") or "none") != "single_line":
        return False
    try:
        quality = float(getattr(lane_observation, "quality", 0.0) or 0.0)
    except (TypeError, ValueError):
        quality = 0.0
    debug = getattr(lane_observation, "debug", None)
    control_source = (
        str(debug.get("control_lateral_error_source") or "")
        if isinstance(debug, dict)
        else ""
    )
    yaw_rejected = _single_line_path_only_yaw_rejected(path_plan)
    if yaw_rejected and control_source not in {
        "single_line_boundary_recovery",
        "single_line_reacquire",
    }:
        return False
    min_quality = (
        float(_VISUAL_LANE_REENTRY_SINGLE_LINE_REACQUIRE_MIN_QUALITY)
        if control_source == "single_line_reacquire"
        else float(_VISUAL_LANE_REENTRY_SINGLE_LINE_MIN_QUALITY)
    )
    if quality < min_quality:
        return False

    if bool(getattr(lane_observation, "direct_error_valid", False)):
        for field_name in (
            "control_lateral_error_m",
            "line_center_offset_m",
            "direct_error_m",
            "lateral_offset_m",
        ):
            try:
                value = float(getattr(lane_observation, field_name, None))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return True

    if control_source in {
        "single_line_boundary_recovery",
        "single_line_physical",
        "single_line_reacquire",
        "single_line_visual_waypoint",
        "single_line_visual_waypoint_low_authority",
    }:
        value = _optional_float(getattr(lane_observation, "control_lateral_error_m", None))
        if value is not None:
            return True

    finite_waypoints = 0
    for waypoint in tuple(getattr(lane_observation, "center_waypoints_body", ()) or ()):
        if len(waypoint) < 2:
            continue
        try:
            x_fwd = float(waypoint[0])
            y_left = float(waypoint[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_fwd) and math.isfinite(y_left):
            finite_waypoints += 1
            if finite_waypoints >= 3:
                return True
    return False


def _lane_observation_lateral_error_mag_m(lane_observation) -> float | None:
    if lane_observation is None:
        return None
    fields = ("control_lateral_error_m", "line_center_offset_m", "direct_error_m", "lateral_offset_m")
    if not bool(getattr(lane_observation, "direct_error_valid", False)):
        fields = ("lateral_offset_m",)
    for field_name in fields:
        try:
            value = float(getattr(lane_observation, field_name, None))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return abs(value)
    return None


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
        self._prev_target_path_source: str | None = None
        self._prev_safe_target_path: np.ndarray | None = None
        self._prev_safe_left_bound: np.ndarray | None = None
        self._prev_safe_right_bound: np.ndarray | None = None
        self._prev_safe_blend_signature: _BlendSignature | None = None
        self._prev_safe_path_source: str | None = None
        self._containment_infeasible_ticks: int = 0
        self._route_tracking_visual_hold_ticks: int = 0
        self._line_loss_straight_hold_ticks: int = 0
        self._line_loss_untrusted_route_release_ticks: int = 0
        self._single_line_transition_hold_ticks: int = 0
        self._prev_lane_measurement_mode: str | None = None
        self._visual_lane_reentry_shift_m: float = 0.0
        self._visual_lane_reentry_decay_ticks: int = 0
        self._visual_lane_reentry_route_id: str | None = None
        self._visual_lane_reentry_anchor_sign: int = 0
        self._visual_lane_reentry_opposite_sign: int = 0
        self._visual_lane_reentry_opposite_ticks: int = 0
        self._visual_lane_reentry_zero_hold_ticks: int = 0
        self._visual_lane_control_memory_state: _VisualLaneReentryState | None = None
        self._visual_lane_control_memory_ticks: int = 0

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
        path_source = _path_note_str(path_plan, "path_source")
        path_source_visual = path_source == "visual_lane_waypoints"

        # === BYPASS estilo urt-ref ===
        # urt-ref hace: Dijkstra → spline densa → state_refs directo al MPC.
        # NO aplica smoothing/blending/containment/prefix_stabilization. El
        # path interpolado del mapa YA contiene la geometría correcta del
        # corredor (incluyendo curvas). Aplastar el path con heading caps
        # ALEJA al coche del corredor real.
        #
        # Cuando el path viene de route_waypoints (no visual), retornamos
        # el raw_path resampleado a horizon_n+1 sin más procesamiento. El
        # MPC penaliza la deviación naturalmente.
        # Import lazy para no romper si config no está disponible (tests).
        try:
            import config as _cfg_for_bypass
            _bypass_enabled = bool(getattr(_cfg_for_bypass, "BEHAVIOR_PATH_OPTIMIZER_BYPASS_ROUTE", False))
        except Exception:
            _bypass_enabled = False
        if (
            _bypass_enabled
            and path_source == "route_waypoints"
            and not path_source_visual
        ):
            step_arc_bypass = max(0.05, float(ref_speed_mps) * float(ctx.dt))
            sampled_xy = _resample_polyline(
                raw_path[:, :2],
                step_arc=step_arc_bypass,
                n_samples=ctx.horizon_n + 1,
            )
            # === CONTAINMENT OBLIGATORIO ===
            # El path NUNCA puede salir del lane corridor del mapa. Aplicamos
            # clamp al corredor del lanelet activo igual que el path_optimizer
            # full hace. Sin esto el path puede pasar por encima del lane line
            # si el raw_path original tiene curvas tangentes al borde.
            try:
                containment = _contain_path_within_corridor(
                    sampled_xy=sampled_xy,
                    path_plan=path_plan,
                    ctx=ctx,
                    blend_signature=blend_signature,
                    prev_safe_target_path=self._prev_safe_target_path,
                    prev_safe_left_bound=self._prev_safe_left_bound,
                    prev_safe_right_bound=self._prev_safe_right_bound,
                    prev_safe_path_source=self._prev_safe_path_source,
                    prev_safe_signature=self._prev_safe_blend_signature,
                )
                clamped_xy = containment.target_xy
                left_bound = containment.left_bound
                right_bound = containment.right_bound
                containment_mode = containment.mode
            except Exception as exc:
                # Si el clamp falla (sin lanelet map, etc.), retornar el path
                # sin clamp pero loguear el fallo.
                clamped_xy = sampled_xy
                half = float(_LANE_HALF_WIDTH_M)
                left_bound = sampled_xy + np.array([0.0, half])
                right_bound = sampled_xy + np.array([0.0, -half])
                containment_mode = f"clamp_failed:{exc}"

            # Inline heading calculation con el path YA clampado
            diffs = np.diff(clamped_xy, axis=0)
            psi = np.zeros(clamped_xy.shape[0], dtype=float)
            psi[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
            if psi.shape[0] >= 2:
                psi[-1] = psi[-2]
            target_xy = np.column_stack([clamped_xy, psi])
            live_log(
                "path_optimizer",
                event="bypass_route",
                path_source=path_source,
                step_arc_m=float(step_arc_bypass),
                n_samples=int(clamped_xy.shape[0]),
                containment_mode=str(containment_mode),
            )
            return OptimizedPathResult(
                target_path=target_xy,
                drivable_left_bound=left_bound,
                drivable_right_bound=right_bound,
                notes={
                    "path_source": path_source,
                    "optimizer_bypassed": True,
                    "containment_mode": str(containment_mode),
                },
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
            path_source=path_source,
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
            visual_lane_reentry_active=precision_step_meta["visual_lane_reentry_active"],
            visual_lane_error_m=precision_step_meta["visual_lane_error_m"],
            precision_map_match_limit_m=precision_step_meta["precision_map_match_limit_m"],
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
        # Cuando el path viene de waypoints visuales, sample[0] está en el
        # centro del carril (offset lateral del ego). Forzarlo a la pose
        # crearía una "L" entre sample[0] y sample[1] que el MPC interpreta
        # como giro brusco. En ese caso dejamos el path tal como vino y el
        # MPC penaliza la deviación naturalmente (paradigma urt-ref).
        if not path_source_visual:
            sampled_xy[0] = pose_xy
        if protected_prefix_xy is not None:
            sampled_xy[: protected_prefix_xy.shape[0]] = protected_prefix_xy
            if not path_source_visual:
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
                    max_heading_delta_deg = (
                        float(_VISUAL_BLEND_SAMPLE1_MAX_HEADING_DELTA_DEG)
                        if path_source_visual
                        else float(_BLEND_SAMPLE1_MAX_HEADING_DELTA_DEG)
                    )
                    if (
                        sample1_heading_delta_deg is not None
                        and sample1_heading_delta_deg > max_heading_delta_deg
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
            if not path_source_visual:
                sampled_xy[0] = pose_xy
        if protected_prefix_xy is not None:
            sampled_xy[: protected_prefix_xy.shape[0]] = protected_prefix_xy
            if not path_source_visual:
                sampled_xy[0] = pose_xy

        (
            visual_lane_state,
            visual_lane_shift_m,
            visual_lane_smoothing_notes,
        ) = self._smooth_visual_lane_reentry_shift(
            state=_visual_lane_reentry_state(ctx),
            path_source=_path_note_str(path_plan, "path_source"),
            blend_signature=blend_signature,
            ctx=ctx,
        )
        sampled_xy, visual_lane_notes = _apply_visual_lane_reentry_bias(
            sampled_xy=sampled_xy,
            path_plan=path_plan,
            ctx=ctx,
            state_override=visual_lane_state,
            base_shift_override_m=visual_lane_shift_m,
            smoothing_notes=visual_lane_smoothing_notes,
        )

        sampled_xy, visual_route_blend_notes = _maybe_blend_route_with_visual_waypoints(
            sampled_xy=sampled_xy,
            path_plan=path_plan,
            ctx=ctx,
        )

        sampled_xy, visual_prefix_notes = _stabilize_visual_path_prefix(
            sampled_xy=sampled_xy,
            path_plan=path_plan,
            ctx=ctx,
        )

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

        single_line_transition_hold = _maybe_hold_previous_path_on_single_line_transition(
            sampled_xy=sampled_xy,
            path_plan=path_plan,
            ctx=ctx,
            blend_signature=blend_signature,
            prev_blend_signature=self._prev_blend_signature,
            prev_lane_measurement_mode=self._prev_lane_measurement_mode,
            prev_target_path=self._prev_target_path,
            prev_safe_target_path=self._prev_safe_target_path,
            prev_safe_signature=self._prev_safe_blend_signature,
            hold_ticks=self._single_line_transition_hold_ticks,
        )
        if single_line_transition_hold is not None:
            sampled_xy = np.array(single_line_transition_hold.target_xy, copy=True)
            self._single_line_transition_hold_ticks += 1
            live_log(
                "path_optimizer",
                event="single_line_transition_hold",
                applied=True,
                hold_ticks=int(self._single_line_transition_hold_ticks),
                reason=str(
                    single_line_transition_hold.notes.get(
                        "single_line_transition_hold_reason",
                        "unknown",
                    )
                ),
                sample1_gap_m=single_line_transition_hold.notes.get("single_line_transition_hold_sample1_gap_m"),
                heading_delta_deg=single_line_transition_hold.notes.get(
                    "single_line_transition_hold_heading_delta_deg"
                ),
                direct_error_m=single_line_transition_hold.notes.get("single_line_transition_hold_direct_error_m"),
                previous_path_source=single_line_transition_hold.notes.get(
                    "single_line_transition_hold_previous_path_source"
                ),
            )
        else:
            self._single_line_transition_hold_ticks = 0

        visual_sampled_xy = np.array(sampled_xy, copy=True)
        containment = _contain_path_within_corridor(
            sampled_xy=sampled_xy,
            path_plan=path_plan,
            ctx=ctx,
            blend_signature=blend_signature,
            prev_safe_target_path=self._prev_safe_target_path,
            prev_safe_left_bound=self._prev_safe_left_bound,
            prev_safe_right_bound=self._prev_safe_right_bound,
            prev_safe_path_source=self._prev_safe_path_source,
            prev_safe_signature=self._prev_safe_blend_signature,
        )
        single_line_flip_override = _maybe_override_single_line_visual_corridor_flip(
            raw_visual_xy=visual_sampled_xy,
            containment=containment,
            path_plan=path_plan,
            ctx=ctx,
        )
        if single_line_flip_override is not None:
            containment = single_line_flip_override
        sampled_xy = containment.target_xy
        route_prefix_notes: dict[str, object] = {}
        previous_line_loss_hold_ticks = int(self._line_loss_straight_hold_ticks)
        line_loss_straight_hold = _maybe_hold_straight_on_line_loss(
            path_plan=path_plan,
            ctx=ctx,
            prev_lane_measurement_mode=self._prev_lane_measurement_mode,
            prev_target_path=self._prev_target_path,
            prev_target_path_source=self._prev_target_path_source,
            hold_ticks=self._line_loss_straight_hold_ticks,
            step_arc=step_arc,
        )
        route_tracking_visual_hold = None
        if line_loss_straight_hold is not None:
            target_path = np.array(line_loss_straight_hold.target_path, copy=True)
            left_bound = np.array(line_loss_straight_hold.left_bound, copy=True)
            right_bound = np.array(line_loss_straight_hold.right_bound, copy=True)
            self._line_loss_straight_hold_ticks += 1
            self._line_loss_untrusted_route_release_ticks = 0
            self._route_tracking_visual_hold_ticks = 0
            live_log(
                "path_optimizer",
                event="line_loss_straight_hold",
                applied=True,
                hold_ticks=int(self._line_loss_straight_hold_ticks),
                reason=str(line_loss_straight_hold.notes.get("line_loss_straight_hold_reason", "unknown")),
                measurement_mode=line_loss_straight_hold.notes.get("line_loss_measurement_mode"),
                previous_measurement_mode=line_loss_straight_hold.notes.get("line_loss_previous_measurement_mode"),
                step_arc_m=line_loss_straight_hold.notes.get("line_loss_straight_hold_step_arc_m"),
                hold_path_kind=line_loss_straight_hold.notes.get("line_loss_hold_path_kind"),
            )
        else:
            if _should_soft_release_untrusted_line_loss_route(
                path_plan=path_plan,
                ctx=ctx,
                previous_hold_ticks=previous_line_loss_hold_ticks,
            ):
                self._line_loss_untrusted_route_release_ticks = max(
                    int(self._line_loss_untrusted_route_release_ticks),
                    int(_LINE_LOSS_UNTRUSTED_ROUTE_RELEASE_TICKS),
                )
                live_log(
                    "path_optimizer",
                    event="line_loss_untrusted_route_release",
                    started=True,
                    release_ticks=int(self._line_loss_untrusted_route_release_ticks),
                    previous_hold_ticks=int(previous_line_loss_hold_ticks),
                )
            self._line_loss_straight_hold_ticks = 0
            route_tracking_visual_hold = _maybe_hold_previous_visual_path_on_route_tracking(
                sampled_xy=sampled_xy,
                path_plan=path_plan,
                ctx=ctx,
                prev_target_path=self._prev_target_path,
                prev_target_path_source=self._prev_target_path_source,
                prev_safe_target_path=self._prev_safe_target_path,
                prev_safe_left_bound=self._prev_safe_left_bound,
                prev_safe_right_bound=self._prev_safe_right_bound,
                prev_safe_path_source=self._prev_safe_path_source,
                hold_ticks=self._route_tracking_visual_hold_ticks,
            )
        if route_tracking_visual_hold is not None:
            target_path = np.array(route_tracking_visual_hold.target_path, copy=True)
            left_bound = np.array(route_tracking_visual_hold.left_bound, copy=True)
            right_bound = np.array(route_tracking_visual_hold.right_bound, copy=True)
            self._line_loss_untrusted_route_release_ticks = 0
            self._route_tracking_visual_hold_ticks += 1
            live_log(
                "path_optimizer",
                event="route_tracking_visual_hold",
                applied=True,
                hold_ticks=int(self._route_tracking_visual_hold_ticks),
                reason=str(route_tracking_visual_hold.notes.get("route_tracking_visual_hold_reason", "unknown")),
                sample1_x_fwd_m=route_tracking_visual_hold.notes.get("route_tracking_visual_hold_sample1_x_fwd_m"),
                sample1_y_left_m=route_tracking_visual_hold.notes.get("route_tracking_visual_hold_sample1_y_left_m"),
                map_match_error_m=route_tracking_visual_hold.notes.get("route_tracking_visual_hold_map_match_error_m"),
                previous_path_source=route_tracking_visual_hold.notes.get("route_tracking_visual_hold_previous_path_source"),
            )
        elif line_loss_straight_hold is None:
            self._route_tracking_visual_hold_ticks = 0
            route_release_ticks = int(self._line_loss_untrusted_route_release_ticks)
            sampled_xy, route_prefix_notes = _stabilize_route_path_prefix(
                sampled_xy=sampled_xy,
                path_plan=path_plan,
                ctx=ctx,
                line_loss_untrusted_release_ticks=route_release_ticks,
            )
            if route_release_ticks > 0:
                if _path_note_str(path_plan, "path_source") == "route_waypoints":
                    self._line_loss_untrusted_route_release_ticks = max(0, route_release_ticks - 1)
                else:
                    self._line_loss_untrusted_route_release_ticks = 0
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
        if (
            line_loss_straight_hold is None
            and route_tracking_visual_hold is None
            and containment.corridor_available
        ):
            left_bound = np.asarray(containment.left_bound, dtype=float)
            right_bound = np.asarray(containment.right_bound, dtype=float)
        elif line_loss_straight_hold is None and route_tracking_visual_hold is None:
            left_bound, right_bound = _build_drivable_bounds(target_path, half_width_m=_DRIVABLE_HALF_WIDTH_M)

        if containment.corridor_available:
            if containment.infeasible:
                self._containment_infeasible_ticks += 1
            else:
                self._containment_infeasible_ticks = 0
        else:
            self._containment_infeasible_ticks = 0

        optimizer_notes = dict(visual_lane_notes)
        optimizer_notes.update(dict(visual_route_blend_notes))
        optimizer_notes.update(dict(visual_prefix_notes))
        if single_line_transition_hold is not None:
            optimizer_notes.update(dict(single_line_transition_hold.notes))
        optimizer_notes.update(dict(route_prefix_notes))
        optimizer_notes.update(dict(containment.notes))
        if line_loss_straight_hold is not None:
            optimizer_notes.update(dict(line_loss_straight_hold.notes))
        if route_tracking_visual_hold is not None:
            optimizer_notes.update(dict(route_tracking_visual_hold.notes))
        optimizer_notes.setdefault("containment_mode", containment.mode)
        if line_loss_straight_hold is not None:
            optimizer_notes["containment_mode"] = "line_loss_straight_hold"
        if route_tracking_visual_hold is not None:
            optimizer_notes["containment_mode"] = "route_tracking_visual_hold"
        optimizer_notes["corridor_violation_m"] = float(containment.max_violation_m)
        optimizer_notes["corridor_left_bound"] = np.array(left_bound, copy=True)
        optimizer_notes["corridor_right_bound"] = np.array(right_bound, copy=True)
        optimizer_notes["ego_corridor_error_m"] = float(containment.ego_corridor_error_m)
        optimizer_notes["corridor_max_abs_offset_m"] = float(containment.max_abs_offset_m)
        optimizer_notes["corridor_touches_bound"] = bool(containment.touches_bound)
        optimizer_notes["used_prev_safe_path"] = bool(containment.used_prev_safe_path)
        optimizer_notes["containment_infeasible_ticks"] = int(self._containment_infeasible_ticks)
        optimizer_notes["containment_stop_after_ticks"] = _CONTAINMENT_INFEASIBLE_STOP_TICKS
        if containment.first_infeasible_index is not None:
            optimizer_notes["first_infeasible_index"] = int(containment.first_infeasible_index)

        live_log(
            "path_optimizer",
            event="containment",
            mode=str(containment.mode),
            corridor_available=bool(containment.corridor_available),
            corridor_violation_m=float(containment.max_violation_m),
            ego_corridor_error_m=float(containment.ego_corridor_error_m),
            corridor_max_abs_offset_m=float(containment.max_abs_offset_m),
            corridor_touches_bound=bool(containment.touches_bound),
            used_prev_safe_path=bool(containment.used_prev_safe_path),
            infeasible=bool(containment.infeasible),
            infeasible_ticks=int(self._containment_infeasible_ticks),
            first_infeasible_index=containment.first_infeasible_index,
        )

        self._prev_target_path = np.array(target_path, copy=True)
        self._prev_blend_signature = blend_signature
        if line_loss_straight_hold is not None:
            self._prev_target_path_source = str(
                line_loss_straight_hold.notes.get(
                    "line_loss_hold_path_source",
                    "line_loss_straight_hold",
                )
            )
        elif route_tracking_visual_hold is not None:
            self._prev_target_path_source = str(
                route_tracking_visual_hold.notes.get(
                    "route_tracking_visual_hold_previous_path_source",
                    "visual_lane_waypoints",
                )
            )
        else:
            self._prev_target_path_source = _path_note_str(path_plan, "path_source")
        if containment.corridor_available and not containment.infeasible:
            self._prev_safe_target_path = np.array(target_path, copy=True)
            self._prev_safe_left_bound = np.array(left_bound, copy=True)
            self._prev_safe_right_bound = np.array(right_bound, copy=True)
            self._prev_safe_blend_signature = blend_signature
            self._prev_safe_path_source = self._prev_target_path_source
        lane_observation = getattr(ctx, "lane_observation", None)
        self._prev_lane_measurement_mode = str(
            getattr(lane_observation, "measurement_mode", "none") or "none"
        )
        return OptimizedPathResult(
            target_path=target_path,
            drivable_left_bound=left_bound,
            drivable_right_bound=right_bound,
            notes=optimizer_notes,
        )

    def _smooth_visual_lane_reentry_shift(
        self,
        *,
        state: "_VisualLaneReentryState",
        path_source: str,
        blend_signature: "_BlendSignature",
        ctx: PlanningContext,
    ) -> tuple["_VisualLaneReentryState", float, dict[str, object]]:
        route_id = str(blend_signature.route_id or "")
        route_changed = route_id != str(self._visual_lane_reentry_route_id or "")
        if path_source != "route_waypoints" or route_changed:
            self._visual_lane_reentry_shift_m = 0.0
            self._visual_lane_reentry_decay_ticks = 0
            self._visual_lane_reentry_route_id = route_id
            self._visual_lane_reentry_anchor_sign = 0
            self._visual_lane_reentry_opposite_sign = 0
            self._visual_lane_reentry_opposite_ticks = 0
            self._visual_lane_reentry_zero_hold_ticks = 0
            if route_changed:
                self._visual_lane_control_memory_state = None
                self._visual_lane_control_memory_ticks = 0

        previous_shift_m = float(self._visual_lane_reentry_shift_m)
        raw_shift_m = _visual_lane_reentry_base_shift_m(state) if state.active else 0.0
        target_shift_m = float(raw_shift_m)
        smoothing_reason = "raw"
        applying_state = state
        max_step_m = float(_VISUAL_LANE_REENTRY_SHIFT_MAX_STEP_M)
        previous_sign = _visual_lane_shift_sign(previous_shift_m)
        raw_sign = _visual_lane_shift_sign(raw_shift_m)
        if self._visual_lane_reentry_anchor_sign == 0 and previous_sign != 0:
            self._visual_lane_reentry_anchor_sign = int(previous_sign)

        visual_memory_active = False
        visual_memory_source = "none"
        visual_memory_mode = "none"
        visual_memory_ticks = int(self._visual_lane_control_memory_ticks)
        if state.active:
            self._visual_lane_control_memory_state = state
            self._visual_lane_control_memory_ticks = 0
            visual_memory_ticks = 0
        elif (
            str(state.measurement_mode) in {"two_line", "single_line"}
            and str(state.reason) == "below_activation_threshold"
        ):
            self._visual_lane_control_memory_state = None
            self._visual_lane_control_memory_ticks = 0
            visual_memory_ticks = 0

        if state.active:
            self._visual_lane_reentry_decay_ticks = 0
            anchor_sign = int(self._visual_lane_reentry_anchor_sign)
            if raw_sign == 0:
                target_shift_m = 0.0
                smoothing_reason = "raw_below_deadband"
                self._visual_lane_reentry_opposite_sign = 0
                self._visual_lane_reentry_opposite_ticks = 0
                self._visual_lane_reentry_zero_hold_ticks = 0
            elif anchor_sign == 0:
                self._visual_lane_reentry_anchor_sign = int(raw_sign)
                self._visual_lane_reentry_opposite_sign = 0
                self._visual_lane_reentry_opposite_ticks = 0
                self._visual_lane_reentry_zero_hold_ticks = 0
            elif raw_sign == anchor_sign:
                self._visual_lane_reentry_opposite_sign = 0
                self._visual_lane_reentry_opposite_ticks = 0
                self._visual_lane_reentry_zero_hold_ticks = 0
            else:
                if abs(raw_shift_m) >= float(_VISUAL_LANE_REENTRY_SIGN_FLIP_CONFIRM_M):
                    if self._visual_lane_reentry_opposite_sign == raw_sign:
                        self._visual_lane_reentry_opposite_ticks += 1
                    else:
                        self._visual_lane_reentry_opposite_sign = int(raw_sign)
                        self._visual_lane_reentry_opposite_ticks = 1
                        self._visual_lane_reentry_zero_hold_ticks = 0
                else:
                    self._visual_lane_reentry_opposite_sign = int(raw_sign)
                    self._visual_lane_reentry_opposite_ticks = 0
                    self._visual_lane_reentry_zero_hold_ticks = 0
                target_shift_m = 0.0
                if abs(raw_shift_m) < float(_VISUAL_LANE_REENTRY_SIGN_FLIP_CONFIRM_M):
                    smoothing_reason = "weak_sign_flip_to_zero"
                elif self._visual_lane_reentry_opposite_ticks < int(
                    _VISUAL_LANE_REENTRY_SIGN_FLIP_CONFIRM_TICKS
                ):
                    smoothing_reason = "opposite_sign_pending"
                elif abs(previous_shift_m) > float(_VISUAL_LANE_REENTRY_SHIFT_DEADBAND_M):
                    smoothing_reason = "opposite_sign_confirmed_decelerate"
                elif self._visual_lane_reentry_zero_hold_ticks < int(
                    _VISUAL_LANE_REENTRY_SIGN_FLIP_ZERO_HOLD_TICKS
                ):
                    self._visual_lane_reentry_zero_hold_ticks += 1
                    smoothing_reason = "opposite_sign_zero_hold"
                else:
                    self._visual_lane_reentry_anchor_sign = int(raw_sign)
                    self._visual_lane_reentry_opposite_sign = 0
                    self._visual_lane_reentry_opposite_ticks = 0
                    self._visual_lane_reentry_zero_hold_ticks = 0
                    target_shift_m = float(raw_shift_m)
                    smoothing_reason = "opposite_sign_confirmed"
        elif (
            path_source == "route_waypoints"
            and self._visual_lane_control_memory_state is not None
            and self._visual_lane_control_memory_ticks < int(_VISUAL_CONTROL_MEMORY_MAX_TICKS)
            and self._visual_lane_reentry_decay_ticks < int(_VISUAL_LANE_REENTRY_DECAY_HOLD_TICKS)
            and _visual_control_memory_allowed(ctx, path_source=path_source)
        ):
            memory_state = self._visual_lane_control_memory_state
            memory_ticks = int(self._visual_lane_control_memory_ticks)
            memory_max_shift_m = min(
                float(memory_state.max_shift_m),
                float(_VISUAL_CONTROL_MEMORY_MAX_SHIFT_M),
            )
            memory_shift_state = _VisualLaneReentryState(
                active=True,
                error_m=float(memory_state.error_m),
                quality=float(memory_state.quality),
                reason=str(memory_state.reason),
                measurement_mode=str(memory_state.measurement_mode),
                measurement_source=str(memory_state.measurement_source),
                control_source=str(memory_state.control_source),
                gain=float(memory_state.gain),
                max_shift_m=float(memory_max_shift_m),
                fade_distance_m=float(memory_state.fade_distance_m),
            )
            memory_fade = max(
                0.0,
                1.0 - (float(memory_ticks) / max(float(_VISUAL_CONTROL_MEMORY_MAX_TICKS), 1.0)),
            )
            raw_shift_m = float(_visual_lane_reentry_base_shift_m(memory_shift_state) * memory_fade)
            target_shift_m = float(raw_shift_m)
            smoothing_reason = "visual_control_memory"
            visual_memory_active = True
            visual_memory_ticks = int(memory_ticks) + 1
            visual_memory_source = str(memory_state.control_source or memory_state.measurement_source)
            visual_memory_mode = str(memory_state.measurement_mode)
            self._visual_lane_control_memory_ticks = int(memory_ticks) + 1
            self._visual_lane_reentry_decay_ticks += 1
            self._visual_lane_reentry_opposite_sign = 0
            self._visual_lane_reentry_opposite_ticks = 0
            self._visual_lane_reentry_zero_hold_ticks = 0
            applying_state = _VisualLaneReentryState(
                active=True,
                error_m=float(memory_state.error_m) * float(memory_fade),
                quality=float(memory_state.quality),
                reason="visual_control_memory",
                measurement_mode=str(memory_state.measurement_mode),
                measurement_source="visual_control_memory",
                control_source=str(memory_state.control_source or memory_state.measurement_source),
                gain=float(memory_state.gain),
                max_shift_m=float(memory_max_shift_m),
                fade_distance_m=float(memory_state.fade_distance_m),
            )
        elif (
            path_source == "route_waypoints"
            and abs(previous_shift_m) > float(_VISUAL_LANE_REENTRY_SHIFT_DEADBAND_M)
            and self._visual_lane_reentry_decay_ticks < int(_VISUAL_LANE_REENTRY_DECAY_HOLD_TICKS)
        ):
            self._visual_lane_reentry_decay_ticks += 1
            self._visual_lane_reentry_opposite_sign = 0
            self._visual_lane_reentry_opposite_ticks = 0
            self._visual_lane_reentry_zero_hold_ticks = 0
            target_shift_m = 0.0
            max_step_m = float(_VISUAL_LANE_REENTRY_SHIFT_DECAY_STEP_M)
            smoothing_reason = "measurement_lost_decay"
            applying_state = _VisualLaneReentryState(
                active=True,
                error_m=0.0,
                quality=1.0,
                reason="filtered_decay",
                measurement_mode="filtered",
                measurement_source="filtered_shift",
                gain=0.0,
                max_shift_m=0.0,
                fade_distance_m=float(_VISUAL_LANE_REENTRY_FADE_DISTANCE_M),
            )
        elif (
            path_source == "route_waypoints"
            and self._visual_lane_reentry_anchor_sign != 0
            and self._visual_lane_reentry_decay_ticks < int(_VISUAL_LANE_REENTRY_DECAY_HOLD_TICKS)
        ):
            self._visual_lane_reentry_decay_ticks += 1
            self._visual_lane_reentry_shift_m = 0.0
            self._visual_lane_reentry_opposite_sign = 0
            self._visual_lane_reentry_opposite_ticks = 0
            self._visual_lane_reentry_zero_hold_ticks = 0
            return state, 0.0, {
                "visual_lane_raw_shift_m": float(raw_shift_m),
                "visual_lane_filtered_shift_m": 0.0,
                "visual_lane_shift_smoothing": "measurement_lost_memory",
                "visual_lane_previous_shift_m": float(previous_shift_m),
                "visual_lane_anchor_sign": int(self._visual_lane_reentry_anchor_sign),
                "visual_lane_opposite_sign": int(self._visual_lane_reentry_opposite_sign),
                "visual_lane_opposite_ticks": int(self._visual_lane_reentry_opposite_ticks),
                "visual_lane_zero_hold_ticks": int(self._visual_lane_reentry_zero_hold_ticks),
                "visual_control_memory_active": False,
                "visual_control_memory_ticks": int(self._visual_lane_control_memory_ticks),
                "visual_control_memory_source": "none",
                "visual_control_memory_mode": "none",
            }
        else:
            self._visual_lane_reentry_shift_m = 0.0
            self._visual_lane_reentry_decay_ticks = 0
            self._visual_lane_reentry_anchor_sign = 0
            self._visual_lane_reentry_opposite_sign = 0
            self._visual_lane_reentry_opposite_ticks = 0
            self._visual_lane_reentry_zero_hold_ticks = 0
            return state, 0.0, {
                "visual_lane_raw_shift_m": float(raw_shift_m),
                "visual_lane_filtered_shift_m": 0.0,
                "visual_lane_shift_smoothing": "inactive",
                "visual_lane_previous_shift_m": float(previous_shift_m),
                "visual_lane_anchor_sign": 0,
                "visual_lane_opposite_sign": 0,
                "visual_lane_opposite_ticks": 0,
                "visual_lane_zero_hold_ticks": 0,
                "visual_control_memory_active": False,
                "visual_control_memory_ticks": int(self._visual_lane_control_memory_ticks),
                "visual_control_memory_source": "none",
                "visual_control_memory_mode": "none",
            }

        delta_m = float(target_shift_m) - float(previous_shift_m)
        delta_m = max(-max_step_m, min(max_step_m, delta_m))
        filtered_shift_m = float(previous_shift_m) + float(delta_m)
        if abs(filtered_shift_m) <= float(_VISUAL_LANE_REENTRY_SHIFT_DEADBAND_M):
            filtered_shift_m = 0.0

        filtered_sign = _visual_lane_shift_sign(filtered_shift_m)
        if self._visual_lane_reentry_anchor_sign == 0 and filtered_sign != 0:
            self._visual_lane_reentry_anchor_sign = int(filtered_sign)
        self._visual_lane_reentry_shift_m = float(filtered_shift_m)
        return applying_state, float(filtered_shift_m), {
            "visual_lane_raw_shift_m": float(raw_shift_m),
            "visual_lane_filtered_shift_m": float(filtered_shift_m),
            "visual_lane_shift_smoothing": str(smoothing_reason),
            "visual_lane_previous_shift_m": float(previous_shift_m),
            "visual_lane_anchor_sign": int(self._visual_lane_reentry_anchor_sign),
            "visual_lane_opposite_sign": int(self._visual_lane_reentry_opposite_sign),
            "visual_lane_opposite_ticks": int(self._visual_lane_reentry_opposite_ticks),
            "visual_lane_zero_hold_ticks": int(self._visual_lane_reentry_zero_hold_ticks),
            "visual_control_memory_active": bool(visual_memory_active),
            "visual_control_memory_ticks": int(visual_memory_ticks),
            "visual_control_memory_source": str(visual_memory_source),
            "visual_control_memory_mode": str(visual_memory_mode),
        }


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
    path_source: str | None = None,
) -> tuple[float, dict[str, object]]:
    route = getattr(ctx, "route", None)
    visual_lane_reentry = _visual_lane_reentry_state(ctx)
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
        "precision_map_match_limit_m": float(_PRECISION_ROUTE_MAX_MAP_MATCH_ERROR_M),
        "visual_lane_reentry_active": bool(visual_lane_reentry.active),
        "visual_lane_error_m": float(visual_lane_reentry.error_m),
        "visual_lane_reason": str(visual_lane_reentry.reason),
    }
    if route is None or not bool(getattr(route, "route_active", False)):
        meta["reason"] = "route_inactive"
        return float(step_arc), meta
    if str(path_source or "") == "visual_lane_waypoints":
        meta["reason"] = "visual_primary_path"
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
    waypoint_mode_active = bool(getattr(route, "waypoint_mode_active", False))
    allowed_map_match_error_m = float(_PRECISION_ROUTE_MAX_MAP_MATCH_ERROR_M)
    if waypoint_mode_active and visual_lane_reentry.active:
        allowed_map_match_error_m = max(
            allowed_map_match_error_m,
            float(_VISUAL_LANE_REENTRY_RELAXED_MAP_MATCH_ERROR_M),
        )
    meta["precision_map_match_limit_m"] = float(allowed_map_match_error_m)
    if map_match_error_m > allowed_map_match_error_m:
        meta["reason"] = "map_match_unreliable"
        return float(step_arc), meta

    local_straight = _raw_path_is_locally_straight(
        raw_path,
        window_m=min(_PRECISION_ROUTE_STRAIGHT_WINDOW_M, max(float(next_distance), float(step_arc))),
        heading_tol_deg=_PRECISION_ROUTE_STRAIGHT_HEADING_TOL_DEG,
    )
    meta["local_straight"] = bool(local_straight)
    if not local_straight and not waypoint_mode_active:
        meta["reason"] = "local_curve_present"
        return float(step_arc), meta

    if not waypoint_mode_active and next_distance > _PRECISION_ROUTE_ARM_DISTANCE_M:
        meta["reason"] = "semantic_far"
        return float(step_arc), meta

    effective_ref_speed_mps = _precision_route_effective_speed_mps(
        ref_speed_mps=ref_speed_mps,
        ctx=ctx,
    )
    meta["precision_speed_cap_mps"] = float(effective_ref_speed_mps)
    min_step_m = (
        _PRECISION_ROUTE_WAYPOINT_MIN_STEP_M
        if waypoint_mode_active
        else _PRECISION_ROUTE_MIN_STEP_M
    )
    desired_step_arc = max(min_step_m, abs(float(effective_ref_speed_mps)) * float(ctx.dt))
    if desired_step_arc + 1e-9 >= float(step_arc):
        meta["reason"] = "already_time_scaled"
        return float(step_arc), meta

    meta["applied"] = True
    meta["reason"] = (
        "precision_waypoint_mode"
        if waypoint_mode_active
        else "precision_time_scaled"
    )
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


def _visual_lane_reentry_state(ctx: PlanningContext) -> _VisualLaneReentryState:
    lane_observation = getattr(ctx, "lane_observation", None)
    if lane_observation is None:
        return _VisualLaneReentryState(active=False, reason="lane_observation_unusable")
    measurement_mode = str(getattr(lane_observation, "measurement_mode", "none") or "none")
    measurement_source = "direct_error_m"
    debug = getattr(lane_observation, "debug", None)
    control_source = (
        str(debug.get("control_lateral_error_source") or "none")
        if isinstance(debug, dict)
        else "none"
    )
    direct_error_valid = bool(getattr(lane_observation, "direct_error_valid", False))
    measurement = _optional_float(getattr(lane_observation, "control_lateral_error_m", None))
    if measurement is not None:
        measurement_source = "control_lateral_error_m"
    elif direct_error_valid:
        measurement = getattr(lane_observation, "line_center_offset_m", None)
        if measurement is not None:
            measurement_source = "line_center_offset_m"
        else:
            measurement = getattr(lane_observation, "direct_error_m", None)
            if measurement is not None:
                measurement_source = "direct_error_m"
    elif measurement_mode == "single_line":
        measurement = _single_line_visual_waypoint_error_m(ctx, lane_observation)
        if measurement is not None:
            measurement_source = "visual_waypoint_center"
    if measurement is None and direct_error_valid:
        measurement = getattr(lane_observation, "direct_error_m", None)
    if measurement is None:
        measurement = getattr(lane_observation, "lateral_offset_m", None)
        measurement_source = "lateral_offset_m"
    if measurement is None:
        return _VisualLaneReentryState(active=False, reason="no_lateral_measurement")
    try:
        error_m = float(measurement)
    except (TypeError, ValueError):
        return _VisualLaneReentryState(
            active=False,
            reason="invalid_lateral_measurement",
            measurement_mode=measurement_mode,
            measurement_source=measurement_source,
            control_source=control_source,
        )
    quality = float(getattr(lane_observation, "quality", 0.0) or 0.0)
    detected_sides = tuple(getattr(lane_observation, "detected_sides", ()) or ())
    boundary_recovery_active = bool(
        isinstance(debug, dict) and debug.get("single_line_boundary_recovery_active", False)
    )
    boundary_recovery_active = boundary_recovery_active or control_source == "single_line_boundary_recovery"
    reacquire_active = control_source == "single_line_reacquire"
    waypoint_control_source = control_source in {
        "single_line_visual_waypoint",
        "single_line_visual_waypoint_low_authority",
    }
    reason = "unsupported_measurement_mode"
    gain = 0.0
    max_shift_m = 0.0
    fade_distance_m = 0.0
    if measurement_mode == "two_line":
        if len(detected_sides) != 2:
            return _VisualLaneReentryState(
                active=False,
                error_m=float(error_m),
                quality=float(quality),
                reason="two_line_side_count_mismatch",
                measurement_mode=measurement_mode,
                measurement_source=measurement_source,
                control_source=control_source,
            )
        if quality < float(_VISUAL_LANE_REENTRY_TWO_LINE_MIN_QUALITY):
            return _VisualLaneReentryState(
                active=False,
                error_m=float(error_m),
                quality=float(quality),
                reason="two_line_low_quality",
                measurement_mode=measurement_mode,
                measurement_source=measurement_source,
                control_source=control_source,
            )
        reason = "two_line_reentry"
        gain = float(_VISUAL_LANE_REENTRY_GAIN)
        max_shift_m = float(_VISUAL_LANE_REENTRY_MAX_SHIFT_M)
        fade_distance_m = float(_VISUAL_LANE_REENTRY_FADE_DISTANCE_M)
    elif measurement_mode == "single_line":
        if len(detected_sides) != 1:
            return _VisualLaneReentryState(
                active=False,
                error_m=float(error_m),
                quality=float(quality),
                reason="single_line_side_count_mismatch",
                measurement_mode=measurement_mode,
                measurement_source=measurement_source,
                control_source=control_source,
            )
        min_quality = (
            float(_VISUAL_LANE_REENTRY_SINGLE_LINE_REACQUIRE_MIN_QUALITY)
            if reacquire_active
            else float(_VISUAL_LANE_REENTRY_SINGLE_LINE_MIN_QUALITY)
        )
        if quality < min_quality:
            return _VisualLaneReentryState(
                active=False,
                error_m=float(error_m),
                quality=float(quality),
                reason="single_line_low_quality",
                measurement_mode=measurement_mode,
                measurement_source=measurement_source,
                control_source=control_source,
            )
        if reacquire_active and not _route_semantics_allow_line_loss_straight_hold(ctx):
            return _VisualLaneReentryState(
                active=False,
                error_m=float(error_m),
                quality=float(quality),
                reason="single_line_reacquire_semantic_block",
                measurement_mode=measurement_mode,
                measurement_source=measurement_source,
                control_source=control_source,
            )
        if (
            measurement_source != "visual_waypoint_center"
            and not waypoint_control_source
            and not boundary_recovery_active
            and _single_line_direction_conflicts_with_route(ctx)
        ):
            return _VisualLaneReentryState(
                active=False,
                error_m=float(error_m),
                quality=float(quality),
                reason="single_line_direction_conflict",
                measurement_mode=measurement_mode,
                measurement_source=measurement_source,
                control_source=control_source,
            )
        if measurement_source == "visual_waypoint_center" or waypoint_control_source:
            reason = "single_line_visual_waypoint_reentry"
            gain = float(_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_GAIN)
            max_shift_m = float(_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_MAX_SHIFT_M)
            fade_distance_m = float(_VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_FADE_DISTANCE_M)
        elif reacquire_active:
            reason = "single_line_reacquire"
            gain = float(_VISUAL_LANE_REENTRY_SINGLE_LINE_REACQUIRE_GAIN)
            max_shift_m = float(_VISUAL_LANE_REENTRY_SINGLE_LINE_REACQUIRE_MAX_SHIFT_M)
            fade_distance_m = float(_VISUAL_LANE_REENTRY_SINGLE_LINE_REACQUIRE_FADE_DISTANCE_M)
        else:
            reason = "single_line_boundary_recovery" if boundary_recovery_active else "single_line_physical_reentry"
            gain = float(_VISUAL_LANE_REENTRY_SINGLE_LINE_GAIN)
            max_shift_m = float(_VISUAL_LANE_REENTRY_SINGLE_LINE_MAX_SHIFT_M)
            fade_distance_m = float(_VISUAL_LANE_REENTRY_SINGLE_LINE_FADE_DISTANCE_M)
    else:
        return _VisualLaneReentryState(
            active=False,
            error_m=float(error_m),
            quality=float(quality),
            reason=f"unsupported_measurement_mode:{measurement_mode}",
            measurement_mode=measurement_mode,
            measurement_source=measurement_source,
            control_source=control_source,
        )
    if abs(error_m) < float(_VISUAL_LANE_REENTRY_MIN_ERROR_M):
        return _VisualLaneReentryState(
            active=False,
            error_m=float(error_m),
            quality=float(quality),
            reason="below_activation_threshold",
            measurement_mode=measurement_mode,
            measurement_source=measurement_source,
            control_source=control_source,
        )
    return _VisualLaneReentryState(
        active=True,
        error_m=float(error_m),
        quality=float(quality),
        reason=reason,
        measurement_mode=measurement_mode,
        measurement_source=measurement_source,
        control_source=control_source,
        gain=gain,
        max_shift_m=max_shift_m,
        fade_distance_m=fade_distance_m,
    )


def _apply_visual_lane_reentry_bias(
    *,
    sampled_xy: np.ndarray,
    path_plan: BehaviorPathPlan,
    ctx: PlanningContext,
    state_override: _VisualLaneReentryState | None = None,
    base_shift_override_m: float | None = None,
    smoothing_notes: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    path_source = _path_note_str(path_plan, "path_source")
    route_corridor_authoritative = bool(
        path_source == "route_waypoints" and _route_corridor_is_authoritative(ctx)
    )
    if route_corridor_authoritative and not bool(_ROUTE_VISUAL_REENTRY_ENABLED):
        notes: dict[str, object] = {
            "visual_lane_reentry_active": False,
            "visual_lane_reentry_reason": "route_visual_reentry_disabled",
            "visual_lane_measurement_mode": "",
            "visual_lane_control_error_source": "none",
            "visual_lane_reentry_applied": False,
            "visual_lane_error_m": 0.0,
            "visual_lane_quality": 0.0,
            "visual_lane_shift_m": 0.0,
            "visual_lane_prefix_samples": 0,
            "route_corridor_authoritative": True,
        }
        live_log(
            "path_optimizer",
            event="visual_lane_reentry_bias",
            applied=False,
            reason="route_visual_reentry_disabled",
            error_m=0.0,
            quality=0.0,
            shift_m=0.0,
            prefix_samples=0,
            path_source=path_source,
            route_corridor_authoritative=True,
        )
        return np.array(sampled_xy, copy=True), notes
    if path_source == "visual_lane_waypoints":
        # La corrección visual ya está horneada en los waypoints — un sesgo
        # perpendicular adicional sería doble corrección (paradigma urt-ref).
        # Aun así propagamos el error/quality visual real: el MotionController
        # lo usa para abrir temporalmente el cap/rate de steering cuando el
        # coche ya está cerca del borde físico del carril.
        state = _visual_lane_reentry_state(ctx)
        notes: dict[str, object] = {
            "visual_lane_reentry_active": bool(state.active),
            "visual_lane_reentry_reason": "path_source_visual_waypoints",
            "visual_lane_measurement_mode": str(state.measurement_mode),
            "visual_lane_measurement_source": str(state.measurement_source),
            "visual_lane_control_error_source": str(state.control_source),
            "visual_lane_reentry_applied": False,
            "visual_lane_error_m": float(state.error_m),
            "visual_lane_quality": float(state.quality),
            "visual_lane_shift_m": 0.0,
            "visual_lane_prefix_samples": 0,
        }
        live_log(
            "path_optimizer",
            event="visual_lane_reentry_bias",
            applied=False,
            reason="path_source_visual_waypoints",
            error_m=float(state.error_m),
            quality=float(state.quality),
            shift_m=0.0,
            prefix_samples=0,
            path_source=path_source,
            measurement_source=str(state.measurement_source),
            control_source=str(state.control_source),
        )
        return np.array(sampled_xy, copy=True), notes
    state = state_override if state_override is not None else _visual_lane_reentry_state(ctx)
    notes: dict[str, object] = {
        "visual_lane_reentry_active": bool(state.active),
        "visual_lane_reentry_reason": str(state.reason),
        "visual_lane_measurement_mode": str(state.measurement_mode),
        "visual_lane_measurement_source": str(state.measurement_source),
        "visual_lane_control_error_source": str(state.control_source),
        "visual_lane_reentry_applied": False,
        "visual_lane_error_m": float(state.error_m),
        "visual_lane_quality": float(state.quality),
        "visual_lane_shift_m": 0.0,
        "visual_lane_prefix_samples": 0,
        "route_corridor_authoritative": bool(route_corridor_authoritative),
    }
    if smoothing_notes:
        notes.update(smoothing_notes)
    if sampled_xy.shape[0] <= 1 or not state.active:
        live_log(
            "path_optimizer",
            event="visual_lane_reentry_bias",
            applied=False,
            reason=str(state.reason),
            error_m=float(state.error_m),
            quality=float(state.quality),
            shift_m=0.0,
            prefix_samples=0,
            path_source=_path_note_str(path_plan, "path_source"),
            route_corridor_authoritative=bool(route_corridor_authoritative),
            measurement_source=str(state.measurement_source),
            control_source=str(state.control_source),
            **dict(smoothing_notes or {}),
        )
        return np.array(sampled_xy, copy=True), notes

    base_shift_m = (
        float(base_shift_override_m)
        if base_shift_override_m is not None
        else _visual_lane_reentry_base_shift_m(state)
    )
    if abs(base_shift_m) < 1e-6:
        notes["visual_lane_reentry_reason"] = "zero_shift_after_scaling"
        live_log(
            "path_optimizer",
            event="visual_lane_reentry_bias",
            applied=False,
            reason="zero_shift_after_scaling",
            error_m=float(state.error_m),
            quality=float(state.quality),
            shift_m=0.0,
            prefix_samples=0,
            path_source=_path_note_str(path_plan, "path_source"),
            route_corridor_authoritative=bool(route_corridor_authoritative),
            measurement_source=str(state.measurement_source),
            control_source=str(state.control_source),
            **dict(smoothing_notes or {}),
        )
        return np.array(sampled_xy, copy=True), notes

    biased_xy = np.array(sampled_xy, copy=True)
    traveled_m = 0.0
    prefix_samples = 0
    max_shift_m = 0.0
    for idx in range(1, sampled_xy.shape[0]):
        traveled_m += float(np.linalg.norm(sampled_xy[idx] - sampled_xy[idx - 1]))
        fade = max(
            0.0,
            1.0 - (traveled_m / max(float(state.fade_distance_m), 1e-6)),
        )
        if fade <= 1e-6:
            break
        path_psi = _tangent_heading_at(sampled_xy, idx)
        if path_psi is None:
            path_psi = _segment_heading(sampled_xy)
        if path_psi is None:
            path_psi = float(ctx.pose.fused_pose.yaw)
        shift_m = float(base_shift_m * fade)
        perp_psi = float(path_psi) + (math.pi * 0.5)
        dx = shift_m * math.cos(perp_psi)
        dy = shift_m * math.sin(perp_psi)
        biased_xy[idx] = biased_xy[idx] - np.array([dx, dy], dtype=float)
        prefix_samples = idx
        max_shift_m = max(max_shift_m, abs(shift_m))

    notes["visual_lane_shift_m"] = float(base_shift_m)
    notes["visual_lane_prefix_samples"] = int(prefix_samples)
    notes["visual_lane_reentry_applied"] = bool(prefix_samples > 0)
    live_log(
        "path_optimizer",
        event="visual_lane_reentry_bias",
        applied=bool(prefix_samples > 0),
        reason=str(state.reason),
        error_m=float(state.error_m),
        quality=float(state.quality),
        shift_m=float(base_shift_m),
        max_shift_m=float(max_shift_m),
        prefix_samples=int(prefix_samples),
        path_source=_path_note_str(path_plan, "path_source"),
        route_corridor_authoritative=bool(route_corridor_authoritative),
        measurement_source=str(state.measurement_source),
        control_source=str(state.control_source),
        **dict(smoothing_notes or {}),
    )
    return biased_xy, notes


def _route_corridor_is_authoritative(ctx: PlanningContext) -> bool:
    route = getattr(ctx, "route", None)
    if route is None:
        return False
    return bool(getattr(route, "route_active", False)) and bool(
        getattr(route, "route_waypoints", None)
    )


def _route_waypoint_visual_priority_allowed(ctx: PlanningContext) -> bool:
    route = getattr(ctx, "route", None)
    if route is None or not bool(getattr(route, "route_active", False)):
        return False
    if bool(getattr(route, "waypoint_mode_active", False)):
        return False
    next_semantic_type = str(getattr(route, "next_semantic_type", "") or "")
    if next_semantic_type in _PRECISION_ROUTE_SEMANTIC_TYPES:
        next_distance = getattr(route, "next_semantic_distance_m", None)
        if next_distance is None:
            return False
        try:
            if float(next_distance) <= float(_PRECISION_ROUTE_ARM_DISTANCE_M):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _maybe_blend_route_with_visual_waypoints(
    *,
    sampled_xy: np.ndarray,
    path_plan: BehaviorPathPlan,
    ctx: PlanningContext,
) -> tuple[np.ndarray, dict[str, object]]:
    notes: dict[str, object] = {
        "visual_route_blend_applied": False,
        "visual_route_blend_points": 0,
        "visual_route_blend_max_shift_m": 0.0,
        "visual_route_blend_reason": "inactive",
    }
    if _path_note_str(path_plan, "path_source") != "route_waypoints":
        notes["visual_route_blend_reason"] = "path_source_not_route"
        return np.array(sampled_xy, copy=True), notes

    lane_observation = getattr(ctx, "lane_observation", None)
    if lane_observation is None:
        notes["visual_route_blend_reason"] = "lane_observation_missing"
        return np.array(sampled_xy, copy=True), notes
    if str(getattr(lane_observation, "measurement_mode", "none") or "none") != "single_line":
        notes["visual_route_blend_reason"] = "measurement_not_single_line"
        return np.array(sampled_xy, copy=True), notes
    if bool(getattr(lane_observation, "direct_error_valid", False)):
        notes["visual_route_blend_reason"] = "physical_error_available"
        return np.array(sampled_xy, copy=True), notes
    if not _route_context_allows_derived_single_line_reentry(ctx):
        notes["visual_route_blend_reason"] = "route_attr_blocked"
        return np.array(sampled_xy, copy=True), notes
    if float(getattr(lane_observation, "quality", 0.0) or 0.0) < float(
        _VISUAL_LANE_REENTRY_DERIVED_SINGLE_LINE_MIN_QUALITY
    ):
        notes["visual_route_blend_reason"] = "low_quality"
        return np.array(sampled_xy, copy=True), notes

    waypoints_body = tuple(getattr(lane_observation, "center_waypoints_body", ()) or ())
    if len(waypoints_body) < 3 or sampled_xy.shape[0] < 3:
        notes["visual_route_blend_reason"] = "insufficient_points"
        return np.array(sampled_xy, copy=True), notes

    pose_x = float(ctx.pose.fused_pose.x)
    pose_y = float(ctx.pose.fused_pose.y)
    pose_yaw = float(ctx.pose.fused_pose.yaw)
    visual_points: list[np.ndarray] = []
    previous_x_fwd = -float("inf")
    for x_fwd_m, y_left_m, _psi_rad in waypoints_body:
        try:
            x_fwd = float(x_fwd_m)
            y_left = float(y_left_m)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x_fwd) and math.isfinite(y_left)):
            continue
        if x_fwd < 0.02:
            continue
        if x_fwd + 1e-4 < previous_x_fwd:
            continue
        previous_x_fwd = x_fwd
        visual_points.append(
            _body_to_world_xy(
                x_fwd_m=x_fwd,
                y_left_m=y_left,
                pose_x=pose_x,
                pose_y=pose_y,
                pose_yaw=pose_yaw,
            )
        )
    if len(visual_points) < 3:
        notes["visual_route_blend_reason"] = "insufficient_forward_points"
        return np.array(sampled_xy, copy=True), notes

    route_steps = [
        float(np.linalg.norm(sampled_xy[idx] - sampled_xy[idx - 1]))
        for idx in range(1, sampled_xy.shape[0])
    ]
    positive_steps = [step for step in route_steps if step > 1e-6]
    step_arc = float(np.median(positive_steps)) if positive_steps else _MIN_STEP_M
    visual_polyline = np.asarray(visual_points, dtype=float)
    visual_xy = _resample_polyline(
        visual_polyline,
        step_arc=max(_MIN_STEP_M, step_arc),
        n_samples=sampled_xy.shape[0],
    )

    blended = np.array(sampled_xy, copy=True)
    max_shift = 0.0
    points = 0
    for idx in range(1, min(blended.shape[0], visual_xy.shape[0])):
        route_heading = _tangent_heading_at(sampled_xy, idx)
        if route_heading is None:
            route_heading = float(ctx.pose.fused_pose.yaw)
        lateral_dir = np.array(
            [-math.sin(float(route_heading)), math.cos(float(route_heading))],
            dtype=float,
        )
        observed_shift_m = float(np.dot(visual_xy[idx] - sampled_xy[idx], lateral_dir))
        if not math.isfinite(observed_shift_m):
            continue
        observed_shift_m = float(np.clip(
            observed_shift_m,
            -float(_VISUAL_ROUTE_BLEND_SINGLE_LINE_MAX_OBSERVATION_M),
            float(_VISUAL_ROUTE_BLEND_SINGLE_LINE_MAX_OBSERVATION_M),
        ))
        shift_m = float(np.clip(
            observed_shift_m * float(_VISUAL_ROUTE_BLEND_SINGLE_LINE_WEIGHT),
            -float(_VISUAL_ROUTE_BLEND_SINGLE_LINE_MAX_SHIFT_M),
            float(_VISUAL_ROUTE_BLEND_SINGLE_LINE_MAX_SHIFT_M),
        ))
        if abs(shift_m) < float(_VISUAL_ROUTE_BLEND_SINGLE_LINE_MIN_SHIFT_M):
            continue
        blended[idx] = sampled_xy[idx] + (shift_m * lateral_dir)
        max_shift = max(max_shift, abs(shift_m))
        points = idx

    notes["visual_route_blend_applied"] = bool(points > 0)
    notes["visual_route_blend_points"] = int(points)
    notes["visual_route_blend_max_shift_m"] = float(max_shift)
    notes["visual_route_blend_reason"] = "single_line_route_lane"
    live_log(
        "path_optimizer",
        event="visual_route_blend",
        applied=bool(points > 0),
        points=int(points),
        max_shift_m=float(max_shift),
        reason=str(notes["visual_route_blend_reason"]),
        measurement_mode=str(getattr(lane_observation, "measurement_mode", "none") or "none"),
        quality=float(getattr(lane_observation, "quality", 0.0) or 0.0),
    )
    return blended, notes


def _stabilize_visual_path_prefix(
    *,
    sampled_xy: np.ndarray,
    path_plan: BehaviorPathPlan,
    ctx: PlanningContext,
) -> tuple[np.ndarray, dict[str, object]]:
    path_source = _path_note_str(path_plan, "path_source")
    lane_observation = getattr(ctx, "lane_observation", None)
    measurement_mode = str(getattr(lane_observation, "measurement_mode", "none") or "none")
    visual_heading_limit_deg = 0.0
    visual_heading_limit_rad = None
    single_line_heading_limit_deg = 0.0
    entry_heading_limit_deg = float(_VISUAL_PREFIX_MAX_ENTRY_HEADING_DEG)
    if measurement_mode == "two_line":
        error_mag_m = _lane_observation_lateral_error_mag_m(lane_observation)
        progress = 0.0
        if error_mag_m is not None:
            relax_start_m = float(_LANE_HALF_WIDTH_M) * float(_VISUAL_PREFIX_TWO_LINE_RELAX_START_RATIO)
            relax_end_m = max(
                relax_start_m + 1e-6,
                float(_LANE_HALF_WIDTH_M) * float(_VISUAL_PREFIX_TWO_LINE_RELAX_END_RATIO),
            )
            progress = min(
                1.0,
                max(0.0, (float(error_mag_m) - relax_start_m) / (relax_end_m - relax_start_m)),
            )
        visual_heading_limit_deg = float(
            _VISUAL_PREFIX_TWO_LINE_MAX_HEADING_DEG
            + (
                (
                    _VISUAL_PREFIX_TWO_LINE_HIGH_ERROR_MAX_HEADING_DEG
                    - _VISUAL_PREFIX_TWO_LINE_MAX_HEADING_DEG
                )
                * progress
            )
        )
        entry_heading_limit_deg = max(float(entry_heading_limit_deg), float(visual_heading_limit_deg))
        visual_heading_limit_rad = math.radians(visual_heading_limit_deg)
    elif measurement_mode == "single_line":
        error_mag_m = _lane_observation_lateral_error_mag_m(lane_observation)
        progress = 0.5
        if error_mag_m is not None and math.isfinite(float(error_mag_m)):
            progress = min(1.0, max(0.0, float(error_mag_m) / max(_LANE_HALF_WIDTH_M, 1e-6)))
        single_line_heading_limit_deg = float(
            _SINGLE_LINE_VISUAL_PREFIX_MIN_HEADING_DEG
            + (
                (_SINGLE_LINE_VISUAL_PREFIX_MAX_HEADING_DEG - _SINGLE_LINE_VISUAL_PREFIX_MIN_HEADING_DEG)
                * progress
            )
        )
        visual_heading_limit_deg = float(single_line_heading_limit_deg)
        if error_mag_m is not None and math.isfinite(float(error_mag_m)):
            entry_heading_limit_deg = max(float(entry_heading_limit_deg), float(visual_heading_limit_deg))
        visual_heading_limit_rad = math.radians(visual_heading_limit_deg)
    notes: dict[str, object] = {
        "visual_prefix_stabilization_applied": False,
        "visual_prefix_stabilization_points": 0,
        "visual_prefix_sample1_x_fwd_before_m": 0.0,
        "visual_prefix_sample1_y_left_before_m": 0.0,
        "visual_prefix_sample1_x_fwd_after_m": 0.0,
        "visual_prefix_sample1_y_left_after_m": 0.0,
        "visual_prefix_starts_at_ego": False,
        "visual_prefix_entry_heading_limit_deg": float(entry_heading_limit_deg),
        "visual_prefix_entry_heading_cap_applied": False,
        "visual_prefix_heading_limit_deg": float(visual_heading_limit_deg),
        "visual_prefix_heading_cap_applied": False,
        "visual_prefix_single_line_heading_limit_deg": float(single_line_heading_limit_deg),
        "visual_prefix_single_line_heading_cap_applied": False,
    }
    if path_source != "visual_lane_waypoints" or sampled_xy.shape[0] <= 1:
        return np.array(sampled_xy, copy=True), notes

    pose_x = float(ctx.pose.fused_pose.x)
    pose_y = float(ctx.pose.fused_pose.y)
    pose_yaw = float(ctx.pose.fused_pose.yaw)
    stabilized_xy = np.array(sampled_xy, copy=True)
    first_body_x, first_body_y = _world_to_body_xy(
        point_xy=stabilized_xy[0],
        pose_x=pose_x,
        pose_y=pose_y,
        pose_yaw=pose_yaw,
    )
    visual_starts_at_ego = bool(
        math.hypot(float(first_body_x), float(first_body_y)) <= float(_VISUAL_PREFIX_EGO_START_TOL_M)
        or float(first_body_x) <= float(_VISUAL_PREFIX_EGO_START_TOL_M)
    )
    cumulative_arc_m = 0.0
    stabilized_points = 0
    prev_body_x = 0.0 if visual_starts_at_ego else float(first_body_x)
    prev_body_y = 0.0 if visual_starts_at_ego else float(first_body_y)
    heading_cap_applied = False
    entry_heading_cap_applied = False
    sample1_before: tuple[float, float] | None = None
    sample1_after: tuple[float, float] | None = None
    start_idx = 1 if visual_starts_at_ego else 0

    for idx in range(start_idx, stabilized_xy.shape[0]):
        if idx == 0:
            cumulative_arc_m = math.hypot(float(first_body_x), float(first_body_y))
        else:
            cumulative_arc_m += float(np.linalg.norm(stabilized_xy[idx] - stabilized_xy[idx - 1]))
        body_x, body_y = _world_to_body_xy(
            point_xy=stabilized_xy[idx],
            pose_x=pose_x,
            pose_y=pose_y,
            pose_yaw=pose_yaw,
        )
        if (
            visual_starts_at_ego
            and idx > start_idx
            and float(body_x) > float(_VISUAL_PREFIX_STABILIZE_DISTANCE_M)
            and float(prev_body_x) >= float(_VISUAL_PREFIX_STABILIZE_DISTANCE_M)
        ):
            break
        if (not visual_starts_at_ego) and float(body_x) > float(_VISUAL_PREFIX_LATERAL_RAMP_DISTANCE_M):
            break
        if idx == 1:
            sample1_before = (float(body_x), float(body_y))

        min_forward_m = max(
            float(_VISUAL_PREFIX_MIN_FORWARD_M),
            float(cumulative_arc_m) * float(_VISUAL_PREFIX_FORWARD_PROGRESS_RATIO),
        )
        min_progress_x = (
            float(min_forward_m)
            if idx == start_idx
            else max(float(min_forward_m), float(prev_body_x) + float(_VISUAL_PREFIX_FORWARD_STEP_EPS_M))
        )
        adjusted_x = max(float(body_x), min_progress_x)
        adjusted_y = float(body_y)
        if visual_starts_at_ego:
            max_abs_lateral_m = max(
                float(_VISUAL_PREFIX_MIN_LATERAL_LIMIT_M),
                float(adjusted_x) * float(_VISUAL_PREFIX_MAX_LATERAL_RATIO),
            )
            adjusted_y = float(np.clip(float(body_y), -max_abs_lateral_m, max_abs_lateral_m))
        else:
            max_abs_lateral_m = max(
                0.0,
                math.tan(math.radians(float(entry_heading_limit_deg))) * max(
                    float(adjusted_x),
                    float(_VISUAL_PREFIX_MIN_FORWARD_M),
                ),
            )
            capped_y = float(np.clip(float(body_y), -max_abs_lateral_m, max_abs_lateral_m))
            if abs(capped_y - float(body_y)) > 1e-6:
                entry_heading_cap_applied = True
            adjusted_y = capped_y
        if visual_heading_limit_rad is not None and (visual_starts_at_ego or idx > start_idx):
            delta_x = max(float(adjusted_x) - float(prev_body_x), 1e-6)
            max_delta_y = float(math.tan(visual_heading_limit_rad)) * float(delta_x)
            capped_y = float(
                prev_body_y
                + np.clip(float(adjusted_y) - float(prev_body_y), -max_delta_y, max_delta_y)
            )
            if abs(capped_y - adjusted_y) > 1e-6:
                heading_cap_applied = True
            adjusted_y = capped_y

        if (
            abs(adjusted_x - float(body_x)) > 1e-6
            or abs(adjusted_y - float(body_y)) > 1e-6
        ):
            stabilized_xy[idx] = _body_to_world_xy(
                x_fwd_m=adjusted_x,
                y_left_m=adjusted_y,
                pose_x=pose_x,
                pose_y=pose_y,
                pose_yaw=pose_yaw,
            )
            stabilized_points = idx
        prev_body_x = adjusted_x
        prev_body_y = adjusted_y
        if idx == 1:
            sample1_after = (adjusted_x, adjusted_y)

    if sample1_before is None:
        sample1_before = (0.0, 0.0)
    if sample1_after is None:
        sample1_after = sample1_before

    notes["visual_prefix_stabilization_applied"] = bool(stabilized_points > 0)
    notes["visual_prefix_stabilization_points"] = int(stabilized_points)
    notes["visual_prefix_sample1_x_fwd_before_m"] = float(sample1_before[0])
    notes["visual_prefix_sample1_y_left_before_m"] = float(sample1_before[1])
    notes["visual_prefix_sample1_x_fwd_after_m"] = float(sample1_after[0])
    notes["visual_prefix_sample1_y_left_after_m"] = float(sample1_after[1])
    notes["visual_prefix_starts_at_ego"] = bool(visual_starts_at_ego)
    notes["visual_prefix_entry_heading_cap_applied"] = bool(entry_heading_cap_applied)
    notes["visual_prefix_heading_cap_applied"] = bool(heading_cap_applied)
    notes["visual_prefix_single_line_heading_cap_applied"] = bool(
        measurement_mode == "single_line" and heading_cap_applied
    )
    live_log(
        "path_optimizer",
        event="visual_prefix_stabilization",
        applied=bool(stabilized_points > 0),
        stabilized_points=int(stabilized_points),
        starts_at_ego=bool(visual_starts_at_ego),
        entry_heading_limit_deg=float(entry_heading_limit_deg),
        entry_heading_cap_applied=bool(entry_heading_cap_applied),
        sample1_x_fwd_before_m=float(sample1_before[0]),
        sample1_y_left_before_m=float(sample1_before[1]),
        sample1_x_fwd_after_m=float(sample1_after[0]),
        sample1_y_left_after_m=float(sample1_after[1]),
        heading_limit_deg=float(visual_heading_limit_deg),
        heading_cap_applied=bool(heading_cap_applied),
        single_line_heading_limit_deg=float(single_line_heading_limit_deg),
        single_line_heading_cap_applied=bool(
            measurement_mode == "single_line" and heading_cap_applied
        ),
        path_source=path_source,
    )
    return stabilized_xy, notes


def _stabilize_route_path_prefix(
    *,
    sampled_xy: np.ndarray,
    path_plan: BehaviorPathPlan,
    ctx: PlanningContext,
    line_loss_untrusted_release_ticks: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    release_active = int(line_loss_untrusted_release_ticks) > 0
    entry_heading_limit_deg = float(_ROUTE_PREFIX_MAX_ENTRY_HEADING_DEG)
    stabilize_distance_m = float(_ROUTE_PREFIX_STABILIZE_DISTANCE_M)
    if release_active:
        entry_heading_limit_deg = min(
            entry_heading_limit_deg,
            float(_LINE_LOSS_UNTRUSTED_ROUTE_RELEASE_ENTRY_HEADING_DEG),
        )
        stabilize_distance_m = max(
            stabilize_distance_m,
            float(_LINE_LOSS_UNTRUSTED_ROUTE_RELEASE_DISTANCE_M),
        )
    notes: dict[str, object] = {
        "route_prefix_stabilization_applied": False,
        "route_prefix_stabilization_points": 0,
        "route_prefix_entry_heading_limit_deg": float(entry_heading_limit_deg),
        "route_prefix_sample1_x_fwd_before_m": 0.0,
        "route_prefix_sample1_y_left_before_m": 0.0,
        "route_prefix_sample1_x_fwd_after_m": 0.0,
        "route_prefix_sample1_y_left_after_m": 0.0,
        "route_prefix_line_loss_untrusted_release_active": bool(release_active),
        "route_prefix_line_loss_untrusted_release_ticks": int(line_loss_untrusted_release_ticks),
        "route_prefix_stabilize_distance_m": float(stabilize_distance_m),
    }
    if _path_note_str(path_plan, "path_source") != "route_waypoints" or sampled_xy.shape[0] <= 1:
        return np.array(sampled_xy, copy=True), notes

    pose_x = float(ctx.pose.fused_pose.x)
    pose_y = float(ctx.pose.fused_pose.y)
    pose_yaw = float(ctx.pose.fused_pose.yaw)
    stabilized_xy = np.array(sampled_xy, copy=True)
    max_heading_rad = math.radians(float(entry_heading_limit_deg))
    prev_body_x = 0.0
    prev_body_y = 0.0
    stabilized_points = 0
    sample1_before: tuple[float, float] | None = None
    sample1_after: tuple[float, float] | None = None

    for idx in range(1, stabilized_xy.shape[0]):
        body_x, body_y = _world_to_body_xy(
            point_xy=stabilized_xy[idx],
            pose_x=pose_x,
            pose_y=pose_y,
            pose_yaw=pose_yaw,
        )
        if idx > 1 and float(body_x) > float(stabilize_distance_m) and (
            float(prev_body_x) >= float(stabilize_distance_m)
        ):
            break
        if idx == 1:
            sample1_before = (float(body_x), float(body_y))

        min_x = max(
            float(_ROUTE_PREFIX_MIN_FORWARD_M),
            float(prev_body_x) + float(_ROUTE_PREFIX_FORWARD_STEP_EPS_M),
        )
        adjusted_x = max(float(body_x), float(min_x))
        delta_x = max(float(adjusted_x) - float(prev_body_x), 1e-6)
        max_delta_y = math.tan(max_heading_rad) * delta_x
        adjusted_y = float(
            prev_body_y
            + np.clip(float(body_y) - float(prev_body_y), -max_delta_y, max_delta_y)
        )

        if (
            abs(adjusted_x - float(body_x)) > 1e-6
            or abs(adjusted_y - float(body_y)) > 1e-6
        ):
            stabilized_xy[idx] = _body_to_world_xy(
                x_fwd_m=adjusted_x,
                y_left_m=adjusted_y,
                pose_x=pose_x,
                pose_y=pose_y,
                pose_yaw=pose_yaw,
            )
            stabilized_points = idx
        prev_body_x = float(adjusted_x)
        prev_body_y = float(adjusted_y)
        if idx == 1:
            sample1_after = (float(adjusted_x), float(adjusted_y))

    if sample1_before is None:
        sample1_before = (0.0, 0.0)
    if sample1_after is None:
        sample1_after = sample1_before

    notes["route_prefix_stabilization_applied"] = bool(stabilized_points > 0)
    notes["route_prefix_stabilization_points"] = int(stabilized_points)
    notes["route_prefix_sample1_x_fwd_before_m"] = float(sample1_before[0])
    notes["route_prefix_sample1_y_left_before_m"] = float(sample1_before[1])
    notes["route_prefix_sample1_x_fwd_after_m"] = float(sample1_after[0])
    notes["route_prefix_sample1_y_left_after_m"] = float(sample1_after[1])
    live_log(
        "path_optimizer",
        event="route_prefix_stabilization",
        applied=bool(stabilized_points > 0),
        stabilized_points=int(stabilized_points),
        sample1_x_fwd_before_m=float(sample1_before[0]),
        sample1_y_left_before_m=float(sample1_before[1]),
        sample1_x_fwd_after_m=float(sample1_after[0]),
        sample1_y_left_after_m=float(sample1_after[1]),
        entry_heading_limit_deg=float(entry_heading_limit_deg),
        line_loss_untrusted_release_active=bool(release_active),
        line_loss_untrusted_release_ticks=int(line_loss_untrusted_release_ticks),
        stabilize_distance_m=float(stabilize_distance_m),
        path_source=_path_note_str(path_plan, "path_source"),
    )
    return stabilized_xy, notes


def _single_line_direction_conflicts_with_route(ctx: PlanningContext) -> bool:
    lane_observation = getattr(ctx, "lane_observation", None)
    route = getattr(ctx, "route", None)
    if lane_observation is None or route is None:
        return False
    if str(getattr(lane_observation, "measurement_mode", "none") or "none") != "single_line":
        return False

    pose_yaw = float(getattr(ctx.pose.fused_pose, "yaw", 0.0) or 0.0)
    route_path_yaw = float(getattr(route, "path_psi", getattr(route.matched_pose, "yaw", pose_yaw)) or pose_yaw)
    route_delta = _wrap_angle(route_path_yaw - pose_yaw)

    heading_hint = None
    cam_yaw = getattr(lane_observation, "camera_yaw_hint_rad", None)
    cam_conf = float(getattr(lane_observation, "camera_yaw_hint_confidence", 0.0) or 0.0)
    if cam_yaw is not None and cam_conf > 0.3:
        heading_hint = _wrap_angle(float(cam_yaw) - pose_yaw)
    else:
        try:
            heading_hint = float(getattr(lane_observation, "heading_error_rad", 0.0) or 0.0)
        except (TypeError, ValueError):
            heading_hint = None

    if heading_hint is None:
        return False

    if abs(route_delta) <= _SINGLE_LINE_ROUTE_STRAIGHT_TOL_RAD:
        return abs(float(heading_hint)) > _SINGLE_LINE_ROUTE_DIRECTION_MARGIN_RAD

    if (
        abs(float(heading_hint)) > _SINGLE_LINE_ROUTE_STRAIGHT_TOL_RAD
        and math.copysign(1.0, float(heading_hint)) != math.copysign(1.0, float(route_delta))
    ):
        return True

    return abs(_wrap_angle(float(heading_hint) - float(route_delta))) > _SINGLE_LINE_ROUTE_MAX_YAW_MISMATCH_RAD


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


def _blend_signatures_share_path_context(
    previous_signature: _BlendSignature | None,
    current_signature: _BlendSignature,
) -> bool:
    if previous_signature is None:
        return False
    return (
        previous_signature.scenario_name == current_signature.scenario_name
        and previous_signature.route_id == current_signature.route_id
        and previous_signature.current_lanelet_id == current_signature.current_lanelet_id
        and previous_signature.first_next_lanelet_id == current_signature.first_next_lanelet_id
    )


def _sample1_heading_delta_from_pose_deg(samples_xy: np.ndarray, pose_yaw: float) -> float | None:
    if samples_xy.shape[0] < 2:
        return None
    if samples_xy.shape[0] >= 3:
        dx = float(samples_xy[2, 0] - samples_xy[1, 0])
        dy = float(samples_xy[2, 1] - samples_xy[1, 1])
    else:
        dx = float(samples_xy[1, 0] - samples_xy[0, 0])
        dy = float(samples_xy[1, 1] - samples_xy[0, 1])
    if math.hypot(dx, dy) < 1e-6:
        return None
    heading = math.atan2(dy, dx)
    return math.degrees(_wrap_angle(float(heading) - float(pose_yaw)))


def _wrap_angle(angle_rad: float) -> float:
    out = float(angle_rad)
    while out > math.pi:
        out -= 2.0 * math.pi
    while out < -math.pi:
        out += 2.0 * math.pi
    return out


def _world_to_body_xy(
    *,
    point_xy: np.ndarray,
    pose_x: float,
    pose_y: float,
    pose_yaw: float,
) -> tuple[float, float]:
    dx = float(point_xy[0]) - float(pose_x)
    dy = float(point_xy[1]) - float(pose_y)
    x_fwd_m = (math.cos(float(pose_yaw)) * dx) + (math.sin(float(pose_yaw)) * dy)
    y_left_m = (math.sin(float(pose_yaw)) * dx) - (math.cos(float(pose_yaw)) * dy)
    return float(x_fwd_m), float(y_left_m)


def _maybe_hold_previous_path_on_single_line_transition(
    *,
    sampled_xy: np.ndarray,
    path_plan: BehaviorPathPlan,
    ctx: PlanningContext,
    blend_signature: _BlendSignature,
    prev_blend_signature: _BlendSignature | None,
    prev_lane_measurement_mode: str | None,
    prev_target_path: np.ndarray | None,
    prev_safe_target_path: np.ndarray | None,
    prev_safe_signature: _BlendSignature | None,
    hold_ticks: int,
) -> _SingleLineTransitionHoldResult | None:
    if _path_note_str(path_plan, "path_source") != "visual_lane_waypoints":
        return None
    lane_observation = getattr(ctx, "lane_observation", None)
    if lane_observation is None:
        return None
    if str(getattr(lane_observation, "measurement_mode", "none") or "none") != "single_line":
        return None
    if not bool(getattr(lane_observation, "planner_priority_active", False)):
        return None
    direct_error_m = None
    if bool(getattr(lane_observation, "direct_error_valid", False)):
        direct_error_m = getattr(lane_observation, "direct_error_m", None)
    if direct_error_m is None:
        direct_error_m = _single_line_visual_waypoint_error_m(ctx, lane_observation)
    if direct_error_m is None:
        return None
    direct_error_m = abs(float(direct_error_m))
    if direct_error_m > float(_SINGLE_LINE_TRANSITION_HOLD_MAX_DIRECT_ERROR_M):
        return None
    if sampled_xy.shape[0] < 2:
        return None

    candidate_xy = None
    candidate_source = None
    if (
        prev_safe_target_path is not None
        and prev_safe_target_path.shape[0] >= sampled_xy.shape[0]
        and prev_safe_target_path.shape[1] >= 2
        and _blend_signatures_share_path_context(prev_safe_signature, blend_signature)
    ):
        candidate_xy = np.asarray(prev_safe_target_path[: sampled_xy.shape[0], :2], dtype=float).copy()
        candidate_source = "prev_safe_target_path"
    elif (
        prev_target_path is not None
        and prev_target_path.shape[0] >= sampled_xy.shape[0]
        and prev_target_path.shape[1] >= 2
        and _blend_signatures_share_path_context(prev_blend_signature, blend_signature)
    ):
        candidate_xy = np.asarray(prev_target_path[: sampled_xy.shape[0], :2], dtype=float).copy()
        candidate_source = "prev_target_path"
    if candidate_xy is None or candidate_source is None:
        return None

    sample1_gap_m = _sample_gap_m(sampled_xy, candidate_xy, idx=1)
    heading_delta_deg = _sample_heading_delta_deg(sampled_xy, candidate_xy)
    if sample1_gap_m is None or heading_delta_deg is None:
        return None

    hold_reason = None
    if int(hold_ticks) > 0:
        if int(hold_ticks) >= int(_SINGLE_LINE_TRANSITION_HOLD_MAX_TICKS):
            return None
        if sample1_gap_m > max(
            float(_SINGLE_LINE_TRANSITION_HOLD_MAX_SAMPLE1_GAP_M),
            float(_VISUAL_PREVIOUS_PATH_MAX_SAMPLE1_GAP_M),
        ):
            return None
        if heading_delta_deg < float(_SINGLE_LINE_TRANSITION_HOLD_RELEASE_HEADING_DELTA_DEG):
            return None
        hold_reason = "active_hold"
    elif (
        str(prev_lane_measurement_mode or "none") in {"two_line", "single_line"}
        and float(_VISUAL_PREVIOUS_PATH_MAX_SAMPLE1_GAP_M) > 0.0
        and sample1_gap_m >= float(_VISUAL_PREVIOUS_PATH_MAX_SAMPLE1_GAP_M)
    ):
        hold_reason = "single_line_path_lateral_jump"
    elif (
        str(prev_lane_measurement_mode or "none") == "two_line"
        and sample1_gap_m <= float(_SINGLE_LINE_TRANSITION_HOLD_MAX_SAMPLE1_GAP_M)
        and heading_delta_deg >= float(_SINGLE_LINE_TRANSITION_HOLD_MIN_HEADING_DELTA_DEG)
    ):
        hold_reason = "two_line_to_single_line_heading_jump"
    if hold_reason is None:
        return None

    return _SingleLineTransitionHoldResult(
        target_xy=candidate_xy,
        notes={
            "single_line_transition_hold_applied": True,
            "single_line_transition_hold_reason": str(hold_reason),
            "single_line_transition_hold_ticks": int(hold_ticks) + 1,
            "single_line_transition_hold_sample1_gap_m": float(sample1_gap_m),
            "single_line_transition_hold_heading_delta_deg": float(heading_delta_deg),
            "single_line_transition_hold_direct_error_m": float(direct_error_m),
            "single_line_transition_hold_previous_path_source": str(candidate_source),
        },
    )


def _route_semantics_allow_line_loss_straight_hold(ctx: PlanningContext) -> bool:
    route = getattr(ctx, "route", None)
    if route is None:
        return True
    semantic_values = {
        str(getattr(route, "expected_control_type", "") or "").lower(),
        str(getattr(route, "next_semantic_type", "") or "").lower(),
    }
    blocking = semantic_values.intersection(_LINE_LOSS_STRAIGHT_HOLD_BLOCKING_SEMANTICS)
    if not blocking:
        return True
    try:
        distance_m = float(getattr(route, "next_semantic_distance_m", 0.0) or 0.0)
    except (TypeError, ValueError):
        distance_m = 0.0
    return distance_m > float(_LINE_LOSS_STRAIGHT_HOLD_BLOCK_DISTANCE_M)


def _line_loss_route_turn_reason(ctx: PlanningContext, path_plan: BehaviorPathPlan) -> str | None:
    route = getattr(ctx, "route", None)
    if route is not None:
        for key in ("heading_rad", "path_heading_change_rad"):
            try:
                value = float(getattr(route, key, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if math.isfinite(value) and abs(value) >= float(_LINE_LOSS_STRAIGHT_HOLD_ROUTE_HEADING_MAX_RAD):
                return key
        try:
            path_kappa = float(getattr(route, "path_kappa", 0.0) or 0.0)
        except (TypeError, ValueError):
            path_kappa = 0.0
        if math.isfinite(path_kappa) and abs(path_kappa) >= float(_LINE_LOSS_STRAIGHT_HOLD_ROUTE_KAPPA_MAX):
            return "path_kappa"

    raw_path = getattr(path_plan, "raw_path", None)
    try:
        samples = np.asarray(raw_path, dtype=float)
    except (TypeError, ValueError):
        return None
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 2:
        return None

    pose_x = float(ctx.pose.fused_pose.x)
    pose_y = float(ctx.pose.fused_pose.y)
    pose_yaw = float(ctx.pose.fused_pose.yaw)
    max_heading = float(_LINE_LOSS_STRAIGHT_HOLD_ROUTE_PATH_HEADING_MAX_RAD)
    min_lookahead = float(_LINE_LOSS_STRAIGHT_HOLD_ROUTE_LOOKAHEAD_MIN_M)

    for point in samples[1 : min(samples.shape[0], 8), :2]:
        x_fwd, y_left = _world_to_body_xy(
            point_xy=point,
            pose_x=pose_x,
            pose_y=pose_y,
            pose_yaw=pose_yaw,
        )
        distance = math.hypot(float(x_fwd), float(y_left))
        if distance < min_lookahead or x_fwd < 0.0:
            continue
        body_heading = math.atan2(float(y_left), max(1e-6, float(x_fwd)))
        if abs(body_heading) >= max_heading:
            return "route_raw_path_body_heading"

    if samples.shape[1] >= 3:
        for idx in range(1, min(samples.shape[0], 6)):
            try:
                route_yaw = float(samples[idx, 2])
            except (TypeError, ValueError):
                continue
            if math.isfinite(route_yaw) and abs(_wrap_angle(route_yaw - pose_yaw)) >= max_heading:
                return "route_raw_path_yaw"

    return None


def _should_soft_release_untrusted_line_loss_route(
    *,
    path_plan: BehaviorPathPlan,
    ctx: PlanningContext,
    previous_hold_ticks: int,
) -> bool:
    if _path_note_str(path_plan, "path_source") != "route_waypoints":
        return False
    if int(previous_hold_ticks) < int(_LINE_LOSS_STRAIGHT_HOLD_MAX_TICKS):
        return False
    lane_observation = getattr(ctx, "lane_observation", None)
    if lane_observation is None:
        return False
    measurement_mode = str(getattr(lane_observation, "measurement_mode", "none") or "none")
    if measurement_mode not in _LINE_LOSS_STRAIGHT_HOLD_MODES:
        return False
    if measurement_mode == "single_line" and _single_line_has_usable_lateral_guidance(
        lane_observation,
        path_plan=path_plan,
    ):
        return False
    route_turn_reason = _line_loss_route_turn_reason(ctx, path_plan)
    if route_turn_reason is None:
        return False
    route = getattr(ctx, "route", None)
    try:
        route_map_match_error_m = abs(float(getattr(route, "map_match_error_m", 0.0) or 0.0))
    except (TypeError, ValueError):
        route_map_match_error_m = 0.0
    if route_map_match_error_m > float(_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M):
        return True
    # A long visual-loss hold can end on a single "trusted" map tick even while
    # the next route samples still demand a sharp turn. Keep that handoff smooth.
    return int(previous_hold_ticks) >= int(_LINE_LOSS_STRAIGHT_HOLD_MAX_TICKS)


def _build_straight_hold_path_from_pose(
    *,
    ctx: PlanningContext,
    step_arc: float,
) -> np.ndarray:
    horizon_n = max(1, int(getattr(ctx, "horizon_n", 1) or 1))
    pose_x = float(ctx.pose.fused_pose.x)
    pose_y = float(ctx.pose.fused_pose.y)
    pose_yaw = float(ctx.pose.fused_pose.yaw)
    step_m = max(0.02, float(step_arc) if math.isfinite(float(step_arc)) else _MIN_STEP_M)
    distances = np.arange(horizon_n + 1, dtype=float) * step_m
    cos_y = math.cos(pose_yaw)
    sin_y = math.sin(pose_yaw)
    x = pose_x + (distances * cos_y)
    y = pose_y + (distances * sin_y)
    yaw = np.full(horizon_n + 1, pose_yaw, dtype=float)
    return np.column_stack([x, y, yaw])


def _build_previous_visual_hold_path_from_pose(
    *,
    ctx: PlanningContext,
    prev_target_path: np.ndarray | None,
    prev_target_path_source: str | None,
    step_arc: float,
) -> np.ndarray | None:
    if not bool(_LINE_LOSS_PREVIOUS_VISUAL_HOLD_ENABLED):
        return None
    if str(prev_target_path_source or "") not in _VISUAL_HOLD_PATH_SOURCES:
        return None
    try:
        previous = np.asarray(prev_target_path, dtype=float)
    except (TypeError, ValueError):
        return None
    if previous.ndim != 2 or previous.shape[0] < 2 or previous.shape[1] < 2:
        return None

    horizon_n = max(1, int(getattr(ctx, "horizon_n", 1) or 1))
    pose_x = float(ctx.pose.fused_pose.x)
    pose_y = float(ctx.pose.fused_pose.y)
    pose_yaw = float(ctx.pose.fused_pose.yaw)
    step_m = max(0.02, float(step_arc) if math.isfinite(float(step_arc)) else _MIN_STEP_M)
    ahead_rows: list[np.ndarray] = []
    for row in previous:
        if row.shape[0] < 2:
            continue
        x_fwd, _ = _world_to_body_xy(
            point_xy=row[:2],
            pose_x=pose_x,
            pose_y=pose_y,
            pose_yaw=pose_yaw,
        )
        if x_fwd >= -0.02:
            ahead_rows.append(np.asarray(row, dtype=float))
    if len(ahead_rows) < 2:
        return None

    xy_rows = [row[:2].astype(float, copy=True) for row in ahead_rows[: horizon_n + 1]]
    if len(xy_rows) < 2:
        return None
    while len(xy_rows) < horizon_n + 1:
        if len(xy_rows) >= 2:
            dx = float(xy_rows[-1][0] - xy_rows[-2][0])
            dy = float(xy_rows[-1][1] - xy_rows[-2][1])
            norm = math.hypot(dx, dy)
            if norm < 1e-6:
                heading = pose_yaw
            else:
                heading = math.atan2(dy, dx)
        else:
            heading = pose_yaw
        xy_rows.append(
            np.array(
                [
                    float(xy_rows[-1][0]) + step_m * math.cos(heading),
                    float(xy_rows[-1][1]) + step_m * math.sin(heading),
                ],
                dtype=float,
            )
        )
    xy = np.vstack(xy_rows[: horizon_n + 1])
    headings = _compute_headings(xy, initial_yaw=pose_yaw)
    return np.column_stack([xy, headings])


def _maybe_hold_straight_on_line_loss(
    *,
    path_plan: BehaviorPathPlan,
    ctx: PlanningContext,
    prev_lane_measurement_mode: str | None,
    prev_target_path: np.ndarray | None,
    prev_target_path_source: str | None,
    hold_ticks: int,
    step_arc: float,
) -> _LineLossStraightHoldResult | None:
    path_source = _path_note_str(path_plan, "path_source")
    if path_source not in {"route_waypoints", "lanelet_centerline"}:
        return None
    lane_observation = getattr(ctx, "lane_observation", None)
    if lane_observation is None:
        return None
    measurement_mode = str(getattr(lane_observation, "measurement_mode", "none") or "none")
    if measurement_mode not in _LINE_LOSS_STRAIGHT_HOLD_MODES:
        return None
    if measurement_mode == "single_line" and _single_line_has_usable_lateral_guidance(
        lane_observation,
        path_plan=path_plan,
    ):
        return None
    previous_mode = str(prev_lane_measurement_mode or "none")
    just_lost_line = previous_mode in {"two_line", "single_line"}
    active_hold = int(hold_ticks) > 0
    if not (just_lost_line or active_hold):
        return None
    if not _route_semantics_allow_line_loss_straight_hold(ctx):
        return None
    route_turn_reason = _line_loss_route_turn_reason(ctx, path_plan)
    route = getattr(ctx, "route", None)
    try:
        route_map_match_error_m = abs(float(getattr(route, "map_match_error_m", 0.0) or 0.0))
    except (TypeError, ValueError):
        route_map_match_error_m = 0.0
    route_turn_trusted = route_map_match_error_m <= float(_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M)
    if measurement_mode in {"none", "blind"} and route_turn_trusted:
        return None
    untrusted_route_turn = route_turn_reason is not None and not route_turn_trusted
    max_hold_ticks = int(_LINE_LOSS_STRAIGHT_HOLD_MAX_TICKS)
    if untrusted_route_turn:
        max_hold_ticks = max(
            max_hold_ticks,
            int(_LINE_LOSS_STRAIGHT_HOLD_UNTRUSTED_ROUTE_MAX_TICKS),
        )
    if (
        route_turn_reason is not None
        and int(hold_ticks) >= int(_LINE_LOSS_STRAIGHT_HOLD_ROUTE_ESCAPE_MIN_TICKS)
        and route_turn_trusted
    ):
        live_log(
            "behavior",
            event="line_loss_straight_hold_blocked",
            reason="route_requires_turn",
            route_turn_reason=str(route_turn_reason),
            hold_ticks=int(hold_ticks),
            measurement_mode=measurement_mode,
            previous_measurement_mode=previous_mode,
            map_match_error_m=float(route_map_match_error_m),
            map_match_error_limit_m=float(_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M),
        )
        return None
    if int(hold_ticks) >= max_hold_ticks:
        live_log(
            "behavior",
            event="line_loss_straight_hold_blocked",
            reason="max_ticks",
            route_turn_reason=(str(route_turn_reason) if route_turn_reason is not None else None),
            route_turn_trusted=bool(route_turn_trusted),
            hold_ticks=int(hold_ticks),
            measurement_mode=measurement_mode,
            previous_measurement_mode=previous_mode,
            map_match_error_m=float(route_map_match_error_m),
            map_match_error_limit_m=float(_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M),
            max_ticks=int(max_hold_ticks),
            base_max_ticks=int(_LINE_LOSS_STRAIGHT_HOLD_MAX_TICKS),
        )
        return None
    if (
        route_turn_reason is not None
        and int(hold_ticks) >= int(_LINE_LOSS_STRAIGHT_HOLD_ROUTE_ESCAPE_MIN_TICKS)
        and not route_turn_trusted
    ):
        live_log(
            "behavior",
            event="line_loss_route_turn_deferred",
            reason="map_match_error_high",
            route_turn_reason=str(route_turn_reason),
            hold_ticks=int(hold_ticks),
            measurement_mode=measurement_mode,
            previous_measurement_mode=previous_mode,
            map_match_error_m=float(route_map_match_error_m),
            map_match_error_limit_m=float(_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M),
        )

    target_path = _build_previous_visual_hold_path_from_pose(
        ctx=ctx,
        prev_target_path=prev_target_path,
        prev_target_path_source=prev_target_path_source,
        step_arc=step_arc,
    )
    hold_path_kind = "previous_visual_path" if target_path is not None else "straight_pose"
    hold_path_source = "line_loss_visual_hold" if target_path is not None else "line_loss_straight_hold"
    if target_path is None:
        target_path = _build_straight_hold_path_from_pose(ctx=ctx, step_arc=step_arc)
    left_bound, right_bound = _build_drivable_bounds(target_path, half_width_m=_DRIVABLE_HALF_WIDTH_M)
    reason = "line_loss_after_visual" if just_lost_line else "active_line_loss_hold"
    return _LineLossStraightHoldResult(
        target_path=target_path,
        left_bound=left_bound,
        right_bound=right_bound,
        notes={
            "line_loss_straight_hold_applied": True,
            "line_loss_straight_hold_reason": reason,
            "line_loss_hold_path_kind": str(hold_path_kind),
            "line_loss_hold_path_source": str(hold_path_source),
            "line_loss_straight_hold_ticks": int(hold_ticks) + 1,
            "line_loss_measurement_mode": measurement_mode,
            "line_loss_previous_measurement_mode": previous_mode,
            "line_loss_route_turn_reason": (
                str(route_turn_reason) if route_turn_reason is not None else None
            ),
            "line_loss_route_turn_trusted": bool(route_turn_trusted),
            "line_loss_route_map_match_error_m": float(route_map_match_error_m),
            "line_loss_route_map_match_limit_m": float(_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M),
            "line_loss_untrusted_route_extension_active": bool(
                untrusted_route_turn
                and int(max_hold_ticks) > int(_LINE_LOSS_STRAIGHT_HOLD_MAX_TICKS)
            ),
            "line_loss_straight_hold_step_arc_m": float(
                max(0.02, float(step_arc) if math.isfinite(float(step_arc)) else _MIN_STEP_M)
            ),
            "line_loss_straight_hold_base_max_ticks": int(_LINE_LOSS_STRAIGHT_HOLD_MAX_TICKS),
            "line_loss_straight_hold_max_ticks": int(max_hold_ticks),
        },
    )


def _maybe_hold_previous_visual_path_on_route_tracking(
    *,
    sampled_xy: np.ndarray,
    path_plan: BehaviorPathPlan,
    ctx: PlanningContext,
    prev_target_path: np.ndarray | None,
    prev_target_path_source: str | None,
    prev_safe_target_path: np.ndarray | None,
    prev_safe_left_bound: np.ndarray | None,
    prev_safe_right_bound: np.ndarray | None,
    prev_safe_path_source: str | None,
    hold_ticks: int,
) -> _RouteTrackingVisualHoldResult | None:
    if _path_note_str(path_plan, "path_source") != "route_waypoints":
        return None
    lane_observation = getattr(ctx, "lane_observation", None)
    if lane_observation is None:
        return None
    if str(getattr(lane_observation, "measurement_mode", "none") or "none") != "route_tracking":
        return None
    if int(hold_ticks) >= int(_ROUTE_TRACKING_VISUAL_HOLD_MAX_TICKS):
        return None

    route = getattr(ctx, "route", None)
    route_map_match_error_m = abs(float(getattr(route, "map_match_error_m", 0.0) or 0.0))
    if sampled_xy.shape[0] < 2:
        return None

    pose_x = float(ctx.pose.fused_pose.x)
    pose_y = float(ctx.pose.fused_pose.y)
    pose_yaw = float(ctx.pose.fused_pose.yaw)

    candidate_target_path = None
    candidate_left_bound = None
    candidate_right_bound = None
    candidate_source = None
    if (
        prev_safe_target_path is not None
        and str(prev_safe_path_source or "") in _VISUAL_HOLD_PATH_SOURCES
        and prev_safe_left_bound is not None
        and prev_safe_right_bound is not None
    ):
        candidate_target_path = np.array(prev_safe_target_path, copy=True)
        candidate_left_bound = np.array(prev_safe_left_bound, copy=True)
        candidate_right_bound = np.array(prev_safe_right_bound, copy=True)
        candidate_source = str(prev_safe_path_source or "visual_lane_waypoints")
    elif (
        prev_target_path is not None
        and str(prev_target_path_source or "") in _VISUAL_HOLD_PATH_SOURCES
    ):
        candidate_target_path = np.array(prev_target_path, copy=True)
        candidate_left_bound, candidate_right_bound = _build_drivable_bounds(
            candidate_target_path,
            half_width_m=_DRIVABLE_HALF_WIDTH_M,
        )
        candidate_source = str(prev_target_path_source or "visual_lane_waypoints")
    else:
        return None

    sample1_x_fwd_m, sample1_y_left_m = _world_to_body_xy(
        point_xy=sampled_xy[1],
        pose_x=pose_x,
        pose_y=pose_y,
        pose_yaw=pose_yaw,
    )
    route_heading_delta_deg = _sample1_heading_delta_from_pose_deg(sampled_xy, pose_yaw)
    visual_heading_delta_deg = _sample1_heading_delta_from_pose_deg(candidate_target_path[:, :2], pose_yaw)
    hold_reason = None
    if (
        route_map_match_error_m >= float(_ROUTE_TRACKING_VISUAL_HOLD_MIN_MAP_MATCH_ERROR_M)
        and sample1_x_fwd_m < float(_ROUTE_TRACKING_VISUAL_HOLD_MIN_FORWARD_M)
    ):
        hold_reason = "route_sample1_behind_ego"
    elif (
        route_map_match_error_m >= float(_ROUTE_TRACKING_VISUAL_HOLD_MIN_MAP_MATCH_ERROR_M)
        and sample1_x_fwd_m < float(_ROUTE_TRACKING_VISUAL_HOLD_NEAR_FORWARD_M)
        and abs(sample1_y_left_m) > float(_ROUTE_TRACKING_VISUAL_HOLD_MAX_LATERAL_M)
    ):
        hold_reason = "route_sample1_exits_laterally"
    elif (
        route_map_match_error_m >= float(_ROUTE_TRACKING_VISUAL_HOLD_HEADING_FLIP_MIN_MAP_MATCH_ERROR_M)
        and sample1_x_fwd_m < float(_ROUTE_TRACKING_VISUAL_HOLD_HEADING_FLIP_MAX_FORWARD_M)
        and route_heading_delta_deg is not None
        and visual_heading_delta_deg is not None
        and abs(route_heading_delta_deg) >= float(_ROUTE_TRACKING_VISUAL_HOLD_HEADING_FLIP_MIN_HEADING_DEG)
        and abs(visual_heading_delta_deg) >= float(_ROUTE_TRACKING_VISUAL_HOLD_HEADING_FLIP_MIN_HEADING_DEG)
        and math.copysign(1.0, route_heading_delta_deg) != math.copysign(1.0, visual_heading_delta_deg)
    ):
        hold_reason = "route_heading_flip_from_visual"
    if hold_reason is None:
        return None

    return _RouteTrackingVisualHoldResult(
        target_path=candidate_target_path,
        left_bound=candidate_left_bound,
        right_bound=candidate_right_bound,
        notes={
            "route_tracking_visual_hold_applied": True,
            "route_tracking_visual_hold_reason": str(hold_reason),
            "route_tracking_visual_hold_ticks": int(hold_ticks) + 1,
            "route_tracking_visual_hold_map_match_error_m": float(route_map_match_error_m),
            "route_tracking_visual_hold_sample1_x_fwd_m": float(sample1_x_fwd_m),
            "route_tracking_visual_hold_sample1_y_left_m": float(sample1_y_left_m),
            "route_tracking_visual_hold_previous_path_source": str(candidate_source),
            "route_tracking_visual_hold_route_heading_delta_deg": (
                float(route_heading_delta_deg) if route_heading_delta_deg is not None else None
            ),
            "route_tracking_visual_hold_visual_heading_delta_deg": (
                float(visual_heading_delta_deg) if visual_heading_delta_deg is not None else None
            ),
        },
    )


def _body_to_world_xy(
    *,
    x_fwd_m: float,
    y_left_m: float,
    pose_x: float,
    pose_y: float,
    pose_yaw: float,
) -> np.ndarray:
    cos_y = math.cos(float(pose_yaw))
    sin_y = math.sin(float(pose_yaw))
    world_x = float(pose_x) + (cos_y * float(x_fwd_m)) + (sin_y * float(y_left_m))
    world_y = float(pose_y) + (sin_y * float(x_fwd_m)) - (cos_y * float(y_left_m))
    return np.array([world_x, world_y], dtype=float)


def _maybe_override_single_line_visual_corridor_flip(
    *,
    raw_visual_xy: np.ndarray,
    containment: _ContainmentResult,
    path_plan: BehaviorPathPlan,
    ctx: PlanningContext,
) -> _ContainmentResult | None:
    if _path_note_str(path_plan, "path_source") != "visual_lane_waypoints":
        return None
    lane_observation = getattr(ctx, "lane_observation", None)
    if lane_observation is None:
        return None
    if str(getattr(lane_observation, "measurement_mode", "none") or "none") != "single_line":
        return None
    if not bool(getattr(lane_observation, "planner_priority_active", False)):
        return None
    if not bool(getattr(lane_observation, "direct_error_valid", False)):
        return None
    if containment.mode != "corridor_clamp":
        return None
    if raw_visual_xy.shape[0] < 2 or containment.target_xy.shape[0] < 2:
        return None
    if float(containment.max_violation_m) < float(_SINGLE_LINE_VISUAL_CORRIDOR_FLIP_MIN_VIOLATION_M):
        return None
    if float(containment.ego_corridor_error_m) > float(_SINGLE_LINE_VISUAL_CORRIDOR_FLIP_MAX_EGO_ERROR_M):
        return None

    observation_debug = getattr(lane_observation, "debug", {})
    if not isinstance(observation_debug, dict):
        observation_debug = {}
    resolved_side = str(observation_debug.get("single_line_resolved_side", "") or "")
    if resolved_side not in {"left", "right"}:
        detected_sides = tuple(getattr(lane_observation, "detected_sides", ()) or ())
        if len(detected_sides) == 1 and detected_sides[0] in {"left", "right"}:
            resolved_side = str(detected_sides[0])
    if resolved_side not in {"left", "right"}:
        return None
    if bool(observation_debug.get("single_line_outer_confirmed", False)):
        return None

    try:
        direct_error_m = float(getattr(lane_observation, "direct_error_m", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if abs(direct_error_m) < float(_SINGLE_LINE_VISUAL_CORRIDOR_FLIP_MIN_DIRECT_ERROR_M):
        return None

    pose_x = float(ctx.pose.fused_pose.x)
    pose_y = float(ctx.pose.fused_pose.y)
    pose_yaw = float(ctx.pose.fused_pose.yaw)

    def _sample1_heading_delta_deg(samples_xy: np.ndarray) -> float | None:
        if samples_xy.shape[0] < 2:
            return None
        if samples_xy.shape[0] >= 3:
            dx = float(samples_xy[2, 0] - samples_xy[1, 0])
            dy = float(samples_xy[2, 1] - samples_xy[1, 1])
        else:
            dx = float(samples_xy[1, 0] - samples_xy[0, 0])
            dy = float(samples_xy[1, 1] - samples_xy[0, 1])
        if math.hypot(dx, dy) < 1e-6:
            return None
        heading = math.atan2(dy, dx)
        return math.degrees(_wrap_angle(float(heading) - pose_yaw))

    raw_heading_delta_deg = _sample1_heading_delta_deg(raw_visual_xy)
    clamped_heading_delta_deg = _sample1_heading_delta_deg(containment.target_xy)
    if raw_heading_delta_deg is None or clamped_heading_delta_deg is None:
        return None

    if (
        abs(raw_heading_delta_deg) < float(_SINGLE_LINE_VISUAL_CORRIDOR_FLIP_MIN_HEADING_DEG)
        and raw_visual_xy.shape[0] >= 2
    ):
        raw_heading_delta_deg = 0.0
    if abs(clamped_heading_delta_deg) < float(_SINGLE_LINE_VISUAL_CORRIDOR_FLIP_MIN_HEADING_DEG):
        return None

    expected_sign = -1.0 if resolved_side == "right" else 1.0
    raw_heading_sign = 0.0 if abs(raw_heading_delta_deg) < 1e-6 else math.copysign(1.0, raw_heading_delta_deg)
    clamped_heading_sign = math.copysign(1.0, clamped_heading_delta_deg)

    raw_sample1_x_fwd_m, raw_sample1_y_left_m = _world_to_body_xy(
        point_xy=raw_visual_xy[1],
        pose_x=pose_x,
        pose_y=pose_y,
        pose_yaw=pose_yaw,
    )
    clamped_sample1_x_fwd_m, clamped_sample1_y_left_m = _world_to_body_xy(
        point_xy=containment.target_xy[1],
        pose_x=pose_x,
        pose_y=pose_y,
        pose_yaw=pose_yaw,
    )
    expected_lateral_sign = -1.0 if resolved_side == "right" else 1.0
    if (
        abs(raw_sample1_y_left_m) >= 1e-3
        and math.copysign(1.0, raw_sample1_y_left_m) != expected_lateral_sign
    ):
        return None
    if raw_heading_sign not in {0.0, expected_sign}:
        return None
    if clamped_heading_sign == expected_sign:
        return None

    left_bound, right_bound = _synthetic_bounds_from_xy(
        raw_visual_xy,
        half_width_m=_DRIVABLE_HALF_WIDTH_M,
    )
    return _ContainmentResult(
        target_xy=np.array(raw_visual_xy, copy=True),
        left_bound=left_bound,
        right_bound=right_bound,
        corridor_available=True,
        infeasible=False,
        used_prev_safe_path=False,
        first_infeasible_index=None,
        max_violation_m=0.0,
        max_abs_offset_m=float(containment.max_abs_offset_m),
        ego_corridor_error_m=float(containment.ego_corridor_error_m),
        touches_bound=False,
        mode="visual_single_line_override",
        notes={
            "corridor_source": "visual_single_line_override",
            "reference_corridor_source": str(containment.notes.get("corridor_source", "unknown")),
            "corridor_required": bool(containment.notes.get("corridor_required", False)),
            "corridor_required_missing": False,
            "corridor_visual_lane_guidance_active": bool(
                containment.notes.get("corridor_visual_lane_guidance_active", False)
            ),
            "corridor_visual_lane_guidance_reason": str(
                containment.notes.get("corridor_visual_lane_guidance_reason", "inactive")
            ),
            "corridor_visual_lane_guidance_mode": str(
                containment.notes.get("corridor_visual_lane_guidance_mode", "single_line")
            ),
            "corridor_visual_lane_guidance_shift_m": float(
                containment.notes.get("corridor_visual_lane_guidance_shift_m", 0.0) or 0.0
            ),
            "corridor_visual_lane_guidance_prefix_samples": 0,
            "corridor_visual_lane_guidance_max_shift_m": 0.0,
            "corridor_visual_lane_guidance_max_limit_m": 0.0,
            "corridor_visual_path_priority_active": bool(
                containment.notes.get("corridor_visual_path_priority_active", True)
            ),
            "single_line_visual_route_override_active": True,
            "single_line_visual_route_override_reason": "corridor_heading_flip",
            "single_line_visual_route_override_threshold_m": float(
                _SINGLE_LINE_VISUAL_CORRIDOR_FLIP_MIN_VIOLATION_M
            ),
            "single_line_visual_route_override_map_match_error_m": float(
                abs(float(getattr(getattr(ctx, "route", None), "map_match_error_m", 0.0) or 0.0))
            ),
            "single_line_visual_route_override_ego_corridor_error_m": float(containment.ego_corridor_error_m),
            "single_line_visual_corridor_flip_override_active": True,
            "single_line_visual_corridor_flip_raw_heading_delta_deg": float(raw_heading_delta_deg),
            "single_line_visual_corridor_flip_clamped_heading_delta_deg": float(clamped_heading_delta_deg),
            "single_line_visual_corridor_flip_raw_sample1_x_fwd_m": float(raw_sample1_x_fwd_m),
            "single_line_visual_corridor_flip_raw_sample1_y_left_m": float(raw_sample1_y_left_m),
            "single_line_visual_corridor_flip_clamped_sample1_x_fwd_m": float(clamped_sample1_x_fwd_m),
            "single_line_visual_corridor_flip_clamped_sample1_y_left_m": float(clamped_sample1_y_left_m),
            "single_line_visual_corridor_flip_direct_error_m": float(direct_error_m),
            "single_line_visual_corridor_flip_resolved_side": str(resolved_side),
            "single_line_visual_corridor_flip_violation_m": float(containment.max_violation_m),
        },
    )


def _compute_headings(samples_xy: np.ndarray, *, initial_yaw: float) -> np.ndarray:
    n = int(samples_xy.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.array([float(initial_yaw)], dtype=float)

    psi = np.zeros(n, dtype=float)
    for idx in range(n):
        if idx == 0:
            dx = float(samples_xy[1, 0] - samples_xy[0, 0])
            dy = float(samples_xy[1, 1] - samples_xy[0, 1])
        elif idx == (n - 1):
            dx = float(samples_xy[-1, 0] - samples_xy[-2, 0])
            dy = float(samples_xy[-1, 1] - samples_xy[-2, 1])
        else:
            dx = float(samples_xy[idx + 1, 0] - samples_xy[idx - 1, 0])
            dy = float(samples_xy[idx + 1, 1] - samples_xy[idx - 1, 1])
        if math.hypot(dx, dy) < 1e-6:
            psi[idx] = psi[idx - 1] if idx > 0 else float(initial_yaw)
        else:
            psi[idx] = math.atan2(dy, dx)
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


def _contain_path_within_corridor(
    *,
    sampled_xy: np.ndarray,
    path_plan: BehaviorPathPlan,
    ctx: PlanningContext,
    blend_signature: _BlendSignature,
    prev_safe_target_path: np.ndarray | None,
    prev_safe_left_bound: np.ndarray | None,
    prev_safe_right_bound: np.ndarray | None,
    prev_safe_path_source: str | None,
    prev_safe_signature: _BlendSignature | None,
) -> _ContainmentResult:
    corridor, corridor_build_notes = _build_lanelet_corridor(ctx)
    if corridor is None:
        empty = np.zeros((0, 2), dtype=float)
        notes = dict(corridor_build_notes)
        notes.setdefault("corridor_source", "synthetic_bounds")
        notes["corridor_required"] = bool(_requires_real_corridor(ctx))
        notes["corridor_required_missing"] = bool(notes["corridor_required"])
        return _ContainmentResult(
            target_xy=np.array(sampled_xy, copy=True),
            left_bound=empty,
            right_bound=empty,
            corridor_available=False,
            infeasible=False,
            used_prev_safe_path=False,
            first_infeasible_index=None,
            max_violation_m=0.0,
            max_abs_offset_m=0.0,
            ego_corridor_error_m=0.0,
            touches_bound=False,
            mode="synthetic_bounds",
            notes=notes,
        )

    target_xy = np.array(sampled_xy, copy=True)
    left_bound = np.zeros((sampled_xy.shape[0], 2), dtype=float)
    right_bound = np.zeros((sampled_xy.shape[0], 2), dtype=float)
    first_infeasible_index: int | None = None
    max_violation_m = 0.0
    max_abs_offset_m = 0.0
    ego_corridor_error_m = 0.0
    touches_bound = False
    traveled_m = 0.0
    visual_lane_state = _visual_lane_reentry_state(ctx)
    visual_lane_base_shift_m = _visual_lane_reentry_base_shift_m(visual_lane_state)
    corridor_visual_lane_guidance_active = (
        bool(visual_lane_state.active)
        and abs(float(visual_lane_base_shift_m)) >= 1e-6
    )
    path_source = _path_note_str(path_plan, "path_source")
    path_source_visual = path_source == "visual_lane_waypoints"
    # A high-quality two-line measurement is the best source of the physical
    # lane center. Allow the target to use the same relaxed corridor budget for
    # route waypoints too; otherwise small OSM-vs-texture offsets force the MPC
    # to hug the visual lane edge even while both painted lines are visible.
    corridor_visual_path_priority_active = (
        bool(corridor_visual_lane_guidance_active)
        and bool(getattr(ctx.lane_observation, "planner_priority_active", False))
        and (
            bool(path_source_visual)
            or (
                path_source == "route_waypoints"
                and str(visual_lane_state.measurement_mode) == "two_line"
                and _route_waypoint_visual_priority_allowed(ctx)
            )
        )
    )
    corridor_visual_lane_prefix_samples = 0
    corridor_visual_lane_max_shift_m = 0.0
    corridor_visual_lane_max_limit_m = 0.0
    corridor_reference_projection = (
        _project_point_to_corridor(corridor, sampled_xy[0])
        if sampled_xy.shape[0] > 0
        else None
    )
    corridor_reference_error_m = (
        abs(float(corridor_reference_projection.center_offset_m))
        if corridor_reference_projection is not None
        else 0.0
    )
    route_map_match_error_m = abs(float(getattr(getattr(ctx, "route", None), "map_match_error_m", 0.0) or 0.0))
    lane_observation = getattr(ctx, "lane_observation", None)
    measurement_mode = str(getattr(lane_observation, "measurement_mode", "none") or "none")
    detected_sides = tuple(getattr(lane_observation, "detected_sides", ()) or ())

    def _visual_corridor_metrics(samples_xy: np.ndarray) -> tuple[float, float, bool]:
        visual_max_abs_offset_m = float(corridor_reference_error_m)
        visual_max_violation_m = 0.0
        visual_touches_bound = False
        for point in samples_xy:
            projection = _project_point_to_corridor(corridor, point)
            if projection is None:
                continue
            limit_half_width_m = _corridor_offset_limit_m(
                projection.width_m,
                visual_path_priority_active=True,
                visual_requested_shift_m=visual_lane_base_shift_m,
            )
            offset_m = abs(float(projection.center_offset_m))
            visual_max_abs_offset_m = max(visual_max_abs_offset_m, offset_m)
            visual_max_violation_m = max(visual_max_violation_m, max(0.0, offset_m - limit_half_width_m))
            if limit_half_width_m > 1e-6 and offset_m >= max(0.0, limit_half_width_m - 1e-3):
                visual_touches_bound = True
        return float(visual_max_abs_offset_m), float(visual_max_violation_m), bool(visual_touches_bound)

    def _previous_visual_hold_result(
        *,
        reason: str,
        mode: str,
        violation_m: float,
        max_abs_offset_m: float,
        touches_bound_value: bool,
        limit_m: float,
    ) -> _ContainmentResult | None:
        if (
            prev_safe_target_path is None
            or str(prev_safe_path_source or "") not in _VISUAL_HOLD_PATH_SOURCES
            or prev_safe_target_path.shape[0] < sampled_xy.shape[0]
            or prev_safe_target_path.shape[1] < 2
        ):
            return None
        held_xy = np.asarray(prev_safe_target_path[: sampled_xy.shape[0], :2], dtype=float).copy()
        if (
            prev_safe_left_bound is not None
            and prev_safe_right_bound is not None
            and prev_safe_left_bound.shape[0] >= sampled_xy.shape[0]
            and prev_safe_right_bound.shape[0] >= sampled_xy.shape[0]
        ):
            held_left_bound = np.asarray(prev_safe_left_bound[: sampled_xy.shape[0]], dtype=float).copy()
            held_right_bound = np.asarray(prev_safe_right_bound[: sampled_xy.shape[0]], dtype=float).copy()
        else:
            held_left_bound, held_right_bound = _synthetic_bounds_from_xy(
                held_xy,
                half_width_m=_DRIVABLE_HALF_WIDTH_M,
            )
        return _ContainmentResult(
            target_xy=held_xy,
            left_bound=held_left_bound,
            right_bound=held_right_bound,
            corridor_available=True,
            infeasible=False,
            used_prev_safe_path=True,
            first_infeasible_index=None,
            max_violation_m=float(violation_m),
            max_abs_offset_m=float(max_abs_offset_m),
            ego_corridor_error_m=float(corridor_reference_error_m),
            touches_bound=bool(touches_bound_value),
            mode=str(mode),
            notes={
                "corridor_source": str(mode),
                "reference_corridor_source": corridor.source,
                "corridor_lanelet_ids": list(corridor.lanelet_ids),
                "corridor_required": bool(_requires_real_corridor(ctx)),
                "corridor_required_missing": False,
                "corridor_visual_lane_guidance_active": bool(corridor_visual_lane_guidance_active),
                "corridor_visual_lane_guidance_reason": str(visual_lane_state.reason),
                "corridor_visual_lane_guidance_mode": str(visual_lane_state.measurement_mode),
                "corridor_visual_lane_guidance_shift_m": float(visual_lane_base_shift_m),
                "corridor_visual_lane_guidance_prefix_samples": 0,
                "corridor_visual_lane_guidance_max_shift_m": 0.0,
                "corridor_visual_lane_guidance_max_limit_m": 0.0,
                "corridor_visual_path_priority_active": True,
                "visual_primary_corridor_override_active": False,
                "visual_primary_corridor_override_rejected_reason": str(reason),
                "visual_primary_corridor_absurd_violation_m": float(violation_m),
                "visual_primary_corridor_absurd_limit_m": float(limit_m),
                "visual_primary_previous_hold_active": True,
                "visual_primary_previous_hold_source": str(prev_safe_path_source or ""),
                "visual_primary_corridor_reference_error_m": float(corridor_reference_error_m),
                "visual_primary_corridor_map_match_error_m": float(route_map_match_error_m),
            },
        )

    visual_two_line_override_active = (
        bool(path_source_visual)
        and measurement_mode == "two_line"
        and len(detected_sides) == 2
        and float(getattr(lane_observation, "quality", 0.0) or 0.0)
        >= float(_VISUAL_LANE_REENTRY_TWO_LINE_MIN_QUALITY)
    )
    if visual_two_line_override_active:
        visual_max_abs_offset_m, visual_max_violation_m, visual_touches_bound = _visual_corridor_metrics(sampled_xy)
        absurd_limit_m = float(_VISUAL_TWO_LINE_ABSURD_CORRIDOR_VIOLATION_M)
        visual_violation_absurd = (
            float(absurd_limit_m) > 0.0
            and visual_max_violation_m >= float(absurd_limit_m)
        )
        if visual_violation_absurd:
            held = _previous_visual_hold_result(
                reason="absurd_corridor_violation",
                mode="visual_two_line_previous_hold",
                violation_m=visual_max_violation_m,
                max_abs_offset_m=visual_max_abs_offset_m,
                touches_bound_value=visual_touches_bound,
                limit_m=absurd_limit_m,
            )
            if held is not None:
                return held
        left_bound, right_bound = _synthetic_bounds_from_xy(
            sampled_xy,
            half_width_m=_DRIVABLE_HALF_WIDTH_M,
        )
        return _ContainmentResult(
            target_xy=np.array(sampled_xy, copy=True),
            left_bound=left_bound,
            right_bound=right_bound,
            corridor_available=True,
            infeasible=False,
            used_prev_safe_path=False,
            first_infeasible_index=None,
            max_violation_m=float(visual_max_violation_m),
            max_abs_offset_m=float(visual_max_abs_offset_m),
            ego_corridor_error_m=float(corridor_reference_error_m),
            touches_bound=bool(visual_touches_bound),
            mode="visual_two_line_corridor_override",
            notes={
                "corridor_source": "visual_two_line_override",
                "reference_corridor_source": corridor.source,
                "corridor_lanelet_ids": list(corridor.lanelet_ids),
                "corridor_required": bool(_requires_real_corridor(ctx)),
                "corridor_required_missing": False,
                "corridor_visual_lane_guidance_active": bool(corridor_visual_lane_guidance_active),
                "corridor_visual_lane_guidance_reason": str(visual_lane_state.reason),
                "corridor_visual_lane_guidance_mode": str(visual_lane_state.measurement_mode),
                "corridor_visual_lane_guidance_shift_m": float(visual_lane_base_shift_m),
                "corridor_visual_lane_guidance_prefix_samples": 0,
                "corridor_visual_lane_guidance_max_shift_m": 0.0,
                "corridor_visual_lane_guidance_max_limit_m": 0.0,
                "corridor_visual_path_priority_active": True,
                "visual_primary_corridor_override_active": True,
                "visual_primary_corridor_override_reason": "two_line_visual_primary",
                "visual_primary_corridor_reference_error_m": float(corridor_reference_error_m),
                "visual_primary_corridor_map_match_error_m": float(route_map_match_error_m),
            },
        )
    single_line_visual_override_reason = "route_map_match"
    if corridor_reference_error_m >= route_map_match_error_m:
        single_line_visual_override_reason = "corridor_pose_mismatch"
    single_line_route_override_map_match_trigger = (
        route_map_match_error_m >= float(_SINGLE_LINE_VISUAL_ROUTE_OVERRIDE_MAP_MATCH_ERROR_M)
    )
    single_line_route_override_corridor_trigger = (
        corridor_reference_error_m >= float(_SINGLE_LINE_VISUAL_ROUTE_OVERRIDE_EGO_CORRIDOR_ERROR_M)
    )
    single_line_visual_override_active = (
        bool(path_source_visual)
        and measurement_mode == "single_line"
        and bool(getattr(lane_observation, "planner_priority_active", False))
        and (
            single_line_route_override_map_match_trigger
            or single_line_route_override_corridor_trigger
        )
    )
    if bool(path_source_visual) and measurement_mode == "single_line":
        visual_max_abs_offset_m, visual_max_violation_m, visual_touches_bound = _visual_corridor_metrics(sampled_xy)
        if (
            float(_VISUAL_ABSURD_CORRIDOR_VIOLATION_M) > 0.0
            and visual_max_violation_m >= float(_VISUAL_ABSURD_CORRIDOR_VIOLATION_M)
        ):
            held = _previous_visual_hold_result(
                reason="absurd_single_line_corridor_violation",
                mode="visual_single_line_previous_hold",
                violation_m=visual_max_violation_m,
                max_abs_offset_m=visual_max_abs_offset_m,
                touches_bound_value=visual_touches_bound,
                limit_m=float(_VISUAL_ABSURD_CORRIDOR_VIOLATION_M),
            )
            if held is not None:
                return held
            single_line_visual_override_active = False
    if single_line_visual_override_active:
        active_threshold_m = (
            float(_SINGLE_LINE_VISUAL_ROUTE_OVERRIDE_EGO_CORRIDOR_ERROR_M)
            if single_line_visual_override_reason == "corridor_pose_mismatch"
            else float(_SINGLE_LINE_VISUAL_ROUTE_OVERRIDE_MAP_MATCH_ERROR_M)
        )
        left_bound, right_bound = _synthetic_bounds_from_xy(
            sampled_xy,
            half_width_m=_DRIVABLE_HALF_WIDTH_M,
        )
        return _ContainmentResult(
            target_xy=np.array(sampled_xy, copy=True),
            left_bound=left_bound,
            right_bound=right_bound,
            corridor_available=True,
            infeasible=False,
            used_prev_safe_path=False,
            first_infeasible_index=None,
            max_violation_m=0.0,
            max_abs_offset_m=float(corridor_reference_error_m),
            ego_corridor_error_m=float(corridor_reference_error_m),
            touches_bound=False,
            mode="visual_single_line_override",
            notes={
                "corridor_source": "visual_single_line_override",
                "reference_corridor_source": corridor.source,
                "corridor_lanelet_ids": list(corridor.lanelet_ids),
                "corridor_required": bool(_requires_real_corridor(ctx)),
                "corridor_required_missing": False,
                "corridor_visual_lane_guidance_active": bool(corridor_visual_lane_guidance_active),
                "corridor_visual_lane_guidance_reason": str(visual_lane_state.reason),
                "corridor_visual_lane_guidance_mode": str(visual_lane_state.measurement_mode),
                "corridor_visual_lane_guidance_shift_m": float(visual_lane_base_shift_m),
                "corridor_visual_lane_guidance_prefix_samples": 0,
                "corridor_visual_lane_guidance_max_shift_m": 0.0,
                "corridor_visual_lane_guidance_max_limit_m": 0.0,
                "corridor_visual_path_priority_active": bool(corridor_visual_path_priority_active),
                "single_line_visual_route_override_active": True,
                "single_line_visual_route_override_reason": str(single_line_visual_override_reason),
                "single_line_visual_route_override_threshold_m": float(active_threshold_m),
                "single_line_visual_route_override_map_match_threshold_m": float(
                    _SINGLE_LINE_VISUAL_ROUTE_OVERRIDE_MAP_MATCH_ERROR_M
                ),
                "single_line_visual_route_override_ego_threshold_m": float(
                    _SINGLE_LINE_VISUAL_ROUTE_OVERRIDE_EGO_CORRIDOR_ERROR_M
                ),
                "single_line_visual_route_override_map_match_triggered": bool(
                    single_line_route_override_map_match_trigger
                ),
                "single_line_visual_route_override_ego_triggered": bool(
                    single_line_route_override_corridor_trigger
                ),
                "single_line_visual_route_override_map_match_error_m": float(route_map_match_error_m),
                "single_line_visual_route_override_ego_corridor_error_m": float(corridor_reference_error_m),
            },
        )

    for idx, point in enumerate(sampled_xy):
        if idx > 0:
            traveled_m += float(np.linalg.norm(sampled_xy[idx] - sampled_xy[idx - 1]))
        projection = _project_point_to_corridor(corridor, point)
        if projection is None:
            if first_infeasible_index is None:
                first_infeasible_index = max(1, int(idx))
            if idx > 0:
                target_xy[idx] = target_xy[idx - 1]
                left_bound[idx] = left_bound[idx - 1]
                right_bound[idx] = right_bound[idx - 1]
            continue

        left_bound[idx] = projection.left_point
        right_bound[idx] = projection.right_point
        offset_m = float(projection.center_offset_m)
        max_abs_offset_m = max(max_abs_offset_m, abs(offset_m))
        if idx == 0:
            ego_corridor_error_m = abs(offset_m)
            continue

        limit_half_width_m = _corridor_offset_limit_m(
            projection.width_m,
            visual_path_priority_active=corridor_visual_path_priority_active,
            visual_requested_shift_m=visual_lane_base_shift_m,
        )
        corridor_visual_lane_max_limit_m = max(
            corridor_visual_lane_max_limit_m,
            float(limit_half_width_m),
        )
        if limit_half_width_m <= 1e-6 and first_infeasible_index is None:
            first_infeasible_index = idx
        violation_m = max(0.0, abs(offset_m) - limit_half_width_m)
        max_violation_m = max(max_violation_m, violation_m)
        if limit_half_width_m > 1e-6 and abs(offset_m) >= max(0.0, limit_half_width_m - 1e-3):
            touches_bound = True
        clamped_offset_m = float(np.clip(offset_m, -limit_half_width_m, limit_half_width_m))
        apply_corridor_visual_guidance = (
            idx > 0
            and corridor_visual_lane_guidance_active
            and limit_half_width_m > 1e-6
        )
        if apply_corridor_visual_guidance:
            fade = max(
                0.0,
                1.0 - (traveled_m / max(float(visual_lane_state.fade_distance_m), 1e-6)),
            )
            if fade > 1e-6:
                desired_offset_m = float(np.clip(
                    -float(visual_lane_base_shift_m) * fade,
                    -limit_half_width_m,
                    limit_half_width_m,
                ))
                if corridor_visual_path_priority_active:
                    blend = (
                        float(_VISUAL_PATH_PRIORITY_TWO_LINE_BLEND)
                        if str(visual_lane_state.measurement_mode) == "two_line"
                        else float(_VISUAL_PATH_PRIORITY_SINGLE_LINE_BLEND)
                    )
                else:
                    blend = (
                        1.0
                        if str(visual_lane_state.measurement_mode) == "two_line"
                        else 0.85
                    )
                clamped_offset_m = float(np.clip(
                    clamped_offset_m + blend * (desired_offset_m - clamped_offset_m),
                    -limit_half_width_m,
                    limit_half_width_m,
                ))
                corridor_visual_lane_prefix_samples = idx
                corridor_visual_lane_max_shift_m = max(
                    corridor_visual_lane_max_shift_m,
                    abs(desired_offset_m),
                )
        target_xy[idx] = projection.center_point + (clamped_offset_m * projection.lateral_dir)

    used_prev_safe_path = False
    mode = "corridor_nominal"
    if max_violation_m > 1e-6 or touches_bound:
        mode = "corridor_clamp"
    infeasible = first_infeasible_index is not None
    if infeasible:
        mode = "corridor_infeasible"
        if (
            prev_safe_target_path is not None
            and prev_safe_signature == blend_signature
            and prev_safe_target_path.shape[0] >= target_xy.shape[0]
            and prev_safe_target_path.shape[1] >= 2
        ):
            target_xy = np.asarray(prev_safe_target_path[: target_xy.shape[0], :2], dtype=float).copy()
            target_xy[0] = np.array([ctx.pose.fused_pose.x, ctx.pose.fused_pose.y], dtype=float)
            if (
                prev_safe_left_bound is not None
                and prev_safe_right_bound is not None
                and prev_safe_left_bound.shape[0] >= left_bound.shape[0]
                and prev_safe_right_bound.shape[0] >= right_bound.shape[0]
            ):
                left_bound = np.asarray(prev_safe_left_bound[: left_bound.shape[0]], dtype=float).copy()
                right_bound = np.asarray(prev_safe_right_bound[: right_bound.shape[0]], dtype=float).copy()
            used_prev_safe_path = True
            mode = "reused_prev_safe_path"

    notes = {
        "corridor_source": corridor.source,
        "corridor_lanelet_ids": list(corridor.lanelet_ids),
        "corridor_required": bool(_requires_real_corridor(ctx)),
        "corridor_required_missing": False,
        "corridor_visual_lane_guidance_active": bool(corridor_visual_lane_guidance_active),
        "corridor_visual_lane_guidance_reason": str(visual_lane_state.reason),
        "corridor_visual_lane_guidance_mode": str(visual_lane_state.measurement_mode),
        "corridor_visual_lane_guidance_shift_m": float(visual_lane_base_shift_m),
        "corridor_visual_lane_guidance_prefix_samples": int(corridor_visual_lane_prefix_samples),
        "corridor_visual_lane_guidance_max_shift_m": float(corridor_visual_lane_max_shift_m),
        "corridor_visual_lane_guidance_max_limit_m": float(corridor_visual_lane_max_limit_m),
        "corridor_visual_path_priority_active": bool(corridor_visual_path_priority_active),
    }
    return _ContainmentResult(
        target_xy=target_xy,
        left_bound=left_bound,
        right_bound=right_bound,
        corridor_available=True,
        infeasible=infeasible,
        used_prev_safe_path=used_prev_safe_path,
        first_infeasible_index=first_infeasible_index,
        max_violation_m=max_violation_m,
        max_abs_offset_m=max_abs_offset_m,
        ego_corridor_error_m=ego_corridor_error_m,
        touches_bound=touches_bound,
        mode=mode,
        notes=notes,
    )


def _build_lanelet_corridor(ctx: PlanningContext) -> tuple[_CorridorGeometry | None, dict[str, Any]]:
    lanelet_ids = _extract_lanelet_corridor_ids(ctx)
    if not lanelet_ids:
        return None, {"corridor_source": "missing_route_lanelet_ids"}

    lanelet_map = _resolve_lanelet_map_for_corridor(ctx, lanelet_ids)
    if lanelet_map is None:
        return None, {
            "corridor_source": "missing_lanelet_map",
            "corridor_lanelet_ids": list(lanelet_ids),
        }

    axis_parts: list[np.ndarray] = []
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    used_lanelet_ids: list[str] = []
    missing_lanelet_ids: list[str] = []
    for lanelet_id in lanelet_ids:
        lanelet = lanelet_map.get_lanelet(str(lanelet_id))
        if lanelet is None:
            missing_lanelet_ids.append(str(lanelet_id))
            continue
        axis = np.asarray(lanelet.centerline, dtype=float)
        if axis.ndim != 2 or axis.shape[0] < 2:
            continue
        left = np.asarray(getattr(lanelet, "left_boundary", np.zeros((0, 2), dtype=float)), dtype=float)
        right = np.asarray(getattr(lanelet, "right_boundary", np.zeros((0, 2), dtype=float)), dtype=float)
        if left.ndim != 2 or left.shape[0] < 2 or right.ndim != 2 or right.shape[0] < 2:
            left, right = _synthetic_bounds_from_xy(axis, half_width_m=_DRIVABLE_HALF_WIDTH_M)
        else:
            left = _resample_polyline_n(left, axis.shape[0])
            right = _resample_polyline_n(right, axis.shape[0])
        axis_parts.append(axis)
        left_parts.append(left)
        right_parts.append(right)
        used_lanelet_ids.append(str(lanelet_id))

    if not axis_parts:
        return None, {
            "corridor_source": "missing_lanelet_geometry",
            "corridor_lanelet_ids": list(lanelet_ids),
            "corridor_missing_lanelet_ids": missing_lanelet_ids,
        }
    axis, left, right = _concat_corridor_triplets(axis_parts, left_parts, right_parts)
    if axis.shape[0] < 2 or left.shape[0] != axis.shape[0] or right.shape[0] != axis.shape[0]:
        return None, {
            "corridor_source": "invalid_lanelet_geometry",
            "corridor_lanelet_ids": list(used_lanelet_ids or lanelet_ids),
        }
    source = "lanelet_bounds"
    if getattr(ctx, "lanelet_map", None) is not lanelet_map:
        source = "lanelet_bounds_tracking_recovery"
    return (
        _CorridorGeometry(
            axis=axis,
            left=left,
            right=right,
            lanelet_ids=tuple(used_lanelet_ids),
            source=source,
        ),
        {
            "corridor_lanelet_ids": list(used_lanelet_ids),
        },
    )


def _resolve_lanelet_map_for_corridor(
    ctx: PlanningContext,
    lanelet_ids: tuple[str, ...],
):
    lanelet_map = getattr(ctx, "lanelet_map", None)
    if lanelet_map_covers_ids(lanelet_map, lanelet_ids):
        return lanelet_map
    force_reload = lanelet_map is not None
    recovered_map = load_tracking_lanelet_map(force_reload=force_reload)
    if lanelet_map_covers_ids(recovered_map, lanelet_ids):
        return recovered_map
    return lanelet_map if lanelet_map_covers_ids(lanelet_map, lanelet_ids) else None


def _requires_real_corridor(ctx: PlanningContext) -> bool:
    return bool(
        getattr(ctx.route, "route_active", False)
        and getattr(ctx.route, "current_lanelet_id", None)
    )


def _extract_lanelet_corridor_ids(ctx: PlanningContext) -> tuple[str, ...]:
    ordered: list[str] = []
    if getattr(ctx.route, "current_lanelet_id", None):
        ordered.append(str(ctx.route.current_lanelet_id))
    ordered.extend(str(item) for item in (ctx.route.next_lanelet_ids or ()) if str(item))
    return tuple(dict.fromkeys(ordered))


def _concat_corridor_parts(parts: list[np.ndarray]) -> np.ndarray:
    combined: list[np.ndarray] = []
    for part in parts:
        arr = np.asarray(part, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        if combined and np.linalg.norm(combined[-1][-1] - arr[0]) <= 1e-6:
            combined.append(arr[1:])
        else:
            combined.append(arr)
    if not combined:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(combined)


def _concat_corridor_triplets(
    axis_parts: list[np.ndarray],
    left_parts: list[np.ndarray],
    right_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_combined: list[np.ndarray] = []
    left_combined: list[np.ndarray] = []
    right_combined: list[np.ndarray] = []
    for axis_part, left_part, right_part in zip(axis_parts, left_parts, right_parts):
        axis = np.asarray(axis_part, dtype=float)
        left = np.asarray(left_part, dtype=float)
        right = np.asarray(right_part, dtype=float)
        if (
            axis.ndim != 2
            or left.ndim != 2
            or right.ndim != 2
            or axis.shape[0] == 0
            or left.shape[0] != axis.shape[0]
            or right.shape[0] != axis.shape[0]
        ):
            continue
        start_idx = 0
        if axis_combined and np.linalg.norm(axis_combined[-1][-1] - axis[0]) <= 1e-6:
            start_idx = 1
        axis_combined.append(axis[start_idx:])
        left_combined.append(left[start_idx:])
        right_combined.append(right[start_idx:])
    if not axis_combined:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty, empty
    return np.vstack(axis_combined), np.vstack(left_combined), np.vstack(right_combined)


def _resample_polyline_n(polyline: np.ndarray, n_samples: int) -> np.ndarray:
    arr = np.asarray(polyline, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((max(0, int(n_samples)), 2), dtype=float)
    if arr.shape[0] == 1:
        return np.repeat(arr, max(1, int(n_samples)), axis=0)
    n_samples = max(2, int(n_samples))
    seg_lens = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cumulative[-1])
    if total <= 1e-9:
        return np.repeat(arr[:1], n_samples, axis=0)
    target_arcs = np.linspace(0.0, total, n_samples)
    samples = np.zeros((n_samples, 2), dtype=float)
    for idx, arc in enumerate(target_arcs):
        seg_idx = int(np.searchsorted(cumulative, arc, side="right") - 1)
        seg_idx = max(0, min(seg_idx, arr.shape[0] - 2))
        seg_len = float(seg_lens[seg_idx])
        if seg_len <= 1e-9:
            samples[idx] = arr[seg_idx]
            continue
        t = (arc - float(cumulative[seg_idx])) / seg_len
        samples[idx] = arr[seg_idx] + t * (arr[seg_idx + 1] - arr[seg_idx])
    return samples


def _project_point_to_corridor(
    corridor: _CorridorGeometry,
    point_xy: np.ndarray,
) -> _CorridorProjection | None:
    axis = np.asarray(corridor.axis, dtype=float)
    left = np.asarray(corridor.left, dtype=float)
    right = np.asarray(corridor.right, dtype=float)
    if axis.shape[0] < 2 or left.shape != axis.shape or right.shape != axis.shape:
        return None

    best: _CorridorProjection | None = None
    best_dist = math.inf
    cumulative_arc_m = 0.0
    px = float(point_xy[0])
    py = float(point_xy[1])
    for idx in range(axis.shape[0] - 1):
        a = axis[idx]
        b = axis[idx + 1]
        seg = b - a
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq <= 1e-12:
            continue
        seg_len = math.sqrt(seg_len_sq)
        t = ((px - float(a[0])) * float(seg[0]) + (py - float(a[1])) * float(seg[1])) / seg_len_sq
        t = max(0.0, min(1.0, float(t)))
        center = a + t * seg
        dist = float(np.linalg.norm(np.array([px, py], dtype=float) - center))
        if dist >= best_dist:
            cumulative_arc_m += seg_len
            continue
        left_point = left[idx] + t * (left[idx + 1] - left[idx])
        right_point = right[idx] + t * (right[idx + 1] - right[idx])
        span = left_point - right_point
        width_m = float(np.linalg.norm(span))
        if width_m > 1e-9:
            lateral_dir = span / width_m
            center_point = (left_point + right_point) * 0.5
        else:
            nx = -float(seg[1]) / seg_len
            ny = float(seg[0]) / seg_len
            lateral_dir = np.array([nx, ny], dtype=float)
            center_point = np.array(center, copy=True, dtype=float)
        offset_m = float(np.dot(np.array([px, py], dtype=float) - center_point, lateral_dir))
        best_dist = dist
        best = _CorridorProjection(
            center_point=np.asarray(center_point, dtype=float),
            left_point=np.asarray(left_point, dtype=float),
            right_point=np.asarray(right_point, dtype=float),
            lateral_dir=np.asarray(lateral_dir, dtype=float),
            center_offset_m=offset_m,
            width_m=width_m,
            arc_m=float(cumulative_arc_m + t * seg_len),
        )
        cumulative_arc_m += seg_len
    return best


def _safe_half_width_from_span(span_m: float) -> float:
    dynamic_half_width = (0.5 * max(0.0, float(span_m))) - _VEHICLE_HALF_WIDTH_M - float(_BEHAVIOR_CONTAINMENT_CLEARANCE_M)
    return max(0.0, min(_SAFE_HALF_WIDTH_M, dynamic_half_width))


def _corridor_offset_limit_m(
    span_m: float,
    *,
    visual_path_priority_active: bool,
    visual_requested_shift_m: float,
) -> float:
    safe_half_width_m = _safe_half_width_from_span(span_m)
    if not visual_path_priority_active:
        return safe_half_width_m

    relaxed_half_width_m = (0.5 * max(0.0, float(span_m))) - float(_BEHAVIOR_CONTAINMENT_CLEARANCE_M)
    if relaxed_half_width_m <= safe_half_width_m + 1e-9:
        return safe_half_width_m

    extra_allowance_m = abs(float(visual_requested_shift_m))
    if extra_allowance_m <= 1e-9:
        return safe_half_width_m

    return max(
        safe_half_width_m,
        min(relaxed_half_width_m, safe_half_width_m + extra_allowance_m),
    )


def _synthetic_bounds_from_xy(
    samples_xy: np.ndarray,
    *,
    half_width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(samples_xy, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty
    pseudo_path = np.column_stack([arr, _compute_headings(arr, initial_yaw=0.0)])
    return _build_drivable_bounds(pseudo_path, half_width_m=half_width_m)
