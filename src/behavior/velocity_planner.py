from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.behavior.context import PlanningContext
from src.core.types.behavior import BehaviorOutput, BehaviorPathPlan
from src.utils.live_log import live_log

_STOPLINE_STOP_RANGE_M = 5.0
_CROSSWALK_SLOWDOWN_RANGE_M = 4.0
_CROSSWALK_SPEED_MPS = 0.30
_INTERSECTION_SPEED_MPS = 0.40
_INTERSECTION_RANGE_M = 6.0
_CURVATURE_A_LAT_MAX_MPS2 = 0.45

try:
    from config import BEHAVIOR_MIN_SPEED_MPS as _BEHAVIOR_MIN_SPEED_MPS
except Exception:
    _BEHAVIOR_MIN_SPEED_MPS = 0.20

_CURVATURE_SPEED_FLOOR_MPS = float(_BEHAVIOR_MIN_SPEED_MPS)
_LANE_CONTAINMENT_WARN_CAP_MPS = float(_BEHAVIOR_MIN_SPEED_MPS)

try:
    from config import (
        BEHAVIOR_CONTAINMENT_CRAWL_ERROR_M as _BEHAVIOR_CONTAINMENT_CRAWL_ERROR_M,
    )
except Exception:
    _BEHAVIOR_CONTAINMENT_CRAWL_ERROR_M = 0.07

try:
    from config import (
        BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS as _BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS,
    )
except Exception:
    _BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS = 0.04

try:
    from config import (
        BEHAVIOR_CONTAINMENT_WARN_ERROR_M as _BEHAVIOR_CONTAINMENT_WARN_ERROR_M,
    )
except Exception:
    _BEHAVIOR_CONTAINMENT_WARN_ERROR_M = 0.05

try:
    from config import (
        BEHAVIOR_CONTAINMENT_STUCK_TICKS as _BEHAVIOR_CONTAINMENT_STUCK_TICKS,
    )
except Exception:
    _BEHAVIOR_CONTAINMENT_STUCK_TICKS = 40

try:
    from config import (
        BEHAVIOR_CONTAINMENT_RECOVERY_SPEED_MPS as _BEHAVIOR_CONTAINMENT_RECOVERY_SPEED_MPS,
    )
except Exception:
    _BEHAVIOR_CONTAINMENT_RECOVERY_SPEED_MPS = 0.08

try:
    from config import (
        BEHAVIOR_VISUAL_INSTABILITY_CRAWL_SPEED_MPS as _BEHAVIOR_VISUAL_INSTABILITY_CRAWL_SPEED_MPS,
    )
except Exception:
    _BEHAVIOR_VISUAL_INSTABILITY_CRAWL_SPEED_MPS = 0.12

try:
    from config import (
        BEHAVIOR_VISUAL_INSTABILITY_FLIP_HOLD_TICKS as _BEHAVIOR_VISUAL_INSTABILITY_FLIP_HOLD_TICKS,
    )
except Exception:
    _BEHAVIOR_VISUAL_INSTABILITY_FLIP_HOLD_TICKS = 8

try:
    from config import (
        BEHAVIOR_VISUAL_INSTABILITY_MIN_ERROR_M as _BEHAVIOR_VISUAL_INSTABILITY_MIN_ERROR_M,
    )
except Exception:
    _BEHAVIOR_VISUAL_INSTABILITY_MIN_ERROR_M = 0.04

try:
    from config import (
        BEHAVIOR_VISUAL_BOUNDARY_CRAWL_ERROR_M as _BEHAVIOR_VISUAL_BOUNDARY_CRAWL_ERROR_M,
    )
except Exception:
    _BEHAVIOR_VISUAL_BOUNDARY_CRAWL_ERROR_M = 0.10

# Histeresis para que el error tenga que decrecer al menos 5 mm por tick
# para considerarse "el robot está recuperando"; valores menores son ruido.
_LANE_CONTAINMENT_DECREASE_EPS_M = 0.005


@dataclass(frozen=True)
class VelocityRuleResult:
    speed_profile: np.ndarray
    stop_required: bool
    notes: dict
    triggered: bool


class VelocityRule(Protocol):
    name: str

    def apply(
        self,
        *,
        speed_profile: np.ndarray,
        target_path: np.ndarray,
        ctx: PlanningContext,
        stop_required: bool,
        planning_notes: dict,
    ) -> VelocityRuleResult: ...


class GlobalSpeedCapRule:
    name = "global_speed_cap"

    def apply(
        self,
        *,
        speed_profile: np.ndarray,
        target_path: np.ndarray,
        ctx: PlanningContext,
        stop_required: bool,
        planning_notes: dict,
    ) -> VelocityRuleResult:
        capped = np.minimum(speed_profile, float(ctx.max_speed_mps))
        triggered = bool(np.any(capped < speed_profile - 1e-9))
        return VelocityRuleResult(
            speed_profile=capped,
            stop_required=bool(stop_required),
            notes={"kind": self.name, "cap_mps": float(ctx.max_speed_mps)},
            triggered=triggered,
        )


class CompetitionSpeedBoundsRule:
    name = "competition_speed_bounds"

    def apply(
        self,
        *,
        speed_profile: np.ndarray,
        target_path: np.ndarray,
        ctx: PlanningContext,
        stop_required: bool,
        planning_notes: dict,
    ) -> VelocityRuleResult:
        allow_sub_min_speed = _planning_notes_allow_sub_min_speed(planning_notes)
        bounded = _apply_competition_speed_bounds(
            speed_profile,
            min_speed_mps=(0.0 if allow_sub_min_speed else float(_BEHAVIOR_MIN_SPEED_MPS)),
            max_speed_mps=float(ctx.max_speed_mps),
        )
        triggered = bool(np.any(np.abs(bounded - np.asarray(speed_profile, dtype=float)) > 1e-9))
        return VelocityRuleResult(
            speed_profile=bounded,
            stop_required=bool(stop_required),
            notes={
                "kind": self.name,
                "min_moving_mps": (0.0 if allow_sub_min_speed else float(_BEHAVIOR_MIN_SPEED_MPS)),
                "allow_sub_min_speed": bool(allow_sub_min_speed),
                "max_mps": float(ctx.max_speed_mps),
            },
            triggered=triggered,
        )


class LaneContainmentRule:
    name = "lane_containment"

    def __init__(self) -> None:
        # Estado para detectar el robot atascado en crawl.
        # Si el error lateral no decrece por _BEHAVIOR_CONTAINMENT_STUCK_TICKS
        # ticks consecutivos en crawl, escalamos a recovery_speed para que
        # la cinemática (v · tan δ / L) pueda recuperar el centerline.
        self._prev_effective_error_m: float = 0.0
        self._stuck_tick_count: int = 0
        self._recovery_active: bool = False
        self._prev_visual_side: str = "none"
        self._visual_instability_hold_ticks: int = 0

    def apply(
        self,
        *,
        speed_profile: np.ndarray,
        target_path: np.ndarray,
        ctx: PlanningContext,
        stop_required: bool,
        planning_notes: dict,
    ) -> VelocityRuleResult:
        speed = np.array(speed_profile, copy=True, dtype=float)
        visual_error_m = _visual_lane_error_m(ctx, planning_notes)
        visual_side = _visual_lane_side(ctx, planning_notes)
        visual_side_flip = _visual_lane_side_flip(
            previous_side=self._prev_visual_side,
            current_side=visual_side,
            planning_notes=planning_notes,
        )
        if visual_side_flip:
            self._visual_instability_hold_ticks = int(_BEHAVIOR_VISUAL_INSTABILITY_FLIP_HOLD_TICKS)
        elif self._visual_instability_hold_ticks > 0:
            self._visual_instability_hold_ticks -= 1
        if visual_side in {"left", "right", "both"}:
            self._prev_visual_side = visual_side

        corridor_error_m = abs(float(planning_notes.get("ego_corridor_error_m", 0.0) or 0.0))
        effective_error_m = max(corridor_error_m, abs(visual_error_m) if visual_error_m is not None else 0.0)
        touches_bound = bool(planning_notes.get("corridor_touches_bound", False))
        used_prev_safe_path = bool(planning_notes.get("used_prev_safe_path", False))
        infeasible_ticks = int(planning_notes.get("containment_infeasible_ticks", 0) or 0)
        stop_after_ticks = int(planning_notes.get("containment_stop_after_ticks", 4) or 4)
        first_infeasible_index = planning_notes.get("first_infeasible_index")

        trigger_warn = effective_error_m >= float(_BEHAVIOR_CONTAINMENT_WARN_ERROR_M)
        trigger_crawl = (
            effective_error_m >= float(_BEHAVIOR_CONTAINMENT_CRAWL_ERROR_M)
            or touches_bound
            or used_prev_safe_path
        )
        trigger_visual_instability_crawl = (
            self._visual_instability_hold_ticks > 0
            and visual_error_m is not None
            and abs(float(visual_error_m)) >= float(_BEHAVIOR_VISUAL_INSTABILITY_MIN_ERROR_M)
        )
        trigger_visual_boundary_crawl = (
            visual_error_m is not None
            and abs(float(visual_error_m)) >= float(_BEHAVIOR_VISUAL_BOUNDARY_CRAWL_ERROR_M)
            and _visual_lane_measurement_source(planning_notes) in {
                "single_line_boundary_hint",
                "line_center_offset_m",
                "direct_error_m",
            }
        )
        trigger_crawl = trigger_crawl or trigger_visual_instability_crawl or trigger_visual_boundary_crawl

        # Tracking de stuck-recovery. Solo cuenta cuando estamos en crawl;
        # fuera de crawl, reset total.
        if trigger_crawl:
            decreased = effective_error_m < (
                self._prev_effective_error_m - _LANE_CONTAINMENT_DECREASE_EPS_M
            )
            if decreased:
                self._stuck_tick_count = 0
            else:
                self._stuck_tick_count += 1
        else:
            self._stuck_tick_count = 0
            self._recovery_active = False

        # Entrar a recovery si llevamos demasiados ticks atascados.
        recovery_was_active = self._recovery_active
        if self._stuck_tick_count >= int(_BEHAVIOR_CONTAINMENT_STUCK_TICKS):
            self._recovery_active = True

        # Salir de recovery solo cuando el error baja del threshold de crawl
        # (histeresis: no oscilar en el límite).
        if self._recovery_active and effective_error_m < float(_BEHAVIOR_CONTAINMENT_CRAWL_ERROR_M):
            self._recovery_active = False
            self._stuck_tick_count = 0

        if self._recovery_active != recovery_was_active:
            live_log(
                "velocity_planner",
                event="containment_recovery",
                active=bool(self._recovery_active),
                effective_error_m=float(effective_error_m),
                stuck_tick_count=int(self._stuck_tick_count),
                crawl_speed_mps=float(_BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS),
                recovery_speed_mps=float(_BEHAVIOR_CONTAINMENT_RECOVERY_SPEED_MPS),
            )

        self._prev_effective_error_m = float(effective_error_m)

        if not trigger_warn and not trigger_crawl and infeasible_ticks <= 0:
            return VelocityRuleResult(speed, bool(stop_required), {}, False)

        stop_req = bool(stop_required)
        note: dict[str, object] = {
            "kind": self.name,
            "effective_error_m": float(effective_error_m),
            "corridor_error_m": float(corridor_error_m),
            "visual_error_m": float(visual_error_m) if visual_error_m is not None else None,
            "visual_side": str(visual_side),
            "visual_side_flip": bool(visual_side_flip),
            "visual_instability_hold_ticks": int(self._visual_instability_hold_ticks),
            "touches_bound": bool(touches_bound),
            "used_prev_safe_path": bool(used_prev_safe_path),
            "containment_infeasible_ticks": int(infeasible_ticks),
            "stuck_tick_count": int(self._stuck_tick_count),
        }

        if (
            infeasible_ticks >= stop_after_ticks
            and first_infeasible_index is not None
            and speed.shape[0] > 0
        ):
            zero_from_idx = max(0, min(speed.shape[0] - 1, int(first_infeasible_index) - 1))
            if zero_from_idx <= 0:
                speed[:] = 0.0
                stop_req = True
                note["mode"] = "stop"
                note["zero_from_idx"] = int(zero_from_idx)
                return VelocityRuleResult(speed, stop_req, note, True)
            speed = np.minimum(speed, float(_BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS))
            note["mode"] = "future_horizon_crawl"
            note["zero_from_idx"] = int(zero_from_idx)
            note["cap_mps"] = float(_BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS)
            note["allow_sub_min_speed"] = True
            return VelocityRuleResult(speed, stop_req, note, True)

        if self._recovery_active:
            cap_mps = float(_BEHAVIOR_CONTAINMENT_RECOVERY_SPEED_MPS)
            note["mode"] = "stuck_recovery"
        elif trigger_visual_instability_crawl and not (touches_bound or used_prev_safe_path or infeasible_ticks > 0):
            cap_mps = float(_BEHAVIOR_VISUAL_INSTABILITY_CRAWL_SPEED_MPS)
            note["mode"] = "visual_instability_crawl"
            note["allow_sub_min_speed"] = True
        elif trigger_visual_boundary_crawl and not (touches_bound or used_prev_safe_path or infeasible_ticks > 0):
            cap_mps = float(_BEHAVIOR_VISUAL_INSTABILITY_CRAWL_SPEED_MPS)
            note["mode"] = "visual_boundary_crawl"
            note["allow_sub_min_speed"] = True
        elif trigger_crawl or infeasible_ticks > 0:
            cap_mps = float(_BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS)
            note["mode"] = "crawl"
            note["allow_sub_min_speed"] = True
        else:
            cap_mps = float(_LANE_CONTAINMENT_WARN_CAP_MPS)
            note["mode"] = "warn"
        note["cap_mps"] = float(cap_mps)
        speed = np.minimum(speed, cap_mps)
        return VelocityRuleResult(speed, stop_req, note, True)


class CurvatureConstraintRule:
    name = "curvature_constraint"

    def apply(
        self,
        *,
        speed_profile: np.ndarray,
        target_path: np.ndarray,
        ctx: PlanningContext,
        stop_required: bool,
        planning_notes: dict,
    ) -> VelocityRuleResult:
        kappas = _compute_curvature(target_path)
        if kappas.size == 0:
            return VelocityRuleResult(speed_profile, bool(stop_required), {}, False)
        caps = np.full(speed_profile.shape[0], np.inf, dtype=float)
        n = min(speed_profile.shape[0], kappas.shape[0])
        for idx in range(n):
            kappa = abs(float(kappas[idx]))
            if kappa < 1e-4:
                continue
            caps[idx] = max(
                _CURVATURE_SPEED_FLOOR_MPS,
                math.sqrt(_CURVATURE_A_LAT_MAX_MPS2 / kappa),
            )
        constrained = np.minimum(speed_profile, caps)
        triggered = bool(np.any(constrained < speed_profile - 1e-9))
        return VelocityRuleResult(
            speed_profile=constrained,
            stop_required=bool(stop_required),
            notes={
                "kind": self.name,
                "peak_abs_kappa": float(np.max(np.abs(kappas))) if kappas.size else 0.0,
                "a_lat_max_mps2": _CURVATURE_A_LAT_MAX_MPS2,
            },
            triggered=triggered,
        )


class RegulatoryElementRule:
    name = "regulatory_elements"

    def apply(
        self,
        *,
        speed_profile: np.ndarray,
        target_path: np.ndarray,
        ctx: PlanningContext,
        stop_required: bool,
        planning_notes: dict,
    ) -> VelocityRuleResult:
        speed = np.array(speed_profile, copy=True, dtype=float)
        notes: list[dict] = []
        stop_req = bool(stop_required)

        for reg in ctx.route.regulatory_ahead:
            kind = str(reg.kind or "").lower()
            distance_m = float(reg.data.get("distance_m", 0.0)) if reg.data else 0.0

            if kind == "stopline" and distance_m <= _STOPLINE_STOP_RANGE_M:
                obs_dist = getattr(ctx.stopline_observation, "distance_m", None)
                if obs_dist is not None:
                    try:
                        distance_m = min(distance_m, float(obs_dist))
                    except (TypeError, ValueError):
                        pass
                speed = _ramp_to_zero(speed, dt=ctx.dt, distance_to_stop_m=distance_m)
                stop_req = True
                notes.append({"kind": kind, "distance_m": float(distance_m)})
            elif kind == "crosswalk" and distance_m <= _CROSSWALK_SLOWDOWN_RANGE_M:
                speed = np.minimum(speed, _CROSSWALK_SPEED_MPS)
                notes.append(
                    {"kind": kind, "distance_m": float(distance_m), "cap_mps": _CROSSWALK_SPEED_MPS}
                )
            elif kind == "intersection" and distance_m <= _INTERSECTION_RANGE_M:
                speed = np.minimum(speed, _INTERSECTION_SPEED_MPS)
                notes.append(
                    {"kind": kind, "distance_m": float(distance_m), "cap_mps": _INTERSECTION_SPEED_MPS}
                )
            elif kind == "speed_limit":
                limit = float(reg.data.get("speed_mps", ctx.max_speed_mps))
                speed = np.minimum(speed, limit)
                notes.append({"kind": kind, "cap_mps": limit})

        return VelocityRuleResult(
            speed_profile=speed,
            stop_required=stop_req,
            notes={"kind": self.name, "hits": notes},
            triggered=bool(notes),
        )


class BehaviorVelocityPlanner:
    """Genera el perfil de velocidad final para el MPC.

    Toma el path optimizado del behavior planner y aplica módulos de velocidad
    independientes, inspirados en `behavior_velocity_planner` y
    `motion_velocity_planner` de Autoware:
      - cap global del vehículo,
      - restricción por curvatura,
      - reglas regulatorias del mapa (stopline/crosswalk/intersection/etc.).
    """

    def __init__(self, rules: list[VelocityRule] | None = None) -> None:
        self._rules: list[VelocityRule] = list(
            rules
            or [
                GlobalSpeedCapRule(),
                LaneContainmentRule(),
                CurvatureConstraintRule(),
                RegulatoryElementRule(),
                CompetitionSpeedBoundsRule(),
            ]
        )

    def build_output(
        self,
        *,
        path_plan: BehaviorPathPlan,
        target_path: np.ndarray,
        drivable_left_bound: np.ndarray,
        drivable_right_bound: np.ndarray,
        optimizer_notes: dict | None = None,
        ctx: PlanningContext,
    ) -> BehaviorOutput:
        speed = _fit_speed_profile(
            base_speed_profile=path_plan.base_speed_profile,
            horizon_n=ctx.horizon_n,
            fallback_speed_mps=ctx.nominal_speed_mps,
            force_stop=bool(path_plan.stop_required),
        )
        stop_required = bool(path_plan.stop_required)
        notes = dict(path_plan.notes)
        if optimizer_notes:
            notes.update(dict(optimizer_notes))
        notes["turn_signal"] = str(path_plan.turn_signal)
        notes["drivable_left_bound"] = np.asarray(drivable_left_bound, dtype=float)
        notes["drivable_right_bound"] = np.asarray(drivable_right_bound, dtype=float)
        notes.setdefault("velocity_modules", [])

        for rule in self._rules:
            result = rule.apply(
                speed_profile=speed,
                target_path=target_path,
                ctx=ctx,
                stop_required=stop_required,
                planning_notes=notes,
            )
            speed = np.asarray(result.speed_profile, dtype=float)
            stop_required = bool(result.stop_required)
            if result.triggered:
                notes["velocity_modules"].append(result.notes)
                if bool(result.notes.get("allow_sub_min_speed", False)):
                    notes["allow_sub_min_speed"] = True

        return BehaviorOutput(
            timestamp=float(path_plan.timestamp or ctx.now_s),
            dt=float(ctx.dt),
            target_path=np.asarray(target_path, dtype=float),
            speed_profile=speed,
            scenario_name=str(path_plan.scenario_name),
            valid=bool(path_plan.valid),
            stop_required=bool(stop_required),
            notes=notes,
        )


def _fit_speed_profile(
    *,
    base_speed_profile: np.ndarray,
    horizon_n: int,
    fallback_speed_mps: float,
    force_stop: bool,
) -> np.ndarray:
    if horizon_n <= 0:
        return np.zeros(0, dtype=float)
    if force_stop:
        return np.zeros(horizon_n, dtype=float)

    speed = np.asarray(base_speed_profile, dtype=float).reshape(-1)
    if speed.size == 0:
        return np.full(horizon_n, float(fallback_speed_mps), dtype=float)
    if speed.size == horizon_n:
        return np.array(speed, copy=True)
    if speed.size == 1:
        return np.full(horizon_n, float(speed[0]), dtype=float)

    x_old = np.linspace(0.0, 1.0, num=speed.size)
    x_new = np.linspace(0.0, 1.0, num=horizon_n)
    return np.interp(x_new, x_old, speed).astype(float)


def _apply_competition_speed_bounds(
    speed_profile: np.ndarray,
    *,
    min_speed_mps: float,
    max_speed_mps: float,
) -> np.ndarray:
    """Keep moving commands inside competition bounds while preserving stops."""
    speed = np.asarray(speed_profile, dtype=float)
    bounded = np.minimum(np.array(speed, copy=True), float(max_speed_mps))
    moving = bounded > 1e-6
    effective_min_mps = min(float(min_speed_mps), float(max_speed_mps))
    bounded[moving] = np.maximum(bounded[moving], effective_min_mps)
    return bounded


def _ramp_to_zero(speed: np.ndarray, *, dt: float, distance_to_stop_m: float) -> np.ndarray:
    n = int(speed.shape[0])
    if n == 0 or distance_to_stop_m <= 0.0:
        return np.zeros_like(speed)

    out = np.array(speed, copy=True, dtype=float)
    cumulative = 0.0
    stop_step = n - 1
    for idx in range(n):
        cumulative += max(0.0, float(out[idx])) * float(dt)
        if cumulative >= float(distance_to_stop_m):
            stop_step = idx
            break
    ramp_end = max(0, min(stop_step, n - 1))
    v0 = max(0.0, float(out[0]))
    ramp = np.linspace(v0, 0.0, ramp_end + 1)
    out[: ramp_end + 1] = np.minimum(out[: ramp_end + 1], ramp)
    out[ramp_end + 1 :] = 0.0
    return out


def _planning_notes_allow_sub_min_speed(planning_notes: dict) -> bool:
    if bool(planning_notes.get("allow_sub_min_speed", False)):
        return True
    modules = planning_notes.get("velocity_modules")
    if not isinstance(modules, list):
        return False
    return any(
        isinstance(note, dict) and bool(note.get("allow_sub_min_speed", False))
        for note in modules
    )


def _visual_lane_side(ctx: PlanningContext, planning_notes: dict) -> str:
    note_side = str(planning_notes.get("visual_lane_detected_side", "") or "")
    if note_side in {"left", "right", "both"}:
        return note_side
    lane_observation = getattr(ctx, "lane_observation", None)
    detected_sides = tuple(getattr(lane_observation, "detected_sides", ()) or ())
    if len(detected_sides) >= 2:
        return "both"
    if len(detected_sides) == 1 and detected_sides[0] in {"left", "right"}:
        return str(detected_sides[0])
    return "none"


def _visual_lane_side_flip(
    *,
    previous_side: str,
    current_side: str,
    planning_notes: dict,
) -> bool:
    if bool(planning_notes.get("visual_lane_shift_side_flip", False)) or bool(
        planning_notes.get("visual_lane_side_switch_pending", False)
    ):
        return True
    return (
        previous_side in {"left", "right"}
        and current_side in {"left", "right"}
        and previous_side != current_side
    )


def _visual_lane_measurement_source(planning_notes: dict) -> str:
    return str(planning_notes.get("visual_lane_measurement_source", "") or "")


def _visual_lane_error_m(ctx: PlanningContext, planning_notes: dict | None = None) -> float | None:
    planning_notes = planning_notes if isinstance(planning_notes, dict) else {}
    note_source = str(planning_notes.get("visual_lane_measurement_source", "") or "")
    note_mode = str(planning_notes.get("visual_lane_measurement_mode", "") or "")
    if note_source in {
        "line_center_offset_m",
        "direct_error_m",
        "single_line_boundary_hint",
        "visual_waypoint_center",
        "lateral_offset_m",
    }:
        value = planning_notes.get("visual_lane_error_m")
        try:
            error_m = float(value)
        except (TypeError, ValueError):
            error_m = None
        if error_m is not None and math.isfinite(error_m):
            quality = _finite_float(planning_notes.get("visual_lane_quality"), default=1.0)
            min_quality = 0.55 if note_mode == "single_line" else 0.75
            if quality >= min_quality:
                return error_m

    lane_observation = getattr(ctx, "lane_observation", None)
    if lane_observation is None:
        return None
    if not bool(getattr(lane_observation, "direct_error_valid", False)):
        return None
    measurement_mode = str(getattr(lane_observation, "measurement_mode", "none"))
    if measurement_mode not in {"two_line", "single_line"}:
        return None
    min_quality = 0.8 if measurement_mode == "two_line" else 0.60
    if float(getattr(lane_observation, "quality", 0.0) or 0.0) < min_quality:
        return None
    value = getattr(lane_observation, "direct_error_m", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite_float(value: object, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _compute_curvature(target_path: np.ndarray) -> np.ndarray:
    if target_path.shape[0] < 3:
        return np.zeros(max(0, target_path.shape[0] - 1), dtype=float)
    xy = np.asarray(target_path[:, :2], dtype=float)
    kappas = np.zeros(target_path.shape[0] - 1, dtype=float)
    for idx in range(1, xy.shape[0] - 1):
        p0 = xy[idx - 1]
        p1 = xy[idx]
        p2 = xy[idx + 1]
        a = float(np.linalg.norm(p1 - p0))
        b = float(np.linalg.norm(p2 - p1))
        c = float(np.linalg.norm(p2 - p0))
        denom = max(a * b * c, 1e-9)
        area2 = abs(
            (p1[0] - p0[0]) * (p2[1] - p0[1])
            - (p1[1] - p0[1]) * (p2[0] - p0[0])
        )
        kappas[idx] = 2.0 * area2 / denom
    if kappas.size >= 2:
        kappas[0] = kappas[1]
    return kappas
