from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.behavior.context import PlanningContext
from src.core.types.behavior import BehaviorOutput, BehaviorPathPlan

_STOPLINE_STOP_RANGE_M = 5.0
_CROSSWALK_SLOWDOWN_RANGE_M = 4.0
_CROSSWALK_SPEED_MPS = 0.30
_INTERSECTION_SPEED_MPS = 0.40
_INTERSECTION_RANGE_M = 6.0
_CURVATURE_A_LAT_MAX_MPS2 = 0.45
_CURVATURE_SPEED_FLOOR_MPS = 0.05


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
    ) -> VelocityRuleResult:
        capped = np.minimum(speed_profile, float(ctx.max_speed_mps))
        triggered = bool(np.any(capped < speed_profile - 1e-9))
        return VelocityRuleResult(
            speed_profile=capped,
            stop_required=bool(stop_required),
            notes={"kind": self.name, "cap_mps": float(ctx.max_speed_mps)},
            triggered=triggered,
        )


class CurvatureConstraintRule:
    name = "curvature_constraint"

    def apply(
        self,
        *,
        speed_profile: np.ndarray,
        target_path: np.ndarray,
        ctx: PlanningContext,
        stop_required: bool,
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
                CurvatureConstraintRule(),
                RegulatoryElementRule(),
            ]
        )

    def build_output(
        self,
        *,
        path_plan: BehaviorPathPlan,
        target_path: np.ndarray,
        drivable_left_bound: np.ndarray,
        drivable_right_bound: np.ndarray,
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
            )
            speed = np.asarray(result.speed_profile, dtype=float)
            stop_required = bool(result.stop_required)
            if result.triggered:
                notes["velocity_modules"].append(result.notes)

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
