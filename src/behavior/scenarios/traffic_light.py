"""Scenario de luz semáforo (RED/YELLOW/GREEN).

Plan C1.3: la lógica de traffic light estaba en ``regulators`` del route
planner, dispersa entre varios módulos. Acá la centralizamos como un
scenario explícito que el ``PlannerManager`` invoca con prioridad 92
(por encima de StopSign=90 porque una luz roja es más urgente que un
stop sign: el stop tiene hold finito de 3s, la luz puede ser indefinida
hasta que cambie).

Inputs:
  * ``ctx.sign_hints`` con ``kind in {red_light, yellow_light, green_light}``.
  * ``ctx.pose.speed_mps`` para detectar si el ego ya frenó.

Output:
  * RED_LIGHT a < react_range_m → speed=0 hasta GREEN_LIGHT o max_wait_s.
  * YELLOW_LIGHT → desacelera a yellow_speed_mps (crawl).
  * GREEN_LIGHT → libera (is_active=False).

Decisiones explícitas:
  * NO copiar HSV color discrimination del REF. Mi YOLO multiclase ya
    publica las clases red/yellow/green; usarlas directo es más robusto
    bajo iluminación variable.
  * No requiere confirmación LiDAR (las luces no tienen firma lidar útil).
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.behavior.sign_utils import normalize_sign_kind, sign_hints_for
from src.core.runtime_params import params as _runtime_params
from src.core.types.behavior import BehaviorOutput, MotionState, ScenarioName


_DEFAULT_REACT_RANGE_M = 4.0
_DEFAULT_RED_SPEED_MPS = 0.0
_DEFAULT_YELLOW_SPEED_MPS = 0.05
_DEFAULT_MAX_WAIT_S = 15.0
_DEFAULT_RELEASE_SUPPRESS_S = 2.0  # post-green, no re-armar 2s

_RED_KINDS = frozenset({"red_light"})
_YELLOW_KINDS = frozenset({"yellow_light"})
_GREEN_KINDS = frozenset({"green_light"})
_ANY_LIGHT_KINDS = _RED_KINDS | _YELLOW_KINDS | _GREEN_KINDS


class TrafficLight(BaseScenario):
    """Scenario de semáforo. Activo cuando hay una luz visible cerca."""

    name = ScenarioName.TRAFFIC_LIGHT.value
    # Más urgente que StopSign (90) porque puede ser indefinido. Más
    # urgente que cualquier scenario de marcha. Pedestrian (95) lo supera.
    priority = 92

    def __init__(self) -> None:
        self._waiting_since_s: float = 0.0
        self._suppress_until_s: float = 0.0
        self._last_seen_kind: str = ""
        self._last_seen_distance_m: float | None = None

    def is_active(self, ctx: PlanningContext) -> bool:
        now = float(ctx.now_s)
        # Post-green suppress: si liberamos hace poco, no re-armar.
        if self._suppress_until_s > now:
            return False
        nearest = self._nearest_light(ctx)
        if nearest is None:
            return False
        kind = normalize_sign_kind(nearest.get("kind") or "")
        self._last_seen_kind = kind
        self._last_seen_distance_m = self._distance_of(nearest)

        if kind in _RED_KINDS or kind in _YELLOW_KINDS:
            if self._waiting_since_s == 0.0:
                self._waiting_since_s = now
            return True
        if kind in _GREEN_KINDS:
            # Llega un GREEN_LIGHT — soltar el hold (si lo había) y
            # supresar 2s para evitar oscilaciones por detección parcial.
            self._suppress_until_s = max(self._suppress_until_s, now + _DEFAULT_RELEASE_SUPPRESS_S)
            self._waiting_since_s = 0.0
            return False
        return False

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        now = float(ctx.now_s)
        kind = self._last_seen_kind or ""
        distance = self._last_seen_distance_m
        max_wait_s = self._param_float("traffic_light.max_wait_s", _DEFAULT_MAX_WAIT_S)
        yellow_speed = self._param_float(
            "traffic_light.yellow_speed_mps", _DEFAULT_YELLOW_SPEED_MPS
        )
        red_speed = self._param_float("traffic_light.red_speed_mps", _DEFAULT_RED_SPEED_MPS)

        # Failsafe: si esperamos demasiado, asumir luz muerta y avanzar.
        if (
            self._waiting_since_s > 0.0
            and (now - self._waiting_since_s) >= max_wait_s
        ):
            self._suppress_until_s = now + _DEFAULT_RELEASE_SUPPRESS_S
            self._waiting_since_s = 0.0
            plan = self._build_constant_speed_plan(
                ctx=ctx,
                target_speed_mps=max(0.0, float(ctx.nominal_speed_mps)),
                scenario_name=self.name,
                notes={
                    "reason": "traffic_light_timeout_failsafe",
                    "waited_s": float(max_wait_s),
                    "kind": kind,
                },
            )
            return replace(plan, motion_state=MotionState.MOVING.value)

        if kind in _YELLOW_KINDS:
            plan = self._build_constant_speed_plan(
                ctx=ctx,
                target_speed_mps=float(yellow_speed),
                scenario_name=self.name,
                notes={
                    "reason": "traffic_light_yellow",
                    "kind": kind,
                    "distance_m": float(distance) if distance is not None else None,
                },
            )
            return replace(plan, motion_state=MotionState.WAITING_FOR_TRAFFIC_LIGHT.value)

        # RED por default si llegamos acá.
        plan = self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=float(red_speed),
            scenario_name=self.name,
            notes={
                "reason": "traffic_light_red",
                "kind": kind or "red_light",
                "distance_m": float(distance) if distance is not None else None,
                "waited_s": float(now - self._waiting_since_s)
                if self._waiting_since_s > 0.0
                else 0.0,
            },
        )
        return replace(
            plan,
            stop_required=True,
            motion_state=MotionState.WAITING_FOR_TRAFFIC_LIGHT.value,
        )

    # ─── Helpers ──────────────────────────────────────────────────────
    def _nearest_light(self, ctx: PlanningContext) -> dict | None:
        max_range = self._param_float("traffic_light.react_range_m", _DEFAULT_REACT_RANGE_M)
        candidates: list[dict] = []
        for hint in sign_hints_for(ctx, _ANY_LIGHT_KINDS, max_distance_m=max_range):
            dist = self._distance_of(hint)
            if dist is None or dist > max_range:
                continue
            candidates.append({**hint, "distance_m": dist})
        if not candidates:
            return None
        return min(candidates, key=lambda h: float(h["distance_m"]))

    @staticmethod
    def _distance_of(hint: dict | None) -> float | None:
        if not isinstance(hint, dict):
            return None
        value = hint.get("distance_m")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _param_float(key: str, default: float) -> float:
        try:
            value = _runtime_params.get(key, default)
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return float(default)
