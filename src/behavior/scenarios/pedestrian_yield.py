"""Scenario de yielding a peatón fuera de crosswalk.

Plan C1.4: hoy el ``Crosswalk`` scenario sólo bloquea si hay peatón en la
zona de crosswalk activa. Un peatón cruzando una recta fuera de crosswalk
NO disparaba freno en mi repo. El REF también lo ignora — este scenario
cierra ese gap.

Prioridad **95** (más alta que cualquier otro) — pisar a alguien siempre es
lo peor. Sobreescribe StopSign, TrafficLight, Intersection.

Activación:
  * Hay ``TrackedObject.PEDESTRIAN`` con ``confidence >= 0.5``,
    ``age_frames >= 2``.
  * Proyección al path ego < ``PEDESTRIAN_YIELD_DISTANCE_M`` (2.5 m).
  * Ángulo respecto a heading ego < ``PEDESTRIAN_YIELD_HALF_FOV_DEG`` (45°).

Liberación:
  * Cuando el track no se ve por ≥ 0.4 s (age_frames > max_age o ausencia).
  * Cuando se sale del cono de detección.

Sólo aplica si el peatón está EN el cono del path. Si está al costado pero
caminando lejos, no freno (no false positive).
"""

from __future__ import annotations

import math
from dataclasses import replace

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.core.runtime_params import params as _runtime_params
from src.core.types.behavior import BehaviorOutput, MotionState, ScenarioName
from src.perception.object_classes import TrackedObject as _ObjectClass


_DEFAULT_YIELD_DISTANCE_M = 2.5
_DEFAULT_YIELD_HALF_FOV_DEG = 45.0
_DEFAULT_YIELD_MIN_CONFIDENCE = 0.50
_DEFAULT_YIELD_MIN_AGE_FRAMES = 2
_DEFAULT_RELEASE_GRACE_S = 0.4


class PedestrianYield(BaseScenario):
    """Yielding a peatón fuera de crosswalk. Prioridad máxima."""

    name = ScenarioName.PEDESTRIAN_YIELD.value
    priority = 95

    def __init__(self) -> None:
        self._yielding: bool = False
        self._last_seen_at_s: float = 0.0

    def is_active(self, ctx: PlanningContext) -> bool:
        now = float(ctx.now_s)
        any_relevant = self._has_relevant_pedestrian(ctx)
        if any_relevant:
            self._yielding = True
            self._last_seen_at_s = now
            return True
        # Grace window: liberar después de N segundos sin ver al peatón.
        if self._yielding:
            release_grace_s = self._param_float(
                "pedestrian.release_grace_s", _DEFAULT_RELEASE_GRACE_S
            )
            if (now - self._last_seen_at_s) >= release_grace_s:
                self._yielding = False
                return False
            return True
        return False

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        plan = self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=0.0,
            scenario_name=self.name,
            notes={
                "reason": "pedestrian_in_path",
                "yielding_active": self._yielding,
                "last_seen_at_s": float(self._last_seen_at_s),
            },
        )
        return replace(
            plan,
            stop_required=True,
            motion_state=MotionState.PEDESTRIAN_YIELDING.value,
        )

    # ─── Helpers ──────────────────────────────────────────────────────
    def _has_relevant_pedestrian(self, ctx: PlanningContext) -> bool:
        tracked = list(getattr(ctx, "tracked_objects", ()) or ())
        if not tracked:
            return False

        yield_dist = self._param_float("pedestrian.yield_distance_m", _DEFAULT_YIELD_DISTANCE_M)
        half_fov_deg = self._param_float(
            "pedestrian.yield_half_fov_deg", _DEFAULT_YIELD_HALF_FOV_DEG
        )
        min_conf = self._param_float(
            "pedestrian.yield_min_confidence", _DEFAULT_YIELD_MIN_CONFIDENCE
        )
        min_age = int(
            self._param_float("pedestrian.yield_min_age_frames", _DEFAULT_YIELD_MIN_AGE_FRAMES)
        )

        cos_max = math.cos(math.radians(half_fov_deg))
        ego_x = float(ctx.pose.fused_pose.x)
        ego_y = float(ctx.pose.fused_pose.y)
        ego_yaw = float(ctx.pose.fused_pose.yaw)
        # Vector unitario heading ego
        hx = math.cos(ego_yaw)
        hy = math.sin(ego_yaw)

        for obj in tracked:
            if not self._is_pedestrian(obj):
                continue
            if float(getattr(obj, "confidence", 0.0)) < min_conf:
                continue
            if int(getattr(obj, "age_frames", 0)) < min_age:
                continue
            pos = getattr(obj, "position_world_xy", None)
            if pos is None:
                continue
            try:
                dx = float(pos[0]) - ego_x
                dy = float(pos[1]) - ego_y
            except (TypeError, ValueError, IndexError):
                continue
            dist = math.hypot(dx, dy)
            if dist <= 1e-3 or dist > yield_dist:
                continue
            # Proyección sobre heading ego
            dot = (dx * hx + dy * hy) / dist
            if dot < cos_max:
                continue  # fuera del cono
            return True
        return False

    @staticmethod
    def _is_pedestrian(obj) -> bool:
        cid = int(getattr(obj, "class_id", -1))
        if cid == int(_ObjectClass.PEDESTRIAN):
            return True
        name = str(getattr(obj, "class_name", "")).lower()
        return name in {"pedestrian", "person"}

    @staticmethod
    def _param_float(key: str, default: float) -> float:
        try:
            return float(_runtime_params.get(key, default))
        except (TypeError, ValueError):
            return float(default)
