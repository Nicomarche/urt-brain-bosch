# src/behavior/scenarios/base.py
#
# `BaseScenario` — clase abstracta común. NO es la `IScenario` Protocol
# de `core/interfaces/planner.py`; es una implementación parcial que
# centraliza utilities compartidas (construcción de speed_profile uniforme,
# manejo de fallback BehaviorOutput, helpers de histeresis).
#
# Las subclases concretas (LaneKeep, Intersection, etc.) heredan de
# `BaseScenario` Y satisfacen el Protocol `IScenario` (estructural).
# Esto da SRP (BaseScenario = utilities) + DIP (planner depende de
# IScenario, no de BaseScenario).

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.behavior.context import PlanningContext
from src.behavior.trajectory_builder import (
    build_target_path,
    build_target_path_from_route,
    build_target_path_from_visual,
)
from src.behavior.visual_primary_gate import VisualPrimaryGate
from src.core.types.behavior import BehaviorOutput, ScenarioName


@dataclass
class HysteresisGate:
    """Gate con histeresis para activación/desactivación estable.

    Uso:
        gate = HysteresisGate(enter=0.4, exit=0.6)
        gate.update(measure)  # recibe valor a comparar
        if gate.active: ...    # estado actual con histeresis aplicada

    Convención: `enter` ≤ `exit`. El gate se activa cuando el valor
    cae bajo `enter`, se desactiva cuando supera `exit`. Útil para
    señales tipo "distancia al stopline" donde queremos activar al
    acercarnos y desactivar cuando ya pasamos.
    """

    enter: float
    exit: float
    active: bool = False

    def update(self, value: float) -> bool:
        if self.active and value > self.exit:
            self.active = False
        elif (not self.active) and value < self.enter:
            self.active = True
        return self.active


class BaseScenario(ABC):
    """Esqueleto común para escenarios de comportamiento.

    Subclases DEBEN definir:
      - `name: str` (tipicamente `ScenarioName.X.value`)
      - `priority: int` (mayor = se evalúa primero)
      - `is_active(ctx) -> bool`
      - `plan(ctx) -> BehaviorOutput`

    Helpers provistos:
      - `_build_constant_speed_plan(...)`: caso común — construir un
        BehaviorOutput a velocidad constante siguiendo la lanelet actual.
      - `_fallback_plan(...)`: BehaviorOutput inválido para situaciones
        en las que `is_active` dice True pero `plan` no puede producir
        un resultado utilizable.
    """

    name: str = "base"
    priority: int = 0

    @abstractmethod
    def is_active(self, ctx: PlanningContext) -> bool: ...

    @abstractmethod
    def plan(self, ctx: PlanningContext) -> BehaviorOutput: ...

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _build_constant_speed_plan(
        self,
        ctx: PlanningContext,
        target_speed_mps: float,
        scenario_name: str,
        notes: dict | None = None,
    ) -> BehaviorOutput:
        """Construye un plan a velocidad constante siguiendo la ruta activa.

        Los detalles que comparten todos los scenarios "siga el lane":
          - Si el RoutePlanner ya publicó la ruta densa activa
            (`ctx.route.route_waypoints`), seguir ESE corredor.
          - Si no hay ruta densa, caer al lanelet local
            (`ctx.route.current_lanelet_id`) como fallback legacy.
          - Si no hay ninguna de las dos referencias, devolver fallback inválido.
          - speed_profile uniforme.
        """
        notes = dict(notes or {})
        route_waypoints = list(ctx.route.route_waypoints or [])
        lane_obs = getattr(ctx, "lane_observation", None)
        route_corridor_available = bool(ctx.route.route_active) and bool(route_waypoints)
        lanelet_corridor_available = bool(ctx.lanelet_map is not None and ctx.route.current_lanelet_id)
        visual_decision = self._visual_primary_gate().decide(
            ctx=ctx,
            lane_observation=lane_obs,
            scenario_name=scenario_name,
            route_corridor_available=route_corridor_available,
            lanelet_corridor_available=lanelet_corridor_available,
        )
        notes.update(visual_decision.notes)
        visual_path_reason = str(visual_decision.reason)
        map_authority_active = bool(visual_decision.map_authority_active)
        use_visual_path = bool(visual_decision.use_visual_path)
        if use_visual_path:
            visual_measurement_mode = str(getattr(lane_obs, "measurement_mode", "none") or "none")
            connect_visual_path_from_ego = visual_measurement_mode != "single_line"
            target_path = build_target_path_from_visual(
                center_waypoints_body=lane_obs.center_waypoints_body,
                ego_pose=ctx.pose.fused_pose,
                target_speed_mps=target_speed_mps,
                horizon_n=ctx.horizon_n,
                dt=ctx.dt,
                connect_from_ego_pose=connect_visual_path_from_ego,
            )
            notes.setdefault("path_source", "visual_lane_waypoints")
            notes.setdefault("recovery_source", "visual_lane_waypoints")
            notes["path_authority"] = "visual"
            notes["mpc_weight_profile"] = "lane_keep_visual"
            notes["steer_rate_limit_deg_s"] = 180.0
            notes["visual_lane_waypoint_count"] = int(len(lane_obs.center_waypoints_body or ()))
            notes["visual_path_connected_from_ego_pose"] = bool(connect_visual_path_from_ego)
            if lane_obs.extrapolated_side:
                notes["visual_lane_extrapolated_side"] = str(lane_obs.extrapolated_side)
            if lane_obs.lane_width_m is not None:
                notes["visual_lane_width_m"] = float(lane_obs.lane_width_m)
            notes["visual_path_primary_reason"] = str(visual_path_reason)
        elif route_corridor_available:
            target_path, route_bridge_meta = build_target_path_from_route(
                route_waypoints=route_waypoints,
                matched_idx=int(ctx.route.matched_idx or 0),
                start_xy=(ctx.pose.fused_pose.x, ctx.pose.fused_pose.y),
                start_yaw_rad=float(ctx.pose.fused_pose.yaw),
                matched_xy=(
                    float(ctx.route.matched_pose.x),
                    float(ctx.route.matched_pose.y),
                ),
                target_speed_mps=target_speed_mps,
                horizon_n=ctx.horizon_n,
                dt=ctx.dt,
                return_metadata=True,
            )
            notes.setdefault("path_source", "route_waypoints")
            notes.setdefault("recovery_source", "route_waypoints_reentry")
            if map_authority_active:
                notes["path_authority"] = "map"
                notes["mpc_weight_profile"] = "map_turn_authority"
                notes["steer_rate_limit_deg_s"] = 160.0
            else:
                notes.setdefault("path_authority", "route")
            if visual_path_reason:
                notes["visual_path_primary_rejected_reason"] = str(visual_path_reason)
            for key in ("bridge_mode", "protected_prefix_m", "merge_start_idx", "merge_end_idx"):
                if route_bridge_meta.get(key) is not None:
                    notes[key] = route_bridge_meta[key]
        elif lanelet_corridor_available:
            target_path = build_target_path(
                lanelet_map=ctx.lanelet_map,
                start_lanelet_id=ctx.route.current_lanelet_id,
                start_xy=(ctx.pose.fused_pose.x, ctx.pose.fused_pose.y),
                target_speed_mps=target_speed_mps,
                horizon_n=ctx.horizon_n,
                dt=ctx.dt,
                next_lanelet_hint_ids=ctx.route.next_lanelet_ids,
            )
            notes.setdefault("path_source", "lanelet_centerline")
            notes.setdefault("recovery_source", "lanelet_corridor")
            if map_authority_active:
                notes["path_authority"] = "map"
                notes["mpc_weight_profile"] = "map_turn_authority"
                notes["steer_rate_limit_deg_s"] = 160.0
            else:
                notes.setdefault("path_authority", "lanelet")
            if visual_path_reason:
                notes["visual_path_primary_rejected_reason"] = str(visual_path_reason)
        else:
            return self._fallback_plan(ctx, reason="no_lanelet_map_or_id")
        if lane_obs is not None and _supports_visual_lane_reentry_bias(lane_obs):
            notes["visual_lane_error_m"] = float(lane_obs.direct_error_m or 0.0)
            notes["visual_lane_quality"] = float(lane_obs.quality or 0.0)
        speed_profile = np.full(ctx.horizon_n, float(target_speed_mps), dtype=float)
        stop_required = False

        return BehaviorOutput(
            timestamp=ctx.now_s,
            dt=ctx.dt,
            target_path=target_path,
            speed_profile=speed_profile,
            scenario_name=scenario_name,
            valid=True,
            stop_required=stop_required,
            notes=notes,
        )

    def _visual_primary_gate(self) -> VisualPrimaryGate:
        gate = getattr(self, "_lane_visual_primary_gate", None)
        if gate is None:
            gate = VisualPrimaryGate()
            setattr(self, "_lane_visual_primary_gate", gate)
        return gate

    def _fallback_plan(self, ctx: PlanningContext, reason: str) -> BehaviorOutput:
        """BehaviorOutput inválido — el `safety_gate` toma control.

        Devuelve speed_profile = ceros y `valid=False`. El controller
        debe llamar `safety_gate.fallback()` cuando vea esto.
        """
        return BehaviorOutput(
            timestamp=ctx.now_s if ctx is not None else time.time(),
            dt=ctx.dt if ctx is not None else 0.05,
            target_path=np.zeros((1, 3)),
            speed_profile=np.zeros(0),
            scenario_name=ScenarioName.FALLBACK.value,
            valid=False,
            stop_required=True,
            notes={"reason": reason},
        )


def _supports_visual_lane_reentry_bias(lane_observation) -> bool:
    return (
        lane_observation is not None
        and bool(getattr(lane_observation, "direct_error_valid", False))
        and str(getattr(lane_observation, "measurement_mode", "none")) == "two_line"
        and float(getattr(lane_observation, "quality", 0.0) or 0.0) >= 0.8
        and getattr(lane_observation, "direct_error_m", None) is not None
    )
