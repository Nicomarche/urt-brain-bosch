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
    blend_target_paths,
    build_target_path,
    build_target_path_from_route,
    build_target_path_from_visual,
)
from src.behavior.visual_dr_blender import VisualDrBlender
from src.behavior.visual_primary_gate import VisualPrimaryGate
from src.core.types.behavior import BehaviorOutput, ScenarioName


def _load_lane_blend_dominant_threshold() -> float:
    """Umbral α para elegir el perfil de pesos MPC en blend (visual vs route)."""
    try:
        import config as _cfg

        return float(getattr(_cfg, "LANE_BLEND_DOMINANT_THRESHOLD", 0.5))
    except Exception:
        return 0.5


_LANE_BLEND_DOMINANT_THRESHOLD = _load_lane_blend_dominant_threshold()


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
        # Decisor de blend: cuando `LANE_BLEND_ENABLED=False` emite α∈{0,1}
        # idéntico al `VisualPrimaryGate` clásico, preservando comportamiento.
        blend_decision = self._visual_dr_blender().decide(
            ctx=ctx,
            lane_observation=lane_obs,
            scenario_name=scenario_name,
            route_corridor_available=route_corridor_available,
            lanelet_corridor_available=lanelet_corridor_available,
            now_s=float(getattr(ctx, "now_s", 0.0) or 0.0),
        )
        notes.update(blend_decision.notes)
        alpha = float(blend_decision.alpha)
        visual_path_reason = str(blend_decision.gate_decision.reason)
        map_authority_active = bool(blend_decision.map_authority_active)
        use_visual_path = alpha >= 1.0 - 1e-6
        use_dr_path = alpha <= 1e-6
        if use_visual_path:
            target_path = self._build_visual_target_path(
                ctx=ctx,
                lane_obs=lane_obs,
                target_speed_mps=target_speed_mps,
                notes=notes,
                visual_path_reason=visual_path_reason,
            )
        elif use_dr_path:
            dr_result = self._build_dr_target_path(
                ctx=ctx,
                target_speed_mps=target_speed_mps,
                route_waypoints=route_waypoints,
                route_corridor_available=route_corridor_available,
                lanelet_corridor_available=lanelet_corridor_available,
                map_authority_active=map_authority_active,
                notes=notes,
                visual_path_reason=visual_path_reason,
            )
            if dr_result is None:
                return self._fallback_plan(ctx, reason="no_lanelet_map_or_id")
            target_path = dr_result
        else:
            # ── Blend convexo visual ↔ DR ─────────────────────────────────
            # Construimos ambos paths en el mismo frame y los mezclamos punto
            # a punto con `alpha` como peso del visual. Esto da una
            # transición SUAVE en single_line (típicamente α≈0.3): el path
            # sigue mayormente el corredor DR pero se desvía hacia la
            # tangente visual, que mantiene la curvatura del polinomio
            # incluso con una sola línea (la otra se sintetiza con LANE_WIDTH).
            visual_notes: dict = {}
            visual_path = self._build_visual_target_path(
                ctx=ctx,
                lane_obs=lane_obs,
                target_speed_mps=target_speed_mps,
                notes=visual_notes,
                visual_path_reason=visual_path_reason,
            )
            dr_notes: dict = {}
            dr_path = self._build_dr_target_path(
                ctx=ctx,
                target_speed_mps=target_speed_mps,
                route_waypoints=route_waypoints,
                route_corridor_available=route_corridor_available,
                lanelet_corridor_available=lanelet_corridor_available,
                map_authority_active=map_authority_active,
                notes=dr_notes,
                visual_path_reason=visual_path_reason,
            )
            if dr_path is None:
                # Sin DR no hay con qué mezclar — degradar a visual puro y
                # mergear los notes del visual.
                target_path = visual_path
                notes.update(visual_notes)
            else:
                target_path = blend_target_paths(visual_path, dr_path, alpha)
                # Mergear notes de ambas ramas con el de mayor peso ganando
                # los keys conflictivos (perfil de pesos / steer rate limit).
                dominant_visual = alpha >= _LANE_BLEND_DOMINANT_THRESHOLD
                # Empezamos por el lado menos dominante para que el dominante
                # sobreescriba sus keys.
                if dominant_visual:
                    notes.update(dr_notes)
                    notes.update(visual_notes)
                    notes["path_authority"] = "blended_visual"
                    notes["mpc_weight_profile"] = "lane_keep_visual"
                    notes["steer_rate_limit_deg_s"] = 180.0
                else:
                    notes.update(visual_notes)
                    notes.update(dr_notes)
                    notes["path_authority"] = "blended_route"
                    # Si DR estaba bajo autoridad de mapa, mantenemos sus límites
                    # (no queremos relajar steer rate cerca de una intersección).
                    if map_authority_active:
                        notes["mpc_weight_profile"] = "map_turn_authority"
                        notes["steer_rate_limit_deg_s"] = 160.0
                notes["path_source"] = "blended_visual_dr"
                notes["recovery_source"] = "blended_visual_dr"
                notes["visual_path_primary_reason"] = str(visual_path_reason)
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

    def _build_visual_target_path(
        self,
        *,
        ctx: PlanningContext,
        lane_obs,
        target_speed_mps: float,
        notes: dict,
        visual_path_reason: str,
    ) -> np.ndarray:
        """Arma el target_path desde los waypoints visuales y publica los notes
        canónicos de la rama visual. Mutates `notes` in place.
        """
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
        return target_path

    def _build_dr_target_path(
        self,
        *,
        ctx: PlanningContext,
        target_speed_mps: float,
        route_waypoints: list,
        route_corridor_available: bool,
        lanelet_corridor_available: bool,
        map_authority_active: bool,
        notes: dict,
        visual_path_reason: str,
    ) -> np.ndarray | None:
        """Arma el target_path desde la ruta densa o, en su defecto, la lanelet
        centerline. Devuelve `None` si no hay ninguna referencia DR (caller
        decide cómo recuperar — típicamente `_fallback_plan`).
        """
        if route_corridor_available:
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
            return target_path
        if lanelet_corridor_available:
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
            return target_path
        return None

    def _visual_primary_gate(self) -> VisualPrimaryGate:
        gate = getattr(self, "_lane_visual_primary_gate", None)
        if gate is None:
            gate = VisualPrimaryGate()
            setattr(self, "_lane_visual_primary_gate", gate)
        return gate

    def _visual_dr_blender(self) -> VisualDrBlender:
        blender = getattr(self, "_lane_visual_dr_blender", None)
        if blender is None:
            blender = VisualDrBlender()
            setattr(self, "_lane_visual_dr_blender", blender)
        # El blender reusa internamente un VisualPrimaryGate. Para que el comportamiento
        # con `LANE_BLEND_ENABLED=False` sea bit-for-bit idéntico al pre-blender,
        # exponemos el gate del blender como el gate "clásico" del scenario. Así la
        # histeresis (enter/exit ticks) se mantiene en un solo lugar y los tests
        # legacy que prueban hysteresis siguen pasando.
        if getattr(self, "_lane_visual_primary_gate", None) is not blender.gate:
            self._lane_visual_primary_gate = blender.gate
        return blender

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
