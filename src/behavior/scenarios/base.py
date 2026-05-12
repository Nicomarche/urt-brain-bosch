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

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import config as _config
from src.behavior.context import PlanningContext
from src.behavior.trajectory_builder import (
    build_target_path,
    build_target_path_from_route,
    build_target_path_from_visual,
)
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.core.types.perception import lane_observation_has_visual_path


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
        min_visual_quality = float(
            getattr(_config, "LANE_VISUAL_MIN_QUALITY_FOR_PRIMARY_PATH", 0.55)
        )
        min_visual_points = int(
            getattr(_config, "LANE_VISUAL_MIN_POLY_POINTS", 8)
        )
        route_corridor_available = bool(ctx.route.route_active) and bool(route_waypoints)
        lanelet_corridor_available = bool(ctx.lanelet_map is not None and ctx.route.current_lanelet_id)
        use_visual_path, visual_path_reason = _select_visual_primary_path(
            ctx=ctx,
            lane_observation=lane_obs,
            min_quality=min_visual_quality,
            min_points=min_visual_points,
            scenario_name=scenario_name,
            route_corridor_available=route_corridor_available,
            lanelet_corridor_available=lanelet_corridor_available,
        )
        if use_visual_path:
            connect_visual_path_from_ego = False
            # Opción C (paso 6): cuando la fuente del path es visual, lo
            # entregamos al MPC en frame body. El motion_controller debe
            # detectar `path_source=="visual_lane_waypoints"` y usar
            # `x_current=(0,0,0)` para resolver el OCP en el mismo frame.
            visual_path_frame_body = bool(
                getattr(_config, "LANE_VISUAL_PATH_FRAME_BODY", True)
            )
            target_path = build_target_path_from_visual(
                center_waypoints_body=lane_obs.center_waypoints_body,
                ego_pose=ctx.pose.fused_pose,
                target_speed_mps=target_speed_mps,
                horizon_n=ctx.horizon_n,
                dt=ctx.dt,
                connect_from_ego_pose=connect_visual_path_from_ego,
                frame_body=visual_path_frame_body,
            )
            notes.setdefault("path_source", "visual_lane_waypoints")
            notes.setdefault("recovery_source", "visual_lane_waypoints")
            notes["visual_lane_waypoint_count"] = int(len(lane_obs.center_waypoints_body or ()))
            notes["visual_path_connected_from_ego_pose"] = bool(connect_visual_path_from_ego)
            notes["visual_path_frame_body"] = bool(visual_path_frame_body)
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
            if visual_path_reason:
                notes["visual_path_primary_rejected_reason"] = str(visual_path_reason)
        else:
            return self._fallback_plan(ctx, reason="no_lanelet_map_or_id")
        if lane_obs is not None and _supports_visual_lane_reentry_bias(lane_obs):
            control_error_m = getattr(lane_obs, "control_lateral_error_m", None)
            notes["visual_lane_error_m"] = float(
                control_error_m if control_error_m is not None else (lane_obs.direct_error_m or 0.0)
            )
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
        and (
            getattr(lane_observation, "control_lateral_error_m", None) is not None
            or getattr(lane_observation, "direct_error_m", None) is not None
        )
    )


def _select_visual_primary_path(
    *,
    ctx: PlanningContext,
    lane_observation,
    min_quality: float,
    min_points: int,
    scenario_name: str,
    route_corridor_available: bool,
    lanelet_corridor_available: bool,
) -> tuple[bool, str]:
    if not bool(getattr(_config, "LANE_VISUAL_PRIMARY_ENABLED", True)):
        return False, "visual_primary_disabled"
    measurement_mode = str(getattr(lane_observation, "measurement_mode", "none") or "none")
    visual_min_points = int(min_points)
    if measurement_mode == "two_line":
        visual_min_points = min(
            visual_min_points,
            max(
                2,
                int(getattr(_config, "LANE_VISUAL_TWO_LINE_PRIMARY_MIN_POINTS", 4)),
            ),
        )
    if not lane_observation_has_visual_path(
        lane_observation,
        min_quality=min_quality,
        min_points=visual_min_points,
    ):
        return False, "visual_path_gate_rejected"

    if str(scenario_name or "") in {
        ScenarioName.CROSSWALK.value,
        ScenarioName.PARKING.value,
        ScenarioName.ROUNDABOUT.value,
    }:
        return False, f"scenario_uses_route:{scenario_name}"

    precision_context = _route_precision_context(ctx)
    if precision_context is not None:
        return False, precision_context

    if measurement_mode == "two_line":
        reason = "two_line_primary"
        if route_corridor_available or lanelet_corridor_available:
            reason = "two_line_visual_primary_over_route"
        return True, reason

    if measurement_mode == "single_line":
        # Estilo urt-ref: cuando se pierde una línea, el detector sintetiza la
        # ausente desplazando lane_width sobre el polinomio visible. El path
        # resultante mantiene la curvatura del polinomio — es tan válido como
        # two_line. Por eso bajamos el bar de aceptación a `0.6` y eliminamos
        # los gates redundantes (transition_error_coherent, streak, map_match
        # obligatorio, assist_primary). Mantenemos solo: calidad mínima y los
        # waypoints viables (que ya validó `lane_observation_has_visual_path`
        # antes de llegar acá).
        min_single_quality = float(
            getattr(_config, "LANE_VISUAL_SINGLE_LINE_PRIMARY_MIN_QUALITY", 0.6)
        )
        quality = float(getattr(lane_observation, "quality", 0.0) or 0.0)
        if quality < min_single_quality:
            return False, "single_line_quality_below_primary"
        reason = "single_line_primary"
        if route_corridor_available or lanelet_corridor_available:
            reason = "single_line_visual_primary_over_route"
        return True, reason

    return False, f"unsupported_measurement_mode:{measurement_mode}"


def _visual_heading_change_rad(lane_obs) -> float:
    """Signed heading change (rad) over visual lane center waypoints in body frame.

    Positive = left curve, negative = right curve — same convention as
    route.path_heading_change_rad so the two values can be compared directly.
    Returns 0.0 on any missing / malformed data.
    """
    try:
        wps = tuple(getattr(lane_obs, "center_waypoints_body", None) or ())
        if len(wps) < 2:
            return 0.0
        delta = float(wps[-1][2]) - float(wps[0][2])
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        return delta
    except (TypeError, ValueError, IndexError):
        return 0.0


def _route_precision_context(ctx: PlanningContext) -> str | None:
    route = getattr(ctx, "route", None)
    if route is None:
        return None
    semantic = str(
        getattr(route, "expected_control_type", None)
        or ""
    ).lower()
    if semantic in {"intersection", "stopline", "crosswalk", "parking", "roundabout"}:
        return f"route_precision_context:{semantic}"
    # Detección geométrica de intersección: si el path GPS del próximo
    # tramo tiene un cambio de heading grande (>25°), el coche está
    # entrando a un giro de intersección. En esos casos el visual sigue
    # ciegamente las líneas rectas del cruce y no dobla; debemos preferir
    # la ruta GPS que conoce la geometría del giro.
    try:
        heading_change_rad = abs(float(getattr(route, "path_heading_change_rad", 0.0) or 0.0))
    except (TypeError, ValueError):
        heading_change_rad = 0.0
    heading_change_threshold_rad = float(
        getattr(_config, "LANE_VISUAL_ROUTE_INTERSECTION_HEADING_CHANGE_RAD", 0.436)
    )  # default ≈ 25°
    if heading_change_rad >= heading_change_threshold_rad:
        # Override visual: cuando la cámara ve un two_line con quality ≥ umbral,
        # tiene la geometría REAL de la curva, no la del OSM. Esto se origina
        # en el run manual run_manual_20260511_192422 donde lanelets 60/70/85
        # reportaron heading_change OSM de 79-90° (artifacts del mapa) mientras
        # la visual veía una curva continua con q=1.0; el operador siguió la
        # cámara y mantuvo la trayectoria, el MPC siguió OSM y habría comandado
        # ±25° en sentido opuesto. El semántico-check de arriba (intersection /
        # stopline / crosswalk / parking / roundabout) queda intacto: en
        # cruces reales sin línea continua sí necesitamos la ruta.
        override_enabled = bool(
            getattr(_config, "LANE_VISUAL_OVERRIDES_HEADING_CHANGE_RULE", True)
        )
        if override_enabled:
            lane = getattr(ctx, "lane_observation", None)
            if lane is not None:
                mode = str(getattr(lane, "measurement_mode", "") or "")
                quality = float(getattr(lane, "quality", 0.0) or 0.0)
                min_quality = float(
                    getattr(
                        _config,
                        "LANE_VISUAL_OVERRIDE_HEADING_CHANGE_MIN_QUALITY",
                        0.95,
                    )
                )
                if mode == "two_line" and quality >= min_quality:
                    # Verificación de dirección: si la visual apunta en sentido
                    # opuesto a la ruta, es una bifurcación donde la cámara ve
                    # ambas ramas y el centro calculado apunta al fork incorrecto.
                    # En ese caso NO hacer el override — usar la ruta.
                    route_heading_signed = float(
                        getattr(route, "path_heading_change_rad", 0.0) or 0.0
                    )
                    visual_change_rad = _visual_heading_change_rad(lane)
                    conflict_threshold_rad = float(
                        getattr(_config, "LANE_VISUAL_ROUTE_CONFLICT_MIN_RAD", 0.174)
                    )
                    if (
                        abs(visual_change_rad) >= conflict_threshold_rad
                        and visual_change_rad * route_heading_signed < 0
                    ):
                        return (
                            f"route_precision_context:visual_route_conflict_"
                            f"{math.degrees(visual_change_rad):.0f}vs"
                            f"{math.degrees(route_heading_signed):.0f}deg"
                        )
                    return None
        return f"route_precision_context:turn_heading_change_{math.degrees(heading_change_rad):.0f}deg"
    # Chequeo de alineación: si el heading del robot difiere del waypoint de ruta
    # en más del umbral, el robot está entrando a una bifurcación o giro donde la
    # cámara ve ambas ramas y genera un centro erróneo. Este chequeo persiste
    # naturalmente hasta que el robot gire y se alinee con la ruta.
    # Se evalúa sobre el waypoint actual Y el siguiente para ser robusto a la
    # oscilación del matched_idx (el localizador alterna entre idx y idx+1 en bordes).
    try:
        route_waypoints = list(getattr(route, "route_waypoints", None) or [])
        matched_idx = int(getattr(route, "matched_idx", 0) or 0)
        robot_yaw = float(ctx.pose.fused_pose.yaw)
        alignment_delta_rad = 0.0
        n_wps = len(route_waypoints)
        for check_idx in (matched_idx, min(matched_idx + 1, n_wps - 1)):
            if 0 <= check_idx < n_wps:
                wp_yaw = float(route_waypoints[check_idx][2])
                d = (wp_yaw - robot_yaw + math.pi) % (2.0 * math.pi) - math.pi
                alignment_delta_rad = max(alignment_delta_rad, abs(d))
    except (TypeError, ValueError, IndexError):
        alignment_delta_rad = 0.0
    alignment_threshold_rad = float(
        getattr(_config, "LANE_VISUAL_ROUTE_ALIGNMENT_MAX_RAD", 0.349)  # 20°
    )
    if alignment_delta_rad >= alignment_threshold_rad:
        return f"route_precision_context:heading_misaligned_{math.degrees(alignment_delta_rad):.0f}deg"
    return None


def _visual_map_match_primary_gate(
    ctx: PlanningContext,
    *,
    measurement_mode: str,
    route_corridor_available: bool,
    lanelet_corridor_available: bool,
) -> str | None:
    if not (route_corridor_available or lanelet_corridor_available):
        return None
    if not bool(getattr(_config, "VISUAL_MAP_MATCH_ENABLED", True)):
        return None
    if (
        str(measurement_mode or "none") != "single_line"
        or not bool(getattr(_config, "LANE_VISUAL_SINGLE_LINE_REQUIRE_MAP_MATCH_WITH_ROUTE", True))
    ):
        return None
    match = getattr(getattr(ctx, "pose", None), "visual_lane_match", None)
    if match is None:
        return "single_line_visual_map_match_missing"

    near_yaw = getattr(match, "near_yaw_error_rad", None)
    if near_yaw is None:
        near_yaw = getattr(match, "yaw_error_rad", 0.0)
    try:
        near_yaw_abs = abs(float(near_yaw or 0.0))
    except (TypeError, ValueError):
        return "single_line_visual_map_match_invalid_yaw"

    max_sample_yaw = getattr(match, "max_abs_yaw_error_rad", None)
    try:
        max_sample_yaw_abs = abs(float(max_sample_yaw or near_yaw_abs))
    except (TypeError, ValueError):
        return "single_line_visual_map_match_invalid_sample_yaw"
    lateral_error = getattr(match, "lateral_error_m", 0.0)
    max_sample_lateral = getattr(match, "max_abs_lateral_error_m", None)
    try:
        lateral_abs = abs(float(lateral_error or 0.0))
        max_sample_lateral_abs = abs(float(max_sample_lateral if max_sample_lateral is not None else lateral_abs))
    except (TypeError, ValueError):
        return "single_line_visual_map_match_invalid_lateral"

    primary_yaw_limit = math.radians(
        float(getattr(_config, "VISUAL_MAP_MATCH_PRIMARY_MAX_YAW_ERROR_DEG", 5.0) or 5.0)
    )
    primary_sample_yaw_limit = math.radians(
        float(
            getattr(
                _config,
                "VISUAL_MAP_MATCH_PRIMARY_MAX_SAMPLE_YAW_ERROR_DEG",
                math.degrees(primary_yaw_limit),
            )
            or math.degrees(primary_yaw_limit)
        )
    )
    lateral_limit = float(
        getattr(
            _config,
            "VISUAL_MAP_MATCH_MAX_LATERAL_ERROR_M",
            float(getattr(_config, "LANE_WIDTH_CM", 35.0)) / 200.0,
        )
        or 0.175
    )
    sample_lateral_limit = max(lateral_limit, lateral_limit * 1.35)
    if near_yaw_abs > primary_yaw_limit:
        return "single_line_visual_map_yaw_mismatch"
    if max_sample_yaw_abs > primary_sample_yaw_limit:
        return "single_line_visual_map_sample_yaw_mismatch"
    if lateral_abs > lateral_limit:
        return "single_line_visual_map_lateral_mismatch"
    if max_sample_lateral_abs > sample_lateral_limit:
        return "single_line_visual_map_sample_lateral_mismatch"
    if not bool(getattr(match, "accepted", False)):
        reason = str(getattr(match, "reason", "rejected") or "rejected")
        return f"single_line_visual_map_match_rejected:{reason}"
    return None
