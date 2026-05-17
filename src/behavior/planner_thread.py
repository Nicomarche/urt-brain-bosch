# src/behavior/planner_thread.py
#
# Thread que ejecuta el `BehaviorPlanner` a tasa fija. ÚNICA fuente de
# verdad de velocidad y target_path en el sistema. El thread:
#
#   1. Lee snapshots inmutables de los inputs desde buffers thread-safe:
#        - PoseEstimate           (de threadPoseEstimator)
#        - RouteContext           (de threadNavigationPlanner)
#        - LaneObservation        (de threadLocalPerception/lane_observer)
#        - StoplineObservation    (de threadLocalPerception)
#        - tuple[TrackedObject]   (de threadObjectTracker — Phase 5)
#   2. Construye un `PlanningContext` con estos inputs + dt/horizon/speeds
#      desde config.
#   3. Llama `BehaviorPlanner.plan(ctx) -> BehaviorOutput`.
#   4. Escribe el `BehaviorOutput` en el buffer de salida (consumido por
#      el MotionController, Phase 6).
#   5. Publica un snapshot serializable al dashboard vía `BEHAVIOR_OUTPUT_MSG`.
#
# Diseño:
#   - El thread es DELGADO: cero lógica de decisión. Sólo pega buffers→
#     planner→buffers. Toda la inteligencia vive en `BehaviorPlanner` y
#     en los `IScenario`. Esto es SRP estricto.
#   - El planner se compone fuera (en `main.py`) y se inyecta — el thread
#     no conoce los scenarios concretos. Esto es DIP.
#   - Si pose o route no están disponibles aún (bootstrap), el thread emite
#     un BehaviorOutput inválido (valid=False, stop_required=True) y deja
#     que el safety_gate del controller pare el auto. NO inventa pose.
#
# Comentario crítico sobre concurrencia: los buffers usan locks internos
# y devuelven referencias inmutables (frozen dataclasses + ndarray que
# tratamos como read-only). El BehaviorOutput resultante es frozen, así
# que el consumer (MPC thread) puede leer sin race.

from __future__ import annotations

import time
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

from src.behavior.context import PlanningContext
from src.core.bus.topics import BEHAVIOR_OUTPUT_MSG, BEHAVIOR_PLANNER_STATUS, STATE_CHANGE
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.core.types.lidar import LidarObstacle, LidarScan
from src.core.types.perception import LaneObservation, StoplineObservation, TrackedObject
from src.core.types.pose import PoseEstimate
from src.core.types.routing import RouteContext
from src.routing.lanelet.runtime_loader import (
    lanelet_map_covers_ids, load_tracking_lanelet_map,
)
from src.templates.threadwithstop import ThreadWithStop
from src.utils.live_log import live_log

if TYPE_CHECKING:
    from src.behavior.planner import BehaviorPlanner
    from src.core.messaging.buffers import LatestValueBuffer
    from src.routing.lanelet.lanelet_map import LaneletMap


def _load_lanelet_map_from_tracking_config(*, force_reload: bool = False):
    """Load the Lanelet map directly from tracking config as a recovery path."""
    return load_tracking_lanelet_map(force_reload=force_reload)


class threadBehaviorPlanner(ThreadWithStop):
    """Ejecuta el BehaviorPlanner a tasa fija y publica BehaviorOutput.

    Constructor recibe TODO inyectado — el thread no construye nada
    importante. Esto facilita test e isolation: en tests usás un planner
    fake y buffers en memoria sin tocar nada del thread.
    """

    def __init__(
        self,
        queuesList,
        planner: "BehaviorPlanner",
        lanelet_map: "LaneletMap | None",
        pose_estimate_buffer: "LatestValueBuffer",
        route_context_buffer: "LatestValueBuffer",
        lane_observation_buffer: "LatestValueBuffer",
        stopline_observation_buffer: "LatestValueBuffer",
        tracked_objects_buffer: "LatestValueBuffer | None",
        behavior_output_buffer: "LatestValueBuffer",
        lidar_scan_buffer: "LatestValueBuffer | None" = None,
        lidar_obstacles_buffer: "LatestValueBuffer | None" = None,
        sign_hints_buffer: "LatestValueBuffer | None" = None,
        *,
        dt_s: float,
        horizon_n: int,
        nominal_speed_mps: float,
        max_speed_mps: float,
        pause_s: float = 0.05,
        logging=None,
        debugging: bool = False,
    ) -> None:
        super().__init__(pause=float(pause_s))
        self.queuesList = queuesList
        self._planner = planner
        self._lanelet_map = lanelet_map
        self._lanelet_map_last_recovery_attempt_mono = 0.0
        self._lanelet_map_recovery_retry_s = 1.0
        self._pose_buf = pose_estimate_buffer
        self._route_buf = route_context_buffer
        self._lane_buf = lane_observation_buffer
        self._stopline_buf = stopline_observation_buffer
        self._tracked_buf = tracked_objects_buffer
        self._lidar_scan_buf = lidar_scan_buffer
        self._lidar_obs_buf = lidar_obstacles_buffer
        self._sign_hints_buf = sign_hints_buffer
        self._output_buf = behavior_output_buffer
        self._dt_s = float(dt_s)
        self._horizon_n = int(horizon_n)
        self._nominal_speed_mps = float(nominal_speed_mps)
        self._max_speed_mps = float(max_speed_mps)
        self.logging = logging
        self.debugging = bool(debugging)

        # Ramp de arranque: la velocidad de referencia sube gradualmente hasta
        # _nominal_speed_mps, pero si se mueve respeta el mínimo competitivo.
        self._current_speed_mps: float = 0.0
        try:
            from config import BEHAVIOR_ACCEL_MPS2, BEHAVIOR_MIN_SPEED_MPS
            self._accel_mps2: float = float(BEHAVIOR_ACCEL_MPS2)
            self._min_moving_speed_mps: float = float(BEHAVIOR_MIN_SPEED_MPS)
        except ImportError:
            self._accel_mps2 = 0.25  # 0→0.50 m/s en 2 s
            self._min_moving_speed_mps = 0.20

        # Senders al dashboard. Mantenemos dos canales: el plan completo
        # (target_path + speed_profile) y un status corto para gauges.
        self._output_sender = messageHandlerSender(queuesList, BEHAVIOR_OUTPUT_MSG)
        self._status_sender = messageHandlerSender(queuesList, BEHAVIOR_PLANNER_STATUS)

        # Modo del sistema vía IPC STATE_CHANGE — mismo mecanismo que el
        # dispatcher. El Manager proxy (__sm_state__) no lo actualiza el
        # dashboard subprocess (eventlet rompe el proxy bajo spawn), así que
        # el pipe IPC es la única fuente confiable para todos los writers.
        self._stateChangeSubscriber = messageHandlerSubscriber(
            queuesList, STATE_CHANGE, "lastOnly", True
        )
        self._current_mode: str = "STOP"

        # Métricas para el status — no afectan a la lógica de decisión.
        self._frame_idx = 0
        self._last_status_publish_t = 0.0
        self._status_period_s = 0.5  # 2 Hz al dashboard alcanza
        self._last_plan_dt_ms = 0.0

    def _route_corridor_ids(self, route: RouteContext | None) -> tuple[str, ...]:
        if route is None:
            return ()
        ordered: list[str] = []
        if route.current_lanelet_id:
            ordered.append(str(route.current_lanelet_id))
        ordered.extend(str(item) for item in (route.next_lanelet_ids or ()) if str(item))
        return tuple(dict.fromkeys(ordered))

    def _ensure_lanelet_map(self, route: RouteContext | None = None) -> None:
        corridor_ids = self._route_corridor_ids(route)
        missing_map = self._lanelet_map is None
        missing_route_geometry = bool(corridor_ids) and not lanelet_map_covers_ids(
            self._lanelet_map,
            corridor_ids,
        )
        if not missing_map and not missing_route_geometry:
            return
        now_mono = time.monotonic()
        if (
            self._lanelet_map_last_recovery_attempt_mono > 0.0
            and (now_mono - self._lanelet_map_last_recovery_attempt_mono) < self._lanelet_map_recovery_retry_s
        ):
            return
        self._lanelet_map_last_recovery_attempt_mono = now_mono
        force_reload = bool(missing_route_geometry)
        try:
            try:
                recovered_map = _load_lanelet_map_from_tracking_config(force_reload=force_reload)
            except TypeError:
                recovered_map = _load_lanelet_map_from_tracking_config()
        except Exception as exc:
            live_log(
                "behavior_planner",
                event="lanelet_map_recovery",
                recovered=False,
                reason="exception",
                detail=str(exc),
                current_lanelet_id=(route.current_lanelet_id if route is not None else None),
                requested_lanelet_ids=list(corridor_ids),
            )
            return
        if recovered_map is None:
            live_log(
                "behavior_planner",
                event="lanelet_map_recovery",
                recovered=False,
                reason="tracking_config_unavailable",
                current_lanelet_id=(route.current_lanelet_id if route is not None else None),
                requested_lanelet_ids=list(corridor_ids),
            )
            return
        if corridor_ids and not lanelet_map_covers_ids(recovered_map, corridor_ids):
            live_log(
                "behavior_planner",
                event="lanelet_map_recovery",
                recovered=False,
                reason="route_lanelet_missing",
                current_lanelet_id=(route.current_lanelet_id if route is not None else None),
                requested_lanelet_ids=list(corridor_ids),
            )
            return
        self._lanelet_map = recovered_map
        try:
            lanelet_count = int(len(recovered_map))
        except Exception:
            lanelet_count = 0
        live_log(
            "behavior_planner",
            event="lanelet_map_recovery",
            recovered=True,
            reason=("route_lanelet_reload" if force_reload else "tracking_config_load"),
            lanelet_count=lanelet_count,
            current_lanelet_id=(route.current_lanelet_id if route is not None else None),
            requested_lanelet_ids=list(corridor_ids),
        )

    # ----------------------------------------------------------------
    # Loop principal
    # ----------------------------------------------------------------
    def thread_work(self) -> None:
        """Un tick: leer inputs → planear → publicar."""
        tick_start = time.monotonic()

        # Actualizar modo desde IPC STATE_CHANGE (mismo override que el dispatcher).
        try:
            msg = self._stateChangeSubscriber.receive()
            if msg is not None:
                self._current_mode = str(msg).upper()
        except Exception:
            pass

        # Fuera de AUTO/PARKING no planear ni correr el MPC.
        if self._current_mode not in {"AUTO", "PARKING"}:
            self._current_speed_mps = 0.0
            self._publish_empty(reason="mode_not_auto")
            return

        ctx = self._build_context(now_s=time.time())
        if ctx is None:
            # Inputs incompletos — emitir BehaviorOutput vacío para que el
            # downstream pare. NO arrancar a inventar plan basura.
            self._current_speed_mps = 0.0
            self._publish_empty(reason="inputs_unavailable")
            return
        if str(getattr(ctx.pose, "relocalization_mode", "") or "") == "auto_gps_entry_pending":
            self._current_speed_mps = 0.0
            self._publish_empty(reason="auto_gps_entry_pending")
            return
        if not bool(getattr(ctx.route, "route_active", False)):
            self._current_speed_mps = 0.0
            self._publish_empty(reason="route_inactive")
            return

        # Avanza el ramp de velocidad solo cuando AUTO ya tiene pose y ruta.
        # Subimos desde 0 de verdad; el piso de movimiento se aplica aguas abajo
        # recién cuando el target ya llegó a ese umbral.
        self._current_speed_mps = min(
            self._nominal_speed_mps,
            max(0.0, self._current_speed_mps + self._accel_mps2 * self._dt_s),
        )
        ctx = replace(ctx, nominal_speed_mps=self._current_speed_mps)

        live_log(
            "behavior_planner", event="context_snapshot",
            route_id=ctx.route.route_id,
            route_source=ctx.route.route_source,
            route_active=bool(ctx.route.route_active),
            route_waypoints_count=int(len(ctx.route.route_waypoints or [])),
            current_lanelet_id=ctx.route.current_lanelet_id,
            next_lanelet_ids=list(ctx.route.next_lanelet_ids or ()),
            current_node_id=ctx.route.current_node_id,
            upcoming_node_id=ctx.route.upcoming_node_id,
            current_attr=int(ctx.route.current_node_attr or 0),
            upcoming_attr=int(ctx.route.upcoming_node_attr or 0),
            matched_idx=int(ctx.route.matched_idx),
            target_idx=int(ctx.route.target_idx),
            map_match_error_m=float(ctx.route.map_match_error_m or 0.0),
            tracking_error_m=float(ctx.route.error_m or 0.0),
            heading_rad=float(ctx.route.heading_rad or 0.0),
            waypoint_mode_active=bool(ctx.route.waypoint_mode_active),
            next_semantic_type=ctx.route.next_semantic_type,
            next_semantic_distance_m=ctx.route.next_semantic_distance_m,
            expected_control_type=ctx.route.expected_control_type,
            lane_observation_available=bool(ctx.lane_observation is not None),
            stopline_observation_available=bool(ctx.stopline_observation is not None),
            tracked_object_count=int(len(ctx.tracked_objects or ())),
            lidar_obstacle_count=int(len(ctx.lidar_obstacles or ())),
            lidar_scan_age_s=(ctx.now_s - ctx.lidar_scan.timestamp) if ctx.lidar_scan is not None else None,
            sign_hints_count=int(len(ctx.sign_hints or ())),
            sign_hints=_slim_sign_hints(ctx.sign_hints),
            pose_x=float(ctx.pose.fused_pose.x),
            pose_y=float(ctx.pose.fused_pose.y),
            pose_yaw_rad=float(ctx.pose.fused_pose.yaw),
            nominal_speed_mps=float(ctx.nominal_speed_mps),
        )
        plan = self._planner.plan(ctx)

        # Si el planner pide parar (semáforo, obstáculo, etc.), reseteamos
        # el ramp para que el próximo arranque también sea suave.
        if plan.stop_required or not plan.valid:
            import logging as _log
            _log.getLogger(__name__).warning(
                "[ BehPlan ] ramp RESET — stop_req=%s valid=%s scen=%s "
                "ramp_before=%.3f notes=%s lanelet=%r",
                plan.stop_required,
                plan.valid,
                plan.scenario_name,
                self._current_speed_mps,
                plan.notes.get("reason", "?") if isinstance(plan.notes, dict) else str(plan.notes),
                ctx.route.current_lanelet_id,
            )
            self._current_speed_mps = 0.0

        # Persistir en buffer (consumer = MPC thread).
        self._output_buf.write(plan, timestamp=plan.timestamp)

        # Live JSONL: cada plan emitido. Cubre el caso "el auto no se
        # mueve / no sigue carril" porque desde acá ves scenario+priority
        # elegido, sp[0] (velocidad target inmediata), notes.reason si
        # cayó a fallback, y el current_lanelet_id que el LaneKeep necesita.
        try:
            sp_first3 = (plan.speed_profile[:3].tolist()
                         if hasattr(plan.speed_profile, "tolist")
                         else list(plan.speed_profile[:3]))
        except Exception:
            sp_first3 = []
        try:
            tp_first3 = (plan.target_path[:3].tolist()
                         if hasattr(plan.target_path, "tolist")
                         else list(plan.target_path[:3]))
        except Exception:
            tp_first3 = []
        notes_reason = (plan.notes.get("reason")
                        if isinstance(plan.notes, dict) else None)
        path_source = (plan.notes.get("path_source")
                       if isinstance(plan.notes, dict) else None)
        stop_mode = (plan.notes.get("stop_mode")
                     if isinstance(plan.notes, dict) else None)
        stop_source = (plan.notes.get("stop_source")
                       if isinstance(plan.notes, dict) else None)
        distance_source = (plan.notes.get("distance_source")
                           if isinstance(plan.notes, dict) else None)
        stop_distance_m = (plan.notes.get("distance_m")
                           if isinstance(plan.notes, dict) else None)
        candidate_modules = (plan.notes.get("candidate_modules")
                             if isinstance(plan.notes, dict) else None)
        approved_module = (plan.notes.get("approved_module")
                           if isinstance(plan.notes, dict) else None)
        overtake_corridor_escape = (plan.notes.get("overtake_corridor_escape")
                                    if isinstance(plan.notes, dict) else None)
        overtake_effective_lateral_offset_m = (
            plan.notes.get("overtake_effective_lateral_offset_m")
            if isinstance(plan.notes, dict) else None
        )
        overtake_clearance_m = (plan.notes.get("overtake_clearance_m")
                                if isinstance(plan.notes, dict) else None)
        containment_mode = (plan.notes.get("containment_mode")
                            if isinstance(plan.notes, dict) else None)
        corridor_touches_bound = (plan.notes.get("corridor_touches_bound")
                                  if isinstance(plan.notes, dict) else None)
        velocity_modules = (plan.notes.get("velocity_modules")
                            if isinstance(plan.notes, dict) else None)
        sp0 = float(plan.speed_profile[0]) if len(plan.speed_profile) > 0 else 0.0
        live_log(
            "behavior_planner", event="plan_output",
            scenario=plan.scenario_name,
            valid=bool(plan.valid),
            stop_required=bool(plan.stop_required),
            ramp_speed_mps=float(self._current_speed_mps),
            nominal_speed_mps=float(ctx.nominal_speed_mps),
            path_source=path_source,
            sp0=sp0,
            sp_first3=sp_first3,
            tp_first3=tp_first3,
            notes_reason=notes_reason,
            stop_mode=stop_mode,
            stop_source=stop_source,
            distance_source=distance_source,
            stop_distance_m=_simple(stop_distance_m),
            candidate_modules=_simple(candidate_modules),
            approved_module=approved_module,
            overtake_corridor_escape=_simple(overtake_corridor_escape),
            overtake_effective_lateral_offset_m=_simple(overtake_effective_lateral_offset_m),
            overtake_clearance_m=_simple(overtake_clearance_m),
            containment_mode=containment_mode,
            corridor_touches_bound=_simple(corridor_touches_bound),
            velocity_modules=_simple(velocity_modules),
            route_id=ctx.route.route_id,
            route_source=ctx.route.route_source,
            current_lanelet_id=ctx.route.current_lanelet_id,
            next_lanelet_ids=list(ctx.route.next_lanelet_ids or ()),
            current_node_id=ctx.route.current_node_id,
            upcoming_node_id=ctx.route.upcoming_node_id,
            matched_idx=int(ctx.route.matched_idx),
            target_idx=int(ctx.route.target_idx),
            map_match_error_m=float(ctx.route.map_match_error_m or 0.0),
            waypoint_mode_active=bool(ctx.route.waypoint_mode_active),
            next_semantic_type=ctx.route.next_semantic_type,
            expected_control_type=ctx.route.expected_control_type,
            pose_x=float(ctx.pose.fused_pose.x),
            pose_y=float(ctx.pose.fused_pose.y),
            pose_yaw_rad=float(ctx.pose.fused_pose.yaw),
            mode=self._current_mode,
            ts_plan=float(plan.timestamp),
        )

        # Publicar snapshot serializable al dashboard. Hacemos esta
        # serialización fuera del lock del buffer para minimizar contención.
        self._publish_output(plan)

        # Métricas.
        self._last_plan_dt_ms = (time.monotonic() - tick_start) * 1000.0
        self._frame_idx += 1
        if (time.monotonic() - self._last_status_publish_t) >= self._status_period_s:
            self._publish_status(plan)
            self._last_status_publish_t = time.monotonic()
            sp0 = float(plan.speed_profile[0]) if len(plan.speed_profile) > 0 else 0.0
            notes_reason = plan.notes.get("reason", "?") if isinstance(plan.notes, dict) else "?"
            lid = ctx.route.current_lanelet_id
            lm_ok = self._lanelet_map is not None
            import logging as _log
            _log.getLogger(__name__).info(
                "[ BehPlan ] scen=%-20s valid=%-5s stop_req=%-5s "
                "sp[0]=%.3f ramp=%.3f nominal=%.3f "
                "lanelet=%r lm=%s notes=%s",
                plan.scenario_name,
                plan.valid,
                plan.stop_required,
                sp0,
                self._current_speed_mps,
                ctx.nominal_speed_mps,
                lid,
                "OK" if lm_ok else "NONE",
                notes_reason,
            )
            if not plan.valid:
                print(
                    f"\033[1;97m[ BehaviorPlanner ] :\033[0m "
                    f"\033[1;93mWARN\033[0m invalid plan "
                    f"reason={notes_reason} "
                    f"current_lanelet_id={lid!r} "
                    f"lanelet_map={'OK' if lm_ok else 'NONE'}"
                )

    # ----------------------------------------------------------------
    # Construcción del PlanningContext
    # ----------------------------------------------------------------
    def _build_context(self, *, now_s: float) -> PlanningContext | None:
        """Lee buffers y arma un PlanningContext. None si pose/route faltan."""
        pose = self._pose_buf.read_latest()
        if not isinstance(pose, PoseEstimate):
            return None

        route = self._route_buf.read_latest()
        if not isinstance(route, RouteContext):
            # Sin route, no hay current_lanelet ni regulators. Sigamos con
            # un RouteContext vacío para que LaneKeep al menos intente —
            # el `start_lanelet_id` del scenario evaluará None y caerá en
            # fallback. Es válido: el bootstrap necesita primero NavPlanner.
            route = RouteContext()
        self._ensure_lanelet_map(route)

        lane_obs = self._lane_buf.read_latest() if self._lane_buf is not None else None
        if not isinstance(lane_obs, LaneObservation):
            lane_obs = None

        stopline_obs = (
            self._stopline_buf.read_latest() if self._stopline_buf is not None else None
        )
        if not isinstance(stopline_obs, StoplineObservation):
            stopline_obs = None

        tracked: tuple[TrackedObject, ...] = ()
        if self._tracked_buf is not None:
            payload = self._tracked_buf.read_latest()
            if isinstance(payload, tuple) and all(
                isinstance(o, TrackedObject) for o in payload
            ):
                tracked = payload

        lidar_scan = self._lidar_scan_buf.read_latest() if self._lidar_scan_buf is not None else None
        if not isinstance(lidar_scan, LidarScan):
            lidar_scan = None

        lidar_obstacles: tuple[LidarObstacle, ...] = ()
        if self._lidar_obs_buf is not None:
            payload = self._lidar_obs_buf.read_latest()
            if isinstance(payload, tuple) and all(isinstance(o, LidarObstacle) for o in payload):
                lidar_obstacles = payload

        sign_hints: tuple[dict, ...] = ()
        if self._sign_hints_buf is not None:
            payload = self._sign_hints_buf.read_latest()
            if isinstance(payload, (list, tuple)):
                sign_hints = _filter_recent_sign_hints(payload, now_s=now_s)

        return PlanningContext(
            now_s=float(now_s),
            dt=self._dt_s,
            horizon_n=self._horizon_n,
            nominal_speed_mps=self._current_speed_mps,   # rampado, no el nominal fijo
            max_speed_mps=self._max_speed_mps,
            pose=pose,
            route=route,
            lane_observation=lane_obs,
            stopline_observation=stopline_obs,
            tracked_objects=tracked,
            lidar_scan=lidar_scan,
            lidar_obstacles=lidar_obstacles,
            lanelet_map=self._lanelet_map,
            sign_hints=sign_hints,
        )

    # ----------------------------------------------------------------
    # Publicación al dashboard
    # ----------------------------------------------------------------
    def _publish_output(self, plan: BehaviorOutput) -> None:
        """Serializa BehaviorOutput a dict para el canal IPC."""
        payload: dict[str, Any] = {
            "timestamp": float(plan.timestamp),
            "dt": float(plan.dt),
            "scenario_name": str(plan.scenario_name),
            "valid": bool(plan.valid),
            "stop_required": bool(plan.stop_required),
            "min_moving_speed_mps": (
                float(plan.min_moving_speed_mps)
                if plan.min_moving_speed_mps is not None
                else None
            ),
            "target_path": plan.target_path.tolist(),
            "speed_profile": plan.speed_profile.tolist(),
            "notes": _serialize_notes(plan.notes),
        }
        try:
            self._output_sender.send(payload)
        except Exception:
            # IPC no debe romper el ciclo del planner — logueamos y seguimos.
            if self.logging is not None:
                self.logging.exception("failed to publish BEHAVIOR_OUTPUT_MSG")

    def _publish_status(self, plan: BehaviorOutput) -> None:
        """Status para gauges del dashboard (2 Hz)."""
        fps = 1.0 / self._last_plan_dt_ms * 1000.0 if self._last_plan_dt_ms > 0 else 0.0
        payload = {
            "active_scenario": str(plan.scenario_name),
            "horizon_n": int(plan.horizon_n),
            "fps": float(fps),
            "last_plan_dt_ms": float(self._last_plan_dt_ms),
        }
        try:
            self._status_sender.send(payload)
        except Exception:
            if self.logging is not None:
                self.logging.exception("failed to publish BEHAVIOR_PLANNER_STATUS")

    def _publish_empty(self, *, reason: str) -> None:
        """Cuando faltan inputs: publicar plan vacío con stop_required."""
        empty = BehaviorOutput(
            timestamp=time.time(),
            dt=self._dt_s,
            scenario_name=ScenarioName.FALLBACK.value,
            valid=False,
            stop_required=True,
            notes={"reason": reason},
        )
        self._output_buf.write(empty, timestamp=empty.timestamp)
        self._publish_output(empty)
        route = self._route_buf.read_latest() if self._route_buf is not None else None
        if not isinstance(route, RouteContext):
            route = None
        pose = self._pose_buf.read_latest() if self._pose_buf is not None else None
        if not isinstance(pose, PoseEstimate):
            pose = None
        live_log(
            "behavior_planner", event="plan_empty",
            reason=reason, mode=self._current_mode,
            ramp_speed_mps=float(self._current_speed_mps),
            route_active=bool(route.route_active) if route is not None else False,
            route_id=route.route_id if route is not None else None,
            current_lanelet_id=route.current_lanelet_id if route is not None else None,
            current_node_id=route.current_node_id if route is not None else None,
            upcoming_node_id=route.upcoming_node_id if route is not None else None,
            map_match_error_m=float(route.map_match_error_m or 0.0) if route is not None else None,
            pose_x=float(pose.fused_pose.x) if pose is not None else None,
            pose_y=float(pose.fused_pose.y) if pose is not None else None,
            pose_yaw_rad=float(pose.fused_pose.yaw) if pose is not None else None,
        )


def _serialize_notes(notes: dict[str, Any]) -> dict[str, Any]:
    """Convierte el `notes` dict a algo serializable por JSON.

    Los scenarios pueden meter ndarray, dataclasses, etc. en `notes`.
    Acá los pasamos a tipos primitivos para que no exploten la cola.
    """
    out: dict[str, Any] = {}
    for k, v in notes.items():
        if hasattr(v, "tolist"):
            out[str(k)] = v.tolist()
        elif hasattr(v, "__dataclass_fields__"):
            out[str(k)] = asdict(v)
        elif isinstance(v, (list, tuple)):
            out[str(k)] = [_simple(x) for x in v]
        elif isinstance(v, dict):
            out[str(k)] = {str(kk): _simple(vv) for kk, vv in v.items()}
        else:
            out[str(k)] = _simple(v)
    return out


def _slim_sign_hints(sign_hints: tuple[dict, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hint in tuple(sign_hints or ())[:6]:
        if not isinstance(hint, dict):
            continue
        out.append({
            "kind": _simple(hint.get("kind")),
            "raw_class": _simple(hint.get("raw_class")),
            "distance_m": _simple(hint.get("distance_m")),
            "distance_source": _simple(hint.get("distance_source")),
            "confidence": _simple(hint.get("confidence")),
            "box_area": _simple(hint.get("box_area")),
            "timestamp": _simple(hint.get("timestamp")),
        })
    return out


def _filter_recent_sign_hints(payload, *, now_s: float) -> tuple[dict, ...]:
    try:
        import config as _cfg
        max_age_s = float(getattr(_cfg, "SIGN_HINT_MAX_AGE_S", 1.5))
    except Exception:
        max_age_s = 1.5
    hints: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        hint = dict(item)
        try:
            ts = float(hint.get("timestamp", now_s))
        except (TypeError, ValueError):
            ts = float(now_s)
        if (float(now_s) - ts) > max_age_s:
            continue
        hints.append(hint)
    return tuple(hints)


def _simple(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "__dataclass_fields__"):
        return asdict(v)
    if isinstance(v, (list, tuple)):
        return [_simple(item) for item in v]
    if isinstance(v, dict):
        return {str(key): _simple(value) for key, value in v.items()}
    return str(v)
