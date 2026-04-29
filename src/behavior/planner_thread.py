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
#   5. Publica un snapshot serializable al dashboard vía `BehaviorOutputMsg`.
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
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from src.behavior.context import PlanningContext
from src.core.messaging.allMessages import BehaviorOutputMsg, BehaviorPlannerStatus
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.core.types.perception import LaneObservation, StoplineObservation, TrackedObject
from src.core.types.pose import PoseEstimate
from src.core.types.routing import RouteContext
from src.templates.threadwithstop import ThreadWithStop

if TYPE_CHECKING:
    from src.behavior.planner import BehaviorPlanner
    from src.core.messaging.buffers import LatestValueBuffer
    from src.routing.lanelet.lanelet_map import LaneletMap


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
        self._pose_buf = pose_estimate_buffer
        self._route_buf = route_context_buffer
        self._lane_buf = lane_observation_buffer
        self._stopline_buf = stopline_observation_buffer
        self._tracked_buf = tracked_objects_buffer
        self._output_buf = behavior_output_buffer
        self._dt_s = float(dt_s)
        self._horizon_n = int(horizon_n)
        self._nominal_speed_mps = float(nominal_speed_mps)
        self._max_speed_mps = float(max_speed_mps)
        self.logging = logging
        self.debugging = bool(debugging)

        # Senders al dashboard. Mantenemos dos canales: el plan completo
        # (target_path + speed_profile) y un status corto para gauges.
        self._output_sender = messageHandlerSender(queuesList, BehaviorOutputMsg)
        self._status_sender = messageHandlerSender(queuesList, BehaviorPlannerStatus)

        # Métricas para el status — no afectan a la lógica de decisión.
        self._frame_idx = 0
        self._last_status_publish_t = 0.0
        self._status_period_s = 0.5  # 2 Hz al dashboard alcanza
        self._last_plan_dt_ms = 0.0

    # ----------------------------------------------------------------
    # Loop principal
    # ----------------------------------------------------------------
    def thread_work(self) -> None:
        """Un tick: leer inputs → planear → publicar."""
        tick_start = time.monotonic()

        ctx = self._build_context(now_s=time.time())
        if ctx is None:
            # Inputs incompletos — emitir BehaviorOutput vacío para que el
            # downstream pare. NO arrancar a inventar plan basura.
            self._publish_empty(reason="inputs_unavailable")
            return

        plan = self._planner.plan(ctx)

        # Persistir en buffer (consumer = MPC thread).
        self._output_buf.write(plan, timestamp=plan.timestamp)

        # Publicar snapshot serializable al dashboard. Hacemos esta
        # serialización fuera del lock del buffer para minimizar contención.
        self._publish_output(plan)

        # Métricas.
        self._last_plan_dt_ms = (time.monotonic() - tick_start) * 1000.0
        self._frame_idx += 1
        if (time.monotonic() - self._last_status_publish_t) >= self._status_period_s:
            self._publish_status(plan)
            self._last_status_publish_t = time.monotonic()

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

        return PlanningContext(
            now_s=float(now_s),
            dt=self._dt_s,
            horizon_n=self._horizon_n,
            nominal_speed_mps=self._nominal_speed_mps,
            max_speed_mps=self._max_speed_mps,
            pose=pose,
            route=route,
            lane_observation=lane_obs,
            stopline_observation=stopline_obs,
            tracked_objects=tracked,
            lanelet_map=self._lanelet_map,
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
            "target_path": plan.target_path.tolist(),
            "speed_profile": plan.speed_profile.tolist(),
            "notes": _serialize_notes(plan.notes),
        }
        try:
            self._output_sender.send(payload)
        except Exception:
            # IPC no debe romper el ciclo del planner — logueamos y seguimos.
            if self.logging is not None:
                self.logging.exception("failed to publish BehaviorOutputMsg")

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
                self.logging.exception("failed to publish BehaviorPlannerStatus")

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


def _simple(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "__dataclass_fields__"):
        return asdict(v)
    return str(v)
