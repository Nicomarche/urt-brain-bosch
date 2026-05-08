# src/control/motor_command_dispatcher.py
#
# `threadMotorCommandDispatcher` — el ÚNICO writer de `SpeedMotor` y
# `SteerMotor` en el path de competencia. Lee el `motor_command_buffer`
# producido por el `MotionController`, lo pasa por el `SafetyGate`, y
# despacha el resultado al firmware Nucleo vía las colas IPC.
#
# Vaciado completo respecto del pre-Fase 6:
#   - Eliminada `NavigationControlPolicy` (lógica de "candidate priority"
#     entre lane follower / sign override / planner). En el nuevo modelo
#     hay UN solo candidate: `MotorCommand` del `MotionController`.
#   - Eliminado `VisualControlCandidate` consumption — ese tipo y su
#     productor (`threadVisualController`) son legacy.
#   - Eliminada la lógica de resume-after-sign — los escenarios del
#     `BehaviorPlanner` se hacen cargo de la coordinación speed/scenario.
#
# Pipeline:
#
#   motor_command_buffer ─► dispatcher ─► SafetyGate.evaluate()
#                                         │
#                                         ├─ valid+passed → SpeedMotor + SteerMotor
#                                         └─ stale/invalid → fallback (0, 0)
#
# Diseño:
#   - Thread DELGADO (SRP): cero decisión propia. Toda la lógica de
#     "es seguro despachar?" vive en `safety_gate.py`. El despacho a la
#     queue IPC es la única responsabilidad acá.
#   - Conversión de unidades: `MotorCommand.steering_deg` (grados de
#     ruedas) y `speed_mps` (m/s) se traducen al formato esperado por el
#     firmware Nucleo (décimas de grado, cm/s × 10). Esto vive ACÁ
#     porque es un detalle del transporte.
#   - State gate: si el dashboard puso al sistema en estado distinto de
#     {AUTO, PARKING}, no escribimos al motor (handover a control manual
#     o pause). El gate sigue activo, solo no se manda.

from __future__ import annotations

import time

from src.control.safety_gate import SafetyGate
from src.core.messaging.allMessages import (
    MotorCommandMsg,
    SpeedMotor,
    StateChange,
    SteerMotor,
)
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.types.control import MotorCommand
from src.templates.threadwithstop import ThreadWithStop
from src.utils.live_log import live_log


class threadMotorCommandDispatcher(ThreadWithStop):
    """Despacha `MotorCommand` al firmware. Sin decisión propia."""

    def __init__(
        self,
        queuesList,
        motor_command_buffer,
        *,
        safety_gate: SafetyGate | None = None,
        behavior_output_buffer=None,
        pose_estimate_buffer=None,
        pause_s: float = 0.02,
        logging=None,
        debugging: bool = False,
    ) -> None:
        # 50 Hz default — algo más rápido que el MPC (20 Hz) para minimizar
        # latencia de actuación.
        super().__init__(pause=float(pause_s))
        self.queuesList = queuesList
        self._motor_cmd_buf = motor_command_buffer
        self._behavior_buf = behavior_output_buffer
        self._pose_buf = pose_estimate_buffer
        self._gate = safety_gate or SafetyGate()
        self.logging = logging
        self.debugging = bool(debugging)

        self._speedMotorSender = messageHandlerSender(queuesList, SpeedMotor)
        self._steerMotorSender = messageHandlerSender(queuesList, SteerMotor)
        self._motorCmdMsgSender = messageHandlerSender(queuesList, MotorCommandMsg)
        self._stateChangeSubscriber = messageHandlerSubscriber(
            queuesList, StateChange, "lastOnly", True
        )
        # Estado del sistema (controlado por el dashboard). Solo en
        # {AUTO, PARKING} se mandan comandos al motor; en DEFAULT/MANUAL
        # se respeta al operador.
        self._current_state: str = "DEFAULT"

        # Fallback directo al StateMachine shared state (Manager proxy).
        # Está disponible cuando main.py inyecta los proxies en queuesList
        # (claves "__sm_state__" y "__sm_lock__"). Cubre el caso de
        # BrokenPipeError en el gateway en spawn mode — si el pipe falla,
        # este path sigue dando el estado real sin depender del gateway.
        self._sm_state = queuesList.get("__sm_state__")
        self._sm_lock = queuesList.get("__sm_lock__")

        # Diagnóstico — última decisión publicada (para el dashboard).
        self._last_dispatched: MotorCommand | None = None

        # Una vez que el IPC entrega su primer StateChange, el Manager proxy
        # no debe volver a pisarlo (el proxy es stale en spawn mode: eventlet
        # en processDashboard rompe la conexión al Manager, por lo que el proxy
        # queda congelado en "STOP" mientras el IPC sí entrega "AUTO").
        self._ipc_mode_received: bool = False

        # Stats para log periódico. Imprime un resumen cada N segundos para
        # que cualquier regresión (fallback continuo, state stuck en DEFAULT,
        # etc.) sea visible sin tener que abrir el dashboard. NO afecta al
        # path crítico — es puramente informativo.
        self._stats_period_s = 2.0
        self._stats_last_t = time.time()
        self._stats_count = 0
        self._stats_last_reason = ""
        self._stats_last_speed = 0.0
        self._stats_last_steer = 0.0

    # ----------------------------------------------------------------
    def thread_work(self) -> None:
        self._consume_state_change()

        # Lee el último MotorCommand. Si todavía no hay nada, el gate emite
        # fallback con razón "no_motor_command".
        motor_cmd = self._motor_cmd_buf.read_latest() if self._motor_cmd_buf else None

        # Timestamps de los inputs upstream (para el watchdog).
        behavior_ts = (
            self._read_timestamp(self._behavior_buf) if self._behavior_buf else None
        )
        pose_ts = (
            self._read_timestamp(self._pose_buf) if self._pose_buf else None
        )

        cmd = self._gate.evaluate(
            motor_command=motor_cmd if isinstance(motor_cmd, MotorCommand) else None,
            behavior_timestamp=behavior_ts,
            pose_timestamp=pose_ts,
        )
        self._last_dispatched = cmd

        # Live JSONL: cada decisión del SafetyGate. Si state != AUTO, igual
        # logueamos para ver por qué el dispatcher no manda nada (típico:
        # el operador olvidó clickear AUTO).
        now_t = time.time()
        live_log(
            "dispatcher", event="dispatch_decision",
            state=self._current_state,
            cmd_valid=bool(cmd.valid),
            cmd_source=getattr(cmd, "source", None),
            cmd_reason=str(getattr(cmd, "reason", "")),
            speed_mps=float(cmd.speed_mps),
            steering_deg=float(cmd.steering_deg),
            behavior_age_s=(now_t - behavior_ts) if behavior_ts else None,
            pose_age_s=(now_t - pose_ts) if pose_ts else None,
            dispatched=(self._current_state in {"AUTO", "PARKING"}),
            had_motor_cmd=isinstance(motor_cmd, MotorCommand),
        )

        # Acumulación para el log periódico. Se hace ANTES del state-gate
        # para que el log siga emitiéndose con state=DEFAULT/MANUAL — útil
        # cuando el operador está debug-eando sin estar todavía en AUTO.
        self._stats_count += 1
        self._stats_last_reason = str(cmd.reason)
        self._stats_last_speed = float(cmd.speed_mps)
        self._stats_last_steer = float(cmd.steering_deg)
        self._maybe_log_stats()

        # Snapshot al dashboard (telemetría — no path crítico).
        try:
            self._motorCmdMsgSender.send(self._serialize(cmd))
        except Exception:
            if self.logging is not None:
                self.logging.exception("failed to publish MotorCommandMsg")

        # Si el estado no es operativo, no escribimos. El gate sigue
        # corriendo (logging continúa para diagnóstico).
        if self._current_state not in {"AUTO", "PARKING"}:
            return

        self._send(cmd)

    # ----------------------------------------------------------------
    def _maybe_log_stats(self) -> None:
        """Imprime un resumen cada `_stats_period_s` segundos.

        Salida típica con todo OK:
            [ Dispatcher ] state=AUTO rate=49.8Hz speed=+0.50m/s steer=+0.0° reason=passthrough

        Salida típica con LaneKeep en fallback (current_lanelet_id=None):
            [ Dispatcher ] state=AUTO rate=49.8Hz speed=+0.00m/s steer=+0.0° reason=motor_command_invalid

        Salida típica con state-gate cerrado:
            [ Dispatcher ] state=DEFAULT rate=49.8Hz speed=+0.00m/s steer=+0.0° reason=no_motor_command
        """
        now = time.time()
        dt = now - self._stats_last_t
        if dt < self._stats_period_s:
            return
        rate = self._stats_count / dt if dt > 0 else 0.0
        # Color según si vamos a despachar realmente (state operativo y
        # speed!=0). Rojo = bloqueado por state, amarillo = state OK pero
        # gate metió fallback, verde = pipeline produciendo movimiento.
        if self._current_state not in {"AUTO", "PARKING"}:
            color = "\033[1;91m"  # rojo
        elif abs(self._stats_last_speed) < 1e-3:
            color = "\033[1;93m"  # amarillo
        else:
            color = "\033[1;92m"  # verde
        msg = (
            f"[ Dispatcher ] state={self._current_state} "
            f"rate={rate:.1f}Hz "
            f"speed={self._stats_last_speed:+.2f}m/s "
            f"steer={self._stats_last_steer:+.1f}deg "
            f"reason={self._stats_last_reason}"
        )
        # Usamos logging en vez de print() — en subprocesos spawn de macOS,
        # print() puede quedar en el buffer del proceso hijo sin aparecer en
        # el log. logging.getLogger() escribe a stderr que sí es unbuffered.
        import logging as _logging
        _logging.getLogger().info(msg)
        self._stats_last_t = now
        self._stats_count = 0

    # ----------------------------------------------------------------
    def _send(self, cmd: MotorCommand) -> None:
        """Convierte unidades y publica a las colas Critical."""
        # `MotorCommand` sigue ISO 8855: +deg = ruedas a la izquierda.
        # El wire legacy `SteerMotor` que consumen Nucleo/sim_bridge usa
        # la convención histórica del proyecto: +deg = ruedas a la derecha.
        # La inversión de signo vive acá para aislar el transporte del MPC.
        steer_value = self._steer_deg_to_wire_x10(cmd.steering_deg)
        self._steerMotorSender.send(str(steer_value))

        # Speed: m/s → cm/s × 10 (firmware espera décimas de cm/s).
        speed_cms_x10 = int(round(float(cmd.speed_mps) * 100.0 * 10.0))
        self._speedMotorSender.send(str(speed_cms_x10))

    # ----------------------------------------------------------------
    @staticmethod
    def _steer_deg_to_wire_x10(steering_deg: float) -> int:
        """Traduce `+izquierda` (MPC) a `+derecha` (wire legacy)."""
        return int(round(-float(steering_deg) * 10.0))

    # ----------------------------------------------------------------
    def _consume_state_change(self) -> None:
        # Primario: IPC StateChange — la única fuente confiable para cambios
        # iniciados desde el dashboard. El Manager proxy NO se actualiza en
        # processDashboard porque eventlet rompe la conexión al Manager bajo
        # spawn mode en macOS (el proxy queda congelado en "STOP").
        try:
            msg = self._stateChangeSubscriber.receive()
            if msg is not None:
                self._current_state = str(msg).upper()
                self._ipc_mode_received = True
                return
        except Exception:
            pass

        # Secundario (solo antes del primer IPC): Manager proxy como fuente de
        # arranque. Cubre el caso en que el sistema arrancó en un estado distinto
        # de DEFAULT antes de que el IPC llegara (ej. AUTO_STATE_RUN_IN_SIM).
        # En cuanto llega el primer mensaje IPC, esta rama queda inactiva para
        # siempre — evita que el proxy stale sobre-escriba el estado real.
        if not self._ipc_mode_received and self._sm_state is not None:
            try:
                mode = self._sm_state.get("mode", None)
                if mode is not None:
                    self._current_state = mode.name.upper()
            except Exception:
                pass

    # ----------------------------------------------------------------
    @staticmethod
    def _read_timestamp(buf) -> float | None:
        """Lee el timestamp del último write al buffer. None si vacío."""
        try:
            _, ts, _ = buf.read_latest(with_metadata=True)
        except Exception:
            return None
        return float(ts) if ts else None

    # ----------------------------------------------------------------
    @staticmethod
    def _serialize(cmd: MotorCommand) -> dict:
        """Serialización mínima para MotorCommandMsg (cola IPC)."""
        return {
            "timestamp": float(cmd.timestamp),
            "steering_deg": float(cmd.steering_deg),
            "speed_mps": float(cmd.speed_mps),
            "valid": bool(cmd.valid),
            "source": str(cmd.source),
            "reason": str(cmd.reason),
        }

    # ----------------------------------------------------------------
    @property
    def last_dispatched(self) -> MotorCommand | None:
        """Para tests/diagnóstico. No usar como API estable."""
        return self._last_dispatched
