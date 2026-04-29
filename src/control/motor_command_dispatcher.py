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

        # Diagnóstico — última decisión publicada (para el dashboard).
        self._last_dispatched: MotorCommand | None = None

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
    def _send(self, cmd: MotorCommand) -> None:
        """Convierte unidades y publica a las colas Critical."""
        # Steering: grados → décimas de grado (formato firmware).
        steer_value = int(round(float(cmd.steering_deg) * 10.0))
        self._steerMotorSender.send(str(steer_value))

        # Speed: m/s → cm/s × 10 (firmware espera décimas de cm/s).
        speed_cms_x10 = int(round(float(cmd.speed_mps) * 100.0 * 10.0))
        self._speedMotorSender.send(str(speed_cms_x10))

    # ----------------------------------------------------------------
    def _consume_state_change(self) -> None:
        msg = self._stateChangeSubscriber.receive()
        if msg is None:
            return
        self._current_state = str(msg or "").strip().upper() or "DEFAULT"

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
