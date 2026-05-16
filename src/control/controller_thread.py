# src/control/controller_thread.py
#
# `threadMotionController` — corre el `MotionController` (MPC Acados) a
# la tasa del planner. Lee `BehaviorOutput` + `PoseEstimate`, llama
# `motion_controller.compute()`, escribe `MotorCommand` al buffer.
#
# Pipeline:
#
#   behavior_output_buffer ─┐
#                           ├──► MotionController.compute() ──► motor_command_buffer
#   pose_estimate_buffer ───┘
#
# Diseño:
#   - Thread DELGADO: cero lógica de control. Toda la traducción
#     `BehaviorOutput → MotorCommand` vive en `MotionController`.
#   - dt del MPC ya está embebido en el solver Acados (generado offline
#     con N pasos de duración fija). Acá no se calcula dt.
#   - Sigue el patrón establecido por `threadBehaviorPlanner` y
#     `threadPoseEstimator`: read inputs, call pure component, write
#     output buffer. SRP estricto.
#
# Por qué un thread separado del dispatcher:
#   - El controller corre a la tasa del planner (~20 Hz, igual al
#     `BEHAVIOR_DT_S`); el dispatcher corre más rápido (~50 Hz) para
#     minimizar latencia de actuación. Separarlos permite ritmos
#     distintos sin coordinación interna.
#   - El gate del dispatcher se ejecuta INDEPENDIENTE del controller —
#     si el controller se cuelga, el dispatcher detecta `motor_command`
#     stale y emite fallback sin bloquearse.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.messaging.allMessages import StateChange
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.types.behavior import BehaviorOutput
from src.core.types.pose import PoseEstimate
from src.templates.threadwithstop import ThreadWithStop

if TYPE_CHECKING:
    from src.core.interfaces.controller import IMotionController
    from src.core.messaging.buffers import LatestValueBuffer


class threadMotionController(ThreadWithStop):
    """Ejecuta `MotionController.compute()` a la tasa del BehaviorPlanner."""

    def __init__(
        self,
        queuesList,
        controller: "IMotionController",
        behavior_output_buffer: "LatestValueBuffer",
        pose_estimate_buffer: "LatestValueBuffer",
        motor_command_buffer: "LatestValueBuffer",
        *,
        pause_s: float = 0.05,
        logging=None,
        debugging: bool = False,
    ) -> None:
        super().__init__(pause=float(pause_s))
        self.queuesList = queuesList
        self._controller = controller
        self._behavior_buf = behavior_output_buffer
        self._pose_buf = pose_estimate_buffer
        self._motor_cmd_buf = motor_command_buffer
        self.logging = logging
        self.debugging = bool(debugging)
        self._state_sub = messageHandlerSubscriber(
            queuesList, StateChange, "lastOnly", subscribe=True
        )
        self._current_state = "DEFAULT"

        # Sequence guard: no recomputamos si el behavior_output no cambió.
        # Recordá que el `BehaviorPlanner` puede correr más lento que el
        # controller en algún edge case; sin guard duplicaríamos comandos.
        self._last_processed_seq: int = -1

    # ----------------------------------------------------------------
    def thread_work(self) -> None:
        self._consume_state_change()

        # 1. Leer último BehaviorOutput. Si es el mismo seq que el anterior,
        #    saltamos (ya fue computado).
        behavior, _, b_seq = self._behavior_buf.read_latest(with_metadata=True)
        if b_seq == self._last_processed_seq or behavior is None:
            return
        if not isinstance(behavior, BehaviorOutput):
            self._warn("behavior_output_buffer payload no es BehaviorOutput")
            return
        self._last_processed_seq = b_seq

        # 2. Leer pose. Sin pose, el MPC no puede computar — emitimos
        #    MotorCommand inválido y dejamos que el dispatcher haga
        #    fallback.
        pose = self._pose_buf.read_latest()
        if not isinstance(pose, PoseEstimate):
            self._publish_invalid("no_pose_estimate")
            return

        # 3. Compute. El controller maneja todos los casos de error
        #    devolviendo MotorCommand con valid=False.
        try:
            cmd = self._controller.compute(behavior, pose)
        except Exception:
            if self.logging is not None:
                self.logging.exception("MotionController.compute crashed")
            self._publish_invalid("motion_controller_exception")
            return

        # 4. Publicar al buffer. El dispatcher lee desde acá.
        self._motor_cmd_buf.write(cmd, timestamp=cmd.timestamp)

    def _consume_state_change(self) -> None:
        try:
            msg = self._state_sub.receive()
        except Exception:
            return
        if msg is None:
            return
        state_name = str(msg or "").strip().upper() or "DEFAULT"
        if state_name == self._current_state:
            return
        self._current_state = state_name
        reset = getattr(self._controller, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                if self.logging is not None:
                    self.logging.exception("MotionController.reset failed on state change")

    # ----------------------------------------------------------------
    def _publish_invalid(self, reason: str) -> None:
        from src.core.types.control import MotorCommand
        import time
        cmd = MotorCommand(
            timestamp=time.time(),
            steering_deg=0.0,
            speed_mps=0.0,
            valid=False,
            source="controller_thread",
            reason=reason,
        )
        self._motor_cmd_buf.write(cmd, timestamp=cmd.timestamp)

    # ----------------------------------------------------------------
    def _warn(self, message: str, **kw: Any) -> None:
        if self.logging is not None:
            self.logging.warning(f"threadMotionController: {message}")
