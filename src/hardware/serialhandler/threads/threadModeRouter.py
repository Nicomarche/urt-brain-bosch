"""Brain-side DRIVING_MODE → StateMachine router.

The legacy ``processDashboard.handle_driving_mode`` used to translate the
GUI's ``DRIVING_MODE="auto"`` envelope into a ``STATE_CHANGE="AUTO"``
publish AND mutate the Manager-proxied shared state dict so other
processes (e.g. :mod:`motor_command_dispatcher`) could read the current
mode without snooping the bus. Removing the dashboard left that
translation un-homed, so the GUI was publishing ``STATE_CHANGE`` directly
and the shared dict drifted stale.

This thread restores the contract:

  * Subscribe to ``DRIVING_MODE`` (commands from the GUI).
  * Call ``StateMachine.request_mode(action)`` — same code path the
    legacy dashboard used. ``request_mode`` runs the TransitionTable,
    updates ``_shared_state["mode"]`` AND publishes ``STATE_CHANGE``.
  * Reject invalid transitions just like the dashboard did (legacy
    behaviour: silently no-op; we log instead).

Lives in ``processSerialHandler`` because it has the smallest thread set
and runs as long as the robot is up. Doesn't actually touch serial — the
placement is purely operational.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.bus.topics import DRIVING_MODE
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.statemachine.stateMachine import StateMachine
from src.templates.threadwithstop import ThreadWithStop


class threadModeRouter(ThreadWithStop):
    """Drains ``DRIVING_MODE`` commands and routes them through StateMachine."""

    _VALID_MODES = {"stop", "manual", "legacy", "auto", "parking"}

    def __init__(self, queuesList, logger: Optional[logging.Logger] = None,
                 debugging: bool = False) -> None:
        super().__init__(pause=0.02)
        self.queuesList = queuesList
        self.logger = logger
        self.debugging = debugging
        self._sub = messageHandlerSubscriber(
            queuesList, DRIVING_MODE, "fifo", subscribe=True
        )
        self._sub.subscribe()
        self._sm: Optional[StateMachine] = None

    def _ensure_state_machine(self) -> Optional[StateMachine]:
        """Lazy lookup so we tolerate the spawn-mode race where the child
        starts before ``StateMachine.initialize_shared_state`` runs in the
        parent. After main.py initialises it the singleton is reachable.
        """
        if self._sm is not None:
            return self._sm
        try:
            self._sm = StateMachine.get_instance()
        except RuntimeError:
            return None
        return self._sm

    def thread_work(self) -> None:
        for _ in range(8):
            cmd = self._sub.receive()
            if cmd is None:
                return
            self._dispatch(cmd)

    def _dispatch(self, raw: object) -> None:
        # GUI sends the bare mode string ("auto", "stop", …) — same as the
        # value the dashboard used to receive in dataDict["Value"].
        mode = str(raw).strip().lower()
        if mode not in self._VALID_MODES:
            self._log(
                "warning",
                f"DRIVING_MODE ignored — unknown value {raw!r}",
            )
            return
        sm = self._ensure_state_machine()
        if sm is None:
            self._log(
                "warning",
                "DRIVING_MODE received before StateMachine initialised; dropping",
            )
            return
        action = f"dashboard_{mode}_button"
        try:
            sm.request_mode(action)
        except Exception as exc:
            self._log(
                "warning",
                f"StateMachine.request_mode({action!r}) failed: {exc}",
            )

    def _log(self, level: str, message: str) -> None:
        if self.logger is not None:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(f"[ threadModeRouter ] {level.upper()} - {message}", flush=True)
