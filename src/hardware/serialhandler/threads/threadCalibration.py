"""Robot-side calibration thread.

Subscribes to the ``Calibration`` bus topic (commands sent by the GUI
wizard) and dispatches them to the existing :class:`Calibration` class.
The class's ``socketio.emit(...)`` calls are routed back to the bus via
:class:`BusSocketAdapter`, preserving the wire format the wizard expects.

Previously this work happened inside ``processDashboard`` — moved here
when the SocketIO/Flask bridge was deleted so the calibration loop stays
close to the serial hardware it drives.
"""

from __future__ import annotations

import time

from src.core.bus.topics import CALIBRATION as CalibrationMsg
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.gui.components._bus_adapter import BusSocketAdapter
from src.gui.components.calibration import Calibration
from src.templates.threadwithstop import ThreadWithStop


class threadCalibration(ThreadWithStop):
    """Drains the Calibration command topic and runs the calibration FSM."""

    def __init__(self, queuesList, logger=None, debugging=False) -> None:
        super().__init__(pause=0.05)
        self.queuesList = queuesList
        self.logger = logger
        self.debugging = debugging

        self._adapter = BusSocketAdapter()
        self._calibration = Calibration(queuesList, self._adapter)
        self._cmd_subscriber = messageHandlerSubscriber(
            queuesList, CalibrationMsg, "fifo", subscribe=True
        )
        # Materialize the socket immediately so the wizard's first command
        # is not lost to the slow-joiner window.
        self._cmd_subscriber.subscribe()

    def thread_work(self) -> None:
        for _ in range(8):
            cmd = self._cmd_subscriber.receive()
            if cmd is None:
                return
            # Filter out our own replies (responses use lowercase ``action``).
            if not isinstance(cmd, dict) or "Action" not in cmd:
                continue
            # D4: inyectamos el correlation_id del request al adapter para
            # que CADA emit subsiguiente de Calibration lo cargue. Esto
            # convierte el canal CALIBRATION (pub/sub mismo topic para
            # request y reply) en un par request↔reply matchable, sin
            # tocar la lógica de la clase legacy Calibration.
            correlation_id = cmd.get("correlation_id") if isinstance(cmd, dict) else None
            self._adapter.set_correlation_id(correlation_id)
            try:
                self._calibration.handle_calibration_signal(cmd, socketId=None)
            except Exception as exc:
                if self.logger is not None:
                    self.logger.warning("Calibration handler failed: %s", exc)
                else:
                    print(
                        "[ threadCalibration ] ERROR - handler failed:", exc, flush=True
                    )
            finally:
                # Limpieza explícita — futuros emits sin request asociado
                # NO deben pegarle un correlation_id viejo.
                self._adapter.set_correlation_id(None)
