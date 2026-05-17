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
        # Sprint 7: NO materializar el socket en __init__. Si el ZMQ context
        # se inicializa en este proceso después de abrir el puerto serial,
        # la RX del CDC ACM en la Jetson se queda en in_waiting=0 indefinido
        # (bytes_total=0). El socket se crea en el primer thread_work() —
        # el primer click del wizard puede perderse por slow-joiner y el
        # operador re-clickea (UX aceptable, la calibración es manual).

    def thread_work(self) -> None:
        for _ in range(8):
            cmd = self._cmd_subscriber.receive()
            if cmd is None:
                return
            # Filter out our own replies (responses use lowercase ``action``).
            if not isinstance(cmd, dict) or "Action" not in cmd:
                continue
            try:
                self._calibration.handle_calibration_signal(cmd, socketId=None)
            except Exception as exc:
                if self.logger is not None:
                    self.logger.warning("Calibration handler failed: %s", exc)
                else:
                    print(
                        "[ threadCalibration ] ERROR - handler failed:", exc, flush=True
                    )
