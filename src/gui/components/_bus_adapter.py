"""Tiny adapter exposing ``socketio``-style ``emit()`` over the bus.

The legacy :class:`Calibration` class published progress/results via
``self.socketio.emit("Calibration", payload, room=socketId)``. With the
Flask + Flask-SocketIO bridge gone, we keep the Calibration class
unchanged and route its emits to bus topics instead. One adapter
instance per process is enough — it caches lazy senders inside.
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.bus.shim import messageHandlerSender
from src.core.bus.topics import CALIBRATION as CalibrationMsg, CALIB_PWM_DATA, CALIB_RUN_DONE


class BusSocketAdapter:
    """Looks like Flask-SocketIO to the Calibration class.

    Only the subset Calibration actually calls is implemented::

        socketio.emit(event_name, payload, room=<ignored>)

    Anything else logs and drops.
    """

    _EVENT_TO_ENUM = {
        "Calibration": CalibrationMsg,
        "CALIB_RUN_DONE": CALIB_RUN_DONE,
        "CALIB_PWM_DATA": CALIB_PWM_DATA,
    }

    def __init__(self) -> None:
        self._senders: Dict[str, messageHandlerSender] = {}

    def emit(self, event_name: str, payload: Any = None, room: Any = None) -> None:
        # ``room`` is the SocketIO single-client target. The bus is pub/sub
        # so we broadcast — there's exactly one GUI listening, so it's the
        # same effect.
        enum_cls = self._EVENT_TO_ENUM.get(event_name)
        if enum_cls is None:
            return
        sender = self._senders.get(event_name)
        if sender is None:
            sender = messageHandlerSender({}, enum_cls)
            self._senders[event_name] = sender
        sender.send(payload)
