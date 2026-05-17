"""Publishes CPU/memory/temperature to the bus once a second.

Replaces ``processDashboard``'s ``update_hardware_data`` + ``socketio.emit('memory_channel'/'cpu_channel')`` pair. Lives in
``processSerialHandler`` because it has the smallest existing thread set
and runs the whole time the robot is up.

The payload shape matches what the GUI's ``RESOURCE_MONITOR`` handler
expects in :class:`ZmqBusClient`::

    {"memory": <percent>, "cpu": {"usage": <percent>, "temp": <celsius>}}
"""

from __future__ import annotations

import time

import psutil

from src.core.bus.topics import RESOURCE_MONITOR
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.templates.threadwithstop import ThreadWithStop


class threadResourceMonitor(ThreadWithStop):
    """Polls psutil every second and publishes a ``RESOURCE_MONITOR`` message."""

    def __init__(self, queuesList, logger=None, debugging=False, period_s: float = 1.0) -> None:
        super().__init__(pause=period_s)
        self.queuesList = queuesList
        self.logger = logger
        self.debugging = debugging
        self._sender = messageHandlerSender(queuesList, RESOURCE_MONITOR)
        self._cpu_temp_c: float = 0.0
        # Warm-up call so the next ``psutil.cpu_percent(interval=None, ...)``
        # call returns a meaningful delta instead of 0.
        psutil.cpu_percent(interval=None, percpu=False)

    def thread_work(self) -> None:
        cpu_usage = psutil.cpu_percent(interval=None, percpu=False)
        memory_pct = psutil.virtual_memory().percent

        try:
            temps = psutil.sensors_temperatures()
            if temps:
                first_sensor = next(iter(temps.values()))
                if first_sensor and first_sensor[0].current is not None:
                    self._cpu_temp_c = round(first_sensor[0].current)
        except Exception:
            # sensors_temperatures isn't supported on macOS dev boxes.
            pass

        self._sender.send({
            "memory": float(memory_pct),
            "cpu": {"usage": float(cpu_usage), "temp": float(self._cpu_temp_c)},
            "ts": time.time(),
        })
