"""Verify the ``Record`` button routes to the correct enum after the
``CMD_RECORDING = "Record"`` fix.

Bug: the GUI used to publish to the ``Recording`` enum (state topic,
Owner=threadCamera) but threadCamera subscribes to ``Record`` (command
topic, Owner=Dashboard). The button silently no-op'd.
"""

from __future__ import annotations

import time

import pytest

from src.core.bus.shim import messageHandlerSubscriber
from src.core.bus.topics import RECORD
from src.gui.client import events as ev


def test_cmd_recording_name_targets_record_enum():
    """The constant the widget sends must match the enum threadCamera
    subscribes to (``Record``), not the state-topic enum (``Recording``)."""
    assert ev.CMD_RECORDING == "Record"


def test_emit_record_reaches_record_subscriber(broker, qapp):
    """End-to-end: ZmqBusClient.emit_message("Record", True) → bus →
    ``Record`` subscriber receives True. Pre-fix this would silently land
    on the ``Recording`` topic and ``Record`` would stay empty.
    """
    from src.gui.client.zmq_bus_client import ZmqBusClient

    sub = messageHandlerSubscriber({}, RECORD, "lastonly", subscribe=True)
    sub.subscribe()

    client = ZmqBusClient(host="127.0.0.1")
    client.start()
    time.sleep(0.15)  # slow joiner

    received = None
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        client.emit_message(ev.CMD_RECORDING, True)
        time.sleep(0.05)
        msg = sub.receive()
        if msg is not None:
            received = msg
            break

    client.stop()
    sub.unsubscribe()

    assert received is True


@pytest.fixture
def qapp():
    """Headless QApplication so ZmqBusClient's Qt signals work without a
    display."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
