"""Plan TANDA 1.3 — Matching request↔reply de CALIBRATION via correlation_id.

Antes el canal CALIBRATION era pub/sub mismo topic para request y reply,
sin matching: con dos requests concurrentes (race), las dos respuestas
volvían sin manera de parear cuál es de cuál.

Ahora:
  * El cliente bus (``emit_calibration``) auto-genera un UUID y lo
    adjunta al payload.
  * El thread serial (``threadCalibration``) inyecta el correlation_id
    en el adapter, y CADA ``emit`` posterior del FSM Calibration legacy
    lo carga automáticamente.
  * El reply lleva el mismo correlation_id que el request → matchable.

Este test mockea el cliente bus y el adapter, simula 3 requests
concurrentes con UUIDs distintos, y verifica que cada reply caza con su
request.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.gui.components._bus_adapter import BusSocketAdapter


def _make_adapter_with_mock_sender():
    """Crea un BusSocketAdapter con un sender mockeado para Calibration.

    Pre-poplamos ``_senders`` para evitar que ``emit`` instancie el
    sender real (que intentaría conectarse al bus ZMQ).
    """
    adapter = BusSocketAdapter()
    mock_sender = MagicMock()
    adapter._senders["Calibration"] = mock_sender
    return adapter, mock_sender


def test_adapter_injects_correlation_id_into_reply():
    """El adapter adjunta correlation_id a cada payload de emit()."""
    adapter, sender = _make_adapter_with_mock_sender()
    adapter.set_correlation_id("uuid-1234")
    adapter.emit("Calibration", {"action": "current_angle", "data": 12.5})

    sender.send.assert_called_once()
    sent_payload = sender.send.call_args[0][0]
    assert sent_payload["correlation_id"] == "uuid-1234"
    assert sent_payload["action"] == "current_angle"


def test_adapter_preserves_explicit_correlation_id():
    """Si el caller setea su propio correlation_id en el payload, no se
    sobrescribe con el adapter — esto permite que un test (o un futuro
    helper) emita réplicas con un id explícito."""
    adapter, sender = _make_adapter_with_mock_sender()
    adapter.set_correlation_id("adapter-default")
    adapter.emit("Calibration", {"action": "x", "correlation_id": "explicit"})

    sent_payload = sender.send.call_args[0][0]
    assert sent_payload["correlation_id"] == "explicit"


def test_adapter_clears_correlation_id_between_turns():
    """Después de set_correlation_id(None), emits subsiguientes NO
    arrastran el id viejo (evita confundir replies de comandos sin
    correlation con replies del comando anterior)."""
    adapter, sender = _make_adapter_with_mock_sender()
    adapter.set_correlation_id("uuid-1")
    adapter.emit("Calibration", {"action": "step1"})
    adapter.set_correlation_id(None)
    adapter.emit("Calibration", {"action": "broadcast_unrelated"})

    payloads = [call.args[0] for call in sender.send.call_args_list]
    assert payloads[0]["correlation_id"] == "uuid-1"
    assert "correlation_id" not in payloads[1]


def test_three_concurrent_requests_each_reply_matches():
    """Simula 3 requests con UUIDs distintos. El adapter procesa cada
    uno secuencialmente (un único thread serial worker) y cada reply
    debe llevar el correlation_id del request que la disparó.
    """
    adapter, sender = _make_adapter_with_mock_sender()

    requests = [
        ("uuid-A", {"action": "current_angle", "data": 12.5}),
        ("uuid-B", {"action": "current_speed", "data": 300}),
        ("uuid-C", {"action": "calibration_run_done"}),
    ]

    # Simular: cada request abre/cierra una "ventana" de correlation_id
    # idéntica a como lo hace threadCalibration.thread_work().
    for correlation_id, reply_payload in requests:
        adapter.set_correlation_id(correlation_id)
        try:
            adapter.emit("Calibration", reply_payload)
        finally:
            adapter.set_correlation_id(None)

    payloads = [call.args[0] for call in sender.send.call_args_list]
    assert len(payloads) == 3
    assert payloads[0]["correlation_id"] == "uuid-A"
    assert payloads[0]["action"] == "current_angle"
    assert payloads[1]["correlation_id"] == "uuid-B"
    assert payloads[1]["action"] == "current_speed"
    assert payloads[2]["correlation_id"] == "uuid-C"
    assert payloads[2]["action"] == "calibration_run_done"


def test_client_emit_calibration_generates_unique_ids():
    """``zmq_bus_client.emit_calibration`` debe generar un UUID distinto
    por llamada para evitar colisiones."""
    from src.gui.client.zmq_bus_client import ZmqBusClient

    client = ZmqBusClient.__new__(ZmqBusClient)
    captured = []
    # Stub emit_message para capturar el payload sin tocar la red.
    client.emit_message = lambda name, value: captured.append(  # type: ignore[method-assign]
        (name, dict(value) if isinstance(value, dict) else value)
    )

    id1 = client.emit_calibration({"Action": "start"})
    id2 = client.emit_calibration({"Action": "start"})  # Mismo payload
    id3 = client.emit_calibration({"Action": "start"})

    assert id1 != id2
    assert id2 != id3
    assert captured[0][1]["correlation_id"] == id1
    assert captured[1][1]["correlation_id"] == id2
    assert captured[2][1]["correlation_id"] == id3
    # Forma UUID4 hex (32 chars, sólo hex)
    for uuid_hex in (id1, id2, id3):
        assert len(uuid_hex) == 32
        assert all(c in "0123456789abcdef" for c in uuid_hex)
