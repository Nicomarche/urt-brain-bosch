#!/usr/bin/env python3
"""
headless_controller — controla el brain via SocketIO sin GUI.

Conecta al dashboard del brain en http://localhost:5005, manda
DrivingMode=auto al iniciar, espera N segundos, y manda DrivingMode=stop
al terminar. Usado por run_test.sh para orquestar tests reproducibles
sin que el operador toque el dashboard.

Uso:
    python3 tools/dev/headless_controller.py --duration 60

Implementación:
- python-socketio.Client con reconnection automática.
- emit('message', json.dumps({"Name":"DrivingMode","Value":"auto"})) — el
  handler del dashboard (processDashboard.handle_message) parsea ese JSON
  y llama TransitionTable.get_next_mode("dashboard_auto_button").
- SIGTERM/SIGINT: shutdown limpio (manda stop, disconnect, exit 0).

KL=30 ya está auto-on en sim por AUTO_KL_RUN_IN_SIM=True (config.py),
no hace falta mandarlo desde acá.
"""

import argparse
import json
import signal
import sys
import time

import socketio


def _emit_mode(sio: socketio.Client, value: str) -> None:
    """Envía un cambio de modo (auto / stop / manual / parking)."""
    payload = json.dumps({"Name": "DrivingMode", "Value": value})
    sio.emit("message", payload)
    print(f"[headless_controller] sent DrivingMode={value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--host", default="http://localhost:5005",
        help="dashboard SocketIO endpoint (default: http://localhost:5005)",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="segundos en AUTO antes de mandar STOP (default: 60)",
    )
    parser.add_argument(
        "--mode", default="auto",
        help="modo a activar (default: auto). Otros: parking, manual, stop",
    )
    args = parser.parse_args()

    sio = socketio.Client(reconnection=True, reconnection_attempts=10,
                          reconnection_delay=0.5, reconnection_delay_max=2.0)

    @sio.event
    def connect() -> None:
        print(f"[headless_controller] connected to {args.host}")

    @sio.event
    def disconnect() -> None:
        print("[headless_controller] disconnected")

    # Shutdown handlers — mandan STOP y desconectan limpio.
    shutdown_called = {"v": False}

    def _shutdown(signum: int = 0, *_: object) -> None:
        if shutdown_called["v"]:
            return
        shutdown_called["v"] = True
        try:
            if sio.connected:
                _emit_mode(sio, "stop")
                # Pequeño delay para que el server procese antes de cerrar.
                time.sleep(0.3)
                sio.disconnect()
        except Exception as exc:
            print(f"[headless_controller] shutdown error: {exc}", file=sys.stderr)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        sio.connect(args.host, wait_timeout=10.0)
    except Exception as exc:
        print(f"[headless_controller] FATAL: cannot connect to {args.host}: {exc}",
              file=sys.stderr)
        return 1

    # Pequeña espera para que el server registre nuestro socket en
    # sessionActive antes de mandar el primer comando — sino el handler
    # nos puede rechazar como "unauthorized user" hasta que el primer
    # heartbeat llegue.
    time.sleep(1.0)

    _emit_mode(sio, args.mode)

    print(f"[headless_controller] running for {args.duration:.1f}s in "
          f"{args.mode.upper()} mode...")
    end_t = time.monotonic() + args.duration
    while time.monotonic() < end_t and not shutdown_called["v"]:
        time.sleep(0.5)

    _shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
