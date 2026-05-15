"""Qt-friendly SocketIO client.

Wraps ``socketio.Client`` (from ``python-socketio[client]``) in a ``QObject``
so widgets can ``QObject.connect`` to one ``pyqtSignal`` per event instead of
juggling raw callbacks across threads.

Why a wrapper:

* The python-socketio client lives in its own thread (background asyncio).
  Emitting Qt signals from there is the canonical thread-safe way to push
  data into the GUI — Qt marshals the call onto the main event loop.
* Widgets stay dumb: they ``client.location_signal.connect(self.on_location)``
  and call ``client.emit_message(CMD_KLEM, "30")``. They don't know about
  socketio at all.
* Camera frames are decoded here (JPEG / base64 PNG → ``QImage``) so the
  rest of the GUI never sees raw bytes.

Connection model:

* ``start(host, port)`` — non-blocking, schedules connect + reconnect.
* ``stop()`` — graceful shutdown.
* Auto-reconnect uses python-socketio's built-in exponential backoff. We
  bubble the state up through ``connection_status_changed("connected" |
  "disconnected" | "error")`` so the UI can show a banner.

Envelope conventions (mirrors ``processDashboard.py`` ↔ Angular contract):

* Outgoing ``message`` is a JSON-serialised string ``{"Name": ..., "Value":
  ...}`` — see ``emit_message``.
* Incoming events come in two shapes; we normalise both to "the inner
  payload" before emitting:
    - ``{"value": X}``   for telemetry/state (BatteryLvl, CurrentSpeed, ...)
    - ``{"data":  X}``   for system events (heartbeat, session_access, ...)
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from typing import Any, Optional

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage

try:
    import socketio  # python-socketio[client]
except ImportError as exc:  # pragma: no cover - dependency error surfaces at runtime
    raise ImportError(
        "python-socketio[client] is required for the dashboard GUI. "
        "Install it via `pip install -r requirements.txt`."
    ) from exc

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    # OpenCV is optional for the GUI: without it we still display frames using
    # Qt's own JPEG/PNG loader (slower but works). The runtime always has cv2,
    # but a remote `--monitor` machine might not.
    _HAS_CV2 = False

from . import events as ev


_logger = logging.getLogger(__name__)


def _unwrap_payload(payload: Any) -> Any:
    """Return the inner payload from a ``{"value": ...}`` / ``{"data": ...}``
    envelope, or the payload itself if it doesn't follow the convention.

    The Flask backend isn't 100% consistent about which key it uses, so the
    GUI normalises here once and the widgets only see the value.
    """
    if isinstance(payload, dict):
        if "value" in payload:
            return payload["value"]
        if "data" in payload:
            return payload["data"]
    return payload


def _decode_jpeg_bytes(data: bytes) -> Optional[QImage]:
    """Decode a JPEG byte string into a ``QImage`` (or ``None`` on failure).

    Prefers OpenCV (faster) and falls back to ``QImage.loadFromData`` so the
    monitor-mode laptop doesn't have to install OpenCV.
    """
    if not data:
        return None
    if _HAS_CV2:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        # OpenCV gives BGR; Qt wants RGB. Convert once.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        # Copy so the QImage owns its memory once the numpy buffer is gone.
        return QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    img = QImage()
    if img.loadFromData(data):
        return img
    return None


def _decode_base64_image(b64: str) -> Optional[QImage]:
    """Decode a base64-encoded PNG/JPEG string into a ``QImage``.

    Used as a fallback when the backend hasn't been upgraded to emit the
    binary ``serialCamera_bin`` event yet.
    """
    if not b64:
        return None
    try:
        # The legacy frontend prefixes data with `data:image/png;base64,`;
        # be lenient and strip it.
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
    except (ValueError, TypeError):
        return None
    img = QImage()
    if img.loadFromData(raw):
        return img
    return None


class SocketIOClient(QObject):
    """Single SocketIO bus for the entire GUI.

    One instance lives in the ``MainWindow``; widgets receive a reference and
    connect to whichever signals they care about.
    """

    # ---- Connection lifecycle -------------------------------------------
    connection_status_changed = pyqtSignal(str)  # "connected"/"disconnected"/"error"

    # ---- Telemetry (backend → GUI) --------------------------------------
    # Each pyqtSignal carries the *unwrapped* payload (see _unwrap_payload),
    # which is whatever the backend put inside the {value/data: ...} envelope.
    enable_button_signal = pyqtSignal(bool)
    response_signal = pyqtSignal(object)
    session_access_signal = pyqtSignal(bool)
    serial_state_signal = pyqtSignal(bool)
    heartbeat_signal = pyqtSignal(object)
    heartbeat_disconnect_signal = pyqtSignal(object)
    memory_signal = pyqtSignal(float)
    cpu_signal = pyqtSignal(object)              # {"usage": float, "temp": float}
    battery_signal = pyqtSignal(float)
    instant_consumption_signal = pyqtSignal(float)
    current_speed_signal = pyqtSignal(int)
    current_steer_signal = pyqtSignal(int)
    steering_limits_signal = pyqtSignal(object)  # {"upper": int, "lower": int}
    state_change_signal = pyqtSignal(str)
    warning_signal_signal = pyqtSignal(str)
    recording_signal = pyqtSignal(bool)
    location_signal = pyqtSignal(object)         # {"x": float, "y": float, ...}
    cars_signal = pyqtSignal(object)             # list[dict]
    semaphores_signal = pyqtSignal(object)       # list[dict]
    navigation_status_signal = pyqtSignal(object)
    behavior_output_signal = pyqtSignal(object)   # dict: target_path, speed_profile, notes
    lidar_scan_signal = pyqtSignal(object)
    lidar_status_signal = pyqtSignal(object)
    detected_objects_signal = pyqtSignal(object)
    tracked_objects_signal = pyqtSignal(object)
    motor_command_signal = pyqtSignal(object)     # dict: steering_deg, speed_mps, valid, source, reason
    gps_fix_signal = pyqtSignal(object)           # dict: world_x, world_y, timestamp
    camera_frame_signal = pyqtSignal(QImage)     # decoded JPEG/PNG → QImage
    line_following_debug_signal = pyqtSignal(QImage)
    line_following_status_signal = pyqtSignal(object)
    calibration_signal = pyqtSignal(object)
    calib_run_done_signal = pyqtSignal(object)
    calib_pwm_data_signal = pyqtSignal(object)
    console_log_signal = pyqtSignal(str)
    load_back_signal = pyqtSignal(object)        # response to a `load` request

    def __init__(self, host: str = "localhost", port: int = 5005,
                 reconnect: bool = True, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._host = host
        self._port = port
        self._url = f"http://{host}:{port}"
        self._connect_lock = threading.Lock()
        self._started = False

        # python-socketio handles reconnect internally; we just configure it.
        self._sio = socketio.Client(
            reconnection=reconnect,
            reconnection_attempts=0,        # 0 = infinite
            reconnection_delay=1,
            reconnection_delay_max=10,
            randomization_factor=0.5,
            logger=False,
            engineio_logger=False,
        )

        self._wire_handlers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Connect (in a background thread) to ``http://host:port``.

        Returns immediately — the connection happens asynchronously and the
        ``connection_status_changed`` signal reports the outcome.
        """
        if host is not None:
            self._host = host
        if port is not None:
            self._port = port
        self._url = f"http://{self._host}:{self._port}"

        with self._connect_lock:
            if self._started:
                return
            self._started = True

        threading.Thread(
            target=self._connect_loop, name="SocketIOClient-connect", daemon=True
        ).start()

    def stop(self) -> None:
        """Disconnect cleanly. Safe to call multiple times."""
        with self._connect_lock:
            if not self._started:
                return
            self._started = False
        try:
            if self._sio.connected:
                self._sio.disconnect()
        except Exception as exc:  # pragma: no cover - best effort
            _logger.debug("SocketIO disconnect error: %s", exc)

    def is_connected(self) -> bool:
        return self._sio.connected

    @property
    def url(self) -> str:
        return self._url

    # ---- Outgoing -----------------------------------------------------
    def emit_message(self, name: str, value: Any = None) -> None:
        """Send a ``{"Name": name, "Value": value}`` envelope as the
        ``message`` event (the contract every Angular component used).

        ``value`` may be a primitive (``str``/``int``/``bool``) or a dict.
        We always JSON-serialise the whole envelope to match what the
        backend's ``handle_message`` does (``json.loads(data)``).
        """
        if not self._sio.connected:
            _logger.debug("emit_message(%s) dropped: not connected", name)
            return
        envelope: dict[str, Any] = {"Name": name}
        if value is not None:
            envelope["Value"] = value
        try:
            self._sio.emit("message", json.dumps(envelope))
        except Exception as exc:  # pragma: no cover - log and move on
            _logger.warning("emit_message(%s) failed: %s", name, exc)

    def emit_save(self, table_state: dict) -> None:
        """Send the table-state ``save`` event (legacy routes/params persist)."""
        if not self._sio.connected:
            return
        try:
            self._sio.emit(ev.ROUTE_SAVE, table_state)
        except Exception as exc:
            _logger.warning("emit_save failed: %s", exc)

    def emit_load(self) -> None:
        """Request the backend re-emit the saved table state via ``loadBack``."""
        if not self._sio.connected:
            return
        try:
            self._sio.emit(ev.ROUTE_LOAD)
        except Exception as exc:
            _logger.warning("emit_load failed: %s", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _connect_loop(self) -> None:
        """Initial connect + python-socketio's own reconnect handles the rest."""
        try:
            self._sio.connect(self._url, transports=["websocket", "polling"])
            # ``wait()`` blocks until disconnect; reconnect is handled internally.
            self._sio.wait()
        except socketio.exceptions.ConnectionError as exc:
            _logger.warning("SocketIO connect failed: %s", exc)
            self.connection_status_changed.emit("error")
        except Exception as exc:  # pragma: no cover - last resort
            _logger.error("SocketIO loop crashed: %s", exc)
            self.connection_status_changed.emit("error")

    def _wire_handlers(self) -> None:
        """Bind every incoming event to a private handler that emits a signal."""

        @self._sio.event
        def connect():  # noqa: ANN001 - socketio decorators
            _logger.info("SocketIO connected to %s", self._url)
            self.connection_status_changed.emit("connected")

        @self._sio.event
        def disconnect():  # noqa: ANN001
            _logger.info("SocketIO disconnected")
            self.connection_status_changed.emit("disconnected")

        @self._sio.event
        def connect_error(data):  # noqa: ANN001
            _logger.warning("SocketIO connect_error: %s", data)
            self.connection_status_changed.emit("error")

        # ---- Buttons / state ----
        @self._sio.on(ev.EVT_ENABLE_BUTTON)
        def _on_enable_button(data):  # noqa: ANN001
            self.enable_button_signal.emit(bool(_unwrap_payload(data)))

        @self._sio.on(ev.EVT_RESPONSE)
        def _on_response(data):  # noqa: ANN001
            self.response_signal.emit(data)

        @self._sio.on(ev.EVT_SESSION_ACCESS)
        def _on_session_access(data):  # noqa: ANN001
            self.session_access_signal.emit(bool(_unwrap_payload(data)))

        @self._sio.on(ev.EVT_CURRENT_SERIAL_STATE)
        def _on_serial_state(data):  # noqa: ANN001
            self.serial_state_signal.emit(bool(_unwrap_payload(data)))

        # ---- Heartbeat ----
        @self._sio.on(ev.EVT_HEARTBEAT)
        def _on_heartbeat(data):  # noqa: ANN001
            self.heartbeat_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_HEARTBEAT_DISCONNECT)
        def _on_heartbeat_disconnect(data):  # noqa: ANN001
            self.heartbeat_disconnect_signal.emit(_unwrap_payload(data))

        # ---- Hardware ----
        @self._sio.on(ev.EVT_MEMORY)
        def _on_memory(data):  # noqa: ANN001
            value = _unwrap_payload(data)
            try:
                self.memory_signal.emit(float(value))
            except (TypeError, ValueError):
                _logger.debug("Bad memory payload: %r", value)

        @self._sio.on(ev.EVT_CPU)
        def _on_cpu(data):  # noqa: ANN001
            self.cpu_signal.emit(_unwrap_payload(data))

        # ---- Telemetry ----
        @self._sio.on(ev.EVT_BATTERY)
        def _on_battery(data):  # noqa: ANN001
            try:
                self.battery_signal.emit(float(_unwrap_payload(data)))
            except (TypeError, ValueError):
                pass

        @self._sio.on(ev.EVT_INSTANT_CONSUMPTION)
        def _on_consumption(data):  # noqa: ANN001
            try:
                self.instant_consumption_signal.emit(float(_unwrap_payload(data)))
            except (TypeError, ValueError):
                pass

        @self._sio.on(ev.EVT_CURRENT_SPEED)
        def _on_speed(data):  # noqa: ANN001
            # Wire format: el brain emite CurrentSpeed en mm/s (matchea el
            # encoder del Nucleo y el `speed_mmps` que sintetiza
            # threadSimFeedback). El Speedometer y el chart están escalados
            # a cm/s con range [-50, 50]. Sin esta división, comandar 5
            # cm/s = 50 mm/s pega el gauge al máximo y se ve binario
            # (0 ↔ 50). Dividimos acá una sola vez para que TODOS los
            # consumers (Speedometer, chart, futuros readouts) reciban la
            # misma unidad declarada por la UI: cm/s.
            try:
                self.current_speed_signal.emit(
                    int(round(float(_unwrap_payload(data)) / 10.0))
                )
            except (TypeError, ValueError):
                pass

        @self._sio.on(ev.EVT_CURRENT_STEER)
        def _on_steer(data):  # noqa: ANN001
            # Mismo motivo que speed: el brain emite CurrentSteer en
            # décimas de grado (×10, range ±250) y los gauges/chart
            # esperan grados (range ±25). Sin esta división el
            # SteeringIndicator y la curva del chart quedan saturados al
            # primer toque de flecha.
            try:
                self.current_steer_signal.emit(
                    int(round(float(_unwrap_payload(data)) / 10.0))
                )
            except (TypeError, ValueError):
                pass

        @self._sio.on(ev.EVT_STEERING_LIMITS)
        def _on_limits(data):  # noqa: ANN001
            self.steering_limits_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_STATE_CHANGE)
        def _on_state(data):  # noqa: ANN001
            value = _unwrap_payload(data)
            self.state_change_signal.emit(str(value) if value is not None else "")

        @self._sio.on(ev.EVT_WARNING_SIGNAL)
        def _on_warning(data):  # noqa: ANN001
            value = _unwrap_payload(data)
            self.warning_signal_signal.emit(str(value) if value is not None else "")

        @self._sio.on(ev.EVT_RECORDING)
        def _on_recording(data):  # noqa: ANN001
            self.recording_signal.emit(bool(_unwrap_payload(data)))

        @self._sio.on(ev.EVT_LOCATION)
        def _on_location(data):  # noqa: ANN001
            self.location_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_CARS)
        def _on_cars(data):  # noqa: ANN001
            self.cars_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_SEMAPHORES)
        def _on_semaphores(data):  # noqa: ANN001
            self.semaphores_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_NAVIGATION_STATUS)
        def _on_nav_status(data):  # noqa: ANN001
            self.navigation_status_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_BEHAVIOR_OUTPUT)
        def _on_behavior_output(data):  # noqa: ANN001
            self.behavior_output_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_LIDAR_SCAN)
        def _on_lidar_scan(data):  # noqa: ANN001
            self.lidar_scan_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_LIDAR_STATUS)
        def _on_lidar_status(data):  # noqa: ANN001
            self.lidar_status_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_DETECTED_OBJECTS)
        def _on_detected_objects(data):  # noqa: ANN001
            self.detected_objects_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_TRACKED_OBJECTS)
        def _on_tracked_objects(data):  # noqa: ANN001
            self.tracked_objects_signal.emit(_unwrap_payload(data))

        # ---- Camera ----
        # Preferred (binary JPEG): payload is raw bytes.
        @self._sio.on(ev.EVT_CAMERA_BIN)
        def _on_camera_bin(data):  # noqa: ANN001
            img = _decode_jpeg_bytes(data if isinstance(data, (bytes, bytearray))
                                     else bytes(data) if data else b"")
            if img is not None and not img.isNull():
                self.camera_frame_signal.emit(img)

        # Legacy fallback (base64 PNG inside {value: ...}).
        @self._sio.on(ev.EVT_CAMERA_BASE64)
        def _on_camera_b64(data):  # noqa: ANN001
            payload = _unwrap_payload(data)
            if isinstance(payload, str):
                img = _decode_base64_image(payload)
                if img is not None and not img.isNull():
                    self.camera_frame_signal.emit(img)

        # ---- Line following ----
        @self._sio.on(ev.EVT_LINE_FOLLOWING_DEBUG)
        def _on_lf_debug(data):  # noqa: ANN001
            payload = _unwrap_payload(data)
            img = None
            if isinstance(payload, str):
                img = _decode_base64_image(payload)
            elif isinstance(payload, (bytes, bytearray)):
                img = _decode_jpeg_bytes(bytes(payload))
            if img is not None and not img.isNull():
                self.line_following_debug_signal.emit(img)

        @self._sio.on(ev.EVT_LINE_FOLLOWING_STATUS)
        def _on_lf_status(data):  # noqa: ANN001
            self.line_following_status_signal.emit(_unwrap_payload(data))

        # ---- Calibration ----
        @self._sio.on(ev.EVT_CALIBRATION)
        def _on_calibration(data):  # noqa: ANN001
            self.calibration_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_CALIB_RUN_DONE)
        def _on_calib_done(data):  # noqa: ANN001
            self.calib_run_done_signal.emit(_unwrap_payload(data))

        @self._sio.on(ev.EVT_CALIB_PWM_DATA)
        def _on_calib_pwm(data):  # noqa: ANN001
            self.calib_pwm_data_signal.emit(_unwrap_payload(data))

        # ---- Misc ----
        @self._sio.on(ev.EVT_CONSOLE_LOG)
        def _on_console(data):  # noqa: ANN001
            value = _unwrap_payload(data)
            self.console_log_signal.emit(str(value) if value is not None else "")

        @self._sio.on(ev.EVT_LOAD_BACK)
        def _on_load_back(data):  # noqa: ANN001
            self.load_back_signal.emit(_unwrap_payload(data))

        # ---- MPC / dispatcher telemetry ----
        @self._sio.on(ev.EVT_MOTOR_COMMAND_MSG)
        def _on_motor_cmd(data):  # noqa: ANN001
            self.motor_command_signal.emit(_unwrap_payload(data))

        # ---- GPS / LoCSys fix (from threadLocSys via processDashboard) ----
        @self._sio.on(ev.EVT_GPS_FIX)
        def _on_gps_fix(data):  # noqa: ANN001
            self.gps_fix_signal.emit(_unwrap_payload(data))
