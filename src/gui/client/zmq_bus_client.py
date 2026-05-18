"""ZeroMQ-backed replacement for :mod:`socketio_client`.

Drop-in replacement: exposes the same ``pyqtSignal`` surface and the same
``emit_message`` / ``emit_save`` / ``emit_load`` API the rest of the GUI
already uses, so widgets keep working without changes.

Architecture::

    Robot processBus (XSUB tcp://*:5556, XPUB tcp://*:5557)
        ▲              ▼
        │              │
        │ commands     │ telemetry
        │ (zmq.PUB)    │ (zmq.SUB)
        │              │
        └──── GUI ZmqBusClient ────┘

The client uses the legacy ``messageHandlerSender`` / ``messageHandlerSubscriber``
shims under the hood — they already know how to translate an ``allMessages``
enum into a topic name + QoS profile. We just plumb the resulting payloads
into Qt signals.

Compared to the SocketIO client:

* Camera frames are NOT carried here — they arrive over UDP via
  :class:`UdpVideoClient`. ``camera_frame_signal`` is still declared so
  widgets connecting to it work, but it's never emitted from this class.
* SocketIO's ``session_access`` / ``heartbeat`` / ``response`` round-trips
  with the dashboard are removed. The GUI fakes ``session_access_signal``
  immediately because there is no server-side auth in the new bus.
* ``emit_save`` / ``emit_load`` persist to the local filesystem
  (``~/.config/urt-brain-bosch/routes.json``) — there is no remote
  dashboard to ship them to.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
from typing import Any, Callable, Optional

import zmq
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage

from src.core.bus.shim import messageHandlerSender, messageHandlerSubscriber
from src.core.bus.topics import LEGACY_NAME_TO_TOPIC, Topic

_logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _routes_local_path() -> pathlib.Path:
    """File where ``emit_save`` writes and ``emit_load`` reads from."""
    base = pathlib.Path(
        os.environ.get(
            "URT_GUI_CONFIG_DIR",
            pathlib.Path.home() / ".config" / "urt-brain-bosch",
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    return base / "routes.json"


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


# ``SystemMode`` is only used to validate the Localisation gate ("manual
# localisation only in STOP mode") which we still enforce client-side. In
# pure-monitor mode (no brain modules installed) the import fails and the
# gate becomes a no-op — safe because the only path that hits it is the
# operator clicking the map, which requires the brain to be alive anyway.
try:
    from src.statemachine.systemMode import SystemMode
    _HAS_STATE_MACHINE = True
except Exception:  # pragma: no cover — monitor mode w/o brain modules
    SystemMode = None  # type: ignore[assignment]
    _HAS_STATE_MACHINE = False


class ZmqBusClient(QObject):
    """Replacement for :class:`SocketIOClient` over the ZMQ pub/sub bus."""

    # ------------------------------------------------------------------
    # Signal surface (identical names to SocketIOClient so widgets that
    # connect to ``client.foo_signal`` keep working)
    # ------------------------------------------------------------------
    connection_status_changed = pyqtSignal(str)  # "connected"/"disconnected"/"error"

    enable_button_signal = pyqtSignal(bool)
    response_signal = pyqtSignal(object)
    session_access_signal = pyqtSignal(bool)
    serial_state_signal = pyqtSignal(bool)
    heartbeat_signal = pyqtSignal(object)
    heartbeat_disconnect_signal = pyqtSignal(object)
    memory_signal = pyqtSignal(float)
    cpu_signal = pyqtSignal(object)
    battery_signal = pyqtSignal(float)
    instant_consumption_signal = pyqtSignal(float)
    current_speed_signal = pyqtSignal(int)
    current_steer_signal = pyqtSignal(int)
    current_distance_signal = pyqtSignal(float)
    actuator_status_signal = pyqtSignal(object)
    imu_data_signal = pyqtSignal(object)
    steering_limits_signal = pyqtSignal(object)
    state_change_signal = pyqtSignal(str)
    warning_signal_signal = pyqtSignal(str)
    recording_signal = pyqtSignal(bool)
    location_signal = pyqtSignal(object)
    cars_signal = pyqtSignal(object)
    semaphores_signal = pyqtSignal(object)
    navigation_status_signal = pyqtSignal(object)
    behavior_output_signal = pyqtSignal(object)
    lidar_scan_signal = pyqtSignal(object)
    lidar_status_signal = pyqtSignal(object)
    detected_objects_signal = pyqtSignal(object)
    tracked_objects_signal = pyqtSignal(object)
    motor_command_signal = pyqtSignal(object)
    gps_fix_signal = pyqtSignal(object)
    # ``camera_frame_signal`` only exists for widgets that connect to it on
    # the (legacy) telemetry client; the actual frame source is
    # :class:`UdpVideoClient`.
    camera_frame_signal = pyqtSignal(QImage)
    line_following_debug_signal = pyqtSignal(QImage)
    line_following_status_signal = pyqtSignal(object)
    calibration_signal = pyqtSignal(object)
    calib_run_done_signal = pyqtSignal(object)
    calib_pwm_data_signal = pyqtSignal(object)
    console_log_signal = pyqtSignal(str)
    load_back_signal = pyqtSignal(object)

    # ------------------------------------------------------------------
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5005,                # legacy SocketIO port (ignored)
        sub_port: int = 5557,
        pub_port: int = 5556,
        reconnect: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._sub_port = int(os.environ.get("URT_BUS_GUI_SUB_PORT", str(sub_port)))
        self._pub_port = int(os.environ.get("URT_BUS_GUI_PUB_PORT", str(pub_port)))

        # The shim's _process_node() lazily creates a Node using the env vars
        # set here. We MUST set them before any subscriber/publisher is
        # materialized — that's why ZmqBusClient does it in __init__.
        os.environ.setdefault(
            "URT_BUS_XSUB_ENDPOINT", f"tcp://{self._host}:{self._pub_port}"
        )
        os.environ.setdefault(
            "URT_BUS_XPUB_ENDPOINT", f"tcp://{self._host}:{self._sub_port}"
        )

        self._topic_by_name = LEGACY_NAME_TO_TOPIC

        self._started = False
        self._stop_event = threading.Event()
        self._poller_thread: Optional[threading.Thread] = None

        # Lazy per-name caches.
        self._senders: dict[str, messageHandlerSender] = {}

        # Latched copy of the brain's mode, updated on every ``StateChange``.
        # Used by the Localisation gate in ``emit_message``. Starts ``None``
        # so the gate defaults to "permit" until the brain advertises a
        # mode — avoids dropping the operator's first map click during the
        # startup window.
        self._current_mode = None

        # Liveness probe state (set by the poll loop).
        self._last_recv_monotonic: float = 0.0
        self._last_status: str = "disconnected"

        # (subscriber, handler) tuples — built in start().
        self._subscribers: list[tuple[messageHandlerSubscriber, Callable[[Any], None]]] = []

    # ─── Public lifecycle ──────────────────────────────────────────────
    def start(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Open all subscriptions and start the poller thread."""
        if host is not None:
            self._host = host
        # port (legacy SocketIO arg) is kept for signature compat; we don't
        # use it — ZMQ ports come from URT_BUS_GUI_*_PORT.

        if self._started:
            return
        self._started = True

        self._build_subscribers()
        # Fake the connect handshake the SocketIO client used to emit so
        # widgets that wait on ``connection_status_changed`` don't sit
        # forever showing "Disconnected".
        self.connection_status_changed.emit("connected")
        # Same idea for session_access — no auth in the new bus.
        self.session_access_signal.emit(True)
        # And the EnableButton the dashboard used to push on connect.
        self.enable_button_signal.emit(True)

        self._stop_event.clear()
        self._poller_thread = threading.Thread(
            target=self._poll_loop, name="ZmqBusClient-poller", daemon=True
        )
        self._poller_thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._stop_event.set()
        thread = self._poller_thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._poller_thread = None
        for sub, _ in self._subscribers:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        self._subscribers.clear()
        self.connection_status_changed.emit("disconnected")

    def is_connected(self) -> bool:
        """True if we received real bus traffic in the last liveness window.

        Before the first message arrives we return ``self._started`` so
        widgets that gate publish calls on connect (e.g. SessionManager
        deferring its first SessionAccess) don't deadlock.
        """
        if not self._started:
            return False
        return self._last_status == "connected"

    @property
    def url(self) -> str:
        return f"tcp://{self._host}:{self._sub_port}"

    # ─── Outgoing ─────────────────────────────────────────────────────
    def emit_calibration(self, payload: dict) -> str:
        """Emite un command de calibración con ``correlation_id`` auto-generado.

        Devuelve el ``correlation_id`` (uuid4 hex) para que el caller pueda
        matchear contra la respuesta del thread. Esto convierte CALIBRATION
        (pub/sub mismo topic para request y reply) en un par request↔reply
        sin race condition entre requests concurrentes.
        """
        import uuid
        correlation_id = uuid.uuid4().hex
        full_payload = dict(payload)
        full_payload["correlation_id"] = correlation_id
        # Reusa la lógica de emit_message para todos los gates y locks.
        self.emit_message("Calibration", full_payload)
        return correlation_id

    def emit_message(self, name: str, value: Any = None) -> None:
        """Publish ``value`` on the bus topic associated with ``name``.

        ``name`` matches the legacy SocketIO ``Name`` field that
        ``processDashboard.handle_message`` used to route. Three names
        that used to be handled server-side are short-circuited here:

        * ``SessionAccess`` / ``Heartbeat`` / ``SessionEnd`` → no-op (auth
          is gone in the new bus).
        * ``GetCurrentSerialConnectionState`` → no-op (the brain publishes
          ``SerialConnectionState`` periodically; the GUI just subscribes).

        ``DrivingMode`` used to be translated to ``StateChange`` client-side
        but the race with ``_last_known_mode`` could fire the wrong action
        on startup. Now we just publish ``DrivingMode`` to the bus and the
        brain-side ``threadModeRouter`` runs ``StateMachine.request_mode``.
        """
        if name in {"SessionAccess", "Heartbeat", "SessionEnd",
                    "GetCurrentSerialConnectionState"}:
            return

        # Safety gate: manual Localisation is only valid in STOP mode.
        # The legacy dashboard enforced this; we keep the contract on the
        # client because the brain's pose estimator doesn't filter. We use
        # the LIVE mode latched from ``state_change_signal`` — race-safe
        # because Localisation clicks always follow GUI startup by seconds.
        if name == "Localisation" and self._is_manual_localisation(value):
            if _HAS_STATE_MACHINE and SystemMode is not None:
                last = self._current_mode  # latched in _on_state_change
                if last is not None and last is not SystemMode.STOP:
                    _logger.warning(
                        "Manual Localisation dropped: mode=%s (only STOP allowed)",
                        getattr(last, "name", last),
                    )
                    self.response_signal.emit({
                        "error": "Manual localisation is only allowed in STOP mode",
                    })
                    return

        sender = self._sender_for(name)
        if sender is None:
            _logger.warning("emit_message(%s): unknown topic", name)
            return
        try:
            sender.send(value)
        except Exception as exc:
            _logger.warning("emit_message(%s) failed: %s", name, exc)

    def emit_save(self, table_state: dict) -> None:
        """Persist routes/params to the GUI's own filesystem."""
        path = _routes_local_path()
        try:
            path.write_text(json.dumps(table_state, indent=2), encoding="utf-8")
            self.response_signal.emit({"data": f"Saved to {path}"})
        except Exception as exc:
            self.response_signal.emit({"error": f"Failed to save: {exc}"})

    def emit_load(self) -> None:
        """Read routes/params back from the GUI's filesystem.

        Mimics the dashboard's behaviour of emitting ``loadBack`` once on
        success or ``response`` with an error on failure.
        """
        path = _routes_local_path()
        if not path.exists():
            self.response_signal.emit({"error": "No saved routes yet."})
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.response_signal.emit({"error": f"Failed to load: {exc}"})
            return
        self.load_back_signal.emit(payload)

    # ─── Internals ────────────────────────────────────────────────────
    @staticmethod
    def _is_manual_localisation(value: Any) -> bool:
        """A Localisation payload carrying ``world_x`` / ``world_y`` from the
        operator (the map click handler) — not a periodic GPS fix from
        threadLocSys (which sets ``meta.source = "gps_localisation"``).
        """
        if not isinstance(value, dict):
            return False
        meta = value.get("meta")
        if isinstance(meta, dict):
            source = str(meta.get("source") or "").lower()
            if source.startswith("gps_") or source.startswith("ego_pose"):
                return False
        return "world_x" in value and "world_y" in value

    def _sender_for(self, name: str) -> Optional[messageHandlerSender]:
        cached = self._senders.get(name)
        if cached is not None:
            return cached
        topic = self._topic_by_name.get(name)
        if topic is None:
            return None
        sender = messageHandlerSender({}, topic)
        self._senders[name] = sender
        return sender

    def _build_subscribers(self) -> None:
        """Open one ZMQ subscriber per topic the GUI cares about.

        Topics that the dashboard previously ignored (``mainCamera``,
        ``LocalLanePerception``, etc.) stay unsubscribed — the GUI doesn't
        consume them either.
        """
        register = self._register_subscriber  # local alias for brevity
        msgs = self._topic_by_name

        # Telemetry — scalars & state
        register(msgs.get("EnableButton"), self._on_enable_button)
        register(msgs.get("SerialConnectionState"), self._on_serial_state)
        register(msgs.get("BatteryLvl"), self._on_battery)
        register(msgs.get("InstantConsumption"), self._on_instant_consumption)
        register(msgs.get("CurrentSpeed"), self._on_current_speed)
        register(msgs.get("CurrentSteer"), self._on_current_steer)
        register(msgs.get("CurrentDistance"), self._on_current_distance)
        register(msgs.get("ActuatorCommandStatus"), self._on_actuator_status)
        register(msgs.get("ImuData"), self._on_imu_data)
        register(msgs.get("SteeringLimits"), self._on_steering_limits)
        register(msgs.get("StateChange"), self._on_state_change)
        register(msgs.get("WarningSignal"), self._on_warning_signal)
        register(msgs.get("Recording"), self._on_recording)

        # Topology
        register(msgs.get("Location"), self._on_location)
        register(msgs.get("Cars"), self._on_cars)
        register(msgs.get("Semaphores"), self._on_semaphores)
        register(msgs.get("NavigationStatus"), self._on_navigation_status)
        register(msgs.get("BehaviorOutputMsg"), self._on_behavior_output)
        register(msgs.get("LidarScanMsg"), self._on_lidar_scan)
        register(msgs.get("LidarStatusMsg"), self._on_lidar_status)
        register(msgs.get("DetectedObjectsMsg"), self._on_detected_objects)
        register(msgs.get("TrackedObjectsMsg"), self._on_tracked_objects)
        register(msgs.get("MotorCommandMsg"), self._on_motor_command)

        # Calibration responses (commands go out via emit_message)
        register(msgs.get("Calibration"), self._on_calibration)
        register(msgs.get("CalibRunDone"), self._on_calib_run_done)
        register(msgs.get("CalibPWMData"), self._on_calib_pwm_data)

        # Resource monitoring (now published from a robot-side thread)
        register(msgs.get("ResourceMonitor"), self._on_resource_monitor)

        # Line following debug — JPEG bytes, decoded here.
        register(msgs.get("LineFollowingDebug"), self._on_line_following_debug)
        register(msgs.get("LineFollowingStatus"), self._on_line_following_status)

        # GPS fix from threadLocSys (separate publisher path).
        register(msgs.get("Localisation"), self._on_localisation_for_gps)

    def _register_subscriber(self, topic: Optional[Topic], handler):
        if topic is None:
            return
        sub = messageHandlerSubscriber({}, topic, "lastonly", subscribe=True)
        # Materialize the socket so its filter starts propagating now —
        # without this the first burst of messages could be missed.
        sub.subscribe()
        self._subscribers.append((sub, handler))

    def _poll_loop(self) -> None:
        """Single zmq.Poller across every shim subscriber socket.

        Replaces processDashboard's ``eventlet.spawn_after(0.15, ...)``
        with a tight poller. We sleep 10 ms when there's no traffic so
        the CPU stays cool while still giving sub-50 ms latency on bursts.
        """
        # Build socket→handler map by materialising every subscriber's
        # underlying socket. The shim guarantees the socket exists after
        # ``sub.subscribe()``.
        poller = zmq.Poller()
        socket_to_handler: dict[Any, tuple[messageHandlerSubscriber, Callable[[Any], None]]] = {}
        for sub_obj, handler in self._subscribers:
            inner = sub_obj._sub                    # ZMQ-side Subscriber
            if inner is None:
                continue
            sock = inner.socket
            poller.register(sock, zmq.POLLIN)
            socket_to_handler[sock] = (sub_obj, handler)

        # Synthetic heartbeat keeps the SessionManager watchdog from firing
        # — the legacy dashboard emitted ``heartbeat`` over SocketIO every
        # 20 s and the watchdog used that to gate "connection lost". With
        # no SocketIO bridge there's nothing emitting that signal, so we
        # produce one here. The payload mirrors the dashboard's wire shape.
        heartbeat_period_s = 20.0
        last_heartbeat = time.monotonic()

        # Liveness probe: if no real (non-heartbeat) bus traffic arrives
        # within this window, treat the connection as lost. 5 s is enough
        # to ride out one missed poll cycle but tight enough that the
        # operator sees a "disconnected" badge before the missing
        # telemetry becomes confusing.
        liveness_timeout_s = 5.0
        # ``start()`` initialised us as "connected" optimistically; track
        # transitions so we only emit on edges (Qt UIs hate spam).
        self._last_status = "connected"
        self._last_recv_monotonic = time.monotonic()
        while not self._stop_event.is_set():
            try:
                ready = dict(poller.poll(200))
            except zmq.ZMQError:
                # Context terminated during shutdown.
                return
            for sock in ready:
                entry = socket_to_handler.get(sock)
                if entry is None:
                    continue
                sub_obj, handler = entry
                try:
                    value = sub_obj.receive()
                except Exception as exc:
                    _logger.debug("subscriber recv error: %s", exc)
                    continue
                if value is None:
                    continue
                # Real traffic — refresh liveness watermark BEFORE invoking
                # the handler so a handler exception doesn't mask the fact
                # that we did hear from the bus.
                self._last_recv_monotonic = time.monotonic()
                if self._last_status != "connected":
                    self._last_status = "connected"
                    self.connection_status_changed.emit("connected")
                try:
                    handler(value)
                except Exception as exc:
                    _logger.warning("handler error: %s", exc)

            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_period_s:
                self.heartbeat_signal.emit({"data": "synthetic"})
                last_heartbeat = now

            # Liveness check: if we haven't heard real traffic in
            # ``liveness_timeout_s`` AND we currently think we're connected,
            # flip to disconnected. The synthetic heartbeat does NOT count
            # — it's a local emit, not bus traffic.
            if (
                self._last_status == "connected"
                and now - self._last_recv_monotonic > liveness_timeout_s
            ):
                self._last_status = "disconnected"
                self.connection_status_changed.emit("disconnected")

    # ─── Per-topic handlers ────────────────────────────────────────────
    def _on_enable_button(self, value): self.enable_button_signal.emit(bool(value))
    def _on_serial_state(self, value):  self.serial_state_signal.emit(bool(value))

    def _on_battery(self, value):
        v = _safe_float(value)
        if v is not None:
            self.battery_signal.emit(v)

    def _on_instant_consumption(self, value):
        v = _safe_float(value)
        if v is not None:
            self.instant_consumption_signal.emit(v)

    def _on_current_speed(self, value):
        # Wire format from the brain: mm/s. UI: cm/s. Same scaling the
        # legacy SocketIOClient did in _on_speed.
        v = _safe_float(value)
        if v is not None:
            self.current_speed_signal.emit(int(round(v / 10.0)))

    def _on_current_steer(self, value):
        # Decideg → deg.
        v = _safe_float(value)
        if v is not None:
            self.current_steer_signal.emit(int(round(v / 10.0)))

    def _on_current_distance(self, value):
        # mm → m.
        v = _safe_float(value)
        if v is not None:
            self.current_distance_signal.emit(v / 1000.0)

    def _on_actuator_status(self, value):
        self.actuator_status_signal.emit(value)

    def _on_imu_data(self, value):
        self.imu_data_signal.emit(value)

    def _on_steering_limits(self, value):
        self.steering_limits_signal.emit(value)

    def _on_state_change(self, value):
        text = str(value) if value is not None else ""
        # Latch the live mode so the Localisation safety gate in
        # ``emit_message`` can compare against the brain's reality, not a
        # stale GUI assumption.
        if _HAS_STATE_MACHINE and SystemMode is not None:
            try:
                self._current_mode = SystemMode[text]
            except KeyError:
                pass
        self.state_change_signal.emit(text)

    def _on_warning_signal(self, value):
        self.warning_signal_signal.emit(str(value) if value is not None else "")

    def _on_recording(self, value):
        self.recording_signal.emit(bool(value))

    def _on_location(self, value):
        # Legacy dashboard demuxed Location into either "Location" (ego)
        # or "Cars" (other car) based on payload.meta.source. We preserve
        # that here for backwards compat with map widgets.
        if isinstance(value, dict):
            meta = value.get("meta")
            if isinstance(meta, dict):
                source = str(meta.get("source") or "").lower()
                if not source.startswith("ego_pose"):
                    # Treat as another car
                    car = {k: v for k, v in value.items() if k != "meta"}
                    car["id"] = car.get("id") or meta.get("source")
                    self.cars_signal.emit([car])
                    return
        self.location_signal.emit(value)

    def _on_cars(self, value):              self.cars_signal.emit(value)
    def _on_semaphores(self, value):        self.semaphores_signal.emit(value)
    def _on_navigation_status(self, value): self.navigation_status_signal.emit(value)
    def _on_behavior_output(self, value):   self.behavior_output_signal.emit(value)
    def _on_lidar_scan(self, value):        self.lidar_scan_signal.emit(value)
    def _on_lidar_status(self, value):      self.lidar_status_signal.emit(value)
    def _on_detected_objects(self, value):  self.detected_objects_signal.emit(value)
    def _on_tracked_objects(self, value):   self.tracked_objects_signal.emit(value)
    def _on_motor_command(self, value):     self.motor_command_signal.emit(value)

    def _on_calibration(self, value):
        # Skip own-publish echoes (commands have an "Action" key, responses
        # use "action"); without this filter the wizard would receive its
        # own command payloads and re-process them.
        if isinstance(value, dict) and "Action" in value and "action" not in value:
            return
        self.calibration_signal.emit(value)

    def _on_calib_run_done(self, value): self.calib_run_done_signal.emit(value)
    def _on_calib_pwm_data(self, value): self.calib_pwm_data_signal.emit(value)

    def _on_resource_monitor(self, value):
        """ResourceMonitor payload: ``{"memory": float, "cpu": {"usage": float,
        "temp": float}}``. Demux into the two signals the GUI expects.
        """
        if not isinstance(value, dict):
            return
        mem = _safe_float(value.get("memory"))
        if mem is not None:
            self.memory_signal.emit(mem)
        cpu = value.get("cpu")
        if isinstance(cpu, dict):
            self.cpu_signal.emit(cpu)

    def _on_line_following_debug(self, value):
        """JPEG bytes → QImage."""
        if not isinstance(value, (bytes, bytearray)):
            return
        img = QImage()
        if img.loadFromData(bytes(value)) and not img.isNull():
            self.line_following_debug_signal.emit(img)

    def _on_line_following_status(self, value):
        self.line_following_status_signal.emit(value)

    def _on_localisation_for_gps(self, value):
        """Surface GPS-sourced ``Localisation`` payloads on ``gps_fix_signal``.

        threadLocSys publishes Localisation with ``meta.source == "gps_localisation"``;
        every other Localisation message goes through the Location/Cars demuxer
        above. Filter on the meta tag to avoid mixing the two.
        """
        if not isinstance(value, dict):
            return
        meta = value.get("meta")
        if not isinstance(meta, dict):
            return
        if str(meta.get("source") or "").lower() == "gps_localisation":
            self.gps_fix_signal.emit(value)
