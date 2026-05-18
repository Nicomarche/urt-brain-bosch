"""Recorder exhaustivo de runs en modo MANUAL/AUTO.

Cada vez que el vehículo entra a ``SystemMode.MANUAL`` o ``SystemMode.AUTO``
(publicación en ``STATE_CHANGE``) este proceso abre una **sesión**: crea una
subcarpeta ``manual_<YYYYMMDD_HHMMSS>/`` o ``auto_<YYYYMMDD_HHMMSS>/`` dentro
del run actual y drena todo el bus a disco hasta que se sale de ese modo o se
apaga el sistema.

Layout por sesión::

    $URT_LOG_RUN_DIR/manual_<ts>/
        manifest.json       — metadata (start/end/host/git_sha/reason)
        events.jsonl        — stream unificado {ts, mono, topic, payload}
        per_topic/
            imu_data.jsonl
            current_speed.jsonl
            ...

Los grabadores de video viven en ``processCamera`` (intra-proceso, sin
pasar por el bus) y escriben ``raw_video.avi``/``annotated_video.avi`` a
la **misma subcarpeta**. La coordinación se hace por el topic LATCHED
:data:`src.core.bus.topics.MANUAL_RECORDING_SESSION` que este recorder
publica con el ``session_dir`` absoluto.

Diseño:

* Single source of truth para el ciclo de vida: este recorder. Los
  productores de video reaccionan al topic.
* Sin throttle — cada mensaje que llegue a los subscribers del recorder se
  persiste. Los subscribers del recorder usan FIFO profundo y ``CONFLATE``
  apagado, aunque el resto del sistema consuma esos mismos topics como
  newest-wins.
* Bootstrap gratis: ``STATE_CHANGE`` es LATCHED, así que al primer
  ``receive()`` el recorder ve el modo actual aunque haya arrancado
  después de la transición.
* Cierre limpio: ``stop()`` cierra la sesión activa con
  ``reason="shutdown"`` antes de retornar. SIGKILL deja ``end_ts: null``
  en el manifest, diagnosticable.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

from src.core.bus import qos as _qos
from src.core.bus.node import Node
from src.core.bus.qos import QoSProfile
from src.core.bus.topics import (
    DRIVING_MODE,
    MANUAL_RECORDING_SESSION,
    STATE_CHANGE,
    Topic,
)
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.recording.campaign_artifacts import write_campaign_artifacts
from src.recording.topic_catalog import filename_for_topic, iter_loggable_topics
from src.templates.workerprocess import WorkerProcess
from src.utils.live_log import live_log


_MANUAL_MODE = "MANUAL"
_AUTO_MODE = "AUTO"
_RECORDED_MODES = {_MANUAL_MODE, _AUTO_MODE}
_STOP_MODE = "STOP"
_DRAIN_PAUSE_S = 0.01      # 100 Hz polling loop — fino para no acumular
_STATE_POLL_PAUSE_S = 0.05  # cuando no hay sesión activa, latencia OK
_RECORDER_SUB_DEPTH = 4096
_SESSION_REANNOUNCE_S = 0.5


_QOS_PRESETS: dict[str, QoSProfile] = {
    "STREAM": _qos.STREAM,
    "COMMAND": _qos.COMMAND,
    "LATCHED": _qos.LATCHED,
    "EVENT": _qos.EVENT,
}


def _safe_session_path(now_ts: float, mode: str = _MANUAL_MODE) -> Path:
    """Resuelve ``$URT_LOG_RUN_DIR/<mode>_<ts>``.

    Fallback a ``./temp/<mode>_<ts>`` si la env var no está seteada
    (invocación legacy de ``main.py`` sin pasar por ``run.sh``).
    """
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_ts))
    prefix = str(mode or _MANUAL_MODE).strip().lower()
    if prefix not in {"manual", "auto"}:
        prefix = "session"
    run_dir_env = os.environ.get("URT_LOG_RUN_DIR", "").strip()
    if run_dir_env:
        base = Path(run_dir_env)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        base = repo_root / "temp"
    return base / f"{prefix}_{ts}"


def _read_git_sha() -> Optional[str]:
    """SHA del HEAD para trazabilidad del binario que generó el dataset.

    Lee ``.git/HEAD`` y resuelve la ref a mano para no depender de git en
    runtime. Devuelve ``None`` si no es un repo git o falta la ref.
    """
    try:
        repo_root = Path(__file__).resolve().parents[2]
        head_file = repo_root / ".git" / "HEAD"
        if not head_file.exists():
            return None
        head = head_file.read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref_path = repo_root / ".git" / head[5:].strip()
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            return None
        return head
    except Exception:
        return None


def _recorder_qos(topic: Topic) -> QoSProfile:
    """QoS especial del recorder: FIFO profundo, nunca conflate.

    El shim legacy deriva ``STREAM`` como ``conflate=True`` aunque el caller
    pida ``deliveryMode="fifo"``. Para la decisión del usuario ("TODO sin
    throttle") el recorder abre sus propios subscribers con CONFLATE apagado
    y un RCVHWM grande, sin cambiar la semántica global del resto del stack.
    """
    base = _QOS_PRESETS.get(topic.qos_preset, _qos.EVENT)
    return replace(
        base,
        allow_pickle=True,
        conflate=False,
        depth=max(int(base.depth), _RECORDER_SUB_DEPTH),
    )


class _SessionState:
    """File handles + estado de una sesión MANUAL activa.

    Mantiene un FD por archivo JSONL. ``os.write`` con ``O_APPEND`` da
    atomicidad por línea hasta PIPE_BUF (~4 KB en Linux/macOS); por encima
    de eso truncamos como hace :func:`src.utils.live_log.live_log`.
    """

    _MAX_LINE_BYTES = 3072

    def __init__(self, session_dir: Path, topics: list[Topic], started_at: float, mode: str) -> None:
        self.session_dir = session_dir
        self.started_at = started_at
        self.session_id = session_dir.name
        self.mode = str(mode or _MANUAL_MODE).upper()
        self._closed = False

        (session_dir / "per_topic").mkdir(parents=True, exist_ok=True)

        # Master event stream (cross-topic, ordenado por llegada).
        self._events_fd = os.open(
            str(session_dir / "events.jsonl"),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )

        # Un FD por topic (per_topic/<name>.jsonl).
        self._topic_fds: Dict[str, int] = {}
        for topic in topics:
            filename = filename_for_topic(topic) + ".jsonl"
            path = session_dir / "per_topic" / filename
            self._topic_fds[topic.name] = os.open(
                str(path),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o644,
            )

        # Manifest inicial (se rescribe al cierre con end_ts y reason).
        self._manifest_path = session_dir / "manifest.json"
        manifest = {
            "session_id": self.session_id,
            "mode": self.mode,
            "session_dir": str(session_dir),
            "started_at": started_at,
            "started_at_monotonic": time.monotonic(),
            "started_at_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(started_at)
            ),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "git_sha": _read_git_sha(),
            "topics_subscribed": sorted(t.name for t in topics),
            "end_ts": None,
            "end_ts_iso": None,
            "reason": None,
            "events_count": 0,
        }
        self._manifest = manifest
        self._write_manifest()

    def write(self, topic_name: str, payload) -> None:
        """Persiste un mensaje en ``events.jsonl`` y en ``per_topic/<name>.jsonl``."""
        if self._closed:
            return
        record = {
            "ts": time.time(),
            "mono": time.monotonic(),
            "topic": topic_name,
            "payload": payload,
        }
        try:
            line = json.dumps(record, default=str, separators=(",", ":"))
        except (TypeError, ValueError):
            return
        encoded = line.encode("utf-8") + b"\n"
        if len(encoded) > self._MAX_LINE_BYTES:
            stub = {
                "ts": record["ts"],
                "mono": record["mono"],
                "topic": topic_name,
                "_truncated": True,
                "_original_size": len(encoded),
            }
            encoded = (
                json.dumps(stub, separators=(",", ":")).encode("utf-8") + b"\n"
            )
        try:
            os.write(self._events_fd, encoded)
        except OSError:
            pass
        topic_fd = self._topic_fds.get(topic_name)
        if topic_fd is not None:
            try:
                os.write(topic_fd, encoded)
            except OSError:
                pass
        self._manifest["events_count"] += 1

    def close(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        end_ts = time.time()
        self._manifest["end_ts"] = end_ts
        self._manifest["end_ts_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(end_ts)
        )
        self._manifest["reason"] = reason
        try:
            self._write_manifest()
        except Exception:
            pass
        for fd in list(self._topic_fds.values()):
            try:
                os.close(fd)
            except OSError:
                pass
        self._topic_fds.clear()
        try:
            os.close(self._events_fd)
        except OSError:
            pass

    def _write_manifest(self) -> None:
        tmp = self._manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, indent=2, default=str)
        os.replace(tmp, self._manifest_path)


class ManualSessionRecorder(WorkerProcess):
    """Proceso worker que graba el bus completo durante runs MANUAL/AUTO.

    El constructor sigue la convención ``(queueList, logger, ...)`` del
    resto de procesos del stack (ver ``main.py``). El logger es opcional;
    el recorder usa :func:`live_log` para que su lifecycle aparezca en
    ``brain.jsonl`` junto al resto del control plane.
    """

    def __init__(self, queueList, logger=None, daemon=True) -> None:
        super().__init__(queueList, ready_event=None, daemon=daemon)
        self._logger = logger or logging.getLogger("manual_session_recorder")
        # Lista cacheada de topics. Se materializa en el child process
        # (post-fork/spawn) para que la introspección use el módulo
        # importado por ese proceso.
        self._topics: list[Topic] = []
        # Estado de la sesión actual (None si no hay sesión).
        self._session: Optional[_SessionState] = None
        # Subscribers se crean lazy en el child (no se pueden picklear).
        self._state_sub = None
        self._driving_mode_sub = None
        self._state_node: Optional[Node] = None
        self._topic_subs: list = []
        self._data_node: Optional[Node] = None
        self._session_pub = None
        self._last_mode: Optional[str] = None
        self._last_session_announce_mono = 0.0

    # ------------------------------------------------------------------
    # WorkerProcess hooks
    # ------------------------------------------------------------------

    def _init_threads(self) -> None:
        # No tenemos threads internos; toda la lógica corre en
        # ``process_work`` que el supervisor de WorkerProcess llama en
        # bucle. Override vacío para no levantar NotImplementedError.
        self.threads = []

    def run(self) -> None:  # type: ignore[override]
        try:
            super().run()
        finally:
            # Este finally corre en el proceso hijo. ``stop()`` corre en el
            # padre y sólo levanta el Event compartido; no puede cerrar los FD
            # que el hijo abrió para la sesión activa.
            self._close_session(reason="shutdown")
            self._stop_topic_subscribers()
            if self._state_node is not None:
                try:
                    self._state_node.close()
                except Exception:
                    pass
                self._state_node = None

    def process_work(self) -> None:  # type: ignore[override]
        try:
            self._ensure_subscribers()
            self._handle_state_change()
            self._handle_driving_mode_command()
            self._drain_topics()
            if self._session is not None:
                self._maybe_reannounce_active_session()
                self._blocker.wait(_DRAIN_PAUSE_S)
            else:
                self._blocker.wait(_STATE_POLL_PAUSE_S)
        except Exception:
            # Nunca dejar caer el proceso por una excepción de bus/io:
            # logear y seguir. El supervisor del WorkerProcess no nos
            # cubre acá porque heredamos process_work directamente.
            self._logger.exception("manual_session_recorder loop error")
            try:
                live_log(
                    "manual_recorder",
                    event="loop_error",
                    error=traceback.format_exc(limit=2),
                )
            except Exception:
                pass

    def stop(self) -> None:  # type: ignore[override]
        # Corre en el padre: sólo señaliza al hijo. El flush real de archivos
        # está en ``run(... finally ...)`` dentro del proceso hijo.
        super().stop()

    # ------------------------------------------------------------------
    # Setup lazy (child process)
    # ------------------------------------------------------------------

    def _ensure_subscribers(self) -> None:
        if self._state_sub is not None:
            return
        # Materializar la lista de topics dentro del child para que las
        # importaciones sean coherentes con el módulo del proceso.
        self._topics = list(iter_loggable_topics())
        # STATE_CHANGE es LATCHED, pero el QoS base tiene depth=1. Usamos el
        # QoS del recorder (FIFO profundo, conflate off) para no perder una
        # sesión manual corta por compresión de cola.
        self._state_node = Node(name="manual_session_recorder_state")
        self._state_sub = self._state_node.subscriber(
            STATE_CHANGE, qos=_recorder_qos(STATE_CHANGE)
        )
        self._driving_mode_sub = self._state_node.subscriber(
            DRIVING_MODE, qos=_recorder_qos(DRIVING_MODE)
        )
        self._start_topic_subscribers()
        # Publisher para anunciar la sesión a los grabadores de video.
        self._session_pub = messageHandlerSender(
            self.queuesList, MANUAL_RECORDING_SESSION
        )

    def _start_topic_subscribers(self) -> None:
        """Abre subscribers FIFO para todos los topics de la sesión actual."""
        self._stop_topic_subscribers()
        self._data_node = Node(name="manual_session_recorder")
        self._topic_subs = [
            (topic, self._data_node.subscriber(topic, qos=_recorder_qos(topic)))
            for topic in self._topics
        ]

    def _stop_topic_subscribers(self) -> None:
        """Cierra sockets de datos; el ``STATE_CHANGE`` queda vivo."""
        node = self._data_node
        if node is not None:
            try:
                node.close()
            except Exception:
                pass
        self._data_node = None
        self._topic_subs = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _handle_state_change(self) -> None:
        if self._state_sub is None:
            return
        # Procesar todas las transiciones en orden. Si comprimimos a "latest"
        # podemos perder una sesión manual corta (AUTO→MANUAL→AUTO entre dos
        # polls) y, más importante, no crear la subcarpeta prometida.
        while True:
            msg = self._state_sub.try_recv()
            if msg is None:
                break
            self._handle_mode(str(msg).upper(), source="state_change")

    def _handle_driving_mode_command(self) -> None:
        if self._driving_mode_sub is None:
            return
        while True:
            msg = self._driving_mode_sub.try_recv()
            if msg is None:
                break
            mode = _normalize_mode_value(msg)
            if mode is None:
                continue
            self._handle_mode(mode, source="driving_mode")

    def _handle_mode(self, mode: str, *, source: str = "state_change") -> None:
        if mode == self._last_mode:
            return
        previous = self._last_mode
        self._last_mode = mode

        if mode in _RECORDED_MODES:
            if self._session is not None:
                self._drain_topics()
                self._close_session(reason="mode_switch")
            self._open_session(
                mode=mode,
                reason_for_open=(
                    ("startup" if previous is None else "mode_enter")
                    if source == "state_change"
                    else f"{source}_{mode.lower()}"
                )
            )
        elif self._session is not None:
            # Capturar la cola que ya llegó antes del StateChange de salida,
            # luego cerrar manifest y sockets.
            self._drain_topics()
            self._close_session(
                reason=("stop_mode" if mode == _STOP_MODE else "mode_exit")
            )

    def _open_session(self, mode: str, reason_for_open: str) -> None:
        now = time.time()
        session_dir = _safe_session_path(now, mode)
        # Si entran 2 sesiones en el mismo segundo (raro pero posible)
        # apilamos un sufijo numérico.
        attempt = 0
        while session_dir.exists():
            attempt += 1
            session_dir = session_dir.with_name(f"{session_dir.name}_{attempt}")
        session_dir.mkdir(parents=True, exist_ok=True)
        self._session = _SessionState(session_dir, self._topics, now, mode)
        self._announce_session(active=True)
        try:
            live_log(
                "manual_recorder",
                event="session_start",
                session_dir=str(session_dir),
                mode=str(mode).upper(),
                trigger=reason_for_open,
            )
        except Exception:
            pass
        self._logger.info(
            "%s session opened: %s (trigger=%s)", str(mode).upper(), session_dir, reason_for_open,
        )

    def _close_session(self, reason: str) -> None:
        if self._session is None:
            return
        session = self._session
        self._announce_closed_session(session, reason)
        # Drenar el propio MANUAL_RECORDING_SESSION active=False y cualquier
        # telemetría que llegó justo antes del cierre.
        self._drain_topics()
        session.close(reason=reason)
        self._session = None
        try:
            live_log(
                "manual_recorder",
                event="session_end",
                session_dir=str(session.session_dir),
                mode=session.mode,
                reason=reason,
            )
        except Exception:
            pass
        self._logger.info(
            "%s session closed: %s (reason=%s)", session.mode, session.session_dir, reason,
        )
        try:
            artifacts = write_campaign_artifacts(session.session_dir)
            live_log(
                "manual_recorder",
                event="campaign_artifacts",
                session_dir=str(session.session_dir),
                artifacts=artifacts,
            )
        except Exception:
            self._logger.exception("failed writing campaign-style artifacts")

    def _session_payload(self, session: _SessionState, active: bool, **extra) -> dict:
        payload = {
            "active": active,
            "mode": session.mode,
            "session_dir": str(session.session_dir),
            "session_id": session.session_id,
            "started_at": session.started_at,
        }
        payload.update(extra)
        return payload

    def _announce_session(self, *, active: bool) -> None:
        if self._session is None or self._session_pub is None:
            return
        try:
            self._session_pub.send(self._session_payload(self._session, active=active))
            self._last_session_announce_mono = time.monotonic()
        except Exception:
            self._logger.exception("failed publishing MANUAL_RECORDING_SESSION")

    def _announce_closed_session(self, session: _SessionState, reason: str) -> None:
        if self._session_pub is None:
            return
        try:
            self._session_pub.send(
                self._session_payload(session, active=False, reason=reason)
            )
            # Darle un beat a threadCamera/threadLocalPerception para recibir
            # active=False y cerrar los VideoWriter antes de extraer frames.
            self._blocker.wait(0.25)
        except Exception:
            self._logger.exception("failed publishing MANUAL_RECORDING_SESSION close")

    def _maybe_reannounce_active_session(self) -> None:
        if time.monotonic() - self._last_session_announce_mono < _SESSION_REANNOUNCE_S:
            return
        self._announce_session(active=True)

    # ------------------------------------------------------------------
    # Drain
    # ------------------------------------------------------------------

    def _drain_topics(self) -> None:
        for topic, sub in self._topic_subs:
            # Drenamos hasta vaciar por topic — sin throttle (decisión
            # del usuario). El loop interno no bloquea: try_recv en el
            # subscriber directo es non-blocking.
            while True:
                try:
                    msg = sub.try_recv()
                except Exception:
                    msg = None
                if msg is None:
                    break
                if self._session is not None:
                    self._session.write(topic.name, _coerce_payload(msg))


def _coerce_payload(msg):
    """Normaliza el payload a algo JSON-serializable.

    Los topics pueden traer bytes (cámara), str (IMU CSV-ish), ints/floats
    (encoders), dicts. ``json.dumps(default=str)`` cubre la mayoría; pero
    ``bytes`` se hace ``str()`` como ``b'...'`` lo cual no es ideal, así
    que lo serializamos como base64 explícito.
    """
    if isinstance(msg, (bytes, bytearray)):
        import base64
        return {"_bytes_b64": base64.b64encode(bytes(msg)).decode("ascii")}
    return msg


def _normalize_mode_value(value) -> Optional[str]:
    text = str(value).strip().lower()
    mapping = {
        "manual": "MANUAL",
        "auto": "AUTO",
        "stop": "STOP",
        "parking": "PARKING",
        "legacy": "LEGACY",
        "default": "DEFAULT",
    }
    return mapping.get(text)


__all__ = ["ManualSessionRecorder"]
