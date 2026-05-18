"""Tests del :class:`ManualSessionRecorder`.

Probamos las piezas que no requieren un proceso real: introspección de
topics, ciclo de vida de archivos (_SessionState), serialización de
payloads y transiciones de modo. El loop completo del WorkerProcess
no se ejercita (requiere ZMQ + un broker), pero sí se valida que la
lógica interna escribe los archivos esperados.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.bus import topics as bus_topics
from src.recording.campaign_artifacts import write_campaign_artifacts
from src.recording.manual_session_recorder import (
    ManualSessionRecorder,
    _SessionState,
    _coerce_payload,
    _normalize_mode_value,
    _recorder_qos,
    _safe_session_path,
)
from src.recording.topic_catalog import filename_for_topic, iter_loggable_topics


# ─── topic_catalog ─────────────────────────────────────────────────────────

def test_topic_catalog_includes_imu_and_state_change():
    names = {t.name for t in iter_loggable_topics()}
    assert bus_topics.IMU_DATA.name in names
    assert bus_topics.STATE_CHANGE.name in names
    assert bus_topics.LOCAL_LANE_PERCEPTION.name in names


def test_topic_catalog_includes_video_toggles_and_recorder_lifecycle():
    names = {t.name for t in iter_loggable_topics()}
    assert bus_topics.MAIN_CAMERA.name in names
    assert bus_topics.SERIAL_CAMERA.name in names
    assert bus_topics.TOGGLE_IMU_DATA.name in names
    assert bus_topics.TOGGLE_BATTERY_LVL.name in names
    assert bus_topics.VIDEO_SUBSCRIBE.name in names
    assert bus_topics.MANUAL_RECORDING_SESSION.name in names


def test_topic_catalog_matches_unique_topic_constants():
    expected = {
        obj.name
        for obj in vars(bus_topics).values()
        if isinstance(obj, bus_topics.Topic)
    }
    actual = {t.name for t in iter_loggable_topics()}
    assert actual == expected


def test_filename_for_topic_strips_prefix():
    assert filename_for_topic(bus_topics.IMU_DATA) == "imudata"
    assert filename_for_topic(bus_topics.STATE_CHANGE) == "statechange"


# ─── _safe_session_path ────────────────────────────────────────────────────

def test_session_path_uses_env_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("URT_LOG_RUN_DIR", str(tmp_path))
    out = _safe_session_path(0.0)
    assert str(out).startswith(str(tmp_path))
    assert out.name.startswith("manual_")


def test_session_path_uses_mode_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("URT_LOG_RUN_DIR", str(tmp_path))
    out = _safe_session_path(0.0, "AUTO")
    assert str(out).startswith(str(tmp_path))
    assert out.name.startswith("auto_")


def test_session_path_falls_back_to_temp(monkeypatch):
    monkeypatch.delenv("URT_LOG_RUN_DIR", raising=False)
    out = _safe_session_path(0.0)
    # Fallback: <repo>/temp/manual_<ts>
    assert "temp" in out.parts
    assert out.name.startswith("manual_")


# ─── _coerce_payload ───────────────────────────────────────────────────────

def test_coerce_payload_passes_through_dict():
    assert _coerce_payload({"x": 1}) == {"x": 1}


def test_coerce_payload_encodes_bytes_as_base64():
    out = _coerce_payload(b"abc")
    assert "_bytes_b64" in out
    import base64
    assert base64.b64decode(out["_bytes_b64"]) == b"abc"


def test_normalize_mode_value_accepts_dashboard_strings():
    assert _normalize_mode_value("manual") == "MANUAL"
    assert _normalize_mode_value("stop") == "STOP"
    assert _normalize_mode_value("AUTO") == "AUTO"
    assert _normalize_mode_value("unknown") is None


def test_thread_camera_record_parser_handles_start_stop_strings():
    from src.hardware.camera.threads.threadCamera import threadCamera

    camera = threadCamera.__new__(threadCamera)
    parser = threadCamera._record_command_enabled.__get__(camera, threadCamera)

    assert parser("start") is True
    assert parser(True) is True
    assert parser("stop") is False
    assert parser(False) is False


def test_recorder_qos_disables_stream_conflate():
    qos = _recorder_qos(bus_topics.IMU_DATA)
    assert qos.conflate is False
    assert qos.depth >= 4096
    assert qos.allow_pickle is True


# ─── _SessionState ─────────────────────────────────────────────────────────

@pytest.fixture
def session_state(tmp_path: Path) -> _SessionState:
    session_dir = tmp_path / "manual_20260518_000000"
    session_dir.mkdir()
    topics = [bus_topics.IMU_DATA, bus_topics.LOCAL_LANE_PERCEPTION]
    state = _SessionState(session_dir, topics, started_at=1000.0, mode="MANUAL")
    yield state
    state.close(reason="test")


def test_session_state_creates_layout(session_state: _SessionState):
    base = session_state.session_dir
    assert (base / "per_topic").is_dir()
    assert (base / "events.jsonl").exists()
    assert (base / "manifest.json").exists()
    assert (base / "per_topic" / "imudata.jsonl").exists()
    assert (base / "per_topic" / "locallaneperception.jsonl").exists()


def test_session_state_writes_to_both_streams(tmp_path: Path):
    session_dir = tmp_path / "manual_test"
    session_dir.mkdir()
    state = _SessionState(session_dir, [bus_topics.IMU_DATA], started_at=1.0, mode="MANUAL")
    state.write(bus_topics.IMU_DATA.name, {"yaw": 0.1})
    state.close(reason="test")

    events = (session_dir / "events.jsonl").read_text().strip().splitlines()
    per_topic = (session_dir / "per_topic" / "imudata.jsonl").read_text().strip().splitlines()
    assert len(events) == 1
    assert len(per_topic) == 1
    parsed = json.loads(events[0])
    assert parsed["topic"] == bus_topics.IMU_DATA.name
    assert parsed["payload"] == {"yaw": 0.1}


def test_session_state_closes_manifest_with_reason(tmp_path: Path):
    session_dir = tmp_path / "manual_test"
    session_dir.mkdir()
    state = _SessionState(session_dir, [bus_topics.IMU_DATA], started_at=10.0, mode="AUTO")
    state.write(bus_topics.IMU_DATA.name, {"a": 1})
    state.write(bus_topics.IMU_DATA.name, {"a": 2})
    state.close(reason="mode_exit")
    manifest = json.loads((session_dir / "manifest.json").read_text())
    assert manifest["reason"] == "mode_exit"
    assert manifest["end_ts"] is not None
    assert manifest["events_count"] == 2
    assert manifest["session_dir"] == str(session_dir)
    assert manifest["mode"] == "AUTO"


def test_session_state_silently_ignores_unserializable(tmp_path: Path):
    session_dir = tmp_path / "manual_test"
    session_dir.mkdir()
    state = _SessionState(session_dir, [bus_topics.IMU_DATA], started_at=1.0, mode="MANUAL")

    class Foo:
        pass

    # No-serializable: la línea se descarta; no debe romper.
    state.write(bus_topics.IMU_DATA.name, {"obj": Foo()})
    state.close(reason="test")
    # default=str cubre cualquier objeto, así que en realidad el record sí
    # se escribe — sólo nos importa que no levanta excepción.


def test_campaign_artifacts_create_camera_frames_and_map_overlays(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    run_dir = tmp_path / "run_test"
    session_dir = run_dir / "manual_20260518_000000"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps({"started_at": 1000.0}),
        encoding="utf-8",
    )
    brain_events = [
        {
            "ts": 1000.0,
            "thread": "pose_estimator",
            "event": "pose_published",
            "fused_x": 1.0,
            "fused_y": 2.0,
        },
        {
            "ts": 1000.1,
            "thread": "lane_observer",
            "event": "lane_obs",
            "quality": 1.0,
            "line_center_offset_m": 0.12,
        },
        {
            "ts": 1000.2,
            "thread": "pose_estimator",
            "event": "pose_published",
            "fused_x": 1.2,
            "fused_y": 2.2,
        },
    ]
    (run_dir / "brain.jsonl").write_text(
        "\n".join(json.dumps(item) for item in brain_events) + "\n",
        encoding="utf-8",
    )

    video_path = session_dir / "camera.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        5.0,
        (32, 24),
    )
    assert writer.isOpened()
    for value in (0, 120, 240):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()

    artifacts = write_campaign_artifacts(session_dir, run_dir=run_dir)

    assert (session_dir / "frames" / "index.json").exists()
    assert (session_dir / "frames" / "frame_000001.png").exists()
    assert (session_dir / "overlay.png").exists()
    assert (session_dir / "overlay_visual_lane.png").exists()
    assert (run_dir / "camera.avi").exists()
    assert (run_dir / "frames").exists()
    assert "camera_video" in artifacts


# ─── Construction smoke-test ───────────────────────────────────────────────

def test_recorder_init_does_not_create_subscribers():
    """El recorder difiere la creación de subs hasta el child process.

    En el parent (donde corre este test) los Subscribers son ``None``
    hasta que ``process_work`` corre dentro del proceso fork/spawn.
    """
    rec = ManualSessionRecorder({})
    assert rec._state_sub is None
    assert rec._topic_subs == []
    assert rec._session is None


def test_recorder_can_be_constructed_with_logger():
    import logging
    log = logging.getLogger("test")
    rec = ManualSessionRecorder({}, logger=log)
    assert rec._logger is log


def test_drain_discards_when_no_session():
    class Sub:
        def __init__(self):
            self.items = ["first", "second"]

        def try_recv(self):
            if not self.items:
                return None
            return self.items.pop(0)

    rec = ManualSessionRecorder({})
    sub = Sub()
    rec._topic_subs = [(bus_topics.IMU_DATA, sub)]
    rec._session = None

    rec._drain_topics()

    assert sub.items == []


def test_drain_writes_when_session_is_active():
    class Sub:
        def __init__(self):
            self.items = [{"yaw": 1.0}]

        def try_recv(self):
            if not self.items:
                return None
            return self.items.pop(0)

    class Session:
        def __init__(self):
            self.records = []

        def write(self, topic_name, payload):
            self.records.append((topic_name, payload))

    rec = ManualSessionRecorder({})
    session = Session()
    rec._topic_subs = [(bus_topics.IMU_DATA, Sub())]
    rec._session = session

    rec._drain_topics()

    assert session.records == [(bus_topics.IMU_DATA.name, {"yaw": 1.0})]


def test_recorder_mode_handler_preserves_short_manual_session(monkeypatch):
    rec = ManualSessionRecorder({})
    calls = []

    class Session:
        def __init__(self, mode):
            self.mode = mode

    def open_session(mode, reason_for_open):
        calls.append(("open", mode, reason_for_open))
        rec._session = Session(mode)

    def close_session(reason):
        calls.append(("close", reason))
        rec._session = None

    monkeypatch.setattr(rec, "_open_session", open_session)
    monkeypatch.setattr(rec, "_close_session", close_session)
    monkeypatch.setattr(rec, "_drain_topics", lambda: calls.append(("drain", None)))

    rec._handle_mode("AUTO")
    rec._handle_mode("MANUAL")
    rec._handle_mode("AUTO")

    assert calls == [
        ("open", "AUTO", "startup"),
        ("drain", None),
        ("close", "mode_switch"),
        ("open", "MANUAL", "mode_enter"),
        ("drain", None),
        ("close", "mode_switch"),
        ("open", "AUTO", "mode_enter"),
    ]


def test_recorder_mode_handler_closes_manual_session_on_stop(monkeypatch):
    rec = ManualSessionRecorder({})
    calls = []

    monkeypatch.setattr(rec, "_drain_topics", lambda: calls.append(("drain", None)))
    monkeypatch.setattr(
        rec,
        "_close_session",
        lambda reason: calls.append(("close", reason)),
    )

    rec._last_mode = "MANUAL"
    rec._session = object()
    rec._handle_mode("STOP")

    assert calls == [("drain", None), ("close", "stop_mode")]


def test_driving_mode_manual_command_opens_session(monkeypatch):
    rec = ManualSessionRecorder({})
    calls = []

    def open_session(mode, reason_for_open):
        calls.append(("open", mode, reason_for_open))
        rec._session = object()

    monkeypatch.setattr(rec, "_open_session", open_session)

    rec._handle_mode("MANUAL", source="driving_mode")

    assert calls == [("open", "MANUAL", "driving_mode_manual")]


def test_driving_mode_auto_command_opens_session(monkeypatch):
    rec = ManualSessionRecorder({})
    calls = []

    def open_session(mode, reason_for_open):
        calls.append(("open", mode, reason_for_open))
        rec._session = object()

    monkeypatch.setattr(rec, "_open_session", open_session)

    rec._handle_mode("AUTO", source="driving_mode")

    assert calls == [("open", "AUTO", "driving_mode_auto")]


def test_recorder_records_real_bus_messages_and_closes_on_stop(
    broker, monkeypatch, tmp_path
):
    monkeypatch.setenv("URT_LOG_RUN_DIR", str(tmp_path))

    rec = ManualSessionRecorder({})
    rec._ensure_subscribers()

    mode_pub = messageHandlerSender({}, bus_topics.DRIVING_MODE)
    imu_pub = messageHandlerSender({}, bus_topics.IMU_DATA)
    toggle_pub = messageHandlerSender({}, bus_topics.TOGGLE_IMU_DATA)
    mode_pub._ensure_pub()
    imu_pub._ensure_pub()
    toggle_pub._ensure_pub()

    # Dejar que las subscriptions lleguen al broker antes de publicar.
    time.sleep(0.2)
    mode_pub.send("manual")
    for _ in range(20):
        rec.process_work()
        if rec._session is not None:
            break
        time.sleep(0.01)

    assert rec._session is not None
    session_dir = rec._session.session_dir

    imu_pub.send("yaw=1.0,pitch=0.0")
    toggle_pub.send("on")
    for _ in range(20):
        rec.process_work()
        time.sleep(0.005)

    mode_pub.send("stop")
    for _ in range(20):
        rec.process_work()
        if rec._session is None:
            break
        time.sleep(0.01)

    rec._stop_topic_subscribers()
    if rec._state_node is not None:
        rec._state_node.close()
        rec._state_node = None

    assert rec._session is None
    manifest = json.loads((session_dir / "manifest.json").read_text())
    assert manifest["reason"] == "stop_mode"
    assert (session_dir / "per_topic" / "imudata.jsonl").read_text().strip()
    assert (session_dir / "per_topic" / "toggleimudata.jsonl").read_text().strip()
