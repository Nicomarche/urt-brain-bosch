#!/usr/bin/env python3
"""Remote bus logger for manual steering and constant-turn tests.

This script is meant to run on the operator laptop before switching the car
to MANUAL mode. It connects to the Jetson's ZMQ bus, records the most useful
telemetry for steering/speed validation, and writes two artifacts:

  * raw.jsonl      - every received message, one JSON object per line
  * timeseries.csv - fixed-rate sampled snapshot of the latest known state

The output is intentionally notebook-local so the operator can review the run
without SSHing back into the Jetson.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return repr(value)


def _parse_imu_payload(payload: Any) -> dict[str, float] | None:
    if isinstance(payload, dict):
        data = payload
    elif isinstance(payload, str):
        try:
            parsed = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None
        data = parsed
    else:
        return None

    out: dict[str, float] = {}
    for key in ("roll", "pitch", "yaw", "accelx", "accely", "accelz"):
        value = _safe_float(data.get(key))
        if value is not None:
            out[key] = value
    return out or None


@dataclass
class TopicWatcher:
    name: str
    topic: Any
    subscriber: Any = None
    count: int = 0

    def drain_latest(self) -> Any:
        if self.subscriber is None:
            return None
        latest = None
        while True:
            payload = self.subscriber.receive()
            if payload is None:
                break
            latest = payload
        if latest is not None:
            self.count += 1
        return latest

    def close(self) -> None:
        if self.subscriber is not None:
            try:
                self.subscriber.unsubscribe()
            except Exception:
                pass
            self.subscriber = None


@dataclass
class LatestState:
    logger_started_wall_s: float
    logger_started_monotonic_s: float
    host: str
    current_state: str | None = None
    cmd_speed_cms: float | None = None
    cmd_speed_mps: float | None = None
    cmd_steer_deg_wire: float | None = None
    cmd_blocked_reason: str | None = None
    cmd_engine_enabled: bool | None = None
    cmd_serial_connected: bool | None = None
    cmd_klem_mode: int | None = None
    motor_speed_mps: float | None = None
    motor_steer_deg_iso: float | None = None
    motor_valid: bool | None = None
    motor_source: str | None = None
    motor_reason: str | None = None
    feedback_speed_cms: float | None = None
    feedback_speed_mps: float | None = None
    feedback_steer_deg_wire: float | None = None
    distance_m: float | None = None
    distance_zero_m: float | None = None
    steering_limit_lower_deg_wire: float | None = None
    steering_limit_upper_deg_wire: float | None = None
    imu_roll_deg: float | None = None
    imu_pitch_deg: float | None = None
    imu_yaw_deg: float | None = None
    pose_x_m: float | None = None
    pose_y_m: float | None = None
    pose_yaw_deg: float | None = None
    pose_yaw_rad: float | None = None
    raw_pose_x_m: float | None = None
    raw_pose_y_m: float | None = None
    raw_pose_yaw_deg: float | None = None
    matched_pose_x_m: float | None = None
    matched_pose_y_m: float | None = None
    matched_pose_yaw_deg: float | None = None
    localization_confidence: float | None = None
    relocalization_mode: str | None = None
    relocalization_source: str | None = None
    relocalization_error_m: float | None = None
    first_payload_seen_wall_s: float | None = None
    first_payload_seen_monotonic_s: float | None = None
    states_seen: set[str] = field(default_factory=set)

    def note_payload_seen(self, wall_s: float, monotonic_s: float) -> None:
        if self.first_payload_seen_wall_s is None:
            self.first_payload_seen_wall_s = wall_s
            self.first_payload_seen_monotonic_s = monotonic_s

    def apply(self, topic_name: str, payload: Any) -> None:
        if topic_name == "state_change":
            text = str(payload) if payload is not None else ""
            self.current_state = text
            if text:
                self.states_seen.add(text)
            return

        if topic_name == "current_speed":
            raw = _safe_float(payload)
            if raw is None:
                return
            self.feedback_speed_cms = raw / 10.0
            self.feedback_speed_mps = raw / 1000.0
            return

        if topic_name == "current_steer":
            raw = _safe_float(payload)
            if raw is None:
                return
            self.feedback_steer_deg_wire = raw / 10.0
            return

        if topic_name == "current_distance":
            raw = _safe_float(payload)
            if raw is None:
                return
            self.distance_m = raw / 1000.0
            if self.distance_zero_m is None:
                self.distance_zero_m = self.distance_m
            return

        if topic_name == "actuator_status":
            if not isinstance(payload, dict):
                return
            speed_x10 = _safe_float(payload.get("speed_x10"))
            steer_x10 = _safe_float(payload.get("steer_x10"))
            self.cmd_speed_cms = None if speed_x10 is None else speed_x10 / 10.0
            self.cmd_speed_mps = None if speed_x10 is None else speed_x10 / 1000.0
            self.cmd_steer_deg_wire = None if steer_x10 is None else steer_x10 / 10.0
            self.cmd_blocked_reason = (
                str(payload.get("blocked_reason"))
                if payload.get("blocked_reason") is not None
                else None
            )
            self.cmd_engine_enabled = (
                bool(payload.get("engine_enabled"))
                if payload.get("engine_enabled") is not None
                else None
            )
            self.cmd_serial_connected = (
                bool(payload.get("serial_connected"))
                if payload.get("serial_connected") is not None
                else None
            )
            klem_mode = payload.get("klem_mode")
            self.cmd_klem_mode = None if klem_mode is None else int(klem_mode)
            return

        if topic_name == "motor_command":
            if not isinstance(payload, dict):
                return
            self.motor_speed_mps = _safe_float(payload.get("speed_mps"))
            self.motor_steer_deg_iso = _safe_float(payload.get("steering_deg"))
            self.motor_valid = (
                bool(payload.get("valid"))
                if payload.get("valid") is not None
                else None
            )
            self.motor_source = (
                str(payload.get("source"))
                if payload.get("source") is not None
                else None
            )
            self.motor_reason = (
                str(payload.get("reason"))
                if payload.get("reason") is not None
                else None
            )
            return

        if topic_name == "steering_limits":
            if not isinstance(payload, dict):
                return
            upper = _safe_float(payload.get("upperLimit"))
            lower = _safe_float(payload.get("lowerLimit"))
            self.steering_limit_upper_deg_wire = None if upper is None else upper / 10.0
            self.steering_limit_lower_deg_wire = None if lower is None else lower / 10.0
            return

        if topic_name == "imu_data":
            data = _parse_imu_payload(payload)
            if not data:
                return
            self.imu_roll_deg = data.get("roll")
            self.imu_pitch_deg = data.get("pitch")
            self.imu_yaw_deg = data.get("yaw")
            return

        if topic_name == "localisation":
            if not isinstance(payload, dict):
                return
            meta = payload.get("meta")
            if not isinstance(meta, dict):
                return
            if str(meta.get("source") or "") != "ego_pose_pose_estimator":
                return
            self.pose_x_m = _safe_float(payload.get("x"))
            self.pose_y_m = _safe_float(payload.get("y"))
            self.pose_yaw_deg = _safe_float(payload.get("yaw"))
            self.pose_yaw_rad = _safe_float(payload.get("yaw_rad"))
            self.raw_pose_x_m = _safe_float(payload.get("raw_x"))
            self.raw_pose_y_m = _safe_float(payload.get("raw_y"))
            self.raw_pose_yaw_deg = _safe_float(payload.get("raw_yaw"))
            self.matched_pose_x_m = _safe_float(payload.get("matched_x"))
            self.matched_pose_y_m = _safe_float(payload.get("matched_y"))
            self.matched_pose_yaw_deg = _safe_float(payload.get("matched_yaw"))
            self.localization_confidence = _safe_float(meta.get("localization_confidence"))
            self.relocalization_mode = (
                str(meta.get("relocalization_mode"))
                if meta.get("relocalization_mode") is not None
                else None
            )
            self.relocalization_source = (
                str(meta.get("last_relocalization_source"))
                if meta.get("last_relocalization_source") is not None
                else None
            )
            self.relocalization_error_m = _safe_float(meta.get("last_relocalization_error_m"))

    def to_timeseries_row(self, wall_s: float, monotonic_s: float) -> dict[str, Any]:
        distance_rel_m = None
        if self.distance_m is not None and self.distance_zero_m is not None:
            distance_rel_m = self.distance_m - self.distance_zero_m
        return {
            "wall_time_s": round(wall_s, 6),
            "monotonic_s": round(monotonic_s, 6),
            "logger_elapsed_s": round(monotonic_s - self.logger_started_monotonic_s, 6),
            "host": self.host,
            "state": self.current_state or "",
            "cmd_speed_cms": self.cmd_speed_cms,
            "cmd_speed_mps": self.cmd_speed_mps,
            "cmd_steer_deg_wire": self.cmd_steer_deg_wire,
            "cmd_blocked_reason": self.cmd_blocked_reason or "",
            "cmd_engine_enabled": self.cmd_engine_enabled,
            "cmd_serial_connected": self.cmd_serial_connected,
            "cmd_klem_mode": self.cmd_klem_mode,
            "motor_speed_mps": self.motor_speed_mps,
            "motor_steer_deg_iso": self.motor_steer_deg_iso,
            "motor_valid": self.motor_valid,
            "motor_source": self.motor_source or "",
            "motor_reason": self.motor_reason or "",
            "feedback_speed_cms": self.feedback_speed_cms,
            "feedback_speed_mps": self.feedback_speed_mps,
            "feedback_steer_deg_wire": self.feedback_steer_deg_wire,
            "distance_m": self.distance_m,
            "distance_rel_m": distance_rel_m,
            "steering_limit_lower_deg_wire": self.steering_limit_lower_deg_wire,
            "steering_limit_upper_deg_wire": self.steering_limit_upper_deg_wire,
            "imu_roll_deg": self.imu_roll_deg,
            "imu_pitch_deg": self.imu_pitch_deg,
            "imu_yaw_deg": self.imu_yaw_deg,
            "pose_x_m": self.pose_x_m,
            "pose_y_m": self.pose_y_m,
            "pose_yaw_deg": self.pose_yaw_deg,
            "pose_yaw_rad": self.pose_yaw_rad,
            "raw_pose_x_m": self.raw_pose_x_m,
            "raw_pose_y_m": self.raw_pose_y_m,
            "raw_pose_yaw_deg": self.raw_pose_yaw_deg,
            "matched_pose_x_m": self.matched_pose_x_m,
            "matched_pose_y_m": self.matched_pose_y_m,
            "matched_pose_yaw_deg": self.matched_pose_yaw_deg,
            "localization_confidence": self.localization_confidence,
            "relocalization_mode": self.relocalization_mode or "",
            "relocalization_source": self.relocalization_source or "",
            "relocalization_error_m": self.relocalization_error_m,
        }


def _default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "temp" / "manual_bus_logs" / f"run_{stamp}"


def _write_run_metadata(path: Path, *, args, fields: list[str]) -> None:
    payload = {
        "created_at_wall_s": time.time(),
        "host": args.host,
        "sub_port": args.sub_port,
        "sample_hz": args.sample_hz,
        "duration_s": args.duration,
        "timeseries_fields": fields,
        "steering_conventions": {
            "cmd_steer_deg_wire_positive": "right",
            "cmd_steer_deg_wire_negative": "left",
            "feedback_steer_deg_wire_positive": "right",
            "feedback_steer_deg_wire_negative": "left",
            "motor_steer_deg_iso_positive": "left",
            "motor_steer_deg_iso_negative": "right",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_summary(
    *,
    state: LatestState,
    watchers: list[TopicWatcher],
    started_wall_s: float,
    ended_wall_s: float,
    sample_rows: int,
) -> dict[str, Any]:
    return {
        "host": state.host,
        "started_wall_s": started_wall_s,
        "ended_wall_s": ended_wall_s,
        "duration_s": round(ended_wall_s - started_wall_s, 3),
        "sample_rows": sample_rows,
        "first_payload_seen_wall_s": state.first_payload_seen_wall_s,
        "states_seen": sorted(state.states_seen),
        "final_state": state.current_state,
        "final_cmd_speed_cms": state.cmd_speed_cms,
        "final_cmd_steer_deg_wire": state.cmd_steer_deg_wire,
        "final_feedback_speed_cms": state.feedback_speed_cms,
        "final_feedback_steer_deg_wire": state.feedback_steer_deg_wire,
        "final_distance_m": state.distance_m,
        "distance_delta_m": (
            None
            if state.distance_m is None or state.distance_zero_m is None
            else round(state.distance_m - state.distance_zero_m, 6)
        ),
        "topic_message_counts": {watcher.name: watcher.count for watcher in watchers},
    }


def _format_optional(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _open_watchers() -> list[TopicWatcher]:
    try:
        from src.core.bus.shim import messageHandlerSubscriber as _subscriber_cls
        from src.core.bus.topics import (
            ACTUATOR_COMMAND_STATUS,
            CURRENT_DISTANCE,
            CURRENT_SPEED,
            CURRENT_STEER,
            IMU_DATA,
            LOCALISATION,
            MOTOR_COMMAND_MSG,
            STATE_CHANGE,
            STEERING_LIMITS,
        )
    except ModuleNotFoundError as exc:
        missing = str(getattr(exc, "name", "") or exc)
        raise SystemExit(
            "Missing runtime dependency for bus logging "
            f"({missing}). Run this script with the same Python environment "
            "you use for the monitor, or install the repo requirements first."
        ) from exc

    watchers = [
        TopicWatcher("state_change", STATE_CHANGE),
        TopicWatcher("current_speed", CURRENT_SPEED),
        TopicWatcher("current_steer", CURRENT_STEER),
        TopicWatcher("current_distance", CURRENT_DISTANCE),
        TopicWatcher("actuator_status", ACTUATOR_COMMAND_STATUS),
        TopicWatcher("motor_command", MOTOR_COMMAND_MSG),
        TopicWatcher("steering_limits", STEERING_LIMITS),
        TopicWatcher("imu_data", IMU_DATA),
        TopicWatcher("localisation", LOCALISATION),
    ]
    for watcher in watchers:
        sub = _subscriber_cls({}, watcher.topic, "lastonly", subscribe=True)
        sub.subscribe()
        watcher.subscriber = sub
    return watchers


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Log remote Jetson bus telemetry locally for manual steering and turn tests."
        )
    )
    parser.add_argument("--host", required=True, help="Jetson IP or hostname")
    parser.add_argument(
        "--sub-port",
        type=int,
        default=5557,
        help="Remote XPUB port used for telemetry subscriptions (default: 5557)",
    )
    parser.add_argument(
        "--pub-port",
        type=int,
        default=5556,
        help="Remote XSUB port used by GUI commands (stored for completeness)",
    )
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=20.0,
        help="Timeseries sampling frequency in Hz (default: 20)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional auto-stop duration in seconds. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: temp/manual_bus_logs/run_<timestamp>",
    )
    parser.add_argument(
        "--heartbeat-s",
        type=float,
        default=1.0,
        help="Console heartbeat period in seconds (default: 1.0, 0 disables)",
    )
    args = parser.parse_args()

    os.environ["URT_BUS_XPUB_ENDPOINT"] = f"tcp://{args.host}:{args.sub_port}"
    os.environ["URT_BUS_XSUB_ENDPOINT"] = f"tcp://{args.host}:{args.pub_port}"

    out_dir = (args.out_dir or _default_output_dir()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    csv_path = out_dir / "timeseries.csv"
    meta_path = out_dir / "meta.json"
    summary_path = out_dir / "summary.json"

    started_wall_s = time.time()
    started_monotonic_s = time.monotonic()
    state = LatestState(
        logger_started_wall_s=started_wall_s,
        logger_started_monotonic_s=started_monotonic_s,
        host=str(args.host),
    )

    watchers = _open_watchers()

    fieldnames = list(
        state.to_timeseries_row(started_wall_s, started_monotonic_s).keys()
    )
    _write_run_metadata(meta_path, args=args, fields=fieldnames)

    stop_requested = False

    def _request_stop(_signum=None, _frame=None) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)

    sample_period_s = 1.0 / max(float(args.sample_hz), 0.1)
    next_sample_monotonic = time.monotonic()
    deadline_monotonic = (
        None
        if float(args.duration) <= 0.0
        else started_monotonic_s + float(args.duration)
    )
    sample_rows = 0
    heartbeat_period_s = max(float(args.heartbeat_s), 0.0)
    next_heartbeat_monotonic = started_monotonic_s + heartbeat_period_s

    print(f"[manual_bus_logger] host={args.host} sub_port={args.sub_port}")
    print(f"[manual_bus_logger] output_dir={out_dir}")
    print("[manual_bus_logger] start this before switching the car to MANUAL.")

    with raw_path.open("w", encoding="utf-8") as raw_fh, csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_fh:
        csv_writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
        csv_writer.writeheader()

        while not stop_requested:
            wall_s = time.time()
            monotonic_s = time.monotonic()

            if deadline_monotonic is not None and monotonic_s >= deadline_monotonic:
                break

            for watcher in watchers:
                payload = watcher.drain_latest()
                if payload is None:
                    continue
                state.note_payload_seen(wall_s, monotonic_s)
                state.apply(watcher.name, payload)
                raw_fh.write(
                    json.dumps(
                        {
                            "wall_time_s": round(wall_s, 6),
                            "monotonic_s": round(monotonic_s, 6),
                            "topic": watcher.name,
                            "payload": _to_jsonable(payload),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

            if monotonic_s >= next_sample_monotonic:
                row = state.to_timeseries_row(wall_s, monotonic_s)
                csv_writer.writerow(row)
                sample_rows += 1
                next_sample_monotonic += sample_period_s
                if next_sample_monotonic < monotonic_s:
                    next_sample_monotonic = monotonic_s + sample_period_s

            if heartbeat_period_s > 0.0 and monotonic_s >= next_heartbeat_monotonic:
                elapsed_s = monotonic_s - started_monotonic_s
                print(
                    "[manual_bus_logger] "
                    f"t={elapsed_s:6.1f}s "
                    f"cmd_steer={_format_optional(state.cmd_steer_deg_wire, digits=1)}deg "
                    f"fb_steer={_format_optional(state.feedback_steer_deg_wire, digits=1)}deg "
                    f"cmd_speed={_format_optional(state.cmd_speed_cms, digits=1)}cm/s "
                    f"fb_speed={_format_optional(state.feedback_speed_cms, digits=1)}cm/s "
                    f"KL={_format_optional(state.cmd_klem_mode, digits=0)} "
                    f"eng={_format_optional(state.cmd_engine_enabled)} "
                    f"counts(act={next((w.count for w in watchers if w.name == 'actuator_status'), 0)}, "
                    f"motor={next((w.count for w in watchers if w.name == 'motor_command'), 0)}, "
                    f"fbsteer={next((w.count for w in watchers if w.name == 'current_steer'), 0)}, "
                    f"imu={next((w.count for w in watchers if w.name == 'imu_data'), 0)})"
                )
                next_heartbeat_monotonic += heartbeat_period_s
                if next_heartbeat_monotonic < monotonic_s:
                    next_heartbeat_monotonic = monotonic_s + heartbeat_period_s

            raw_fh.flush()
            csv_fh.flush()
            time.sleep(0.01)

    ended_wall_s = time.time()
    summary = _build_summary(
        state=state,
        watchers=watchers,
        started_wall_s=started_wall_s,
        ended_wall_s=ended_wall_s,
        sample_rows=sample_rows,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    for watcher in watchers:
        watcher.close()

    print(f"[manual_bus_logger] wrote raw log: {raw_path}")
    print(f"[manual_bus_logger] wrote timeseries: {csv_path}")
    print(f"[manual_bus_logger] wrote summary: {summary_path}")
    counts = summary["topic_message_counts"]
    print(
        "[manual_bus_logger] final counts "
        f"act={counts.get('actuator_status', 0)} "
        f"motor={counts.get('motor_command', 0)} "
        f"fbsteer={counts.get('current_steer', 0)} "
        f"fbspeed={counts.get('current_speed', 0)} "
        f"imu={counts.get('imu_data', 0)} "
        f"pose={counts.get('localisation', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
