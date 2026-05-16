from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import config
from src.core.messaging.allMessages import LidarScanMsg, LidarStatusMsg
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.types.lidar import LidarHealth, LidarObstacle, LidarPoint, LidarScan
from src.core.types.pose import PoseEstimate
from src.perception.lidar.processing import (
    RollingLidarBuffer,
    cluster_obstacles,
    raw_points_to_body_points,
    scan_from_points,
    signed_angle_rad,
    transform_obstacles_to_world,
)
from src.perception.lidar.reader import LidarReader
from src.templates.threadwithstop import ThreadWithStop
from src.utils.live_log import live_log

if TYPE_CHECKING:
    from src.core.messaging.buffers import LatestValueBuffer


class threadLidar(ThreadWithStop):
    """LiDAR runtime thread. Critical data goes through LatestValueBuffer."""

    def __init__(
        self,
        queuesList,
        logger=None,
        debugger: bool = False,
        *,
        lidar_scan_buffer: "LatestValueBuffer | None" = None,
        lidar_obstacles_buffer: "LatestValueBuffer | None" = None,
        pose_estimate_buffer: "LatestValueBuffer | None" = None,
        pause_s: float = 0.005,
    ) -> None:
        super().__init__(pause=float(pause_s))
        self.queuesList = queuesList
        self.logging = logger
        self.debugger = bool(debugger)
        self._scan_buf = lidar_scan_buffer
        self._obs_buf = lidar_obstacles_buffer
        self._pose_buf = pose_estimate_buffer
        self._enabled = bool(getattr(config, "LIDAR_ENABLED", True))
        self._backend = str(getattr(config, "LIDAR_BACKEND", "zmq")).lower()
        self._scan_sender = messageHandlerSender(queuesList, LidarScanMsg)
        self._status_sender = messageHandlerSender(queuesList, LidarStatusMsg)
        self._rolling = RollingLidarBuffer(
            ttl_s=float(getattr(config, "LIDAR_ROLLING_TTL_S", 0.5))
        )
        self._reader: LidarReader | None = None
        self._zmq_context = None
        self._zmq_sub = None
        self._last_scan_ts = 0.0
        self._last_status_t = 0.0
        self._last_scan_count = 0
        self._scan_hz = 0.0
        self._hz_window_t = time.monotonic()
        self._crc_errors = 0
        self._resyncs = 0
        self._last_detail = ""

    def thread_work(self) -> None:
        if not self._enabled:
            self._publish_status("disabled", "LIDAR_ENABLED=False")
            time.sleep(0.1)
            return
        try:
            if self._backend == "serial":
                scan = self._read_serial_scan()
            elif self._backend == "zmq":
                scan = self._read_zmq_scan()
            else:
                self._publish_status("error", f"unknown backend {self._backend!r}")
                time.sleep(0.2)
                return
        except Exception as exc:
            self._last_detail = str(exc)
            self._publish_status("error", str(exc))
            if self._backend == "serial" and self._reader is not None:
                try:
                    self._reader.close()
                except Exception:
                    pass
                self._reader = None
            if self.logging is not None:
                self.logging.exception("threadLidar failed")
            time.sleep(1.0 if self._backend == "serial" else 0.1)
            return

        if scan is None:
            self._publish_status("waiting_for_scan", self._last_detail)
            return
        self._publish_scan(scan)

    def stop(self) -> None:
        super().stop()
        if self._reader is not None:
            self._reader.close()
        if self._zmq_sub is not None:
            try:
                self._zmq_sub.close(linger=0)
            except Exception:
                pass

    def _read_serial_scan(self) -> LidarScan | None:
        if self._reader is None:
            self._reader = LidarReader(
                port=getattr(config, "LIDAR_SERIAL_PORT", None),
                baudrate=int(getattr(config, "LIDAR_SERIAL_BAUD", 230400)),
                timeout_s=float(getattr(config, "LIDAR_SERIAL_TIMEOUT_S", 0.5)),
            )
            self._reader.open()
            self._last_detail = f"serial port={self._reader.port}"
        packet = self._reader.read_packet()
        self._crc_errors = self._reader.crc_errors
        self._resyncs = self._reader.resyncs
        if packet is None:
            return None
        points = raw_points_to_body_points(
            packet.points,
            yaw_offset_deg=float(getattr(config, "LIDAR_YAW_OFFSET_DEG", 0.0)),
            clockwise=bool(getattr(config, "LIDAR_CLOCKWISE", False)),
        )
        now = time.time()
        self._rolling.add_points(points, timestamp=now)
        return self._rolling.to_scan(
            timestamp=now,
            frame_id="lidar",
            bins=int(getattr(config, "LIDAR_BINS", 720)),
            range_min_m=float(getattr(config, "LIDAR_RANGE_MIN_M", 0.05)),
            range_max_m=float(getattr(config, "LIDAR_RANGE_MAX_M", 4.0)),
            source="serial",
        )

    def _read_zmq_scan(self) -> LidarScan | None:
        if self._zmq_sub is None:
            try:
                import zmq
            except ImportError as exc:
                raise RuntimeError("pyzmq is required for ZMQ LiDAR backend") from exc
            self._zmq_context = zmq.Context.instance()
            self._zmq_sub = self._zmq_context.socket(zmq.SUB)
            self._zmq_sub.connect(str(getattr(config, "ZMQ_LIDAR_ENDPOINT", "tcp://localhost:5578")))
            topic = getattr(config, "ZMQ_LIDAR_TOPIC", b"lidar")
            if isinstance(topic, str):
                topic = topic.encode("utf-8")
            self._zmq_sub.setsockopt(zmq.SUBSCRIBE, bytes(topic))
            self._zmq_sub.setsockopt(zmq.RCVTIMEO, 1)
            self._last_detail = f"zmq endpoint={getattr(config, 'ZMQ_LIDAR_ENDPOINT', '')}"

        try:
            topic, payload = self._zmq_sub.recv_multipart(flags=0)
        except Exception:
            return None
        del topic
        data = json.loads(payload.decode("utf-8"))
        return _scan_from_zmq_payload(data)

    def _publish_scan(self, scan: LidarScan) -> None:
        pose = self._pose_buf.read_latest() if self._pose_buf is not None else None
        obstacles = cluster_obstacles(scan)
        if isinstance(pose, PoseEstimate):
            obstacles = transform_obstacles_to_world(obstacles, pose)

        if self._scan_buf is not None:
            self._scan_buf.write(scan, timestamp=scan.timestamp)
        if self._obs_buf is not None:
            self._obs_buf.write(tuple(obstacles), timestamp=scan.timestamp)

        now_mono = time.monotonic()
        self._last_scan_count += 1
        dt = now_mono - self._hz_window_t
        if dt >= 1.0:
            self._scan_hz = self._last_scan_count / dt
            self._last_scan_count = 0
            self._hz_window_t = now_mono

        self._last_scan_ts = float(scan.timestamp)
        try:
            self._scan_sender.send(_serialize_scan(scan, obstacles))
        except Exception:
            if self.logging is not None:
                self.logging.exception("failed to publish LidarScanMsg")
        self._publish_status("ok", "scan fresh", scan=scan, obstacles=obstacles)

        live_log(
            "lidar",
            event="scan",
            backend=self._backend,
            source=scan.source,
            points=len(scan.ranges_m),
            obstacles=len(obstacles),
            scan_age_s=max(0.0, time.time() - float(scan.timestamp)),
        )

    def _publish_status(
        self,
        status: str,
        detail: str = "",
        *,
        scan: LidarScan | None = None,
        obstacles: tuple[LidarObstacle, ...] = (),
    ) -> None:
        now = time.time()
        if status == "waiting_for_scan" and (now - self._last_status_t) < 0.5:
            return
        if status == "ok" and (now - self._last_status_t) < 0.5:
            return
        self._last_status_t = now
        age_s = (now - self._last_scan_ts) if self._last_scan_ts > 0.0 else None
        health = LidarHealth(
            timestamp=now,
            enabled=self._enabled,
            backend=self._backend,
            status=str(status),
            detail=str(detail or ""),
            scan_hz=float(self._scan_hz),
            points=len(scan.ranges_m) if scan is not None else 0,
            obstacles=len(obstacles),
            age_s=age_s,
            crc_errors=int(self._crc_errors),
            resyncs=int(self._resyncs),
            source=scan.source if scan is not None else self._backend,
        )
        try:
            self._status_sender.send(asdict(health))
        except Exception:
            if self.logging is not None:
                self.logging.exception("failed to publish LidarStatusMsg")


def _scan_from_zmq_payload(data: dict[str, Any]) -> LidarScan:
    now = time.time()
    raw_ts = float(data.get("timestamp", 0.0) or 0.0)
    # Gazebo often sends sim time (small seconds since world start). Watchdogs
    # compare with wall-clock, so normalize small/implausible stamps here.
    timestamp = raw_ts
    if timestamp < 1_000_000_000.0 or abs(now - timestamp) > 60.0:
        timestamp = now

    ranges = tuple(_coerce_range(v) for v in data.get("ranges", ()))
    intensities = tuple(_coerce_float(v, 0.0) for v in data.get("intensities", ()))
    if len(intensities) != len(ranges):
        intensities = tuple(20.0 for _ in ranges)
    elif intensities and all(float(v) == 0.0 for v in intensities):
        intensities = tuple(20.0 for _ in intensities)

    angle_min = float(data.get("angle_min", -math.pi))
    angle_max = float(data.get("angle_max", math.pi))
    angle_step = float(data.get("angle_step", 0.0) or 0.0)
    if angle_step == 0.0 and len(ranges) > 1:
        angle_step = (angle_max - angle_min) / float(len(ranges) - 1)

    points = []
    for idx, dist in enumerate(ranges):
        if not math.isfinite(dist):
            continue
        intensity = intensities[idx] if idx < len(intensities) else 20.0
        points.append(
            LidarPoint(
                angle_rad=angle_min + idx * angle_step,
                distance_m=dist,
                intensity=float(intensity),
            )
        )
    body_points = raw_points_to_body_points(
        points,
        yaw_offset_deg=float(getattr(config, "LIDAR_YAW_OFFSET_DEG", 0.0)),
        clockwise=bool(getattr(config, "LIDAR_CLOCKWISE", False)),
    )
    return scan_from_points(
        body_points,
        timestamp=timestamp,
        frame_id=str(data.get("frame_id", "lidar") or "lidar"),
        bins=int(getattr(config, "LIDAR_BINS", len(ranges) or 720)),
        range_min_m=float(data.get("range_min", getattr(config, "LIDAR_RANGE_MIN_M", 0.05))),
        range_max_m=float(data.get("range_max", getattr(config, "LIDAR_RANGE_MAX_M", 4.0))),
        source="zmq",
    )


def _serialize_scan(
    scan: LidarScan,
    obstacles: tuple[LidarObstacle, ...],
) -> dict[str, Any]:
    return {
        "timestamp": float(scan.timestamp),
        "frame_id": str(scan.frame_id),
        "angle_min_rad": float(scan.angle_min_rad),
        "angle_max_rad": float(scan.angle_max_rad),
        "angle_step_rad": float(scan.angle_step_rad),
        "range_min_m": float(scan.range_min_m),
        "range_max_m": float(scan.range_max_m),
        "ranges_m": [_json_range(v) for v in scan.ranges_m],
        "intensities": [float(v) for v in scan.intensities],
        "source": str(scan.source),
        "obstacles": [_serialize_obstacle(obs) for obs in obstacles],
    }


def _serialize_obstacle(obs: LidarObstacle) -> dict[str, Any]:
    return {
        "timestamp": float(obs.timestamp),
        "distance_min_m": float(obs.distance_min_m),
        "angle_center_rad": float(signed_angle_rad(obs.angle_center_rad)),
        "x_m": float(obs.x_m),
        "y_m": float(obs.y_m),
        "width_rad": float(obs.width_rad),
        "point_count": int(obs.point_count),
        "confidence": float(obs.confidence),
        "world_xy": list(obs.world_xy) if obs.world_xy is not None else None,
        "source": str(obs.source),
        "class_name": str(obs.class_name),
    }


def _coerce_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _coerce_range(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.inf
    return out if math.isfinite(out) and out > 0.0 else math.inf


def _json_range(value: float) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
