"""Pure distance-based dead reckoning for dashboard visualisation.

This thread intentionally does not feed path planning or control. It consumes
the same Nucleo-shaped feedback as the fused pose estimator, integrates
position from accumulated odometer distance deltas, and publishes a separate
green dashboard marker so drift can be inspected against GPS.
"""

from __future__ import annotations

import ast
import math
import time

from src.core.messaging.allMessages import (
    CurrentSteer,
    DeadReckoningPose,
    ImuData,
    Location,
    OdoDistance,
)
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop
from src.utils.live_log import live_log

try:
    import config as cfg

    _WHEELBASE_M = float(getattr(cfg, "TRACKING_WHEELBASE_M", 0.260) or 0.260)
    _STEER_SIGN_DR = float(getattr(cfg, "TRACKING_STEER_SIGN_DR", 1.0) or 1.0)
    _IMU_YAW_SIGN = float(getattr(cfg, "TRACKING_IMU_YAW_SIGN", -1.0) or -1.0)
except Exception:
    _WHEELBASE_M = 0.260
    _STEER_SIGN_DR = 1.0
    _IMU_YAW_SIGN = -1.0


def _wrap_pi(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def _as_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class threadDeadReckoningPure(ThreadWithStop):
    """Integrate pure DR from odometer distance deltas and IMU absolute yaw."""

    def __init__(self, queueList, logging=None, debugging: bool = False):
        super().__init__(pause=0.02)
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging

        self._pose_sender = messageHandlerSender(self.queuesList, DeadReckoningPose)
        self._odo_sub = messageHandlerSubscriber(
            self.queuesList, OdoDistance, "lastOnly", subscribe=True
        )
        self._steer_sub = messageHandlerSubscriber(
            self.queuesList, CurrentSteer, "lastOnly", subscribe=True
        )
        self._imu_sub = messageHandlerSubscriber(
            self.queuesList, ImuData, "lastOnly", subscribe=True
        )
        self._location_sub = messageHandlerSubscriber(
            self.queuesList, Location, "lastOnly", subscribe=True
        )

        self._x = 0.0
        self._y = 0.0
        self._yaw_offset_rad = 0.0
        self._last_odo_mm: int | None = None
        self._anchored = False
        self._last_steer_rad = 0.0
        self._last_imu_yaw_rad = 0.0
        self._imu_received: bool = False
        self._last_imu_t: float | None = None
        self._stale_imu_logged: bool = False
        self._last_unanchored_publish_t = 0.0

    @staticmethod
    def _parse_steer_rad(raw_value) -> float | None:
        try:
            wire_to_math_sign = -1.0
            return (
                math.radians(float(raw_value) / 10.0)
                * wire_to_math_sign
                * float(_STEER_SIGN_DR)
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_imu_yaw_rad(raw_value) -> float | None:
        try:
            imu_dict = ast.literal_eval(str(raw_value))
            yaw_deg = float(imu_dict.get("yaw", 0.0))
            return _IMU_YAW_SIGN * math.radians(yaw_deg)
        except Exception:
            return None

    @staticmethod
    def _location_xy(payload: dict) -> tuple[float, float] | None:
        x = _as_float(payload.get("x", payload.get("world_x", payload.get("posA"))))
        y = _as_float(payload.get("y", payload.get("world_y", payload.get("posB"))))
        if x is None or y is None:
            return None
        return x, y

    @staticmethod
    def _location_yaw_rad(payload: dict) -> float | None:
        yaw_rad = _as_float(payload.get("yaw_rad"))
        if yaw_rad is not None:
            return yaw_rad
        yaw_deg = _as_float(payload.get("yaw", payload.get("yaw_deg")))
        if yaw_deg is None:
            return None
        return math.radians(yaw_deg)

    def _consume_feedback(self) -> None:
        steer_raw = self._steer_sub.receive()
        if steer_raw is not None:
            steer_rad = self._parse_steer_rad(steer_raw)
            if steer_rad is not None:
                self._last_steer_rad = float(steer_rad)

        imu_raw = self._imu_sub.receive()
        if imu_raw is not None:
            imu_yaw_rad = self._parse_imu_yaw_rad(imu_raw)
            if imu_yaw_rad is not None:
                self._last_imu_yaw_rad = float(imu_yaw_rad)
                self._imu_received = True
                self._last_imu_t = time.monotonic()
                self._stale_imu_logged = False

    def _try_anchor(self) -> bool:
        if self._anchored:
            return False
        if not self._imu_received:
            return False
        imu_age = (
            time.monotonic() - self._last_imu_t
            if self._last_imu_t is not None
            else math.inf
        )
        if imu_age >= 1.0:
            return False

        payload = self._location_sub.receive()
        if not isinstance(payload, dict):
            return False
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        source = str(meta.get("source", ""))
        if source not in {"ego_pose_pose_estimator", "ego_pose_relocalization"}:
            return False

        xy = self._location_xy(payload)
        if xy is None:
            return False

        location_yaw_rad = self._location_yaw_rad(payload)
        self._x, self._y = xy
        self._yaw_offset_rad = (
            _wrap_pi(float(location_yaw_rad) - float(self._last_imu_yaw_rad))
            if location_yaw_rad is not None
            else 0.0
        )
        self._anchored = True
        # Reset the odometer reference so the next delta only counts movement
        # after the anchor. Otherwise, any pre-anchor odo samples would be
        # integrated from the anchor pose and the DR dot would appear displaced.
        self._last_odo_mm = None

        live_log(
            "dead_reckoning",
            event="dr_anchored",
            anchor_x=float(self._x),
            anchor_y=float(self._y),
            anchor_yaw_rad=(
                float(location_yaw_rad) if location_yaw_rad is not None else None
            ),
            anchor_source=source,
            imu_yaw_rad=float(self._last_imu_yaw_rad),
            yaw_offset_rad=float(self._yaw_offset_rad),
        )
        if self.debugging:
            msg = (
                "[dead_reckoning] anchored at "
                f"x={self._x:.3f} y={self._y:.3f} "
                f"source={source} "
                f"offset={math.degrees(self._yaw_offset_rad):+.1f}deg"
            )
            if self.logging:
                self.logging.info(msg)
            else:
                print(msg)
        return True

    def _publish_unanchored(self) -> None:
        now = time.time()
        if now - self._last_unanchored_publish_t < 0.5:
            return
        self._last_unanchored_publish_t = now
        self._pose_sender.send({"timestamp": now, "anchored": False})

    def _publish_pose(self, *, delta_d_m: float, odo_mm: int) -> None:
        yaw_effective = _wrap_pi(float(self._last_imu_yaw_rad) + float(self._yaw_offset_rad))
        payload = {
            "x": float(self._x),
            "y": float(self._y),
            "yaw_deg": float(math.degrees(yaw_effective)),
            "timestamp": time.time(),
            "anchored": True,
        }
        self._pose_sender.send(payload)
        live_log(
            "dead_reckoning",
            event="dr_pose",
            x=float(self._x),
            y=float(self._y),
            yaw_rad=float(yaw_effective),
            delta_d_m=float(delta_d_m),
            odo_mm=int(odo_mm),
            steer_rad=float(self._last_steer_rad),
            imu_yaw_rad=float(self._last_imu_yaw_rad),
            anchored=True,
        )

    def thread_work(self) -> None:
        self._consume_feedback()
        imu_age = (
            time.monotonic() - self._last_imu_t
            if self._last_imu_t is not None
            else math.inf
        )
        imu_fresh = imu_age < 1.0
        if not imu_fresh and self._anchored:
            if not self._stale_imu_logged:
                live_log(
                    "dead_reckoning",
                    event="dr_imu_stale",
                    imu_age_s=float(imu_age) if math.isfinite(imu_age) else None,
                )
                self._stale_imu_logged = True
            self._publish_unanchored()
            return

        just_anchored = self._try_anchor()

        odo_raw = self._odo_sub.receive()
        if odo_raw is None:
            if just_anchored and self._last_odo_mm is not None:
                self._publish_pose(delta_d_m=0.0, odo_mm=int(self._last_odo_mm))
            elif not self._anchored:
                self._publish_unanchored()
            return

        try:
            odo_mm = int(odo_raw)
        except (TypeError, ValueError):
            if not self._anchored:
                self._publish_unanchored()
            return

        if self._last_odo_mm is None:
            self._last_odo_mm = odo_mm
            if self._anchored:
                self._publish_pose(delta_d_m=0.0, odo_mm=odo_mm)
            else:
                self._publish_unanchored()
            return

        delta_d_m = (float(odo_mm) - float(self._last_odo_mm)) / 1000.0
        if delta_d_m > 0.0 and self._anchored:
            yaw_effective = _wrap_pi(
                float(self._last_imu_yaw_rad) + float(self._yaw_offset_rad)
            )
            delta_yaw_rad = -(
                float(delta_d_m) / float(_WHEELBASE_M)
            ) * math.tan(float(self._last_steer_rad))
            yaw_mid = yaw_effective + 0.5 * delta_yaw_rad
            self._x += float(delta_d_m) * math.cos(yaw_mid)
            self._y += float(delta_d_m) * math.sin(yaw_mid)
        elif delta_d_m <= 0.0:
            delta_d_m = 0.0

        self._last_odo_mm = odo_mm
        if self._anchored:
            self._publish_pose(delta_d_m=delta_d_m, odo_mm=odo_mm)
        else:
            self._publish_unanchored()
