"""SQLite store para calibración persistente entre runs.

Plan E2: el repo REF guarda estos datos en `share/database.db`. Adoptamos
el mismo patrón con stdlib `sqlite3` (sin C++, sin migraciones a Alembic).

Schema:
  * ``camera_intrinsics(mode, name, value)`` — fy/cx/cy/pitch por modo
    (real/sim).
  * ``imu_offsets(axis, offset, calibrated_at)`` — bias del IMU por eje.
  * ``wheel_calibration(param, value, calibrated_at)`` — slip, radio,
    steer_offset.
  * ``last_pose(id, x, y, yaw, ts, lanelet_id)`` — pose final del run
    anterior, útil para boot-from-last-known en lugar del JSON legacy
    ``config/sim_start_pose.json``.

Single writer (brain) + lecturas concurrentes (GUI, tools/) → WAL mode.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def _default_db_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "calibration.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS camera_intrinsics (
    mode TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (mode, name)
);

CREATE TABLE IF NOT EXISTS imu_offsets (
    axis TEXT PRIMARY KEY,
    offset REAL NOT NULL,
    calibrated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS wheel_calibration (
    param TEXT PRIMARY KEY,
    value REAL NOT NULL,
    calibrated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS last_pose (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    x REAL NOT NULL,
    y REAL NOT NULL,
    yaw REAL NOT NULL,
    ts REAL NOT NULL,
    lanelet_id TEXT
);
"""


class CalibrationDB:
    """Acceso thread-safe a la DB de calibración del brain."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else _default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # check_same_thread=False permite usar la misma conexión desde
        # múltiples threads del brain, custodiada por self._lock.
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, timeout=10.0
        )
        # WAL — múltiples readers, un writer no bloquean entre sí.
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ─── Camera intrinsics ────────────────────────────────────────────
    def get_camera_intrinsics(self, mode: str) -> dict[str, float]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT name, value FROM camera_intrinsics WHERE mode=?", (mode,)
            )
            return {row[0]: float(row[1]) for row in cur.fetchall()}

    def set_camera_intrinsic(self, mode: str, name: str, value: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO camera_intrinsics(mode, name, value) VALUES(?,?,?) "
                "ON CONFLICT(mode, name) DO UPDATE SET value=excluded.value",
                (str(mode), str(name), float(value)),
            )
            self._conn.commit()

    # ─── IMU offsets ──────────────────────────────────────────────────
    def get_imu_offset(self, axis: str) -> float | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT offset FROM imu_offsets WHERE axis=?", (axis,)
            )
            row = cur.fetchone()
            return float(row[0]) if row else None

    def set_imu_offset(self, axis: str, offset: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO imu_offsets(axis, offset, calibrated_at) VALUES(?,?,?) "
                "ON CONFLICT(axis) DO UPDATE SET offset=excluded.offset, calibrated_at=excluded.calibrated_at",
                (str(axis), float(offset), float(time.time())),
            )
            self._conn.commit()

    # ─── Wheel calibration ────────────────────────────────────────────
    def get_wheel_param(self, param: str) -> float | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT value FROM wheel_calibration WHERE param=?", (param,)
            )
            row = cur.fetchone()
            return float(row[0]) if row else None

    def set_wheel_param(self, param: str, value: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO wheel_calibration(param, value, calibrated_at) VALUES(?,?,?) "
                "ON CONFLICT(param) DO UPDATE SET value=excluded.value, calibrated_at=excluded.calibrated_at",
                (str(param), float(value), float(time.time())),
            )
            self._conn.commit()

    # ─── Last pose (boot-from-last-known) ─────────────────────────────
    def get_last_pose(self) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT x, y, yaw, ts, lanelet_id FROM last_pose WHERE id=1"
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "x": float(row[0]),
                "y": float(row[1]),
                "yaw": float(row[2]),
                "ts": float(row[3]),
                "lanelet_id": row[4],
            }

    def set_last_pose(
        self,
        *,
        x: float,
        y: float,
        yaw: float,
        lanelet_id: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO last_pose(id, x, y, yaw, ts, lanelet_id) "
                "VALUES(1, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET x=excluded.x, y=excluded.y, "
                "yaw=excluded.yaw, ts=excluded.ts, lanelet_id=excluded.lanelet_id",
                (float(x), float(y), float(yaw), float(time.time()), lanelet_id),
            )
            self._conn.commit()

    # ─── Lifecycle ────────────────────────────────────────────────────
    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "CalibrationDB":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
