"""Tests de `src/core/persistence/calibration_db.py`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.persistence.calibration_db import CalibrationDB


def _fresh_db(tmp_path: Path) -> CalibrationDB:
    return CalibrationDB(tmp_path / "cal.db")


def test_schema_creates_all_tables(tmp_path):
    db = _fresh_db(tmp_path)
    cur = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    names = {row[0] for row in cur.fetchall()}
    assert {"camera_intrinsics", "imu_offsets", "wheel_calibration", "last_pose"} <= names
    db.close()


def test_camera_intrinsics_roundtrip(tmp_path):
    db = _fresh_db(tmp_path)
    db.set_camera_intrinsic("real", "fy", 640.0)
    db.set_camera_intrinsic("real", "cy", 240.0)
    db.set_camera_intrinsic("sim", "fy", 720.0)
    real = db.get_camera_intrinsics("real")
    assert real == {"fy": 640.0, "cy": 240.0}
    sim = db.get_camera_intrinsics("sim")
    assert sim == {"fy": 720.0}
    db.close()


def test_camera_intrinsics_upsert(tmp_path):
    db = _fresh_db(tmp_path)
    db.set_camera_intrinsic("real", "fy", 500.0)
    db.set_camera_intrinsic("real", "fy", 640.0)  # overwrite
    assert db.get_camera_intrinsics("real")["fy"] == 640.0
    db.close()


def test_imu_offset_roundtrip(tmp_path):
    db = _fresh_db(tmp_path)
    assert db.get_imu_offset("yaw") is None
    db.set_imu_offset("yaw", 0.012)
    assert db.get_imu_offset("yaw") == pytest.approx(0.012)
    db.close()


def test_wheel_param_roundtrip(tmp_path):
    db = _fresh_db(tmp_path)
    db.set_wheel_param("steer_offset_deg", -1.5)
    assert db.get_wheel_param("steer_offset_deg") == pytest.approx(-1.5)
    db.close()


def test_last_pose_roundtrip(tmp_path):
    db = _fresh_db(tmp_path)
    assert db.get_last_pose() is None
    db.set_last_pose(x=10.0, y=20.0, yaw=0.5, lanelet_id="L7")
    p = db.get_last_pose()
    assert p is not None
    assert p["x"] == 10.0 and p["y"] == 20.0
    assert p["yaw"] == pytest.approx(0.5)
    assert p["lanelet_id"] == "L7"
    db.close()


def test_last_pose_singleton(tmp_path):
    """id=1 con CHECK constraint — sólo existe una fila."""
    db = _fresh_db(tmp_path)
    db.set_last_pose(x=1.0, y=2.0, yaw=0.0)
    db.set_last_pose(x=3.0, y=4.0, yaw=1.0)  # update, no insert
    cur = db._conn.execute("SELECT COUNT(*) FROM last_pose")
    assert cur.fetchone()[0] == 1
    p = db.get_last_pose()
    assert p["x"] == 3.0 and p["y"] == 4.0
    db.close()


def test_creates_parent_directory(tmp_path):
    """Si la carpeta data/ no existe, la creamos al instanciar."""
    deep = tmp_path / "nested" / "subdir" / "x.db"
    db = CalibrationDB(deep)
    assert deep.exists()
    db.close()
