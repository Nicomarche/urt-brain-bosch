"""Tests de `src/core/persistence/sessions_db.py`."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.core.persistence.sessions_db import SessionsDB


def _fresh(tmp_path: Path) -> SessionsDB:
    return SessionsDB(tmp_path / "sessions.db")


def test_schema_created(tmp_path):
    db = _fresh(tmp_path)
    cur = db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = {row[0] for row in cur.fetchall()}
    assert "runs" in names
    db.close()


def test_insert_and_update_run(tmp_path):
    db = _fresh(tmp_path)
    db.insert_run_start("run_20260517_120000", sim_mode=True)
    info = db.get_run("run_20260517_120000")
    assert info is not None
    assert info["sim_mode"] is True
    assert info["end_ts"] is None

    time.sleep(0.05)
    db.update_run_end(
        "run_20260517_120000", total_distance_m=42.3, n_errors=2, notes="OK"
    )
    info2 = db.get_run("run_20260517_120000")
    assert info2["end_ts"] is not None
    assert info2["total_distance_m"] == 42.3
    assert info2["n_errors"] == 2
    assert info2["notes"] == "OK"
    db.close()


def test_list_recent_runs_order(tmp_path):
    db = _fresh(tmp_path)
    db.insert_run_start("a", sim_mode=False)
    time.sleep(0.01)
    db.insert_run_start("b", sim_mode=False)
    time.sleep(0.01)
    db.insert_run_start("c", sim_mode=True)
    rows = db.list_recent_runs(n=5)
    assert [r["run_id"] for r in rows] == ["c", "b", "a"]
    db.close()


def test_insert_idempotent(tmp_path):
    db = _fresh(tmp_path)
    db.insert_run_start("x", sim_mode=False)
    db.insert_run_start("x", sim_mode=False)  # INSERT OR IGNORE
    rows = db.list_recent_runs()
    assert len([r for r in rows if r["run_id"] == "x"]) == 1
    db.close()


def test_get_run_missing(tmp_path):
    db = _fresh(tmp_path)
    assert db.get_run("nonexistent") is None
    db.close()
