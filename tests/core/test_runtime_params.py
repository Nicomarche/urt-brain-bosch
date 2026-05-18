"""Tests de `src/core/runtime_params.py` — hot-reload y dotted lookup."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.core.runtime_params import RuntimeParams


def _write_yaml(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def test_initial_load_reads_yaml(tmp_path):
    yml = tmp_path / "runtime.yaml"
    _write_yaml(
        yml,
        """
parking:
  search_speed_mps: 0.12
""",
    )
    # Romper el singleton para este test
    RuntimeParams._instance = None
    rp = RuntimeParams(yml)
    assert rp.get("parking.search_speed_mps") == 0.12


def test_dotted_lookup_missing_returns_default(tmp_path):
    yml = tmp_path / "runtime.yaml"
    _write_yaml(yml, "parking:\n  search_speed_mps: 0.1\n")
    RuntimeParams._instance = None
    rp = RuntimeParams(yml)
    assert rp.get("does.not.exist", default=42) == 42
    assert rp.get("parking.nonexistent", default="fallback") == "fallback"


def test_nested_navigation(tmp_path):
    yml = tmp_path / "runtime.yaml"
    _write_yaml(
        yml,
        """
visual_correction_profiles:
  parking:
    gain: 0.0
    max_m: 0.0
""",
    )
    RuntimeParams._instance = None
    rp = RuntimeParams(yml)
    assert rp.get("visual_correction_profiles.parking.gain") == 0.0


def test_hot_reload_picks_up_new_value(tmp_path):
    yml = tmp_path / "runtime.yaml"
    _write_yaml(yml, "x:\n  v: 1\n")
    RuntimeParams._instance = None
    rp = RuntimeParams(yml)
    rp._poll_interval_s = 0.05  # acelera el test
    rp.start()
    try:
        assert rp.get("x.v") == 1
        # Avanzamos mtime (algunos FS tienen granularidad de 1s).
        time.sleep(1.1)
        _write_yaml(yml, "x:\n  v: 99\n")
        # Esperar a que el poller detecte
        deadline = time.time() + 5.0
        while time.time() < deadline and rp.get("x.v") != 99:
            time.sleep(0.1)
        assert rp.get("x.v") == 99
    finally:
        rp.stop()


def test_listener_fires_on_reload(tmp_path):
    yml = tmp_path / "runtime.yaml"
    _write_yaml(yml, "x:\n  v: 1\n")
    RuntimeParams._instance = None
    rp = RuntimeParams(yml)
    rp._poll_interval_s = 0.05
    calls: list[int] = []
    rp.add_listener(lambda: calls.append(rp.get("x.v")))
    rp.start()
    try:
        time.sleep(1.1)
        _write_yaml(yml, "x:\n  v: 7\n")
        deadline = time.time() + 5.0
        while time.time() < deadline and not calls:
            time.sleep(0.05)
        assert calls and calls[-1] == 7
    finally:
        rp.stop()


def test_snapshot_returns_copy(tmp_path):
    yml = tmp_path / "runtime.yaml"
    _write_yaml(yml, "x: { v: 1 }\n")
    RuntimeParams._instance = None
    rp = RuntimeParams(yml)
    snap = rp.snapshot()
    snap["x"]["v"] = 99
    assert rp.get("x.v") == 1  # interno no afectado


def test_missing_file_does_not_raise(tmp_path):
    """Si el YAML no existe, get() devuelve default — no excepción."""
    RuntimeParams._instance = None
    rp = RuntimeParams(tmp_path / "no_existe.yaml")
    assert rp.get("anything", default=None) is None
