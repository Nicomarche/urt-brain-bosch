from __future__ import annotations

import json

from src.utils import gps_mode


def test_gps_mode_defaults_to_enabled_when_file_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(gps_mode, "GPS_MODE_PATH", tmp_path / "gps_mode.json")

    assert gps_mode.is_gps_disabled() is False


def test_gps_mode_persists_disabled_flag(monkeypatch, tmp_path) -> None:
    path = tmp_path / "gps_mode.json"
    monkeypatch.setattr(gps_mode, "GPS_MODE_PATH", path)

    payload = gps_mode.save_gps_disabled(True, source="test")

    assert payload == {"disabled": True, "source": "test"}
    assert gps_mode.is_gps_disabled() is True
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_gps_mode_ignores_corrupt_file(monkeypatch, tmp_path) -> None:
    path = tmp_path / "gps_mode.json"
    path.write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr(gps_mode, "GPS_MODE_PATH", path)

    assert gps_mode.is_gps_disabled() is False
