from __future__ import annotations

import pytest

from src.hardware.gps import threadLocSys as gps_mod


def test_locidsub_track_bottom_left_frame_maps_to_lanelet_world(monkeypatch) -> None:
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_COORD_FRAME", "track_bottom_left")
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_COORD_SCALE", 1.0)
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_ORIGIN_WORLD_X", "auto")
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_ORIGIN_WORLD_Y", "auto")
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_MAP_WIDTH_M", "auto")
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_MAP_HEIGHT_M", "auto")
    monkeypatch.setattr(
        gps_mod,
        "_TRACK_WORLD_BOUNDS_CACHE",
        {"x_min": -10.0, "x_max": 10.0, "y_min": -7.0, "y_max": 7.0},
    )

    client = gps_mod.threadLocSys.__new__(gps_mod.threadLocSys)

    fix = client._traffic_subscription_fix_from_message(
        {"type": "location", "x": 19.6, "y": 17.2, "z": 1.0}
    )

    assert fix is not None
    assert fix["x"] == pytest.approx(9.6)
    assert fix["y"] == pytest.approx(10.2)
    assert fix["gps_raw_traffic_x"] == pytest.approx(19.6)
    assert fix["gps_raw_traffic_y"] == pytest.approx(17.2)
    assert fix["gps_frame_width_m"] == pytest.approx(20.0)
    assert fix["gps_frame_height_m"] == pytest.approx(14.0)
    assert fix["gps_frame_in_bounds"] is False

    traffic_x, traffic_y = client._world_xy_to_traffic_subscription_frame(9.6, 10.2)
    assert traffic_x == pytest.approx(19.6)
    assert traffic_y == pytest.approx(17.2)


def test_locidsub_frame_size_can_rescale_traffic_coordinates(monkeypatch) -> None:
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_COORD_FRAME", "track_bottom_left")
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_COORD_SCALE", 1.0)
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_ORIGIN_WORLD_X", "auto")
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_ORIGIN_WORLD_Y", "auto")
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_MAP_WIDTH_M", 40.0)
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_MAP_HEIGHT_M", 28.0)
    monkeypatch.setattr(
        gps_mod,
        "_TRACK_WORLD_BOUNDS_CACHE",
        {"x_min": -10.0, "x_max": 10.0, "y_min": -7.0, "y_max": 7.0},
    )

    client = gps_mod.threadLocSys.__new__(gps_mod.threadLocSys)

    fix = client._traffic_subscription_fix_from_message(
        {"type": "location", "x": 39.2, "y": 34.4}
    )

    assert fix is not None
    assert fix["x"] == pytest.approx(9.6)
    assert fix["y"] == pytest.approx(10.2)
    assert fix["gps_frame_scale_x"] == pytest.approx(0.5)
    assert fix["gps_frame_scale_y"] == pytest.approx(0.5)


def test_locidsub_world_frame_keeps_coordinates(monkeypatch) -> None:
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_COORD_FRAME", "world")
    monkeypatch.setattr(gps_mod, "_TRAFFIC_COMM_LOCSYS_SUB_COORD_SCALE", 1.0)

    client = gps_mod.threadLocSys.__new__(gps_mod.threadLocSys)

    fix = client._traffic_subscription_fix_from_message(
        {"type": "location", "x": 19.6, "y": 17.2}
    )

    assert fix is not None
    assert fix["x"] == pytest.approx(19.6)
    assert fix["y"] == pytest.approx(17.2)
