"""Tests de `src/perception/object_classes.py`."""

from __future__ import annotations

import pytest

from src.perception.object_classes import (
    CLASS_NAME_TO_ENUM,
    ENUM_TO_CLASS_NAME,
    LIDAR_CONFIRMATION_REQUIRED,
    SIGN_CONFIDENCE_PER_CLASS,
    TrackedObject,
    confidence_threshold_for,
    needs_lidar_confirmation,
    to_enum,
)


def test_enum_ids_match_ref_convention():
    """IDs 0-12 deben coincidir con OBJECT_DICT del REF."""
    assert int(TrackedObject.ONEWAY) == 0
    assert int(TrackedObject.HIGHWAY_ENTRANCE) == 1
    assert int(TrackedObject.STOP_SIGN) == 2
    assert int(TrackedObject.ROUNDABOUT) == 3
    assert int(TrackedObject.PARK) == 4
    assert int(TrackedObject.CROSSWALK) == 5
    assert int(TrackedObject.NO_ENTRY) == 6
    assert int(TrackedObject.HIGHWAY_EXIT) == 7
    assert int(TrackedObject.PRIORITY) == 8
    assert int(TrackedObject.TRAFFIC_LIGHT) == 9
    assert int(TrackedObject.BLOCK) == 10
    assert int(TrackedObject.PEDESTRIAN) == 11
    assert int(TrackedObject.CAR) == 12


def test_extensions_above_12():
    assert int(TrackedObject.RED_LIGHT) == 13
    assert int(TrackedObject.YELLOW_LIGHT) == 14
    assert int(TrackedObject.GREEN_LIGHT) == 15
    assert int(TrackedObject.STOPLINE_VISUAL) == 16


def test_to_enum_string_canonical():
    assert to_enum("stop") is TrackedObject.STOP_SIGN
    assert to_enum("crosswalk") is TrackedObject.CROSSWALK
    assert to_enum("pedestrian") is TrackedObject.PEDESTRIAN


def test_to_enum_string_alias_normalizes():
    """`stopsign`, `stop_sign`, `Stop-Sign` deberían mapear a STOP_SIGN."""
    assert to_enum("stopsign") is TrackedObject.STOP_SIGN
    assert to_enum("stop_sign") is TrackedObject.STOP_SIGN
    assert to_enum("Stop-Sign") is TrackedObject.STOP_SIGN


def test_to_enum_id_round_trip():
    assert to_enum(2) is TrackedObject.STOP_SIGN
    assert to_enum(11) is TrackedObject.PEDESTRIAN
    assert to_enum(99) is None  # ID inválido


def test_to_enum_passthrough_enum():
    """to_enum es idempotente sobre TrackedObject."""
    assert to_enum(TrackedObject.CAR) is TrackedObject.CAR


def test_to_enum_none_returns_none():
    assert to_enum(None) is None
    assert to_enum("") is None
    assert to_enum("unknown_garbage_class") is None


def test_enum_to_class_name_bidirectional():
    for enum_val, name in ENUM_TO_CLASS_NAME.items():
        assert CLASS_NAME_TO_ENUM[name] is enum_val


def test_confidence_threshold_critical_signs_at_least_pedestrian():
    """Plan TANDA 0.2: PEDESTRIAN subió a 0.65 al sacar LiDAR obligatorio.

    Ahora PEDESTRIAN está en el mismo nivel de estrictez que STOP_SIGN
    porque sin confirmación LiDAR, YOLO solo es quien decide si frenamos
    al auto — vale la pena ser estricto."""
    assert confidence_threshold_for(TrackedObject.STOP_SIGN) >= confidence_threshold_for(
        TrackedObject.PEDESTRIAN
    )
    assert confidence_threshold_for(TrackedObject.RED_LIGHT) >= 0.65
    assert confidence_threshold_for(TrackedObject.PEDESTRIAN) >= 0.65


def test_confidence_threshold_unknown_class_fallback():
    assert confidence_threshold_for("nonsense") == 0.55


def test_lidar_confirmation_pedestrian_no():
    """Plan TANDA 0.2: PEDESTRIAN ya no requiere LiDAR.

    El requisito viejo (True) hacía que peatones detectados por YOLO
    quedaran con ``position_world_xy=None`` cuando el LiDAR estaba
    ausente/tardío, y el scenario PedestrianYield los ignoraba en
    silencio. Compensamos con threshold de confianza más alto."""
    assert needs_lidar_confirmation(TrackedObject.PEDESTRIAN) is False


def test_lidar_confirmation_red_light_no():
    """Luz semáforo no tiene firma LiDAR útil."""
    assert needs_lidar_confirmation(TrackedObject.RED_LIGHT) is False


def test_lidar_confirmation_unknown_default_false():
    """Por defecto no requerir LiDAR (no-op safe)."""
    assert needs_lidar_confirmation("nonsense") is False
