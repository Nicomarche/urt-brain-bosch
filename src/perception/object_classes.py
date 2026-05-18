"""Catálogo unificado de clases de objetos detectados — IntEnum + helpers.

Plan C1.1: el repo REF tiene un dict ``OBJECT_DICT`` con 13 clases que el
detector reporta como ID entero. Mi repo trabaja con strings (más
flexible pero propenso a typos). Acá centralizamos AMBAS convenciones:

  * ``TrackedObject`` IntEnum — value = ID compatible con el dict del REF
    (oneway=0, highway_entrance=1, ...).
  * ``CLASS_NAME_TO_ENUM`` y ``ENUM_TO_CLASS_NAME`` — mapeo bidireccional
    str ↔ enum.
  * ``SIGN_CONFIDENCE_PER_CLASS`` — threshold por clase (reemplaza el
    global ``LOCAL_AI_SIGN_MIN_CONFIDENCE`` para ser estricto en signs
    críticos sin perder peatones a baja confianza).
  * ``LIDAR_CONFIRMATION_REQUIRED`` — flag por clase (plan C4).

Convención de IDs del REF (mantenemos compatibilidad por si el detector
publica IDs en vez de strings):
    0 ONEWAY · 1 HIGHWAY_ENTRANCE · 2 STOP_SIGN · 3 ROUNDABOUT
    4 PARK · 5 CROSSWALK · 6 NO_ENTRY · 7 HIGHWAY_EXIT
    8 PRIORITY · 9 TRAFFIC_LIGHT · 10 BLOCK · 11 PEDESTRIAN · 12 CAR

Extensiones MI repo (>=13):
    13 RED_LIGHT · 14 YELLOW_LIGHT · 15 GREEN_LIGHT
    16 STOPLINE_VISUAL (detección CV clásica, no YOLO)
"""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable


class TrackedObject(IntEnum):
    """Catálogo de clases detectadas por el pipeline de percepción.

    IDs 0-12 alineados byte-a-byte con OBJECT_DICT del repo REF; 13+ son
    extensiones de mi repo (luces semaforo separadas por color, stopline
    visual). Si en el futuro YOLO publica directamente ID entero, este
    enum es el contrato.
    """

    ONEWAY = 0
    HIGHWAY_ENTRANCE = 1
    STOP_SIGN = 2
    ROUNDABOUT = 3
    PARK = 4
    CROSSWALK = 5
    NO_ENTRY = 6
    HIGHWAY_EXIT = 7
    PRIORITY = 8
    TRAFFIC_LIGHT = 9
    BLOCK = 10
    PEDESTRIAN = 11
    CAR = 12
    # Extensiones MI repo
    RED_LIGHT = 13
    YELLOW_LIGHT = 14
    GREEN_LIGHT = 15
    STOPLINE_VISUAL = 16


# Mapeo string canonical → TrackedObject. Convención: snake_case. Toma
# los strings que sign_classifier.normalize_sign_name() ya produce.
CLASS_NAME_TO_ENUM: dict[str, TrackedObject] = {
    "oneway": TrackedObject.ONEWAY,
    "highway_entrance": TrackedObject.HIGHWAY_ENTRANCE,
    "stop": TrackedObject.STOP_SIGN,
    "stop_sign": TrackedObject.STOP_SIGN,
    "stopsign": TrackedObject.STOP_SIGN,
    "roundabout": TrackedObject.ROUNDABOUT,
    "park": TrackedObject.PARK,
    "parking": TrackedObject.PARK,
    "crosswalk": TrackedObject.CROSSWALK,
    "no_entry": TrackedObject.NO_ENTRY,
    "noentry": TrackedObject.NO_ENTRY,
    "highway_exit": TrackedObject.HIGHWAY_EXIT,
    "priority": TrackedObject.PRIORITY,
    "traffic_light": TrackedObject.TRAFFIC_LIGHT,
    "lights": TrackedObject.TRAFFIC_LIGHT,
    "block": TrackedObject.BLOCK,
    "pedestrian": TrackedObject.PEDESTRIAN,
    "car": TrackedObject.CAR,
    "red_light": TrackedObject.RED_LIGHT,
    "yellow_light": TrackedObject.YELLOW_LIGHT,
    "green_light": TrackedObject.GREEN_LIGHT,
    "stopline": TrackedObject.STOPLINE_VISUAL,
    "stopline_visual": TrackedObject.STOPLINE_VISUAL,
}

# Inverso para serialización canónica al bus / GUI.
ENUM_TO_CLASS_NAME: dict[TrackedObject, str] = {
    TrackedObject.ONEWAY: "oneway",
    TrackedObject.HIGHWAY_ENTRANCE: "highway_entrance",
    TrackedObject.STOP_SIGN: "stop",
    TrackedObject.ROUNDABOUT: "roundabout",
    TrackedObject.PARK: "park",
    TrackedObject.CROSSWALK: "crosswalk",
    TrackedObject.NO_ENTRY: "no_entry",
    TrackedObject.HIGHWAY_EXIT: "highway_exit",
    TrackedObject.PRIORITY: "priority",
    TrackedObject.TRAFFIC_LIGHT: "traffic_light",
    TrackedObject.BLOCK: "block",
    TrackedObject.PEDESTRIAN: "pedestrian",
    TrackedObject.CAR: "car",
    TrackedObject.RED_LIGHT: "red_light",
    TrackedObject.YELLOW_LIGHT: "yellow_light",
    TrackedObject.GREEN_LIGHT: "green_light",
    TrackedObject.STOPLINE_VISUAL: "stopline_visual",
}


# Plan C1.1: thresholds por clase. Permite ser ESTRICTO en signs críticos
# (stop/red_light) sin perder peatones que pueden detectarse a baja
# confianza. Reemplaza el global LOCAL_AI_SIGN_MIN_CONFIDENCE=0.55.
SIGN_CONFIDENCE_PER_CLASS: dict[TrackedObject, float] = {
    TrackedObject.STOP_SIGN: 0.65,
    TrackedObject.RED_LIGHT: 0.70,
    TrackedObject.YELLOW_LIGHT: 0.65,
    TrackedObject.GREEN_LIGHT: 0.65,
    TrackedObject.TRAFFIC_LIGHT: 0.65,
    # Plan TANDA 0.2: compensamos sacar la confirmación LiDAR obligatoria
    # con un threshold más estricto. Antes 0.55 + LIDAR_REQUIRED=True; ahora
    # 0.65 sin LiDAR obligatorio para que YOLO solo pueda frenar al auto.
    TrackedObject.PEDESTRIAN: 0.65,
    TrackedObject.CAR: 0.60,
    TrackedObject.CROSSWALK: 0.55,
    TrackedObject.NO_ENTRY: 0.70,
    TrackedObject.PRIORITY: 0.60,
    TrackedObject.ROUNDABOUT: 0.60,
    TrackedObject.HIGHWAY_ENTRANCE: 0.60,
    TrackedObject.HIGHWAY_EXIT: 0.60,
    TrackedObject.PARK: 0.60,
    TrackedObject.ONEWAY: 0.55,
    TrackedObject.BLOCK: 0.55,
    TrackedObject.STOPLINE_VISUAL: 0.50,
}


# Plan C4: confirmación LiDAR obligatoria por clase. True = el hint solo
# emite como ``actionable`` si hay match con un cluster LiDAR; sino,
# queda ``tentative`` (no dispara reacción).
LIDAR_CONFIRMATION_REQUIRED: dict[TrackedObject, bool] = {
    TrackedObject.STOP_SIGN: True,
    TrackedObject.RED_LIGHT: False,      # luz no tiene firma lidar útil
    TrackedObject.YELLOW_LIGHT: False,
    TrackedObject.GREEN_LIGHT: False,
    TrackedObject.TRAFFIC_LIGHT: False,
    # Plan TANDA 0.2: ANTES True — el peatón era ignorado cuando el LiDAR
    # estaba ausente/tardío, porque position_world_xy quedaba None y el
    # scenario lo descartaba. Ahora False; compensamos subiendo el
    # threshold de confianza YOLO a 0.65 (vs el 0.55 viejo). El fallback
    # de distancia con cámara pinhole (proyección al piso) le da
    # position_world_xy razonable a 1-3m.
    TrackedObject.PEDESTRIAN: False,
    TrackedObject.CROSSWALK: False,      # zona pintada, sin firma lidar
    TrackedObject.CAR: True,
    TrackedObject.BLOCK: True,
    TrackedObject.NO_ENTRY: False,
    TrackedObject.PRIORITY: False,
    TrackedObject.ROUNDABOUT: False,
    TrackedObject.HIGHWAY_ENTRANCE: False,
    TrackedObject.HIGHWAY_EXIT: False,
    TrackedObject.PARK: False,
    TrackedObject.ONEWAY: False,
    TrackedObject.STOPLINE_VISUAL: False,
}


def to_enum(class_name_or_id: str | int | TrackedObject | None) -> TrackedObject | None:
    """Convierte un string o ID a TrackedObject. ``None`` si no matchea.

    Es tolerante: si recibe un IntEnum ya retorna; si es int matchea por
    valor; si es str pasa por ``CLASS_NAME_TO_ENUM`` con normalización.
    """
    if class_name_or_id is None:
        return None
    if isinstance(class_name_or_id, TrackedObject):
        return class_name_or_id
    if isinstance(class_name_or_id, int):
        try:
            return TrackedObject(class_name_or_id)
        except ValueError:
            return None
    if isinstance(class_name_or_id, str):
        normalized = class_name_or_id.strip().lower().replace("-", "_").replace(" ", "_")
        return CLASS_NAME_TO_ENUM.get(normalized)
    return None


def confidence_threshold_for(obj: TrackedObject | str | int) -> float:
    """Devuelve el umbral de confianza para una clase. Fallback global=0.55."""
    enum_val = to_enum(obj)
    if enum_val is None:
        return 0.55
    return SIGN_CONFIDENCE_PER_CLASS.get(enum_val, 0.55)


def needs_lidar_confirmation(obj: TrackedObject | str | int) -> bool:
    """Plan C4: True si la clase requiere match LiDAR para ser actionable."""
    enum_val = to_enum(obj)
    if enum_val is None:
        return False
    return LIDAR_CONFIRMATION_REQUIRED.get(enum_val, False)


__all__ = [
    "TrackedObject",
    "CLASS_NAME_TO_ENUM",
    "ENUM_TO_CLASS_NAME",
    "SIGN_CONFIDENCE_PER_CLASS",
    "LIDAR_CONFIRMATION_REQUIRED",
    "to_enum",
    "confidence_threshold_for",
    "needs_lidar_confirmation",
]
