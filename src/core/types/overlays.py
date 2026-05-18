"""Tipos para overlays vectoriales separados del JPEG raw.

Plan F3: hoy los productores (threadLocalPerception, lane_observer)
queman overlays sobre el JPEG raw (lane lines, sign boxes, lidar pts)
ANTES de codificar. Esto:
  * impide toggleo individual de capas en el GUI.
  * dobla el ancho de banda (raw + overlay quemado).
  * pierde precisión sub-píxel.

Solución: productores publican `OverlayLayer` con primitivas
vectoriales; el GUI las dibuja sobre el QPixmap raw con un check
por capa.

El topic ``OVERLAY_LAYER_MSG`` lleva un layer a la vez (un overlay
por publicación). Las publicaciones llevan ``header.frame_id`` para
que el GUI sincronice con el frame raw correspondiente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Tipos primitivos para dibujar — coordenadas en píxeles del frame raw.
Line2D = tuple[float, float, float, float]  # (x0, y0, x1, y1)
Polygon2D = list[tuple[float, float]]
Box2D = tuple[float, float, float, float]   # (x0, y0, x1, y1) bbox xyxy


@dataclass(frozen=True)
class OverlayLayer:
    """Una capa vectorial sobre el frame de cámara raw.

    Attributes:
        layer_name: ``"lane"``, ``"signs"``, ``"lidar_obstacles"``,
            ``"detected_objects"``. Toggle por nombre en el GUI.
        frame_id: Identificador del frame raw al que pertenece (debe
            coincidir con el ``header.frame_id`` del MAIN_CAMERA).
        timestamp: Cuando se computó la capa (s, UNIX).
        lines: Líneas a dibujar (cada línea = 4 floats).
        polygons: Polígonos (lista de listas de puntos).
        boxes: Bounding boxes (cada uno xyxy).
        color_hex: Color por defecto si la primitiva no trae el suyo.
        labels: Lista de ``(x, y, text)`` para anotar.
    """

    layer_name: str
    frame_id: str = ""
    timestamp: float = 0.0
    lines: list[Line2D] = field(default_factory=list)
    polygons: list[Polygon2D] = field(default_factory=list)
    boxes: list[Box2D] = field(default_factory=list)
    color_hex: str = "#FFD700"
    labels: list[tuple[float, float, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": str(self.layer_name),
            "frame_id": str(self.frame_id),
            "timestamp": float(self.timestamp),
            "lines": [list(l) for l in self.lines],
            "polygons": [[list(p) for p in poly] for poly in self.polygons],
            "boxes": [list(b) for b in self.boxes],
            "color_hex": str(self.color_hex),
            "labels": [list(t) for t in self.labels],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OverlayLayer":
        if not isinstance(data, dict):
            return cls(layer_name="")
        return cls(
            layer_name=str(data.get("layer_name", "")),
            frame_id=str(data.get("frame_id", "")),
            timestamp=float(data.get("timestamp", 0.0)),
            lines=[tuple(map(float, l)) for l in (data.get("lines") or [])],  # type: ignore[misc]
            polygons=[
                [tuple(map(float, p)) for p in poly]
                for poly in (data.get("polygons") or [])
            ],
            boxes=[tuple(map(float, b)) for b in (data.get("boxes") or [])],  # type: ignore[misc]
            color_hex=str(data.get("color_hex", "#FFD700")),
            labels=[tuple(l) for l in (data.get("labels") or [])],  # type: ignore[misc]
        )


__all__ = ["OverlayLayer", "Line2D", "Polygon2D", "Box2D"]
