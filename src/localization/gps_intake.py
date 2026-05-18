"""GpsIntake — gestor de fixes GPS para el pose_estimator.

Plan A2.2: 11 helpers de GPS conviven en ``pose_estimator_thread`` (líneas
269-1013). Esta clase los agrupa con API limpia, mantiene los buffers
internos, y queda lista para integrar al orchestrator (A3).

NO duplica la lógica legacy compleja de auto-detección de inicio o de
GPS calibration — esa lógica de fondo sigue en el thread por ahora.
Acá vive el contrato mínimo que el orchestrator usa.

Pipeline:
  * ``receive_latest_fix()`` drena el FIFO del subscriber LOCALISATION y
    devuelve el último fix válido.
  * Historial ``_fix_history`` mantenido para estadísticas de auto-start.
  * ``apply_calibration_offset(payload, meta)`` aplica el offset cuando el
    operador inicia calibración GPS desde el dashboard.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GpsFix:
    """Una medición GPS proyectada al frame del mapa."""

    timestamp: float
    x: float
    y: float
    raw_payload: dict[str, Any] = field(default_factory=dict)


class GpsIntake:
    """Subscriber + buffers del flujo de fixes GPS para el pose_estimator.

    El thread compone esta clase y delega cualquier operación de bus a
    ella. Cuando se construye con ``subscriber=None``, opera en modo
    "puro" — útil para tests sin queues.
    """

    def __init__(
        self,
        *,
        subscriber: Any = None,
        history_max: int = 60,
        auto_start_sample_count: int = 6,
    ) -> None:
        self._subscriber = subscriber
        self._history: deque[GpsFix] = deque(maxlen=int(max(2, history_max)))
        self._auto_start_n = int(auto_start_sample_count)
        self._last_fix: GpsFix | None = None
        self._calibration_offset_xy: tuple[float, float] | None = None
        self._calibration_pending: bool = False
        self._auto_gps_pending: bool = False
        self._auto_gps_started_monotonic: float = 0.0

    # ─── Receive ──────────────────────────────────────────────────────
    def receive_latest_fix(self) -> GpsFix | None:
        """Drena el FIFO del subscriber, devuelve el último fix.

        Si no hay subscriber (modo puro), devuelve el último ``push_fix()``.
        Cada fix drenado se agrega al historial (no sólo el último).
        """
        if self._subscriber is None:
            return self._last_fix
        last: GpsFix | None = None
        for _ in range(10):
            raw = self._subscriber.receive()
            if raw is None:
                break
            parsed = self._parse_fix(raw)
            if parsed is None:
                continue
            self._history.append(parsed)
            last = parsed
        if last is not None:
            self._last_fix = last
        return last

    def push_fix(self, fix: GpsFix) -> None:
        """Inyección directa para tests sin queues."""
        self._last_fix = fix
        self._history.append(fix)

    # ─── Calibration / auto-start ─────────────────────────────────────
    def try_start_calibration(self, yaw_rad: float, now_s: float) -> bool:
        """Marca calibración pendiente.

        El thread legacy aplica el offset cuando llega el próximo fix.
        Acá sólo gestionamos el flag.
        """
        self._calibration_pending = True
        return True

    def try_auto_gps_entry(
        self,
        yaw_rad: float,
        now_s: float,
        lanelet_map: Any = None,
    ) -> bool:
        """Marca auto-start GPS pendiente."""
        self._auto_gps_pending = True
        self._auto_gps_started_monotonic = float(now_s)
        return True

    def apply_calibration_offset(
        self,
        payload: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> tuple[float, float] | None:
        """Aplica un offset (x, y) calibrado desde el dashboard.

        Args:
            payload: ``{"offset_x": ..., "offset_y": ...}`` o similar.

        Returns:
            ``(offset_x, offset_y)`` aplicado, o ``None`` si payload
            inválido.
        """
        try:
            ox = float(payload.get("offset_x", 0.0))
            oy = float(payload.get("offset_y", 0.0))
        except (TypeError, ValueError):
            return None
        self._calibration_offset_xy = (ox, oy)
        self._calibration_pending = False
        return (ox, oy)

    # ─── State accessors ──────────────────────────────────────────────
    @property
    def last_fix(self) -> GpsFix | None:
        return self._last_fix

    @property
    def history(self) -> tuple[GpsFix, ...]:
        return tuple(self._history)

    @property
    def has_calibration(self) -> bool:
        return self._calibration_offset_xy is not None

    @property
    def calibration_offset(self) -> tuple[float, float] | None:
        return self._calibration_offset_xy

    @property
    def calibration_pending(self) -> bool:
        return self._calibration_pending

    @property
    def auto_gps_pending(self) -> bool:
        return self._auto_gps_pending

    def clear_auto_gps(self) -> None:
        self._auto_gps_pending = False

    # ─── Helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _parse_fix(raw: Any) -> GpsFix | None:
        if not isinstance(raw, dict):
            return None
        try:
            x = float(raw.get("x", 0.0))
            y = float(raw.get("y", 0.0))
            ts = float(raw.get("timestamp", time.time()))
        except (TypeError, ValueError):
            return None
        return GpsFix(timestamp=ts, x=x, y=y, raw_payload=dict(raw))


__all__ = ["GpsFix", "GpsIntake"]
