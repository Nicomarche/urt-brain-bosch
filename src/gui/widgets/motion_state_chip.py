"""motion_state_chip — widget mostrando el sub-estado granular del planner.

Plan F2: ``BehaviorOutput.motion_state`` (plan B3) reporta sub-fase
("moving", "approaching_intersection", "parking", ...). Esto vive como
un QLabel chip con color asignado por categoría, en el toolbar top del
main_window.

Colores (QSS):
  * verde   → marcha sin restricciones (moving, highway_cruise)
  * amarillo → atención (approaching_*, in_curve_*, exiting_*)
  * rojo    → frenando o parado (waiting_for_*, parking, pedestrian_yielding)
  * gris    → idle (done, fallback)
"""

from __future__ import annotations

from typing import Any

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QLabel
except ImportError:  # pragma: no cover - tests sin Qt
    Qt = None
    QLabel = None  # type: ignore[assignment]


_GREEN = "#2ed47a"
_YELLOW = "#e6c34a"
_RED = "#e25555"
_GRAY = "#888888"


_COLOR_BY_STATE: dict[str, str] = {
    # Verde (marcha)
    "moving": _GREEN,
    "highway_cruise": _GREEN,
    # Amarillo (precaución)
    "approaching_intersection": _YELLOW,
    "approaching_parking": _YELLOW,
    "in_intersection": _YELLOW,
    "exiting_intersection": _YELLOW,
    "in_curve_left": _YELLOW,
    "in_curve_right": _YELLOW,
    "entering_roundabout": _YELLOW,
    "in_roundabout": _YELLOW,
    "exiting_roundabout": _YELLOW,
    "exiting_parking": _YELLOW,
    "overtake_active": _YELLOW,
    # Rojo (freno / yield)
    "waiting_for_stop_sign": _RED,
    "waiting_for_traffic_light": _RED,
    "parking": _RED,
    "crosswalk_yielding": _RED,
    "pedestrian_yielding": _RED,
    # Gris (idle)
    "parked": _GRAY,
    "done": _GRAY,
    "fallback": _GRAY,
}


def color_for_motion_state(state: str) -> str:
    """Devuelve hex color para un motion_state. Default = gris."""
    return _COLOR_BY_STATE.get(str(state or "").strip().lower(), _GRAY)


if QLabel is not None:

    class MotionStateChip(QLabel):
        """QLabel chip que se actualiza desde BEHAVIOR_OUTPUT_MSG.motion_state."""

        def __init__(self, parent=None) -> None:
            super().__init__("—", parent)
            self.setAlignment(Qt.AlignCenter)
            self.setStyleSheet(self._qss(_GRAY))
            self.setMinimumWidth(160)

        def update_from_payload(self, payload: Any) -> None:
            """Lee ``motion_state`` (string) de un BEHAVIOR_OUTPUT_MSG."""
            if not isinstance(payload, dict):
                return
            state = str(payload.get("motion_state", "")).strip().lower()
            if not state:
                return
            color = color_for_motion_state(state)
            self.setText(state.replace("_", " ").upper())
            self.setStyleSheet(self._qss(color))

        @staticmethod
        def _qss(color: str) -> str:
            return (
                "QLabel {"
                f"  background: {color};"
                "  color: #1a1a1a;"
                "  border-radius: 10px;"
                "  padding: 4px 10px;"
                "  font-weight: 600;"
                "  font-size: 10pt;"
                "}"
            )

else:  # pragma: no cover - Qt no instalado
    class MotionStateChip:  # type: ignore[no-redef]
        """Stub sin Qt — sólo permite tests del color helper."""

        pass


__all__ = ["MotionStateChip", "color_for_motion_state"]
