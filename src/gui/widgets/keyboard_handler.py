"""Global keyboard driver for manual driving.

Installed as an *event filter* on the **QApplication** (NOT on the
MainWindow) so the keys work no matter which widget has focus — same
UX as the Angular ``@HostListener('window:keydown')``.

Why on the QApplication: a filter on a QWidget (incluso el MainWindow)
solo intercepta eventos dirigidos *a ese widget*. Cuando arranca la
GUI el foco cae en la `QListWidget` de la sidebar, y esa lista tiene
``keyboardSearch`` habilitado por default — al presionar W/A/S/D el
QListWidget los interpreta como búsqueda incremental y los CONSUME
antes de bubblear. Idem los QPushButton que comen Space/Enter. El
filter en `QApplication.instance()` corre ANTES que el widget
enfocado, así nuestra captura es la primera en ver la tecla.

Bindings (only active when ``DrivingModel.can_drive()`` is True):

  W / ↑      → speed +5
  S / ↓      → speed -5
  A / ← held → ramp steer left
  D / → held → ramp steer right
  A/D / ←/→ release → decay back to 0
  Space      → emergency brake reset

Why two key sets: WASD respects the existing muscle memory of devs que
ya usaban el dashboard, and arrows are the natural "drive a car"
binding for end-users. Same `DrivingModel` action bajo ambos — no
forking of state.

Why we *consume* (return True) cualquier tecla en la que actuamos:
estando con filter global, si no consumimos el evento llega también
al widget enfocado. Pressing W con foco en la sidebar dispararía
`increase_speed()` Y la búsqueda incremental del QListWidget; pressing
→ con foco en el QButtonGroup de StateSwitch movería el modo (Manual
→ Legacy) además de girar. Consumimos solo cuando *actuamos* — si
``can_drive=False`` o la tecla no es nuestra, devolvemos False y el
evento sigue su curso normal (text widgets, navegación de la sidebar,
etc. siguen funcionando fuera de manual).

Auto-repeat from the OS is suppressed by checking the event's
``isAutoRepeat()`` flag — Qt fires repeated keyDowns while a key is held,
which would otherwise stack accelerations.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtGui import QKeyEvent

from .state_switch import DrivingModel


_logger = logging.getLogger(__name__)


_STEER_KEYS = (Qt.Key_A, Qt.Key_D, Qt.Key_Left, Qt.Key_Right)


def _key_name(key: int) -> str:
    """Nombre legible de una Qt key. Para los logs."""
    return {
        Qt.Key_W: "W", Qt.Key_S: "S", Qt.Key_A: "A", Qt.Key_D: "D",
        Qt.Key_Up: "Up", Qt.Key_Down: "Down",
        Qt.Key_Left: "Left", Qt.Key_Right: "Right",
        Qt.Key_Space: "Space",
    }.get(key, f"key={key}")


class KeyboardHandler(QObject):
    """Event filter that drives a ``DrivingModel`` from WASD/arrows + space."""

    def __init__(self, model: DrivingModel, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._model = model
        self._active_keys: set[int] = set()

    # ------------------------------------------------------------------
    def install(self, target: QObject) -> None:
        """Install as an event filter on ``target``.

        For manejar manual queremos `QApplication.instance()` para que
        el filter vea TODAS las teclas antes que el widget enfocado.
        """
        target.installEventFilter(self)

    # ------------------------------------------------------------------
    def eventFilter(self, _obj, event):  # noqa: N802 - Qt convention
        et = event.type()
        if et == QEvent.KeyPress:
            handled = self._on_key_press(event)
        elif et == QEvent.KeyRelease:
            handled = self._on_key_release(event)
        else:
            # Filter en QApplication recibe TODOS los eventos (mouse,
            # paint, resize…). Salida temprana barata para no agregar
            # overhead al loop de eventos.
            return False
        # Consumimos toda tecla en la que ACTUAMOS — devuelve False
        # cuando no fue una de las nuestras o `can_drive=False`, así
        # los eventos siguen fluyendo normalmente fuera de manual.
        return bool(handled)

    # ------------------------------------------------------------------
    def _on_key_press(self, event: QKeyEvent) -> bool:
        if event.isAutoRepeat():
            return False
        key = event.key()
        # Visibilidad: TODA tecla relevante se loguea, incluso si la
        # rechazamos por can_drive=False. Así el operador ve si el
        # event filter se está disparando o no, y por qué la rechazamos.
        # Si esta línea no aparece al pulsar W, el filter no está
        # instalado o la ventana no tiene foco.
        relevant = key in (
            Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D, Qt.Key_Space,
            Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right,
        )
        if relevant:
            cd = self._model.can_drive()
            print(
                f"\033[1;97m[ Keyboard ] :\033[0m \033[1;96mPRESS\033[0m "
                f"key={_key_name(key)} can_drive={cd} "
                f"mode={self._model.current_mode} kl={self._model.kl_state}"
            )
            if not cd:
                return False
        else:
            return False
        if key in self._active_keys:
            return False
        self._active_keys.add(key)

        if key in (Qt.Key_W, Qt.Key_Up):
            self._model.increase_speed()
            return True
        if key in (Qt.Key_S, Qt.Key_Down):
            self._model.decrease_speed()
            return True
        if key in (Qt.Key_A, Qt.Key_Left):
            self._model.start_steer_left()
            return True
        if key in (Qt.Key_D, Qt.Key_Right):
            self._model.start_steer_right()
            return True
        if key == Qt.Key_Space:
            self._model.brake_reset()
            return True
        return False

    def _on_key_release(self, event: QKeyEvent) -> bool:
        if event.isAutoRepeat():
            return False
        key = event.key()
        if key not in self._active_keys:
            return False
        self._active_keys.discard(key)

        if key in _STEER_KEYS:
            self._model.stop_steer(decay=True)
            return True
        return False
