"""Debug pad to bisect the manual-driving pipeline.

Why this widget exists
----------------------
"Pulso W y el auto no se mueve" tiene tres causas posibles:

  1. La captura de teclado no dispara (focus, event filter mal instalado,
     `DrivingModel.can_drive()` False, etc.).
  2. El emit a SocketIO se hace pero no llega al brain (sesión inactiva,
     gateway saturado).
  3. Llega al brain pero `threadWrite` no despacha (engine off, serial
     desconectado, sim_bridge no suscripto).

Los botones de este pad **emiten el wire-level CMD_SPEED/CMD_STEER/CMD_BRAKE
directamente al `SocketIOClient`**, bypaseando `DrivingModel` y su
`can_drive()` gate. Eso aísla el camino *send → backend → firmware* del
camino *teclado → modelo*. Si los botones mueven el auto pero W no, la
falla está en la captura. Si ni los botones lo mueven, está aguas abajo.

Es un widget de DEBUG: valores fijos, sin acumulador, sin ramp. No
reemplaza a la rampa del `DrivingModel`.

Layout
------

      ┌─ Speed=200 ─┐
      ┌── Steer=-100 ─┐ ┌─── Stop ───┐ ┌── Steer=+100 ─┐
      ┌─ Speed=-200 ┐  ┌── Brake ──┐
      ┌─ Speed=50 ─┐  ┌─ Speed=0 ─┐
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..client import events as ev
from ..client.zmq_bus_client import ZmqBusClient as SocketIOClient


_logger = logging.getLogger(__name__)


# Wire-format de speed/steer (lo que viaja por SocketIO):
#   speed: cm/s × 10. 50 = 5 cm/s, 200 = 20 cm/s.
#   steer: grados × 10. 100 = 10°, -100 = -10°.
# Estos valores son los mismos que el dashboard real envía cuando
# acumulás presses de W o sostenés A/D — solo que acá los mandamos
# de un saque.
_SPEED_LOW = 50      # 5 cm/s — debajo de la fricción estática típica
_SPEED_HIGH = 200    # 20 cm/s — debería arrancar en cualquier setup
_SPEED_REV = -200    # marcha atrás
_STEER_FULL = 100    # 10° (la mitad del rango ±25°)


class ManualTestPad(QWidget):
    """Wire-test buttons. Bypassean DrivingModel y emiten CMD_* crudos."""

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._client = client

        title = QLabel("Manual Test Pad — bypassea DrivingModel (DEBUG)")
        title.setStyleSheet("color:#ffd866; font-weight:600; padding:2px;")

        # Fila 1: speed forward + reverse
        b_spd_low = self._btn(f"▲ Speed={_SPEED_LOW}",
                              lambda: self._emit(ev.CMD_SPEED, _SPEED_LOW))
        b_spd_high = self._btn(f"▲▲ Speed={_SPEED_HIGH}",
                               lambda: self._emit(ev.CMD_SPEED, _SPEED_HIGH))
        b_spd_rev = self._btn(f"▼ Speed={_SPEED_REV}",
                              lambda: self._emit(ev.CMD_SPEED, _SPEED_REV))
        b_spd_zero = self._btn("Speed=0",
                               lambda: self._emit(ev.CMD_SPEED, 0))

        # Fila 2: steer L / center / R
        b_str_l = self._btn(f"◀ Steer={-_STEER_FULL}",
                            lambda: self._emit(ev.CMD_STEER, -_STEER_FULL))
        b_str_0 = self._btn("Steer=0",
                            lambda: self._emit(ev.CMD_STEER, 0))
        b_str_r = self._btn(f"Steer={_STEER_FULL} ▶",
                            lambda: self._emit(ev.CMD_STEER, _STEER_FULL))

        # Fila 3: brake (mantengo separado porque corta speed Y steer en firmware)
        b_brake = self._btn("⏹ BRAKE",
                            lambda: self._emit(ev.CMD_BRAKE, 0))
        b_brake.setStyleSheet(self._btn_style(accent="#d9534f"))

        grid = QGridLayout()
        grid.setSpacing(6)
        grid.addWidget(b_spd_high, 0, 0)
        grid.addWidget(b_spd_low,  0, 1)
        grid.addWidget(b_spd_zero, 0, 2)
        grid.addWidget(b_spd_rev,  0, 3)

        grid.addWidget(b_str_l, 1, 0)
        grid.addWidget(b_str_0, 1, 1, 1, 2)
        grid.addWidget(b_str_r, 1, 3)

        grid.addWidget(b_brake, 2, 0, 1, 4)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.addWidget(title)
        outer.addLayout(grid)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.setStyleSheet("ManualTestPad { background:#1a1a1a; border:1px solid #333; border-radius:6px; }")

    # ------------------------------------------------------------------
    def _emit(self, cmd: str, value: int) -> None:
        """Manda un CMD_* al backend. Loguea para que el operador vea
        que el botón disparó el send (y pueda comparar con la cadena
        teclado → DrivingModel → backend)."""
        wire = str(value)
        print(
            f"\033[1;97m[ ManualTestPad ] :\033[0m \033[1;96mSEND\033[0m"
            f" {cmd}={wire}"
        )
        self._client.emit_message(cmd, wire)

    # ------------------------------------------------------------------
    def _btn(self, label: str, slot) -> QPushButton:
        btn = QPushButton(label)
        btn.setStyleSheet(self._btn_style())
        btn.clicked.connect(slot)
        return btn

    @staticmethod
    def _btn_style(accent: str = "#5cb85c") -> str:
        return (
            "QPushButton { background:#2a2a2a; color:#eee; border:1px solid #444; "
            "padding:8px 12px; font-weight:600; }"
            "QPushButton:hover { background:#3a3a3a; }"
            f"QPushButton:pressed {{ background:{accent}; color:white; }}"
        )
