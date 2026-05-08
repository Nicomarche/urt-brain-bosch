"""Driving-mode switch + speed/steer model.

Replaces the Angular ``state-switch.component``. This widget owns the
*authoritative* speed/steer state for manual mode — the backend only
holds the last value we sent, so when KL goes off or the user switches
to auto/legacy we have to reset locally and re-emit. Centralising that
logic here keeps it from being duplicated in the keyboard handler and
the on-screen joystick.

Public surface (used by KeyboardHandler and ControlPanel):

* ``current_mode`` — one of ``MODE_STOP/MANUAL/LEGACY/AUTO/PARKING``
* ``set_mode(name)`` — changes mode (also emits DrivingMode to backend)
* ``can_drive()`` — True iff manual + KL=30 (gates speed/steer commands)
* ``increase_speed()`` / ``decrease_speed()`` — clamped, emits SpeedMotor
* ``start_steer_left()`` / ``start_steer_right()`` / ``stop_steer()``
* ``brake_reset()`` — zero speed/steer + emit Brake

Wire protocol details (mirrors Angular exactly):

* Speed payload is ``round(speed_units * 10)`` as a string, max ±50.
* Steer payload is ``round(steer_units * 10)`` as a string, default ±25.
* Steering is incremental: while held, a QTimer at 50 ms ticks +/- by
  ``maxSteer/10`` and emits each step. On release, a separate timer
  ramps back to 0.
* ``SteeringLimits`` event from backend (re)configures min/max steer
  (raw values are ×10 so we divide).
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..client import events as ev
from ..client.socketio_client import SocketIOClient


_logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Internal model — separated from the QWidget so the keyboard handler can
# drive it without going through button presses.
# ----------------------------------------------------------------------
class DrivingModel(QObject):
    """Owns speed/steer, emits commands to the backend, signals UI updates."""

    mode_changed = pyqtSignal(str)
    speed_changed = pyqtSignal(int)   # rounded units (-50..50)
    steer_changed = pyqtSignal(int)   # rounded units (-max..max)
    can_drive_changed = pyqtSignal(bool)

    SPEED_STEP = 5
    SPEED_MIN = -50
    SPEED_MAX = 50
    STEER_NUM_STEPS = 10
    STEER_INTERVAL_MS = 50

    def __init__(self, client: SocketIOClient, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._client = client

        self._mode: str = ev.MODE_STOP
        self._kl_state: str = ev.KL_OFF
        self._speed: int = 0
        self._steer: float = 0.0
        # ±25° matchea el límite físico del servo Nucleo en el coche real
        # (BFMC firmware spec). En sim teníamos ±60° para "giros agresivos"
        # pero a esa apertura la geometría Ackermann mete diferencial
        # extremo (rueda interior trasera baja a 49% de ω nominal con
        # wheelbase 0.27/axletrack 0.16) → el chasis no puede acomodar
        # el giro y se ve como una caída de velocidad al doblar.
        # ±25° da diferencial ±14%, suave y sin slowdown perceptible.
        # Si el brain emite EVT_STEERING_LIMITS más estrictos,
        # `_on_steer_limits` los aplica encima.
        self._max_steer: float = 25.0
        self._min_steer: float = -25.0

        # Steering ramp timers. ``_ramp_dir`` is +1, -1 or 0 (decay).
        self._ramp_timer = QTimer(self)
        self._ramp_timer.setInterval(self.STEER_INTERVAL_MS)
        self._ramp_timer.timeout.connect(self._on_ramp_tick)
        self._ramp_dir: int = 0
        self._is_steering = False

        self._client.steering_limits_signal.connect(self._on_steer_limits)
        self._client.state_change_signal.connect(self._on_state_change_from_backend)

    # ------------------------------------------------------------------
    # Mode + KL integration
    # ------------------------------------------------------------------
    @property
    def current_mode(self) -> str:
        return self._mode

    @property
    def kl_state(self) -> str:
        return self._kl_state

    def set_mode(self, mode: str, emit_to_backend: bool = True) -> None:
        if mode == self._mode:
            return
        # Leaving manual → safe-reset.
        if self._mode == ev.MODE_MANUAL and mode != ev.MODE_MANUAL:
            self.brake_reset()
            self.stop_steer(decay=False)
        self._mode = mode
        self.mode_changed.emit(mode)
        if emit_to_backend:
            self._client.emit_message(ev.CMD_DRIVING_MODE, mode)
        self.can_drive_changed.emit(self.can_drive())

    def set_kl(self, kl: str) -> None:
        if kl == self._kl_state:
            return
        self._kl_state = kl
        # When KL drops out of RUN, kill any in-flight motion locally.
        if kl != ev.KL_RUN:
            if self._speed != 0:
                self._speed = 0
                self.speed_changed.emit(0)
            if self._steer != 0:
                self._steer = 0.0
                self.steer_changed.emit(0)
            self.stop_steer(decay=False)
        self.can_drive_changed.emit(self.can_drive())

    def can_drive(self) -> bool:
        return self._mode == ev.MODE_MANUAL and self._kl_state == ev.KL_RUN

    # ------------------------------------------------------------------
    # Speed
    # ------------------------------------------------------------------
    def increase_speed(self) -> None:
        if not self.can_drive():
            return
        self._speed = min(self.SPEED_MAX, self._speed + self.SPEED_STEP)
        self._emit_speed()

    def decrease_speed(self) -> None:
        if not self.can_drive():
            return
        self._speed = max(self.SPEED_MIN, self._speed - self.SPEED_STEP)
        self._emit_speed()

    def speed_reset(self) -> None:
        self._speed = 0
        self._emit_speed()

    def _emit_speed(self) -> None:
        wire = round(self._speed * 10)
        # Log de cada emit: deja claro que el modelo emitió y a qué
        # valor de wire. Si W dispara `[ Keyboard ] PRESS` pero NO esta
        # línea, el problema está entre el filter y el modelo (poco
        # probable — son una llamada directa). Si esta línea sale pero
        # `[ Dashboard ] RECV SpeedMotor` no, la sesión SocketIO está
        # rechazando el message (chequear `sessionActive`/`activeUser`).
        print(
            f"\033[1;97m[ DrivingModel ] :\033[0m \033[1;96mEMIT\033[0m"
            f" CMD_SPEED wire={wire} (units={self._speed})"
        )
        self._client.emit_message(ev.CMD_SPEED, str(wire))
        self.speed_changed.emit(int(self._speed))

    # ------------------------------------------------------------------
    # Steering — ramp up while held, decay back when released.
    # ------------------------------------------------------------------
    def start_steer_left(self) -> None:
        if not self.can_drive():
            return
        self._is_steering = True
        self._ramp_dir = -1
        if not self._ramp_timer.isActive():
            self._ramp_timer.start()

    def start_steer_right(self) -> None:
        if not self.can_drive():
            return
        self._is_steering = True
        self._ramp_dir = +1
        if not self._ramp_timer.isActive():
            self._ramp_timer.start()

    def stop_steer(self, decay: bool = True) -> None:
        """Release the steering hold. ``decay=True`` ramps back to 0."""
        self._is_steering = False
        if decay and self._steer != 0:
            self._ramp_dir = 0  # 0 means decay-toward-zero
            if not self._ramp_timer.isActive():
                self._ramp_timer.start()
        else:
            self._ramp_dir = 0
            if not decay:
                self._ramp_timer.stop()
                self._steer = 0.0
                self._emit_steer()

    def _on_ramp_tick(self) -> None:
        step = max(0.001, abs(self._max_steer) / self.STEER_NUM_STEPS)
        if self._ramp_dir == +1:
            self._steer = min(self._max_steer, self._steer + step)
        elif self._ramp_dir == -1:
            self._steer = max(self._min_steer, self._steer - step)
        else:
            # Decay toward zero.
            if self._steer > 0:
                self._steer = max(0.0, self._steer - step)
            elif self._steer < 0:
                self._steer = min(0.0, self._steer + step)
            else:
                self._ramp_timer.stop()
                return
        self._emit_steer()
        # Stop emitting once we've landed on the rail (Angular semantic).
        if self._ramp_dir == +1 and self._steer >= self._max_steer:
            return
        if self._ramp_dir == -1 and self._steer <= self._min_steer:
            return
        if self._ramp_dir == 0 and self._steer == 0.0:
            self._ramp_timer.stop()

    def _emit_steer(self) -> None:
        wire = round(self._steer * 10)
        # Solo logueamos cuando el wire CAMBIA (la rampa tickea cada
        # 50ms ⇒ 20Hz). Sin esto floodearíamos stdout durante hold.
        if wire != getattr(self, "_last_emit_steer_wire", None):
            print(
                f"\033[1;97m[ DrivingModel ] :\033[0m \033[1;96mEMIT\033[0m"
                f" CMD_STEER wire={wire} (deg={self._steer:+.1f}°)"
            )
            self._last_emit_steer_wire = wire
        self._client.emit_message(ev.CMD_STEER, str(wire))
        self.steer_changed.emit(int(round(self._steer)))

    # ------------------------------------------------------------------
    # Brake (spacebar)
    # ------------------------------------------------------------------
    def brake_reset(self) -> None:
        self._speed = 0
        self._steer = 0.0
        self._ramp_dir = 0
        self._ramp_timer.stop()
        print(
            f"\033[1;97m[ DrivingModel ] :\033[0m \033[1;96mEMIT\033[0m"
            f" CMD_BRAKE (full reset speed/steer)"
        )
        self._client.emit_message(ev.CMD_BRAKE, "0")
        self.speed_changed.emit(0)
        self.steer_changed.emit(0)

    # ------------------------------------------------------------------
    # Backend → us
    # ------------------------------------------------------------------
    def _on_steer_limits(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        try:
            upper = float(payload.get("upperLimit", 250)) / 10.0
            lower = float(payload.get("lowerLimit", -250)) / 10.0
        except (TypeError, ValueError):
            return
        self._max_steer = upper
        self._min_steer = lower

    def _on_state_change_from_backend(self, mode: str) -> None:
        """The brain (or another client) changed mode — sync without
        re-emitting (would create a feedback loop)."""
        if not mode:
            return
        normalised = mode.lower()
        if normalised in {ev.MODE_STOP, ev.MODE_MANUAL, ev.MODE_LEGACY,
                          ev.MODE_AUTO, ev.MODE_PARKING}:
            self.set_mode(normalised, emit_to_backend=False)


# ----------------------------------------------------------------------
# UI: a row of buttons, one per mode
# ----------------------------------------------------------------------
class StateSwitch(QWidget):
    """Five-button driving-mode selector backed by a ``DrivingModel``."""

    MODES = (
        (ev.MODE_STOP,    "Stop",    "#d9534f"),
        (ev.MODE_MANUAL,  "Manual",  "#f0ad4e"),
        (ev.MODE_LEGACY,  "Legacy",  "#2b8fd1"),
        (ev.MODE_AUTO,    "Auto",    "#5cb85c"),
        (ev.MODE_PARKING, "Parking", "#a87cff"),
    )

    def __init__(self, model: DrivingModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model = model

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        row = QHBoxLayout()
        row.setSpacing(4)
        for mode, label, color in self.MODES:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setStyleSheet(self._stylesheet(color))
            btn.clicked.connect(lambda _checked=False, m=mode: self._on_clicked(m))
            self._group.addButton(btn)
            self._buttons[mode] = btn
            row.addWidget(btn)

        wrapper = QVBoxLayout(self)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addLayout(row)

        self._model.mode_changed.connect(self._sync_buttons)
        self._sync_buttons(self._model.current_mode)

    @staticmethod
    def _stylesheet(color: str) -> str:
        return (
            "QPushButton { background:#222; color:#ccc; border:1px solid #333; "
            "padding:8px; font-weight:600; }"
            f"QPushButton:checked {{ background:{color}; color:white; "
            f"border:1px solid {color}; }}"
        )

    def _on_clicked(self, mode: str) -> None:
        self._model.set_mode(mode)

    def _sync_buttons(self, mode: str) -> None:
        if mode in self._buttons:
            self._buttons[mode].setChecked(True)
