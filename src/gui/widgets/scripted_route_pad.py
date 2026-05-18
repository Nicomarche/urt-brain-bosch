"""Open-loop scripted driving widget for fixed-track fallback testing.

This widget exists for one very specific workflow: when the regular AUTO
stack is misbehaving, the operator may still want to replay a fixed,
time-based sequence of raw speed/steer commands to verify the car can at
least drive a known open-loop route around the track.

Important properties:

* It does NOT use the route planner, behavior planner, camera, IMU or map.
* It sends raw ``SpeedMotor`` / ``SteerMotor`` commands directly from the
  GUI using a timed script loaded from ``missions/open_loop_route.json``.
* It forces ``DrivingMode=manual`` before the sequence starts so the AUTO
  dispatcher stops fighting the operator script.
* It requires KL=30 / engine enabled. Safety-wise we do not auto-arm KL.

The route file is versioned in the repo so the team can tweak durations and
angles without touching Python code for every experiment.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..client import events as ev
from ..client.zmq_bus_client import ZmqBusClient as SocketIOClient


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "missions" / "open_loop_route.json"
_START_DELAY_S = 0.30
_STATUS_REFRESH_MS = 100


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _ScriptStep:
    label: str
    duration_s: float
    speed_wire: int
    steer_wire: int

    @property
    def speed_cms(self) -> float:
        return self.speed_wire / 10.0

    @property
    def steer_deg(self) -> float:
        return self.steer_wire / 10.0


class ScriptedRoutePad(QFrame):
    """Run a repo-local open-loop route script from the Driving tab."""

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._client = client
        self._script_name = "Open-loop route"
        self._script_description = ""
        self._steps: list[_ScriptStep] = []
        self._final_brake = True

        self._klem_mode = 0
        self._engine_enabled = False
        self._current_mode = ""
        self._running = False
        self._active_step_idx = -1
        self._step_started_monotonic = 0.0
        self._step_deadline_monotonic = 0.0
        self._pending_start = False

        title = QLabel("Scripted Route — open-loop fallback")
        title.setStyleSheet("color:#9fd3ff; font-weight:600; padding:2px;")

        self._name_label = QLabel("")
        self._name_label.setStyleSheet("color:#ddd; font-weight:600;")

        self._desc_label = QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color:#aaa;")

        self._status_label = QLabel("Idle")
        self._status_label.setStyleSheet("color:#ffd866; font-weight:600;")

        self._step_label = QLabel("No step running")
        self._step_label.setStyleSheet("color:#ccc;")

        self._path_label = QLabel(str(_SCRIPT_PATH))
        self._path_label.setStyleSheet("color:#666; font-size:9pt;")

        self._start_btn = QPushButton("Run Open-Loop Route")
        self._start_btn.setStyleSheet(self._btn_style("#5cb85c"))
        self._start_btn.clicked.connect(self._start_route)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setStyleSheet(self._btn_style("#d9534f"))
        self._stop_btn.clicked.connect(self._stop_route)
        self._stop_btn.setEnabled(False)

        self._reload_btn = QPushButton("Reload Script")
        self._reload_btn.setStyleSheet(self._btn_style("#2b8fd1"))
        self._reload_btn.clicked.connect(self._reload_script)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self._start_btn)
        buttons.addWidget(self._stop_btn)
        buttons.addWidget(self._reload_btn)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addWidget(title)
        layout.addWidget(self._name_label)
        layout.addWidget(self._desc_label)
        layout.addLayout(buttons)
        layout.addWidget(self._status_label)
        layout.addWidget(self._step_label)
        layout.addWidget(self._path_label)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.setStyleSheet(
            "ScriptedRoutePad { background:#151515; border:1px solid #2f2f2f; border-radius:6px; }"
        )

        self._step_timer = QTimer(self)
        self._step_timer.setSingleShot(True)
        self._step_timer.timeout.connect(self._advance_step)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(_STATUS_REFRESH_MS)
        self._status_timer.timeout.connect(self._refresh_running_status)

        self._client.actuator_status_signal.connect(self._on_actuator_status)
        self._client.state_change_signal.connect(self._on_state_change)

        self._reload_script()

    @staticmethod
    def _btn_style(accent: str) -> str:
        return (
            "QPushButton { background:#242424; color:#eee; border:1px solid #444; "
            "padding:8px 12px; font-weight:600; }"
            "QPushButton:hover { background:#303030; }"
            f"QPushButton:pressed {{ background:{accent}; color:white; }}"
            "QPushButton:disabled { color:#666; border:1px solid #333; }"
        )

    def _reload_script(self) -> None:
        if self._running:
            self._set_status("Stop the route before reloading the script.")
            return
        try:
            payload = json.loads(_SCRIPT_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._steps = []
            self._script_name = "Open-loop route"
            self._script_description = "Script file not found."
            self._render_script_summary()
            self._set_status(f"Missing script file: {_SCRIPT_PATH}")
            return
        except Exception as exc:
            self._steps = []
            self._script_name = "Open-loop route"
            self._script_description = f"Failed to parse JSON: {exc}"
            self._render_script_summary()
            self._set_status("Script reload failed.")
            return

        steps = self._parse_steps(payload.get("steps"))
        self._script_name = str(payload.get("name") or "Open-loop route")
        self._script_description = str(payload.get("description") or "")
        self._final_brake = bool(payload.get("final_brake", True))
        self._steps = steps
        self._render_script_summary()
        self._set_status(
            f"Loaded {len(self._steps)} step(s). Requires KL=30. "
            f"Will force MANUAL on start."
        )

    def _parse_steps(self, raw_steps) -> list[_ScriptStep]:
        parsed: list[_ScriptStep] = []
        for idx, item in enumerate(list(raw_steps or []), start=1):
            if not isinstance(item, dict):
                continue
            duration_s = _safe_float(item.get("duration_s"))
            if duration_s is None or duration_s <= 0.0:
                continue

            speed_wire = _safe_int(item.get("speed_x10"))
            if speed_wire is None:
                speed_cms = _safe_float(item.get("speed_cms"))
                speed_wire = 0 if speed_cms is None else int(round(speed_cms * 10.0))

            steer_wire = _safe_int(item.get("steer_x10"))
            if steer_wire is None:
                steer_deg = _safe_float(item.get("steer_deg"))
                steer_wire = 0 if steer_deg is None else int(round(steer_deg * 10.0))

            label = str(item.get("label") or f"Step {idx}")
            parsed.append(
                _ScriptStep(
                    label=label,
                    duration_s=float(duration_s),
                    speed_wire=int(speed_wire),
                    steer_wire=int(steer_wire),
                )
            )
        return parsed

    def _render_script_summary(self) -> None:
        self._name_label.setText(self._script_name)
        self._desc_label.setText(
            self._script_description
            or "Time-based raw speed/steer route loaded from the repo."
        )
        if not self._steps:
            self._step_label.setText("No steps loaded.")
            return
        total_s = sum(step.duration_s for step in self._steps)
        self._step_label.setText(
            f"Loaded {len(self._steps)} step(s), total {total_s:.1f}s. "
            f"First: {self._steps[0].label} "
            f"({self._steps[0].speed_cms:.1f} cm/s, {self._steps[0].steer_deg:+.1f} deg)"
        )

    def _on_actuator_status(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        klem_mode = payload.get("klem_mode")
        if klem_mode is not None:
            try:
                self._klem_mode = int(klem_mode)
            except (TypeError, ValueError):
                self._klem_mode = 0
        self._engine_enabled = bool(payload.get("engine_enabled"))
        if self._running and (self._klem_mode != 30 or not self._engine_enabled):
            self._finish_route("KL/engine dropped during scripted route.")

    def _on_state_change(self, mode: str) -> None:
        self._current_mode = str(mode or "")

    def _start_route(self) -> None:
        if self._running or self._pending_start:
            return
        if not self._steps:
            self._set_status("No route steps loaded.")
            return
        if self._klem_mode != 30 or not self._engine_enabled:
            self._set_status("Scripted route requires KL=30 and engine enabled.")
            return

        print(
            f"\033[1;97m[ ScriptedRoutePad ] :\033[0m \033[1;96mSTART\033[0m"
            f" name={self._script_name!r} steps={len(self._steps)}"
        )
        self._client.emit_message(ev.CMD_DRIVING_MODE, ev.MODE_MANUAL)
        self._client.emit_message(ev.CMD_SPEED, "0")
        self._client.emit_message(ev.CMD_STEER, "0")

        self._pending_start = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._set_status("Arming MANUAL and starting scripted route...")
        self._step_timer.start(int(round(_START_DELAY_S * 1000.0)))
        self._active_step_idx = -1
        self._status_timer.start()

    def _advance_step(self) -> None:
        if self._pending_start:
            self._pending_start = False
            self._running = True
            next_idx = 0
        else:
            next_idx = self._active_step_idx + 1

        if next_idx >= len(self._steps):
            self._finish_route("Scripted route completed.")
            return

        step = self._steps[next_idx]
        self._active_step_idx = next_idx
        self._step_started_monotonic = time.monotonic()
        self._step_deadline_monotonic = self._step_started_monotonic + step.duration_s

        print(
            f"\033[1;97m[ ScriptedRoutePad ] :\033[0m \033[1;96mSTEP\033[0m"
            f" idx={next_idx + 1}/{len(self._steps)}"
            f" label={step.label!r}"
            f" speed_wire={step.speed_wire} steer_wire={step.steer_wire}"
            f" duration_s={step.duration_s:.2f}"
        )
        self._client.emit_message(ev.CMD_SPEED, str(step.speed_wire))
        self._client.emit_message(ev.CMD_STEER, str(step.steer_wire))
        self._refresh_running_status()
        self._step_timer.start(max(1, int(round(step.duration_s * 1000.0))))

    def _refresh_running_status(self) -> None:
        if self._pending_start:
            self._step_label.setText("Preparing MANUAL mode...")
            return
        if not self._running or not (0 <= self._active_step_idx < len(self._steps)):
            return
        step = self._steps[self._active_step_idx]
        remaining_s = max(0.0, self._step_deadline_monotonic - time.monotonic())
        self._step_label.setText(
            f"Step {self._active_step_idx + 1}/{len(self._steps)} — {step.label} | "
            f"speed={step.speed_cms:.1f} cm/s steer={step.steer_deg:+.1f} deg | "
            f"remaining={remaining_s:.1f}s"
        )
        self._set_status("Running scripted route (open-loop, sensors ignored by design).")

    def _stop_route(self) -> None:
        if not self._running and not self._pending_start:
            return
        self._finish_route("Scripted route stopped by operator.")

    def _finish_route(self, reason: str) -> None:
        was_running = self._running or self._pending_start
        self._running = False
        self._pending_start = False
        self._step_timer.stop()
        self._status_timer.stop()
        self._active_step_idx = -1
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._client.emit_message(ev.CMD_STEER, "0")
        self._client.emit_message(ev.CMD_SPEED, "0")
        if self._final_brake:
            self._client.emit_message(ev.CMD_BRAKE, "0")
        self._render_script_summary()
        self._set_status(reason)
        if was_running:
            print(
                f"\033[1;97m[ ScriptedRoutePad ] :\033[0m \033[1;96mSTOP\033[0m"
                f" reason={reason}"
            )

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)
