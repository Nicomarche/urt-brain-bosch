"""Calibration wizard — drive the backend's ``Calibration`` state machine.

The backend at ``src/gui/components/calibration.py`` owns the actual
calibration sequence (steering left → right → backward → test run →
polynomial fit → save). We just talk to it over SocketIO using the
existing protocol, exactly like the Angular ``calibration.component`` did:

Outgoing (wrapped in ``message`` event with ``Name="Calibration"``)::

    Value = {"Action": "start"}                         # begin
    Value = {"Action": "run", "Direction": "left"}      # one PWM run
    Value = {"Action": "current_angle", "Direction": …} # what step we're on
    Value = {"Action": "submit_measurements",
             "Direction": "left",
             "Distances": {"d1": …, "d2": …, "d3": …}}  # what the operator measured
    Value = {"Action": "test_run"}
    Value = {"Action": "test_run_done"}
    Value = {"Action": "save_calibration"}              # → server emits zip
    Value = {"Action": "get_status"}                    # snapshot
    Value = {"Action": "get_polynomial_data"}           # for the chart

Incoming on the ``Calibration`` event (dict with ``action`` key)::

    {action: "calibration_status", left/right/test_run/backward: bool}
    {action: "current_angle", data: <degrees>}
    {action: "current_speed", data: <mm/s>}      # backward only
    {action: "calibration_run_done"}
    {action: "test_run_done"}
    {action: "polynomial_data", speedData/steerData/limitPointsData}
    {action: "calibration_saved", success, zipData (base64)}

The flow is **stepped** rather than one-shot: per direction, the operator
hits "Run", waits for ``calibration_run_done`` to come back, *measures*
the wheel deflection by hand, types the three distances, submits, and
moves to the next PWM step. The wizard counts 7 PWM steps per direction
(matching ``commands_template`` in the backend).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..client import events as ev
from ..client.socketio_client import SocketIOClient
from ..config import persistence, settings


_logger = logging.getLogger(__name__)

# pyqtgraph is optional (heavy import); the polynomial step degrades to a
# text dump if it isn't available.
try:
    import pyqtgraph as pg  # type: ignore
    _PG_OK = True
except Exception:  # pragma: no cover - dev-time absence is fine
    pg = None  # type: ignore
    _PG_OK = False


# Number of steps per direction in the backend ``commands_template``.
# Hard-coded here so we can show progress without needing a backend
# round-trip; if the template ever changes, update this constant.
STEPS_PER_DIRECTION = 7
BACKWARD_STEPS = 5


# ----------------------------------------------------------------------
# Step 0 — placement / safety briefing
# ----------------------------------------------------------------------
class _PlacementStep(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        title = QLabel("Calibration — placement")
        title.setStyleSheet("font-size:18pt; font-weight:600;")
        layout.addWidget(title, alignment=Qt.AlignCenter)
        body = QLabel(
            "1. Place the car on the calibration mat with the front "
            "wheels on the marked starting line.\n"
            "2. Make sure the rear wheels are clear and the car can "
            "swing freely without hitting anything.\n"
            "3. Have a ruler handy: you'll be measuring three "
            "displacement distances per run.\n\n"
            "Press Start to put the backend into calibration mode."
        )
        body.setWordWrap(True)
        body.setStyleSheet("font-size:11pt; color:#ccc;")
        layout.addWidget(body)


# ----------------------------------------------------------------------
# Step 1/2/3 — drive a direction (left/right/backward)
# ----------------------------------------------------------------------
class _RunDirectionStep(QWidget):
    """One panel reused for left/right/backward runs.

    Shows: which step we're on (1..N), the desired angle/speed, a "Run"
    button to fire that step on the car, then form fields for the
    operator to type the measured distances and submit.
    """

    def __init__(self, client: SocketIOClient, direction: str):
        super().__init__()
        self._client = client
        self._direction = direction
        self._is_backward = direction == "backward"
        self._total_steps = BACKWARD_STEPS if self._is_backward else STEPS_PER_DIRECTION
        self._step = 0
        self._target_value: float = 0.0  # current desired steer/speed
        self._completed = False

        # ---- Layout ----
        layout = QVBoxLayout(self)

        title = QLabel(f"Run — {direction.capitalize()}")
        title.setStyleSheet("font-size:16pt; font-weight:600;")
        layout.addWidget(title)

        self._progress = QProgressBar()
        self._progress.setRange(0, self._total_steps)
        self._progress.setFormat("%v / %m steps")
        layout.addWidget(self._progress)

        self._target_label = QLabel("Target: —")
        self._target_label.setStyleSheet("font-size:13pt; color:#5af;")
        layout.addWidget(self._target_label)

        # Run button strip.
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run step")
        self._run_btn.clicked.connect(self._run_step)
        self._rerun_btn = QPushButton("Re-run last step")
        self._rerun_btn.clicked.connect(self._rerun_last_step)
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._rerun_btn)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        # Measurement form. For left/right we want d1/d2/d3 (mm); for
        # backward, a single distance d.
        form_box = QGroupBox("Measurements (mm)")
        form = QFormLayout(form_box)
        if self._is_backward:
            self._d  = self._spin()
            form.addRow("Distance d", self._d)
        else:
            self._d1 = self._spin()
            self._d2 = self._spin()
            self._d3 = self._spin()
            form.addRow("d1", self._d1)
            form.addRow("d2", self._d2)
            form.addRow("d3", self._d3)
        layout.addWidget(form_box)

        self._submit_btn = QPushButton("Submit measurement")
        self._submit_btn.clicked.connect(self._submit_measurement)
        self._submit_btn.setEnabled(False)
        layout.addWidget(self._submit_btn)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#aaa;")
        layout.addWidget(self._status)
        layout.addStretch(1)

        # Backend signals.
        client.calibration_signal.connect(self._on_calibration)

    # ------------------------------------------------------------------
    @staticmethod
    def _spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-2000, 2000)
        s.setDecimals(1)
        s.setSingleStep(1.0)
        s.setSuffix(" mm")
        return s

    def reset(self) -> None:
        self._step = 0
        self._completed = False
        self._progress.setValue(0)
        self._target_label.setText("Target: —")
        self._submit_btn.setEnabled(False)
        self._run_btn.setEnabled(True)
        self._set_status("Ready — press Run to start.")

    # ------------------------------------------------------------------
    def _run_step(self) -> None:
        if self._completed:
            self._set_status("Direction already completed.")
            return
        # Ask the backend what the upcoming desired angle/speed is, then
        # dispatch the run. The backend echoes the value back via
        # `current_angle` / `current_speed`.
        self._client.emit_message(ev.CMD_CALIBRATION, {
            "Action": "current_angle",
            "Direction": self._direction,
        })
        self._client.emit_message(ev.CMD_CALIBRATION, {
            "Action": "run",
            "Direction": self._direction,
        })
        self._set_status(f"Running step {self._step + 1}/{self._total_steps}…")
        self._run_btn.setEnabled(False)
        self._submit_btn.setEnabled(False)

    def _rerun_last_step(self) -> None:
        self._client.emit_message(ev.CMD_CALIBRATION, {"Action": "re-run"})
        if self._step > 0:
            self._step -= 1
            self._progress.setValue(self._step)
        self._set_status("Backed up one step.")

    def _submit_measurement(self) -> None:
        if self._is_backward:
            distances = {"d": self._d.value()}
        else:
            distances = {
                "d1": self._d1.value(),
                "d2": self._d2.value(),
                "d3": self._d3.value(),
            }
        self._client.emit_message(ev.CMD_CALIBRATION, {
            "Action": "submit_measurements",
            "Direction": self._direction,
            "Distances": distances,
        })
        self._set_status("Measurement submitted.")
        self._submit_btn.setEnabled(False)
        if self._step >= self._total_steps:
            self._completed = True
            self._run_btn.setEnabled(False)
            self._set_status("Direction complete — move to the next step.")

    # ------------------------------------------------------------------
    def _on_calibration(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        action = payload.get("action")
        if action == "current_angle":
            self._target_value = float(payload.get("data", 0.0) or 0.0)
            self._target_label.setText(f"Target steer: {self._target_value:.1f}°")
        elif action == "current_speed":
            self._target_value = float(payload.get("data", 0.0) or 0.0)
            self._target_label.setText(f"Target speed: {self._target_value:.0f} mm/s")
        elif action == "calibration_run_done":
            # Step finished on the car; let the operator measure.
            self._step += 1
            self._progress.setValue(self._step)
            self._submit_btn.setEnabled(True)
            self._run_btn.setEnabled(self._step < self._total_steps)
            self._set_status(
                f"Step {self._step}/{self._total_steps} complete — "
                f"measure and submit."
            )

    def _set_status(self, msg: str) -> None:
        self._status.setText(msg)


# ----------------------------------------------------------------------
# Step 4 — test run (validate offset)
# ----------------------------------------------------------------------
class _TestRunStep(QWidget):
    def __init__(self, client: SocketIOClient):
        super().__init__()
        self._client = client

        layout = QVBoxLayout(self)
        title = QLabel("Test run")
        title.setStyleSheet("font-size:16pt; font-weight:600;")
        layout.addWidget(title)

        body = QLabel(
            "After the left and right runs, drive the car forward with "
            "steer = 0 to see how it tracks. The backend will fit a cubic "
            "spline to your earlier measurements and compute a steering "
            "offset to compensate."
        )
        body.setWordWrap(True)
        body.setStyleSheet("color:#bbb;")
        layout.addWidget(body)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run test")
        self._run_btn.clicked.connect(self._run_test)
        self._done_btn = QPushButton("Mark test complete")
        self._done_btn.clicked.connect(self._mark_done)
        self._done_btn.setEnabled(False)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._done_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#aaa;")
        layout.addWidget(self._status)
        layout.addStretch(1)

        client.calibration_signal.connect(self._on_calibration)

    def _run_test(self) -> None:
        self._client.emit_message(ev.CMD_CALIBRATION, {"Action": "test_run"})
        self._status.setText("Test run dispatched — watch the car.")
        self._run_btn.setEnabled(False)

    def _mark_done(self) -> None:
        self._client.emit_message(ev.CMD_CALIBRATION, {"Action": "test_run_done"})
        self._status.setText("Test run accepted.")
        self._done_btn.setEnabled(False)

    def _on_calibration(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        action = payload.get("action")
        if action == "calibration_run_done":
            self._status.setText("Test run finished — confirm or re-run.")
            self._run_btn.setEnabled(True)
            self._done_btn.setEnabled(True)
        elif action == "test_run_done":
            self._status.setText("Test run confirmed.")
            self._done_btn.setEnabled(False)


# ----------------------------------------------------------------------
# Step 5 — polynomial visualisation
# ----------------------------------------------------------------------
class _PolynomialStep(QWidget):
    """Plot of the calibration data + cubic spline fit.

    Backend payload ``polynomial_data`` shape::

        {
          action: "polynomial_data",
          hasData: bool,
          speedData:  {points: [[pwm, mm/s], ...], spline: {...}, error: ...},
          steerData:  {points: [[pwm, deg],  ...], spline: {...}, error: ...},
          limitPointsData: {points: [[pwm, deg], ...], type: "limit"}
        }
    """

    def __init__(self, client: SocketIOClient):
        super().__init__()
        self._client = client

        layout = QVBoxLayout(self)
        title = QLabel("Polynomial fit")
        title.setStyleSheet("font-size:16pt; font-weight:600;")
        layout.addWidget(title)

        refresh = QPushButton("Reload data")
        refresh.clicked.connect(self._request_data)
        layout.addWidget(refresh, alignment=Qt.AlignLeft)

        if _PG_OK:
            self._speed_plot = pg.PlotWidget(title="Speed: PWM → mm/s")
            self._steer_plot = pg.PlotWidget(title="Steer: PWM → degrees")
            for p in (self._speed_plot, self._steer_plot):
                p.setBackground("#101010")
                p.showGrid(x=True, y=True, alpha=0.3)
            grid = QGridLayout()
            grid.addWidget(self._speed_plot, 0, 0)
            grid.addWidget(self._steer_plot, 0, 1)
            layout.addLayout(grid, 1)
        else:
            self._fallback = QTextEdit()
            self._fallback.setReadOnly(True)
            self._fallback.setPlaceholderText(
                "(pyqtgraph not available — payload is shown as JSON)")
            layout.addWidget(self._fallback, 1)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color:#888;")
        layout.addWidget(self._error_label)

        client.calibration_signal.connect(self._on_calibration)

    def _request_data(self) -> None:
        self._client.emit_message(ev.CMD_CALIBRATION, {"Action": "get_polynomial_data"})
        self._client.emit_message(ev.CMD_CALIBRATION, {"Action": "get_zero_offset_spline_data"})

    def _on_calibration(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("action") != "polynomial_data":
            return
        if _PG_OK:
            self._render_plots(payload)
        else:
            import json
            self._fallback.setText(json.dumps(payload, indent=2))
        if not payload.get("hasData", False):
            self._error_label.setText("Not enough data yet — finish a few runs first.")
        else:
            self._error_label.setText("")

    def _render_plots(self, payload) -> None:
        # ``speedData`` / ``steerData`` are independent; render whichever
        # are present. Each holds raw points (scatter) + spline curve.
        for plot, key, point_color in (
            (self._speed_plot, "speedData", (0, 200, 255)),
            (self._steer_plot, "steerData", (255, 200, 0)),
        ):
            plot.clear()
            block = payload.get(key)
            if not isinstance(block, dict):
                continue
            pts = block.get("points") or []
            if pts:
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                plot.plot(
                    xs, ys, pen=None, symbol="o",
                    symbolBrush=pg.mkBrush(*point_color),
                )
            spline = block.get("spline")
            if isinstance(spline, dict):
                xs2, ys2 = self._eval_spline(spline)
                if xs2:
                    plot.plot(xs2, ys2, pen=pg.mkPen(*point_color, width=2))
        # Limit markers (only relevant for steer).
        limits = payload.get("limitPointsData", {}).get("points") or []
        if limits and _PG_OK:
            xs = [float(p[0]) for p in limits]
            ys = [float(p[1]) for p in limits]
            self._steer_plot.plot(
                xs, ys, pen=None, symbol="x",
                symbolBrush=pg.mkBrush(255, 80, 80), symbolSize=14,
            )

    @staticmethod
    def _eval_spline(spline: dict, samples_per_segment: int = 20):
        """Evaluate the backend's CubicSpline payload to (x, y) lists.

        The payload stores ``knots`` and ``coefficients`` (cs.c.T) — same
        shape SciPy uses, so we recreate the polynomial for each segment
        and sample it densely.
        """
        knots = spline.get("knots") or []
        coeffs = spline.get("coefficients") or []
        if len(knots) < 2 or len(coeffs) != len(knots) - 1:
            return [], []
        xs: list[float] = []
        ys: list[float] = []
        for i, (x0, x1) in enumerate(zip(knots[:-1], knots[1:])):
            seg = coeffs[i]
            # CubicSpline.c.T row layout: [a, b, c, d] for
            # y(x) = a*(x-x0)^3 + b*(x-x0)^2 + c*(x-x0) + d
            a, b, c, d = (seg + [0, 0, 0, 0])[:4]
            for s in range(samples_per_segment):
                t = s / samples_per_segment
                x = x0 + t * (x1 - x0)
                dx = x - x0
                y = a * dx**3 + b * dx**2 + c * dx + d
                xs.append(x)
                ys.append(y)
        # Append the very last knot for a closed line.
        xs.append(float(knots[-1]))
        # Evaluate the last segment at its right endpoint.
        if coeffs:
            a, b, c, d = (coeffs[-1] + [0, 0, 0, 0])[:4]
            dx = knots[-1] - knots[-2]
            ys.append(a * dx**3 + b * dx**2 + c * dx + d)
        return xs, ys


# ----------------------------------------------------------------------
# Step 6 — save (download zip)
# ----------------------------------------------------------------------
class _SaveStep(QWidget):
    def __init__(self, client: SocketIOClient):
        super().__init__()
        self._client = client
        layout = QVBoxLayout(self)
        title = QLabel("Save calibration")
        title.setStyleSheet("font-size:16pt; font-weight:600;")
        layout.addWidget(title)

        body = QLabel(
            "Generate the calibration source bundle. The backend fits "
            "polynomials to all collected points and returns a zip of "
            "the C++/Python sources to drop on the Nucleo / brain."
        )
        body.setWordWrap(True)
        body.setStyleSheet("color:#bbb;")
        layout.addWidget(body)

        btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save calibration")
        self._save_btn.setStyleSheet("font-weight:600;")
        self._save_btn.clicked.connect(self._save)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#aaa;")
        layout.addWidget(self._status)
        layout.addStretch(1)

        client.calibration_signal.connect(self._on_calibration)

    def _save(self) -> None:
        self._client.emit_message(ev.CMD_CALIBRATION, {"Action": "save_calibration"})
        self._status.setText("Requested — waiting for backend zip…")
        self._save_btn.setEnabled(False)

    def _on_calibration(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("action") != "calibration_saved":
            return
        success = bool(payload.get("success"))
        if not success:
            self._status.setText("Backend reported failure.")
            self._save_btn.setEnabled(True)
            return
        zip_b64 = payload.get("zipData")
        if not isinstance(zip_b64, str):
            self._status.setText("Saved (no zip payload returned).")
            return
        # Persist the local snapshot config too — purely a record of
        # "we calibrated on <date>" for later debugging.
        try:
            persistence.save_config("calibration", {
                "saved": True,
                "zip_size_b64": len(zip_b64),
            })
        except Exception:
            pass
        # Offer to write the zip to disk.
        default_name = "calibration_sources.zip"
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save calibration zip",
            str(settings.REPO_ROOT / default_name), "Zip archive (*.zip)"
        )
        if not path:
            self._status.setText("Save cancelled by user.")
            self._save_btn.setEnabled(True)
            return
        try:
            Path(path).write_bytes(base64.b64decode(zip_b64))
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            self._save_btn.setEnabled(True)
            return
        self._status.setText(f"Wrote {path}")
        self._save_btn.setEnabled(True)


# ----------------------------------------------------------------------
# Wizard shell
# ----------------------------------------------------------------------
class CalibrationWizard(QWidget):
    """Step-by-step calibration UI.

    Implemented as a ``QStackedWidget`` rather than a true ``QWizard``
    because the backend's protocol is interactive (some steps need
    multiple round-trips) and ``QWizard``'s "next/back" semantics fit
    poorly with that.
    """

    STEP_LABELS = (
        "1. Placement",
        "2. Left runs",
        "3. Right runs",
        "4. Backward runs",
        "5. Test run",
        "6. Polynomial fit",
        "7. Save",
    )

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._client = client
        self._current = 0
        self._build_ui()
        self._wire_signals()
        # Ask for current backend status the moment the wizard appears,
        # so the operator knows what's already done.
        self._client.emit_message(ev.CMD_CALIBRATION, {"Action": "get_status"})

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Header: numbered step indicator that highlights the current one.
        self._step_label = QLabel("")
        self._step_label.setStyleSheet("font-size:11pt; color:#888;")

        self._stack = QStackedWidget()
        self._placement = _PlacementStep()
        self._left      = _RunDirectionStep(self._client, "left")
        self._right     = _RunDirectionStep(self._client, "right")
        self._backward  = _RunDirectionStep(self._client, "backward")
        self._test_run  = _TestRunStep(self._client)
        self._poly      = _PolynomialStep(self._client)
        self._save      = _SaveStep(self._client)
        for w in (self._placement, self._left, self._right, self._backward,
                  self._test_run, self._poly, self._save):
            self._stack.addWidget(w)

        # Footer: Previous / Start / Next / Cancel buttons.
        self._prev_btn = QPushButton("◀  Previous")
        self._next_btn = QPushButton("Next  ▶")
        self._start_btn = QPushButton("Start calibration")
        self._cancel_btn = QPushButton("Cancel")
        self._prev_btn.clicked.connect(self._go_prev)
        self._next_btn.clicked.connect(self._go_next)
        self._start_btn.clicked.connect(self._start)
        self._cancel_btn.clicked.connect(self._cancel)

        button_row = QHBoxLayout()
        button_row.addWidget(self._cancel_btn)
        button_row.addStretch(1)
        button_row.addWidget(self._prev_btn)
        button_row.addWidget(self._start_btn)
        button_row.addWidget(self._next_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._step_label)
        layout.addWidget(self._stack, 1)
        layout.addLayout(button_row)

        self._refresh_step_label()

    def _wire_signals(self) -> None:
        self._client.calibration_signal.connect(self._on_calibration)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _refresh_step_label(self) -> None:
        parts = []
        for i, label in enumerate(self.STEP_LABELS):
            if i == self._current:
                parts.append(f"<b style='color:#5af;'>{label}</b>")
            else:
                parts.append(label)
        self._step_label.setText("  ·  ".join(parts))
        self._stack.setCurrentIndex(self._current)
        # Hide Start outside step 0.
        self._start_btn.setVisible(self._current == 0)
        self._prev_btn.setEnabled(self._current > 0)
        self._next_btn.setEnabled(self._current < len(self.STEP_LABELS) - 1)
        # Reset run panels when entering them so old state doesn't bleed.
        widget = self._stack.currentWidget()
        if isinstance(widget, _RunDirectionStep):
            # Tell the backend to reset its `current_step` between
            # directions (the backend uses `done`/`continue` for that).
            self._client.emit_message(ev.CMD_CALIBRATION, {"Action": "done"})
            widget.reset()

    def _go_prev(self) -> None:
        if self._current > 0:
            self._current -= 1
            self._refresh_step_label()

    def _go_next(self) -> None:
        if self._current < len(self.STEP_LABELS) - 1:
            self._current += 1
            self._refresh_step_label()
            # Pre-fetch the polynomial data when arriving on that step.
            if self._stack.currentWidget() is self._poly:
                self._poly._request_data()

    def _start(self) -> None:
        self._client.emit_message(ev.CMD_CALIBRATION, {"Action": "start"})
        self._go_next()

    def _cancel(self) -> None:
        if QMessageBox.question(
            self, "Cancel calibration",
            "Stop the calibration session? Progress on the backend "
            "will be reset."
        ) != QMessageBox.Yes:
            return
        self._client.emit_message(ev.CMD_CALIBRATION, {"Action": "exit"})
        self._current = 0
        self._refresh_step_label()

    # ------------------------------------------------------------------
    # Backend status
    # ------------------------------------------------------------------
    def _on_calibration(self, payload) -> None:
        # Only react to the snapshot here; per-step events live on the
        # individual panels.
        if not isinstance(payload, dict):
            return
        if payload.get("action") == "calibration_status":
            done = []
            if payload.get("left"):     done.append("left")
            if payload.get("right"):    done.append("right")
            if payload.get("test_run"): done.append("test")
            if payload.get("backward"): done.append("backward")
            if done:
                _logger.info("Calibration already done: %s", ", ".join(done))
