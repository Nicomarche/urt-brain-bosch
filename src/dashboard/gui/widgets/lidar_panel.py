from __future__ import annotations

import math
import time
from typing import Any

try:
    import config
except Exception:  # pragma: no cover - dashboard can still render with defaults.
    config = None

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..client.socketio_client import SocketIOClient


class LidarPanel(QWidget):
    """LiDAR health, sector distances, obstacle table and compact log."""

    def __init__(self, client: SocketIOClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._scan: dict[str, Any] | None = None
        self._status: dict[str, Any] | None = None
        self._detected: dict[str, Any] | None = None
        self._tracked: dict[str, Any] | None = None
        self._last_log_t = 0.0
        self._last_signature = ""

        self._build_ui()
        client.lidar_scan_signal.connect(self._on_scan)
        client.lidar_status_signal.connect(self._on_status)
        client.detected_objects_signal.connect(self._on_detected)
        client.tracked_objects_signal.connect(self._on_tracked)

    def _build_ui(self) -> None:
        self.setStyleSheet(
            "QWidget { background: #f7f8fa; color: #111; }"
            "QGroupBox { color: #111; font-weight: 600; border: 1px solid #cfd5dd; "
            "border-radius: 6px; margin-top: 10px; padding: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            "QLabel { color: #111; }"
            "QTableWidget { background: #ffffff; color: #111; gridline-color: #d9dee7; }"
            "QHeaderView::section { background: #e9edf3; color: #111; padding: 4px; }"
            "QTextEdit { background: #ffffff; color: #111; border: 1px solid #cfd5dd; }"
            "QPushButton { color: #111; background: #e8edf5; border: 1px solid #b8c0cc; "
            "border-radius: 4px; padding: 5px 10px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        health = QGroupBox("Health")
        grid = QGridLayout(health)
        self._health_labels: dict[str, QLabel] = {}
        labels = [
            "status", "backend", "enabled", "scan_hz", "points", "obstacles",
            "ai_objects", "age", "crc_errors", "resyncs", "source", "frame", "range", "detail",
        ]
        for idx, key in enumerate(labels):
            title = QLabel(f"{key}:")
            value = QLabel("-")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._health_labels[key] = value
            row, col = divmod(idx, 4)
            grid.addWidget(title, row, col * 2)
            grid.addWidget(value, row, col * 2 + 1)
        layout.addWidget(health)

        sectors = QGroupBox("Nearest Sectors")
        sgrid = QGridLayout(sectors)
        self._sector_labels: dict[str, QLabel] = {}
        for idx, name in enumerate(("front +/-10 deg", "left +45 deg", "right -45 deg")):
            title = QLabel(name)
            value = QLabel("-")
            self._sector_labels[name] = value
            sgrid.addWidget(title, 0, idx * 2)
            sgrid.addWidget(value, 0, idx * 2 + 1)
        layout.addWidget(sectors)

        self._table = QTableWidget(0, 11)
        self._table.setHorizontalHeaderLabels([
            "#", "src", "class", "dist m", "angle deg", "x m", "y m",
            "width deg", "pts", "conf", "world xy",
        ])
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(False)
        layout.addWidget(self._table, 1)

        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Log"))
        clear_btn = QPushButton("Clear log")
        clear_btn.clicked.connect(self._clear_log)
        log_header.addStretch(1)
        log_header.addWidget(clear_btn)
        layout.addLayout(log_header)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(110)
        layout.addWidget(self._log)

    def _on_scan(self, payload: object) -> None:
        if isinstance(payload, dict):
            self._scan = payload
            self._refresh()

    def _on_status(self, payload: object) -> None:
        if isinstance(payload, dict):
            self._status = payload
            self._refresh()

    def _on_detected(self, payload: object) -> None:
        if isinstance(payload, dict):
            self._detected = payload
            self._refresh()

    def _on_tracked(self, payload: object) -> None:
        if isinstance(payload, dict):
            self._tracked = payload
            self._refresh()

    def _refresh(self) -> None:
        scan = self._scan or {}
        status = self._status or {}
        now = time.time()
        scan_ts = _float(scan.get("timestamp"))
        age = (now - scan_ts) if scan_ts else _float(status.get("age_s"))
        fresh = age is not None and age < 0.5
        status_text = str(status.get("status", "-"))
        if fresh and status_text in {"waiting_for_scan", "-", ""}:
            status_text = "ok (scan fresh)"

        ai_objects = list((self._detected or {}).get("objects", []) or [])
        self._set_health("status", status_text)
        self._set_health("backend", status.get("backend", "-"))
        self._set_health("enabled", status.get("enabled", "-"))
        self._set_health("scan_hz", f"{_float(status.get('scan_hz'), 0.0):.1f}")
        self._set_health("points", len(scan.get("ranges_m", []) or []) or status.get("points", 0))
        self._set_health("obstacles", len(scan.get("obstacles", []) or []) or status.get("obstacles", 0))
        self._set_health("ai_objects", len(ai_objects))
        self._set_health("age", f"{age:.2f}s" if age is not None else "-")
        self._set_health("crc_errors", status.get("crc_errors", 0))
        self._set_health("resyncs", status.get("resyncs", 0))
        self._set_health("source", scan.get("source", status.get("source", "-")))
        self._set_health("frame", scan.get("frame_id", "-"))
        self._set_health("range", f"{scan.get('range_min_m', '-')}-{scan.get('range_max_m', '-')} m")
        self._set_health("detail", status.get("detail", ""))

        self._update_sectors(scan)
        rows = self._build_rows(scan, ai_objects)
        self._fill_table(rows)
        self._maybe_log(scan, rows, ai_objects)

    def _set_health(self, key: str, value: object) -> None:
        label = self._health_labels.get(key)
        if label is not None:
            label.setText(str(value))

    def _update_sectors(self, scan: dict[str, Any]) -> None:
        sectors = {
            "front +/-10 deg": (0.0, math.radians(10.0)),
            "left +45 deg": (math.radians(45.0), math.radians(8.0)),
            "right -45 deg": (math.radians(-45.0), math.radians(8.0)),
        }
        for name, (center, half) in sectors.items():
            dist = _nearest_in_sector(scan, center, half)
            self._sector_labels[name].setText(f"{dist:.2f} m" if dist is not None else "-")

    def _build_rows(self, scan: dict[str, Any], ai_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = [_row_from_obstacle(obs) for obs in (scan.get("obstacles", []) or []) if isinstance(obs, dict)]
        if not rows and scan.get("ranges_m"):
            rows = _raw_rows_from_scan(scan)
        for ai in ai_objects:
            if not isinstance(ai, dict):
                continue
            ai_row = _row_from_ai(ai)
            match = _find_match(rows, ai_row)
            if match is None:
                rows.append(ai_row)
                continue
            source = str(match.get("src", "raw"))
            match["src"] = f"{source}+ai" if not source.endswith("+ai") else source
            match["class"] = ai_row.get("class") or match.get("class") or "object"
            if ai_row.get("conf") is not None:
                match["conf"] = max(float(match.get("conf") or 0.0), float(ai_row["conf"]))
        rows.sort(key=lambda row: _float(row.get("dist"), 999.0) or 999.0)
        return rows[:16]

    def _fill_table(self, rows: list[dict[str, Any]]) -> None:
        self._table.setRowCount(len(rows))
        for idx, row in enumerate(rows):
            values = [
                idx,
                row.get("src", "-"),
                row.get("class", "-"),
                _fmt(row.get("dist"), "{:.2f}"),
                _fmt(row.get("angle_deg"), "{:.1f}"),
                _fmt(row.get("x"), "{:.2f}"),
                _fmt(row.get("y"), "{:.2f}"),
                _fmt(row.get("width_deg"), "{:.1f}"),
                row.get("pts", "-"),
                _fmt(row.get("conf"), "{:.2f}"),
                row.get("world", "-"),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setForeground(QColor("#111111"))
                self._table.setItem(idx, col, item)
        self._table.resizeColumnsToContents()

    def _maybe_log(self, scan: dict[str, Any], rows: list[dict[str, Any]], ai_objects: list[dict[str, Any]]) -> None:
        signature = "|".join(
            f"{r.get('src')}:{r.get('class')}:{_fmt(r.get('dist'), '{:.2f}')}:{_fmt(r.get('angle_deg'), '{:.1f}')}"
            for r in rows[:4]
        )
        now = time.time()
        if signature == self._last_signature and (now - self._last_log_t) < 1.0:
            return
        self._last_signature = signature
        self._last_log_t = now
        lines = [
            f"scan source={scan.get('source', '-')} pts={len(scan.get('ranges_m', []) or [])} "
            f"obs={len(scan.get('obstacles', []) or [])} shown={len(rows)} ai={len(ai_objects)}"
        ]
        for idx, row in enumerate(rows[:3]):
            lines.append(
                f"obs[{idx}] {row.get('src')}/{row.get('class')} "
                f"d={_fmt(row.get('dist'), '{:.2f}')} angle={_fmt(row.get('angle_deg'), '{:.1f}')}"
            )
        if ai_objects:
            summary = " ".join(
                f"{obj.get('kind', 'object')}:{_float(obj.get('confidence'), 0.0):.2f}"
                for obj in ai_objects[:5]
            )
            lines.append(f"ai objects={len(ai_objects)} {summary}")
        self._append_log("\n".join(lines))

    def _append_log(self, text: str) -> None:
        self._log.append(text)
        doc = self._log.document()
        if doc.blockCount() > 220:
            cursor = self._log.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def _clear_log(self) -> None:
        self._log.clear()


def _row_from_obstacle(obs: dict[str, Any]) -> dict[str, Any]:
    world_xy = obs.get("world_xy")
    return {
        "src": obs.get("source", "cluster"),
        "class": obs.get("class_name", "obstacle"),
        "dist": _float(obs.get("distance_min_m")),
        "angle_deg": math.degrees(_float(obs.get("angle_center_rad"), 0.0) or 0.0),
        "x": _float(obs.get("x_m")),
        "y": _float(obs.get("y_m")),
        "width_deg": math.degrees(_float(obs.get("width_rad"), 0.0) or 0.0),
        "pts": int(_float(obs.get("point_count"), 0.0) or 0),
        "conf": _float(obs.get("confidence")),
        "world": _world_text(world_xy),
    }


def _row_from_ai(ai: dict[str, Any]) -> dict[str, Any]:
    distance = _float(ai.get("distance_m"))
    angle = _float(ai.get("angle_rad"))
    source = "ai+lidar" if ai.get("distance_source") == "lidar" else "ai"
    return {
        "src": source,
        "class": ai.get("kind", "object"),
        "dist": distance,
        "angle_deg": math.degrees(angle) if angle is not None else None,
        "x": distance * math.cos(angle) if distance is not None and angle is not None else None,
        "y": distance * math.sin(angle) if distance is not None and angle is not None else None,
        "width_deg": None,
        "pts": "-",
        "conf": _float(ai.get("confidence")),
        "world": _world_text(ai.get("world_xy")),
    }


def _raw_rows_from_scan(scan: dict[str, Any]) -> list[dict[str, Any]]:
    ranges = scan.get("ranges_m", []) or []
    angle_min = _float(scan.get("angle_min_rad"), -math.pi) or -math.pi
    step = _float(scan.get("angle_step_rad"), (2.0 * math.pi) / max(1, len(ranges))) or 0.0
    rmin = _float(scan.get("range_min_m"), 0.05) or 0.05
    max_range = min(_float(scan.get("range_max_m"), 4.0) or 4.0, 1.20)
    min_forward_x = _cfg_float("LIDAR_OBSTACLE_MIN_FORWARD_X_M", 0.0)
    valid = []
    for idx, raw in enumerate(ranges):
        dist = _float(raw)
        if dist is None or not (rmin <= dist <= max_range):
            continue
        angle = _signed(angle_min + idx * step)
        if dist * math.cos(angle) < min_forward_x:
            continue
        valid.append((angle, dist))
    if not valid:
        return []
    valid.sort(key=lambda item: item[0])
    groups = [[valid[0]]]
    max_gap = math.radians(5.0)
    for prev, item in zip(valid, valid[1:]):
        if abs(_signed(item[0] - prev[0])) <= max_gap:
            groups[-1].append(item)
        else:
            groups.append([item])
    if len(groups) > 1 and abs(_signed(groups[0][0][0] - groups[-1][-1][0])) <= max_gap:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    rows = []
    for group in groups:
        best = min(group, key=lambda item: item[1])
        angle = _mean_angle([a for a, _ in group])
        dist = best[1]
        rows.append({
            "src": "raw",
            "class": "obstacle",
            "dist": dist,
            "angle_deg": math.degrees(angle),
            "x": dist * math.cos(angle),
            "y": dist * math.sin(angle),
            "width_deg": max(0.0, math.degrees(abs(_signed(group[-1][0] - group[0][0])))),
            "pts": len(group),
            "conf": min(1.0, 0.30 + 0.05 * len(group)),
            "world": "-",
        })
    rows.sort(key=lambda row: row["dist"])
    return rows[:12]


def _find_match(rows: list[dict[str, Any]], ai_row: dict[str, Any]) -> dict[str, Any] | None:
    ai_dist = _float(ai_row.get("dist"))
    ai_angle = _float(ai_row.get("angle_deg"))
    if ai_dist is None or ai_angle is None:
        return None
    max_dist = _cfg_float("LIDAR_AI_MATCH_DISTANCE_M", 0.30)
    max_angle = math.radians(_cfg_float("LIDAR_AI_MATCH_ANGLE_DEG", 28.0))
    max_xy = _cfg_float("LIDAR_AI_MATCH_XY_M", 0.28)
    min_forward_x = _cfg_float("LIDAR_AI_MATCH_MIN_FORWARD_X_M", -0.02)
    max_abs_angle = math.radians(_cfg_float("LIDAR_AI_MATCH_MAX_ABS_ANGLE_DEG", 120.0))
    max_score = _cfg_float("LIDAR_AI_MATCH_MAX_SCORE", 1.35)
    ai_angle_rad = math.radians(ai_angle)
    ai_x = _float(ai_row.get("x"))
    ai_y = _float(ai_row.get("y"))
    best_row = None
    best_score = math.inf
    for row in rows:
        if str(row.get("src", "")).startswith("ai"):
            continue
        dist = _float(row.get("dist"))
        angle = _float(row.get("angle_deg"))
        if dist is None or angle is None:
            continue
        angle_rad = math.radians(angle)
        if abs(_signed(angle_rad)) > max_abs_angle:
            continue
        row_x = _float(row.get("x"))
        row_y = _float(row.get("y"))
        if row_x is not None and row_x < min_forward_x:
            continue
        dist_delta = abs(dist - ai_dist)
        angle_delta = abs(_signed(angle_rad - ai_angle_rad))
        xy_delta = None
        if ai_x is not None and ai_y is not None and row_x is not None and row_y is not None:
            xy_delta = math.hypot(row_x - ai_x, row_y - ai_y)
        if dist_delta > max_dist and (xy_delta is None or xy_delta > max_xy):
            continue
        if angle_delta > max_angle and (xy_delta is None or xy_delta > max_xy * 0.75):
            continue
        score = 0.42 * (dist_delta / max(max_dist, 1e-6))
        score += 0.33 * (angle_delta / max(max_angle, 1e-6))
        score += 0.25 * ((xy_delta / max(max_xy, 1e-6)) if xy_delta is not None else 1.0)
        if score < best_score:
            best_score = score
            best_row = row
    return best_row if best_score <= max_score else None


def _nearest_in_sector(scan: dict[str, Any], center: float, half: float) -> float | None:
    ranges = scan.get("ranges_m", []) or []
    angle_min = _float(scan.get("angle_min_rad"), -math.pi) or -math.pi
    step = _float(scan.get("angle_step_rad"), (2.0 * math.pi) / max(1, len(ranges))) or 0.0
    rmin = _float(scan.get("range_min_m"), 0.05) or 0.05
    rmax = _float(scan.get("range_max_m"), 4.0) or 4.0
    best = None
    for idx, raw in enumerate(ranges):
        dist = _float(raw)
        if dist is None or not (rmin <= dist <= rmax):
            continue
        angle = _signed(angle_min + idx * step)
        if abs(_signed(angle - center)) <= half:
            best = dist if best is None else min(best, dist)
    return best


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _cfg_float(name: str, default: float) -> float:
    if config is None:
        return default
    return _float(getattr(config, name, default), default) or default


def _fmt(value: Any, fmt: str) -> str:
    num = _float(value)
    return fmt.format(num) if num is not None else "-"


def _signed(angle: float) -> float:
    return ((float(angle) + math.pi) % (2.0 * math.pi)) - math.pi


def _mean_angle(angles: list[float]) -> float:
    if not angles:
        return 0.0
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


def _world_text(value: object) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _float(value[0])
        y = _float(value[1])
        if x is not None and y is not None:
            return f"{x:.2f}, {y:.2f}"
    return "-"
