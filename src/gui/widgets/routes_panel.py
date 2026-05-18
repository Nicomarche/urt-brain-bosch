"""Saved navigation routes — destination lists persisted to JSON.

A "route" is just a named ordered list of node IDs (or x/y coordinates)
that the operator wants to send the car along. Selecting a route and
clicking "Run" emits ``NavigationCommand`` events for each waypoint;
the brain handles the actual planning between them.

Persistence: ``config/routes.json`` (one of the new versioned configs).

Schema::

    {
      "saved_routes": [
        {"name": "Route 1", "waypoints": [{"x": 1.2, "y": 3.4}, ...]},
        ...
      ]
    }

The panel lets you create / delete / rename routes and add waypoints by
hand. Future work: clicking on the map will drop waypoints into the
currently-selected route.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..client import events as ev
from ..client.zmq_bus_client import ZmqBusClient as SocketIOClient
from ..config import persistence


class RoutesPanel(QWidget):
    """Two-column UI: route list (left) + waypoints of the selected route (right)."""

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._client = client

        loaded = persistence.load_config("routes")
        self._routes: list[dict] = list(loaded.get("saved_routes") or [])

        # ---- Left: route list + buttons ----
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_route_selected)

        list_buttons = QHBoxLayout()
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self._new_route)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self._delete_route)
        list_buttons.addWidget(new_btn)
        list_buttons.addWidget(del_btn)
        list_buttons.addStretch(1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self._list, 1)
        left_layout.addLayout(list_buttons)

        # ---- Right: waypoints of selected route ----
        self._waypoints = QTableWidget(0, 2)
        self._waypoints.setHorizontalHeaderLabels(["x", "y"])
        self._waypoints.verticalHeader().setVisible(True)

        wp_buttons = QHBoxLayout()
        add_wp = QPushButton("Add waypoint")
        add_wp.clicked.connect(self._add_waypoint)
        del_wp = QPushButton("Remove waypoint")
        del_wp.clicked.connect(self._remove_waypoint)
        run_btn = QPushButton("Run route")
        run_btn.setStyleSheet("font-weight:600;")
        run_btn.clicked.connect(self._run_route)
        save_btn = QPushButton("Save all")
        save_btn.clicked.connect(self._save)
        wp_buttons.addWidget(add_wp)
        wp_buttons.addWidget(del_wp)
        wp_buttons.addStretch(1)
        wp_buttons.addWidget(run_btn)
        wp_buttons.addWidget(save_btn)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self._waypoints, 1)
        right_layout.addLayout(wp_buttons)

        # ---- Splitter ----
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(splitter)

        self._refresh_list()

    # ------------------------------------------------------------------
    def _refresh_list(self) -> None:
        self._list.clear()
        for route in self._routes:
            self._list.addItem(QListWidgetItem(route.get("name", "(unnamed)")))

    def _current_route(self) -> Optional[dict]:
        row = self._list.currentRow()
        if 0 <= row < len(self._routes):
            return self._routes[row]
        return None

    def _on_route_selected(self, _row: int) -> None:
        route = self._current_route()
        self._waypoints.setRowCount(0)
        if not route:
            return
        for wp in route.get("waypoints", []):
            self._add_waypoint_row(wp.get("x", 0.0), wp.get("y", 0.0))

    def _add_waypoint_row(self, x: float, y: float) -> None:
        row = self._waypoints.rowCount()
        self._waypoints.insertRow(row)
        self._waypoints.setItem(row, 0, QTableWidgetItem(f"{x:.3f}"))
        self._waypoints.setItem(row, 1, QTableWidgetItem(f"{y:.3f}"))

    # ------------------------------------------------------------------
    def _new_route(self) -> None:
        name, ok = QInputDialog.getText(self, "New route", "Route name:")
        if not ok or not name.strip():
            return
        self._routes.append({"name": name.strip(), "waypoints": []})
        self._refresh_list()
        self._list.setCurrentRow(len(self._routes) - 1)

    def _delete_route(self) -> None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._routes):
            return
        if QMessageBox.question(
            self, "Delete route",
            f"Delete '{self._routes[row].get('name', '')}'?"
        ) != QMessageBox.Yes:
            return
        del self._routes[row]
        self._refresh_list()

    def _add_waypoint(self) -> None:
        route = self._current_route()
        if route is None:
            QMessageBox.information(self, "No route", "Create a route first.")
            return
        x, ok = QInputDialog.getDouble(self, "Add waypoint", "x:", 0.0, -100.0, 100.0, 3)
        if not ok:
            return
        y, ok = QInputDialog.getDouble(self, "Add waypoint", "y:", 0.0, -100.0, 100.0, 3)
        if not ok:
            return
        route.setdefault("waypoints", []).append({"x": x, "y": y})
        self._add_waypoint_row(x, y)

    def _remove_waypoint(self) -> None:
        route = self._current_route()
        if route is None:
            return
        row = self._waypoints.currentRow()
        if row < 0 or row >= len(route.get("waypoints", [])):
            return
        del route["waypoints"][row]
        self._waypoints.removeRow(row)

    def _run_route(self) -> None:
        route = self._current_route()
        if not route or not route.get("waypoints"):
            QMessageBox.information(self, "Empty route", "Add at least one waypoint.")
            return
        # Send each waypoint as a separate NavigationCommand. The brain
        # decides how to chain them (it has its own planner queue).
        for wp in route["waypoints"]:
            try:
                payload = {"x": float(wp["x"]), "y": float(wp["y"])}
            except (TypeError, ValueError, KeyError):
                continue
            self._client.emit_message(ev.CMD_NAV_COMMAND, payload)

    def _save(self) -> None:
        # Pull the current cell values into the model first (in case the
        # user has been editing inline without picking a different row).
        route = self._current_route()
        if route is not None:
            new_wps: list[dict] = []
            for r in range(self._waypoints.rowCount()):
                try:
                    x = float(self._waypoints.item(r, 0).text())
                    y = float(self._waypoints.item(r, 1).text())
                except (AttributeError, ValueError):
                    continue
                new_wps.append({"x": x, "y": y})
            route["waypoints"] = new_wps
        persistence.save_config("routes", {"saved_routes": self._routes})
        QMessageBox.information(self, "Saved", "Routes saved to config/routes.json.")
