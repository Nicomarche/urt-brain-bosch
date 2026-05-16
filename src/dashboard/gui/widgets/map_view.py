"""Autoware-style map view — Lanelet2 OSM as source of truth + live overlays.

The visual model follows how Autoware uses RViz:

1. Vector HD map from Lanelet2 (`lanelet2_map.osm`) as the primary layer.
2. Drivable lane polygons and lane boundaries.
3. Lane centerlines / route overlays from the planner.
4. Regulatory markers (stopline / crosswalk / intersection / roundabout).
5. Ego vehicle footprint + raw GPS/debug overlays.

"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import time as _time

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..client import events as ev
from ..client.socketio_client import SocketIOClient
from ..config import persistence, settings
from ._map_path_overlay import (
    extract_control_path_points,
    extract_nav_route_preview_points,
)
from ._map_click_routing import resolve_click_destination_lanelet
from ._pannable_view import PannableZoomableView
from src.routing.lanelet.attributes import (
    ATTR_CROSSWALK,
    ATTR_INTERSECTION,
    ATTR_ROUNDABOUT,
    ATTR_STOPLINE,
)
from src.routing.lanelet.from_osm import parse_lanelet2_osm, load_lanelet2_osm
from src.utils.sim_start_pose import (
    load_saved_start_pose,
    save_saved_start_pose,
    saved_start_pose_path,
)


_logger = logging.getLogger(__name__)

class MapData:
    """Loaded map: Lanelet2 geometry + visual background metadata."""

    def __init__(self, map_dir: Path):
        self.map_dir = Path(map_dir)
        self.meters_per_pixel: float = 0.01
        self.meters_per_pixel_x: float = 0.01
        self.meters_per_pixel_y: float = 0.01
        self.img_w: int = 2000
        self.img_h: int = 1333
        self.y_axis_inverted: bool = True
        self.x_min_m: float = 0.0
        self.y_min_m: float = 0.0
        self.background: Optional[QPixmap] = None
        self.map_source: str = "none"
        self._meta_loaded: bool = False
        self._bounds_loaded_from_meta: bool = False
        self.lanelet_map = None
        self.lanelet_polygons: list[tuple[str, list[tuple[float, float]], dict[str, str]]] = []
        self.lanelet_polygons_by_id: dict[str, tuple[tuple[float, float], ...]] = {}
        self.lanelet_boundaries: list[tuple[str, list[tuple[float, float]], dict[str, str]]] = []
        self.lanelet_centerlines: list[tuple[str, list[tuple[float, float]], dict[str, str]]] = []
        self.semantic_markers: list[tuple[str, str, float, float, str]] = []

        self._load_meta()
        self._load_background()
        self._load_osm()

    # ------------------------------------------------------------------
    def _load_meta(self) -> None:
        meta_path = self.map_dir / "track_meta.json"
        meta = persistence.load_json(meta_path, default={}) or {}
        base_mpp = float(meta.get("metersPerPixel", 0.01))
        self.meters_per_pixel = base_mpp
        self.meters_per_pixel_x = float(meta.get("metersPerPixelX", base_mpp))
        self.meters_per_pixel_y = float(meta.get("metersPerPixelY", base_mpp))
        self.img_w = int(meta.get("imgW", 2000))
        self.img_h = int(meta.get("imgH", 1333))
        self.y_axis_inverted = bool(meta.get("y_axis_inverted", True))
        self._meta_loaded = bool(meta)
        bounds = meta.get("world_bounds") or {}
        try:
            self.x_min_m = float(bounds.get("x_min", self.x_min_m))
            self.y_min_m = float(bounds.get("y_min", self.y_min_m))
            self._bounds_loaded_from_meta = "x_min" in bounds and "y_min" in bounds
        except (TypeError, ValueError):
            pass

    def _load_background(self) -> None:
        # Prefer rasterising the SVG (sharper), fall back to PNG.
        svg = self.map_dir / "track.svg"
        png = self.map_dir / "track.png"
        if svg.exists():
            try:
                renderer = QSvgRenderer(str(svg))
                if renderer.isValid():
                    img = QImage(self.img_w, self.img_h, QImage.Format_ARGB32)
                    img.fill(QColor("#000000"))
                    painter = QPainter(img)
                    renderer.render(painter)
                    painter.end()
                    self.background = QPixmap.fromImage(img)
                    return
            except Exception as exc:
                _logger.warning("SVG rasterisation failed: %s", exc)
        if png.exists():
            self.background = QPixmap(str(png))
            self._maybe_flip_background()

    def _maybe_flip_background(self) -> None:
        """Flippea verticalmente el PNG fallback cuando ``y_axis_inverted=False``.

        Razón: el PNG fue rasterizado con la convención de imagen estándar (origen
        top-left, +Y hacia abajo). Con ``y_axis_inverted=True`` el render
        compensaba ese flip al transformar mapa→pixel — pero entonces la
        orientación visual del dashboard quedaba opuesta a la del gz GUI
        (verificado en screenshots por el usuario). Con ``y_axis_inverted=False``
        no se compensa en `world_to_pixel`, así que tenemos que flipear la
        imagen acá para que el track dibujado quede en la misma orientación
        que el gz GUI muestra el mundo.
        """
        if self.background is None or self.y_axis_inverted:
            return
        try:
            from PyQt5.QtGui import QTransform
            self.background = self.background.transformed(
                QTransform().scale(1, -1)
            )
        except Exception as exc:
            _logger.warning("Could not vertically flip track background: %s", exc)

    def _load_osm(self) -> bool:
        for osm_path in (self.map_dir / "lanelet2_map.osm",):
            if not osm_path.exists():
                continue
            try:
                parsed = parse_lanelet2_osm(str(osm_path))
                lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.20)
            except Exception as exc:
                _logger.warning("Lanelet2 OSM load failed at %s: %s", osm_path, exc)
                continue

            self.lanelet_polygons.clear()
            self.lanelet_polygons_by_id.clear()
            self.lanelet_boundaries.clear()
            self.lanelet_centerlines.clear()
            self.semantic_markers.clear()
            self.map_source = "osm"
            self.lanelet_map = lanelet_map
            self._apply_osm_map_metadata(lanelet_map.get_map_metadata())
            self._build_osm_geometry(parsed=parsed, lanelet_map=lanelet_map)
            if self.lanelet_polygons or self.lanelet_centerlines or self.semantic_markers:
                return True
        return False

    def _apply_osm_map_metadata(self, map_metadata: dict) -> None:
        if not isinstance(map_metadata, dict):
            return
        if not self._meta_loaded:
            try:
                base_mpp = float(map_metadata.get("meters_per_pixel", self.meters_per_pixel))
                self.meters_per_pixel = base_mpp
                self.meters_per_pixel_x = base_mpp
                self.meters_per_pixel_y = base_mpp
            except (TypeError, ValueError):
                pass
            try:
                self.img_w = int(map_metadata.get("image_width_px", self.img_w))
                self.img_h = int(map_metadata.get("image_height_px", self.img_h))
            except (TypeError, ValueError):
                pass
            self.y_axis_inverted = bool(map_metadata.get("y_axis_inverted", self.y_axis_inverted))
        if not self._bounds_loaded_from_meta:
            bounds = map_metadata.get("world_bounds") or {}
            try:
                self.x_min_m = float(bounds.get("x_min", self.x_min_m))
                self.y_min_m = float(bounds.get("y_min", self.y_min_m))
            except (TypeError, ValueError):
                pass

    def _build_osm_geometry(self, *, parsed, lanelet_map) -> None:
        def _rel_member(rel, role: str, member_type: str = "way") -> str | None:
            for m_type, ref, m_role in rel.members:
                if m_type == member_type and m_role == role:
                    return str(ref)
            return None

        def _way_points(way_id: str | None) -> list[tuple[float, float]]:
            if way_id is None:
                return []
            way = parsed.ways.get(str(way_id))
            if way is None:
                return []
            pts = []
            for node_id in way.node_ids:
                node = parsed.nodes.get(str(node_id))
                if node is None:
                    continue
                pts.append((float(node.x), float(node.y)))
            return pts

        def _align_bounds(
            left: list[tuple[float, float]],
            right: list[tuple[float, float]],
        ) -> list[tuple[float, float]]:
            if len(left) < 2 or len(right) < 2:
                return right
            same_dir = (
                ((left[0][0] - right[0][0]) ** 2 + (left[0][1] - right[0][1]) ** 2) +
                ((left[-1][0] - right[-1][0]) ** 2 + (left[-1][1] - right[-1][1]) ** 2)
            )
            reverse_dir = (
                ((left[0][0] - right[-1][0]) ** 2 + (left[0][1] - right[-1][1]) ** 2) +
                ((left[-1][0] - right[0][0]) ** 2 + (left[-1][1] - right[0][1]) ** 2)
            )
            return list(reversed(right)) if reverse_dir < same_dir else right

        def _centerline_points(lanelet_id: str) -> list[tuple[float, float]]:
            ll = lanelet_map.get_lanelet(str(lanelet_id))
            if ll is None or getattr(ll.centerline, "shape", (0,))[0] < 2:
                return []
            return [(float(pt[0]), float(pt[1])) for pt in ll.centerline]

        for rel in parsed.relations.values():
            if str(rel.tags.get("type", "")).lower() != "lanelet":
                continue
            left_way_id = _rel_member(rel, "left")
            right_way_id = _rel_member(rel, "right")
            center_way_id = _rel_member(rel, "centerline")
            left_pts = _way_points(left_way_id)
            right_pts = _align_bounds(left_pts, _way_points(right_way_id))
            if len(left_pts) >= 2 and len(right_pts) >= 2:
                polygon = list(left_pts) + list(reversed(right_pts))
                self.lanelet_polygons.append((rel.relation_id, polygon, dict(rel.tags)))
                self.lanelet_polygons_by_id[str(rel.relation_id)] = tuple(
                    (float(x_m), float(y_m)) for x_m, y_m in polygon
                )
                self.lanelet_boundaries.append(
                    (f"{rel.relation_id}:left", list(left_pts), dict(parsed.ways.get(str(left_way_id)).tags if left_way_id and parsed.ways.get(str(left_way_id)) else {}))
                )
                self.lanelet_boundaries.append(
                    (f"{rel.relation_id}:right", list(right_pts), dict(parsed.ways.get(str(right_way_id)).tags if right_way_id and parsed.ways.get(str(right_way_id)) else {}))
                )
            center_pts = _way_points(center_way_id) if center_way_id else _centerline_points(rel.relation_id)
            if len(center_pts) >= 2:
                self.lanelet_centerlines.append((rel.relation_id, center_pts, dict(rel.tags)))

        marker_keys: set[tuple[str, int, int]] = set()
        for reg_id in getattr(lanelet_map, "regulator_ids", ()):
            reg = lanelet_map.get_regulator(str(reg_id))
            if reg is None:
                continue
            x, y = float(reg.position_xy[0]), float(reg.position_xy[1])
            key = (str(reg.kind), int(round(x * 100.0)), int(round(y * 100.0)))
            if key in marker_keys:
                continue
            marker_keys.add(key)
            self.semantic_markers.append(
                (str(reg.element_id), str(reg.kind), x, y, str(reg.label or reg.kind))
            )

        for lanelet_id in getattr(lanelet_map, "lanelet_ids", ()):
            lanelet = lanelet_map.get_lanelet(str(lanelet_id))
            if lanelet is None or getattr(lanelet.centerline, "shape", (0,))[0] < 2:
                continue
            attr_kind = self._kind_from_attr(int(lanelet.attribute))
            if attr_kind is None:
                continue
            mid = lanelet.centerline[lanelet.centerline.shape[0] // 2]
            x, y = float(mid[0]), float(mid[1])
            key = (attr_kind, int(round(x * 100.0)), int(round(y * 100.0)))
            if key in marker_keys:
                continue
            marker_keys.add(key)
            self.semantic_markers.append(
                (f"{attr_kind}:{lanelet_id}", attr_kind, x, y, f"{attr_kind} {lanelet_id}")
            )

    @staticmethod
    def _kind_from_attr(attr: int) -> str | None:
        if attr == ATTR_STOPLINE:
            return "stopline"
        if attr == ATTR_CROSSWALK:
            return "crosswalk"
        if attr == ATTR_INTERSECTION:
            return "intersection"
        if attr == ATTR_ROUNDABOUT:
            return "roundabout"
        return None

    def world_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Convert world meters → image pixels (top-left origin)."""
        local_x = float(x_m) - float(self.x_min_m)
        local_y = float(y_m) - float(self.y_min_m)
        px = local_x / self.meters_per_pixel_x
        if self.y_axis_inverted:
            py = self.img_h - (local_y / self.meters_per_pixel_y)
        else:
            py = local_y / self.meters_per_pixel_y
        return px, py

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        x_m = float(self.x_min_m) + px * self.meters_per_pixel_x
        if self.y_axis_inverted:
            y_m = float(self.y_min_m) + (self.img_h - py) * self.meters_per_pixel_y
        else:
            y_m = float(self.y_min_m) + py * self.meters_per_pixel_y
        return x_m, y_m

    def scene_rect(self) -> QRectF:
        if self.background is not None:
            return QRectF(0, 0, self.img_w, self.img_h)
        world_points: list[tuple[float, float]] = []
        for _, polygon, _tags in self.lanelet_polygons:
            world_points.extend(polygon)
        for _, centerline, _tags in self.lanelet_centerlines:
            world_points.extend(centerline)
        for _marker_id, _kind, x, y, _label in self.semantic_markers:
            world_points.append((x, y))
        if not world_points:
            return QRectF(0, 0, self.img_w, self.img_h)
        pxs = []
        pys = []
        for x_m, y_m in world_points:
            px, py = self.world_to_pixel(x_m, y_m)
            pxs.append(px)
            pys.append(py)
        margin = 60.0
        min_x = min(pxs) - margin
        max_x = max(pxs) + margin
        min_y = min(pys) - margin
        max_y = max(pys) + margin
        return QRectF(min_x, min_y, max(10.0, max_x - min_x), max(10.0, max_y - min_y))


def _build_world_path(
    points: list[tuple[float, float]],
    *,
    world_to_pixel,
) -> QPainterPath:
    if len(points) < 2:
        return QPainterPath()
    path = QPainterPath()
    first_x_m, first_y_m = points[0]
    first_px, first_py = world_to_pixel(first_x_m, first_y_m)
    path.moveTo(first_px, first_py)
    for x_m, y_m in points[1:]:
        px, py = world_to_pixel(x_m, y_m)
        path.lineTo(px, py)
    return path


# ----------------------------------------------------------------------
# Custom QGraphicsScene that emits world coordinates on click.
# ----------------------------------------------------------------------
class _MapScene(QGraphicsScene):
    map_clicked = pyqtSignal(float, float)  # world_x, world_y

    def __init__(self, data: MapData, parent=None):
        super().__init__(parent)
        self._data = data

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            sp = event.scenePos()
            x_m, y_m = self._data.pixel_to_world(sp.x(), sp.y())
            self.map_clicked.emit(x_m, y_m)
            event.accept()
            return
        super().mousePressEvent(event)


# ----------------------------------------------------------------------
# The car cursor — a triangle that rotates with yaw, plus a halo so it
# stays visible against the dark background and the busy node grid.
# ----------------------------------------------------------------------
class _CarCursor(QGraphicsPolygonItem):
    """Vehicle footprint anchored at the rear axle center.

    The tracking pose (`Location.x/y/yaw`) uses the same rear-axle reference
    as the dead-reckoning model. If the dashboard icon is centred around its
    geometric midpoint instead, rotating it makes the car *look* like it moved
    sideways or backwards even though the pose point stayed fixed.
    """

    _PX_PER_M = 105.0
    _WHEELBASE_M = 0.260
    _REAR_OVERHANG_M = 0.018
    _FRONT_OVERHANG_M = 0.072
    _HALF_WIDTH_M = 0.095

    def __init__(self):
        rear = self._REAR_OVERHANG_M * self._PX_PER_M
        front = (self._WHEELBASE_M + self._FRONT_OVERHANG_M) * self._PX_PER_M
        half_w = self._HALF_WIDTH_M * self._PX_PER_M
        shoulder_x = front * 0.78
        polygon = QPolygonF([
            QPointF(front, 0.0),
            QPointF(shoulder_x, half_w * 0.88),
            QPointF(-rear, half_w),
            QPointF(-rear, -half_w),
            QPointF(shoulder_x, -half_w * 0.88),
        ])
        super().__init__(polygon)
        self.setTransformOriginPoint(QPointF(0.0, 0.0))
        self.setBrush(QBrush(QColor("#ff6b57")))
        self.setPen(QPen(QColor("#ffffff"), 2.0))
        self.setZValue(101)
        self.setVisible(False)


class _CarHalo(QGraphicsEllipseItem):
    """Translucent ring drawn under the car cursor for extra visibility."""

    R = 22

    def __init__(self):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self.setPen(QPen(QColor(255, 59, 48, 220), 3))
        self.setBrush(QBrush(QColor(255, 59, 48, 60)))
        self.setZValue(100)
        self.setVisible(False)


class _GpsDot(QGraphicsEllipseItem):
    """Blue dot marking the last raw GPS fix from LoCSys."""

    R = 7

    def __init__(self):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self.setPen(QPen(QColor("#ffffff"), 1.5))
        self.setBrush(QBrush(QColor(0, 120, 255, 200)))
        self.setZValue(90)
        self.setVisible(False)


class _DrDot(QGraphicsEllipseItem):
    """Green dot marking the pure dead-reckoning estimate."""

    R = 8

    def __init__(self):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self.setPen(QPen(QColor("#ffffff"), 2.0))
        self.setBrush(QBrush(QColor(46, 212, 122, 230)))
        self.setZValue(103)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setVisible(False)


# ----------------------------------------------------------------------
# Public widget
# ----------------------------------------------------------------------
class MapView(QWidget):
    """Autoware-style lanelet visualizer for the dashboard."""

    def __init__(self, client: SocketIOClient, map_dir: Optional[Path] = None,
                 parent: Optional[QWidget] = None, *, compact: bool = False):
        super().__init__(parent)
        self._client = client
        self._compact = bool(compact)
        self._map_dir = Path(map_dir) if map_dir else self._guess_map_dir()
        self._saved_start_pose = load_saved_start_pose(map_dir=self._map_dir)
        self._last_pose_snapshot: dict[str, object] | None = None
        self._data = MapData(self._map_dir)

        self._scene = _MapScene(self._data, self)
        # Custom view: wheel zoom, trackpad pan/pinch, right-drag pan, left
        # button stays available for the scene's own click handling.
        self._view = PannableZoomableView(self._scene, self)
        self._view.setBackgroundBrush(QBrush(QColor("#0c0c0c")))
        self._lanelet_fill_items: dict[str, QGraphicsPolygonItem] = {}
        self._active_current_lanelet_id: str | None = None
        self._active_next_lanelet_ids: tuple[str, ...] = ()
        self._has_centered_on_ego = False
        self._pending_initial_fit = True
        self._syncing_map_calibration_controls = False

        self._build_scene()

        self._location_label = QLabel("(no fix)")
        self._location_label.setMinimumWidth(0)
        self._location_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._location_label.setStyleSheet("color:#aaa;")

        self._gps_label = QLabel("GPS: --")
        self._gps_label.setStyleSheet("color:#666; font-size: 10pt;")
        self._gps_label.setToolTip("Age of last GPS fix from LoCSys")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        if not self._compact:
            toolbar = QHBoxLayout()
            # "Relocate" mode forces every left-click to fire ``Localisation``,
            # so the user can drop the car onto the map regardless of which
            # driving mode is active. In sim, the brain forwards the new pose
            # to its localisation filter; whether the Gazebo body teleports too
            # is up to the sim bridge (separate concern).
            self._relocate_btn = QToolButton()
            self._relocate_btn.setText("Relocate")
            self._relocate_btn.setToolTip(
                "Stop only: in sim, left-click relocalizes and teleports the "
                "simulator pose. On the real car, the same click calibrates GPS "
                "so the latest LoCSys fix is treated as that map point."
            )
            self._relocate_btn.setCheckable(True)
            self._relocate_btn.toggled.connect(self._on_relocate_toggled)
            self._save_start_btn = QPushButton("Save start")
            self._save_start_btn.setToolTip(
                "Stop only: save the current car pose into "
                "config/sim_start_pose.json so the next sim run starts there."
            )
            self._save_start_btn.clicked.connect(self._save_current_pose_as_start)
            self._start_pose_label = QLabel()
            self._start_pose_label.setMinimumWidth(0)
            self._start_pose_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            self._start_pose_label.setStyleSheet("color:#666; font-size: 10pt;")
            self._update_saved_start_label()

            fit_btn = QPushButton("Fit view")
            fit_btn.clicked.connect(self._fit_view)
            load_btn = QPushButton("Load map…")
            load_btn.clicked.connect(self._load_map_dialog)
            clear_route_btn = QPushButton("Clear route")
            clear_route_btn.clicked.connect(self._clear_route)
            self._clear_route_btn = clear_route_btn
            self._map_edit_btn = QToolButton()
            self._map_edit_btn.setText("Edit map")
            self._map_edit_btn.setCheckable(True)
            self._map_edit_btn.setToolTip(
                "Stop only: edit the map calibration saved in track_meta.json. "
                "This changes the real pixel/world conversion used by clicks and relocate."
            )
            self._map_edit_btn.toggled.connect(self._on_map_edit_toggled)
            self._map_x_min_spin = self._make_map_calibration_spin(
                "xMin ",
                decimals=4,
                step=0.01,
                tooltip="world_bounds.x_min in metres. Larger moves lanelets left on the SVG.",
            )
            self._map_y_min_spin = self._make_map_calibration_spin(
                "yMin ",
                decimals=4,
                step=0.01,
                tooltip=(
                    "world_bounds.y_min in metres. With this non-inverted SVG, "
                    "larger moves lanelets up."
                ),
            )
            self._map_scale_spin = self._make_map_calibration_spin(
                "Scale ",
                decimals=5,
                step=0.001,
                minimum=0.5,
                maximum=1.5,
                tooltip=(
                    "Uniform map scale. 1.00000 keeps the Lanelet/SVG scale unchanged; "
                    "larger makes lanelets visually larger."
                ),
            )
            self._map_scale_spin.setValue(1.0)
            self._map_one_to_one_btn = QPushButton("1x")
            self._map_one_to_one_btn.setFixedWidth(44)
            self._map_one_to_one_btn.setToolTip("Reset the scale field to 1.00000.")
            self._map_one_to_one_btn.clicked.connect(self._reset_map_scale_to_one)
            self._map_save_btn = QPushButton("Save meta")
            self._map_save_btn.setFixedWidth(86)
            self._map_save_btn.setToolTip("Save xMin, yMin and scale into track_meta.json.")
            self._map_save_btn.clicked.connect(self._save_map_calibration)
            self._map_calibration_label = QLabel()
            self._map_calibration_label.setMinimumWidth(0)
            self._map_calibration_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            self._map_calibration_label.setStyleSheet("color:#777; font-size: 10pt;")
            self._map_calibration_panel = QWidget()
            self._map_calibration_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            map_edit_layout = QHBoxLayout(self._map_calibration_panel)
            map_edit_layout.setContentsMargins(0, 0, 0, 0)
            map_edit_layout.setSpacing(6)
            map_edit_layout.addWidget(self._map_x_min_spin)
            map_edit_layout.addWidget(self._map_y_min_spin)
            map_edit_layout.addWidget(self._map_scale_spin)
            map_edit_layout.addWidget(self._map_one_to_one_btn)
            map_edit_layout.addWidget(self._map_save_btn)
            map_edit_layout.addWidget(self._map_calibration_label, 1)
            self._map_calibration_panel.setVisible(False)
            self._sync_map_calibration_controls()
            self._refresh_map_calibration_controls()

            hint = QLabel("Wheel/pinch: zoom · Right-drag or two-finger: pan")
            hint.setMinimumWidth(0)
            hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            hint.setStyleSheet("color:#666; font-size: 10pt;")

            toolbar.addWidget(load_btn)
            toolbar.addWidget(fit_btn)
            toolbar.addWidget(clear_route_btn)
            toolbar.addWidget(self._relocate_btn)
            toolbar.addWidget(self._save_start_btn)
            toolbar.addSpacing(10)
            toolbar.addWidget(self._map_edit_btn)
            toolbar.addStretch(1)
            toolbar.addWidget(hint)
            toolbar.addSpacing(12)
            toolbar.addWidget(self._start_pose_label)
            toolbar.addSpacing(12)
            toolbar.addWidget(self._gps_label)
            toolbar.addSpacing(8)
            toolbar.addWidget(self._location_label)
            layout.addLayout(toolbar)
            layout.addWidget(self._map_calibration_panel)
        else:
            compact_header = QHBoxLayout()
            title = QLabel("Map")
            title.setStyleSheet("color:#ddd; font-weight:600;")
            compact_header.addWidget(title)
            compact_header.addStretch(1)
            compact_header.addWidget(self._gps_label)
            compact_header.addSpacing(8)
            compact_header.addWidget(self._location_label)
            layout.addLayout(compact_header)
        layout.addWidget(self._view, 1)

        # GPS state — updated by _on_gps_fix, polled by _gps_age_timer.
        self._last_gps_fix_time: float = 0.0
        self._last_gps_fix_payload: dict[str, object] | None = None
        self._last_location_time: float = 0.0
        self._last_location_yaw_deg: float = 0.0
        self._gps_age_timer = QTimer(self)
        self._gps_age_timer.setInterval(500)
        self._gps_age_timer.timeout.connect(self._update_gps_age_label)
        self._gps_age_timer.start()

        # ---- Wire signals ----
        if not self._compact:
            self._scene.map_clicked.connect(self._on_map_click)
        self._client.location_signal.connect(self._on_location)
        self._client.cars_signal.connect(self._on_cars)
        self._client.semaphores_signal.connect(self._on_semaphores)
        self._client.navigation_status_signal.connect(self._on_nav_status)
        self._client.behavior_output_signal.connect(self._on_behavior_output)
        self._client.state_change_signal.connect(self._on_state_change)
        self._client.gps_fix_signal.connect(self._on_gps_fix)
        self._client.dead_reckoning_signal.connect(self._on_dead_reckoning)
        self._mode = ev.MODE_STOP
        self._relocate_mode = False
        self._refresh_manual_pose_controls()
        self._fit_view()

    # ------------------------------------------------------------------
    def showEvent(self, event):  # noqa: N802 - Qt convention
        super().showEvent(event)
        if not self._pending_initial_fit:
            return
        self._pending_initial_fit = False
        # The compact map inside Driving is laid out by nested splitters, so
        # the first fit done in __init__ often uses a too-small viewport.
        QTimer.singleShot(0, self._fit_view)

    def resizeEvent(self, event):  # noqa: N802 - Qt convention
        super().resizeEvent(event)
        if not self._compact or not self.isVisible():
            return
        # Keep the mini map framed to the available panel size as telemetry
        # grows/shrinks, matching the explicit "Fit view" action.
        QTimer.singleShot(0, self._fit_view)

    # ------------------------------------------------------------------
    def _guess_map_dir(self) -> Path:
        """Default to the same track-map directory resolved by the backend config."""
        return settings.default_track_map_dir()

    # ------------------------------------------------------------------
    def _build_scene(self) -> None:
        self._scene.clear()
        self._lanelet_fill_items.clear()
        self._has_centered_on_ego = False

        if self._data.background is not None:
            bg = QGraphicsPixmapItem(self._data.background)
            bg.setZValue(0)
            self._scene.addItem(bg)
        self._scene.setSceneRect(self._data.scene_rect())

        self._render_lanelet_map()

        # Semantic highlights (intersection rings, etc).
        self._render_semantics()

        # Persistent overlay items (pre-create so we can update in place).
        # Halo first (lower z) so the triangle sits on top.
        self._cursor_halo = _CarHalo()
        self._scene.addItem(self._cursor_halo)
        self._heading_ray = QGraphicsLineItem(0.0, 0.0, 54.0, 0.0)
        self._heading_ray.setPen(QPen(QColor(255, 214, 102, 210), 2.5, Qt.SolidLine, Qt.RoundCap))
        self._heading_ray.setZValue(100)
        self._heading_ray.setVisible(False)
        self._scene.addItem(self._heading_ray)
        self._cursor = _CarCursor()
        self._scene.addItem(self._cursor)
        self._gps_dot = _GpsDot()
        self._scene.addItem(self._gps_dot)
        self._dr_dot = _DrDot()
        self._scene.addItem(self._dr_dot)

        self._cars_items: list[QGraphicsItem] = []
        self._semaphore_items: list[QGraphicsItem] = []
        self._nav_path = QGraphicsPathItem()
        self._nav_path.setPen(QPen(QColor("#7db5ff"), 2, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin))
        self._nav_path.setZValue(50)
        self._scene.addItem(self._nav_path)
        self._control_path = QGraphicsPathItem()
        self._control_path.setPen(QPen(QColor("#2ed47a"), 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        self._control_path.setZValue(51)
        self._scene.addItem(self._control_path)
        self._apply_lanelet_highlights(
            current_lanelet_id=self._active_current_lanelet_id,
            next_lanelet_ids=self._active_next_lanelet_ids,
        )

    def _render_lanelet_map(self) -> None:
        fill_brush = QBrush(QColor(170, 170, 170, 36))
        boundary_pen = QPen(QColor(238, 238, 238, 210), 2.1)
        dashed_boundary_pen = QPen(QColor(190, 190, 190, 180), 1.8, Qt.DashLine)
        center_pen = QPen(QColor(110, 205, 255, 150), 1.5, Qt.DashLine)

        for lanelet_id, polygon_pts, tags in self._data.lanelet_polygons:
            polygon = QPolygonF(
                [QPointF(*self._data.world_to_pixel(x_m, y_m)) for x_m, y_m in polygon_pts]
            )
            item = QGraphicsPolygonItem(polygon)
            item.setBrush(fill_brush)
            item.setPen(QPen(Qt.NoPen))
            item.setZValue(0.5)
            item.setToolTip(f"Lanelet {lanelet_id}\n{tags.get('subtype', 'road')}")
            self._scene.addItem(item)
            self._lanelet_fill_items[str(lanelet_id)] = item

        for boundary_id, points, tags in self._data.lanelet_boundaries:
            if len(points) < 2:
                continue
            path = QPainterPath()
            x0, y0 = self._data.world_to_pixel(*points[0])
            path.moveTo(x0, y0)
            for point in points[1:]:
                px, py = self._data.world_to_pixel(*point)
                path.lineTo(px, py)
            item = QGraphicsPathItem(path)
            subtype = str(tags.get("subtype") or "").lower()
            item.setPen(dashed_boundary_pen if subtype in {"dashed", "dash"} else boundary_pen)
            item.setZValue(1.0)
            item.setToolTip(boundary_id)
            self._scene.addItem(item)

        for lanelet_id, centerline_pts, tags in self._data.lanelet_centerlines:
            if len(centerline_pts) < 2:
                continue
            path = QPainterPath()
            x0, y0 = self._data.world_to_pixel(*centerline_pts[0])
            path.moveTo(x0, y0)
            for point in centerline_pts[1:]:
                px, py = self._data.world_to_pixel(*point)
                path.lineTo(px, py)
            item = QGraphicsPathItem(path)
            item.setPen(center_pen)
            item.setZValue(1.4)
            item.setToolTip(f"Centerline {lanelet_id}\n{tags.get('location', 'road')}")
            self._scene.addItem(item)

    def _apply_lanelet_highlights(
        self,
        *,
        current_lanelet_id: str | None,
        next_lanelet_ids: tuple[str, ...],
    ) -> None:
        if not self._lanelet_fill_items:
            return
        default_brush = QBrush(QColor(170, 170, 170, 36))
        next_brush = QBrush(QColor(54, 214, 156, 68))
        current_brush = QBrush(QColor(255, 107, 87, 110))
        for item in self._lanelet_fill_items.values():
            item.setBrush(default_brush)
        for lanelet_id in next_lanelet_ids:
            item = self._lanelet_fill_items.get(str(lanelet_id))
            if item is not None:
                item.setBrush(next_brush)
        if current_lanelet_id:
            item = self._lanelet_fill_items.get(str(current_lanelet_id))
            if item is not None:
                item.setBrush(current_brush)

    def _render_semantics(self) -> None:
        """Coloured markers for regulatory elements and semantic areas."""
        sem_colors = {
            "intersection": QColor(255, 92, 92),
            "crosswalk": QColor(85, 135, 255),
            "stopline": QColor(255, 196, 64),
            "roundabout": QColor(90, 220, 210),
            "parking": QColor(180, 100, 255),
            "yield": QColor(255, 196, 64),
            "traffic_light": QColor(255, 196, 64),
            "stop": QColor(255, 196, 64),
        }
        for marker_id, kind, x_m, y_m, label in self._data.semantic_markers:
            px, py = self._data.world_to_pixel(x_m, y_m)
            color = sem_colors.get(str(kind).lower(), QColor(140, 140, 140))
            r = 8
            ring = QGraphicsEllipseItem(px - r, py - r, 2 * r, 2 * r)
            ring.setPen(QPen(color, 2))
            ring.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 32)))
            ring.setZValue(3)
            ring.setToolTip(f"{kind}: {label}\n{marker_id}")
            self._scene.addItem(ring)

    def _fit_view(self) -> None:
        if self._scene.sceneRect().isEmpty():
            return
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _make_map_calibration_spin(
        self,
        prefix: str,
        *,
        decimals: int,
        step: float,
        tooltip: str,
        minimum: float = -1000.0,
        maximum: float = 1000.0,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(int(decimals))
        spin.setSingleStep(float(step))
        spin.setPrefix(prefix)
        spin.setFixedWidth(112 if prefix != "Scale " else 104)
        spin.setKeyboardTracking(False)
        spin.setToolTip(tooltip)
        spin.valueChanged.connect(self._on_map_calibration_changed)
        return spin

    def _map_base_mpp(self) -> float:
        try:
            base_mpp = float(self._data.meters_per_pixel)
        except (TypeError, ValueError):
            base_mpp = 0.01
        return base_mpp if base_mpp > 0.0 else 0.01

    def _current_map_scale(self) -> float:
        base_mpp = self._map_base_mpp()
        try:
            mpp_x = float(self._data.meters_per_pixel_x)
            mpp_y = float(self._data.meters_per_pixel_y)
        except (TypeError, ValueError):
            return 1.0
        mpp = (mpp_x + mpp_y) / 2.0
        return base_mpp / mpp if mpp > 0.0 else 1.0

    def _set_spin_value_safely(self, spin: QDoubleSpinBox, value: float) -> None:
        spin.blockSignals(True)
        spin.setValue(float(value))
        spin.blockSignals(False)

    def _sync_map_calibration_controls(self) -> None:
        if self._compact or not hasattr(self, "_map_x_min_spin"):
            return
        self._syncing_map_calibration_controls = True
        try:
            self._set_spin_value_safely(self._map_x_min_spin, self._data.x_min_m)
            self._set_spin_value_safely(self._map_y_min_spin, self._data.y_min_m)
            self._set_spin_value_safely(self._map_scale_spin, self._current_map_scale())
        finally:
            self._syncing_map_calibration_controls = False
        self._update_map_calibration_label()

    def _refresh_map_calibration_controls(self) -> None:
        if self._compact or not hasattr(self, "_map_edit_btn"):
            return
        enabled = bool(self._map_edit_btn.isChecked()) and self._is_stop_mode()
        if hasattr(self, "_map_calibration_panel"):
            self._map_calibration_panel.setVisible(enabled)
        for widget in (
            self._map_x_min_spin,
            self._map_y_min_spin,
            self._map_scale_spin,
            self._map_one_to_one_btn,
            self._map_save_btn,
        ):
            widget.setEnabled(enabled)

    def _update_map_calibration_label(self) -> None:
        if self._compact or not hasattr(self, "_map_calibration_label"):
            return
        scale = self._current_map_scale()
        self._map_calibration_label.setText(
            f"xMin {self._data.x_min_m:.3f} yMin {self._data.y_min_m:.3f} s {scale:.5f}x"
        )
        self._map_calibration_label.setToolTip(
            "Saved in track_meta.json. These values affect the dashboard "
            "pixel/world conversion, including map clicks and Relocate."
        )

    def _on_map_edit_toggled(self, checked: bool) -> None:
        if checked and not self._is_stop_mode():
            self._map_edit_btn.blockSignals(True)
            self._map_edit_btn.setChecked(False)
            self._map_edit_btn.blockSignals(False)
            self._location_label.setText("Map edit is only available in Stop mode")
            checked = False
        if checked:
            self._sync_map_calibration_controls()
        self._refresh_map_calibration_controls()

    def _reset_map_scale_to_one(self) -> None:
        if not hasattr(self, "_map_scale_spin"):
            return
        self._map_scale_spin.setValue(1.0)

    def _on_map_calibration_changed(self, _value: float) -> None:
        if self._syncing_map_calibration_controls:
            return
        if self._compact or not getattr(self, "_map_edit_btn", None) or not self._map_edit_btn.isChecked():
            return
        self._apply_map_calibration_preview()

    def _apply_map_calibration_preview(self) -> None:
        scale = max(0.000001, float(self._map_scale_spin.value()))
        mpp = self._map_base_mpp() / scale
        self._data.x_min_m = float(self._map_x_min_spin.value())
        self._data.y_min_m = float(self._map_y_min_spin.value())
        self._data.meters_per_pixel_x = mpp
        self._data.meters_per_pixel_y = mpp

        center = self._view.mapToScene(self._view.viewport().rect().center())
        self._build_scene()
        self._view.centerOn(center)
        self._redraw_last_pose_after_map_calibration()
        self._update_map_calibration_label()

    def _redraw_last_pose_after_map_calibration(self) -> None:
        if not isinstance(self._last_pose_snapshot, dict):
            return
        try:
            x_m = float(self._last_pose_snapshot["world_x"])
            y_m = float(self._last_pose_snapshot["world_y"])
            yaw_deg = float(self._last_pose_snapshot.get("yaw_deg", self._last_location_yaw_deg))
        except (KeyError, TypeError, ValueError):
            return
        self._set_cursor_pose(x_m, y_m, yaw_deg)

    def _save_map_calibration(self) -> None:
        if self._compact:
            return
        if not self._is_stop_mode():
            self._location_label.setText("Map edit is only available in Stop mode")
            self._refresh_map_calibration_controls()
            return
        self._apply_map_calibration_preview()

        scale = max(0.000001, float(self._map_scale_spin.value()))
        base_mpp = self._map_base_mpp()
        mpp = base_mpp / scale
        x_min = float(self._map_x_min_spin.value())
        y_min = float(self._map_y_min_spin.value())
        meta_path = self._map_dir / "track_meta.json"
        meta = persistence.load_json(meta_path, default={})
        if not isinstance(meta, dict):
            meta = {}
        meta["metersPerPixel"] = base_mpp
        meta["metersPerPixelX"] = mpp
        meta["metersPerPixelY"] = mpp
        meta["imgW"] = int(self._data.img_w)
        meta["imgH"] = int(self._data.img_h)
        meta["coordinate_origin"] = meta.get("coordinate_origin", "bottom_left")
        meta["y_axis_inverted"] = bool(self._data.y_axis_inverted)
        meta["world_bounds"] = {
            "x_min": x_min,
            "x_max": x_min + (float(self._data.img_w) * mpp),
            "y_min": y_min,
            "y_max": y_min + (float(self._data.img_h) * mpp),
        }
        meta.pop("map_to_texture", None)
        meta.pop("last_dashboard_alignment", None)
        meta["manual_calibration"] = {
            "saved_at": round(_time.time(), 3),
            "scale": round(scale, 9),
            "x_min": round(x_min, 9),
            "y_min": round(y_min, 9),
        }
        meta["notes"] = (
            "Calibracion manual del dashboard/simulador: xMin/yMin mueven el "
            "frame mapa->pixel; Scale es uniforme y 1.0 mantiene Lanelet/SVG "
            "a la misma escala. Estos valores se usan para clicks y Relocate."
        )
        try:
            persistence.save_json_atomic(meta_path, meta)
        except Exception as exc:
            _logger.warning("Could not save map calibration to %s: %s", meta_path, exc)
            self._location_label.setText("Could not save track_meta.json")
            return
        self._data = MapData(self._map_dir)
        self._scene._data = self._data
        self._build_scene()
        self._sync_map_calibration_controls()
        self._refresh_map_calibration_controls()
        self._redraw_last_pose_after_map_calibration()
        self._location_label.setText(
            f"Saved map calibration → xMin {x_min:.3f}, yMin {y_min:.3f}, scale {scale:.5f}x"
        )

    def _load_map_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select map directory", str(self._map_dir))
        if not path:
            return
        self._map_dir = Path(path)
        self._data = MapData(self._map_dir)
        self._scene._data = self._data  # keep scene in sync
        self._build_scene()
        self._saved_start_pose = load_saved_start_pose(map_dir=self._map_dir)
        self._update_saved_start_label()
        self._sync_map_calibration_controls()
        self._fit_view()

    def _ensure_scene_contains(self, px: float, py: float, *, margin: float = 80.0) -> None:
        rect = self._scene.sceneRect()
        if rect.contains(QPointF(px, py)):
            return
        if rect.isNull() or rect.isEmpty():
            self._scene.setSceneRect(QRectF(px - margin, py - margin, margin * 2.0, margin * 2.0))
            return
        left = min(rect.left(), px - margin)
        right = max(rect.right(), px + margin)
        top = min(rect.top(), py - margin)
        bottom = max(rect.bottom(), py + margin)
        self._scene.setSceneRect(QRectF(left, top, max(10.0, right - left), max(10.0, bottom - top)))

    def _keep_ego_visible(self, px: float, py: float) -> None:
        self._ensure_scene_contains(px, py)
        if not self._has_centered_on_ego:
            self._view.centerOn(px, py)
            self._has_centered_on_ego = True
            return
        if self._compact:
            self._view.centerOn(px, py)

    def _set_cursor_pose(self, x_m: float, y_m: float, yaw_deg: float) -> None:
        px, py = self._data.world_to_pixel(x_m, y_m)
        self._cursor.setPos(px, py)
        self._cursor_halo.setPos(px, py)
        self._keep_ego_visible(px, py)
        rotation_deg = -yaw_deg if self._data.y_axis_inverted else yaw_deg
        display_yaw_deg = ((rotation_deg + 180.0) % 360.0) - 180.0
        self._heading_ray.setPos(px, py)
        self._heading_ray.setRotation(rotation_deg)
        self._heading_ray.setVisible(True)
        self._cursor.setRotation(rotation_deg)
        self._cursor.setVisible(True)
        self._cursor_halo.setVisible(True)
        self._location_label.setText(
            f"({x_m:.2f}, {y_m:.2f}) m  yaw_vis={display_yaw_deg:.0f}°  raw={yaw_deg:.0f}°"
        )

    def _is_stop_mode(self) -> bool:
        return str(getattr(self, "_mode", ev.MODE_STOP) or "").lower() == ev.MODE_STOP

    @staticmethod
    def _distance_point_to_segment(
        x_m: float,
        y_m: float,
        ax_m: float,
        ay_m: float,
        bx_m: float,
        by_m: float,
    ) -> float:
        seg_dx = float(bx_m) - float(ax_m)
        seg_dy = float(by_m) - float(ay_m)
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq <= 1e-12:
            return math.hypot(float(x_m) - float(ax_m), float(y_m) - float(ay_m))
        rel_x = float(x_m) - float(ax_m)
        rel_y = float(y_m) - float(ay_m)
        proj_t = max(0.0, min(1.0, (rel_x * seg_dx + rel_y * seg_dy) / seg_len_sq))
        proj_x = float(ax_m) + proj_t * seg_dx
        proj_y = float(ay_m) + proj_t * seg_dy
        return math.hypot(float(x_m) - proj_x, float(y_m) - proj_y)

    def _remember_pose_snapshot(
        self,
        x_m: float,
        y_m: float,
        yaw_deg: float,
        *,
        lanelet_id: str | None = None,
    ) -> None:
        snapshot: dict[str, object] = {
            "world_x": float(x_m),
            "world_y": float(y_m),
            "yaw_deg": float(yaw_deg),
        }
        if lanelet_id:
            snapshot["lanelet_id"] = str(lanelet_id)
        self._last_pose_snapshot = snapshot
        self._refresh_manual_pose_controls()

    def _resolve_lanelet_id_for_pose(
        self,
        x_m: float,
        y_m: float,
        *,
        yaw_deg: float | None = None,
        prefer_active_corridor: bool = False,
    ) -> str | None:
        lanelet_map = self._data.lanelet_map
        if lanelet_map is None:
            return None
        if yaw_deg is not None:
            try:
                lanelet_id = lanelet_map.at_pose(
                    float(x_m),
                    float(y_m),
                    yaw_rad=math.radians(float(yaw_deg)),
                )
                if lanelet_id is not None:
                    return str(lanelet_id)
            except Exception:
                pass
        return resolve_click_destination_lanelet(
            x_m=x_m,
            y_m=y_m,
            lanelet_map=lanelet_map,
            lanelet_polygons=self._data.lanelet_polygons_by_id,
            current_lanelet_id=self._active_current_lanelet_id if prefer_active_corridor else None,
            next_lanelet_ids=self._active_next_lanelet_ids if prefer_active_corridor else (),
        )

    def _heading_deg_for_pose(
        self,
        x_m: float,
        y_m: float,
        *,
        lanelet_id: str | None = None,
        fallback_deg: float | None = None,
    ) -> float:
        fallback = float(self._last_location_yaw_deg if fallback_deg is None else fallback_deg)
        lanelet_map = self._data.lanelet_map
        if lanelet_map is None or lanelet_id is None:
            return fallback
        lanelet = lanelet_map.get_lanelet(str(lanelet_id))
        centerline = getattr(lanelet, "centerline", None)
        if lanelet is None or centerline is None or getattr(centerline, "shape", (0,))[0] < 2:
            return fallback
        points = centerline.tolist()
        best_heading_deg = fallback
        best_distance = float("inf")
        for idx in range(len(points) - 1):
            ax_m, ay_m = points[idx]
            bx_m, by_m = points[idx + 1]
            distance_m = self._distance_point_to_segment(x_m, y_m, ax_m, ay_m, bx_m, by_m)
            if distance_m < best_distance:
                best_distance = distance_m
                best_heading_deg = math.degrees(math.atan2(float(by_m) - float(ay_m), float(bx_m) - float(ax_m)))
        return float(best_heading_deg)

    @staticmethod
    def _payload_xy(payload: dict[str, object]) -> tuple[float, float] | None:
        for x_key, y_key in (("world_x", "world_y"), ("x", "y"), ("posA", "posB")):
            if payload.get(x_key) is None or payload.get(y_key) is None:
                continue
            try:
                return float(payload[x_key]), float(payload[y_key])
            except (TypeError, ValueError):
                continue
        return None

    def _gps_calibration_sample(self) -> dict[str, float] | None:
        payload = self._last_gps_fix_payload
        if not isinstance(payload, dict) or self._last_gps_fix_time <= 0.0:
            return None
        age_s = _time.monotonic() - self._last_gps_fix_time
        if age_s > 3.0:
            return None
        xy = self._payload_xy(payload)
        if xy is None:
            return None
        raw_x, raw_y = xy
        sample: dict[str, float] = {
            "gps_raw_world_x": round(float(raw_x), 6),
            "gps_raw_world_y": round(float(raw_y), 6),
            "gps_raw_age_s": round(float(age_s), 3),
        }
        try:
            sample["gps_raw_timestamp"] = float(payload.get("timestamp", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        return sample

    def _manual_localisation_payload(self, x_m: float, y_m: float) -> dict[str, object]:
        lanelet_id = self._resolve_lanelet_id_for_pose(
            x_m,
            y_m,
            prefer_active_corridor=True,
        )
        yaw_deg = self._heading_deg_for_pose(
            x_m,
            y_m,
            lanelet_id=lanelet_id,
            fallback_deg=self._last_location_yaw_deg,
        )
        payload: dict[str, object] = {
            "world_x": round(float(x_m), 6),
            "world_y": round(float(y_m), 6),
            "posA": round(float(x_m), 6),
            "posB": round(float(y_m), 6),
            "rotA": 0.0,
            "rotB": 0.0,
            "yaw_deg": round(float(yaw_deg), 3),
            "timestamp": _time.time(),
            "meta": {
                "source": "manual_dashboard",
                "manual": True,
                "gps_calibration": True,
            },
        }
        gps_sample = self._gps_calibration_sample()
        if gps_sample is not None:
            payload.update(gps_sample)
        if lanelet_id is not None:
            payload["lanelet_id"] = str(lanelet_id)
        return payload

    def _saved_start_pose_payload(self) -> dict[str, object] | None:
        if not isinstance(self._last_pose_snapshot, dict):
            return None
        try:
            x_m = float(self._last_pose_snapshot["world_x"])
            y_m = float(self._last_pose_snapshot["world_y"])
        except (KeyError, TypeError, ValueError):
            return None
        try:
            yaw_deg = float(self._last_pose_snapshot.get("yaw_deg", self._last_location_yaw_deg))
        except (TypeError, ValueError):
            yaw_deg = float(self._last_location_yaw_deg)
        lanelet_id = self._last_pose_snapshot.get("lanelet_id")
        lanelet_text = str(lanelet_id).strip() if lanelet_id is not None else ""
        if not lanelet_text:
            lanelet_text = self._resolve_lanelet_id_for_pose(x_m, y_m, yaw_deg=yaw_deg) or ""
        yaw_deg = self._heading_deg_for_pose(
            x_m,
            y_m,
            lanelet_id=lanelet_text or None,
            fallback_deg=yaw_deg,
        )
        payload: dict[str, object] = {
            "world_x": round(float(x_m), 6),
            "world_y": round(float(y_m), 6),
            "yaw_deg": round(float(yaw_deg), 3),
            "saved_at": round(_time.time(), 3),
            "map_dir": str(self._map_dir.resolve()),
        }
        if lanelet_text:
            payload["lanelet_id"] = lanelet_text
        return payload

    def _update_saved_start_label(self) -> None:
        if self._compact or not hasattr(self, "_start_pose_label"):
            return
        config_hint = str(saved_start_pose_path())
        pose = self._saved_start_pose
        if isinstance(pose, dict):
            try:
                x_m = float(pose["world_x"])
                y_m = float(pose["world_y"])
                yaw_deg = float(
                    pose.get("yaw_deg")
                    if pose.get("yaw_deg") is not None
                    else math.degrees(float(pose["yaw_rad"]))
                    if pose.get("yaw_rad") is not None
                    else 0.0
                )
            except (KeyError, TypeError, ValueError):
                pose = None
            else:
                self._start_pose_label.setText(f"Start: ({x_m:.2f}, {y_m:.2f})")
                self._start_pose_label.setStyleSheet("color:#8fd19e; font-size: 10pt;")
                self._start_pose_label.setToolTip(
                    f"Saved sim start pose for this map.\nYaw: {yaw_deg:.1f}°\n{config_hint}"
                )
                return
        self._start_pose_label.setText("Start: default")
        self._start_pose_label.setStyleSheet("color:#666; font-size: 10pt;")
        self._start_pose_label.setToolTip(
            f"No saved sim start pose for this map.\n{config_hint}"
        )

    def _refresh_manual_pose_controls(self) -> None:
        if self._compact:
            return
        can_edit_pose = self._is_stop_mode()
        if not can_edit_pose and getattr(self, "_relocate_mode", False):
            self._relocate_btn.blockSignals(True)
            self._relocate_btn.setChecked(False)
            self._relocate_btn.blockSignals(False)
            self._on_relocate_toggled(False)
        self._relocate_btn.setEnabled(can_edit_pose)
        if hasattr(self, "_save_start_btn"):
            self._save_start_btn.setEnabled(can_edit_pose and self._last_pose_snapshot is not None)

    def _save_current_pose_as_start(self) -> None:
        if not self._is_stop_mode():
            self._location_label.setText("Save start is only available in Stop mode")
            return
        payload = self._saved_start_pose_payload()
        if payload is None:
            self._location_label.setText("No car pose available yet to save")
            self._refresh_manual_pose_controls()
            return
        try:
            self._saved_start_pose = save_saved_start_pose(payload)
        except Exception as exc:
            _logger.warning("Could not save sim start pose: %s", exc)
            self._location_label.setText("Could not save start pose")
            return
        self._update_saved_start_label()
        self._location_label.setText(
            f"Start saved → ({float(payload['world_x']):.2f}, {float(payload['world_y']):.2f}) m"
        )

    # ------------------------------------------------------------------
    # Backend → us
    # ------------------------------------------------------------------
    def _on_location(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        try:
            x_m = float(payload.get("x", 0.0))
            y_m = float(payload.get("y", 0.0))
            yaw_deg = float(payload.get("yaw", 0.0))
        except (TypeError, ValueError):
            return
        self._last_location_time = _time.monotonic()
        self._last_location_yaw_deg = yaw_deg
        self._remember_pose_snapshot(x_m, y_m, yaw_deg)
        self._set_cursor_pose(x_m, y_m, yaw_deg)

    def _on_cars(self, payload) -> None:
        # Refresh the "other cars" layer.
        for it in self._cars_items:
            self._scene.removeItem(it)
        self._cars_items.clear()
        if not isinstance(payload, list):
            return
        brush = QBrush(QColor(255, 200, 0))
        pen = QPen(QColor("#000"), 1)
        for car in payload:
            if not isinstance(car, dict):
                continue
            try:
                x = float(car.get("x", 0))
                y = float(car.get("y", 0))
            except (TypeError, ValueError):
                continue
            px, py = self._data.world_to_pixel(x, y)
            r = 6
            ell = QGraphicsEllipseItem(px - r, py - r, 2 * r, 2 * r)
            ell.setBrush(brush)
            ell.setPen(pen)
            ell.setZValue(60)
            self._scene.addItem(ell)
            self._cars_items.append(ell)

    def _on_semaphores(self, payload) -> None:
        for it in self._semaphore_items:
            self._scene.removeItem(it)
        self._semaphore_items.clear()
        if not isinstance(payload, list):
            return
        for s in payload:
            if not isinstance(s, dict):
                continue
            try:
                x = float(s.get("x", 0))
                y = float(s.get("y", 0))
            except (TypeError, ValueError):
                continue
            state = str(s.get("state", "")).lower()
            color = {
                "red":    QColor(255, 60, 60),
                "yellow": QColor(255, 200, 60),
                "green":  QColor(60, 220, 60),
            }.get(state, QColor(150, 150, 150))
            px, py = self._data.world_to_pixel(x, y)
            r = 7
            ell = QGraphicsEllipseItem(px - r, py - r, 2 * r, 2 * r)
            ell.setBrush(QBrush(color))
            ell.setPen(QPen(QColor("#000"), 1))
            ell.setZValue(70)
            self._scene.addItem(ell)
            self._semaphore_items.append(ell)

    def _on_nav_status(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        route_active = bool(payload.get("route_active", False))
        if route_active:
            self._active_current_lanelet_id = (
                str(payload.get("current_lanelet_id")).strip()
                if payload.get("current_lanelet_id") is not None
                else None
            ) or None
            self._active_next_lanelet_ids = tuple(
                str(item).strip()
                for item in (payload.get("next_lanelet_ids") or [])
                if str(item).strip()
            )
        else:
            self._active_current_lanelet_id = None
            self._active_next_lanelet_ids = ()
            self._control_path.setPath(QPainterPath())
        self._apply_lanelet_highlights(
            current_lanelet_id=self._active_current_lanelet_id,
            next_lanelet_ids=self._active_next_lanelet_ids,
        )

        route_source = str(payload.get("route_source") or "").strip().lower()
        if route_source == "reference":
            self._nav_path.setPath(QPainterPath())
            return

        route_preview_points = extract_nav_route_preview_points(payload)
        if len(route_preview_points) < 2:
            self._nav_path.setPath(QPainterPath())
            return
        self._nav_path.setPath(
            _build_world_path(route_preview_points, world_to_pixel=self._data.world_to_pixel)
        )

    def _on_behavior_output(self, payload) -> None:
        control_path_points = extract_control_path_points(payload)
        if len(control_path_points) < 2:
            self._control_path.setPath(QPainterPath())
            return
        self._control_path.setPath(
            _build_world_path(control_path_points, world_to_pixel=self._data.world_to_pixel)
        )

    def _on_state_change(self, mode: str) -> None:
        if mode:
            self._mode = mode.lower()
        if (
            hasattr(self, "_map_edit_btn")
            and self._map_edit_btn.isChecked()
            and not self._is_stop_mode()
        ):
            self._map_edit_btn.setChecked(False)
        self._refresh_map_calibration_controls()
        self._refresh_manual_pose_controls()

    def _on_gps_fix(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        xy = self._payload_xy(payload)
        if xy is None:
            return
        x_m, y_m = xy
        px, py = self._data.world_to_pixel(x_m, y_m)
        self._gps_dot.setPos(px, py)
        self._gps_dot.setVisible(True)
        self._ensure_scene_contains(px, py)
        try:
            gps_yaw_deg = float(
                payload.get("yaw_deg")
                if payload.get("yaw_deg") is not None
                else math.degrees(float(payload.get("yaw_rad")))
                if payload.get("yaw_rad") is not None
                else self._last_location_yaw_deg
            )
        except (TypeError, ValueError):
            gps_yaw_deg = float(self._last_location_yaw_deg)
        self._last_gps_fix_time = _time.monotonic()
        self._last_gps_fix_payload = dict(payload)
        if self._last_location_time <= 0.0:
            self._set_cursor_pose(x_m, y_m, gps_yaw_deg)
        self._update_gps_age_label()

    def _on_dead_reckoning(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        if not payload.get("anchored"):
            self._dr_dot.setVisible(False)
            return
        try:
            x_m = float(payload["x"])
            y_m = float(payload["y"])
        except (TypeError, ValueError, KeyError):
            return
        px, py = self._data.world_to_pixel(x_m, y_m)
        self._dr_dot.setPos(px, py)
        self._dr_dot.setVisible(True)

    def _update_gps_age_label(self) -> None:
        if self._last_gps_fix_time <= 0.0:
            self._gps_label.setText("GPS: --")
            self._gps_label.setStyleSheet("color:#666; font-size: 10pt;")
            return
        age_s = _time.monotonic() - self._last_gps_fix_time
        if age_s < 2.0:
            color, style = "#4cd964", "bold"   # green — fresh fix
        elif age_s < 5.0:
            color, style = "#ffcc00", "normal" # yellow — aging
        else:
            color, style = "#ff3b30", "normal" # red — stale
        self._gps_label.setText(f"GPS: {age_s:.1f}s")
        self._gps_label.setStyleSheet(
            f"color:{color}; font-size:10pt; font-weight:{style};"
        )

    # ------------------------------------------------------------------
    # Click handling: relocate vs. navigate vs. snap-to-node.
    # ------------------------------------------------------------------
    def _on_relocate_toggled(self, checked: bool) -> None:
        """Switch the click semantics. Visual cue: tinted toolbar button +
        cross-hair cursor over the viewport so the user knows the next click
        will move the car, not plan a route.
        """
        if checked and not self._is_stop_mode():
            if hasattr(self, "_relocate_btn"):
                self._relocate_btn.blockSignals(True)
                self._relocate_btn.setChecked(False)
                self._relocate_btn.blockSignals(False)
            self._location_label.setText("Relocate is only available in Stop mode")
            checked = False
        self._relocate_mode = checked
        if checked:
            if hasattr(self, "_relocate_btn"):
                self._relocate_btn.setStyleSheet(
                    "QToolButton { background:#cc3232; color:white; "
                    "padding:4px 10px; border-radius:4px; font-weight:600; }"
                )
            self._view.viewport().setCursor(Qt.CrossCursor)
        else:
            if hasattr(self, "_relocate_btn"):
                self._relocate_btn.setStyleSheet("")
            self._view.viewport().unsetCursor()

    def _on_map_click(self, x_m: float, y_m: float) -> None:
        # Two behaviours:
        #
        # 1. Relocate mode: any click → ``Localisation``. In sim this drops the
        #    car there; on the real car it calibrates the latest GPS fix to the
        #    clicked map point.
        # 2. Normal click → ``NavigationCommand`` (plan a route to this point).
        #    This includes AUTO mode + known node — the user wants navigation,
        #    not a teleport.
        if self._relocate_mode:
            if not self._is_stop_mode():
                self._location_label.setText("Relocate is only available in Stop mode")
                self._on_relocate_toggled(False)
                return
            payload = self._manual_localisation_payload(x_m, y_m)
            yaw_deg = float(payload.get("yaw_deg", self._last_location_yaw_deg))
            lanelet_id = payload.get("lanelet_id")
            self._remember_pose_snapshot(
                x_m,
                y_m,
                yaw_deg,
                lanelet_id=str(lanelet_id) if lanelet_id is not None else None,
            )
            self._set_cursor_pose(x_m, y_m, yaw_deg)
            self._client.emit_message(ev.CMD_LOCALISATION, payload)
            gps_raw_x = payload.get("gps_raw_world_x")
            gps_raw_y = payload.get("gps_raw_world_y")
            if gps_raw_x is not None and gps_raw_y is not None:
                self._location_label.setText(
                    f"Relocate/calibrate → ({x_m:.2f}, {y_m:.2f}) m "
                    f"from GPS ({float(gps_raw_x):.2f}, {float(gps_raw_y):.2f})"
                )
                return
            self._location_label.setText(
                f"Relocate/calibrate → ({x_m:.2f}, {y_m:.2f}) m"
            )
            return
        lanelet_id = resolve_click_destination_lanelet(
            x_m=x_m,
            y_m=y_m,
            lanelet_map=self._data.lanelet_map,
            lanelet_polygons=self._data.lanelet_polygons_by_id,
            current_lanelet_id=self._active_current_lanelet_id,
            next_lanelet_ids=self._active_next_lanelet_ids,
        )
        command = {"x": x_m, "y": y_m}
        if lanelet_id is not None:
            command["lanelet_id"] = str(lanelet_id)
        self._client.emit_message(ev.CMD_NAV_COMMAND, command)
        lanelet_text = f" · lanelet {lanelet_id}" if lanelet_id is not None else ""
        if self._mode == ev.MODE_AUTO:
            self._location_label.setText(
                f"Route sent → ({x_m:.2f}, {y_m:.2f}) m{lanelet_text}"
            )
        else:
            self._location_label.setText(
                f"Route armed → ({x_m:.2f}, {y_m:.2f}) m{lanelet_text} · switch to Auto to execute"
            )

    def _clear_route(self) -> None:
        self._client.emit_message(ev.CMD_NAV_COMMAND, {"mode": "clear"})
        self._active_current_lanelet_id = None
        self._active_next_lanelet_ids = ()
        self._apply_lanelet_highlights(
            current_lanelet_id=None,
            next_lanelet_ids=(),
        )
        self._nav_path.setPath(QPainterPath())
        self._control_path.setPath(QPainterPath())
        self._location_label.setText("Route cleared")
