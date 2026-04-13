"""
Track graph loader, global route builder, and dense waypoint generator.

This module keeps two responsibilities separate:
1. A topological directed graph loaded from GraphML for route planning.
2. Dense waypoint interpolation for whatever node sequence is currently active.

The legacy reference loop is still built for backwards compatibility with the
existing visualizer and as a safe default route when no destination was chosen.
Explicit navigation uses shortest-path planning over the graph instead.
"""

from __future__ import annotations

import heapq
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

try:
    from scipy.interpolate import CubicSpline

    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from src.hardware.tracking.trackSemantics import TrackSemantics

ATTR_NORMAL = 0
ATTR_CROSSWALK = 1
ATTR_INTERSECTION = 2
ATTR_ONEWAY = 3
ATTR_HIGHWAY_LEFT = 4
ATTR_HIGHWAY_RIGHT = 5
ATTR_ROUNDABOUT = 6
ATTR_STOPLINE = 7
ATTR_DOTTED = 8
ATTR_DOTTED_CROSSWALK = 9
ATTR_INTERSECTION_EXIT = 11  # End of intersection: keep waypoint mode until two lane lines detected

WAYPOINT_MODE_ATTRS = {ATTR_INTERSECTION, ATTR_STOPLINE, ATTR_INTERSECTION_EXIT}

_ATTR_ROUTE_WEIGHTS = {
    ATTR_NORMAL: 1.00,
    ATTR_CROSSWALK: 1.08,
    ATTR_INTERSECTION: 1.02,
    ATTR_ONEWAY: 0.98,
    ATTR_HIGHWAY_LEFT: 0.95,
    ATTR_HIGHWAY_RIGHT: 0.95,
    ATTR_ROUNDABOUT: 1.04,
    ATTR_STOPLINE: 1.10,
    ATTR_DOTTED: 1.00,
    ATTR_DOTTED_CROSSWALK: 1.08,
    ATTR_INTERSECTION_EXIT: 1.02,
}


@dataclass(frozen=True)
class TrackNode:
    node_id: str
    x: float
    y: float
    attribute: int = ATTR_NORMAL
    is_start: bool = False


@dataclass
class RoutePath:
    node_ids: list[str]
    waypoints: np.ndarray
    wp_node_attrs: np.ndarray
    wp_node_ids: list[str]
    closed_loop: bool = False
    route_id: str | None = None
    source: str = "graph"
    wp_edge_ids: list[str] = field(default_factory=list)
    wp_semantic_ids: list[str | None] = field(default_factory=list)
    wp_semantic_types: list[str] = field(default_factory=list)
    wp_zone_ids: list[list[str]] = field(default_factory=list)
    wp_zone_types: list[list[str]] = field(default_factory=list)
    route_events: list[dict] = field(default_factory=list)
    map_metadata: dict = field(default_factory=dict)
    available_destinations: list[dict] = field(default_factory=list)

    def preview_points(self, max_points: int = 140) -> list[dict[str, float]]:
        if self.waypoints.size == 0:
            return []
        total = int(self.waypoints.shape[0])
        step = max(1, total // max(1, int(max_points)))
        preview = self.waypoints[::step, :2]
        if total > 0 and (len(preview) == 0 or not np.allclose(preview[-1], self.waypoints[-1, :2])):
            preview = np.vstack([preview, self.waypoints[-1, :2]])
        return [
            {"x": round(float(x), 4), "y": round(float(y), 4)}
            for x, y in preview
        ]

    def destination_point(self) -> dict[str, float] | None:
        if self.waypoints.size == 0:
            return None
        x, y, _ = self.waypoints[-1]
        return {"x": round(float(x), 4), "y": round(float(y), 4)}


class TrackGraph:
    """Loads a GraphML track file and provides route/navigation utilities."""

    GRAPHML_NS = "http://graphml.graphdrawing.org/graphml"

    def __init__(self, graphml_path: str, step_m: float = 0.05, semantics_path: str | None = None):
        self.step_m = float(step_m)
        self.nodes: dict[str, TrackNode] = {}
        self.adj: dict[str, list[str]] = defaultdict(list)
        self.edge_lengths: dict[tuple[str, str], float] = {}
        self.edge_ids: dict[tuple[str, str], str] = {}

        self.reference_node_ids: list[str] = []
        self.reference_path: RoutePath = RoutePath([], np.empty((0, 3)), np.empty(0, dtype=int), [])

        # Backwards-compatible fields used by the visualizer and older code.
        self.ordered_nodes: list[TrackNode] = []
        self.waypoints: np.ndarray = np.empty((0, 3))
        self.wp_node_attrs: np.ndarray = np.empty(0, dtype=int)

        self._load(graphml_path)
        if semantics_path is None:
            base_dir = os.path.dirname(os.path.abspath(graphml_path))
            for candidate_name in ("Track Editor Save.json", "track_semantics.json", "Track Semantics.json"):
                candidate = os.path.join(base_dir, candidate_name)
                if os.path.exists(candidate):
                    semantics_path = candidate
                    break
        self.semantics = TrackSemantics(semantics_path)
        self.map_metadata = self._build_map_metadata()
        self._build_reference_path()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self, path: str) -> None:
        tree = ET.parse(path)
        root = tree.getroot()
        ns = self.GRAPHML_NS

        key_map = {}
        for key in root.findall(f"{{{ns}}}key"):
            name = key.get("attr.name", "")
            kid = key.get("id", "")
            key_map[name] = kid

        x_key = key_map.get("x", "d0")
        y_key = key_map.get("y", "d1")
        attr_key = key_map.get("new_attribute", "d2")
        start_key = key_map.get("is_start", "d3")

        graph = root.find(f"{{{ns}}}graph")
        if graph is None:
            raise ValueError(f"No <graph> element found in {path}")

        for node_el in graph.findall(f"{{{ns}}}node"):
            nid = node_el.get("id")
            if nid is None:
                continue
            data = {d.get("key"): d.text for d in node_el.findall(f"{{{ns}}}data")}
            x = float(data.get(x_key, 0.0))
            y = float(data.get(y_key, 0.0))
            attr = int(data.get(attr_key, ATTR_NORMAL) or ATTR_NORMAL)
            is_start = str(data.get(start_key, "false")).lower() == "true"
            self.nodes[nid] = TrackNode(nid, x, y, attr, is_start)

        for edge_el in graph.findall(f"{{{ns}}}edge"):
            src = edge_el.get("source")
            tgt = edge_el.get("target")
            if not src or not tgt or src not in self.nodes or tgt not in self.nodes:
                continue
            self.adj[src].append(tgt)
            self.edge_ids[(src, tgt)] = edge_el.get("id") or f"{src}->{tgt}"
            self.edge_lengths[(src, tgt)] = math.hypot(
                self.nodes[tgt].x - self.nodes[src].x,
                self.nodes[tgt].y - self.nodes[src].y,
            )

    def get_start_node_id(self) -> str:
        for nid, node in self.nodes.items():
            if node.is_start:
                return nid
        return min(self.nodes.keys(), key=lambda key: int(key))

    def _build_map_metadata(self) -> dict:
        xs = [node.x for node in self.nodes.values()]
        ys = [node.y for node in self.nodes.values()]
        x_min = min(xs) if xs else 0.0
        x_max = max(xs) if xs else 0.0
        y_min = min(ys) if ys else 0.0
        y_max = max(ys) if ys else 0.0
        default = {
            "coordinate_origin": "bottom_left",
            "y_axis_inverted": True,
            "width_m": round(float(x_max - x_min), 5),
            "height_m": round(float(y_max - y_min), 5),
            "world_bounds": {
                "x_min": round(float(x_min), 5),
                "x_max": round(float(x_max), 5),
                "y_min": round(float(y_min), 5),
                "y_max": round(float(y_max), 5),
            },
        }
        payload = dict(default)
        payload.update(dict(self.semantics.map_metadata or {}))
        mpp = payload.get("meters_per_pixel")
        ppm = payload.get("pixels_per_meter")
        try:
            if mpp and not ppm:
                payload["pixels_per_meter"] = round(1.0 / float(mpp), 6)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        try:
            if ppm and not mpp:
                payload["meters_per_pixel"] = round(1.0 / float(ppm), 9)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        return payload

    def get_map_metadata(self) -> dict:
        return dict(self.map_metadata)

    def get_available_destinations(self) -> list[dict]:
        destinations = self.semantics.get_available_destinations()
        if destinations:
            return destinations
        start_id = self.get_start_node_id() if self.nodes else None
        return [
            {"id": "reference_start", "label": "Reference Start", "node_id": start_id}
        ] if start_id else []

    def describe_destination(self, node_id: str | None) -> dict | None:
        if node_id is None:
            return None
        node_key = str(node_id)
        for item in self.get_available_destinations():
            if str(item.get("node_id")) == node_key or str(item.get("id")) == node_key:
                return dict(item)
        if node_key in self.nodes:
            return {"id": node_key, "label": f"Node {node_key}", "node_id": node_key}
        return None

    def _map_frame(self) -> dict[str, float | bool]:
        metadata = self.get_map_metadata()
        bounds = dict(metadata.get("world_bounds") or {})
        xs = [node.x for node in self.nodes.values()]
        ys = [node.y for node in self.nodes.values()]
        x_min = float(bounds.get("x_min", min(xs) if xs else 0.0))
        x_max = float(bounds.get("x_max", max(xs) if xs else x_min))
        y_min = float(bounds.get("y_min", min(ys) if ys else 0.0))
        y_max = float(bounds.get("y_max", max(ys) if ys else y_min))
        width_m = float(metadata.get("width_m", max(0.0, x_max - x_min)) or max(0.0, x_max - x_min))
        height_m = float(metadata.get("height_m", max(0.0, y_max - y_min)) or max(0.0, y_max - y_min))
        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "width_m": width_m,
            "height_m": height_m,
            "y_axis_inverted": bool(metadata.get("y_axis_inverted", False)),
        }

    def _reference_successor(self, node_id: str) -> str | None:
        ref_ids = [nid for nid in self.reference_node_ids if nid in self.nodes]
        if len(ref_ids) > 1 and ref_ids[0] == ref_ids[-1]:
            ref_ids = ref_ids[:-1]
        if node_id not in ref_ids:
            return None
        start_idx = ref_ids.index(node_id)
        for offset in range(1, len(ref_ids)):
            candidate = ref_ids[(start_idx + offset) % len(ref_ids)]
            if candidate != node_id:
                return candidate
        return None

    def get_node_pose(self, node_id: str | None) -> tuple[float, float, float] | None:
        if node_id is None:
            return None
        node_key = str(node_id)
        node = self.nodes.get(node_key)
        if node is None:
            return None

        successor_id = self._reference_successor(node_key)
        if successor_id is None:
            for candidate in self.adj.get(node_key, []):
                if candidate in self.nodes and candidate != node_key:
                    successor_id = candidate
                    break

        if successor_id is not None and successor_id in self.nodes:
            successor = self.nodes[successor_id]
            yaw = math.atan2(successor.y - node.y, successor.x - node.x)
            return float(node.x), float(node.y), float(yaw)

        for prev_id, next_ids in self.adj.items():
            if node_key in next_ids and prev_id in self.nodes:
                prev_node = self.nodes[prev_id]
                yaw = math.atan2(node.y - prev_node.y, node.x - prev_node.x)
                return float(node.x), float(node.y), float(yaw)

        return float(node.x), float(node.y), 0.0

    def get_map_nodes(self) -> list[dict]:
        def _sort_key(item: str):
            text = str(item)
            return (0, int(text)) if text.isdigit() else (1, text)

        payload: list[dict] = []
        for node_id in sorted(self.nodes.keys(), key=_sort_key):
            node = self.nodes[node_id]
            pose = self.get_node_pose(node_id)
            events = self.semantics.get_node_events(node_id)
            destination = self.describe_destination(node_id)
            primary_event = events[0] if events else None
            label = None
            if primary_event is not None:
                label = primary_event.get("label")
            if label is None and destination is not None:
                label = destination.get("label")
            if label is None:
                label = f"Node {node_id}"
            payload.append(
                {
                    "id": str(node_id),
                    "node_id": str(node_id),
                    "x": round(float(node.x), 5),
                    "y": round(float(node.y), 5),
                    "yaw": round(float(pose[2] if pose is not None else 0.0), 6),
                    "attr": int(node.attribute),
                    "is_start": bool(node.is_start),
                    "label": str(label),
                    "semantic_types": [
                        str(event.get("type"))
                        for event in events
                        if event.get("type") is not None
                    ],
                }
            )
        return payload

    def world_to_localisation_pose(
        self,
        x: float,
        y: float,
        *,
        yaw: float | None = None,
        timestamp: float | None = None,
    ) -> dict:
        frame = self._map_frame()
        local_x = float(x) - float(frame["x_min"])
        local_y = float(y) - float(frame["y_min"])
        pos_b = (
            float(frame["height_m"]) - local_y
            if bool(frame["y_axis_inverted"])
            else local_y
        )
        payload = {
            "timestamp": float(timestamp if timestamp is not None else 0.0),
            "posA": round(float(local_x), 6),
            "posB": round(float(pos_b), 6),
            "rotA": 0.0,
            "rotB": 0.0,
        }
        return payload

    def localisation_to_world_pose(
        self,
        payload: dict | None,
        *,
        default_yaw: float | None = None,
    ) -> tuple[float, float, float] | None:
        if not isinstance(payload, dict):
            return None
        try:
            pos_a = float(payload.get("posA"))
            pos_b = float(payload.get("posB"))
        except (TypeError, ValueError):
            return None

        frame = self._map_frame()
        x = float(frame["x_min"]) + pos_a
        if bool(frame["y_axis_inverted"]):
            y = float(frame["y_min"]) + (float(frame["height_m"]) - pos_b)
        else:
            y = float(frame["y_min"]) + pos_b

        yaw = float(default_yaw if default_yaw is not None else 0.0)
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        try:
            if payload.get("yaw_rad") is not None:
                yaw = float(payload.get("yaw_rad"))
            elif payload.get("yaw_deg") is not None:
                yaw = math.radians(float(payload.get("yaw_deg")))
            else:
                node_pose = self.get_node_pose(
                    self.resolve_node_id(meta.get("node_id") or payload.get("node_id"))
                )
                if node_pose is not None:
                    yaw = float(node_pose[2])
        except (TypeError, ValueError):
            pass

        return float(x), float(y), float(yaw)

    # ------------------------------------------------------------------
    # Reference loop (compatibility / default route)
    # ------------------------------------------------------------------
    def _build_reference_sequence(self, start_id: str) -> list[str]:
        visited = set()
        path: list[str] = []
        current = start_id
        while current is not None and current not in visited:
            visited.add(current)
            if current not in self.nodes:
                break
            path.append(current)
            successors = self.adj.get(current, [])
            next_node = None
            for succ in successors:
                if succ not in visited:
                    next_node = succ
                    break
            if next_node is None and successors and start_id in successors and len(path) > 1:
                path.append(start_id)
                break
            current = next_node
        return path

    def _build_reference_path(self) -> None:
        start_id = self.get_start_node_id()
        reference_ids = self._build_reference_sequence(start_id)
        if len(reference_ids) < 2 and self.nodes:
            reference_ids = list(self.nodes.keys())

        closed_loop = len(reference_ids) > 1 and reference_ids[0] == reference_ids[-1]
        if not closed_loop and len(reference_ids) > 1:
            last_id = reference_ids[-1]
            first_id = reference_ids[0]
            if first_id in self.adj.get(last_id, []):
                reference_ids = reference_ids + [first_id]
                closed_loop = True

        self.reference_node_ids = list(reference_ids)
        self.reference_path = self.build_dense_path(
            reference_ids,
            closed_loop=closed_loop,
            route_id="reference_loop",
            source="reference",
        )
        self.ordered_nodes = [self.nodes[nid] for nid in reference_ids if nid in self.nodes]
        self.waypoints = self.reference_path.waypoints
        self.wp_node_attrs = self.reference_path.wp_node_attrs

    # ------------------------------------------------------------------
    # Topological planning
    # ------------------------------------------------------------------
    def _edge_cost(self, src: str, tgt: str) -> float:
        base = self.edge_lengths.get((src, tgt))
        if base is None:
            src_node = self.nodes[src]
            tgt_node = self.nodes[tgt]
            base = math.hypot(tgt_node.x - src_node.x, tgt_node.y - src_node.y)
        weight = _ATTR_ROUTE_WEIGHTS.get(self.nodes[tgt].attribute, 1.0)
        return float(base) * float(weight)

    def shortest_path(self, start_id: str, dest_id: str) -> list[str]:
        start_id = str(start_id)
        dest_id = str(dest_id)
        if start_id not in self.nodes or dest_id not in self.nodes:
            return []
        if start_id == dest_id:
            return [start_id]

        dist = {start_id: 0.0}
        prev: dict[str, str] = {}
        heap: list[tuple[float, str]] = [(0.0, start_id)]
        visited = set()

        while heap:
            cost, node_id = heapq.heappop(heap)
            if node_id in visited:
                continue
            visited.add(node_id)
            if node_id == dest_id:
                break
            for succ in self.adj.get(node_id, []):
                edge_cost = cost + self._edge_cost(node_id, succ)
                if edge_cost < dist.get(succ, float("inf")):
                    dist[succ] = edge_cost
                    prev[succ] = node_id
                    heapq.heappush(heap, (edge_cost, succ))

        if dest_id not in dist:
            return []

        path = [dest_id]
        cur = dest_id
        while cur != start_id:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path

    def reference_path_between(self, start_id: str, dest_id: str) -> list[str]:
        if not self.reference_node_ids:
            return []
        ref_ids = list(self.reference_node_ids)
        if len(ref_ids) > 1 and ref_ids[0] == ref_ids[-1]:
            ref_ids = ref_ids[:-1]
        if start_id not in ref_ids or dest_id not in ref_ids:
            return []
        start_idx = ref_ids.index(start_id)
        dest_idx = ref_ids.index(dest_id)
        if start_idx <= dest_idx:
            return ref_ids[start_idx:dest_idx + 1]
        return ref_ids[start_idx:] + ref_ids[:dest_idx + 1]

    def find_nearest_node(self, x: float, y: float, candidate_ids: Iterable[str] | None = None) -> str | None:
        ids = list(candidate_ids) if candidate_ids is not None else list(self.nodes.keys())
        if not ids:
            return None
        best_id = None
        best_dist = None
        for node_id in ids:
            node = self.nodes.get(str(node_id))
            if node is None:
                continue
            dist = math.hypot(float(x) - node.x, float(y) - node.y)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_id = node.node_id
        return best_id

    def resolve_node_id(self, spec) -> str | None:
        if spec is None:
            return None
        destination = self.semantics.resolve_destination(spec)
        if destination is not None:
            return str(destination.get("node_id"))
        if isinstance(spec, dict):
            if "node_id" in spec:
                candidate = str(spec["node_id"])
                return candidate if candidate in self.nodes else None
            if "id" in spec:
                candidate = str(spec["id"])
                return candidate if candidate in self.nodes else None
            if "x" in spec and "y" in spec:
                try:
                    return self.find_nearest_node(float(spec["x"]), float(spec["y"]))
                except (TypeError, ValueError):
                    return None
            return None
        candidate = str(spec)
        if candidate in self.nodes:
            return candidate
        destination = self.semantics.resolve_destination(candidate)
        if destination is not None:
            return str(destination.get("node_id"))
        try:
            numeric = str(int(float(spec)))
            return numeric if numeric in self.nodes else None
        except (TypeError, ValueError):
            return None

    def go_to(self, start_spec, dest_spec) -> RoutePath:
        start_id = self.resolve_node_id(start_spec)
        dest_id = self.resolve_node_id(dest_spec)
        if start_id is None or dest_id is None:
            return RoutePath([], np.empty((0, 3)), np.empty(0, dtype=int), [])

        node_ids = self.shortest_path(start_id, dest_id)
        if not node_ids:
            node_ids = self.reference_path_between(start_id, dest_id)
        if not node_ids:
            return RoutePath([], np.empty((0, 3)), np.empty(0, dtype=int), [])
        return self.build_dense_path(node_ids, closed_loop=False, source="go_to")

    def go_to_multiple(self, start_spec, destinations) -> RoutePath:
        start_id = self.resolve_node_id(start_spec)
        if start_id is None:
            return RoutePath([], np.empty((0, 3)), np.empty(0, dtype=int), [])

        dest_ids = []
        for spec in destinations or []:
            node_id = self.resolve_node_id(spec)
            if node_id is not None:
                dest_ids.append(node_id)
        if not dest_ids:
            return RoutePath([], np.empty((0, 3)), np.empty(0, dtype=int), [])

        full_sequence: list[str] = []
        current = start_id
        for dest_id in dest_ids:
            segment = self.shortest_path(current, dest_id)
            if not segment:
                segment = self.reference_path_between(current, dest_id)
            if not segment:
                continue
            if full_sequence and segment:
                segment = segment[1:]
            full_sequence.extend(segment if full_sequence else segment)
            current = dest_id

        if not full_sequence:
            return RoutePath([], np.empty((0, 3)), np.empty(0, dtype=int), [])
        return self.build_dense_path(full_sequence, closed_loop=False, source="go_to_multiple")

    # ------------------------------------------------------------------
    # Dense waypoint generation
    # ------------------------------------------------------------------
    @staticmethod
    def _semantic_priority(event: dict | None) -> int:
        semantic_type = str((event or {}).get("type") or "")
        order = {
            "control": 0,
            "intersection": 1,
            "crosswalk": 2,
            "parking_spot": 3,
            "destination": 4,
            "zone": 5,
        }
        return order.get(semantic_type, 99)

    def _default_attr_semantic(self, node_id: str, attr: int) -> dict | None:
        attr = int(attr)
        if attr == ATTR_STOPLINE:
            return {
                "id": f"attr-stopline-{node_id}",
                "type": "stopline",
                "label": "Stopline",
                "anchor_node_id": node_id,
                "control_type": "stop",
            }
        if attr == ATTR_INTERSECTION:
            return {
                "id": f"attr-intersection-{node_id}",
                "type": "intersection",
                "label": "Intersection",
                "anchor_node_id": node_id,
            }
        if attr == ATTR_CROSSWALK:
            return {
                "id": f"attr-crosswalk-{node_id}",
                "type": "crosswalk",
                "label": "Crosswalk",
                "anchor_node_id": node_id,
            }
        if attr == ATTR_ROUNDABOUT:
            return {
                "id": f"attr-roundabout-{node_id}",
                "type": "roundabout",
                "label": "Roundabout",
                "anchor_node_id": node_id,
            }
        if attr in (ATTR_HIGHWAY_LEFT, ATTR_HIGHWAY_RIGHT):
            return {
                "id": f"attr-highway-{node_id}",
                "type": "zone",
                "label": "Highway",
                "anchor_node_id": node_id,
                "zone_type": "highway",
            }
        if attr == ATTR_ONEWAY:
            return {
                "id": f"attr-oneway-{node_id}",
                "type": "zone",
                "label": "One Way",
                "anchor_node_id": node_id,
                "zone_type": "oneway",
            }
        return None

    def _annotate_route(self, route: RoutePath) -> RoutePath:
        route.map_metadata = self.get_map_metadata()
        route.available_destinations = self.get_available_destinations()
        if not route.node_ids or not route.wp_node_ids:
            return route

        annotations = self.semantics.build_route_annotations(
            route.node_ids,
            closed_loop=route.closed_loop,
        )
        node_events_by_index = annotations.get("node_events_by_index", [])
        zone_ids_by_index = annotations.get("zone_ids_by_index", [])
        zone_types_by_index = annotations.get("zone_types_by_index", [])
        route.route_events = list(annotations.get("route_events", []))

        node_index_by_id = {
            str(node_id): idx
            for idx, node_id in enumerate(route.node_ids)
        }
        route.wp_semantic_ids = []
        route.wp_semantic_types = []
        route.wp_zone_ids = []
        route.wp_zone_types = []

        seen_route_events = {
            (str(item.get("type")), str(item.get("id")))
            for item in route.route_events
        }

        for wp_idx, node_id in enumerate(route.wp_node_ids):
            route_idx = node_index_by_id.get(str(node_id), 0)
            node_events = list(node_events_by_index[route_idx]) if route_idx < len(node_events_by_index) else []
            zone_ids = list(zone_ids_by_index[route_idx]) if route_idx < len(zone_ids_by_index) else []
            zone_types = list(zone_types_by_index[route_idx]) if route_idx < len(zone_types_by_index) else []

            default_event = self._default_attr_semantic(
                str(node_id),
                int(route.wp_node_attrs[wp_idx]) if wp_idx < len(route.wp_node_attrs) else ATTR_NORMAL,
            )
            if default_event is not None:
                node_events.append(default_event)
                key = (str(default_event.get("type")), str(default_event.get("id")))
                if key not in seen_route_events:
                    seen_route_events.add(key)
                    route.route_events.append(
                        {**default_event, "route_node_index": route_idx, "node_id": str(node_id)}
                    )

            node_events.sort(key=self._semantic_priority)
            primary = node_events[0] if node_events else None
            route.wp_semantic_ids.append(primary.get("id") if primary else None)
            route.wp_semantic_types.append(str(primary.get("type") or "lane_follow") if primary else "lane_follow")
            route.wp_zone_ids.append(zone_ids)
            route.wp_zone_types.append(zone_types)

        first_wp_idx_by_node_id = {}
        for wp_idx, node_id in enumerate(route.wp_node_ids):
            first_wp_idx_by_node_id.setdefault(str(node_id), wp_idx)
        for event in route.route_events:
            node_id = str(event.get("node_id") or "")
            event["waypoint_idx"] = int(first_wp_idx_by_node_id.get(node_id, 0))
        route.route_events.sort(key=lambda item: (int(item.get("route_node_index", 0)), str(item.get("id", ""))))
        return route

    def _finalize_route(self, route: RoutePath) -> RoutePath:
        route.map_metadata = self.get_map_metadata()
        route.available_destinations = self.get_available_destinations()
        if not route.wp_edge_ids:
            route.wp_edge_ids = ["" for _ in route.wp_node_ids]
        return self._annotate_route(route)

    def build_dense_path(
        self,
        node_ids: Iterable[str],
        closed_loop: bool | None = None,
        step_m: float | None = None,
        route_id: str | None = None,
        source: str = "graph",
    ) -> RoutePath:
        clean_ids = [str(node_id) for node_id in node_ids if str(node_id) in self.nodes]
        if not clean_ids:
            return self._finalize_route(
                RoutePath([], np.empty((0, 3)), np.empty(0, dtype=int), [], False, route_id, source)
            )

        deduped = [clean_ids[0]]
        for node_id in clean_ids[1:]:
            if node_id != deduped[-1]:
                deduped.append(node_id)

        if closed_loop is None:
            closed_loop = len(deduped) > 1 and deduped[0] == deduped[-1]
        loop_ids = list(deduped)
        if closed_loop and len(loop_ids) > 1 and loop_ids[0] != loop_ids[-1]:
            loop_ids.append(loop_ids[0])

        if len(loop_ids) == 1:
            node = self.nodes[loop_ids[0]]
            waypoints = np.array([[node.x, node.y, 0.0]], dtype=float)
            attrs = np.array([node.attribute], dtype=int)
            return self._finalize_route(RoutePath(
                node_ids=list(deduped),
                waypoints=waypoints,
                wp_node_attrs=attrs,
                wp_node_ids=[node.node_id],
                closed_loop=bool(closed_loop),
                route_id=route_id,
                source=source,
                wp_edge_ids=[""],
            ))

        xs = np.array([self.nodes[node_id].x for node_id in loop_ids], dtype=float)
        ys = np.array([self.nodes[node_id].y for node_id in loop_ids], dtype=float)
        attrs = np.array([self.nodes[node_id].attribute for node_id in loop_ids], dtype=int)

        dists = np.hypot(np.diff(xs), np.diff(ys))
        dists = np.maximum(dists, 1e-6)
        t = np.concatenate([[0.0], np.cumsum(dists)])
        total_len = float(t[-1])
        step = float(step_m or self.step_m)

        if total_len <= 1e-6:
            waypoints = np.column_stack([xs[:1], ys[:1], np.zeros(1, dtype=float)])
            return self._finalize_route(RoutePath(
                node_ids=list(deduped),
                waypoints=waypoints,
                wp_node_attrs=np.array([attrs[0]], dtype=int),
                wp_node_ids=[loop_ids[0]],
                closed_loop=bool(closed_loop),
                route_id=route_id,
                source=source,
                wp_edge_ids=[""],
            ))

        if closed_loop:
            n_pts = max(3, int(total_len / max(step, 1e-6)))
            t_dense = np.linspace(0.0, total_len, n_pts, endpoint=False)
        else:
            n_pts = max(2, int(total_len / max(step, 1e-6)) + 1)
            t_dense = np.linspace(0.0, total_len, n_pts)

        if _SCIPY_OK and len(loop_ids) >= 3:
            cs_x = CubicSpline(t, xs, bc_type="not-a-knot")
            cs_y = CubicSpline(t, ys, bc_type="not-a-knot")
            wx = cs_x(t_dense)
            wy = cs_y(t_dense)
            dx = cs_x(t_dense, 1)
            dy = cs_y(t_dense, 1)
        else:
            wx = np.interp(t_dense, t, xs)
            wy = np.interp(t_dense, t, ys)
            dx = np.gradient(wx, t_dense)
            dy = np.gradient(wy, t_dense)

        psi = np.arctan2(dy, dx)
        waypoints = np.column_stack([wx, wy, psi])

        wp_attrs = np.zeros(len(t_dense), dtype=int)
        wp_node_ids: list[str] = []
        wp_edge_ids: list[str] = []
        attr_limit = max(0, len(attrs) - 2) if len(attrs) > 1 else 0
        for td in t_dense:
            idx = int(np.searchsorted(t, td, side="right")) - 1
            idx = max(0, min(idx, attr_limit if len(attrs) > 1 else 0))
            wp_attrs[len(wp_node_ids)] = attrs[idx]
            wp_node_ids.append(loop_ids[idx])
            next_idx = min(idx + 1, len(loop_ids) - 1)
            edge_key = (loop_ids[idx], loop_ids[next_idx]) if next_idx != idx else None
            wp_edge_ids.append(self.edge_ids.get(edge_key, "") if edge_key else "")

        return self._finalize_route(RoutePath(
            node_ids=list(deduped),
            waypoints=waypoints,
            wp_node_attrs=wp_attrs,
            wp_node_ids=wp_node_ids,
            closed_loop=bool(closed_loop),
            route_id=route_id,
            source=source,
            wp_edge_ids=wp_edge_ids,
        ))

    # ------------------------------------------------------------------
    # Backwards-compatible path utilities (reference path only)
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        while angle_rad > math.pi:
            angle_rad -= 2.0 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2.0 * math.pi
        return float(angle_rad)

    def find_closest_waypoint(self, x: float, y: float, search_start: int = 0, search_window: int = 60) -> int:
        n = len(self.waypoints)
        if n == 0:
            return 0
        start = int(search_start) % n if self.reference_path.closed_loop else max(0, min(n - 1, int(search_start)))
        window = max(1, int(search_window))
        if n <= window + 5 or not self.reference_path.closed_loop:
            lo = max(0, start - 5)
            hi = min(n, start + window)
            idxs = np.arange(lo, hi, dtype=int)
        else:
            idxs = np.array([(start + off) % n for off in range(-5, window)], dtype=int)
        sub = self.waypoints[idxs, :2]
        dists = np.hypot(sub[:, 0] - x, sub[:, 1] - y)
        return int(idxs[int(np.argmin(dists))])

    def find_waypoint_ahead(self, x: float, y: float, current_idx: int, lookahead_m: float = 0.30) -> int:
        n = len(self.waypoints)
        if n == 0:
            return 0
        steps = max(1, round(lookahead_m / self.step_m))
        if self.reference_path.closed_loop:
            return (current_idx + steps) % n
        return min(n - 1, current_idx + steps)

    def project_pose_to_path(
        self,
        x: float,
        y: float,
        yaw: float,
        search_center: int = 0,
        search_window: int = 18,
        distance_weight: float = 1.0,
        heading_weight: float = 0.35,
    ) -> dict:
        n = len(self.waypoints)
        if n == 0:
            return {
                "matched_idx": 0,
                "matched_x": float(x),
                "matched_y": float(y),
                "path_psi": float(yaw),
                "lateral_error_m": 0.0,
                "heading_error_rad": 0.0,
                "map_match_error_m": 0.0,
                "score": 0.0,
            }
        if n == 1:
            wx, wy, psi = self.waypoints[0]
            return {
                "matched_idx": 0,
                "matched_x": float(wx),
                "matched_y": float(wy),
                "path_psi": float(psi),
                "lateral_error_m": 0.0,
                "heading_error_rad": self._wrap_angle(float(yaw) - float(psi)),
                "map_match_error_m": math.hypot(float(x) - float(wx), float(y) - float(wy)),
                "score": math.hypot(float(x) - float(wx), float(y) - float(wy)),
            }

        center = int(search_center)
        if self.reference_path.closed_loop:
            center %= n
            seg_idxs = [(center + off) % n for off in range(-search_window, search_window + 1)]
        else:
            lo = max(0, center - search_window)
            hi = min(n - 1, center + search_window)
            seg_idxs = list(range(lo, hi))
            if not seg_idxs:
                seg_idxs = [max(0, min(n - 2, center))]

        continuity_weight = max(0.05, 0.25 * float(distance_weight))
        best = None
        for seg_idx in seg_idxs:
            next_idx = (seg_idx + 1) % n if self.reference_path.closed_loop else min(seg_idx + 1, n - 1)
            if next_idx == seg_idx:
                continue
            x0, y0, _ = self.waypoints[seg_idx]
            x1, y1, _ = self.waypoints[next_idx]
            seg_dx = float(x1) - float(x0)
            seg_dy = float(y1) - float(y0)
            seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
            if seg_len2 <= 1e-9:
                continue

            rel_x = float(x) - float(x0)
            rel_y = float(y) - float(y0)
            proj_t = max(0.0, min(1.0, (rel_x * seg_dx + rel_y * seg_dy) / seg_len2))
            proj_x = float(x0) + proj_t * seg_dx
            proj_y = float(y0) + proj_t * seg_dy
            seg_psi = math.atan2(seg_dy, seg_dx)
            dx = float(x) - proj_x
            dy = float(y) - proj_y
            projection_error_m = math.hypot(dx, dy)
            lateral_error_m = -dx * math.sin(seg_psi) + dy * math.cos(seg_psi)
            heading_error_rad = self._wrap_angle(float(yaw) - seg_psi)
            continuity_m = abs(seg_idx - center) * float(self.step_m)
            score = (
                float(distance_weight) * projection_error_m
                + float(heading_weight) * abs(heading_error_rad)
                + continuity_weight * continuity_m
            )
            matched_idx = next_idx if proj_t >= 0.5 else seg_idx
            candidate = {
                "matched_idx": int(matched_idx),
                "matched_x": float(proj_x),
                "matched_y": float(proj_y),
                "path_psi": float(seg_psi),
                "lateral_error_m": float(lateral_error_m),
                "heading_error_rad": float(heading_error_rad),
                "map_match_error_m": float(projection_error_m),
                "score": float(score),
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

        if best is not None:
            return best

        idx = self.find_closest_waypoint(float(x), float(y), search_start=center, search_window=search_window)
        wx, wy, psi = self.waypoints[idx]
        return {
            "matched_idx": int(idx),
            "matched_x": float(wx),
            "matched_y": float(wy),
            "path_psi": float(psi),
            "lateral_error_m": 0.0,
            "heading_error_rad": self._wrap_angle(float(yaw) - float(psi)),
            "map_match_error_m": math.hypot(float(x) - float(wx), float(y) - float(wy)),
            "score": math.hypot(float(x) - float(wx), float(y) - float(wy)),
        }

    def compute_tracking_error(self, x: float, y: float, yaw: float, wp_idx: int) -> tuple[float, float]:
        if len(self.waypoints) == 0:
            return 0.0, 0.0
        idx = int(max(0, min(len(self.waypoints) - 1, wp_idx)))
        xr, yr, psi_r = self.waypoints[idx]
        dx = x - xr
        dy = y - yr
        error_m = -dx * math.sin(psi_r) + dy * math.cos(psi_r)
        heading_rad = self._wrap_angle(yaw - psi_r)
        return float(error_m), float(heading_rad)

    def is_precision_zone(self, wp_idx: int, lookahead_pts: int = 8) -> bool:
        n = len(self.waypoints)
        if n == 0:
            return False
        for i in range(max(1, int(lookahead_pts))):
            idx = (wp_idx + i) % n if self.reference_path.closed_loop else min(n - 1, wp_idx + i)
            if int(self.wp_node_attrs[idx]) in WAYPOINT_MODE_ATTRS:
                return True
        return False

    def get_curvature(self, wp_idx: int) -> float:
        n = len(self.waypoints)
        if n < 3:
            return 0.0
        idx = int(max(0, min(n - 1, wp_idx)))
        if self.reference_path.closed_loop:
            i_prev = (idx - 1) % n
            i_next = (idx + 1) % n
        else:
            i_prev = max(0, idx - 1)
            i_next = min(n - 1, idx + 1)
            if i_prev == idx or i_next == idx:
                return 0.0
        dpsi = float(self.waypoints[i_next][2]) - float(self.waypoints[i_prev][2])
        while dpsi > math.pi:
            dpsi -= 2.0 * math.pi
        while dpsi < -math.pi:
            dpsi += 2.0 * math.pi
        return dpsi / max(2.0 * self.step_m, 1e-6)

    def get_start_pose(self) -> tuple[float, float, float]:
        start_id = self.get_start_node_id()
        node = self.nodes[start_id]
        next_id = None
        for candidate in self.reference_node_ids:
            if candidate != start_id:
                next_id = candidate
                break
        if next_id is not None and next_id in self.nodes:
            next_node = self.nodes[next_id]
            yaw = math.atan2(next_node.y - node.y, next_node.x - node.x)
        elif len(self.waypoints) > 0:
            yaw = float(self.waypoints[0][2])
        else:
            yaw = 0.0
        return node.x, node.y, yaw
