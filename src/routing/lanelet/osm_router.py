from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.routing.lanelet.attributes import (
    ATTR_CROSSWALK,
    ATTR_HIGHWAY_LEFT,
    ATTR_HIGHWAY_RIGHT,
    ATTR_INTERSECTION,
    ATTR_NORMAL,
    ATTR_ONEWAY,
    ATTR_ROUNDABOUT,
    ATTR_ROUTE_WEIGHTS,
    ATTR_STOPLINE,
)
from src.routing.lanelet.from_osm import load_lanelet2_osm
from src.routing.lanelet.route_path import RoutePath


@dataclass(frozen=True)
class RouteAnchor:
    node_id: str
    x: float
    y: float
    attribute: int = ATTR_NORMAL
    is_start: bool = False


class OsmRouteGraph:
    """OSM-only route handler over Lanelet geometry.

    Mirrors the role that Autoware's route_handler plays over a Lanelet map:
    global routing runs on lanelet topology, while the output path is the
    concatenated centerline corridor in the exact same frame as the map.
    """

    def __init__(
        self,
        osm_path: str,
        *,
        step_m: float = 0.05,
        start_lanelet_id: str | None = None,
    ) -> None:
        self.step_m = float(step_m)
        self._osm_path = os.path.abspath(osm_path)
        self.lanelet_map = load_lanelet2_osm(self._osm_path, step_m=self.step_m)
        self.map_metadata = self.lanelet_map.get_map_metadata()

        self._start_lanelet_id = self._resolve_start_lanelet_id(start_lanelet_id)
        self.reference_node_ids, closed_loop = self._build_reference_loop(self._start_lanelet_id)
        self.reference_path = self.build_dense_path(
            self.reference_node_ids,
            closed_loop=closed_loop,
            route_id="reference_loop",
            source="reference",
        )
        self.waypoints = self.reference_path.waypoints
        self.wp_node_attrs = self.reference_path.wp_node_attrs
        self.ordered_nodes = self._build_ordered_nodes()

    def get_map_metadata(self) -> dict:
        return dict(self.map_metadata)

    def get_start_node_id(self) -> str:
        return self._start_lanelet_id

    def get_start_pose(self) -> tuple[float, float, float]:
        lanelet = self.lanelet_map.get_lanelet(self._start_lanelet_id)
        if lanelet is None or lanelet.centerline.shape[0] == 0:
            return 0.0, 0.0, 0.0
        x = float(lanelet.centerline[0, 0])
        y = float(lanelet.centerline[0, 1])
        yaw = _heading_of_centerline(lanelet.centerline)
        return x, y, yaw

    def get_available_destinations(self) -> list[dict]:
        x0, y0, _yaw0 = self.get_start_pose()
        return [{
            "id": "reference_start",
            "label": "Reference Start",
            "lanelet_id": self._start_lanelet_id,
            "x": round(float(x0), 4),
            "y": round(float(y0), 4),
        }]

    def describe_destination(self, spec) -> dict | None:
        lanelet_id = self.resolve_node_id(spec)
        if lanelet_id is None:
            if isinstance(spec, dict) and "x" in spec and "y" in spec:
                try:
                    return {
                        "id": "point",
                        "label": "Point goal",
                        "x": round(float(spec["x"]), 4),
                        "y": round(float(spec["y"]), 4),
                    }
                except (TypeError, ValueError):
                    return None
            return None
        lanelet = self.lanelet_map.get_lanelet(lanelet_id)
        if lanelet is None or lanelet.centerline.shape[0] == 0:
            return {"id": str(lanelet_id), "label": f"Lanelet {lanelet_id}", "lanelet_id": lanelet_id}
        end_xy = lanelet.centerline[-1]
        return {
            "id": str(lanelet_id),
            "label": f"Lanelet {lanelet_id}",
            "lanelet_id": str(lanelet_id),
            "x": round(float(end_xy[0]), 4),
            "y": round(float(end_xy[1]), 4),
        }

    def get_map_nodes(self) -> list[dict]:
        payload: list[dict] = []
        for lanelet_id in self.reference_node_ids:
            lanelet = self.lanelet_map.get_lanelet(str(lanelet_id))
            if lanelet is None or lanelet.centerline.shape[0] == 0:
                continue
            mid = lanelet.centerline[lanelet.centerline.shape[0] // 2]
            payload.append(
                {
                    "id": str(lanelet_id),
                    "lanelet_id": str(lanelet_id),
                    "x": round(float(mid[0]), 5),
                    "y": round(float(mid[1]), 5),
                    "yaw": round(float(_heading_of_centerline(lanelet.centerline)), 6),
                    "attr": int(lanelet.attribute),
                    "label": f"Lanelet {lanelet_id}",
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
        bounds = dict(self.map_metadata.get("world_bounds") or {})
        x_min = float(bounds.get("x_min", 0.0))
        y_min = float(bounds.get("y_min", 0.0))
        height_m = float(self.map_metadata.get("height_m", 0.0) or 0.0)
        local_x = float(x) - x_min
        local_y = float(y) - y_min
        pos_b = (
            height_m - local_y
            if bool(self.map_metadata.get("y_axis_inverted", False))
            else local_y
        )
        payload = {
            "timestamp": float(timestamp if timestamp is not None else 0.0),
            "posA": round(float(local_x), 6),
            "posB": round(float(pos_b), 6),
            "rotA": 0.0,
            "rotB": 0.0,
        }
        if yaw is not None:
            payload["yaw_rad"] = round(float(yaw), 6)
        return payload

    def localisation_to_world_pose(
        self,
        payload: dict | None,
        *,
        default_yaw: float | None = None,
    ) -> tuple[float, float, float] | None:
        if not isinstance(payload, dict):
            return None
        if "world_x" in payload and "world_y" in payload:
            try:
                x = float(payload["world_x"])
                y = float(payload["world_y"])
            except (TypeError, ValueError):
                return None
        else:
            try:
                pos_a = float(payload.get("posA"))
                pos_b = float(payload.get("posB"))
            except (TypeError, ValueError):
                return None
            bounds = dict(self.map_metadata.get("world_bounds") or {})
            x_min = float(bounds.get("x_min", 0.0))
            y_min = float(bounds.get("y_min", 0.0))
            height_m = float(self.map_metadata.get("height_m", 0.0) or 0.0)
            x = x_min + pos_a
            if bool(self.map_metadata.get("y_axis_inverted", False)):
                y = y_min + (height_m - pos_b)
            else:
                y = y_min + pos_b

        yaw = float(default_yaw if default_yaw is not None else 0.0)
        try:
            if payload.get("yaw_rad") is not None:
                yaw = float(payload.get("yaw_rad"))
            elif payload.get("yaw_deg") is not None:
                yaw = math.radians(float(payload.get("yaw_deg")))
            else:
                lanelet_id = self.resolve_node_id(payload.get("lanelet_id"))
                if lanelet_id is None:
                    lanelet_id = self.find_nearest_node(float(x), float(y))
                lanelet = self.lanelet_map.get_lanelet(str(lanelet_id)) if lanelet_id else None
                if lanelet is not None:
                    yaw = _heading_of_centerline(lanelet.centerline)
        except (TypeError, ValueError):
            pass
        return float(x), float(y), float(yaw)

    def find_nearest_node(
        self,
        x: float,
        y: float,
        yaw_rad: float | None = None,
        candidate_ids: Iterable[str] | None = None,
    ) -> str | None:
        if candidate_ids is None and yaw_rad is not None:
            lanelet_id = self.lanelet_map.at_pose(float(x), float(y), yaw_rad=float(yaw_rad))
            if lanelet_id is not None:
                return lanelet_id
        lanelet_ids = [str(item) for item in candidate_ids] if candidate_ids is not None else list(self.lanelet_map.lanelet_ids)
        best_id = None
        best_score = None
        fallback_best_id = None
        fallback_best_score = None
        for lanelet_id in lanelet_ids:
            lanelet = self.lanelet_map.get_lanelet(lanelet_id)
            if lanelet is None:
                continue
            arc_m, lat_m = lanelet.project_arclength(float(x), float(y))
            cl = lanelet.centerline
            if cl.shape[0] == 0:
                continue
            point = _point_at_arclength(cl, arc_m)
            euclid_m = math.hypot(float(x) - float(point[0]), float(y) - float(point[1]))
            score = euclid_m + 0.10 * abs(float(lat_m))
            if fallback_best_score is None or score < fallback_best_score:
                fallback_best_score = score
                fallback_best_id = lanelet_id

            if yaw_rad is not None:
                lanelet_heading = _heading_of_centerline(cl)
                yaw_diff = _wrap_angle(lanelet_heading - float(yaw_rad))
                if abs(yaw_diff) > (math.pi / 2.0):
                    continue
                score += 0.25 * abs(yaw_diff)

            if best_score is None or score < best_score:
                best_score = score
                best_id = lanelet_id
        return best_id if best_id is not None else fallback_best_id

    def resolve_node_id(self, spec) -> str | None:
        if spec is None:
            return None
        if isinstance(spec, dict):
            lanelet_id = spec.get("lanelet_id") or spec.get("id")
            if lanelet_id is not None and str(lanelet_id) in self.lanelet_map.lanelet_ids:
                return str(lanelet_id)
            if "x" in spec and "y" in spec:
                try:
                    yaw_rad = None
                    if spec.get("yaw_rad") is not None:
                        yaw_rad = float(spec["yaw_rad"])
                    elif spec.get("yaw_deg") is not None:
                        yaw_rad = math.radians(float(spec["yaw_deg"]))
                    elif spec.get("yaw") is not None:
                        yaw_rad = float(spec["yaw"])
                    return self.find_nearest_node(float(spec["x"]), float(spec["y"]), yaw_rad=yaw_rad)
                except (TypeError, ValueError):
                    return None
            return None
        candidate = str(spec).strip()
        if candidate in self.lanelet_map.lanelet_ids:
            return candidate
        return None

    def shortest_path(self, start_id: str, dest_id: str) -> list[str]:
        start_id = str(start_id)
        dest_id = str(dest_id)
        if start_id not in self.lanelet_map.lanelet_ids or dest_id not in self.lanelet_map.lanelet_ids:
            return []
        if start_id == dest_id:
            return [start_id]

        dist = {start_id: 0.0}
        prev: dict[str, str] = {}
        heap: list[tuple[float, str]] = [(0.0, start_id)]
        visited: set[str] = set()
        while heap:
            cost, lanelet_id = heapq.heappop(heap)
            if lanelet_id in visited:
                continue
            visited.add(lanelet_id)
            if lanelet_id == dest_id:
                break
            lanelet = self.lanelet_map.get_lanelet(lanelet_id)
            if lanelet is None:
                continue
            for succ in lanelet.successor_ids:
                succ_ll = self.lanelet_map.get_lanelet(str(succ))
                if succ_ll is None:
                    continue
                edge_cost = cost + float(succ_ll.length_m) * float(
                    ATTR_ROUTE_WEIGHTS.get(int(succ_ll.attribute), 1.0)
                )
                if edge_cost < dist.get(str(succ), float("inf")):
                    dist[str(succ)] = edge_cost
                    prev[str(succ)] = lanelet_id
                    heapq.heappush(heap, (edge_cost, str(succ)))
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
        ref_ids = list(self.reference_node_ids)
        if not ref_ids:
            return []
        if start_id not in ref_ids or dest_id not in ref_ids:
            return []
        start_idx = ref_ids.index(start_id)
        dest_idx = ref_ids.index(dest_id)
        if start_idx <= dest_idx:
            return ref_ids[start_idx:dest_idx + 1]
        return ref_ids[start_idx:] + ref_ids[:dest_idx + 1]

    def go_to(self, start_spec, dest_spec) -> RoutePath:
        start_id = self.resolve_node_id(start_spec)
        dest_id = self.resolve_node_id(dest_spec)
        if start_id is None or dest_id is None:
            return self._empty_route(source="go_to")
        lanelet_ids = self.shortest_path(start_id, dest_id)
        if not lanelet_ids:
            lanelet_ids = self.reference_path_between(start_id, dest_id)
        if not lanelet_ids:
            return self._empty_route(source="go_to")
        return self.build_dense_path(
            lanelet_ids,
            closed_loop=False,
            source="go_to",
            start_pose_xy=self._extract_pose_xy(start_spec),
        )

    def go_to_multiple(self, start_spec, destinations) -> RoutePath:
        start_id = self.resolve_node_id(start_spec)
        if start_id is None:
            return self._empty_route(source="go_to_multiple")
        dest_ids = [
            lanelet_id
            for lanelet_id in (self.resolve_node_id(spec) for spec in (destinations or []))
            if lanelet_id is not None
        ]
        if not dest_ids:
            return self._empty_route(source="go_to_multiple")

        full_sequence: list[str] = []
        current = start_id
        for dest_id in dest_ids:
            segment = self.shortest_path(current, dest_id)
            if not segment:
                segment = self.reference_path_between(current, dest_id)
            if not segment:
                continue
            if full_sequence:
                segment = segment[1:]
            full_sequence.extend(segment if full_sequence else segment)
            current = dest_id
        if not full_sequence:
            return self._empty_route(source="go_to_multiple")
        return self.build_dense_path(
            full_sequence,
            closed_loop=False,
            source="go_to_multiple",
            start_pose_xy=self._extract_pose_xy(start_spec),
        )

    def build_dense_path(
        self,
        lanelet_ids: Iterable[str],
        *,
        closed_loop: bool | None = None,
        route_id: str | None = None,
        source: str = "lanelet",
        start_pose_xy: tuple[float, float] | None = None,
    ) -> RoutePath:
        clean_ids = [str(lanelet_id) for lanelet_id in lanelet_ids if str(lanelet_id) in self.lanelet_map.lanelet_ids]
        if not clean_ids:
            return self._empty_route(route_id=route_id, source=source)

        deduped = [clean_ids[0]]
        for lanelet_id in clean_ids[1:]:
            if lanelet_id != deduped[-1]:
                deduped.append(lanelet_id)
        if closed_loop is None:
            closed_loop = len(deduped) > 1 and deduped[0] == deduped[-1]
        route_ids = list(deduped)
        if closed_loop and len(route_ids) > 1 and route_ids[0] == route_ids[-1]:
            route_ids = route_ids[:-1]

        path_points: list[np.ndarray] = []
        wp_lanelet_ids: list[str] = []
        wp_attrs: list[int] = []
        wp_edge_ids: list[str] = []
        for idx, lanelet_id in enumerate(route_ids):
            lanelet = self.lanelet_map.get_lanelet(str(lanelet_id))
            if lanelet is None or lanelet.centerline.shape[0] == 0:
                continue
            cl = np.asarray(lanelet.centerline, dtype=float)
            if idx == 0 and start_pose_xy is not None and not closed_loop:
                cl = _trim_centerline_from_pose(cl, float(start_pose_xy[0]), float(start_pose_xy[1]))
            if path_points and np.linalg.norm(path_points[-1] - cl[0]) <= 1e-6:
                cl = cl[1:]
            if cl.shape[0] == 0:
                continue
            next_id = route_ids[(idx + 1) % len(route_ids)] if (closed_loop and route_ids) else (
                route_ids[idx + 1] if idx + 1 < len(route_ids) else lanelet_id
            )
            for point in cl:
                path_points.append(np.asarray(point, dtype=float))
                wp_lanelet_ids.append(str(lanelet_id))
                wp_attrs.append(int(lanelet.attribute))
                wp_edge_ids.append(f"{lanelet_id}->{next_id}" if next_id != lanelet_id else str(lanelet_id))
        if not path_points:
            return self._empty_route(route_id=route_id, source=source)

        xy = np.vstack(path_points)
        psi = _compute_heading_series(xy, closed_loop=bool(closed_loop))
        waypoints = np.column_stack([xy[:, 0], xy[:, 1], psi])
        route = RoutePath(
            node_ids=list(route_ids),
            waypoints=waypoints,
            wp_node_attrs=np.asarray(wp_attrs, dtype=int),
            wp_node_ids=list(wp_lanelet_ids),
            closed_loop=bool(closed_loop),
            route_id=route_id,
            source=source,
            wp_edge_ids=list(wp_edge_ids),
        )
        return self._finalize_route(route)

    def _empty_route(self, *, route_id: str | None = None, source: str = "lanelet") -> RoutePath:
        route = RoutePath([], np.empty((0, 3)), np.empty(0, dtype=int), [], False, route_id, source)
        return self._finalize_route(route)

    @staticmethod
    def _extract_pose_xy(spec) -> tuple[float, float] | None:
        if not isinstance(spec, dict):
            return None
        if "x" not in spec or "y" not in spec:
            return None
        try:
            return float(spec["x"]), float(spec["y"])
        except (TypeError, ValueError):
            return None

    def _resolve_start_lanelet_id(self, configured_id: str | None) -> str:
        if configured_id is not None:
            candidate = str(configured_id).strip()
            if candidate in self.lanelet_map.lanelet_ids:
                return candidate
        lanelet_ids = list(self.lanelet_map.lanelet_ids)
        if not lanelet_ids:
            return ""
        return min(lanelet_ids, key=_lanelet_sort_key)

    def _build_reference_loop(self, start_lanelet_id: str) -> tuple[list[str], bool]:
        if not start_lanelet_id:
            return [], False
        visited: list[str] = []
        seen: set[str] = set()
        current = str(start_lanelet_id)
        closed_loop = False
        while current and current not in seen:
            visited.append(current)
            seen.add(current)
            next_lanelet = self._choose_reference_successor(current, seen, start_lanelet_id)
            if next_lanelet is None:
                break
            if next_lanelet == start_lanelet_id:
                closed_loop = len(visited) > 1
                break
            current = next_lanelet
        return visited, closed_loop

    def _choose_reference_successor(
        self,
        current_lanelet_id: str,
        seen: set[str],
        start_lanelet_id: str,
    ) -> str | None:
        current = self.lanelet_map.get_lanelet(str(current_lanelet_id))
        if current is None:
            return None
        successors = list(current.successor_ids or ())
        if not successors:
            return None
        if start_lanelet_id in successors and len(seen) > 1:
            return str(start_lanelet_id)
        current_heading = _heading_of_centerline(current.centerline)
        best_id = None
        best_score = None
        for succ_id in successors:
            succ = self.lanelet_map.get_lanelet(str(succ_id))
            if succ is None:
                continue
            if str(succ_id) in seen:
                continue
            succ_heading = _heading_of_centerline(succ.centerline)
            delta = abs(_wrap_angle(succ_heading - current_heading))
            score = (delta, _lanelet_sort_key(str(succ_id)))
            if best_score is None or score < best_score:
                best_score = score
                best_id = str(succ_id)
        return best_id

    def _build_ordered_nodes(self) -> list[RouteAnchor]:
        nodes: list[RouteAnchor] = []
        for idx, lanelet_id in enumerate(self.reference_node_ids):
            lanelet = self.lanelet_map.get_lanelet(str(lanelet_id))
            if lanelet is None or lanelet.centerline.shape[0] == 0:
                continue
            pt = lanelet.centerline[0]
            nodes.append(
                RouteAnchor(
                    node_id=str(lanelet_id),
                    x=float(pt[0]),
                    y=float(pt[1]),
                    attribute=int(lanelet.attribute),
                    is_start=(idx == 0),
                )
            )
        return nodes

    @staticmethod
    def _semantic_priority(event: dict | None) -> int:
        semantic_type = str((event or {}).get("type") or "")
        order = {
            "stopline": 0,
            "yield": 0,
            "traffic_light": 0,
            "intersection": 1,
            "crosswalk": 2,
            "roundabout": 3,
            "zone": 4,
        }
        return order.get(semantic_type, 99)

    def _default_attr_semantic(self, lanelet_id: str, attr: int) -> dict | None:
        attr = int(attr)
        if attr == ATTR_STOPLINE:
            return {
                "id": f"attr-stopline-{lanelet_id}",
                "type": "stopline",
                "label": "Stopline",
                "lanelet_id": lanelet_id,
                "control_type": "stop",
            }
        if attr == ATTR_INTERSECTION:
            return {
                "id": f"attr-intersection-{lanelet_id}",
                "type": "intersection",
                "label": "Intersection",
                "lanelet_id": lanelet_id,
            }
        if attr == ATTR_CROSSWALK:
            return {
                "id": f"attr-crosswalk-{lanelet_id}",
                "type": "crosswalk",
                "label": "Crosswalk",
                "lanelet_id": lanelet_id,
            }
        if attr == ATTR_ROUNDABOUT:
            return {
                "id": f"attr-roundabout-{lanelet_id}",
                "type": "roundabout",
                "label": "Roundabout",
                "lanelet_id": lanelet_id,
            }
        if attr in (ATTR_HIGHWAY_LEFT, ATTR_HIGHWAY_RIGHT):
            return {
                "id": f"attr-highway-{lanelet_id}",
                "type": "zone",
                "label": "Highway",
                "lanelet_id": lanelet_id,
                "zone_type": "highway",
            }
        if attr == ATTR_ONEWAY:
            return {
                "id": f"attr-oneway-{lanelet_id}",
                "type": "zone",
                "label": "One Way",
                "lanelet_id": lanelet_id,
                "zone_type": "oneway",
            }
        return None

    def _lanelet_events(self, lanelet_id: str) -> list[dict]:
        lanelet = self.lanelet_map.get_lanelet(str(lanelet_id))
        if lanelet is None:
            return []
        events: list[dict] = []
        for reg_id in lanelet.regulatory_element_ids:
            reg = self.lanelet_map.get_regulator(str(reg_id))
            if reg is None:
                continue
            events.append(
                {
                    "id": str(reg.element_id),
                    "type": str(reg.kind),
                    "label": str(reg.label or reg.kind),
                    "lanelet_id": str(lanelet_id),
                    **dict(reg.data or {}),
                }
            )
        default_event = self._default_attr_semantic(str(lanelet_id), int(lanelet.attribute))
        if default_event is not None:
            events.append(default_event)
        deduped: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for event in events:
            key = (str(event.get("type") or ""), str(event.get("id") or ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(event)
        deduped.sort(key=self._semantic_priority)
        return deduped

    def _annotate_route(self, route: RoutePath) -> RoutePath:
        route.route_events = []
        route.wp_semantic_ids = []
        route.wp_semantic_types = []
        route.wp_zone_ids = []
        route.wp_zone_types = []
        route.map_metadata = self.get_map_metadata()
        route.available_destinations = self.get_available_destinations()
        if not route.node_ids or not route.wp_node_ids:
            return route

        first_wp_idx_by_lanelet: dict[str, int] = {}
        lanelet_events: dict[str, list[dict]] = {}
        for wp_idx, lanelet_id in enumerate(route.wp_node_ids):
            first_wp_idx_by_lanelet.setdefault(str(lanelet_id), int(wp_idx))
            lanelet_events.setdefault(str(lanelet_id), self._lanelet_events(str(lanelet_id)))

        seen_route_events: set[tuple[str, str]] = set()
        for route_idx, lanelet_id in enumerate(route.node_ids):
            events = lanelet_events.get(str(lanelet_id), [])
            for event in events:
                key = (str(event.get("type") or ""), str(event.get("id") or ""))
                if key in seen_route_events:
                    continue
                seen_route_events.add(key)
                route.route_events.append(
                    {
                        **event,
                        "route_node_index": int(route_idx),
                        "node_id": str(lanelet_id),
                        "waypoint_idx": int(first_wp_idx_by_lanelet.get(str(lanelet_id), 0)),
                    }
                )

        for lanelet_id, attr in zip(route.wp_node_ids, route.wp_node_attrs.tolist()):
            events = lanelet_events.get(str(lanelet_id), [])
            primary = events[0] if events else None
            route.wp_semantic_ids.append(primary.get("id") if primary else None)
            route.wp_semantic_types.append(str(primary.get("type") or "lane_follow") if primary else "lane_follow")

            zone_ids: list[str] = []
            zone_types: list[str] = []
            if int(attr) in (ATTR_HIGHWAY_LEFT, ATTR_HIGHWAY_RIGHT):
                zone_ids.append(f"highway:{lanelet_id}")
                zone_types.append("highway")
            if int(attr) == ATTR_ONEWAY:
                zone_ids.append(f"oneway:{lanelet_id}")
                zone_types.append("oneway")
            route.wp_zone_ids.append(zone_ids)
            route.wp_zone_types.append(zone_types)

        route.route_events.sort(key=lambda item: (int(item.get("waypoint_idx", 0)), str(item.get("id", ""))))
        return route

    def _finalize_route(self, route: RoutePath) -> RoutePath:
        route.map_metadata = self.get_map_metadata()
        route.available_destinations = self.get_available_destinations()
        if not route.wp_edge_ids:
            route.wp_edge_ids = ["" for _ in route.wp_node_ids]
        return self._annotate_route(route)


def _lanelet_sort_key(lanelet_id: str):
    text = str(lanelet_id)
    return (0, int(text)) if text.isdigit() else (1, text)


def _wrap_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return float(angle_rad)


def _heading_of_centerline(centerline: np.ndarray) -> float:
    pts = np.asarray(centerline, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    delta = pts[min(1, pts.shape[0] - 1)] - pts[0]
    return math.atan2(float(delta[1]), float(delta[0]))


def _compute_heading_series(xy: np.ndarray, *, closed_loop: bool) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)
    n = int(pts.shape[0])
    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)
    psi = np.zeros(n, dtype=float)
    for idx in range(n - 1):
        delta = pts[idx + 1] - pts[idx]
        psi[idx] = math.atan2(float(delta[1]), float(delta[0]))
    if closed_loop and n > 2:
        delta = pts[0] - pts[-1]
        psi[-1] = math.atan2(float(delta[1]), float(delta[0]))
    else:
        psi[-1] = psi[-2]
    return psi


def _trim_centerline_from_pose(centerline: np.ndarray, x: float, y: float) -> np.ndarray:
    pts = np.asarray(centerline, dtype=float)
    if pts.shape[0] < 2:
        return pts

    best_seg_idx = 0
    best_proj = pts[0]
    best_dist = math.inf
    for idx in range(pts.shape[0] - 1):
        p0 = pts[idx]
        p1 = pts[idx + 1]
        seg = p1 - p0
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 <= 1e-12:
            continue
        rel = np.asarray([float(x), float(y)], dtype=float) - p0
        t = max(0.0, min(1.0, float(np.dot(rel, seg) / seg_len2)))
        proj = p0 + t * seg
        dist = float(np.linalg.norm(np.asarray([float(x), float(y)], dtype=float) - proj))
        if dist < best_dist:
            best_dist = dist
            best_seg_idx = idx
            best_proj = proj

    tail = pts[best_seg_idx + 1:]
    pieces: list[np.ndarray] = [np.asarray(best_proj, dtype=float)]
    if tail.shape[0] > 0:
        if np.linalg.norm(tail[0] - best_proj) > 1e-6:
            pieces.extend(tail)
        else:
            pieces.extend(tail[1:])
    if len(pieces) < 2:
        if pts.shape[0] >= 2:
            pieces = [np.asarray(best_proj, dtype=float), pts[-1]]
        else:
            pieces = [np.asarray(best_proj, dtype=float), np.asarray(best_proj, dtype=float)]
    return np.vstack(pieces)


def _point_at_arclength(centerline: np.ndarray, arclength_m: float) -> np.ndarray:
    pts = np.asarray(centerline, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros(2, dtype=float)
    if pts.shape[0] == 1:
        return pts[0]
    remaining = float(max(0.0, arclength_m))
    for idx in range(pts.shape[0] - 1):
        p0 = pts[idx]
        p1 = pts[idx + 1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-9:
            continue
        if remaining <= seg_len:
            t = remaining / seg_len
            return p0 + t * seg
        remaining -= seg_len
    return pts[-1]
