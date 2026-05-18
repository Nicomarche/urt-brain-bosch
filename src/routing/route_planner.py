from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.routing.lanelet.attributes import (
    ATTR_CROSSWALK,
    ATTR_HIGHWAY_LEFT,
    ATTR_HIGHWAY_RIGHT,
    ATTR_INTERSECTION,
    ATTR_INTERSECTION_EXIT,
    ATTR_NORMAL,
    ATTR_ONEWAY,
    ATTR_ROUNDABOUT,
    ATTR_STOPLINE,
    WAYPOINT_MODE_ATTRS,
)
from src.routing.lanelet.route_path import RoutePath
from src.utils.live_log import live_log

try:
    from config import (
        TRACKING_ROUTE_GLOBAL_RECOVERY_ERROR_M as _MAP_MATCH_GLOBAL_RECOVERY_ERROR_M,
    )
except Exception:
    _MAP_MATCH_GLOBAL_RECOVERY_ERROR_M = 0.14

try:
    from config import TRACKING_ROUTE_RECOVERY_ERROR_M as _MAP_MATCH_RECOVERY_ERROR_M
except Exception:
    _MAP_MATCH_RECOVERY_ERROR_M = 0.10

_MAP_MATCH_RECOVERY_MIN_IMPROVEMENT_M = 0.02
_MAP_MATCH_RECOVERY_WINDOW_MULT = 6
_MAP_MATCH_GLOBAL_MIN_IMPROVEMENT_M = 0.03


def _preview_waypoints(waypoints, limit: int = 3) -> list[list[float]]:
    out: list[list[float]] = []
    if waypoints is None:
        return out
    try:
        for pt in waypoints[:limit]:
            out.append([
                round(float(pt[0]), 3),
                round(float(pt[1]), 3),
                round(float(pt[2]), 4),
            ])
    except Exception:
        return []
    return out


def _serialize_waypoints(waypoints) -> list[list[float]]:
    out: list[list[float]] = []
    if waypoints is None:
        return out
    try:
        arr = np.asarray(waypoints, dtype=float)
    except Exception:
        return out
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return out
    for pt in arr:
        yaw = float(pt[2]) if arr.shape[1] >= 3 else 0.0
        out.append([float(pt[0]), float(pt[1]), yaw])
    return out


def _preview_ids(values, limit: int = 6) -> list[str]:
    if not values:
        return []
    return [
        str(item)
        for item in list(values)[:limit]
        if str(item)
    ]


def _preview_destinations(specs, limit: int = 4) -> list[dict]:
    preview: list[dict] = []
    for spec in list(specs or [])[:limit]:
        if not isinstance(spec, dict):
            continue
        item: dict[str, object] = {}
        if spec.get("lanelet_id") is not None:
            item["lanelet_id"] = str(spec.get("lanelet_id"))
        if spec.get("id") is not None:
            item["id"] = str(spec.get("id"))
        if spec.get("x") is not None and spec.get("y") is not None:
            try:
                item["x"] = round(float(spec["x"]), 3)
                item["y"] = round(float(spec["y"]), 3)
            except (TypeError, ValueError):
                pass
        if item:
            preview.append(item)
    return preview


def _summarize_command(command: dict | None) -> dict:
    if not isinstance(command, dict):
        return {"type": "invalid"}
    summary: dict[str, object] = {
        "mode": str(command.get("mode", "") or "").lower() or "direct_xy",
    }
    if command.get("lanelet_id") is not None:
        summary["lanelet_id"] = str(command.get("lanelet_id"))
    if command.get("x") is not None and command.get("y") is not None:
        try:
            summary["x"] = round(float(command["x"]), 3)
            summary["y"] = round(float(command["y"]), 3)
        except (TypeError, ValueError):
            pass
    destinations = list(command.get("destinations", []) or [])
    if destinations:
        summary["destination_count"] = len(destinations)
        summary["destinations_preview"] = _preview_destinations(destinations)
    return summary


@dataclass
class PathUpdate:
    route_active: bool
    route_id: str | None
    route_points: list[dict[str, float]]
    route_waypoints: list[list[float]]
    destination_node_id: str | None
    destination_label: str | None
    route_queue: list[dict]
    current_node_id: str | None
    current_node_attr: int
    upcoming_node_id: str | None
    upcoming_node_attr: int
    maneuver_type: str
    route_progress: float
    route_completed: bool
    waypoint_mode_active: bool
    matched_idx: int
    target_idx: int
    matched_x: float
    matched_y: float
    matched_yaw: float
    path_psi: float
    path_kappa: float
    path_heading_change_rad: float
    error_m: float
    heading_rad: float
    map_match_error_m: float
    remaining_distance_m: float
    replans: int
    route_source: str
    destination_point: dict[str, float] | None
    next_semantic_id: str | None
    next_semantic_type: str | None
    next_semantic_label: str | None
    next_semantic_distance_m: float | None
    expected_control_type: str | None
    current_zone_ids: list[str]
    current_zone_types: list[str]
    map_metadata: dict
    available_destinations: list[dict]


class PathManager:
    """Maintains the active navigation route and control targets."""

    def __init__(self, graph):
        self.graph = graph
        self.active_route: RoutePath | None = None
        self.route_active = False
        self.route_completed = False
        self.destination_node_id: str | None = None
        self.destination_specs: list[dict[str, float] | dict] = []
        self.route_id: str | None = None
        self.replans = 0
        self._route_counter = 0
        self.matched_idx = 0
        self.target_idx = 0

        self._clear_active_route()

    # ------------------------------------------------------------------
    # Route selection
    # ------------------------------------------------------------------
    def _next_route_id(self, prefix: str = "route") -> str:
        self._route_counter += 1
        return f"{prefix}-{self._route_counter}"

    def _pose_to_start_spec(self, current_pose: dict | None):
        if not isinstance(current_pose, dict):
            return self.graph.get_start_node_id()
        try:
            spec = {"x": float(current_pose["x"]), "y": float(current_pose["y"])}
            yaw_rad = current_pose.get("yaw_rad", current_pose.get("yaw"))
            if yaw_rad is not None:
                spec["yaw_rad"] = float(yaw_rad)
            return spec
        except (KeyError, TypeError, ValueError):
            return self.graph.get_start_node_id()

    def _activate_route(
        self,
        route: RoutePath,
        destination_specs: list[dict] | None,
        route_id: str,
    ) -> bool:
        if route is None or route.waypoints.size == 0:
            live_log(
                "route_planner", event="route_activation_failed",
                route_id=route_id,
                reason="empty_route",
                destination_specs=_preview_destinations(destination_specs),
            )
            return False
        route.route_id = route_id
        self.active_route = route
        self.route_id = route_id
        self.route_active = True
        self.route_completed = False
        self.destination_specs = list(destination_specs or [])
        self.destination_node_id = route.node_ids[-1] if route.node_ids else None
        self.matched_idx = 0
        self.target_idx = 0
        live_log(
            "route_planner", event="route_activated",
            route_id=route_id,
            route_source=str(getattr(route, "source", "unknown") or "unknown"),
            closed_loop=bool(getattr(route, "closed_loop", False)),
            destination_node_id=self.destination_node_id,
            destination_specs=_preview_destinations(self.destination_specs),
            waypoint_count=int(len(route.waypoints)),
            node_count=int(len(getattr(route, "node_ids", []) or [])),
            node_ids_preview=_preview_ids(getattr(route, "node_ids", []) or []),
            waypoint_preview=_preview_waypoints(route.waypoints),
        )
        return True

    def _clear_active_route(self) -> None:
        self.active_route = None
        self.route_active = False
        self.route_completed = False
        self.destination_specs = []
        self.destination_node_id = None
        self.route_id = None
        self.matched_idx = 0
        self.target_idx = 0

    def reset_route(self, current_pose: dict | None = None) -> bool:
        ref_ids = list(self.graph.reference_node_ids)
        closed_loop = bool(getattr(self.graph.reference_path, "closed_loop", False))
        if len(ref_ids) > 1 and ref_ids[0] == ref_ids[-1]:
            ref_ids = ref_ids[:-1]
            closed_loop = True
        if not ref_ids:
            self._clear_active_route()
            return False

        if current_pose:
            try:
                yaw_rad = current_pose.get("yaw_rad", current_pose.get("yaw"))
                start_node_id = self.graph.find_nearest_node(
                    float(current_pose["x"]),
                    float(current_pose["y"]),
                    yaw_rad=float(yaw_rad) if yaw_rad is not None else None,
                    candidate_ids=ref_ids,
                )
            except (KeyError, TypeError, ValueError):
                start_node_id = ref_ids[0]
            if start_node_id in ref_ids:
                idx = ref_ids.index(start_node_id)
                ref_ids = ref_ids[idx:] + ref_ids[:idx]

        if (
            not closed_loop
            and len(ref_ids) > 1
            and ref_ids[0] in self.graph.adj.get(ref_ids[-1], [])
        ):
            closed_loop = True

        if closed_loop and ref_ids[0] != ref_ids[-1]:
            ref_ids = ref_ids + [ref_ids[0]]
        start_pose_xy = None
        if isinstance(current_pose, dict):
            try:
                start_pose_xy = (float(current_pose["x"]), float(current_pose["y"]))
            except (KeyError, TypeError, ValueError):
                start_pose_xy = None
        route = self.graph.build_dense_path(
            ref_ids,
            closed_loop=closed_loop,
            route_id=self._next_route_id("reference"),
            source="reference",
            start_pose_xy=start_pose_xy,
        )
        return self._activate_route(
            route,
            [route.destination_point() or {"x": float(route.waypoints[-1][0]), "y": float(route.waypoints[-1][1])}],
            route.route_id or self._next_route_id("reference"),
        )

    def set_route(self, destination, current_pose: dict | None = None) -> bool:
        start_spec = self._pose_to_start_spec(current_pose)
        route = self.graph.go_to(start_spec, destination)
        route_id = self._next_route_id("route")
        return self._activate_route(route, [self._normalize_destination_spec(destination)], route_id)

    def set_route_from_lanelet_sequence(
        self,
        lanelet_ids,
        current_pose: dict | None = None,
        *,
        route_id_hint: str | None = None,
    ) -> bool:
        """Activa una ruta siguiendo una secuencia ordenada de lanelets.

        - Valida que TODOS los lanelets existan en el mapa.
        - Valida conectividad par-a-par usando `successor_ids`.
        - No re-resuelve Dijkstra: densifica directamente con `build_dense_path`.
        Devuelve False si algo no cierra (sin intentar reparar).
        """
        clean_ids = [str(item) for item in (lanelet_ids or []) if str(item)]
        if not clean_ids:
            live_log(
                "route_planner", event="lanelet_sequence_empty",
                route_id_hint=route_id_hint,
            )
            return False
        lmap = self.graph.lanelet_map
        missing = [lid for lid in clean_ids if lmap.get_lanelet(lid) is None]
        if missing:
            live_log(
                "route_planner", event="lanelet_sequence_invalid",
                reason="unknown_lanelets",
                missing=missing[:10],
                missing_count=len(missing),
            )
            return False
        broken: list[tuple[str, str]] = []
        for i in range(len(clean_ids) - 1):
            a, b = clean_ids[i], clean_ids[i + 1]
            if a == b:
                continue
            succs = lmap.get_lanelet(a).successor_ids
            if b not in succs:
                broken.append((a, b))
        if broken:
            live_log(
                "route_planner", event="lanelet_sequence_invalid",
                reason="broken_successor_links",
                broken_count=len(broken),
                broken_preview=[f"{a}->{b}" for a, b in broken[:10]],
            )
            return False
        start_pose_xy = None
        if isinstance(current_pose, dict):
            try:
                start_pose_xy = (
                    float(current_pose["x"]),
                    float(current_pose["y"]),
                )
            except (KeyError, TypeError, ValueError):
                start_pose_xy = None
        route_id = route_id_hint or self._next_route_id("seq")
        route = self.graph.build_dense_path(
            clean_ids,
            closed_loop=False,
            route_id=route_id,
            source="lanelet_sequence",
            start_pose_xy=start_pose_xy,
        )
        return self._activate_route(route, [], route_id)

    def set_route_queue(self, destinations, current_pose: dict | None = None) -> bool:
        start_spec = self._pose_to_start_spec(current_pose)
        dest_specs = []
        for spec in destinations or []:
            normalized = self._normalize_destination_spec(spec)
            if normalized is not None:
                dest_specs.append(normalized)
        if not dest_specs:
            return False
        route = self.graph.go_to_multiple(start_spec, dest_specs)
        route_id = self._next_route_id("route")
        return self._activate_route(route, dest_specs, route_id)

    def clear_route(self) -> bool:
        had_route = self.active_route is not None or self.route_active or bool(self.destination_specs)
        prev_route_id = self.route_id
        prev_destination_node_id = self.destination_node_id
        prev_destination_specs = list(self.destination_specs or [])
        self._clear_active_route()
        if had_route:
            live_log(
                "route_planner", event="route_cleared",
                route_id=prev_route_id,
                destination_node_id=prev_destination_node_id,
                destination_specs=_preview_destinations(prev_destination_specs),
            )
        return had_route

    def handle_command(self, command: dict | None, current_pose: dict | None = None) -> bool:
        if not isinstance(command, dict):
            return False
        live_log(
            "route_planner", event="command_received",
            command=_summarize_command(command),
            current_pose=_preview_destinations([current_pose] if isinstance(current_pose, dict) else []),
            had_active_route=bool(self.route_active),
            route_id=self.route_id,
        )
        if "x" in command and "y" in command and "mode" not in command:
            self.replans += 1
            handled = self.set_route(command, current_pose=current_pose)
        else:
            mode = str(command.get("mode", "") or "").lower()
            destinations = list(command.get("destinations", []) or [])
            if mode in {"clear", "cancel"}:
                handled = self.clear_route()
            elif mode == "reset":
                self.replans += 1
                handled = self.reset_route(current_pose=current_pose)
            elif mode == "go_to":
                if not destinations and "x" in command and "y" in command:
                    destinations = [{"x": command.get("x"), "y": command.get("y")}]
                if not destinations:
                    handled = False
                else:
                    self.replans += 1
                    handled = self.set_route(destinations[0], current_pose=current_pose)
            elif mode == "go_to_multiple":
                self.replans += 1
                handled = self.set_route_queue(destinations, current_pose=current_pose)
            elif mode == "lanelet_sequence":
                lanelet_ids = [str(x) for x in (command.get("lanelet_ids") or [])]
                if not lanelet_ids:
                    handled = False
                else:
                    self.replans += 1
                    handled = self.set_route_from_lanelet_sequence(
                        lanelet_ids,
                        current_pose=current_pose,
                        route_id_hint=command.get("plan_name") or command.get("route_id"),
                    )
            else:
                handled = False
        live_log(
            "route_planner", event="command_result",
            command=_summarize_command(command),
            handled=bool(handled),
            route_active=bool(self.route_active),
            route_id=self.route_id,
            route_completed=bool(self.route_completed),
            destination_node_id=self.destination_node_id,
            replans=int(self.replans),
        )
        return handled

    @staticmethod
    def _normalize_destination_spec(spec) -> dict | None:
        if spec is None:
            return None
        if isinstance(spec, dict):
            if "x" in spec and "y" in spec:
                try:
                    normalized = {"x": float(spec["x"]), "y": float(spec["y"])}
                except (TypeError, ValueError):
                    return None
                lanelet_id = spec.get("lanelet_id") or spec.get("id")
                if lanelet_id is not None:
                    normalized["lanelet_id"] = str(lanelet_id)
                return normalized
            lanelet_id = spec.get("lanelet_id") or spec.get("id")
            if lanelet_id is not None:
                return {"lanelet_id": str(lanelet_id)}
            return None
        return None

    def _describe_destination(self, spec) -> dict | None:
        return self.graph.describe_destination(spec)

    def _route_queue_payload(self) -> list[dict]:
        payload = []
        for spec in self.destination_specs:
            info = self._describe_destination(spec)
            if info is None and isinstance(spec, dict) and "x" in spec and "y" in spec:
                info = {
                    "id": "point",
                    "label": "Point goal",
                    "x": round(float(spec["x"]), 4),
                    "y": round(float(spec["y"]), 4),
                }
            if info is None:
                continue
            payload.append(info)
        return payload

    def _current_destination_spec(self):
        return self.destination_specs[-1] if self.destination_specs else None

    @property
    def destination_point(self) -> dict[str, float] | None:
        current = self._current_destination_spec()
        if isinstance(current, dict) and "x" in current and "y" in current:
            return {"x": float(current["x"]), "y": float(current["y"])}
        if self.active_route is not None:
            return self.active_route.destination_point()
        return None

    # ------------------------------------------------------------------
    # Route geometry utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        while angle_rad > math.pi:
            angle_rad -= 2.0 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2.0 * math.pi
        return float(angle_rad)

    def _clamp_idx(self, route: RoutePath, idx: int) -> int:
        n = len(route.waypoints)
        if n == 0:
            return 0
        if route.closed_loop:
            return idx % n
        return max(0, min(n - 1, idx))

    def _find_waypoint_ahead(self, route: RoutePath, current_idx: int, lookahead_m: float) -> int:
        if len(route.waypoints) == 0:
            return 0
        steps = max(1, int(round(float(lookahead_m) / max(self.graph.step_m, 1e-6))))
        if route.closed_loop:
            return (current_idx + steps) % len(route.waypoints)
        return min(len(route.waypoints) - 1, current_idx + steps)

    def _project_pose_to_route(
        self,
        route: RoutePath,
        x: float,
        y: float,
        yaw: float,
        search_center: int,
        search_window: int,
        distance_weight: float,
        heading_weight: float,
        continuity_weight_scale: float = 1.0,
    ) -> dict:
        n = len(route.waypoints)
        if n == 0:
            return {
                "matched_idx": 0,
                "matched_x": float(x),
                "matched_y": float(y),
                "path_psi": float(yaw),
                "map_match_error_m": 0.0,
            }
        if n == 1:
            wx, wy, psi = route.waypoints[0]
            return {
                "matched_idx": 0,
                "matched_x": float(wx),
                "matched_y": float(wy),
                "path_psi": float(psi),
                "map_match_error_m": math.hypot(float(x) - float(wx), float(y) - float(wy)),
            }

        center = self._clamp_idx(route, int(search_center))
        window = max(1, int(search_window))
        continuity_weight = (
            max(0.05, 0.25 * float(distance_weight))
            * max(0.0, float(continuity_weight_scale))
        )
        if route.closed_loop:
            seg_idxs = [(center + off) % n for off in range(-window, window + 1)]
        else:
            lo = max(0, center - window)
            hi = min(n - 1, center + window)
            seg_idxs = list(range(lo, hi))
            if not seg_idxs:
                seg_idxs = [max(0, min(n - 2, center))]

        best = None
        for seg_idx in seg_idxs:
            next_idx = (seg_idx + 1) % n if route.closed_loop else min(seg_idx + 1, n - 1)
            if next_idx == seg_idx:
                continue
            x0, y0, _ = route.waypoints[seg_idx]
            x1, y1, _ = route.waypoints[next_idx]
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
            continuity_m = abs(seg_idx - center) * float(self.graph.step_m)
            score = (
                # Use the real distance to the projected point on the segment, not only the
                # signed lateral component. In tight curves / stopline turns the raw pose can
                # be well ahead along the route while keeping a small lateral error, which
                # makes many old segments look equally good and the continuity term freezes the
                # match on a stale waypoint.
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
                "map_match_error_m": float(projection_error_m),
                "score": float(score),
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

        if best is not None:
            return best

        idx = center
        wx, wy, psi = route.waypoints[idx]
        return {
            "matched_idx": int(idx),
            "matched_x": float(wx),
            "matched_y": float(wy),
            "path_psi": float(psi),
            "map_match_error_m": math.hypot(float(x) - float(wx), float(y) - float(wy)),
        }

    def _compute_tracking_error(self, route: RoutePath, x: float, y: float, yaw: float, wp_idx: int) -> tuple[float, float]:
        if len(route.waypoints) == 0:
            return 0.0, 0.0
        idx = self._clamp_idx(route, wp_idx)
        xr, yr, psi_r = route.waypoints[idx]
        dx = x - xr
        dy = y - yr
        error_m = -dx * math.sin(psi_r) + dy * math.cos(psi_r)
        heading_rad = self._wrap_angle(yaw - psi_r)
        return float(error_m), float(heading_rad)

    def _is_precision_zone(self, route: RoutePath, wp_idx: int, lookahead_pts: int) -> bool:
        n = len(route.waypoints)
        if n == 0:
            return False
        for offset in range(max(1, int(lookahead_pts))):
            idx = (wp_idx + offset) % n if route.closed_loop else min(n - 1, wp_idx + offset)
            if int(route.wp_node_attrs[idx]) in WAYPOINT_MODE_ATTRS:
                return True
        return False

    def _get_curvature(self, route: RoutePath, wp_idx: int) -> float:
        n = len(route.waypoints)
        if n < 3:
            return 0.0
        idx = self._clamp_idx(route, wp_idx)
        if route.closed_loop:
            i_prev = (idx - 1) % n
            i_next = (idx + 1) % n
        else:
            i_prev = max(0, idx - 1)
            i_next = min(n - 1, idx + 1)
            if i_prev == idx or i_next == idx:
                return 0.0
        dpsi = float(route.waypoints[i_next][2]) - float(route.waypoints[i_prev][2])
        while dpsi > math.pi:
            dpsi -= 2.0 * math.pi
        while dpsi < -math.pi:
            dpsi += 2.0 * math.pi
        return dpsi / max(2.0 * self.graph.step_m, 1e-6)

    def _get_heading_change_ahead(self, route: RoutePath, wp_idx: int, lookahead_m: float) -> float:
        """Total heading change (radians) from wp_idx over the next lookahead_m of path.

        Sign follows the graph yaw convention:
        positive = counter-clockwise / left curve, negative = clockwise / right curve.
        Derived directly from the dense path built from graph nodes and edges, so
        this reflects true track geometry rather than camera observations.
        """
        n = len(route.waypoints)
        if n < 2:
            return 0.0
        steps = max(1, int(round(float(lookahead_m) / max(float(self.graph.step_m), 1e-6))))
        idx_start = self._clamp_idx(route, wp_idx)
        idx_end = self._clamp_idx(route, wp_idx + steps)
        if idx_start == idx_end:
            return 0.0
        dpsi = float(route.waypoints[idx_end][2]) - float(route.waypoints[idx_start][2])
        while dpsi > math.pi:
            dpsi -= 2.0 * math.pi
        while dpsi < -math.pi:
            dpsi += 2.0 * math.pi
        return dpsi

    def _remaining_distance(self, route: RoutePath, matched_idx: int) -> float:
        n = len(route.waypoints)
        if n == 0 or route.closed_loop:
            return float("inf") if route.closed_loop else 0.0
        remaining_pts = max(0, (n - 1) - self._clamp_idx(route, matched_idx))
        return float(remaining_pts) * float(self.graph.step_m)

    def _infer_maneuver_type(
        self,
        route: RoutePath,
        target_idx: int,
        lookahead_pts: int,
        next_semantic_type: str | None = None,
        expected_control_type: str | None = None,
    ) -> str:
        if len(route.waypoints) == 0:
            return "none"
        if expected_control_type == "traffic_light":
            return "traffic_light"
        if expected_control_type == "stop":
            return "stopline"
        if next_semantic_type == "parking_spot":
            return "parking_search"
        if next_semantic_type == "crosswalk":
            return "crosswalk"
        if next_semantic_type == "roundabout":
            return "roundabout"
        attr = int(route.wp_node_attrs[self._clamp_idx(route, target_idx)])
        if attr == ATTR_STOPLINE:
            return "stopline"
        if attr == ATTR_CROSSWALK:
            return "crosswalk"
        if attr == ATTR_ROUNDABOUT:
            return "roundabout"
        if attr in (ATTR_HIGHWAY_LEFT, ATTR_HIGHWAY_RIGHT):
            return "highway"
        if attr == ATTR_ONEWAY:
            return "oneway"
        if attr != ATTR_INTERSECTION:
            return "lane_follow"

        n = len(route.waypoints)
        ahead_idx = self._clamp_idx(route, target_idx + max(4, int(lookahead_pts)))
        psi_now = float(route.waypoints[self._clamp_idx(route, target_idx)][2])
        psi_next = float(route.waypoints[ahead_idx][2])
        yaw_delta = self._wrap_angle(psi_next - psi_now)
        if yaw_delta > math.radians(12.0):
            return "turn_left"
        if yaw_delta < -math.radians(12.0):
            return "turn_right"
        return "intersection_straight"

    def _current_route_preview(self) -> list[dict[str, float]]:
        if self.active_route is None:
            return []
        return self.active_route.preview_points()

    def _current_zone_context(self, route: RoutePath, idx: int) -> tuple[list[str], list[str]]:
        zone_ids = []
        zone_types = []
        if getattr(route, "wp_zone_ids", None) and idx < len(route.wp_zone_ids):
            zone_ids = list(route.wp_zone_ids[idx] or [])
        if getattr(route, "wp_zone_types", None) and idx < len(route.wp_zone_types):
            zone_types = list(route.wp_zone_types[idx] or [])
        return zone_ids, zone_types

    def _semantic_context(self, route: RoutePath, matched_idx: int, target_idx: int, lookahead_pts: int) -> dict:
        zone_ids, zone_types = self._current_zone_context(route, self._clamp_idx(route, matched_idx))
        result = {
            "next_semantic_id": None,
            "next_semantic_type": None,
            "next_semantic_label": None,
            "next_semantic_distance_m": None,
            "expected_control_type": None,
            "current_zone_ids": zone_ids,
            "current_zone_types": zone_types,
        }
        if not getattr(route, "route_events", None):
            return result

        n = len(route.waypoints)
        if n == 0:
            return result

        search_start = self._clamp_idx(route, matched_idx)
        candidate_events = []
        for event in route.route_events:
            try:
                event_wp_idx = int(event.get("waypoint_idx", 0))
            except (TypeError, ValueError):
                event_wp_idx = 0
            if route.closed_loop:
                distance_pts = (event_wp_idx - search_start) % n
            else:
                if event_wp_idx < search_start:
                    continue
                distance_pts = event_wp_idx - search_start
            candidate_events.append((distance_pts, event))

        if not candidate_events:
            return result

        candidate_events.sort(key=lambda item: (item[0], str(item[1].get("id", ""))))
        preferred = None
        for distance_pts, event in candidate_events:
            semantic_type = str(event.get("type") or "")
            if semantic_type == "destination" and distance_pts > 0:
                continue
            preferred = (distance_pts, event)
            break
        if preferred is None:
            preferred = candidate_events[0]

        distance_pts, event = preferred
        result.update(
            {
                "next_semantic_id": str(event.get("id")) if event.get("id") is not None else None,
                "next_semantic_type": str(event.get("type")) if event.get("type") is not None else None,
                "next_semantic_label": str(event.get("label")) if event.get("label") is not None else None,
                "next_semantic_distance_m": round(float(distance_pts) * float(self.graph.step_m), 5),
                "expected_control_type": (
                    str(event.get("control_type"))
                    if event.get("control_type") is not None
                    else ("stop" if str(event.get("type") or "") == "stopline" else None)
                ),
            }
        )
        return result

    def get_upcoming_context(self) -> dict:
        if self.active_route is None or len(self.active_route.waypoints) == 0:
            return {
                "route_active": False,
                "current_node_id": None,
                "current_node_attr": ATTR_NORMAL,
                "upcoming_node_id": None,
                "upcoming_node_attr": ATTR_NORMAL,
                "maneuver_type": "none",
                "route_progress": 0.0,
                "destination_node_id": None,
                "next_semantic_id": None,
                "next_semantic_type": None,
                "expected_control_type": None,
            }
        route = self.active_route
        current_idx = self._clamp_idx(route, self.matched_idx)
        target_idx = self._clamp_idx(route, self.target_idx)
        semantic = self._semantic_context(route, current_idx, target_idx, 8)
        return {
            "route_active": self.route_active,
            "current_node_id": route.wp_node_ids[current_idx],
            "current_node_attr": int(route.wp_node_attrs[current_idx]),
            "upcoming_node_id": route.wp_node_ids[target_idx],
            "upcoming_node_attr": int(route.wp_node_attrs[target_idx]),
            "maneuver_type": self._infer_maneuver_type(route, target_idx, 8),
            "route_progress": self._progress(route, current_idx),
            "destination_node_id": self.destination_node_id,
            "next_semantic_id": semantic["next_semantic_id"],
            "next_semantic_type": semantic["next_semantic_type"],
            "expected_control_type": semantic["expected_control_type"],
        }

    def _progress(self, route: RoutePath, idx: int) -> float:
        n = len(route.waypoints)
        if n <= 1:
            return 0.0
        if route.closed_loop:
            return float(self._clamp_idx(route, idx)) / float(n)
        return float(self._clamp_idx(route, idx)) / float(n - 1)

    # ------------------------------------------------------------------
    # Main tracking update
    # ------------------------------------------------------------------
    def update(
        self,
        x: float,
        y: float,
        yaw: float,
        speed_mps: float,
        min_lookahead_m: float,
        lookahead_time_s: float,
        max_lookahead_m: float,
        precision_lookahead_m: float,
        lookahead_pts: int,
        search_window: int,
        distance_weight: float,
        heading_weight: float,
    ) -> PathUpdate:
        route = self.active_route
        if route is None or route.waypoints.size == 0:
            return PathUpdate(
                route_active=False,
                route_id=self.route_id,
                route_points=[],
                route_waypoints=[],
                destination_node_id=self.destination_node_id,
                destination_label=(self._describe_destination(self._current_destination_spec()) or {}).get("label"),
                route_queue=self._route_queue_payload(),
                current_node_id=None,
                current_node_attr=ATTR_NORMAL,
                upcoming_node_id=None,
                upcoming_node_attr=ATTR_NORMAL,
                maneuver_type="none",
                route_progress=0.0,
                route_completed=self.route_completed,
                waypoint_mode_active=False,
                matched_idx=0,
                target_idx=0,
                matched_x=float(x),
                matched_y=float(y),
                matched_yaw=float(yaw),
                path_psi=float(yaw),
                path_kappa=0.0,
                path_heading_change_rad=0.0,
                error_m=0.0,
                heading_rad=0.0,
                map_match_error_m=0.0,
                remaining_distance_m=0.0,
                replans=self.replans,
                route_source="none",
                destination_point=None,
                next_semantic_id=None,
                next_semantic_type=None,
                next_semantic_label=None,
                next_semantic_distance_m=None,
                expected_control_type=None,
                current_zone_ids=[],
                current_zone_types=[],
                map_metadata=self.graph.get_map_metadata(),
                available_destinations=self.graph.get_available_destinations(),
            )

        prev_matched_idx = int(self.matched_idx)
        prev_target_idx = int(self.target_idx)
        # Advance the search centre by the raw position's forward projection along
        # the current path direction.  When the DR position has drifted ahead of the
        # last matched waypoint (common in single-line curve mode where camera
        # corrections stop and speed is low), every segment within the window has a
        # similar lateral error, so the continuity penalty keeps matched_idx frozen.
        # Projecting the raw-to-matched displacement along the path direction gives
        # the number of waypoints the car has actually advanced, and centering the
        # search there lets the matcher find the correct segment instead of staying
        # stuck at the old position.
        _wp = route.waypoints[self._clamp_idx(route, self.matched_idx)]
        _along_path_m = (
            (float(x) - float(_wp[0])) * math.cos(float(_wp[2]))
            + (float(y) - float(_wp[1])) * math.sin(float(_wp[2]))
        )
        _forward_pts = min(
            max(0, int(round(_along_path_m / max(float(self.graph.step_m), 1e-6)))),
            int(search_window) * 2,
        )
        _search_center = self._clamp_idx(route, self.matched_idx + _forward_pts)
        map_match = self._project_pose_to_route(
            route,
            x,
            y,
            yaw,
            search_center=_search_center,
            search_window=search_window,
            distance_weight=distance_weight,
            heading_weight=heading_weight,
        )
        local_match = dict(map_match)
        local_error_m = float(local_match.get("map_match_error_m", 0.0))
        map_match_error_m = local_error_m
        recovery_error_m: float | None = None
        recovery_matched_idx: int | None = None
        global_error_m: float | None = None
        global_matched_idx: int | None = None
        match_source = "local_window"
        # Recovery mode: si el matcher local queda pegado a un tramo viejo y la
        # pose real ya avanzó mucho por el loop, la continuidad local impide
        # saltar a los waypoints correctos. En ese caso abrimos una búsqueda más
        # amplia SOLO hacia adelante sobre la misma ruta y aceptamos el salto
        # únicamente si reduce el error de forma clara.
        if map_match_error_m > _MAP_MATCH_RECOVERY_ERROR_M:
            recovery_half_window = max(
                int(search_window),
                int(search_window) * _MAP_MATCH_RECOVERY_WINDOW_MULT,
            )
            recovery_center = self._clamp_idx(route, self.matched_idx + recovery_half_window)
            recovery_match = self._project_pose_to_route(
                route,
                x,
                y,
                yaw,
                search_center=recovery_center,
                search_window=recovery_half_window,
                distance_weight=distance_weight,
                heading_weight=heading_weight,
            )
            recovery_error_m = float(recovery_match.get("map_match_error_m", map_match_error_m))
            recovery_matched_idx = int(recovery_match.get("matched_idx", self.matched_idx))
            if recovery_error_m + _MAP_MATCH_RECOVERY_MIN_IMPROVEMENT_M < map_match_error_m:
                map_match = recovery_match
                map_match_error_m = recovery_error_m
                match_source = "forward_recovery"
        # Full-route recovery: si incluso la búsqueda amplia hacia adelante
        # sigue lejos, soltamos por completo la penalización de continuidad y
        # buscamos el mejor segmento sobre TODA la ruta activa. Esto corta el
        # caso donde el matcher local queda pegado en un loop cercano pero el
        # ego físico ya está sobre otro tramo que pasa cerca del mismo lugar.
        if map_match_error_m > _MAP_MATCH_GLOBAL_RECOVERY_ERROR_M:
            global_match = self._project_pose_to_route(
                route,
                x,
                y,
                yaw,
                search_center=0,
                search_window=max(1, len(route.waypoints)),
                distance_weight=distance_weight,
                heading_weight=heading_weight,
                continuity_weight_scale=0.0,
            )
            global_error_m = float(global_match.get("map_match_error_m", map_match_error_m))
            global_matched_idx = int(global_match.get("matched_idx", self.matched_idx))
            if global_error_m + _MAP_MATCH_GLOBAL_MIN_IMPROVEMENT_M < map_match_error_m:
                map_match = global_match
                map_match_error_m = global_error_m
                match_source = "global_recovery"
        matched_idx = int(map_match.get("matched_idx", self.matched_idx))
        matched_x = float(map_match.get("matched_x", x))
        matched_y = float(map_match.get("matched_y", y))
        matched_yaw = float(map_match.get("path_psi", yaw))
        self.matched_idx = self._clamp_idx(route, matched_idx)

        lookahead_m = min(
            max(float(min_lookahead_m), abs(float(speed_mps)) * float(lookahead_time_s)),
            float(max_lookahead_m),
        )
        target_idx = self._find_waypoint_ahead(route, self.matched_idx, lookahead_m=lookahead_m)
        # Detect precision nodes both from the current matched pose and from the
        # forward control target. Looking only from target_idx can skip a nearby
        # stopline/intersection that lies between matched_idx and target_idx.
        in_precision_zone = (
            self._is_precision_zone(route, self.matched_idx, lookahead_pts=lookahead_pts) or
            self._is_precision_zone(route, target_idx, lookahead_pts=lookahead_pts)
        )
        if in_precision_zone:
            precision_lookahead = max(float(self.graph.step_m), min(lookahead_m, float(precision_lookahead_m)))
            target_idx = self._find_waypoint_ahead(route, self.matched_idx, lookahead_m=precision_lookahead)

        self.target_idx = self._clamp_idx(route, target_idx)
        error_m, heading_rad = self._compute_tracking_error(route, matched_x, matched_y, matched_yaw, self.target_idx)
        path_psi = float(route.waypoints[self.target_idx][2])
        path_kappa = self._get_curvature(route, self.target_idx)
        path_heading_change_rad = self._get_heading_change_ahead(route, self.matched_idx, lookahead_m=1.5)
        current_node_id = route.wp_node_ids[self.matched_idx] if route.wp_node_ids else None
        upcoming_node_id = route.wp_node_ids[self.target_idx] if route.wp_node_ids else None
        current_attr = int(route.wp_node_attrs[self.matched_idx]) if len(route.wp_node_attrs) else ATTR_NORMAL
        upcoming_attr = int(route.wp_node_attrs[self.target_idx]) if len(route.wp_node_attrs) else ATTR_NORMAL
        semantic_context = self._semantic_context(route, self.matched_idx, self.target_idx, lookahead_pts)
        maneuver_type = self._infer_maneuver_type(
            route,
            self.target_idx,
            lookahead_pts,
            next_semantic_type=semantic_context["next_semantic_type"],
            expected_control_type=semantic_context["expected_control_type"],
        )
        route_progress = self._progress(route, self.matched_idx)
        remaining_distance_m = self._remaining_distance(route, self.matched_idx)

        if not route.closed_loop and remaining_distance_m <= max(0.12, 2.0 * float(self.graph.step_m)):
            self.route_completed = True
            self.route_active = False

        destination_info = self._describe_destination(self._current_destination_spec())

        live_log(
            "route_planner", event="tracking_update",
            route_id=self.route_id,
            route_active=bool(self.route_active),
            route_source=str(getattr(route, "source", "unknown") or "unknown"),
            prev_matched_idx=prev_matched_idx,
            prev_target_idx=prev_target_idx,
            matched_idx=int(self.matched_idx),
            target_idx=int(self.target_idx),
            match_source=match_source,
            local_error_m=local_error_m,
            recovery_error_m=recovery_error_m,
            global_error_m=global_error_m,
            local_matched_idx=int(local_match.get("matched_idx", prev_matched_idx)),
            recovery_matched_idx=recovery_matched_idx,
            global_matched_idx=global_matched_idx,
            along_path_m=float(_along_path_m),
            forward_pts=int(_forward_pts),
            search_center=int(_search_center),
            lookahead_m=float(lookahead_m),
            waypoint_mode_active=bool(in_precision_zone),
            current_node_id=current_node_id,
            upcoming_node_id=upcoming_node_id,
            current_attr=int(current_attr),
            upcoming_attr=int(upcoming_attr),
            matched_x=matched_x,
            matched_y=matched_y,
            matched_yaw=matched_yaw,
            map_match_error_m=map_match_error_m,
            tracking_error_m=float(error_m),
            heading_rad=float(heading_rad),
            remaining_distance_m=float(remaining_distance_m),
            route_progress=float(route_progress),
            route_completed=bool(self.route_completed),
            next_semantic_type=semantic_context["next_semantic_type"],
            next_semantic_distance_m=semantic_context["next_semantic_distance_m"],
            expected_control_type=semantic_context["expected_control_type"],
            destination_node_id=self.destination_node_id,
        )

        return PathUpdate(
            route_active=self.route_active,
            route_id=self.route_id,
            route_points=self._current_route_preview(),
            route_waypoints=_serialize_waypoints(route.waypoints),
            destination_node_id=self.destination_node_id,
            destination_label=(destination_info or {}).get("label"),
            route_queue=self._route_queue_payload(),
            current_node_id=current_node_id,
            current_node_attr=current_attr,
            upcoming_node_id=upcoming_node_id,
            upcoming_node_attr=upcoming_attr,
            maneuver_type=maneuver_type,
            route_progress=route_progress,
            route_completed=self.route_completed,
            waypoint_mode_active=in_precision_zone,
            matched_idx=self.matched_idx,
            target_idx=self.target_idx,
            matched_x=matched_x,
            matched_y=matched_y,
            matched_yaw=matched_yaw,
            path_psi=path_psi,
            path_kappa=path_kappa,
            path_heading_change_rad=path_heading_change_rad,
            error_m=error_m,
            heading_rad=heading_rad,
            map_match_error_m=map_match_error_m,
            remaining_distance_m=remaining_distance_m,
            replans=self.replans,
            route_source=route.source,
            destination_point=route.destination_point(),
            next_semantic_id=semantic_context["next_semantic_id"],
            next_semantic_type=semantic_context["next_semantic_type"],
            next_semantic_label=semantic_context["next_semantic_label"],
            next_semantic_distance_m=semantic_context["next_semantic_distance_m"],
            expected_control_type=semantic_context["expected_control_type"],
            current_zone_ids=semantic_context["current_zone_ids"],
            current_zone_types=semantic_context["current_zone_types"],
            map_metadata=dict(getattr(route, "map_metadata", {}) or self.graph.get_map_metadata()),
            available_destinations=list(getattr(route, "available_destinations", None) or self.graph.get_available_destinations()),
        )

    # ------------------------------------------------------------------
    # MPC reference trajectory extraction
    # ------------------------------------------------------------------
    def get_mpc_references(
        self,
        matched_idx: int,
        N: int,
        T: float,
        v_ref: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Sample *N+1* state refs and *N* input refs for the full Acados MPC.

        Waypoints are sampled at the expected time-proportional positions
        along the route (distance = v_ref * k * T for step *k*), with linear
        interpolation between dense waypoints.

        Returns
        -------
        state_refs : (N+1, 3) ndarray  [x, y, yaw]   or ``None``
        input_refs : (N, 2)   ndarray  [v_ref, 0.0]   or ``None``
        """
        route = self.active_route
        if route is None or route.waypoints.size == 0:
            return None, None

        n = len(route.waypoints)
        step_m = max(float(self.graph.step_m), 1e-6)
        safe_v = max(abs(float(v_ref)), 0.01)

        state_refs = np.empty((N + 1, 3), dtype=np.float64)
        idx0 = self._clamp_idx(route, int(matched_idx))
        state_refs[0] = route.waypoints[idx0]

        for k in range(1, N + 1):
            dist_ahead = safe_v * k * T
            wp_offset = dist_ahead / step_m
            frac_idx = float(idx0) + wp_offset

            idx_lo = int(frac_idx)
            t = frac_idx - idx_lo
            if route.closed_loop:
                lo = idx_lo % n
                hi = (idx_lo + 1) % n
            else:
                lo = min(idx_lo, n - 1)
                hi = min(idx_lo + 1, n - 1)

            wp_lo = route.waypoints[lo]
            wp_hi = route.waypoints[hi]
            state_refs[k, 0] = wp_lo[0] + t * (wp_hi[0] - wp_lo[0])
            state_refs[k, 1] = wp_lo[1] + t * (wp_hi[1] - wp_lo[1])
            yaw_diff = self._wrap_angle(float(wp_hi[2]) - float(wp_lo[2]))
            state_refs[k, 2] = wp_lo[2] + t * yaw_diff

        input_refs = np.column_stack([
            np.full(N, safe_v, dtype=np.float64),
            np.zeros(N, dtype=np.float64),
        ])

        return state_refs, input_refs

    def build_navigation_status(self, update: PathUpdate) -> dict:
        return {
            "route_active": bool(update.route_active),
            "route_id": update.route_id,
            "destination": update.destination_point,
            "destination_point": update.destination_point,
            "destination_lanelet_id": update.destination_node_id,
            "destination_label": update.destination_label,
            "queue": list(update.route_queue),
            "route_queue": list(update.route_queue),
            "current_node_attr": int(update.current_node_attr),
            "upcoming_node_attr": int(update.upcoming_node_attr),
            "maneuver_type": update.maneuver_type,
            "next_semantic_id": update.next_semantic_id,
            "next_semantic_type": update.next_semantic_type,
            "next_semantic_label": update.next_semantic_label,
            "next_semantic_distance_m": update.next_semantic_distance_m,
            "expected_control_type": update.expected_control_type,
            "current_zone_ids": list(update.current_zone_ids),
            "current_zone_types": list(update.current_zone_types),
            "progress": round(float(update.route_progress), 5),
            "route_progress": round(float(update.route_progress), 5),
            "replans": int(update.replans),
            "route_completed": bool(update.route_completed),
            "route_points": list(update.route_points),
            "route_source": update.route_source,
            "waypoint_mode_active": bool(update.waypoint_mode_active),
            "map_metadata": dict(update.map_metadata),
        }
