from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.hardware.tracking.trackGraph import (
    ATTR_CROSSWALK,
    ATTR_HIGHWAY_LEFT,
    ATTR_HIGHWAY_RIGHT,
    ATTR_INTERSECTION,
    ATTR_INTERSECTION_EXIT,
    ATTR_NORMAL,
    ATTR_ONEWAY,
    ATTR_ROUNDABOUT,
    ATTR_STOPLINE,
    RoutePath,
    TrackGraph,
    WAYPOINT_MODE_ATTRS,
)


@dataclass
class PathUpdate:
    route_active: bool
    route_id: str | None
    route_points: list[dict[str, float]]
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

    def __init__(self, graph: TrackGraph):
        self.graph = graph
        self.active_route: RoutePath | None = None
        self.route_active = False
        self.route_completed = False
        self.destination_node_id: str | None = None
        self.destination_node_ids: list[str] = []
        self.route_id: str | None = None
        self.replans = 0
        self._route_counter = 0
        self.matched_idx = 0
        self.target_idx = 0

        self.reset_route()

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
            return {"x": float(current_pose["x"]), "y": float(current_pose["y"])}
        except (KeyError, TypeError, ValueError):
            return self.graph.get_start_node_id()

    def _activate_route(
        self,
        route: RoutePath,
        destination_node_ids: list[str] | None,
        route_id: str,
    ) -> bool:
        if route is None or route.waypoints.size == 0:
            return False
        route.route_id = route_id
        self.active_route = route
        self.route_id = route_id
        self.route_active = True
        self.route_completed = False
        self.destination_node_ids = list(destination_node_ids or [])
        self.destination_node_id = self.destination_node_ids[-1] if self.destination_node_ids else None
        self.matched_idx = 0
        self.target_idx = 0
        return True

    def reset_route(self, current_pose: dict | None = None) -> bool:
        ref_ids = list(self.graph.reference_node_ids)
        closed_loop = bool(getattr(self.graph.reference_path, "closed_loop", False))
        if len(ref_ids) > 1 and ref_ids[0] == ref_ids[-1]:
            ref_ids = ref_ids[:-1]
            closed_loop = True
        if not ref_ids:
            self.active_route = None
            self.route_active = False
            self.route_completed = False
            return False

        if current_pose:
            try:
                start_node_id = self.graph.find_nearest_node(
                    float(current_pose["x"]),
                    float(current_pose["y"]),
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
        route = self.graph.build_dense_path(
            ref_ids,
            closed_loop=closed_loop,
            route_id=self._next_route_id("reference"),
            source="reference",
        )
        return self._activate_route(route, [ref_ids[-1]], route.route_id or self._next_route_id("reference"))

    def set_route(self, destination, current_pose: dict | None = None) -> bool:
        start_spec = self._pose_to_start_spec(current_pose)
        dest_id = self.graph.resolve_node_id(destination)
        if dest_id is None:
            return False
        route = self.graph.go_to(start_spec, dest_id)
        route_id = self._next_route_id("route")
        return self._activate_route(route, [dest_id], route_id)

    def set_route_queue(self, destinations, current_pose: dict | None = None) -> bool:
        start_spec = self._pose_to_start_spec(current_pose)
        dest_ids = []
        for spec in destinations or []:
            node_id = self.graph.resolve_node_id(spec)
            if node_id is not None:
                dest_ids.append(node_id)
        if not dest_ids:
            return False
        route = self.graph.go_to_multiple(start_spec, dest_ids)
        route_id = self._next_route_id("route")
        return self._activate_route(route, dest_ids, route_id)

    def handle_command(self, command: dict | None, current_pose: dict | None = None) -> bool:
        if not isinstance(command, dict):
            return False
        mode = str(command.get("mode", "") or "").lower()
        destinations = list(command.get("destinations", []) or [])
        if mode == "reset":
            self.replans += 1
            return self.reset_route(current_pose=current_pose)
        if mode == "go_to":
            if not destinations:
                return False
            self.replans += 1
            return self.set_route(destinations[0], current_pose=current_pose)
        if mode == "go_to_multiple":
            self.replans += 1
            return self.set_route_queue(destinations, current_pose=current_pose)
        return False

    def _describe_destination(self, node_id: str | None) -> dict | None:
        return self.graph.describe_destination(node_id)

    def _route_queue_payload(self) -> list[dict]:
        payload = []
        for node_id in self.destination_node_ids:
            info = self._describe_destination(node_id)
            if info is None:
                info = {"id": str(node_id), "label": f"Node {node_id}", "node_id": str(node_id)}
            payload.append(info)
        return payload

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
        continuity_weight = max(0.05, 0.25 * float(distance_weight))
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
                destination_node_id=self.destination_node_id,
                destination_label=(self._describe_destination(self.destination_node_id) or {}).get("label"),
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
        matched_idx = int(map_match.get("matched_idx", self.matched_idx))
        matched_x = float(map_match.get("matched_x", x))
        matched_y = float(map_match.get("matched_y", y))
        matched_yaw = float(map_match.get("path_psi", yaw))
        map_match_error_m = float(map_match.get("map_match_error_m", 0.0))
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

        destination_info = self._describe_destination(self.destination_node_id)

        return PathUpdate(
            route_active=self.route_active,
            route_id=self.route_id,
            route_points=self._current_route_preview(),
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

    def build_navigation_status(self, update: PathUpdate) -> dict:
        return {
            "route_active": bool(update.route_active),
            "route_id": update.route_id,
            "destination": update.destination_node_id,
            "destination_node_id": update.destination_node_id,
            "destination_label": update.destination_label,
            "queue": list(update.route_queue),
            "route_queue": list(update.route_queue),
            "destination_point": update.destination_point,
            "current_node": update.current_node_id,
            "current_node_id": update.current_node_id,
            "current_node_attr": int(update.current_node_attr),
            "upcoming_node": update.upcoming_node_id,
            "upcoming_node_id": update.upcoming_node_id,
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
            "map_nodes": self.graph.get_map_nodes(),
            "available_destinations": list(update.available_destinations),
        }
