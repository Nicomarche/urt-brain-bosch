from __future__ import annotations

import math
import os
import time

from src.core.types import Pose2D, PoseEstimate, RouteContext
from src.core.types.routing import RegulatoryElement
from src.routing.route_planner import PathManager
from src.utils.live_log import live_log
from src.localization.relocalization_thread import (
    _ADVANCE_DIST,
    _LOOKAHEAD_PTS,
    _LOOKAHEAD_TIME_S,
    _MAP_MATCH_DISTANCE_W,
    _MAP_MATCH_HEADING_W,
    _MAP_MATCH_SEARCH_WP,
    _MAX_LOOKAHEAD_M,
    _OSM_PATH,
    _PRECISION_LOOKAHEAD_M,
    _START_LANELET_ID,
    _STEP_M,
)
from src.routing.lanelet.osm_router import OsmRouteGraph
from src.templates.threadwithstop import ThreadWithStop
from src.core.messaging.allMessages import NavigationCommand, NavigationStatus
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber

_ROUTE_LANELET_OVERRIDE_ERROR_M = 0.60
_ROUTE_LANELET_OVERRIDE_MIN_IMPROVEMENT_M = 0.10
_ROUTE_LANELET_OVERRIDE_CONFIRM_TICKS = 3
_ROUTE_LANELET_PROMOTION_MAX_ERROR_M = 0.20


def _summarize_nav_command(command: dict | None) -> dict:
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
        preview = []
        for item in destinations[:4]:
            if not isinstance(item, dict):
                continue
            row: dict[str, object] = {}
            if item.get("lanelet_id") is not None:
                row["lanelet_id"] = str(item.get("lanelet_id"))
            if item.get("x") is not None and item.get("y") is not None:
                try:
                    row["x"] = round(float(item["x"]), 3)
                    row["y"] = round(float(item["y"]), 3)
                except (TypeError, ValueError):
                    pass
            if row:
                preview.append(row)
        summary["destinations_preview"] = preview
    return summary


def _max_present_float(*values: float | None) -> float:
    present = [float(item) for item in values if item is not None]
    if not present:
        return 0.0
    return max(present)


class threadNavigationPlanner(ThreadWithStop):
    """Dedicated route/map-matching planner fed from the latest pose estimate."""

    def __init__(
        self,
        queuesList,
        tracking_state,
        pose_estimate_buffer,
        route_context_buffer,
        *,
        lanelet_map=None,
        logging=None,
        debugging: bool = False,
        visualizer=None,
    ):
        super().__init__(pause=0.05)
        self.queuesList = queuesList
        self.tracking_state = tracking_state
        self.pose_estimate_buffer = pose_estimate_buffer
        self.route_context_buffer = route_context_buffer
        # `lanelet_map` (Phase 3): si está presente, se usa para resolver
        # `RouteContext.current_lanelet_id` en cada tick vía `at_pose(x,y)`.
        # Sin esto, LaneKeep cae en `_fallback_plan(reason="no_lanelet_map_or_id")`
        # porque su precondición es exactamente `route.current_lanelet_id is not None`.
        # Es decir: aunque `lanelet_map` esté cargado en `processCamera`, si NO
        # se inyecta acá el campo nunca se llena y la cadena BehaviorPlanner →
        # MotionController → dispatcher emite speed=0 indefinidamente.
        self._lanelet_map = lanelet_map
        self.logging = logging
        self.debugging = debugging
        self.visualizer = visualizer
        self._nav_cmd_sub = messageHandlerSubscriber(
            queuesList, NavigationCommand, "lastOnly", subscribe=True
        )
        self._nav_status_sender = messageHandlerSender(queuesList, NavigationStatus)
        self._graph = self._load_graph()
        self._path_manager = PathManager(self._graph) if self._graph is not None else None
        if self._graph is not None:
            x0, y0, yaw0 = self._graph.get_start_pose()
            self._last_pose = Pose2D(float(x0), float(y0), float(yaw0))
        else:
            self._last_pose = Pose2D()
        self._last_speed = 0.0
        # Tracking del cambio de nodo para emitir un evento `node_change`
        # discreto en el JSONL cada vez que current_node_id cambia. Útil para
        # comparar progreso de ruta contra la ruta nominal en el postmortem.
        self._last_node_id: object = None
        # Tracking del lanelet (para futura hysteresis si se necesita).
        self._last_lanelet_id: str | None = None
        self._last_lanelet_resolution_diag: dict[str, float | str] = {
            "lanelet_source": "none",
            "route_alignment_error_m": 0.0,
        }
        self._route_override_candidate_id: str | None = None
        self._route_override_candidate_hits: int = 0
        # Auto-lap (sim/demo): si nadie manda NavigationCommand, el thread
        # reactiva el loop de lanelets OSM usando la pose actual como
        # punto de arranque, así el coche da vueltas en loop. Se desactiva
        # apenas el dashboard mande un destino real.
        self._user_nav_cmd_received: bool = False
        self._auto_lap_last_attempt_ts: float = 0.0
        self._auto_lap_activated: bool = False
        self._pose_ever_valid: bool = False

    def _load_graph(self):
        osm_path = _OSM_PATH
        if not os.path.isabs(osm_path):
            _here = os.path.dirname(os.path.abspath(__file__))
            _root = os.path.join(_here, "..", "..")
            osm_path = os.path.normpath(os.path.join(_root, osm_path))
        return OsmRouteGraph(
            osm_path,
            step_m=_STEP_M,
            start_lanelet_id=_START_LANELET_ID,
        )

    def _future_lanelet_hints(self, current_lanelet_id: str | None, path_update) -> tuple[str, ...]:
        if self._lanelet_map is None or not current_lanelet_id:
            return ()
        return tuple(self._lanelet_map.successors_of(current_lanelet_id) or ())

    @staticmethod
    def _route_tail_from_lanelet(
        route_lanelet_id: str | None,
        route_next_lanelets: tuple[str, ...],
        anchor_lanelet_id: str | None,
    ) -> tuple[str, ...]:
        if anchor_lanelet_id is None:
            return ()
        ordered_route = []
        if route_lanelet_id:
            ordered_route.append(str(route_lanelet_id))
        ordered_route.extend(str(item) for item in (route_next_lanelets or ()) if str(item))
        if not ordered_route:
            return ()
        anchor = str(anchor_lanelet_id)
        if anchor not in ordered_route:
            return ()
        start_idx = ordered_route.index(anchor)
        return tuple(dict.fromkeys(ordered_route[start_idx:]))

    def _shared_successor_hint_tail(
        self,
        pose_lanelet_id: str | None,
        route_lanelet_id: str | None,
        route_next_lanelets: tuple[str, ...],
    ) -> tuple[str, ...]:
        if self._lanelet_map is None or pose_lanelet_id is None or route_lanelet_id is None:
            return ()
        pose_successors = tuple(self._lanelet_map.successors_of(pose_lanelet_id) or ())
        if not pose_successors:
            return ()
        ordered_route = []
        if route_lanelet_id:
            ordered_route.append(str(route_lanelet_id))
        ordered_route.extend(str(item) for item in (route_next_lanelets or ()) if str(item))
        for lanelet_id in ordered_route:
            if lanelet_id in pose_successors:
                return self._route_tail_from_lanelet(
                    route_lanelet_id,
                    route_next_lanelets,
                    lanelet_id,
                )
        return ()

    def _resolve_lanelet_context(self, path_update) -> tuple[str | None, tuple[str, ...], tuple[RegulatoryElement, ...]]:
        current_lanelet_id: str | None = None
        next_lanelet_ids: tuple[str, ...] = ()
        regulatory_ahead: tuple[RegulatoryElement, ...] = ()
        lanelet_source = "none"
        pose_lanelet_id: str | None = None
        pose_lanelet_error_m: float | None = None
        pose_lanelet_source: str | None = None
        pose_is_predecessor_of_route = False
        route_is_predecessor_of_pose = False
        shared_tail: tuple[str, ...] = ()
        route_tail_ids: tuple[str, ...] = ()
        override_blocked_reason: str | None = None

        route_lanelet_id, route_next_lanelets, route_lanelet_source, route_lanelet_error_m = (
            self._resolve_lanelet_from_active_route(path_update)
        )
        route_lanelet_pose_error_m: float | None = None
        route_alignment_error_m: float = float(path_update.map_match_error_m or 0.0)
        if route_lanelet_id is not None:
            current_lanelet_id = route_lanelet_id
            next_lanelet_ids = route_next_lanelets
            lanelet_source = route_lanelet_source
            route_tail_ids = self._route_tail_from_lanelet(
                route_lanelet_id,
                route_next_lanelets,
                route_lanelet_id,
            )

        # Priorizamos la pose fused real. Cuando el matcher se queda pegado a
        # un tramo viejo, usar `matched_x/y` vuelve a meter al BehaviorPlanner
        # en ese mismo carril stale y el error se auto-refuerza.
        candidate_poses = (
            (
                float(self._last_pose.x),
                float(self._last_pose.y),
                float(self._last_pose.yaw),
                "fused",
            ),
            (
                float(path_update.matched_x or self._last_pose.x),
                float(path_update.matched_y or self._last_pose.y),
                float(path_update.matched_yaw or self._last_pose.yaw),
                "matched",
            ),
        )
        try:
            if self._lanelet_map is not None:
                if route_lanelet_id is not None:
                    route_lanelet_pose_error_m = self._lanelet_lateral_error(
                        route_lanelet_id,
                        float(self._last_pose.x),
                        float(self._last_pose.y),
                    )
                for px, py, pyaw, source in candidate_poses:
                    candidate_lanelet_id = self._lanelet_map.at_pose(px, py, yaw_rad=pyaw)
                    if candidate_lanelet_id:
                        pose_lanelet_id = candidate_lanelet_id
                        pose_lanelet_source = source
                        pose_lanelet_error_m = self._lanelet_lateral_error(
                            candidate_lanelet_id,
                            px,
                            py,
                        )
                        if current_lanelet_id is None:
                            current_lanelet_id = candidate_lanelet_id
                            lanelet_source = source
                            next_lanelet_ids = self._future_lanelet_hints(current_lanelet_id, path_update)
                        break
                # En intersecciones el matcher de ruta puede adelantarse al
                # edge saliente apenas el ego toca el nodo compartido, aunque
                # físicamente el auto todavía esté sobre el carril entrante.
                # Para UI y planning queremos que `current_lanelet_id` siga
                # representando el carril que pisa el ego, usando la cadena de
                # ruta sólo como hint de lo que viene después.
                route_alignment_error_m = _max_present_float(
                    float(path_update.map_match_error_m or 0.0),
                    route_lanelet_pose_error_m,
                    pose_lanelet_error_m,
                )
                if (
                    pose_lanelet_id is not None
                    and route_lanelet_id is not None
                    and pose_lanelet_error_m is not None
                    and current_lanelet_id is not None
                    and pose_lanelet_id != route_lanelet_id
                ):
                    pose_successors = tuple(self._lanelet_map.successors_of(pose_lanelet_id) or ())
                    route_successors = tuple(self._lanelet_map.successors_of(route_lanelet_id) or ())
                    pose_is_predecessor_of_route = route_lanelet_id in pose_successors
                    route_is_predecessor_of_pose = pose_lanelet_id in route_successors
                    route_close_to_pose = (
                        route_lanelet_pose_error_m is None
                        or route_lanelet_pose_error_m <= _ROUTE_LANELET_PROMOTION_MAX_ERROR_M
                    )
                    pose_close_to_lanelet = (
                        pose_lanelet_error_m <= _ROUTE_LANELET_PROMOTION_MAX_ERROR_M
                    )

                    if not (pose_close_to_lanelet and route_close_to_pose):
                        if current_lanelet_id == route_lanelet_id and route_lanelet_id is not None:
                            lanelet_source = f"{route_lanelet_source}_degraded"
                        override_blocked_reason = "alignment_error_exceeds_promotion_threshold"
                    elif pose_is_predecessor_of_route:
                        hinted_tail = [str(route_lanelet_id)]
                        hinted_tail.extend(
                            str(item)
                            for item in (route_next_lanelets or ())
                            if str(item)
                        )
                        current_lanelet_id = pose_lanelet_id
                        next_lanelet_ids = tuple(dict.fromkeys(hinted_tail))
                        lanelet_source = "fused_pose_precedes_route"
                    elif route_is_predecessor_of_pose and pose_close_to_lanelet:
                        current_lanelet_id = pose_lanelet_id
                        next_lanelet_ids = self._future_lanelet_hints(current_lanelet_id, path_update)
                        lanelet_source = "fused_pose_ahead_of_route"
                    else:
                        shared_tail = self._shared_successor_hint_tail(
                            pose_lanelet_id,
                            route_lanelet_id,
                            route_next_lanelets,
                        )
                        if shared_tail and pose_close_to_lanelet and route_close_to_pose:
                            current_lanelet_id = pose_lanelet_id
                            next_lanelet_ids = shared_tail
                            lanelet_source = "fused_pose_shared_successor_hold"
                    if current_lanelet_id == pose_lanelet_id:
                        self._route_override_candidate_id = None
                        self._route_override_candidate_hits = 0
                route_override_metric_m = max(
                    float(path_update.map_match_error_m or 0.0),
                    float(route_lanelet_pose_error_m or 0.0),
                )
                if (
                    pose_lanelet_id is not None
                    and pose_lanelet_error_m is not None
                    and current_lanelet_id is not None
                    and pose_lanelet_id != current_lanelet_id
                    and route_override_metric_m > _ROUTE_LANELET_OVERRIDE_ERROR_M
                    and (pose_lanelet_error_m + _ROUTE_LANELET_OVERRIDE_MIN_IMPROVEMENT_M)
                    < route_override_metric_m
                ):
                    candidate_is_route_relative = bool(
                        (pose_lanelet_id in route_tail_ids)
                        or pose_is_predecessor_of_route
                        or route_is_predecessor_of_pose
                        or shared_tail
                    )
                    if (
                        bool(path_update.route_active)
                        and route_lanelet_id is not None
                        and not candidate_is_route_relative
                    ):
                        override_blocked_reason = "candidate_outside_route_tail"
                        self._route_override_candidate_id = None
                        self._route_override_candidate_hits = 0
                    else:
                        prev_candidate_id = getattr(self, "_route_override_candidate_id", None)
                        prev_candidate_hits = int(
                            getattr(self, "_route_override_candidate_hits", 0) or 0
                        )
                        if prev_candidate_id == pose_lanelet_id:
                            candidate_hits = prev_candidate_hits + 1
                        else:
                            candidate_hits = 1
                        self._route_override_candidate_id = pose_lanelet_id
                        self._route_override_candidate_hits = candidate_hits
                        if candidate_hits >= _ROUTE_LANELET_OVERRIDE_CONFIRM_TICKS:
                            current_lanelet_id = pose_lanelet_id
                            next_lanelet_ids = self._future_lanelet_hints(current_lanelet_id, path_update)
                            lanelet_source = "fused_pose_override"
                            override_blocked_reason = None
                        else:
                            override_blocked_reason = "awaiting_override_confirmation"
                else:
                    self._route_override_candidate_id = None
                    self._route_override_candidate_hits = 0
            if current_lanelet_id is None:
                current_lanelet_id = self._last_lanelet_id
                lanelet_source = "sticky_last" if current_lanelet_id else "none"
            if self._lanelet_map is not None and current_lanelet_id is not None:
                if not next_lanelet_ids:
                    next_lanelet_ids = self._future_lanelet_hints(current_lanelet_id, path_update)
                regulatory_ahead = self._lanelet_map.regulatory_within(
                    current_lanelet_id,
                    distance_m=6.0,
                ) or ()
            self._last_lanelet_id = current_lanelet_id
            self._last_lanelet_resolution_diag = {
                "lanelet_source": str(lanelet_source or "none"),
                "route_alignment_error_m": float(route_alignment_error_m),
            }
            live_log(
                "nav_planner",
                event="lanelet_resolution",
                lanelet_source=lanelet_source,
                route_alignment_error_m=float(route_alignment_error_m),
                current_lanelet_id=current_lanelet_id,
                next_lanelet_ids=list(next_lanelet_ids or ()),
                fused_x=float(self._last_pose.x),
                fused_y=float(self._last_pose.y),
                matched_x=float(path_update.matched_x or 0.0),
                matched_y=float(path_update.matched_y or 0.0),
                map_match_error_m=float(path_update.map_match_error_m or 0.0),
                pose_lanelet_id=pose_lanelet_id,
                pose_lanelet_source=pose_lanelet_source,
                pose_lanelet_error_m=pose_lanelet_error_m,
                route_lanelet_id=route_lanelet_id,
                route_next_lanelets=list(route_next_lanelets or ()),
                route_lanelet_source=route_lanelet_source,
                route_lanelet_error_m=(
                    float(route_lanelet_error_m)
                    if route_lanelet_error_m is not None
                    else None
                ),
                route_lanelet_pose_error_m=route_lanelet_pose_error_m,
                route_tail_lanelet_ids=list(route_tail_ids or ()),
                override_blocked_reason=override_blocked_reason,
                route_override_candidate_hits=int(
                    getattr(self, "_route_override_candidate_hits", 0) or 0
                ),
                route_id=path_update.route_id,
                route_active=bool(path_update.route_active),
                matched_idx=int(path_update.matched_idx or 0),
                target_idx=int(path_update.target_idx or 0),
                current_node_id=path_update.current_node_id,
                upcoming_node_id=path_update.upcoming_node_id,
            )
        except Exception:
            # No queremos romper el ciclo del planner por una query de mapa.
            # Si falla, dejamos current_lanelet_id en None y LaneKeep cae
            # en fallback como antes (estado pre-fix).
            self._last_lanelet_resolution_diag = {
                "lanelet_source": str(lanelet_source or "none"),
                "route_alignment_error_m": float(route_alignment_error_m),
            }
            if self.logging is not None:
                self.logging.exception("lanelet_map query failed")
        return current_lanelet_id, next_lanelet_ids, regulatory_ahead

    @staticmethod
    def _lanelet_chain_from_route(route, wp_idx: int, *, max_lanelets: int = 8) -> tuple[str | None, tuple[str, ...]]:
        lanelet_ids = [str(item) for item in (getattr(route, "wp_node_ids", []) or []) if str(item)]
        if not lanelet_ids:
            return None, ()
        n = len(lanelet_ids)
        idx = max(0, min(n - 1, int(wp_idx)))
        lanelets = [lanelet_ids[idx]]
        cursor = idx + 1
        while cursor < n and len(lanelets) < max_lanelets:
            lanelet_id = lanelet_ids[cursor]
            if lanelet_id != lanelets[-1]:
                lanelets.append(lanelet_id)
            cursor += 1
        if bool(getattr(route, "closed_loop", False)) and len(lanelets) < max_lanelets:
            cursor = 0
            while cursor < idx and len(lanelets) < max_lanelets:
                lanelet_id = lanelet_ids[cursor]
                if lanelet_id != lanelets[-1]:
                    lanelets.append(lanelet_id)
                cursor += 1
        return lanelets[0], tuple(lanelets[1:])

    def _resolve_lanelet_from_active_route(
        self, path_update
    ) -> tuple[str | None, tuple[str, ...], str, float | None]:
        route = getattr(self._path_manager, "active_route", None)
        if route is None or getattr(route, "waypoints", None) is None or route.waypoints.size == 0:
            return None, (), "none", None

        route_idx = int(path_update.matched_idx or 0)
        route_error_m = float(path_update.map_match_error_m or 0.0)
        route_source = "route_matched"

        if route_error_m > 0.60:
            try:
                global_match = self._path_manager._project_pose_to_route(
                    route,
                    float(self._last_pose.x),
                    float(self._last_pose.y),
                    float(self._last_pose.yaw),
                    search_center=0,
                    search_window=max(1, len(route.waypoints)),
                    distance_weight=_MAP_MATCH_DISTANCE_W,
                    heading_weight=_MAP_MATCH_HEADING_W,
                    continuity_weight_scale=0.0,
                )
                global_error_m = float(global_match.get("map_match_error_m", route_error_m))
                if global_error_m + 0.15 < route_error_m:
                    route_idx = int(global_match.get("matched_idx", route_idx))
                    route_error_m = global_error_m
                    route_source = "route_global"
            except Exception:
                if self.logging is not None:
                    self.logging.exception("route-global lanelet recovery failed")

        lanelet_id, next_lanelets = self._lanelet_chain_from_route(route, route_idx)
        return lanelet_id, next_lanelets, route_source, route_error_m

    def _lanelet_lateral_error(self, lanelet_id: str | None, x: float, y: float) -> float | None:
        if self._lanelet_map is None or not lanelet_id:
            return None
        lanelet = self._lanelet_map.get_lanelet(str(lanelet_id))
        if lanelet is None:
            return None
        try:
            _, lateral_error_m = lanelet.project_arclength(float(x), float(y))
        except Exception:
            return None
        return abs(float(lateral_error_m))

    def _build_route_context(self, path_update, timestamp: float) -> RouteContext:
        # Phase 3 enrichment: si tenemos `lanelet_map`, resolvemos la lanelet
        # más cercana al matched-pose y la sucesora "más obvia" (la primera de
        # `successors_of`) para que `LaneKeep._build_constant_speed_plan` pueda
        # construir un target_path. Sin esto el escenario default cae en
        # fallback y el dispatcher emite speed=0 hasta el infinito.
        current_lanelet_id, next_lanelet_ids, regulatory_ahead = self._resolve_lanelet_context(path_update)

        return RouteContext(
            timestamp=float(timestamp),
            route_active=bool(path_update.route_active),
            route_id=path_update.route_id,
            route_points=list(path_update.route_points or []),
            route_waypoints=list(path_update.route_waypoints or []),
            destination_node_id=path_update.destination_node_id,
            destination_label=path_update.destination_label,
            route_queue=list(path_update.route_queue or []),
            current_node_id=path_update.current_node_id,
            current_node_attr=int(path_update.current_node_attr or 0),
            upcoming_node_id=path_update.upcoming_node_id,
            upcoming_node_attr=int(path_update.upcoming_node_attr or 0),
            maneuver_type=str(path_update.maneuver_type or "none"),
            route_progress=float(path_update.route_progress or 0.0),
            route_completed=bool(path_update.route_completed),
            waypoint_mode_active=bool(path_update.waypoint_mode_active),
            matched_idx=int(path_update.matched_idx or 0),
            target_idx=int(path_update.target_idx or 0),
            matched_pose=Pose2D(
                x=float(path_update.matched_x or 0.0),
                y=float(path_update.matched_y or 0.0),
                yaw=float(path_update.matched_yaw or 0.0),
            ),
            path_psi=float(path_update.path_psi or 0.0),
            path_kappa=float(path_update.path_kappa or 0.0),
            path_heading_change_rad=float(path_update.path_heading_change_rad or 0.0),
            error_m=float(path_update.error_m or 0.0),
            heading_rad=float(path_update.heading_rad or 0.0),
            map_match_error_m=float(path_update.map_match_error_m or 0.0),
            remaining_distance_m=float(path_update.remaining_distance_m or 0.0),
            replans=int(path_update.replans or 0),
            route_source=str(path_update.route_source or "none"),
            destination_point=path_update.destination_point,
            next_semantic_id=path_update.next_semantic_id,
            next_semantic_type=path_update.next_semantic_type,
            next_semantic_label=path_update.next_semantic_label,
            next_semantic_distance_m=path_update.next_semantic_distance_m,
            expected_control_type=path_update.expected_control_type,
            current_zone_ids=list(path_update.current_zone_ids or []),
            current_zone_types=list(path_update.current_zone_types or []),
            map_metadata=self._effective_map_metadata(path_update),
            available_destinations=list(path_update.available_destinations or []),
            # Phase 3:
            current_lanelet_id=current_lanelet_id,
            next_lanelet_ids=next_lanelet_ids,
            regulatory_ahead=regulatory_ahead,
        )

    def _effective_map_metadata(self, path_update) -> dict:
        if self._lanelet_map is not None:
            try:
                metadata = dict(self._lanelet_map.get_map_metadata() or {})
                if metadata:
                    return metadata
            except Exception:
                pass
        return dict(path_update.map_metadata or {})

    def thread_work(self):
        if self._path_manager is None:
            return

        pose_estimate, _, _ = self.pose_estimate_buffer.read_latest(with_metadata=True)
        if isinstance(pose_estimate, PoseEstimate):
            self._last_pose = pose_estimate.fused_pose
            self._last_speed = float(pose_estimate.speed_mps or 0.0)
            self._pose_ever_valid = True

        nav_cmd = self._nav_cmd_sub.receive()
        if isinstance(nav_cmd, dict):
            mode = str(nav_cmd.get("mode", "") or "").lower()
            live_log(
                "nav_planner", event="nav_command_received",
                command=_summarize_nav_command(nav_cmd),
                pose_x=float(self._last_pose.x),
                pose_y=float(self._last_pose.y),
                pose_yaw_rad=float(self._last_pose.yaw),
                had_route=bool(getattr(self._path_manager, "route_active", False)),
                route_id=getattr(self._path_manager, "route_id", None),
            )
            handled = self._path_manager.handle_command(
                nav_cmd,
                current_pose={
                    "x": self._last_pose.x,
                    "y": self._last_pose.y,
                    "yaw_rad": self._last_pose.yaw,
                },
            )
            active_route = getattr(self._path_manager, "active_route", None)
            live_log(
                "nav_planner", event="nav_command_result",
                command=_summarize_nav_command(nav_cmd),
                handled=bool(handled),
                route_active=bool(getattr(self._path_manager, "route_active", False)),
                route_id=getattr(self._path_manager, "route_id", None),
                replans=int(getattr(self._path_manager, "replans", 0)),
                destination_node_id=getattr(self._path_manager, "destination_node_id", None),
                waypoint_count=(
                    int(len(active_route.waypoints))
                    if active_route is not None and getattr(active_route, "waypoints", None) is not None
                    else 0
                ),
            )
            # Modos `go_to`/`go_to_multiple` indican intención explícita del
            # usuario → el auto-lap se rinde permanentemente. El modo `reset`
            # NO cuenta como intención del usuario porque es justamente lo
            # que el auto-lap haría por su cuenta.
            if handled and (mode in ("go_to", "go_to_multiple") or ("x" in nav_cmd and "y" in nav_cmd)):
                self._user_nav_cmd_received = True

        self._ensure_auto_lap_route()

        path_update = self._path_manager.update(
            self._last_pose.x,
            self._last_pose.y,
            self._last_pose.yaw,
            speed_mps=self._last_speed,
            min_lookahead_m=_ADVANCE_DIST,
            lookahead_time_s=_LOOKAHEAD_TIME_S,
            max_lookahead_m=_MAX_LOOKAHEAD_M,
            precision_lookahead_m=_PRECISION_LOOKAHEAD_M,
            lookahead_pts=_LOOKAHEAD_PTS,
            search_window=_MAP_MATCH_SEARCH_WP,
            distance_weight=_MAP_MATCH_DISTANCE_W,
            heading_weight=_MAP_MATCH_HEADING_W,
        )

        route_context = self._build_route_context(path_update, time.time())
        lanelet_diag = dict(getattr(self, "_last_lanelet_resolution_diag", {}) or {})
        self.route_context_buffer.write(route_context, timestamp=route_context.timestamp)
        if self.tracking_state is not None and hasattr(self.tracking_state, "update_from_route_context"):
            self.tracking_state.update_from_route_context(route_context)

        live_log(
            "nav_planner", event="route_update",
            route_id=route_context.route_id,
            route_source=route_context.route_source,
            lanelet_source=str(lanelet_diag.get("lanelet_source", "none") or "none"),
            route_alignment_error_m=float(lanelet_diag.get("route_alignment_error_m", 0.0) or 0.0),
            current_node_id=route_context.current_node_id,
            upcoming_node_id=route_context.upcoming_node_id,
            current_lanelet_id=route_context.current_lanelet_id,
            next_lanelet_ids=list(route_context.next_lanelet_ids or ()),
            matched_idx=int(route_context.matched_idx),
            target_idx=int(route_context.target_idx),
            map_match_error_m=float(route_context.map_match_error_m or 0.0),
            tracking_error_m=float(route_context.error_m or 0.0),
            heading_rad=float(route_context.heading_rad or 0.0),
            route_progress=float(route_context.route_progress or 0.0),
            route_active=bool(route_context.route_active),
            route_completed=bool(route_context.route_completed),
            waypoint_mode_active=bool(route_context.waypoint_mode_active),
            current_attr=int(route_context.current_node_attr or 0),
            upcoming_attr=int(route_context.upcoming_node_attr or 0),
            next_semantic_type=route_context.next_semantic_type,
            next_semantic_distance_m=route_context.next_semantic_distance_m,
            expected_control_type=route_context.expected_control_type,
            destination_node_id=route_context.destination_node_id,
            matched_x=float(route_context.matched_pose.x),
            matched_y=float(route_context.matched_pose.y),
            matched_yaw_rad=float(route_context.matched_pose.yaw),
            pose_x=float(self._last_pose.x), pose_y=float(self._last_pose.y),
            speed_mps=float(self._last_speed),
            pose_to_match_distance_m=math.hypot(
                float(self._last_pose.x) - float(route_context.matched_pose.x),
                float(self._last_pose.y) - float(route_context.matched_pose.y),
            ),
        )

        if route_context.current_node_id != self._last_node_id:
            live_log(
                "nav_planner", event="node_change",
                from_node=self._last_node_id,
                to_node=route_context.current_node_id,
                upcoming_node=route_context.upcoming_node_id,
                current_lanelet_id=route_context.current_lanelet_id,
                pose_x=float(self._last_pose.x), pose_y=float(self._last_pose.y),
            )
            self._last_node_id = route_context.current_node_id

        try:
            nav_status = self._path_manager.build_navigation_status(path_update)
            nav_status["map_metadata"] = self._effective_map_metadata(path_update)
            nav_status.update(
                {
                    "current_lanelet_id": route_context.current_lanelet_id,
                    "next_lanelet_ids": list(route_context.next_lanelet_ids or ()),
                    "lanelet_source": str(lanelet_diag.get("lanelet_source", "none") or "none"),
                    "route_alignment_error_m": float(
                        lanelet_diag.get("route_alignment_error_m", 0.0) or 0.0
                    ),
                    "regulatory_ahead": [
                        {
                            "id": reg.element_id,
                            "kind": reg.kind,
                            "x": float(reg.position_xy[0]),
                            "y": float(reg.position_xy[1]),
                            "label": reg.label,
                        }
                        for reg in (route_context.regulatory_ahead or ())
                    ],
                }
            )
            if isinstance(pose_estimate, PoseEstimate):
                nav_status.update(
                    {
                        "relocalization_mode": str(pose_estimate.relocalization_mode or "dead_reckoning"),
                        "last_relocalization_source": str(
                            pose_estimate.last_relocalization_source or "none"
                        ),
                        "last_relocalization_error_m": float(
                            pose_estimate.last_relocalization_error_m or 0.0
                        ),
                        "localization_confidence": float(
                            pose_estimate.localization_confidence or 0.0
                        ),
                    }
                )
            self._nav_status_sender.send(nav_status)
        except Exception:
            pass

        if self.visualizer is not None:
            try:
                self.visualizer.update_state(self.tracking_state.snapshot())
            except Exception:
                pass

    def _ensure_auto_lap_route(self) -> None:
        """Reactiva un loop OSM-only siguiendo la secuencia de lanelets de referencia."""
        if self._path_manager is None:
            return
        try:
            from config import AUTO_LAP_MODE, AUTO_LAP_RETRY_PERIOD_S
        except ImportError:
            return
        if not AUTO_LAP_MODE:
            return
        if self._user_nav_cmd_received:
            return
        if not self._pose_ever_valid:
            return
        if self._path_manager.route_active:
            return

        now = time.time()
        if now - self._auto_lap_last_attempt_ts < float(AUTO_LAP_RETRY_PERIOD_S):
            return
        self._auto_lap_last_attempt_ts = now

        ok = self._path_manager.reset_route(
            current_pose={
                "x": float(self._last_pose.x),
                "y": float(self._last_pose.y),
                "yaw_rad": float(self._last_pose.yaw),
            },
        )
        if ok:
            live_log(
                "nav_planner", event="auto_lap_activated",
                start_pose_x=float(self._last_pose.x),
                start_pose_y=float(self._last_pose.y),
                start_pose_yaw_rad=float(self._last_pose.yaw),
                route_id=self._path_manager.route_id,
                destination_point=getattr(self._path_manager, "destination_point", None),
            )
            self._auto_lap_activated = True
        else:
            live_log(
                "nav_planner", event="auto_lap_failed",
                reason="reset_route_returned_false",
                pose_x=float(self._last_pose.x),
                pose_y=float(self._last_pose.y),
            )
