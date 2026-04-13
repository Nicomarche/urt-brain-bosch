from __future__ import annotations

import os
import time

from src.hardware.pipeline.sharedTypes import Pose2D, PoseEstimate, RouteContext
from src.hardware.tracking.pathManager import PathManager
from src.hardware.tracking.threadTracking import (
    _ADVANCE_DIST,
    _GRAPHML_PATH,
    _LOOKAHEAD_PTS,
    _LOOKAHEAD_TIME_S,
    _MAP_MATCH_DISTANCE_W,
    _MAP_MATCH_HEADING_W,
    _MAP_MATCH_SEARCH_WP,
    _MAX_LOOKAHEAD_M,
    _PRECISION_LOOKAHEAD_M,
    _SEMANTICS_PATH,
    _STEP_M,
)
from src.hardware.tracking.trackGraph import TrackGraph
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import Location, NavigationCommand, NavigationStatus
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber


class threadNavigationPlanner(ThreadWithStop):
    """Dedicated route/map-matching planner fed from the latest pose estimate."""

    def __init__(
        self,
        queuesList,
        tracking_state,
        pose_estimate_buffer,
        route_context_buffer,
        *,
        logging=None,
        debugging: bool = False,
        visualizer=None,
    ):
        super().__init__(pause=0.05)
        self.queuesList = queuesList
        self.tracking_state = tracking_state
        self.pose_estimate_buffer = pose_estimate_buffer
        self.route_context_buffer = route_context_buffer
        self.logging = logging
        self.debugging = debugging
        self.visualizer = visualizer
        self._nav_cmd_sub = messageHandlerSubscriber(
            queuesList, NavigationCommand, "lastOnly", subscribe=True
        )
        self._loc_sender = messageHandlerSender(queuesList, Location)
        self._nav_status_sender = messageHandlerSender(queuesList, NavigationStatus)
        self._graph = self._load_graph()
        self._path_manager = PathManager(self._graph) if self._graph is not None else None
        if self._graph is not None:
            x0, y0, yaw0 = self._graph.get_start_pose()
            self._last_pose = Pose2D(float(x0), float(y0), float(yaw0))
        else:
            self._last_pose = Pose2D()
        self._last_speed = 0.0

    def _load_graph(self):
        graphml_path = _GRAPHML_PATH
        if not os.path.isabs(graphml_path):
            _here = os.path.dirname(os.path.abspath(__file__))
            _root = os.path.join(_here, "..", "..", "..")
            graphml_path = os.path.normpath(os.path.join(_root, graphml_path))

        semantics_path = _SEMANTICS_PATH
        if not os.path.isabs(semantics_path):
            _here = os.path.dirname(os.path.abspath(__file__))
            _root = os.path.join(_here, "..", "..", "..")
            semantics_path = os.path.normpath(os.path.join(_root, semantics_path))
        if not os.path.exists(semantics_path):
            alt_name = os.path.join(os.path.dirname(semantics_path), "Track Semantics.json")
            semantics_path = alt_name if os.path.exists(alt_name) else None

        return TrackGraph(graphml_path, step_m=_STEP_M, semantics_path=semantics_path)

    @staticmethod
    def _build_route_context(path_update, timestamp: float) -> RouteContext:
        return RouteContext(
            timestamp=float(timestamp),
            route_active=bool(path_update.route_active),
            route_id=path_update.route_id,
            route_points=list(path_update.route_points or []),
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
            map_metadata=dict(path_update.map_metadata or {}),
            available_destinations=list(path_update.available_destinations or []),
        )

    def thread_work(self):
        if self._path_manager is None:
            return

        pose_estimate, _, _ = self.pose_estimate_buffer.read_latest(with_metadata=True)
        if isinstance(pose_estimate, PoseEstimate):
            self._last_pose = pose_estimate.fused_pose
            self._last_speed = float(pose_estimate.speed_mps or 0.0)

        nav_cmd = self._nav_cmd_sub.receive()
        if isinstance(nav_cmd, dict):
            self._path_manager.handle_command(
                nav_cmd,
                current_pose={"x": self._last_pose.x, "y": self._last_pose.y},
            )

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
        self.route_context_buffer.write(route_context, timestamp=route_context.timestamp)
        if self.tracking_state is not None and hasattr(self.tracking_state, "update_from_route_context"):
            self.tracking_state.update_from_route_context(route_context)

        try:
            nav_status = self._path_manager.build_navigation_status(path_update)
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

        try:
            self._loc_sender.send(
                {
                    "x": round(float(route_context.matched_pose.x), 4),
                    "y": round(float(route_context.matched_pose.y), 4),
                }
            )
        except Exception:
            pass

        if self.visualizer is not None:
            try:
                self.visualizer.update_state(self.tracking_state.snapshot())
            except Exception:
                pass
