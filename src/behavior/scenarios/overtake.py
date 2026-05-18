from __future__ import annotations

import math
from dataclasses import replace

import config
import numpy as np

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.routing.lanelet.attributes import ATTR_HIGHWAY_LEFT, ATTR_HIGHWAY_RIGHT
from src.core.types.behavior import (
    BehaviorOutput,
    BehaviorPathPlan,
    MotionState,
    ScenarioName,
    TurnSignalCommand,
)
from src.utils.live_log import live_log


class Overtake(BaseScenario):
    """Simple left-side overtake when a slow frontal obstacle is present."""

    name = "overtake"
    priority = 65  # sobre Intersection/Roundabout/Highway; debajo de Stop/Parking/Crosswalk

    def __init__(self) -> None:
        self._raw_lidar_confirm_ticks = 0
        self._raw_lidar_last_xy: tuple[float, float] | None = None

    def is_active(self, ctx: PlanningContext) -> bool:
        if not bool(getattr(config, "OVERTAKE_ENABLED", True)):
            live_log("overtake", event="activation_check", enabled=False, active=False)
            return False
        front_detected = self._front_vehicle_or_obstacle(ctx)
        left_clear = self._left_corridor_clear(ctx)
        active = bool(front_detected and left_clear)
        debug = _overtake_debug_snapshot(
            ctx,
            front_detected=front_detected,
            left_clear=left_clear,
            raw_lidar_confirm_ticks=self._raw_lidar_confirm_ticks,
        )
        live_log(
            "overtake",
            event="activation_check",
            enabled=True,
            active=active,
            front_vehicle_or_obstacle=bool(front_detected),
            left_corridor_clear=bool(left_clear),
            **debug,
        )
        return active

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        speed = float(getattr(config, "OVERTAKE_SPEED_MPS", 0.30))
        offset = _effective_lateral_offset_m()
        nominal_offset = float(getattr(config, "OVERTAKE_LATERAL_OFFSET_M", 0.16))
        clearance = float(getattr(config, "OVERTAKE_MIN_OBSTACLE_CLEARANCE_M", 0.04))
        plan = self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=min(speed, float(ctx.max_speed_mps)),
            scenario_name=ScenarioName.LANE_KEEP.value,
            notes={
                "reason": "overtake_active",
                "overtake": True,
                "overtake_corridor_escape": True,
                "overtake_nominal_lateral_offset_m": nominal_offset,
                "overtake_effective_lateral_offset_m": offset,
                "overtake_clearance_m": clearance,
                "turn_signal": TurnSignalCommand.LEFT.value,
                "lateral_offset_m": offset,
            },
        )
        if not plan.valid or plan.target_path.shape[0] == 0:
            return plan
        target_path = np.array(plan.target_path, copy=True, dtype=float)
        for idx in range(target_path.shape[0]):
            yaw = float(target_path[idx, 2])
            target_path[idx, 0] += -math.sin(yaw) * offset
            target_path[idx, 1] += math.cos(yaw) * offset
        return replace(
            plan,
            target_path=target_path,
            scenario_name=self.name,
            motion_state=MotionState.OVERTAKE_ACTIVE.value,
            notes={**plan.notes, "turn_signal": TurnSignalCommand.LEFT.value},
        )

    def plan_path(self, ctx: PlanningContext) -> BehaviorPathPlan:
        plan = self.plan(ctx)
        notes = dict(plan.notes)
        turn_signal = str(notes.get("turn_signal") or TurnSignalCommand.LEFT.value)
        return BehaviorPathPlan(
            timestamp=float(plan.timestamp or ctx.now_s),
            raw_path=np.asarray(plan.target_path, dtype=float),
            base_speed_profile=np.asarray(plan.speed_profile, dtype=float),
            scenario_name=str(plan.scenario_name),
            valid=bool(plan.valid),
            stop_required=bool(plan.stop_required),
            turn_signal=turn_signal,
            notes=notes,
        )

    def _front_vehicle_or_obstacle(self, ctx: PlanningContext) -> bool:
        fmin = float(getattr(config, "OVERTAKE_FORWARD_MIN_M", 0.25))
        fmax = float(getattr(config, "OVERTAKE_FORWARD_MAX_M", 1.25))
        half_width = float(getattr(config, "LIDAR_OBSTACLE_CORRIDOR_HALF_WIDTH_M", 0.14))

        for obs in ctx.lidar_obstacles:
            if not _is_semantic_lidar_overtake_obstacle(obs):
                continue
            if fmin <= float(obs.x_m) <= fmax and abs(float(obs.y_m)) <= half_width:
                self._reset_raw_lidar_confirmation()
                return True

        ego_x, ego_y = ctx.pose.fused_pose.x, ctx.pose.fused_pose.y
        ego_yaw = ctx.pose.fused_pose.yaw
        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)
        for track in ctx.tracked_objects:
            if str(track.class_name).lower() not in {"car", "vehicle", "bus", "truck", "obstacle"}:
                continue
            tx, ty = track.position_world_xy
            dx, dy = tx - ego_x, ty - ego_y
            forward = dx * cos_y + dy * sin_y
            lateral = -dx * sin_y + dy * cos_y
            if fmin <= forward <= fmax and abs(lateral) <= half_width:
                self._reset_raw_lidar_confirmation()
                return True

        return self._raw_lidar_front_confirmed(ctx, fmin=fmin, half_width=half_width)

    def _raw_lidar_front_confirmed(
        self,
        ctx: PlanningContext,
        *,
        fmin: float,
        half_width: float,
    ) -> bool:
        if not bool(getattr(config, "OVERTAKE_RAW_LIDAR_FALLBACK_ENABLED", True)):
            self._reset_raw_lidar_confirmation()
            return False
        if bool(getattr(config, "OVERTAKE_RAW_LIDAR_REQUIRES_HIGHWAY", True)) and not _route_is_highway_like(ctx):
            self._reset_raw_lidar_confirmation()
            return False

        candidate = _raw_lidar_front_candidate(ctx, fmin=fmin, half_width=half_width)
        if candidate is None:
            self._reset_raw_lidar_confirmation()
            return False

        xy = (float(candidate.x_m), float(candidate.y_m))
        if self._raw_lidar_last_xy is None:
            self._raw_lidar_confirm_ticks = 1
        else:
            dx = abs(xy[0] - self._raw_lidar_last_xy[0])
            dy = abs(xy[1] - self._raw_lidar_last_xy[1])
            if dx <= 0.25 and dy <= 0.08:
                self._raw_lidar_confirm_ticks += 1
            else:
                self._raw_lidar_confirm_ticks = 1
        self._raw_lidar_last_xy = xy
        required = max(1, int(getattr(config, "OVERTAKE_RAW_LIDAR_CONFIRM_TICKS", 3)))
        return self._raw_lidar_confirm_ticks >= required

    def _reset_raw_lidar_confirmation(self) -> None:
        self._raw_lidar_confirm_ticks = 0
        self._raw_lidar_last_xy = None

    def _left_corridor_clear(self, ctx: PlanningContext) -> bool:
        forward_max = float(getattr(config, "OVERTAKE_SIDE_CLEAR_FORWARD_M", 1.25))
        offset = _effective_lateral_offset_m()
        lidar_corridor_half_width = _overtake_lidar_corridor_half_width_m()
        tracked_corridor_half_width = _overtake_tracked_corridor_half_width_m()
        for obs in ctx.lidar_obstacles:
            if _point_hits_overtake_corridor(
                forward_m=float(obs.x_m),
                lateral_m=float(obs.y_m),
                offset_m=offset,
                corridor_half_width_m=lidar_corridor_half_width,
                forward_max_m=forward_max,
            ):
                return False

        ego_x, ego_y = ctx.pose.fused_pose.x, ctx.pose.fused_pose.y
        ego_yaw = ctx.pose.fused_pose.yaw
        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)
        for track in ctx.tracked_objects:
            if str(track.class_name).lower() not in {"car", "vehicle", "bus", "truck", "obstacle"}:
                continue
            tx, ty = track.position_world_xy
            dx, dy = tx - ego_x, ty - ego_y
            forward = dx * cos_y + dy * sin_y
            lateral = -dx * sin_y + dy * cos_y
            if _point_hits_overtake_corridor(
                forward_m=forward,
                lateral_m=lateral,
                offset_m=offset,
                corridor_half_width_m=tracked_corridor_half_width,
                forward_max_m=forward_max,
            ):
                return False
        return True


def _effective_lateral_offset_m() -> float:
    nominal = float(getattr(config, "OVERTAKE_LATERAL_OFFSET_M", 0.16))
    vehicle_width = max(0.0, float(getattr(config, "BEHAVIOR_VEHICLE_WIDTH_M", 0.19)))
    clearance = max(0.0, float(getattr(config, "OVERTAKE_MIN_OBSTACLE_CLEARANCE_M", 0.04)))
    max_offset = max(0.0, float(getattr(config, "OVERTAKE_MAX_LATERAL_OFFSET_M", 0.28)))
    required = vehicle_width + clearance
    return float(min(max_offset, max(nominal, required)))


def _overtake_lidar_corridor_half_width_m() -> float:
    vehicle_width = max(0.0, float(getattr(config, "BEHAVIOR_VEHICLE_WIDTH_M", 0.19)))
    clearance = max(0.0, float(getattr(config, "OVERTAKE_MIN_OBSTACLE_CLEARANCE_M", 0.04)))
    legacy_half_width = max(0.0, float(getattr(config, "OVERTAKE_SIDE_CLEAR_HALF_WIDTH_M", 0.12)))
    return float(max(legacy_half_width, 0.5 * vehicle_width + clearance))


def _overtake_tracked_corridor_half_width_m() -> float:
    vehicle_width = max(0.0, float(getattr(config, "BEHAVIOR_VEHICLE_WIDTH_M", 0.19)))
    clearance = max(0.0, float(getattr(config, "OVERTAKE_MIN_OBSTACLE_CLEARANCE_M", 0.04)))
    return float(vehicle_width + clearance)


def _is_semantic_lidar_overtake_obstacle(obs: object) -> bool:
    class_name = str(getattr(obs, "class_name", "") or "").strip().lower()
    source = str(getattr(obs, "source", "") or "").strip().lower()
    if class_name not in {"car", "vehicle", "bus", "truck", "obstacle", "road_block"}:
        return False
    return any(token in source for token in ("ai", "camera", "semantic", "tracked"))


def _raw_lidar_front_candidate(
    ctx: PlanningContext,
    *,
    fmin: float,
    half_width: float,
) -> object | None:
    fmax = min(
        float(getattr(config, "OVERTAKE_FORWARD_MAX_M", 1.25)),
        float(getattr(config, "OVERTAKE_RAW_LIDAR_FORWARD_MAX_M", 0.70)),
    )
    min_points = int(getattr(config, "OVERTAKE_RAW_LIDAR_MIN_POINTS", 12))
    min_width_rad = math.radians(float(getattr(config, "OVERTAKE_RAW_LIDAR_MIN_WIDTH_DEG", 8.0)))
    min_confidence = float(getattr(config, "OVERTAKE_RAW_LIDAR_MIN_CONFIDENCE", 0.55))
    best = None
    for obs in ctx.lidar_obstacles:
        x = float(getattr(obs, "x_m", 0.0))
        y = float(getattr(obs, "y_m", 0.0))
        if x < float(fmin) or x > fmax or abs(y) > float(half_width):
            continue
        if int(getattr(obs, "point_count", 0) or 0) < min_points:
            continue
        if float(getattr(obs, "width_rad", 0.0) or 0.0) < min_width_rad:
            continue
        if float(getattr(obs, "confidence", 0.0) or 0.0) < min_confidence:
            continue
        if best is None or x < float(getattr(best, "x_m", math.inf)):
            best = obs
    return best


def _route_is_highway_like(ctx: PlanningContext) -> bool:
    attrs = {
        int(getattr(ctx.route, "current_node_attr", 0) or 0),
        int(getattr(ctx.route, "upcoming_node_attr", 0) or 0),
    }
    if attrs.intersection({ATTR_HIGHWAY_LEFT, ATTR_HIGHWAY_RIGHT}):
        return True
    if ctx.lanelet_map is not None and ctx.route.current_lanelet_id:
        ll = ctx.lanelet_map.get_lanelet(ctx.route.current_lanelet_id)
        if ll is not None and int(ll.attribute) in (ATTR_HIGHWAY_LEFT, ATTR_HIGHWAY_RIGHT):
            return True
    for hint in tuple(getattr(ctx, "sign_hints", ()) or ()):
        kind = str(hint.get("kind", "") or hint.get("type", "") or "").strip().lower()
        if kind in {"highway_entrance", "highway_entry"}:
            return True
    return False


def _point_hits_overtake_corridor(
    *,
    forward_m: float,
    lateral_m: float,
    offset_m: float,
    corridor_half_width_m: float,
    forward_max_m: float,
) -> bool:
    if forward_m < 0.0 or forward_m > float(forward_max_m):
        return False
    clearance_gap_m = abs(float(lateral_m) - float(offset_m))
    return clearance_gap_m < max(0.0, float(corridor_half_width_m) - 1e-6)


def _overtake_debug_snapshot(
    ctx: PlanningContext,
    *,
    front_detected: bool,
    left_clear: bool,
    raw_lidar_confirm_ticks: int = 0,
) -> dict[str, object]:
    offset = _effective_lateral_offset_m()
    lidar_corridor_half_width = _overtake_lidar_corridor_half_width_m()
    tracked_corridor_half_width = _overtake_tracked_corridor_half_width_m()
    fmin = float(getattr(config, "OVERTAKE_FORWARD_MIN_M", 0.25))
    fmax = float(getattr(config, "OVERTAKE_FORWARD_MAX_M", 1.25))
    front_half_width = float(getattr(config, "LIDAR_OBSTACLE_CORRIDOR_HALF_WIDTH_M", 0.14))
    side_forward_max = float(getattr(config, "OVERTAKE_SIDE_CLEAR_FORWARD_M", 1.25))

    front_candidates: list[dict[str, object]] = []
    ignored_lidar_front_candidates: list[dict[str, object]] = []
    raw_lidar_front_candidates: list[dict[str, object]] = []
    left_blockers: list[dict[str, object]] = []
    for obs in ctx.lidar_obstacles:
        x = float(obs.x_m)
        y = float(obs.y_m)
        item = {
            "source": "lidar",
            "lidar_source": str(getattr(obs, "source", "")),
            "class_name": str(getattr(obs, "class_name", "")),
            "x_m": round(x, 3),
            "y_m": round(y, 3),
            "distance_min_m": round(float(obs.distance_min_m), 3),
            "width_deg": round(math.degrees(float(getattr(obs, "width_rad", 0.0) or 0.0)), 1),
            "point_count": int(getattr(obs, "point_count", 0) or 0),
            "confidence": round(float(getattr(obs, "confidence", 0.0) or 0.0), 3),
        }
        if fmin <= x <= fmax and abs(y) <= front_half_width:
            if _is_semantic_lidar_overtake_obstacle(obs):
                front_candidates.append(item)
            else:
                ignored_lidar_front_candidates.append(item)
        if _raw_lidar_front_candidate(ctx, fmin=fmin, half_width=front_half_width) is obs:
            raw_lidar_front_candidates.append(item)
        if _point_hits_overtake_corridor(
            forward_m=x,
            lateral_m=y,
            offset_m=offset,
            corridor_half_width_m=lidar_corridor_half_width,
            forward_max_m=side_forward_max,
        ):
            left_blockers.append(item)

    ego_x, ego_y = ctx.pose.fused_pose.x, ctx.pose.fused_pose.y
    ego_yaw = ctx.pose.fused_pose.yaw
    cos_y = math.cos(ego_yaw)
    sin_y = math.sin(ego_yaw)
    for track in ctx.tracked_objects:
        if str(track.class_name).lower() not in {"car", "vehicle", "bus", "truck", "obstacle"}:
            continue
        tx, ty = track.position_world_xy
        dx, dy = tx - ego_x, ty - ego_y
        forward = dx * cos_y + dy * sin_y
        lateral = -dx * sin_y + dy * cos_y
        item = {
            "source": "track",
            "track_id": int(track.track_id),
            "class_name": str(track.class_name),
            "x_m": round(float(forward), 3),
            "y_m": round(float(lateral), 3),
        }
        if fmin <= forward <= fmax and abs(lateral) <= front_half_width:
            front_candidates.append(item)
        if _point_hits_overtake_corridor(
            forward_m=forward,
            lateral_m=lateral,
            offset_m=offset,
            corridor_half_width_m=tracked_corridor_half_width,
            forward_max_m=side_forward_max,
        ):
            left_blockers.append(item)

    front_candidates.sort(key=lambda item: float(item.get("x_m", 0.0) or 0.0))
    ignored_lidar_front_candidates.sort(key=lambda item: float(item.get("x_m", 0.0) or 0.0))
    raw_lidar_front_candidates.sort(key=lambda item: float(item.get("x_m", 0.0) or 0.0))
    left_blockers.sort(key=lambda item: float(item.get("x_m", 0.0) or 0.0))
    return {
        "decision_reason": _overtake_decision_reason(front_detected, left_clear),
        "effective_lateral_offset_m": round(float(offset), 3),
        "clearance_m": round(float(getattr(config, "OVERTAKE_MIN_OBSTACLE_CLEARANCE_M", 0.04)), 3),
        "corridor_half_width_m": round(float(lidar_corridor_half_width), 3),
        "tracked_corridor_half_width_m": round(float(tracked_corridor_half_width), 3),
        "front_semantic_required": True,
        "raw_lidar_fallback_enabled": bool(getattr(config, "OVERTAKE_RAW_LIDAR_FALLBACK_ENABLED", True)),
        "raw_lidar_fallback_allowed": _route_is_highway_like(ctx) or not bool(getattr(config, "OVERTAKE_RAW_LIDAR_REQUIRES_HIGHWAY", True)),
        "raw_lidar_confirm_ticks": int(raw_lidar_confirm_ticks),
        "front_forward_min_m": round(fmin, 3),
        "front_forward_max_m": round(fmax, 3),
        "raw_lidar_forward_max_m": round(float(getattr(config, "OVERTAKE_RAW_LIDAR_FORWARD_MAX_M", 0.70)), 3),
        "front_half_width_m": round(front_half_width, 3),
        "side_forward_max_m": round(side_forward_max, 3),
        "lidar_obstacle_count": int(len(ctx.lidar_obstacles or ())),
        "tracked_object_count": int(len(ctx.tracked_objects or ())),
        "front_candidates": front_candidates[:4],
        "raw_lidar_front_candidates": raw_lidar_front_candidates[:4],
        "ignored_lidar_front_candidates": ignored_lidar_front_candidates[:4],
        "left_blockers": left_blockers[:4],
    }


def _overtake_decision_reason(front_detected: bool, left_clear: bool) -> str:
    if not front_detected:
        return "no_semantic_front_obstacle"
    if not left_clear:
        return "left_corridor_blocked"
    return "active"
