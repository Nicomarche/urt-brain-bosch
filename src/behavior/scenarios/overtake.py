from __future__ import annotations

import math
from dataclasses import replace

import config
import numpy as np

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.core.types.behavior import BehaviorOutput, ScenarioName, TurnSignalCommand


class Overtake(BaseScenario):
    """Simple left-side overtake when a slow frontal obstacle is present."""

    name = "overtake"
    priority = 25

    def is_active(self, ctx: PlanningContext) -> bool:
        if not bool(getattr(config, "OVERTAKE_ENABLED", True)):
            return False
        return self._front_vehicle_or_obstacle(ctx) and self._left_corridor_clear(ctx)

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        speed = float(getattr(config, "OVERTAKE_SPEED_MPS", 0.30))
        offset = float(getattr(config, "OVERTAKE_LATERAL_OFFSET_M", 0.16))
        plan = self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=min(speed, float(ctx.max_speed_mps)),
            scenario_name=ScenarioName.LANE_KEEP.value,
            notes={
                "reason": "overtake_active",
                "overtake": True,
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
            notes={**plan.notes, "turn_signal": TurnSignalCommand.LEFT.value},
        )

    def _front_vehicle_or_obstacle(self, ctx: PlanningContext) -> bool:
        fmin = float(getattr(config, "OVERTAKE_FORWARD_MIN_M", 0.25))
        fmax = float(getattr(config, "OVERTAKE_FORWARD_MAX_M", 1.25))
        half_width = float(getattr(config, "LIDAR_OBSTACLE_CORRIDOR_HALF_WIDTH_M", 0.14))

        for obs in ctx.lidar_obstacles:
            if fmin <= float(obs.x_m) <= fmax and abs(float(obs.y_m)) <= half_width:
                return True

        ego_x, ego_y = ctx.pose.fused_pose.x, ctx.pose.fused_pose.y
        ego_yaw = ctx.pose.fused_pose.yaw
        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)
        for track in ctx.tracked_objects:
            if str(track.class_name).lower() not in {"car", "vehicle", "bus", "obstacle"}:
                continue
            tx, ty = track.position_world_xy
            dx, dy = tx - ego_x, ty - ego_y
            forward = dx * cos_y + dy * sin_y
            lateral = -dx * sin_y + dy * cos_y
            if fmin <= forward <= fmax and abs(lateral) <= half_width:
                return True
        return False

    def _left_corridor_clear(self, ctx: PlanningContext) -> bool:
        forward_max = float(getattr(config, "OVERTAKE_SIDE_CLEAR_FORWARD_M", 1.25))
        half_width = float(getattr(config, "OVERTAKE_SIDE_CLEAR_HALF_WIDTH_M", 0.12))
        offset = float(getattr(config, "OVERTAKE_LATERAL_OFFSET_M", 0.16))
        y_min = max(0.0, offset - half_width)
        y_max = offset + half_width
        for obs in ctx.lidar_obstacles:
            if 0.0 <= float(obs.x_m) <= forward_max and y_min <= float(obs.y_m) <= y_max:
                return False
        return True
