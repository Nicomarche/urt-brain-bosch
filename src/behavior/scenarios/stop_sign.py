from __future__ import annotations

import numpy as np

from dataclasses import replace

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.behavior.sign_utils import normalize_sign_kind, sign_hints_for
from src.core.types.behavior import BehaviorOutput, ScenarioName

try:
    import config as _cfg
except Exception:  # pragma: no cover
    _cfg = None


_STOP_APPROACH_RANGE_M = 5.0
_STOP_COMMAND_DISTANCE_M = 0.40
_STOP_CAMERA_FALLBACK_DISTANCE_M = 0.40
_STOP_CAMERA_FALLBACK_MIN_CONFIDENCE = 0.65
_STOP_CAMERA_FALLBACK_MIN_BOX_AREA = 0.010
_STOP_RETRIGGER_SUPPRESS_S = 5.0
_STOPPED_SPEED_MPS = 0.025


class StopSign(BaseScenario):
    """STOP sign policy: use lidar distance, with a close visual fallback."""

    name = ScenarioName.STOP.value
    priority = 90

    def __init__(self) -> None:
        self._active_key: str | None = None
        self._hold_until_s: float = 0.0
        self._consumed_keys: set[str] = set()
        self._last_candidate: dict | None = None
        self._committed_to_stop: bool = False
        self._suppress_until_s: float = 0.0

    def is_active(self, ctx: PlanningContext) -> bool:
        now = float(ctx.now_s)
        if self._hold_until_s > 0.0 and now >= self._hold_until_s:
            if self._active_key:
                self._consumed_keys.add(self._active_key)
            self._suppress_until_s = max(
                self._suppress_until_s,
                now + _stop_retrigger_suppress_s(),
            )
            self._active_key = None
            self._hold_until_s = 0.0
            self._last_candidate = None
            self._committed_to_stop = False
            return False

        if self._active_key is not None and (
            self._committed_to_stop
            or (self._hold_until_s > now)
        ):
            return True

        if self._suppress_until_s > now:
            return False

        candidate = self._candidate(ctx)
        if candidate is not None:
            key = str(candidate["key"])
            if key not in self._consumed_keys:
                self._active_key = key
                self._last_candidate = candidate
                distance_m = _candidate_distance(candidate)
                if distance_m is not None and distance_m <= _stop_command_distance_m():
                    self._committed_to_stop = True
                return True

        return bool(
            self._active_key is not None
            and (
                self._committed_to_stop
                or (self._hold_until_s > now)
            )
        )

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        now = float(ctx.now_s)
        if self._active_key is not None and (
            self._committed_to_stop
            or (self._hold_until_s > now)
        ):
            candidate = self._last_candidate or {}
        else:
            candidate = self._candidate(ctx) or self._last_candidate or {}
        key = str(candidate.get("key") or self._active_key or "stop")
        self._active_key = key
        self._last_candidate = dict(candidate) if candidate else self._last_candidate

        notes = {
            "reason": "stop_policy",
            "stop_policy_managed": True,
            "stop_key": key,
            "stop_source": str(candidate.get("source", "unknown")),
            "distance_source": str(candidate.get("distance_source", "unknown")),
        }

        distance_m = _candidate_distance(candidate)
        command_distance_m = _stop_command_distance_m()
        if distance_m is not None and distance_m <= command_distance_m:
            self._committed_to_stop = True

        ego_stopped = abs(float(ctx.pose.speed_mps or 0.0)) <= _STOPPED_SPEED_MPS
        if self._committed_to_stop or distance_m is None:
            if ego_stopped and self._hold_until_s <= float(ctx.now_s):
                hold_s = float(getattr(_cfg, "TRAFFIC_SIGN_STOP_HOLD_S", 3.0)) if _cfg is not None else 3.0
                self._hold_until_s = float(ctx.now_s) + hold_s
            notes["stop_mode"] = "hold" if self._hold_until_s > float(ctx.now_s) else "stopping"
            notes["command_distance_m"] = float(command_distance_m)
            if distance_m is not None:
                notes["distance_m"] = float(distance_m)
            if self._hold_until_s > float(ctx.now_s):
                notes["hold_until_s"] = float(self._hold_until_s)
            plan = self._build_constant_speed_plan(
                ctx=ctx,
                target_speed_mps=0.0,
                scenario_name=self.name,
                notes=notes,
            )
            return replace(plan, stop_required=True)

        plan = self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=max(0.0, float(ctx.nominal_speed_mps)),
            scenario_name=self.name,
            notes={**notes, "stop_mode": "approach", "distance_m": float(distance_m)},
        )
        if not plan.valid:
            return plan
        speed = _ramp_to_zero(
            np.asarray(plan.speed_profile, dtype=float),
            dt=float(ctx.dt),
            distance_to_stop_m=float(distance_m),
        )
        return replace(plan, speed_profile=speed, stop_required=False)

    def _candidate(self, ctx: PlanningContext) -> dict | None:
        sign = _nearest_lidar_stop_sign(ctx)
        if sign is not None:
            distance = sign.get("distance_m")
            return {
                "key": _stable_stop_sign_key(ctx, sign),
                "source": "sign_hint_lidar",
                "distance_source": "lidar",
                "distance_m": distance,
            }
        sign = _nearest_camera_fallback_stop_sign(ctx)
        if sign is not None:
            distance = sign.get("distance_m")
            return {
                "key": _stable_stop_sign_key(ctx, sign),
                "source": "sign_hint_camera_fallback",
                "distance_source": "camera",
                "distance_m": distance,
            }
        return None


def _nearest_lidar_stop_sign(ctx: PlanningContext) -> dict | None:
    candidates: list[dict] = []
    for hint in sign_hints_for(ctx, {"stop"}, max_distance_m=_STOP_APPROACH_RANGE_M):
        if str(hint.get("distance_source", "")).strip().lower() != "lidar":
            continue
        distance = _hint_distance_m(hint)
        if distance is None or distance > _STOP_APPROACH_RANGE_M:
            continue
        candidates.append({**hint, "distance_m": distance})
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item["distance_m"]))


def _nearest_camera_fallback_stop_sign(ctx: PlanningContext) -> dict | None:
    candidates: list[dict] = []
    max_distance = _stop_camera_fallback_distance_m()
    min_confidence = _stop_camera_fallback_min_confidence()
    min_box_area = _stop_camera_fallback_min_box_area()
    for hint in sign_hints_for(ctx, {"stop"}, max_distance_m=max_distance):
        if str(hint.get("distance_source", "")).strip().lower() != "camera":
            continue
        distance = _hint_distance_m(hint)
        if distance is None or distance > max_distance:
            continue
        if _hint_float(hint, "confidence", default=0.0) < min_confidence:
            continue
        if _hint_float(hint, "box_area", default=0.0) < min_box_area:
            continue
        candidates.append({**hint, "distance_m": distance})
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item["distance_m"]))


def _stable_stop_sign_key(ctx: PlanningContext, sign: dict) -> str:
    lanelet_id = str(ctx.route.current_lanelet_id or "unknown_lanelet")
    raw = normalize_sign_kind(sign.get("raw_class") or sign.get("kind") or "stop")
    return f"stop_sign:{lanelet_id}:{raw or 'stop'}"


def _candidate_distance(candidate: dict | None) -> float | None:
    if not isinstance(candidate, dict):
        return None
    value = candidate.get("distance_m")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _stop_command_distance_m() -> float:
    if _cfg is None:
        return _STOP_COMMAND_DISTANCE_M
    try:
        return max(0.05, float(getattr(_cfg, "TRAFFIC_SIGN_STOP_COMMAND_DISTANCE_M", _STOP_COMMAND_DISTANCE_M)))
    except (TypeError, ValueError):
        return _STOP_COMMAND_DISTANCE_M


def _stop_camera_fallback_distance_m() -> float:
    if _cfg is None:
        return _STOP_CAMERA_FALLBACK_DISTANCE_M
    try:
        return max(
            0.05,
            float(
                getattr(
                    _cfg,
                    "TRAFFIC_SIGN_STOP_CAMERA_FALLBACK_DISTANCE_M",
                    _STOP_CAMERA_FALLBACK_DISTANCE_M,
                )
            ),
        )
    except (TypeError, ValueError):
        return _STOP_CAMERA_FALLBACK_DISTANCE_M


def _stop_camera_fallback_min_confidence() -> float:
    if _cfg is None:
        return _STOP_CAMERA_FALLBACK_MIN_CONFIDENCE
    try:
        return max(
            0.0,
            float(
                getattr(
                    _cfg,
                    "TRAFFIC_SIGN_STOP_CAMERA_FALLBACK_MIN_CONFIDENCE",
                    _STOP_CAMERA_FALLBACK_MIN_CONFIDENCE,
                )
            ),
        )
    except (TypeError, ValueError):
        return _STOP_CAMERA_FALLBACK_MIN_CONFIDENCE


def _stop_camera_fallback_min_box_area() -> float:
    if _cfg is None:
        return _STOP_CAMERA_FALLBACK_MIN_BOX_AREA
    try:
        return max(
            0.0,
            float(
                getattr(
                    _cfg,
                    "TRAFFIC_SIGN_STOP_CAMERA_FALLBACK_MIN_BOX_AREA",
                    _STOP_CAMERA_FALLBACK_MIN_BOX_AREA,
                )
            ),
        )
    except (TypeError, ValueError):
        return _STOP_CAMERA_FALLBACK_MIN_BOX_AREA


def _stop_retrigger_suppress_s() -> float:
    if _cfg is None:
        return _STOP_RETRIGGER_SUPPRESS_S
    try:
        return max(
            0.0,
            float(getattr(_cfg, "TRAFFIC_SIGN_STOP_RETRIGGER_SUPPRESS_S", _STOP_RETRIGGER_SUPPRESS_S)),
        )
    except (TypeError, ValueError):
        return _STOP_RETRIGGER_SUPPRESS_S


def _hint_distance_m(hint: dict | None) -> float | None:
    if not isinstance(hint, dict):
        return None
    value = hint.get("distance_m")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _hint_float(hint: dict | None, key: str, *, default: float) -> float:
    if not isinstance(hint, dict):
        return float(default)
    value = hint.get(key)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _ramp_to_zero(speed: np.ndarray, *, dt: float, distance_to_stop_m: float) -> np.ndarray:
    n = int(speed.shape[0])
    if n == 0 or distance_to_stop_m <= 0.0:
        return np.zeros_like(speed)
    out = np.array(speed, copy=True, dtype=float)
    cumulative = 0.0
    stop_step = n - 1
    for idx in range(n):
        cumulative += max(0.0, float(out[idx])) * float(dt)
        if cumulative >= float(distance_to_stop_m):
            stop_step = idx
            break
    ramp_end = max(0, min(stop_step, n - 1))
    ramp = np.linspace(max(0.0, float(out[0])), 0.0, ramp_end + 1)
    out[: ramp_end + 1] = np.minimum(out[: ramp_end + 1], ramp)
    out[ramp_end + 1 :] = 0.0
    return out
