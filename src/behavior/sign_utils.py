from __future__ import annotations

from typing import Iterable

from src.behavior.context import PlanningContext


def normalize_sign_kind(value) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def sign_hints_for(
    ctx: PlanningContext,
    kinds: Iterable[str],
    *,
    max_distance_m: float | None = None,
) -> tuple[dict, ...]:
    wanted = {normalize_sign_kind(kind) for kind in kinds}
    out: list[dict] = []
    for hint in tuple(getattr(ctx, "sign_hints", ()) or ()):
        if not isinstance(hint, dict):
            continue
        kind = normalize_sign_kind(hint.get("kind"))
        if kind not in wanted:
            continue
        if max_distance_m is not None:
            distance = hint.get("distance_m")
            if distance is not None:
                try:
                    if float(distance) > float(max_distance_m):
                        continue
                except (TypeError, ValueError):
                    pass
        out.append(dict(hint))
    return tuple(out)


def nearest_sign_hint(
    ctx: PlanningContext,
    kinds: Iterable[str],
    *,
    max_distance_m: float | None = None,
) -> dict | None:
    hints = sign_hints_for(ctx, kinds, max_distance_m=max_distance_m)
    if not hints:
        return None

    def key(hint: dict) -> tuple[float, float]:
        try:
            distance = float(hint.get("distance_m"))
        except (TypeError, ValueError):
            distance = float("inf")
        try:
            ts = float(hint.get("timestamp", 0.0))
        except (TypeError, ValueError):
            ts = 0.0
        return distance, -ts

    return min(hints, key=key)


def sign_hint_key(hint: dict | None, fallback: str) -> str:
    if not isinstance(hint, dict):
        return str(fallback)
    kind = normalize_sign_kind(hint.get("kind")) or str(fallback)
    timestamp = hint.get("timestamp")
    distance = hint.get("distance_m")
    try:
        ts_part = f"{float(timestamp):.2f}"
    except (TypeError, ValueError):
        ts_part = "unknown"
    try:
        dist_part = f"{float(distance):.2f}"
    except (TypeError, ValueError):
        dist_part = "unknown"
    return f"{kind}:{ts_part}:{dist_part}"
