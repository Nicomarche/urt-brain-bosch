# src/behavior/velocity_overlay.py
#
# Función pura: dado un `BehaviorOutput` ya producido por un scenario y
# el `RouteContext` con `regulatory_ahead`, aplica caps de velocidad y
# decide si frenar.
#
# Diseño:
#   - El scenario produce su `speed_profile` "de buena fe" (lo que
#     querría hacer si no hubiera regulators). El overlay se encarga
#     de hacer cumplir las reglas regulatorias EN UN SOLO LUGAR.
#   - Esto evita que cada scenario tenga que duplicar la lógica de
#     "ah, hay un crosswalk, mejor bajo velocidad". El scenario
#     concentra en SU comportamiento; el overlay concentra en regulators.
#
# Reglas implementadas (en orden de aplicación):
#
#   1. CAP global: speed_profile[i] = min(speed_profile[i], max_speed_mps).
#   2. SLOWDOWN por crosswalk: si crosswalk dentro de
#      `_CROSSWALK_SLOWDOWN_RANGE_M`, capear speed a `_CROSSWALK_SPEED_MPS`.
#   3. CAP por intersection: dentro de un intersection, capear a
#      `_INTERSECTION_SPEED_MPS`.
#
# Todas las reglas son monotónicamente DECRECIENTES sobre speed: nunca
# subimos velocidad. Esto preserva la invariante "el overlay nunca
# acelera más allá de lo que el scenario decidió".
#
# El overlay devuelve un `BehaviorOutput` NUEVO (frozen ⇒ replace),
# preservando target_path y scenario_name del original.

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from src.core.types.behavior import BehaviorOutput

if TYPE_CHECKING:
    from src.core.types.routing import RegulatoryElement, RouteContext


# Rangos en metros desde la posición actual del ego — distancias
# relativas al inicio de la lanelet en la que está el ego (proxy
# conservador, ver `LaneletMap.regulatory_within`).
_CROSSWALK_SLOWDOWN_RANGE_M = 4.0
try:
    from config import TRAFFIC_SIGN_LOW_SPEED_MPS as _CROSSWALK_SPEED_MPS
except Exception:
    _CROSSWALK_SPEED_MPS = 0.10
_INTERSECTION_SPEED_MPS = 0.40
_INTERSECTION_RANGE_M = 6.0


def apply_overlay(
    plan: BehaviorOutput,
    route: "RouteContext",
    max_speed_mps: float,
) -> BehaviorOutput:
    """Aplica caps de velocidad por regulatory elements al `plan`.

    Devuelve un BehaviorOutput nuevo (con `speed_profile` posiblemente
    capeado y `stop_required` levantado). Si no hay regulators que apliquen,
    devuelve el plan original sin cambios materiales (sólo el cap global).
    """
    speed = np.array(plan.speed_profile, dtype=float, copy=True)
    if speed.size == 0:
        return plan

    # 1. Cap global.
    speed = np.minimum(speed, float(max_speed_mps))

    stop_required = bool(plan.stop_required)
    notes_extra: dict = dict(plan.notes)
    notes_extra.setdefault("overlay_caps", [])

    # Asumimos que `route.regulatory_ahead` está ordenado por distancia
    # creciente (ver LaneletMap.regulatory_within).
    for reg in route.regulatory_ahead:
        kind = reg.kind.lower()
        # Distancia: usamos el campo data["distance_m"] si está; fallback
        # a una distancia conservadora (el regulator está "cerca").
        distance_m = float(reg.data.get("distance_m", 0.0)) if reg.data else 0.0

        if kind == "crosswalk" and distance_m <= _CROSSWALK_SLOWDOWN_RANGE_M:
            speed = np.minimum(speed, _CROSSWALK_SPEED_MPS)
            notes_extra["overlay_caps"].append(
                {"kind": "crosswalk", "distance_m": distance_m, "cap": _CROSSWALK_SPEED_MPS}
            )
        elif kind == "intersection" and distance_m <= _INTERSECTION_RANGE_M:
            speed = np.minimum(speed, _INTERSECTION_SPEED_MPS)
            notes_extra["overlay_caps"].append(
                {"kind": "intersection", "distance_m": distance_m, "cap": _INTERSECTION_SPEED_MPS}
            )
        elif kind == "speed_limit":
            limit = float(reg.data.get("speed_mps", max_speed_mps))
            speed = np.minimum(speed, limit)
            notes_extra["overlay_caps"].append({"kind": "speed_limit", "cap": limit})

    return replace(
        plan,
        speed_profile=speed,
        stop_required=stop_required,
        notes=notes_extra,
    )
