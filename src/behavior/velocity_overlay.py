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
#   3. STOP por stopline: si stopline dentro de
#      `_STOPLINE_STOP_RANGE_M`, RAMPa lineal a 0 desde la velocidad
#      actual hasta el stopline. `stop_required = True`.
#   4. CAP por intersection: dentro de un intersection, capear a
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
_STOPLINE_STOP_RANGE_M = 5.0
_CROSSWALK_SLOWDOWN_RANGE_M = 4.0
_CROSSWALK_SPEED_MPS = 0.30
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

        if kind == "stopline" and distance_m <= _STOPLINE_STOP_RANGE_M:
            speed = _ramp_to_zero(speed, dt=plan.dt, distance_to_stop_m=distance_m)
            stop_required = True
            notes_extra["overlay_caps"].append({"kind": "stopline", "distance_m": distance_m})
        elif kind == "crosswalk" and distance_m <= _CROSSWALK_SLOWDOWN_RANGE_M:
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


def _ramp_to_zero(speed: np.ndarray, dt: float, distance_to_stop_m: float) -> np.ndarray:
    """Genera un perfil de frenada lineal hasta 0 a la distancia indicada.

    Aproximación:
      - Calcular cuánto step_arc avanzaría el plan original a su velocidad.
      - Identificar el step k* en el cual cumulamos `distance_to_stop_m`.
      - Para steps <= k*: rampa lineal de v0 → 0 que termina exactamente en 0
        en el step k*.
      - Para steps > k*: 0.
      - Si el horizonte es más corto que la distancia (k* >= n), la rampa
        cubre todo el horizonte terminando en 0 en el último step. Esto
        garantiza que `speed_profile[-1] == 0` siempre que se aplique el ramp.
    Si `distance_to_stop_m <= 0`, devolvemos ceros completos.
    """
    n = speed.shape[0]
    if distance_to_stop_m <= 0.0 or n == 0:
        return np.zeros_like(speed)

    cumulative_arc = 0.0
    out = speed.copy()
    stop_step = n
    for k in range(n):
        cumulative_arc += float(out[k]) * float(dt)
        if cumulative_arc >= distance_to_stop_m:
            stop_step = k
            break

    # ramp_end es el índice donde queremos llegar a 0. Si el horizonte
    # es más corto que la distancia requerida, recortamos al último step.
    ramp_end = min(stop_step, n - 1)
    if ramp_end <= 0:
        return np.zeros_like(speed)

    v0 = float(out[0])
    # Rampa lineal: ramp_end+1 puntos desde v0 hasta 0 (inclusive).
    ramp = np.linspace(v0, 0.0, ramp_end + 1)
    out[: ramp_end + 1] = np.minimum(out[: ramp_end + 1], ramp)
    out[ramp_end + 1 :] = 0.0
    return out
