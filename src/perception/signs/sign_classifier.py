# src/perception/signs/sign_classifier.py
#
# Helpers PUROS para clasificar nombres de señales de tráfico. Esta es
# la superficie sobreviviente del antiguo `signActions.py`: lo único
# que realmente usaba el resto del stack era el mapeo Sign → canonical
# name + el predicado "¿es accionable?".
#
# Quién consume este módulo:
#   - `threadLocalPerception._publish_sign(...)` para normalizar el
#     `class` que sale del modelo YOLO antes de mandarlo al gateway.
#   - `ManeuverManager.observe_sign(...)` para decidir si una detección
#     dispara una decisión de maniobra.
#   - Eventualmente, los `IScenario` del BehaviorPlanner que reaccionan
#     a hints de signs (Crosswalk, Highway) cuando se complete la
#     migración Sign → ScenarioHint.
#
# Por qué este módulo existe (y no es parte de `sign_actions.py` viejo):
# en la nueva arquitectura post-Phase 6 ningún componente que NO sea el
# `motor_command_dispatcher` puede escribir motores. Los executors del
# antiguo `SignActions` se desarmaron. Lo que sobrevive es **mapeo y
# clasificación pura**, no ejecución — por eso este archivo es solo
# constantes + funciones libres, sin clases con estado, sin queues, sin
# Events. Es trivialmente testeable y reusable.

from __future__ import annotations

# ----------------------------------------------------------------
# Tablas de mapeo. Se exponen como módulo-level por dos razones:
#   1. son inmutables en runtime — no pertenecen a ningún objeto.
#   2. los tests se las pueden monkeypatch sin instanciar nada.
# ----------------------------------------------------------------

# Nombres alternativos observados en datasets/logs que se pliegan al
# canonical. Si aparece un nombre que no está acá, se devuelve tal cual
# (ya normalizado a snake_case lower).
SIGN_ALIASES: dict[str, str] = {
    "highway_entry": "highway_entrance",
    "highway_entrance": "highway_entrance",
    "highway_exit": "highway_exit",
    "hw_entry": "highway_entrance",
    "hw_entrance": "highway_entrance",
    "hw_exit": "highway_exit",
}

# Set de canonical names que SÍ disparan acciones / decisiones del
# `ManeuverManager`. "parking" se excluye intencionalmente: la state
# machine de parking vive en `threadLineFollowing._poll_sign_detected`,
# no en este path.
ACTIONABLE_SIGNS: frozenset[str] = frozenset({
    "stop",
    "no_entry",
    "crosswalk",
    "red_light",
    "yellow_light",
    "green_light",
    "speed_20",
    "speed_30",
    "highway_entrance",
    "highway_exit",
})

# Agrupación para cooldown de detecciones repetidas. Un cooldown por
# grupo evita que dos signs distintas pero del mismo grupo (ej. dos
# stops detectados muy seguido) re-disparen la maniobra.
ACTION_GROUP: dict[str, str] = {
    "stop": "stop",
    "no_entry": "stop",
    "red_light": "stop",
    "crosswalk": "crosswalk",
    "yellow_light": "traffic_light",
    "green_light": "traffic_light",
    "speed_20": "speed_limit",
    "speed_30": "speed_limit",
    "parking": "parking",
    # Entry/exit separados — un highway_entry reciente NO debe bloquear
    # un highway_exit por compartir grupo.
    "highway_entrance": "highway_entrance",
    "highway_exit": "highway_exit",
}


# ----------------------------------------------------------------
# API pública
# ----------------------------------------------------------------

def normalize_sign_name(sign_name: str | None) -> str:
    """Lleva un nombre crudo de YOLO/dataset a su canonical lower-case.

    Pipeline:
      1. cast a str y trim.
      2. lower-case + dashes/spaces → underscore (snake_case).
      3. lookup en SIGN_ALIASES (devuelve canonical si match, si no el
         valor normalizado).

    Devuelve cadena vacía cuando la entrada es None o vacía.
    """
    normalized = (
        str(sign_name or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return SIGN_ALIASES.get(normalized, normalized)


def is_actionable_sign(sign_name: str | None) -> bool:
    """True si el sign normalizado pertenece al set ACTIONABLE_SIGNS."""
    return normalize_sign_name(sign_name) in ACTIONABLE_SIGNS


def action_group_of(sign_name: str | None) -> str:
    """Devuelve el grupo de cooldown del sign normalizado.

    Si el sign no tiene un grupo declarado, se devuelve el propio
    nombre — efectivamente cada sign desconocido tiene su propio grupo.
    """
    canonical = normalize_sign_name(sign_name)
    return ACTION_GROUP.get(canonical, canonical)
