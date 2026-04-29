# src/core/interfaces/planner.py
#
# Contratos de planning. Hay dos protocolos:
#
#   IBehaviorPlanner — el componente top-level que produce `BehaviorOutput`
#     por tick. Es la **única fuente de verdad de velocidad** del sistema.
#     Toma pose, route, percepción y tracked_objects; devuelve target_path
#     + speed_profile + stop_required.
#
#   IScenario — un escenario activable (lane_keep, intersection, parking,
#     crosswalk, highway, roundabout). El planner mantiene una lista de
#     scenarios ordenados por prioridad y delega `plan(ctx)` al primero
#     que dice `is_active(ctx) == True`.
#
# Histeresis: cada scenario implementa sus propios thresholds de entrada y
# salida para evitar thrashing entre escenarios cuando la pose oscila en
# el borde.
#
# Tipos de input/output (`PlanningContext`, `BehaviorOutput`,
# `ScenarioName`) viven en `src/core/types/behavior.py`, creado en Fase 4.
# Mientras tanto las firmas usan `object` para no acoplar.

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IScenario(Protocol):
    """Un escenario activable (LaneKeep, Intersection, Crosswalk, etc.)."""

    name: str
    priority: int

    def is_active(self, ctx: object) -> bool:
        """¿Aplica este escenario al contexto actual?

        Implementaciones deben aplicar histeresis interna: usar threshold
        de entrada distinto al de salida para evitar oscilaciones.
        """
        ...

    def plan(self, ctx: object) -> object:
        """Produce un `BehaviorOutput` para este tick.

        Pre: `is_active(ctx)` retornó True.
        """
        ...


@runtime_checkable
class IBehaviorPlanner(Protocol):
    """Top-level planner; única fuente de verdad de velocidad."""

    def plan(self, ctx: object) -> object:
        """Itera scenarios por prioridad, delega al primero activo,
        aplica velocity_overlay (caps por regulatory_elements). Devuelve
        `BehaviorOutput` con target_path + speed_profile + stop_required.
        """
        ...
