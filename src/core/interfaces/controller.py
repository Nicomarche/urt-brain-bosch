# src/core/interfaces/controller.py
#
# Contrato del controller de motion. Recibe `BehaviorOutput` (target_path +
# speed_profile) y `PoseEstimate` (estado actual del vehículo); devuelve
# `MotorCommand` (steering + speed para el firmware).
#
# Decisión arquitectónica clave (single source of truth):
#   - El controller es el ÚNICO componente que decide steering en runtime
#     de competencia. No hay Stanley, no hay PID directo, no hay overrides
#     desde signActions sobre la dirección.
#   - El controller NO decide speed: la velocidad sale del BehaviorPlanner
#     (que produce speed_profile) y el controller solo la ejecuta.
#
# Implementación en Fase 6: `MotionController` con Acados MPC acoplado
# (steering + velocity en un único QP no-lineal). Si Acados no compila en
# Jetson Nano, plan B es un MPC scipy con la misma interfaz — el contrato
# no cambia.
#
# El tipo `MotorCommand` lo define `src/core/types/control.py` en Fase 6;
# por ahora la firma usa el legacy `ControlDecision` mientras la transición
# está en curso. Cuando MotorCommand exista, el Protocol se actualiza.

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.types.control import ControlDecision
from src.core.types.pose import PoseEstimate


@runtime_checkable
class IMotionController(Protocol):
    """Convierte la decisión del planner en comandos para el actuador."""

    def compute(self, behavior_output: object, pose: PoseEstimate) -> ControlDecision:
        """Resuelve un step del MPC dado el plan y la pose actual.

        `behavior_output` es `BehaviorOutput` (definido en Fase 4); aquí
        está como `object` para evitar acoplamiento durante la transición.
        Retorno tipo `ControlDecision` será reemplazado por `MotorCommand`
        cuando exista.

        Pre: `pose.timestamp` reciente (< 200 ms). Si stale, el llamador
        debe usar el SafetyGate, no este método.
        """
        ...

    def reset(self) -> None:
        """Limpia el estado interno del solver (warmstart, slacks)."""
        ...
