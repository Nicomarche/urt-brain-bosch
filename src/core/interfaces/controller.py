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
# Pre/postcondiciones:
#   - Pre: `pose.timestamp` reciente (< 200 ms). Si stale, el llamador
#     debe usar el `safety_gate`, NO este método. El controller asume
#     que la pose es buena.
#   - Pre: `behavior_output.valid == True`. Si el planner emitió un plan
#     inválido, el controller no inventa nada — devuelve `MotorCommand`
#     con `valid=False` y razón "behavior_invalid".
#   - Post: `MotorCommand.valid == True` ⇒ steering ∈ [-25°, +25°] y
#     speed >= 0 (el clamp ocurre adentro del compute()).

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.types.behavior import BehaviorOutput
from src.core.types.control import MotorCommand
from src.core.types.pose import PoseEstimate


@runtime_checkable
class IMotionController(Protocol):
    """Convierte la decisión del planner en comandos para el actuador."""

    def compute(
        self, behavior_output: BehaviorOutput, pose: PoseEstimate
    ) -> MotorCommand:
        """Resuelve un step del MPC dado el plan y la pose actual.

        Si `behavior_output.valid == False` o `behavior_output.stop_required
        == True`, debe devolver `MotorCommand(valid=False, ...)` con la
        `reason` correspondiente. NO inventar comandos.
        """
        ...

    def reset(self) -> None:
        """Limpia el estado interno del solver (warmstart, slacks)."""
        ...
