# src/core/types/behavior.py
#
# Tipos de salida del `BehaviorPlanner`. Es la **única fuente de verdad
# de velocidad** del sistema — el `MotionController` (MPC) consume esto
# y produce el `MotorCommand`. No hay otro lugar que decida velocidad.
#
# Diseño:
#   - `BehaviorOutput` es **el contrato** entre planner y controller.
#     Su forma calza 1:1 con la API esperada por Acados MPC:
#       target_path (N+1, 3)  ↔ state_refs en el MPC
#       speed_profile (N,)    ↔ input_refs[:, 0] (velocidad ref)
#     No se inserta glue: el thread del MPC lee `BehaviorOutput`
#     directamente.
#
#   - `ScenarioName` es un Enum cerrado para que el dashboard pueda
#     mostrar el escenario activo de manera estable y los tests puedan
#     validar transiciones sin strings mágicos.
#
#   - `BehaviorOutput.valid = False` significa "el planner no pudo
#     producir un plan en este tick" — el `MotionController` debe
#     llamar a `safety_gate.fallback()` (parar). Esto es preferible a
#     emitir un plan basura.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ScenarioName(str, Enum):
    """Nombres canónicos de escenarios — string-enum para serialización fácil.

    Heredar de `str` permite que `ScenarioName.LANE_KEEP == "lane_keep"`
    sin conversiones, lo que simplifica logging y mensajes JSON al
    dashboard. Los escenarios concretos viven en
    `src/behavior/scenarios/{lane_keep,intersection,...}.py`.
    """

    LANE_KEEP = "lane_keep"
    INTERSECTION = "intersection"
    PARKING = "parking"
    CROSSWALK = "crosswalk"
    HIGHWAY = "highway"
    ROUNDABOUT = "roundabout"
    OVERTAKE = "overtake"
    # FALLBACK no es un scenario propiamente dicho — lo usa el planner
    # cuando ningún scenario está activo, para emitir un BehaviorOutput
    # con stop_required=True. Mantener acá evita strings mágicos.
    FALLBACK = "fallback"


class TurnSignalCommand(str, Enum):
    """Comando de guiño estilo Autoware.

    Por ahora el hardware BFMC no consume esto directamente, pero lo dejamos
    en el contrato interno del planner para que el dashboard y futuros módulos
    de control puedan razonar sobre la intención lateral del auto.
    """

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    HAZARD = "hazard"


@dataclass(frozen=True)
class BehaviorPathPlan:
    """Salida intermedia estilo Autoware entre scene module y controller.

    Este tipo separa tres etapas:
      1. `PlannerManager` elige el scene module activo.
      2. El scene module produce un *path plan* crudo (`raw_path` +
         `base_speed_profile` + intención lateral).
      3. `PathOptimizer` y `BehaviorVelocityPlanner` lo convierten al
         `BehaviorOutput` final que consume el MPC.

    Mantener esta capa intermedia nos permite acercarnos a la arquitectura de
    Autoware sin romper el contrato público actual del repo.
    """

    timestamp: float = 0.0
    raw_path: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    base_speed_profile: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scenario_name: str = ScenarioName.FALLBACK.value
    valid: bool = False
    stop_required: bool = False
    turn_signal: str = TurnSignalCommand.NONE.value
    drivable_left_bound: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    drivable_right_bound: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BehaviorOutput:
    """Salida del `BehaviorPlanner`. Consumida por `MotionController`.

    Invariantes:
      - `target_path.shape == (N+1, 3)`. Columnas: x, y, psi (rad).
        Estado de referencia al inicio (k=0) hasta horizonte (k=N).
        Coordenadas en el frame plano del mapa activo (Lanelet2 OSM proyectado).
      - `speed_profile.shape == (N,)`. Velocidad de referencia para los
        N pasos de control. Unidades m/s. Valores >= 0; el planner de
        velocidad aplica caps y ramps regulatorios antes de llegar al MPC.
      - `dt`: duración de cada step (s). Debe coincidir con el dt del MPC.
      - `valid == True` ⇔ el planner pudo computar un plan utilizable. El
        controller debe verificar este flag antes de consumir `target_path`.
        Si False, el `safety_gate` toma el control (parar).
      - `stop_required == True` ⇔ scenario o velocity_overlay decidió
        parar (stopline, peatón, bloqueo). Distinto de `!valid`: acá el
        plan es válido, sólo que speed_profile va a cero.
      - `scenario_name` es el scene module que originó el plan. El dashboard
        lo muestra para diagnóstico.
      - `notes` es un dict libre para datos auxiliares (regulatory hits,
        razón de freno, etc.). NO consumir programáticamente — sólo
        diagnóstico.

    Diseño: frozen para inmutabilidad cross-thread. Si necesitás
    derivar una variante (ej. después de aplicar overlay), usá
    `dataclasses.replace`. Ese es el patrón usado por `velocity_overlay`.
    """

    timestamp: float = 0.0
    dt: float = 0.05  # default 20 Hz; el MPC operará a esta tasa
    target_path: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    speed_profile: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scenario_name: str = ScenarioName.FALLBACK.value
    valid: bool = False
    stop_required: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def horizon_n(self) -> int:
        """N = número de steps del MPC. Derivado de speed_profile.shape."""
        return int(self.speed_profile.shape[0])
