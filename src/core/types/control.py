# src/core/types/control.py
#
# Tipos de la capa de control. ControlDecision es la salida histórica del
# `threadControlCoordinator` (legacy) que combinaba múltiples candidates
# (lane follower, sign override, manual). En Fase 6 esto colapsa en
# `MotorCommand` emitido por el `MotionController` Acados — el coordinator
# pasa a ser puro dispatch (single source of truth de motor).
#
# `MotorCommand` queda como TODO de Fase 6 — todavía no se define para
# evitar confusión sobre cuál es el contrato vigente. Cuando exista,
# absorbe `steering_deg` + `speed_cmd` con semántica clara y reemplaza
# todas las escrituras al SerialWriter desde fuera de
# `motor_command_dispatcher.py`.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ControlDecision:
    """LEGACY (pre-Fase 6). Decisión final que llega al SerialWriter.

    Hoy lo produce `threadControlCoordinator.py` después de elegir entre
    varios candidates. La elección se basa en `authority` y `safety_override`.
    En Fase 6 el coordinator se vacía: solo hay un candidate (MotorCommand)
    y la decisión es trivial.

    Conventions:
      - `steering_deg`: ángulo de las RUEDAS, no del volante. Rango ±25°.
      - `speed_cmd`: cm/s para el firmware Nucleo (no m/s).
      - `authority`: "lane_follower", "manual", "sign_override", "safety", "none".
      - `safety_override`: True si llegó desde safety_gate (failsafe).
      - `*_sent`: confirmación post-envío al serial (no de pre-envío).
    """

    timestamp: float = 0.0
    steering_deg: float | None = None
    speed_cmd: float | None = None
    authority: str = "none"
    safety_override: bool = False
    reason: str = "none"
    speed_sent: bool = False
    steer_sent: bool = False
    source_candidate: str = "none"
    debug: dict[str, Any] = field(default_factory=dict)
