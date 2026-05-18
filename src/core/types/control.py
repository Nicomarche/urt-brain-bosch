# src/core/types/control.py
#
# Tipos de la capa de control. Dos tipos viven acá:
#
#   1. `MotorCommand` (Fase 6, vigente) — salida del `MotionController`. Es
#      el ÚNICO contrato que llega al `motor_command_dispatcher`, que a su
#      vez es el ÚNICO lugar que escribe `SpeedMotor`/`SteerMotor` al
#      firmware Nucleo. Single source of truth.
#
#   2. `ControlDecision` (legacy, pre-Fase 6) — la salida histórica del
#      `threadControlCoordinator` que combinaba candidates (lane follower,
#      sign override, manual). Se mantiene únicamente porque el dashboard
#      todavía lo serializa para visualización; ningún path de motor
#      depende de él. Se borrará cuando el dashboard migre a `MotorCommand`.
#
# Convenciones:
#   - `steering_deg`: ángulo de las RUEDAS, no del volante. Rango [-25°, +25°].
#     Positivo = giro a la izquierda (convención ISO 8855).
#   - `speed_mps`: metros por segundo, positivo hacia adelante. El
#     dispatcher convierte a cm/s antes de mandarlo al firmware (legacy
#     Nucleo opera en cm/s).
#   - `timestamp`: segundos UNIX (`time.time()`), no monotonic. Permite al
#     `safety_gate` calcular staleness contra `time.time()`.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.messaging.header import MessageHeader


@dataclass(frozen=True)
class MotorCommand:
    """Comando final hacia el firmware del Nucleo. Vigente desde Fase 6.

    Invariantes:
      - `steering_deg ∈ [-25, +25]`. El `MotionController` ya aplica el clamp;
        el dispatcher confía en el rango y no re-clamps.
      - `speed_mps >= 0` en operación normal. La única reversa permitida es
        una maniobra de parking emitida por el BehaviorPlanner/MotionController.
      - `valid == True` ⇔ los valores son utilizables. Si False, el
        dispatcher debe ignorar el comando y llamar a `safety_gate.fallback()`.
      - `source` documenta de dónde viene el comando — "motion_controller"
        en operación nominal, "safety_gate" cuando el watchdog lo emite.
        El dashboard lo usa para mostrar al operador qué subsistema manda.
      - `reason` describe la causa cuando `source != "motion_controller"`
        (ej. "stale_behavior_output", "mpc_solver_failure", "behavior_invalid").
        Vacío en el camino feliz para no inflar logs.

    Diseño: frozen para garantizar inmutabilidad cross-thread (el comando
    cruza de `motion_controller` a `motor_command_dispatcher` por buffer).
    """

    timestamp: float = 0.0
    steering_deg: float = 0.0
    speed_mps: float = 0.0
    valid: bool = False
    source: str = "none"
    reason: str = ""
    debug: dict[str, Any] = field(default_factory=dict)
    # Plan TANDA 2.1: Header opcional (default None mantiene retro-compat).
    header: MessageHeader | None = None


@dataclass(frozen=True)
class ControlDecision:
    """LEGACY (pre-Fase 6). Se mantiene para compatibilidad con dashboard.

    NO se usa en el path crítico de motor desde Fase 6. Cualquier
    componente nuevo debe consumir `MotorCommand`. Cuando dashboard migre,
    este tipo se elimina.
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
