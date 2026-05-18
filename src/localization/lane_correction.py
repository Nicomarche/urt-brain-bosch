"""Función pura `compute_lateral_correction`.

Plan A1.3: extracción del bloque ``_apply_lane_observation`` del
``pose_estimator_thread`` a un módulo testeable sin instanciar el thread.

API mínima — recibe medición + estado + config, devuelve la corrección a
aplicar (en metros) clampeada por max/step. Sin side effects, sin
``threading.Lock``, sin queues. La parte stateful (último timestamp de
corrección, accumulador) vive en el caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.localization.visual_correction_profile import VisualCorrectionProfile


@dataclass(frozen=True)
class LateralCorrectionResult:
    """Resultado de un único cómputo de corrección lateral."""

    correction_m: float                # corrección a aplicar este tick
    applied: bool                      # True si pasó los gates
    reason: str                        # razón humanlegible si applied=False
    raw_correction_m: float = 0.0      # corrección antes del clamp


def compute_lateral_correction(
    *,
    lateral_error_m: float,
    speed_mps: float,
    now_s: float,
    last_correction_time_s: float,
    profile: VisualCorrectionProfile,
    min_speed_mps: float = 0.02,
) -> LateralCorrectionResult:
    """Calcula la corrección lateral aplicable este tick.

    Args:
        lateral_error_m: Error visual medido en metros (positivo = ego está
            a la derecha de la lane center → corrección a la izquierda).
        speed_mps: Velocidad lineal del ego (m/s). Si < ``min_speed_mps``
            la corrección se inhibe (no confiar en visión a baja vel).
        now_s: Tiempo monotónico del tick actual.
        last_correction_time_s: Tiempo del último tick que aplicó corrección.
        profile: ``VisualCorrectionProfile`` con gain/max_m/step_max_m
            del scenario activo.
        min_speed_mps: Umbral de velocidad mínima para confiar en visión.

    Returns:
        ``LateralCorrectionResult`` con la corrección clampeada y razón.
    """
    if profile.is_disabled:
        return LateralCorrectionResult(
            correction_m=0.0,
            applied=False,
            reason="profile_disabled",
        )
    if abs(speed_mps) < min_speed_mps:
        return LateralCorrectionResult(
            correction_m=0.0,
            applied=False,
            reason="speed_below_min",
        )
    # Cooldown: si el último tick aplicado fue muy reciente, esperar.
    # Esto evita amplificar ruido del lane perception entre ticks juntos.
    # El cooldown lo gestiona el caller (no acá), pero damos el campo.

    raw_correction = float(lateral_error_m) * float(profile.gain)
    abs_max = float(profile.max_m)
    if abs(raw_correction) > abs_max:
        clamped = abs_max if raw_correction >= 0 else -abs_max
    else:
        clamped = raw_correction
    step_max = float(profile.step_max_m)
    if step_max > 0.0 and abs(clamped) > step_max:
        clamped = step_max if clamped >= 0 else -step_max
    return LateralCorrectionResult(
        correction_m=float(clamped),
        applied=True,
        reason="applied",
        raw_correction_m=float(raw_correction),
    )


__all__ = ["compute_lateral_correction", "LateralCorrectionResult"]
