"""Helper para resolver el perfil de corrección visual activo.

Plan B2: la autoridad de la visión sobre la pose depende del scenario
activo (lane_keep, intersection, parking, ...). El BehaviorPlanner
publica el ``scenario_name`` activo via ``BEHAVIOR_OUTPUT_MSG``; el
pose_estimator suscribe a eso y mantiene ``current_scenario`` en
``TrackingState``. Cuando aplica corrección lateral, lee el perfil con
``resolve_visual_correction_profile(scenario_name)``.

Fuentes (en orden de prioridad):
  1. ``config.VISUAL_CORRECTION_PROFILES[scenario_name]``
  2. ``runtime.yaml visual_correction_profiles.<name>`` (vía RuntimeParams)
  3. Fallback estándar = perfil "default" hardcodeado.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualCorrectionProfile:
    """Pesos de la corrección visual lateral para un scenario dado."""

    gain: float = 0.18
    max_m: float = 0.02
    step_max_m: float = 0.015

    @property
    def is_disabled(self) -> bool:
        return self.gain <= 1e-9 and self.max_m <= 1e-9


_DEFAULT = VisualCorrectionProfile(gain=0.18, max_m=0.02, step_max_m=0.015)


def resolve_visual_correction_profile(scenario_name: str | None) -> VisualCorrectionProfile:
    """Devuelve el perfil para el scenario dado, con fallback robusto.

    Args:
        scenario_name: ``lane_keep`` / ``parking`` / ``intersection`` / ...
            ``None`` o desconocido → perfil "default".
    """
    name = (scenario_name or "default").strip().lower()
    # 1) Lectura desde config.py
    try:
        import config as _cfg
        profiles = getattr(_cfg, "VISUAL_CORRECTION_PROFILES", None)
        if isinstance(profiles, dict):
            data = profiles.get(name) or profiles.get("default")
            if isinstance(data, dict):
                return _profile_from_dict(data)
    except Exception:
        pass
    # 2) Lectura desde runtime.yaml
    try:
        from src.core.runtime_params import params as _rt_params
        data = _rt_params.get(f"visual_correction_profiles.{name}", None)
        if isinstance(data, dict):
            return _profile_from_dict(data)
        data = _rt_params.get("visual_correction_profiles.default", None)
        if isinstance(data, dict):
            return _profile_from_dict(data)
    except Exception:
        pass
    return _DEFAULT


def _profile_from_dict(data: dict) -> VisualCorrectionProfile:
    try:
        return VisualCorrectionProfile(
            gain=float(data.get("gain", _DEFAULT.gain)),
            max_m=float(data.get("max_m", _DEFAULT.max_m)),
            step_max_m=float(data.get("step_max_m", _DEFAULT.step_max_m)),
        )
    except (TypeError, ValueError):
        return _DEFAULT


__all__ = [
    "VisualCorrectionProfile",
    "resolve_visual_correction_profile",
]
