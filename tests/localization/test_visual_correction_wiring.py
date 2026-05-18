"""Tests del cableado VISUAL_CORRECTION_PROFILES → TrackingState → pose_estimator.

Plan B3.2: el dict ``VISUAL_CORRECTION_PROFILES`` de ``config.py`` se resuelve
en runtime via ``resolve_visual_correction_profile(scenario_name)``. El
BehaviorPlanner escribe el scenario activo en ``TrackingState.set_current_scenario``;
el pose_estimator lo lee al aplicar la corrección lateral.

Estos tests verifican el cableado punta-a-punta sin instanciar threads.
"""

from __future__ import annotations

import pytest

from src.localization.relocalization_thread import TrackingState
from src.localization.visual_correction_profile import (
    resolve_visual_correction_profile,
    VisualCorrectionProfile,
)


def test_tracking_state_default_scenario_is_none():
    state = TrackingState()
    assert state.current_scenario is None


def test_tracking_state_set_and_read_scenario():
    state = TrackingState()
    state.set_current_scenario("lane_keep")
    assert state.current_scenario == "lane_keep"


def test_tracking_state_set_scenario_normalizes_case_and_strips():
    state = TrackingState()
    state.set_current_scenario("  Lane_Keep  ")
    assert state.current_scenario == "lane_keep"


def test_tracking_state_set_scenario_none_resets():
    state = TrackingState()
    state.set_current_scenario("intersection")
    state.set_current_scenario(None)
    assert state.current_scenario is None


def test_tracking_state_set_scenario_empty_string_normalizes_to_none():
    state = TrackingState()
    state.set_current_scenario("")
    assert state.current_scenario is None


def test_profile_lane_keep_uses_aggressive_gain():
    """Plan T1.4: el perfil lane_keep usa autoridad fuerte de cámara.

    Esto resuelve el síntoma reportado de "se guía mucho por la posición y
    muy poco por el carril" en recta. Si alguien baja estos valores, este
    test obliga a revisar la decisión.
    """
    profile = resolve_visual_correction_profile("lane_keep")
    assert profile.gain == pytest.approx(0.45)
    assert profile.max_m == pytest.approx(0.06)
    assert profile.step_max_m == pytest.approx(0.035)
    assert not profile.is_disabled


def test_profile_parking_is_disabled():
    """En parking la cámara NO debe corregir el pose — el MPC parking lleva."""
    profile = resolve_visual_correction_profile("parking")
    assert profile.is_disabled
    assert profile.gain == pytest.approx(0.0)


def test_profile_intersection_is_weak():
    """En intersección la cámara aporta poco — el mapa OSM dicta la trayectoria."""
    profile = resolve_visual_correction_profile("intersection")
    assert profile.gain <= 0.10
    assert profile.max_m <= 0.02


def test_profile_default_fallback_for_unknown_scenario():
    profile = resolve_visual_correction_profile("scenario_inventado_xyz")
    default = resolve_visual_correction_profile("default")
    assert profile.gain == default.gain
    assert profile.max_m == default.max_m
    assert profile.step_max_m == default.step_max_m


def test_profile_default_fallback_for_none():
    profile = resolve_visual_correction_profile(None)
    default = resolve_visual_correction_profile("default")
    assert profile.gain == default.gain


def test_profile_case_insensitive():
    upper = resolve_visual_correction_profile("LANE_KEEP")
    lower = resolve_visual_correction_profile("lane_keep")
    assert upper.gain == lower.gain
    assert upper.max_m == lower.max_m


def test_profile_returns_visualcorrectionprofile_instance():
    profile = resolve_visual_correction_profile("lane_keep")
    assert isinstance(profile, VisualCorrectionProfile)


# ────────────────────────────────────────────────────────────────────
# Política single_line — post-mortem run_20260518_025658
# ────────────────────────────────────────────────────────────────────
# El observador en single_line extrapola la línea faltante asumiendo
# lane_width=0.35 m. Si el ancho real difiere, ``direct_error_m``
# satura (28 cm vistos consistentes). El urt-ref no consume la cámara
# para control (``subLane=False`` en control_node2.py); adoptamos la
# misma postura para la corrección al pose: en single_line se ignora,
# two_line sigue activo.

class TestSingleLinePoseCorrectionPolicy:
    """Verifica que pose_estimator NO usa single_line para mover el pose."""

    def test_single_line_branch_returns_early(self):
        """El código fuente debe contener el guard explícito de single_line.

        Buscamos el patrón canónico: comparar ``measurement_mode`` con
        ``"single_line"`` y retornar sin aplicar corrección.
        """
        from pathlib import Path
        src = Path(
            "src/localization/pose_estimator_thread.py"
        ).read_text(encoding="utf-8")
        # Doble verificación: existe la rama y registra el evento de log.
        assert 'measurement_mode == "single_line"' in src
        assert "lane_correction_single_line_ignored" in src

    def test_old_attenuation_constants_are_gone(self):
        """Garantizamos que no quedaron las constantes de la atenuación
        viejas (atenuación 35%); si reaparecen es porque alguien
        reintrodujo el approach intermedio sin querer."""
        from src.localization import pose_estimator_thread as pet
        assert not hasattr(pet, "_LANE_SINGLE_LINE_ATTENUATION")
        assert not hasattr(pet, "_LANE_SINGLE_LINE_MAX_ERROR_M")
