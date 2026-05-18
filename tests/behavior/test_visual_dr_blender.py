"""Tests para `VisualDrBlender`.

Cubren:
  - Modo legacy (LANE_BLEND_ENABLED=False) → α ∈ {0, 1} replicando el gate clásico.
  - Modo blend (LANE_BLEND_ENABLED=True):
      * 2 líneas con quality alta → α → 1.
      * 1 línea → α se asienta cerca de LANE_BLEND_ALPHA_SINGLE_LINE.
      * 0 líneas → α → 0.
      * Hard-block del gate (intersection/map authority) → α = 0 instantáneo.
      * Quality baja atenúa α target a 0.
      * Constantes de tiempo asimétricas: bajar α (decay) más rápido que subirlo
        (recovery).
      * Sin corredor DR disponible → α delega al gate clásico binario.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.behavior.visual_dr_blender import VisualDrBlender
from src.core.types.perception import LaneObservation
from src.routing.lanelet.lanelet_map import from_track_graph
from tests.behavior.conftest import _build_track_graph, make_context


def _straight_visual_waypoints(n: int = 20, density: float = 0.05):
    return tuple((density * i, 0.0, 0.0) for i in range(n))


def _lookahead_visual_waypoints(n: int = 20, start_m: float = 0.20, density: float = 0.05):
    return tuple((start_m + (density * i), 0.0, 0.0) for i in range(n))


def _make_lane_obs(
    *,
    detected_sides: tuple[str, ...],
    quality: float,
    measurement_mode: str,
    waypoints=None,
    extrapolated_side: str | None = None,
) -> LaneObservation:
    return LaneObservation(
        detected_sides=detected_sides,
        quality=quality,
        measurement_mode=measurement_mode,
        direct_error_valid=True,
        direct_error_m=0.0,
        center_waypoints_body=waypoints or _lookahead_visual_waypoints(),
        extrapolated_side=extrapolated_side,
        lane_width_m=0.35,
        control_policy_mode="ROUTE_TRACKING",
        planner_priority_active=True,
    )


def _make_route_ctx(*, lane_obs: LaneObservation | None = None):
    route_waypoints = [
        [0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0],
        [0.8, 0.0, 0.0],
        [1.2, 0.0, 0.0],
    ]
    lanelet_map = from_track_graph(
        _build_track_graph(
            [("n1", 0.0, 0.0, 0), ("n2", 1.0, 0.0, 0), ("n3", 2.0, 0.0, 0)]
        ),
        step_m=0.20,
    )
    ctx = make_context(
        pose_x=0.02,
        pose_y=0.05,
        pose_yaw=0.0,
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3",),
        lanelet_map=lanelet_map,
        route_waypoints=route_waypoints,
        matched_pose=(0.0, 0.0, 0.0),
    )
    return replace(ctx, lane_observation=lane_obs) if lane_obs is not None else ctx


# ----------------------------------------------------------------------
# Modo legacy
# ----------------------------------------------------------------------


def test_blender_legacy_emits_alpha_1_when_gate_uses_visual(monkeypatch) -> None:
    monkeypatch.setattr("config.LANE_BLEND_ENABLED", False)
    blender = VisualDrBlender()
    ctx = _make_route_ctx(
        lane_obs=_make_lane_obs(
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            waypoints=_straight_visual_waypoints(),
        )
    )
    decision = None
    for _ in range(5):  # cubrir la hysteresis del gate (enter ticks = 3)
        decision = blender.decide(
            ctx=ctx,
            lane_observation=ctx.lane_observation,
            scenario_name="lane_keep",
            route_corridor_available=True,
            lanelet_corridor_available=True,
            now_s=0.0,
        )
    assert decision is not None
    assert decision.alpha == 1.0
    assert decision.gate_decision.use_visual_path is True
    assert decision.notes["lane_blend_enabled"] is False


def test_blender_legacy_emits_alpha_0_when_gate_rejects(monkeypatch) -> None:
    monkeypatch.setattr("config.LANE_BLEND_ENABLED", False)
    blender = VisualDrBlender()
    # Quality muy baja → gate rejecta por low_quality.
    ctx = _make_route_ctx(
        lane_obs=_make_lane_obs(
            detected_sides=("left", "right"),
            quality=0.40,
            measurement_mode="two_line",
            waypoints=_straight_visual_waypoints(),
        )
    )
    decision = blender.decide(
        ctx=ctx,
        lane_observation=ctx.lane_observation,
        scenario_name="lane_keep",
        route_corridor_available=True,
        lanelet_corridor_available=True,
        now_s=0.0,
    )
    assert decision.alpha == 0.0
    assert decision.gate_decision.use_visual_path is False


# ----------------------------------------------------------------------
# Modo blend
# ----------------------------------------------------------------------


def test_blender_blend_two_line_settles_at_alpha_1(monkeypatch) -> None:
    monkeypatch.setattr("config.LANE_BLEND_ENABLED", True)
    monkeypatch.setattr("config.LANE_BLEND_TAU_DECAY_S", 0.05)
    monkeypatch.setattr("config.LANE_BLEND_TAU_RECOVERY_S", 0.05)
    blender = VisualDrBlender()
    ctx = _make_route_ctx(
        lane_obs=_make_lane_obs(
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            waypoints=_straight_visual_waypoints(),
        )
    )
    # Pasamos el enter-ticks del gate + tiempo de suavizado.
    decision = None
    for tick in range(20):
        decision = blender.decide(
            ctx=ctx,
            lane_observation=ctx.lane_observation,
            scenario_name="lane_keep",
            route_corridor_available=True,
            lanelet_corridor_available=True,
            now_s=tick * 0.05,
        )
    assert decision is not None
    assert decision.alpha == pytest.approx(1.0, abs=1e-3)
    assert decision.target_alpha == pytest.approx(1.0, abs=1e-6)
    assert decision.num_lines == 2


def test_blender_blend_single_line_settles_at_configured_alpha(monkeypatch) -> None:
    monkeypatch.setattr("config.LANE_BLEND_ENABLED", True)
    monkeypatch.setattr("config.LANE_BLEND_ALPHA_SINGLE_LINE", 0.3)
    monkeypatch.setattr("config.LANE_BLEND_QUALITY_MIN", 0.5)
    monkeypatch.setattr("config.LANE_BLEND_TAU_DECAY_S", 0.05)
    monkeypatch.setattr("config.LANE_BLEND_TAU_RECOVERY_S", 0.05)
    blender = VisualDrBlender()
    ctx = _make_route_ctx(
        lane_obs=_make_lane_obs(
            detected_sides=("left",),
            quality=1.0,
            measurement_mode="single_line",
            waypoints=_lookahead_visual_waypoints(),
            extrapolated_side="right",
        )
    )
    decision = None
    for tick in range(40):
        decision = blender.decide(
            ctx=ctx,
            lane_observation=ctx.lane_observation,
            scenario_name="lane_keep",
            route_corridor_available=True,
            lanelet_corridor_available=True,
            now_s=tick * 0.05,
        )
    assert decision is not None
    assert decision.num_lines == 1
    # quality_factor = (1.0 - 0.5) / (1.0 - 0.5) = 1.0 ⇒ target = 0.3 * 1.0
    assert decision.target_alpha == pytest.approx(0.3, abs=1e-6)
    assert decision.alpha == pytest.approx(0.3, abs=5e-3)


def test_blender_blend_no_lines_drops_alpha_to_zero(monkeypatch) -> None:
    monkeypatch.setattr("config.LANE_BLEND_ENABLED", True)
    blender = VisualDrBlender()
    ctx = _make_route_ctx(
        lane_obs=LaneObservation(detected_sides=(), quality=0.0, measurement_mode="none"),
    )
    decision = blender.decide(
        ctx=ctx,
        lane_observation=ctx.lane_observation,
        scenario_name="lane_keep",
        route_corridor_available=True,
        lanelet_corridor_available=True,
        now_s=0.0,
    )
    assert decision.alpha == 0.0
    assert decision.num_lines == 0
    assert decision.visual_available is False


def test_blender_low_quality_two_line_attenuates_target_alpha(monkeypatch) -> None:
    monkeypatch.setattr("config.LANE_BLEND_ENABLED", True)
    monkeypatch.setattr("config.LANE_BLEND_QUALITY_MIN", 0.5)
    monkeypatch.setattr("config.LANE_BLEND_TAU_DECAY_S", 0.05)
    monkeypatch.setattr("config.LANE_BLEND_TAU_RECOVERY_S", 0.05)
    monkeypatch.setattr("config.LANE_VISUAL_PRIMARY_TWO_LINE_MIN_QUALITY", 0.75)
    blender = VisualDrBlender()
    # quality justo arriba del gate (0.78) pero la atenuación del blend la baja.
    ctx = _make_route_ctx(
        lane_obs=_make_lane_obs(
            detected_sides=("left", "right"),
            quality=0.78,
            measurement_mode="two_line",
            waypoints=_straight_visual_waypoints(),
        )
    )
    decision = None
    for tick in range(40):
        decision = blender.decide(
            ctx=ctx,
            lane_observation=ctx.lane_observation,
            scenario_name="lane_keep",
            route_corridor_available=True,
            lanelet_corridor_available=True,
            now_s=tick * 0.05,
        )
    assert decision is not None
    # quality_factor = (0.78 - 0.5) / (1.0 - 0.5) = 0.56 ⇒ target = 1.0 * 0.56
    assert decision.target_alpha == pytest.approx(0.56, abs=1e-3)
    # No es 1.0: hay degradación visual real.
    assert decision.alpha < 0.8


def test_blender_hard_block_forces_alpha_zero(monkeypatch) -> None:
    """Un escenario fuera de LANE_KEEP es hard-block del gate ⇒ α=0 instantáneo."""
    monkeypatch.setattr("config.LANE_BLEND_ENABLED", True)
    blender = VisualDrBlender()
    ctx = _make_route_ctx(
        lane_obs=_make_lane_obs(
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            waypoints=_straight_visual_waypoints(),
        )
    )
    # Le pasamos un scenario_name que no está en LANE_VISUAL_PRIMARY_ALLOWED_SCENARIOS.
    decision = blender.decide(
        ctx=ctx,
        lane_observation=ctx.lane_observation,
        scenario_name="intersection",
        route_corridor_available=True,
        lanelet_corridor_available=True,
        now_s=0.0,
    )
    assert decision.alpha == 0.0
    assert decision.reason.startswith("hard_block")


def test_blender_decay_is_faster_than_recovery(monkeypatch) -> None:
    monkeypatch.setattr("config.LANE_BLEND_ENABLED", True)
    monkeypatch.setattr("config.LANE_BLEND_TAU_DECAY_S", 0.1)
    monkeypatch.setattr("config.LANE_BLEND_TAU_RECOVERY_S", 1.0)
    blender = VisualDrBlender()
    two_line_ctx = _make_route_ctx(
        lane_obs=_make_lane_obs(
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            waypoints=_straight_visual_waypoints(),
        )
    )
    no_line_obs = LaneObservation(detected_sides=(), quality=0.0, measurement_mode="none")
    no_line_ctx = replace(two_line_ctx, lane_observation=no_line_obs)
    # Calentamos para llegar a α≈1.
    for tick in range(80):
        blender.decide(
            ctx=two_line_ctx,
            lane_observation=two_line_ctx.lane_observation,
            scenario_name="lane_keep",
            route_corridor_available=True,
            lanelet_corridor_available=True,
            now_s=tick * 0.05,
        )
    assert blender._alpha_prev == pytest.approx(1.0, abs=1e-3)
    # Perdemos visión → α DEBE caer rápido (target=0, hard-instant porque
    # visual_available=False) ⇒ α=0 instantáneo.
    decision_loss = blender.decide(
        ctx=no_line_ctx,
        lane_observation=no_line_obs,
        scenario_name="lane_keep",
        route_corridor_available=True,
        lanelet_corridor_available=True,
        now_s=80 * 0.05,
    )
    assert decision_loss.alpha == 0.0
    # Recuperamos visión: α target=1, pero τ_recovery alto → un solo tick
    # NO debe llegar a 1.
    decision_recover = blender.decide(
        ctx=two_line_ctx,
        lane_observation=two_line_ctx.lane_observation,
        scenario_name="lane_keep",
        route_corridor_available=True,
        lanelet_corridor_available=True,
        now_s=(80 * 0.05) + 0.05,
    )
    assert 0.0 < decision_recover.alpha < 0.3


def test_blender_no_route_corridor_delegates_to_gate(monkeypatch) -> None:
    """Sin DR available, blender no puede mezclar — usa el gate binario."""
    monkeypatch.setattr("config.LANE_BLEND_ENABLED", True)
    blender = VisualDrBlender()
    lane_obs = _make_lane_obs(
        detected_sides=("left",),
        quality=0.9,
        measurement_mode="single_line",
        waypoints=_lookahead_visual_waypoints(),
        extrapolated_side="right",
    )
    ctx = _make_route_ctx(lane_obs=lane_obs)
    decision = None
    for tick in range(10):
        decision = blender.decide(
            ctx=ctx,
            lane_observation=ctx.lane_observation,
            scenario_name="lane_keep",
            route_corridor_available=False,
            lanelet_corridor_available=False,
            now_s=tick * 0.05,
        )
    assert decision is not None
    # Gate aprobó (single_line con quality 0.9 > 0.85) ⇒ α=1.
    assert decision.alpha == 1.0
    assert decision.reason.startswith("no_route_corridor")
