"""Tests para `blend_target_paths`.

Verifica:
  - α=0 → idéntico al path route.
  - α=1 → idéntico al path visual.
  - α=0.5 → interpolación posicional lineal.
  - Headings cerca de ±π no producen saltos (blend angle-aware con atan2).
  - Truncado al mínimo común cuando shapes difieren.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.behavior.trajectory_builder import blend_target_paths


def _path(xs, ys, psis):
    return np.column_stack([np.asarray(xs, dtype=float),
                            np.asarray(ys, dtype=float),
                            np.asarray(psis, dtype=float)])


def test_blend_alpha_zero_returns_route_path() -> None:
    path_visual = _path([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [0.1, 0.1, 0.1])
    path_route = _path([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    blended = blend_target_paths(path_visual, path_route, alpha=0.0)
    np.testing.assert_allclose(blended, path_route)


def test_blend_alpha_one_returns_visual_path() -> None:
    path_visual = _path([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [0.1, 0.1, 0.1])
    path_route = _path([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    blended = blend_target_paths(path_visual, path_route, alpha=1.0)
    np.testing.assert_allclose(blended, path_visual)


def test_blend_alpha_half_interpolates_positions_linearly() -> None:
    path_visual = _path([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    path_route = _path([0.0, 1.0, 2.0], [-1.0, -1.0, -1.0], [0.0, 0.0, 0.0])
    blended = blend_target_paths(path_visual, path_route, alpha=0.5)
    # x igual en ambos ⇒ se mantiene. y = (1.0 + (-1.0)) / 2 = 0.0.
    np.testing.assert_allclose(blended[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(blended[:, 1], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(blended[:, 2], [0.0, 0.0, 0.0], atol=1e-9)


def test_blend_heading_avoids_wraparound_near_pi() -> None:
    """ψ_v = π−ε y ψ_r = −π+ε deberían blendearse a ±π, no a 0."""
    eps = 0.01
    psi_v = math.pi - eps
    psi_r = -math.pi + eps
    path_visual = _path([0.0, 1.0], [0.0, 0.0], [psi_v, psi_v])
    path_route = _path([0.0, 1.0], [0.0, 0.0], [psi_r, psi_r])
    blended = blend_target_paths(path_visual, path_route, alpha=0.5)
    # El blend correcto está cerca de ±π (no cerca de 0). Verificamos que la
    # magnitud del seno sigue cerca de 0 (apunta backwards) y el coseno está
    # cerca de -1.
    for row in blended:
        assert math.cos(row[2]) == pytest.approx(-1.0, abs=1e-2)
        assert abs(math.sin(row[2])) < 0.1


def test_blend_heading_at_quadrant_boundary() -> None:
    """ψ_v = π/2, ψ_r = -π/2 con α=0.5 → resultado mal definido (apunta a 0 ó π).

    Lo importante es que NO sea NaN/infinito y se mantenga finito.
    """
    path_visual = _path([0.0, 1.0], [0.0, 0.0], [math.pi / 2, math.pi / 2])
    path_route = _path([0.0, 1.0], [0.0, 0.0], [-math.pi / 2, -math.pi / 2])
    blended = blend_target_paths(path_visual, path_route, alpha=0.5)
    assert np.all(np.isfinite(blended))


def test_blend_truncates_to_minimum_shape() -> None:
    path_visual = _path([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    path_route = _path([0.0, 1.0], [0.0, 0.0], [0.0, 0.0])
    blended = blend_target_paths(path_visual, path_route, alpha=0.5)
    assert blended.shape == (2, 3)


def test_blend_handles_empty_paths() -> None:
    path_visual = _path([0.0, 1.0], [1.0, 1.0], [0.0, 0.0])
    blended = blend_target_paths(path_visual, np.zeros((0, 3)), alpha=0.5)
    np.testing.assert_allclose(blended, path_visual)
    blended2 = blend_target_paths(np.zeros((0, 3)), path_visual, alpha=0.5)
    np.testing.assert_allclose(blended2, path_visual)


def test_blend_clamps_alpha_to_unit_interval() -> None:
    path_visual = _path([0.0, 1.0], [1.0, 1.0], [0.0, 0.0])
    path_route = _path([0.0, 1.0], [0.0, 0.0], [0.0, 0.0])
    # α > 1 ⇒ devuelve visual puro.
    blended_high = blend_target_paths(path_visual, path_route, alpha=1.5)
    np.testing.assert_allclose(blended_high, path_visual)
    # α < 0 ⇒ devuelve route puro.
    blended_low = blend_target_paths(path_visual, path_route, alpha=-0.5)
    np.testing.assert_allclose(blended_low, path_route)
