# tests/behavior/test_trajectory_builder.py
#
# Tests del trajectory_builder. Validamos:
#   1. La salida tiene shape (N+1, 3) según el contrato.
#   2. El primer waypoint queda cerca del ego (start_xy), no del nodo
#      upstream de la lanelet.
#   3. El espaciado entre waypoints corresponde al `target_speed * dt`.
#   4. Las headings (psi) son consistentes con la dirección de avance
#      (atan2 del segmento adyacente).
#   5. Cuando el ego está al final de un grafo terminal, los waypoints
#      sobrantes se replican (no crashea).

from __future__ import annotations

import math

import numpy as np
import pytest

from src.behavior.trajectory_builder import build_target_path


def test_path_has_correct_shape(straight_lanelet_map) -> None:
    """build_target_path devuelve (N+1, 3)."""
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=0.5,
        horizon_n=10,
        dt=0.05,
    )
    assert path.shape == (11, 3)


def test_path_starts_near_ego(straight_lanelet_map) -> None:
    """El primer waypoint debería estar cerca del ego (proyección)."""
    ego = (0.5, 0.0)
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=ego,
        target_speed_mps=0.5,
        horizon_n=8,
        dt=0.05,
    )
    # Distancia entre ego y first waypoint < step_arc.
    step_arc = 0.5 * 0.05
    dist = math.hypot(path[0, 0] - ego[0], path[0, 1] - ego[1])
    assert dist <= step_arc + 0.10  # tolerance for densification artifacts


def test_path_spacing_matches_target_speed(straight_lanelet_map) -> None:
    """Distance entre samples ≈ max(0.10, target_speed * dt).

    El builder aplica un piso de 0.10 m al step para evitar muestreos
    degenerados a velocidades bajas. El test usa parámetros tales que
    `target_speed * dt > 0.10` para verificar la fórmula directamente.
    """
    target_speed = 2.0
    dt = 0.1
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=target_speed,
        horizon_n=10,
        dt=dt,
    )
    diffs = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    expected_step = max(0.10, target_speed * dt)
    valid_diffs = diffs[diffs > 1e-6]
    assert valid_diffs.size > 3
    assert np.median(valid_diffs) == pytest.approx(expected_step, abs=expected_step * 0.5)


def test_path_headings_match_direction(straight_lanelet_map) -> None:
    """Para una lanelet horizontal +x, todas las headings ≈ 0."""
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=0.5,
        horizon_n=10,
        dt=0.05,
    )
    # Todos los psi cerca de 0.
    np.testing.assert_allclose(path[:, 2], 0.0, atol=1e-3)


def test_path_handles_zero_speed_safely(straight_lanelet_map) -> None:
    """target_speed=0 ⇒ usa step mínimo de 0.10 m, no crashea."""
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=0.0,
        horizon_n=5,
        dt=0.05,
    )
    assert path.shape == (6, 3)


def test_path_replicates_end_when_grid_short(straight_lanelet_map) -> None:
    """Horizon larger than available path ⇒ último waypoint se replica."""
    # El grafo tiene 4 edges de 1 m c/u. Con speed 5 m/s y dt 0.05 s,
    # step_arc = 0.25 m, horizonte 100 ⇒ requeriría 25 m. El path se
    # estira pero los últimos samples coinciden con el endpoint.
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=5.0,
        horizon_n=100,
        dt=0.05,
    )
    assert path.shape == (101, 3)
    # Los últimos pares deberían tener distancia 0 (replicación).
    last_diff = np.linalg.norm(path[-1, :2] - path[-2, :2])
    assert last_diff < 1e-6


def test_path_follows_successor_chain(straight_lanelet_map) -> None:
    """Con horizon largo, el path debe atravesar las sucesoras."""
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=0.5,
        horizon_n=80,
        dt=0.05,
    )
    # Debería llegar al final del grafo (~4 m en eje x).
    final_x = path[-1, 0]
    assert final_x > 1.5  # cruzó al menos a la 2da lanelet


def test_path_zero_horizon_returns_minimal(straight_lanelet_map) -> None:
    """horizon_n=0 ⇒ devuelve un path mínimo de 1 punto."""
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=0.5,
        horizon_n=0,
        dt=0.05,
    )
    assert path.shape == (1, 3)
