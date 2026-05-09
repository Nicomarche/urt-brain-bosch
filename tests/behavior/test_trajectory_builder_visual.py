# tests/behavior/test_trajectory_builder_visual.py
#
# Tests del nuevo `build_target_path_from_visual`. Replica la lógica de
# `urt-ref::LaneDetector::get_waypoints` → `state_refs` del MPC: tomar
# waypoints en frame body del vehículo (x_fwd, y_left, ψ_local) y
# transformarlos rígidamente al frame OSM/mapa usando la pose fusionada.
#
# Convención CLAVE (foco principal de los tests):
#   Body: y_left positivo = a la izquierda del piloto.
#   OSM/mapa: y crece hacia el SUR en pantalla, yaw es CW+. La izquierda
#   del piloto al heading=0 (este) es el norte en pantalla → y NEGATIVO
#   en OSM. `motion_controller._osm_pose_to_controller_frame` espeja
#   después al frame ENU del solver acados.

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.behavior.trajectory_builder import build_target_path_from_visual


def _make_pose(x: float, y: float, yaw: float):
    return SimpleNamespace(x=float(x), y=float(y), yaw=float(yaw))


def _straight_forward(n: int, density: float = 0.05):
    return tuple((density * i, 0.0, 0.0) for i in range(n))


class TestBuildTargetPathFromVisualShape:
    """Validan el contrato de shape (N+1, 3)."""

    @pytest.mark.parametrize("horizon_n", [10, 20, 40])
    def test_path_shape_matches_horizon(self, horizon_n):
        path = build_target_path_from_visual(
            center_waypoints_body=_straight_forward(40),
            ego_pose=_make_pose(0, 0, 0),
            target_speed_mps=1.0,
            horizon_n=horizon_n,
            dt=0.05,
        )
        assert path.shape == (horizon_n + 1, 3)

    def test_empty_waypoints_returns_static_path(self):
        path = build_target_path_from_visual(
            center_waypoints_body=(),
            ego_pose=_make_pose(2.0, 3.0, 0.5),
            target_speed_mps=1.0,
            horizon_n=10,
            dt=0.05,
        )
        assert path.shape == (11, 3)
        # Todos los puntos coinciden con la pose
        np.testing.assert_allclose(path[:, 0], 2.0)
        np.testing.assert_allclose(path[:, 1], 3.0)


class TestBuildTargetPathFromVisualTransform:
    """Validan el rigid-body transform body → OSM mundo."""

    def test_yaw_zero_keeps_x_forward_and_y_zero(self):
        # Sin lateral en body y heading=0 (este), la y queda constante
        path = build_target_path_from_visual(
            center_waypoints_body=_straight_forward(20),
            ego_pose=_make_pose(0, 0, 0),
            target_speed_mps=1.0,
            horizon_n=10,
            dt=0.05,
        )
        assert path[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert path[-1, 0] > 0.4
        np.testing.assert_allclose(path[:, 1], 0.0, atol=1e-6)

    def test_yaw_pi_over_two_in_osm_rotates_forward_to_south_screen(self):
        # En OSM yaw es CW+ desde +x. yaw=π/2 = 90° CW = sur en pantalla.
        path = build_target_path_from_visual(
            center_waypoints_body=_straight_forward(20),
            ego_pose=_make_pose(10.0, 5.0, math.pi / 2.0),
            target_speed_mps=1.0,
            horizon_n=10,
            dt=0.05,
        )
        # x_fwd body con yaw=π/2 → y_screen_south crece (más positivo en OSM).
        # y_left body=0 → x_east mundo=10.
        np.testing.assert_allclose(path[:, 0], 10.0, atol=1e-6)
        assert path[-1, 1] > 5.4

    def test_yaw_negative_pi_over_two_in_osm_rotates_forward_to_north_screen(self):
        # yaw=-π/2 = 90° CCW desde +x = norte en pantalla = -y en OSM.
        path = build_target_path_from_visual(
            center_waypoints_body=_straight_forward(20),
            ego_pose=_make_pose(0.0, 0.0, -math.pi / 2.0),
            target_speed_mps=1.0,
            horizon_n=10,
            dt=0.05,
        )
        np.testing.assert_allclose(path[:, 0], 0.0, atol=1e-6)
        assert path[-1, 1] < -0.4  # Norte en pantalla

    def test_y_left_positive_translates_to_negative_y_in_osm(self):
        # En OSM (y crece al sur), la "izquierda del piloto al heading=0"
        # es el NORTE en pantalla = y NEGATIVO. Esta es la fuente del bug
        # "el coche dirige al lado contrario": si emitís y_left con la
        # convención ENU directa, el path sale espejado.
        wpts = ((0.5, 0.10, 0.0),) + _straight_forward(20)
        path = build_target_path_from_visual(
            center_waypoints_body=wpts,
            ego_pose=_make_pose(0.0, 0.0, 0.0),
            target_speed_mps=1.0,
            horizon_n=20,
            dt=0.05,
        )
        # Algún sample inicial debe estar al NORTE en pantalla (y<0 en OSM)
        # por la ofset lateral izquierdo inyectada en el primer waypoint.
        assert min(path[:5, 1]) < 0.0

    def test_prepends_ego_pose_when_visual_waypoints_start_at_lookahead(self):
        # En runtime los waypoints visuales no arrancan en x_fwd=0 sino en el
        # primer lookahead visible. El target_path debe conectarse desde la
        # pose actual para no dejar un sample0 "saltado" respecto del resto.
        wpts = tuple((0.20 + (0.05 * i), 0.10, 0.0) for i in range(20))
        path = build_target_path_from_visual(
            center_waypoints_body=wpts,
            ego_pose=_make_pose(1.0, 2.0, 0.0),
            target_speed_mps=1.0,
            horizon_n=10,
            dt=0.05,
        )

        assert path[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert path[0, 1] == pytest.approx(2.0, abs=1e-6)
        assert path[1, 0] > path[0, 0]
        assert path[1, 1] < path[0, 1]

    def test_can_skip_ego_pose_connection_for_single_line_visual_paths(self):
        # En single_line preferimos NO unir el path al ego con una diagonal
        # artificial: el MPC debe ver la geometría visual real a partir del
        # primer lookahead, no un giro sintético desde la pose actual.
        wpts = tuple((0.20 + (0.05 * i), 0.10, 0.0) for i in range(20))
        path = build_target_path_from_visual(
            center_waypoints_body=wpts,
            ego_pose=_make_pose(1.0, 2.0, 0.0),
            target_speed_mps=1.0,
            horizon_n=10,
            dt=0.05,
            connect_from_ego_pose=False,
        )

        assert path[0, 0] > 1.15
        assert path[0, 1] < 1.95
        assert path[0, 2] == pytest.approx(0.0, abs=1e-6)

    def test_skips_nonforward_visual_prefix_before_connecting_to_ego(self):
        # Caso observado en logs: el prefijo visual arranca casi lateral /
        # levemente hacia atrás. Si se conecta directo con la pose, el MPC en
        # precision mode re-muestrea un sample1 "en reversa".
        wpts = (
            (0.00, 0.03, 0.0),
            (-0.01, 0.08, 0.0),
            (0.05, 0.14, 0.0),
            (0.14, 0.20, 0.0),
            (0.24, 0.24, 0.0),
            (0.34, 0.27, 0.0),
        )
        pose = _make_pose(0.0, 0.0, 0.0)
        path = build_target_path_from_visual(
            center_waypoints_body=wpts,
            ego_pose=pose,
            target_speed_mps=1.0,
            horizon_n=10,
            dt=0.05,
        )

        dx = float(path[1, 0] - pose.x)
        dy = float(path[1, 1] - pose.y)
        x_fwd = math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy

        assert x_fwd > 0.0
        assert path[1, 0] > path[0, 0]


class TestBuildTargetPathFromVisualExtension:
    """Validan el comportamiento cuando los waypoints no cubren el horizonte."""

    def test_short_polyline_extends_in_straight_line(self):
        # Sólo 5 waypoints en 0.20 m, horizonte de 1.5 m
        path = build_target_path_from_visual(
            center_waypoints_body=_straight_forward(5, density=0.05),
            ego_pose=_make_pose(0.0, 0.0, 0.0),
            target_speed_mps=1.0,
            horizon_n=30,
            dt=0.05,
        )
        # El path tiene shape correcto
        assert path.shape == (31, 3)
        # Y se extiende monotónicamente (sin frenar al borde del polinomio)
        assert path[-1, 0] > path[5, 0]
