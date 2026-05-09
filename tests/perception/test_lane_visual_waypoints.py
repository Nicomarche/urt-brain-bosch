# tests/perception/test_lane_visual_waypoints.py
#
# Cubre el camino crítico del paradigma urt-ref: la percepción ajusta
# polinomios a las líneas, sintetiza la línea ausente desplazando la
# opuesta `LANE_WIDTH_M`, y emite waypoints del centro en frame body
# que el `BehaviorPlanner` consume como referencia primaria del MPC.
#
# Tests:
#   1. `lane_observation_has_visual_path` gatea correctamente sobre
#      cantidad de puntos y calidad.
#   2. `_build_lane_observation` lee `visual_lane_waypoints` del
#      frame_trace y sube la calidad incluso en single_line.
#   3. La extrapolación de single_line preserva la curvatura del
#      polinomio visible (no asume paralelismo) — equivalente al
#      `LaneDetector.hpp:603-616` de urt-ref.

from __future__ import annotations

import math
import unittest

from src.core.types import LaneObservation, VisualStateSnapshot
from src.core.types.perception import lane_observation_has_visual_path
from src.perception.lane.lane_observer_thread import threadLaneObserver


def _straight_waypoints(n: int = 20, density: float = 0.05):
    return tuple((density * i, 0.0, 0.0) for i in range(n))


class HasVisualPathTests(unittest.TestCase):
    def test_returns_false_when_no_observation(self):
        self.assertFalse(lane_observation_has_visual_path(None))

    def test_returns_false_when_too_few_waypoints(self):
        obs = LaneObservation(
            center_waypoints_body=_straight_waypoints(n=4),
            quality=0.85,
        )
        self.assertFalse(lane_observation_has_visual_path(obs, min_points=8))

    def test_returns_false_when_quality_below_threshold(self):
        obs = LaneObservation(
            center_waypoints_body=_straight_waypoints(n=20),
            quality=0.40,
        )
        self.assertFalse(lane_observation_has_visual_path(obs, min_quality=0.55))

    def test_returns_true_for_single_line_with_polynomial_waypoints(self):
        obs = LaneObservation(
            center_waypoints_body=_straight_waypoints(n=20),
            measurement_mode="single_line",
            quality=0.85,
            extrapolated_side="right",
            lane_width_m=0.35,
        )
        self.assertTrue(lane_observation_has_visual_path(obs))

    def test_rejects_non_finite_waypoints(self):
        bad = ((0.0, 0.0, 0.0), (math.inf, 0.0, 0.0)) + _straight_waypoints(n=8)
        obs = LaneObservation(center_waypoints_body=bad, quality=0.85)
        self.assertFalse(lane_observation_has_visual_path(obs))


class LaneObserverVisualPathTests(unittest.TestCase):
    def test_visual_waypoints_in_frame_trace_populate_observation(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        waypoints = _straight_waypoints(n=20)
        snapshot = VisualStateSnapshot(
            timestamp=5.0,
            detection_mode="ai_local",
            local_lane_payload={"lane_side_point_counts": {"left": 6, "right": 0}},
            frame_trace={
                "debug": {"measurement_mode": "single_line"},
                "visual_lane_waypoints": {
                    "center_waypoints_body": waypoints,
                    "left_poly_coeffs": (0.0, 0.0, 0.5, -50.0),
                    "right_poly_coeffs": None,
                    "lane_width_m": 0.35,
                    "extrapolated_side": "right",
                },
            },
        )

        obs = observer._build_lane_observation(snapshot)

        self.assertEqual(len(obs.center_waypoints_body), 20)
        self.assertEqual(obs.extrapolated_side, "right")
        self.assertEqual(obs.measurement_mode, "single_line")
        self.assertGreaterEqual(obs.quality, 0.85)
        self.assertEqual(obs.left_poly_coeffs, (0.0, 0.0, 0.5, -50.0))
        self.assertIsNone(obs.right_poly_coeffs)
        self.assertAlmostEqual(obs.lane_width_m or 0.0, 0.35, places=4)

    def test_visual_waypoints_derive_direct_error_from_first_point(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        # y_left positivo = vehículo desplazado a la izquierda → direct_error negativo
        waypoints = ((0.05, 0.04, 0.0),) + _straight_waypoints(n=19)
        snapshot = VisualStateSnapshot(
            timestamp=5.0,
            detection_mode="ai_local",
            local_lane_payload={"lane_side_point_counts": {"left": 6, "right": 6}},
            frame_trace={
                "debug": {"measurement_mode": "two_line"},
                "visual_lane_waypoints": {
                    "center_waypoints_body": waypoints,
                    "lane_width_m": 0.35,
                },
            },
        )

        obs = observer._build_lane_observation(snapshot)

        self.assertTrue(obs.direct_error_valid)
        self.assertAlmostEqual(obs.direct_error_m or 0.0, -0.04, places=4)

    def test_no_visual_waypoints_keeps_legacy_quality(self):
        observer = threadLaneObserver.__new__(threadLaneObserver)
        snapshot = VisualStateSnapshot(
            timestamp=5.0,
            detection_mode="ai_local",
            local_lane_payload={"lane_side_point_counts": {"left": 6, "right": 0}},
            frame_trace={"debug": {"measurement_mode": "single_line", "sl_direct_error_m": 0.05}},
        )

        obs = observer._build_lane_observation(snapshot)

        self.assertEqual(len(obs.center_waypoints_body), 0)
        self.assertAlmostEqual(obs.quality, 0.65, places=4)


if __name__ == "__main__":
    unittest.main()
