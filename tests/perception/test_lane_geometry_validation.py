# tests/perception/test_lane_geometry_validation.py
#
# Plan TANDA 0.1 — Carril contramano.
#
# Cubre `threadLaneObserver._line_geometry_valid` con énfasis en el caso
# donde el detector ve la línea del carril OPUESTO y la confunde como
# borde propio. El check de consistencia de ancho (suma de distancias ≈
# ancho físico del carril) rechaza esa observación para que LaneKeep
# caiga a route_tracking en vez de cruzar al carril contrario.

from __future__ import annotations

import unittest

from src.perception.lane.lane_observer_thread import threadLaneObserver


class LineGeometryValidationTests(unittest.TestCase):
    """Cubrir los 4 casos canónicos del fix de TANDA 0.1."""

    def test_matched_width_centered_car(self):
        """Auto centrado en su carril: L=0.175, R=0.175, suma=0.35 ≈ W=0.35."""
        valid, reason = threadLaneObserver._line_geometry_valid(0.175, 0.175)
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_matched_width_offset_car(self):
        """Auto desplazado a la izquierda: L=0.05, R=0.30, suma=0.35 ≈ W=0.35."""
        valid, reason = threadLaneObserver._line_geometry_valid(0.05, 0.30)
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_right_from_opposite_lane_rejected(self):
        """El bug clásico: el detector confunde la línea del carril opuesto
        como "right" del carril propio. L=0.05, R=0.40, suma=0.45 > W*1.25.
        Tiene que ser rechazado para no derivar contramano.
        """
        valid, reason = threadLaneObserver._line_geometry_valid(0.05, 0.40)
        self.assertFalse(valid)
        # El check de rango individual (margen ~0.063) rechaza primero
        # porque 0.40 está fuera de [-0.063, 0.413]. Cualquiera de los
        # dos motivos protege contra contramano.
        self.assertIn(reason, ("lane_width_mismatch", "line_distance_out_of_lane_bounds"))

    def test_both_lines_too_wide_lane_width_mismatch(self):
        """Las dos líneas individuales caen dentro del rango pero su SUMA
        no matchea el ancho físico. Es el caso difícil: el check viejo
        (sólo rango individual) las aceptaba; el nuevo rechaza por suma.
        """
        # L=0.20, R=0.20, suma=0.40 vs W=0.35 → diff relativa 14% < 25%
        # Esto NO debería rechazarse.
        valid_borderline, _ = threadLaneObserver._line_geometry_valid(0.20, 0.20)
        self.assertTrue(valid_borderline)
        # L=0.25, R=0.25, suma=0.50 vs W=0.35 → diff relativa 43% > 25%
        # Esto SÍ debe rechazarse por la nueva validación de suma.
        valid_mismatch, reason = threadLaneObserver._line_geometry_valid(0.25, 0.25)
        self.assertFalse(valid_mismatch)
        self.assertEqual(reason, "lane_width_mismatch")

    def test_too_narrow_lane_width_mismatch(self):
        """Suma muy chica respecto al ancho esperado: error de detector que
        proyecta ambas líneas al mismo punto."""
        # L=0.05, R=0.05, suma=0.10 vs W=0.35 → diff relativa 71% > 25%
        valid, reason = threadLaneObserver._line_geometry_valid(0.05, 0.05)
        self.assertFalse(valid)
        self.assertEqual(reason, "lane_width_mismatch")

    def test_single_line_short_circuit(self):
        """Con una sola línea no hay diff que validar — el método retorna
        True (la lógica de fallback la maneja el caller cuando quality
        baja por otros motivos)."""
        valid, reason = threadLaneObserver._line_geometry_valid(0.30, None)
        self.assertTrue(valid)
        self.assertIsNone(reason)
        valid, reason = threadLaneObserver._line_geometry_valid(None, 0.30)
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_negative_distance_out_of_bounds(self):
        """Una distancia negativa significativa indica medición errónea
        (la línea estaría DETRÁS del auto). Rechazar por rango individual."""
        valid, reason = threadLaneObserver._line_geometry_valid(-0.30, 0.30)
        self.assertFalse(valid)
        self.assertEqual(reason, "line_distance_out_of_lane_bounds")

    def test_caller_passes_measured_width_does_not_relax_sum_check(self):
        """Aún si el caller pasa `lane_width_m=0.50` (porque el detector
        midió mal), la consistencia se mide contra el ancho FÍSICO del
        config (35 cm). Esto evita que un detector mentiroso se valide a
        sí mismo."""
        # L=0.30, R=0.20: rango individual permite hasta 0.50+margen, así
        # que con lane_width_m=0.50 pasaría el check de rango — pero la
        # suma (0.50) sigue siendo 43% > 25% más grande que 0.35 físico.
        valid, reason = threadLaneObserver._line_geometry_valid(
            0.30, 0.20, lane_width_m=0.50
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "lane_width_mismatch")


if __name__ == "__main__":
    unittest.main()
