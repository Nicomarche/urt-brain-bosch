# tests/perception/test_lane_polynomial_heading_anchor.py
#
# Cubre el "heading anchor" del polinomio: cuando el polinomio sub-rota
# en curvas (síntoma del usuario: "dobló demasiado abierto"), se rota
# rígidamente para alinear con `heading_error_rad` del near-field legacy.
#
# La función `_extract_lane_polynomials_and_waypoints` necesita una
# instancia de threadLineFollowing con muchas dependencias inicializadas;
# en estos tests instanciamos una versión mínima vía `__new__` y seteamos
# sólo los atributos que la función toca.

from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from src.hardware.camera.threads.threadLineFollowing import threadLineFollowing


def _make_minimal_thread(img_h=480, img_w=640, lane_width_cm=35.0):
    inst = threadLineFollowing.__new__(threadLineFollowing)
    inst.lane_width_cm = lane_width_cm
    inst._last_frame_size = (img_h, img_w)
    inst.show_debug = False
    src = np.float32([
        [img_w * 0.1, img_h],
        [img_w * 0.4, img_h * 0.6],
        [img_w * 0.6, img_h * 0.6],
        [img_w * 0.9, img_h],
    ])
    dst = np.float32([
        [img_w * 0.2, img_h],
        [img_w * 0.2, 0],
        [img_w * 0.8, 0],
        [img_w * 0.8, img_h],
    ])
    inst.perspective_M = cv2.getPerspectiveTransform(src, dst)
    inst.perspective_initialized = True
    bev_lane_w = img_w * 0.6
    inst.bev_cm_per_px = lane_width_cm / bev_lane_w
    return inst, img_h, img_w


def _straight_line_points(img_h, img_w, x_offset_px=0.0, n=30):
    return [
        (img_w / 2.0 + x_offset_px, y)
        for y in np.linspace(img_h * 0.6, img_h, n)
    ]


class TestHeadingAnchor(unittest.TestCase):
    def test_anchor_skips_when_no_hint(self):
        inst, h, w = _make_minimal_thread()
        pts = {
            "left": _straight_line_points(h, w, -120),
            "right": _straight_line_points(h, w, +120),
        }
        out = inst._extract_lane_polynomials_and_waypoints(pts, h, w, heading_hint_rad=None)
        self.assertIsNotNone(out)
        self.assertEqual(out["heading_anchor_applied_rad"], 0.0)

    def test_anchor_skips_when_polynomial_already_rotates(self):
        # Polinomio bien rotado (lateral barre 0.5 px/y) con un hint pequeño.
        inst, h, w = _make_minimal_thread()
        # Líneas inclinadas: lateral grande para forzar polinomio con heading.
        pts = {
            "left": [(img_x, y) for y, img_x in zip(
                np.linspace(h * 0.6, h, 30),
                np.linspace(w / 2.0 - 60, w / 2.0 - 180, 30),
            )],
            "right": [(img_x, y) for y, img_x in zip(
                np.linspace(h * 0.6, h, 30),
                np.linspace(w / 2.0 + 180, w / 2.0 + 60, 30),
            )],
        }
        # Hint pequeño: el polinomio ya rotó más → no anclar
        out = inst._extract_lane_polynomials_and_waypoints(pts, h, w, heading_hint_rad=0.05)
        self.assertIsNotNone(out)
        self.assertEqual(out["heading_anchor_applied_rad"], 0.0)

    def test_anchor_applies_when_polynomial_undershoots(self):
        # Polinomio captura un carril casi recto pero el legacy near-field
        # dice que hay un heading_error grande → anclar.
        inst, h, w = _make_minimal_thread()
        pts = {
            "left": _straight_line_points(h, w, -120),
            "right": _straight_line_points(h, w, +120),
        }
        out = inst._extract_lane_polynomials_and_waypoints(pts, h, w, heading_hint_rad=0.4)
        self.assertIsNotNone(out)
        anchor = float(out["heading_anchor_applied_rad"])
        # heading_hint=+0.4 → target_psi=-0.4. Polinomio era recto → anchor ≈ -0.4.
        self.assertLess(anchor, -0.30)
        self.assertGreater(anchor, -0.50)
        # El primer waypoint debe heading hacia la derecha (negativo en body math)
        wpts = out["center_waypoints_body"]
        self.assertGreaterEqual(len(wpts), 8)
        self.assertLess(wpts[0][2], -0.30)

    def test_anchor_flips_sign_when_polynomial_curves_wrong_direction(self):
        # Polinomio inclinado hacia la IZQUIERDA (heading positivo en body)
        # pero legacy heading_error=+0.3 (lane CW de car → target body psi
        # NEGATIVO). Sign flip claro → anchor.
        inst, h, w = _make_minimal_thread()
        pts = {
            "left": [(img_x, y) for y, img_x in zip(
                np.linspace(h * 0.6, h, 30),
                np.linspace(w / 2.0 - 200, w / 2.0 - 80, 30),
            )],
            "right": [(img_x, y) for y, img_x in zip(
                np.linspace(h * 0.6, h, 30),
                np.linspace(w / 2.0 + 80, w / 2.0 + 200, 30),
            )],
        }
        out = inst._extract_lane_polynomials_and_waypoints(pts, h, w, heading_hint_rad=+0.3)
        self.assertIsNotNone(out)
        self.assertNotEqual(out["heading_anchor_applied_rad"], 0.0)
        # target_psi=-0.3 → primer waypoint psi negativo después del anchor
        wpts = out["center_waypoints_body"]
        self.assertLess(wpts[0][2], -0.10)


class TestLateralAnchor(unittest.TestCase):
    def test_lateral_anchor_aligns_first_waypoint_with_hint(self):
        inst, h, w = _make_minimal_thread()
        pts = {
            "left": _straight_line_points(h, w, -120),
            "right": _straight_line_points(h, w, +120),
        }
        # Sin hint: el path centra ~0
        baseline = inst._extract_lane_polynomials_and_waypoints(pts, h, w)
        self.assertIsNotNone(baseline)
        baseline_y0 = baseline["center_waypoints_body"][0][1]

        # Con hint = +0.10 m (lane center 10 cm a la izquierda en body)
        out = inst._extract_lane_polynomials_and_waypoints(pts, h, w, lateral_hint_m=+0.10)
        self.assertIsNotNone(out)
        anchored_y0 = out["center_waypoints_body"][0][1]
        self.assertAlmostEqual(anchored_y0, 0.10, places=4)
        self.assertNotEqual(out["lateral_anchor_applied_m"], 0.0)
        # La forma se preserva: shift de todos los puntos por el mismo delta
        delta = anchored_y0 - baseline_y0
        for orig, anchored in zip(baseline["center_waypoints_body"], out["center_waypoints_body"]):
            self.assertAlmostEqual(anchored[1] - orig[1], delta, places=4)

    def test_lateral_anchor_skipped_when_difference_below_threshold(self):
        inst, h, w = _make_minimal_thread()
        pts = {
            "left": _straight_line_points(h, w, -120),
            "right": _straight_line_points(h, w, +120),
        }
        # Polinomio centra ~0; hint también ~0 (diff < 2 cm) → skip
        out = inst._extract_lane_polynomials_and_waypoints(pts, h, w, lateral_hint_m=0.005)
        self.assertIsNotNone(out)
        self.assertEqual(out["lateral_anchor_applied_m"], 0.0)


class TestVisualWaypointPayloadFallback(unittest.TestCase):
    def test_uses_single_line_physical_error_cm_when_debug_direct_error_is_missing(self):
        inst, h, w = _make_minimal_thread()
        captured = {}

        def _fake_extract(
            lane_side_points,
            img_h,
            img_w,
            heading_hint_rad=None,
            lateral_hint_m=None,
            single_line_target_factor=None,
        ):
            captured["heading_hint_rad"] = heading_hint_rad
            captured["lateral_hint_m"] = lateral_hint_m
            captured["single_line_target_factor"] = single_line_target_factor
            return {
                "center_waypoints_body": ((0.10, lateral_hint_m or 0.0, 0.0), (0.20, lateral_hint_m or 0.0, 0.0)),
                "left_poly_coeffs": None,
                "right_poly_coeffs": (0.0, 0.0, 1.0),
                "lane_width_m": 0.35,
                "extrapolated_side": "left",
                "samples_used": {"left": 0, "right": 12},
                "heading_anchor_applied_rad": 0.0,
                "lateral_anchor_applied_m": 0.0,
            }

        inst._extract_lane_polynomials_and_waypoints = _fake_extract
        inst._heading_error = None
        inst._last_local_lane_payload = {
            "lane_side_points": {
                "left": [],
                "right": [(406, 360)] * 12,
            },
            "frame_height": h,
            "frame_width": w,
        }
        inst._last_frame_trace = {
            "debug": {
                "measurement_mode": "single_line",
                "local_mask_guidance": {
                    "guidance_mode": "single_line_physical",
                    "error_cm": -10.3,
                },
            }
        }

        payload = inst._compute_visual_lane_waypoints_payload()

        self.assertIsNotNone(payload)
        self.assertAlmostEqual(captured["lateral_hint_m"], 0.103, places=4)
        self.assertIsNone(captured["single_line_target_factor"])

    def test_reassigns_single_line_points_to_resolved_side_and_uses_curve_factor(self):
        inst, h, w = _make_minimal_thread()
        inst.single_line_offset_factor = 0.42
        captured = {}

        def _fake_extract(
            lane_side_points,
            img_h,
            img_w,
            heading_hint_rad=None,
            lateral_hint_m=None,
            single_line_target_factor=None,
        ):
            captured["lane_side_points"] = lane_side_points
            captured["single_line_target_factor"] = single_line_target_factor
            return {
                "center_waypoints_body": ((0.10, 0.0, 0.0), (0.20, 0.0, 0.0)),
                "left_poly_coeffs": None,
                "right_poly_coeffs": (0.0, 0.0, 1.0),
                "lane_width_m": 0.35,
                "extrapolated_side": "left",
                "samples_used": {"left": 0, "right": 12},
                "heading_anchor_applied_rad": 0.0,
                "lateral_anchor_applied_m": 0.0,
            }

        inst._extract_lane_polynomials_and_waypoints = _fake_extract
        inst._heading_error = None
        inst._last_local_lane_payload = {
            "lane_side_points": {
                "left": [(406, 360)] * 12,
                "right": [],
            },
            "frame_height": h,
            "frame_width": w,
        }
        inst._last_frame_trace = {
            "debug": {
                "measurement_mode": "single_line",
                "single_line_detected_side": "left",
                "single_line_resolved_side": "right",
                "local_mask_guidance": {
                    "guidance_mode": "single_line_physical",
                    "detected_sides": ["right"],
                    "single_line_prefer_center": False,
                    "single_line_projection_debug": {
                        "single_line_curve_context": True,
                    },
                },
            }
        }

        payload = inst._compute_visual_lane_waypoints_payload()

        self.assertIsNotNone(payload)
        self.assertEqual(captured["lane_side_points"]["left"], [])
        self.assertEqual(len(captured["lane_side_points"]["right"]), 12)
        self.assertAlmostEqual(captured["single_line_target_factor"], 0.42, places=4)


if __name__ == "__main__":
    unittest.main()
