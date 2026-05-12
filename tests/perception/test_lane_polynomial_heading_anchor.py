# tests/perception/test_lane_polynomial_heading_anchor.py
#
# Estilo urt-ref: los anchors heading/lateral fueron eliminados. El path
# visual sale 1:1 del polinomio fitteado y nunca se rota o traslada por
# `heading_hint_rad`/`lateral_hint_m`. Estos tests verifican explícitamente
# que llegada cualquier hint, el path queda sin modificar.

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

    def test_heading_hint_is_ignored_in_ref_style(self):
        # heading_hint_rad llega pero NO se aplica: el polinomio es el path.
        # Si el detector subrota, se corrige mejorando los puntos del modelo
        # o la calibración BEV, no con anchors legacy ruidosos.
        inst, h, w = _make_minimal_thread()
        pts = {
            "left": _straight_line_points(h, w, -120),
            "right": _straight_line_points(h, w, +120),
        }
        baseline = inst._extract_lane_polynomials_and_waypoints(pts, h, w, heading_hint_rad=None)
        out = inst._extract_lane_polynomials_and_waypoints(pts, h, w, heading_hint_rad=0.4)
        self.assertIsNotNone(out)
        self.assertEqual(out["heading_anchor_applied_rad"], 0.0)
        # Con o sin hint, los waypoints deben ser idénticos: el path es el
        # polinomio, no se rota.
        for base_wp, out_wp in zip(
            baseline["center_waypoints_body"], out["center_waypoints_body"]
        ):
            self.assertAlmostEqual(base_wp[0], out_wp[0], places=6)
            self.assertAlmostEqual(base_wp[1], out_wp[1], places=6)
            self.assertAlmostEqual(base_wp[2], out_wp[2], places=6)


class TestLateralAnchor(unittest.TestCase):
    def test_lateral_hint_is_ignored_in_ref_style(self):
        # lateral_hint_m llega pero NO se aplica.
        inst, h, w = _make_minimal_thread()
        pts = {
            "left": _straight_line_points(h, w, -120),
            "right": _straight_line_points(h, w, +120),
        }
        # Sin hint
        baseline = inst._extract_lane_polynomials_and_waypoints(pts, h, w)
        self.assertIsNotNone(baseline)
        baseline_y0 = baseline["center_waypoints_body"][0][1]

        # Con hint = +0.10 m: NO debe trasladarse el path.
        out = inst._extract_lane_polynomials_and_waypoints(pts, h, w, lateral_hint_m=+0.10)
        self.assertIsNotNone(out)
        self.assertEqual(out["lateral_anchor_applied_m"], 0.0)
        anchored_y0 = out["center_waypoints_body"][0][1]
        self.assertAlmostEqual(anchored_y0, baseline_y0, places=4)


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
