"""Tests de transforms 2D — verifican que la migración no rompe signos.

La sutileza importante: `body_to_world` (matemática y-up) NO es lo mismo
que `body_to_world_yflip` (convención path_optimizer y-down). Cada
convención tiene tests independientes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.frames import (
    FRAME_BASE_LINK,
    FRAME_CAMERA,
    FRAME_MAP,
    body_to_world,
    body_to_world_yflip,
    camera_pixel_to_body,
    world_to_body,
)


# ─── body_to_world (matemática, y-up) ──────────────────────────────────────


def test_body_to_world_identity_at_origin_yaw_zero():
    out = body_to_world((1.0, 2.0), origin_xy=(0.0, 0.0), yaw_rad=0.0)
    np.testing.assert_allclose(out, [1.0, 2.0])


def test_body_to_world_yaw_90_rotates_forward_to_left():
    """Yaw=π/2: forward del body queda hacia +y del mapa."""
    out = body_to_world((1.0, 0.0), origin_xy=(0.0, 0.0), yaw_rad=math.pi / 2)
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-9)


def test_body_to_world_yaw_180_inverts_x():
    out = body_to_world((1.0, 0.0), origin_xy=(0.0, 0.0), yaw_rad=math.pi)
    np.testing.assert_allclose(out, [-1.0, 0.0], atol=1e-9)


def test_body_to_world_translation_adds_origin():
    out = body_to_world((0.0, 0.0), origin_xy=(5.0, 7.0), yaw_rad=0.0)
    np.testing.assert_allclose(out, [5.0, 7.0])


def test_body_to_world_full_compose():
    """origin (5,7) + yaw 90° rotando body (1,0) → world (5, 8)."""
    out = body_to_world((1.0, 0.0), origin_xy=(5.0, 7.0), yaw_rad=math.pi / 2)
    np.testing.assert_allclose(out, [5.0, 8.0], atol=1e-9)


def test_world_to_body_is_inverse_of_body_to_world():
    pose = np.array([3.0, 4.0])
    yaw = 0.7
    body_pt = np.array([2.5, -1.3])
    world = body_to_world(body_pt, origin_xy=pose, yaw_rad=yaw)
    back = world_to_body(world, origin_xy=pose, yaw_rad=yaw)
    np.testing.assert_allclose(back, body_pt, atol=1e-9)


# ─── body_to_world_yflip (convención path_optimizer, y-down) ──────────────


def test_yflip_matches_legacy_path_optimizer_formula():
    """Verifica byte-a-byte equivalencia con la fórmula legacy.

    `_body_to_world_xy` en path_optimizer.py:
        world_x = pose_x + cos*x_fwd + sin*y_left
        world_y = pose_y + sin*x_fwd - cos*y_left
    """
    pose_x, pose_y, pose_yaw = 10.0, 20.0, 0.3
    x_fwd, y_left = 1.5, -0.4
    cos_y = math.cos(pose_yaw)
    sin_y = math.sin(pose_yaw)
    expected = np.array(
        [
            pose_x + cos_y * x_fwd + sin_y * y_left,
            pose_y + sin_y * x_fwd - cos_y * y_left,
        ]
    )
    out = body_to_world_yflip(
        x_fwd_m=x_fwd, y_left_m=y_left, pose_x=pose_x, pose_y=pose_y, pose_yaw=pose_yaw
    )
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_yflip_yaw_zero_forward_along_x():
    out = body_to_world_yflip(x_fwd_m=2.0, y_left_m=0.0, pose_x=0.0, pose_y=0.0, pose_yaw=0.0)
    np.testing.assert_allclose(out, [2.0, 0.0])


def test_yflip_distinct_from_yup_when_y_nonzero():
    """Para body_y!=0, las dos convenciones difieren en el signo Y."""
    a = body_to_world((1.0, 1.0), origin_xy=(0.0, 0.0), yaw_rad=0.0)  # → (1, 1)
    b = body_to_world_yflip(x_fwd_m=1.0, y_left_m=1.0, pose_x=0.0, pose_y=0.0, pose_yaw=0.0)
    # b → (1, -1) en yflip, demostrando que NO son intercambiables
    np.testing.assert_allclose(a, [1.0, 1.0])
    np.testing.assert_allclose(b, [1.0, -1.0])


# ─── camera_pixel_to_body ──────────────────────────────────────────────────


def test_camera_pixel_at_cy_with_pitch_gives_finite_x():
    """Pixel justo en el centro óptico + pitch razonable → distancia finita."""
    x_fwd, y_left = camera_pixel_to_body(
        u_px=320.0, v_px=240.0, fy=300.0, cy=240.0, pitch_rad=0.2, height_m=0.2
    )
    assert math.isfinite(x_fwd) and x_fwd > 0.0
    assert y_left == 0.0


def test_camera_pixel_above_horizon_returns_inf():
    """Pixel muy arriba + pitch ~0 → rayo apunta arriba → inf."""
    x_fwd, _ = camera_pixel_to_body(
        u_px=320.0, v_px=10.0, fy=300.0, cy=240.0, pitch_rad=0.0, height_m=0.2
    )
    assert math.isinf(x_fwd)


def test_camera_pixel_lateral_offset():
    """Pixel desviado de cx → componente lateral no-cero."""
    _, y_left = camera_pixel_to_body(
        u_px=400.0,
        v_px=300.0,
        fy=300.0,
        cy=240.0,
        pitch_rad=0.2,
        height_m=0.2,
        fx=300.0,
        cx=320.0,
    )
    assert abs(y_left) > 0.0


# ─── frame string constants ────────────────────────────────────────────────


def test_frame_constants_match_header_module():
    """Las constantes string deben ser idénticas a las de header.py para
    permitir comparación cross-module sin discrepancias."""
    from src.core.messaging.header import (
        FRAME_BASE_LINK as HDR_BASE,
        FRAME_CAMERA as HDR_CAM,
        FRAME_MAP as HDR_MAP,
    )

    assert FRAME_MAP == HDR_MAP == "map"
    assert FRAME_BASE_LINK == HDR_BASE == "base_link"
    assert FRAME_CAMERA == HDR_CAM == "camera"
