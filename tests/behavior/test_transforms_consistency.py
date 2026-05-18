"""Plan TANDA 1.2 — Consistencia de transforms 2D.

Los call-sites legacy ``_local_to_world`` (trajectory_builder) y
``_body_to_world_xy`` (path_optimizer) ya son wrappers sobre
``src.core.frames`` para mantener single-source-of-truth. Este test:

  1. Confirma byte-a-byte que ambos wrappers reproducen exactamente la
     salida de las funciones canónicas en ``core.frames`` para 20 inputs
     pseudo-aleatorios con seed fijo.
  2. Documenta la equivalencia para que la migración nunca se desande:
     si un futuro refactor rompe el contrato, esta suite explota antes
     de que el robot se vaya al carril contrario.

NO tocar los wrappers internos (son thin y mantienen retro-compat de
imports relativos). Si en el futuro se eliminan, eliminar también este
archivo.
"""

from __future__ import annotations

import math

import numpy as np

from src.behavior.path_optimizer import _body_to_world_xy
from src.behavior.trajectory_builder import _local_to_world
from src.core.frames import body_to_world, body_to_world_yflip


def _seeded_inputs(n: int = 20):
    rng = np.random.default_rng(seed=20260518)
    for _ in range(n):
        bx = float(rng.uniform(-2.0, 2.0))
        by = float(rng.uniform(-2.0, 2.0))
        ox = float(rng.uniform(-50.0, 50.0))
        oy = float(rng.uniform(-50.0, 50.0))
        yaw = float(rng.uniform(-math.pi, math.pi))
        yield bx, by, ox, oy, yaw


def test_local_to_world_matches_body_to_world():
    """Wrapper legacy en trajectory_builder vs core.frames.body_to_world."""
    for bx, by, ox, oy, yaw in _seeded_inputs():
        legacy = _local_to_world(
            np.array([bx, by], dtype=float),
            origin_xy=np.array([ox, oy], dtype=float),
            yaw_rad=yaw,
        )
        canonical = body_to_world(
            (bx, by),
            origin_xy=(ox, oy),
            yaw_rad=yaw,
        )
        # Asserción byte-a-byte (no tolerancia): el wrapper SOLO delega.
        np.testing.assert_array_equal(legacy, canonical)


def test_body_to_world_xy_matches_yflip():
    """Wrapper legacy en path_optimizer vs core.frames.body_to_world_yflip."""
    for bx, by, ox, oy, yaw in _seeded_inputs():
        legacy = _body_to_world_xy(
            x_fwd_m=bx,
            y_left_m=by,
            pose_x=ox,
            pose_y=oy,
            pose_yaw=yaw,
        )
        canonical = body_to_world_yflip(
            x_fwd_m=bx,
            y_left_m=by,
            pose_x=ox,
            pose_y=oy,
            pose_yaw=yaw,
        )
        np.testing.assert_array_equal(legacy, canonical)


def test_yflip_differs_from_yup_when_yleft_nonzero():
    """Las dos convenciones DEBEN diferir cuando hay componente lateral —
    si convergen para todo input es porque alguien rompió la diferencia
    semántica."""
    # y_left=0 → ambas dan el mismo resultado (eje x).
    same = body_to_world((1.0, 0.0), origin_xy=(0.0, 0.0), yaw_rad=0.0)
    flip = body_to_world_yflip(x_fwd_m=1.0, y_left_m=0.0, pose_x=0.0, pose_y=0.0, pose_yaw=0.0)
    np.testing.assert_array_almost_equal(same, flip)

    # y_left≠0 → DEBEN diferir (sign de la componente lateral).
    yup = body_to_world((1.0, 1.0), origin_xy=(0.0, 0.0), yaw_rad=0.0)
    ydown = body_to_world_yflip(x_fwd_m=1.0, y_left_m=1.0, pose_x=0.0, pose_y=0.0, pose_yaw=0.0)
    assert yup[1] != ydown[1]
