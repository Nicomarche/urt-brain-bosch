# tests/perception/tracking/test_kalman_track.py
#
# Tests del TrackState (KF constant-velocity 2D). Validamos:
#   1. Posición inicial coincide con la observación.
#   2. Velocidad inicial = (0, 0).
#   3. Tras 2 updates con desplazamiento conocido, vx/vy se aproximan a
#      la velocidad real.
#   4. predict() avanza el estado proporcional a vx*dt, vy*dt.
#   5. Track sin updates expira (`is_stale`) tras max_age_frames.
#   6. is_confirmed se activa al alcanzar min_hits.
#   7. Covarianza posicional se reduce al recibir observaciones.

from __future__ import annotations

import numpy as np
import pytest

from src.core.types.perception import DetectedObject
from src.perception.tracking.kalman_track import TrackState


def _det(x: float, y: float, t: float = 0.0, conf: float = 0.9) -> DetectedObject:
    return DetectedObject(
        timestamp=t,
        class_id=0,
        class_name="pedestrian",
        confidence=conf,
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        position_world_xy=(x, y),
    )


def test_track_initial_position_matches_observation() -> None:
    tr = TrackState(
        track_id=1,
        initial_xy=(2.0, 1.0),
        class_id=0,
        class_name="pedestrian",
        confidence=0.9,
        timestamp=0.0,
    )
    assert tr.position() == pytest.approx((2.0, 1.0))


def test_track_initial_velocity_is_zero() -> None:
    tr = TrackState(1, (0.0, 0.0), 0, "p", 0.9, 0.0)
    vx, vy = tr.velocity()
    assert vx == pytest.approx(0.0)
    assert vy == pytest.approx(0.0)


def test_track_velocity_estimated_after_observations() -> None:
    """Si el peatón se mueve a ~1 m/s en x, el KF debería estimar vx ≈ 1."""
    tr = TrackState(1, (0.0, 0.0), 0, "p", 0.9, 0.0)
    dt = 0.1
    # 10 frames con +0.1 m/step en x → 1 m/s.
    for k in range(1, 11):
        tr.predict(dt)
        tr.update(_det(0.1 * k, 0.0, t=dt * k))
    vx, vy = tr.velocity()
    assert vx == pytest.approx(1.0, abs=0.2)
    assert abs(vy) < 0.2


def test_track_predict_advances_position() -> None:
    tr = TrackState(1, (0.0, 0.0), 0, "p", 0.9, 0.0)
    dt = 0.1
    # Inyectar velocidad observada.
    for k in range(1, 6):
        tr.predict(dt)
        tr.update(_det(0.5 * k, 0.0, t=dt * k))
    pos_before = tr.position()
    tr.predict(dt)
    pos_after = tr.position()
    # Después de un predict sin update, la posición debería avanzar
    # aproximadamente vx*dt.
    assert pos_after[0] > pos_before[0]


def test_track_age_increments_on_predict() -> None:
    tr = TrackState(1, (0.0, 0.0), 0, "p", 0.9, 0.0)
    age_initial = tr.age
    tr.predict(0.1)
    tr.predict(0.1)
    assert tr.age == age_initial + 2


def test_track_time_since_update_resets_on_match() -> None:
    tr = TrackState(1, (0.0, 0.0), 0, "p", 0.9, 0.0)
    tr.predict(0.1)
    tr.predict(0.1)
    assert tr.time_since_update == 2
    tr.update(_det(0.0, 0.0, t=0.2))
    assert tr.time_since_update == 0


def test_track_is_stale_after_max_age() -> None:
    tr = TrackState(1, (0.0, 0.0), 0, "p", 0.9, 0.0)
    for _ in range(5):
        tr.predict(0.1)
    assert tr.is_stale(max_age_frames=4)
    assert not tr.is_stale(max_age_frames=10)


def test_track_is_confirmed_after_min_hits() -> None:
    tr = TrackState(1, (0.0, 0.0), 0, "p", 0.9, 0.0)
    # hits arranca en 1 (la detección que creó el track).
    assert tr.hits == 1
    tr.predict(0.1)
    tr.update(_det(0.1, 0.0, t=0.1))
    tr.predict(0.1)
    tr.update(_det(0.2, 0.0, t=0.2))
    assert tr.hits == 3
    assert tr.is_confirmed(min_hits=3)
    assert not tr.is_confirmed(min_hits=5)


def test_track_covariance_shrinks_with_updates() -> None:
    tr = TrackState(1, (0.0, 0.0), 0, "p", 0.9, 0.0)
    p_initial = float(np.trace(tr.position_covariance()))
    tr.predict(0.1)
    tr.update(_det(0.0, 0.0, t=0.1))
    p_after_update = float(np.trace(tr.position_covariance()))
    # La covarianza posicional baja al incorporar una observación.
    assert p_after_update < p_initial
