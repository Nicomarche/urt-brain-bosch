# tests/perception/tracking/test_sort_tracker.py
#
# Tests del MOTTracker (SORT-light). Validamos:
#   1. Detección sola con confianza < birth threshold no crea track.
#   2. Detección con confianza suficiente crea track con ID nuevo.
#   3. ID se mantiene cross-frame mientras la detección esté cerca.
#   4. Track expira tras max_age_frames sin matches.
#   5. Track NO es emitido (no aparece en la salida) hasta `min_hits`.
#   6. Detección con position_world_xy = None se ignora.
#   7. Asociación correcta cuando hay 2 peatones moviéndose en paralelo.
#   8. reset() limpia tracks pero mantiene IDs únicos en el futuro.

from __future__ import annotations

import pytest

from src.core.types.perception import DetectedObject
from src.core.types.pose import PoseEstimate
from src.perception.tracking.sort_tracker import MOTTracker


def _det(
    x: float,
    y: float,
    *,
    cls: int = 0,
    name: str = "pedestrian",
    conf: float = 0.9,
    t: float = 0.0,
) -> DetectedObject:
    return DetectedObject(
        timestamp=t,
        class_id=cls,
        class_name=name,
        confidence=conf,
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        position_world_xy=(x, y),
    )


def _pose() -> PoseEstimate:
    return PoseEstimate()


# ---------- birth ----------------------------------------------------------


def test_low_confidence_detection_does_not_birth() -> None:
    t = MOTTracker(min_confidence_to_birth=0.5)
    out = t.step([_det(1.0, 0.0, conf=0.3)], dt=0.1, ego_pose=_pose())
    assert out == []
    assert t.n_tracks == 0


def test_high_confidence_detection_creates_track() -> None:
    t = MOTTracker(min_confidence_to_birth=0.5, min_hits_to_emit=1)
    out = t.step([_det(1.0, 0.0, conf=0.9)], dt=0.1, ego_pose=_pose())
    assert len(out) == 1
    assert out[0].track_id >= 1
    assert t.n_tracks == 1


def test_detection_without_world_position_ignored() -> None:
    t = MOTTracker()
    bad = DetectedObject(
        timestamp=0.0,
        class_id=0,
        class_name="pedestrian",
        confidence=0.9,
        bbox_xyxy=(0, 0, 1, 1),
        position_world_xy=None,
    )
    t.step([bad], dt=0.1, ego_pose=_pose())
    assert t.n_tracks == 0


# ---------- persistence (no ID switch) -------------------------------------


def test_id_persists_across_frames_for_moving_target() -> None:
    """Un peatón que se desplaza ~1 m/s mantiene el mismo ID."""
    t = MOTTracker(min_confidence_to_birth=0.5, min_hits_to_emit=1)
    # Frame 0
    out0 = t.step([_det(0.0, 0.0, t=0.0)], dt=0.1, ego_pose=_pose())
    initial_id = out0[0].track_id
    # Frames 1..5: el peatón se mueve +0.1 m/step en x.
    for k in range(1, 6):
        out = t.step([_det(0.1 * k, 0.0, t=0.1 * k)], dt=0.1, ego_pose=_pose())
        ids = [o.track_id for o in out]
        assert initial_id in ids


def test_two_parallel_targets_get_distinct_ids() -> None:
    """Dos peatones bien separados nunca deberían intercambiar IDs."""
    t = MOTTracker(min_confidence_to_birth=0.5, min_hits_to_emit=1)
    # Frame 0
    out0 = t.step(
        [_det(0.0, 0.0, t=0.0), _det(5.0, 0.0, t=0.0)], dt=0.1, ego_pose=_pose()
    )
    assert len({o.track_id for o in out0}) == 2
    id_a = next(o.track_id for o in out0 if abs(o.position_world_xy[0]) < 1.0)
    id_b = next(o.track_id for o in out0 if abs(o.position_world_xy[0] - 5.0) < 1.0)
    # Frames siguientes: ambos avanzan +0.1 en x.
    for k in range(1, 5):
        out = t.step(
            [_det(0.1 * k, 0.0, t=0.1 * k), _det(5.0 + 0.1 * k, 0.0, t=0.1 * k)],
            dt=0.1,
            ego_pose=_pose(),
        )
        a = next(o for o in out if abs(o.position_world_xy[0] - 0.1 * k) < 1.0)
        b = next(o for o in out if abs(o.position_world_xy[0] - (5.0 + 0.1 * k)) < 1.0)
        assert a.track_id == id_a
        assert b.track_id == id_b


# ---------- death ----------------------------------------------------------


def test_track_dies_after_max_age_without_match() -> None:
    t = MOTTracker(min_confidence_to_birth=0.5, min_hits_to_emit=1, max_age_frames=3)
    t.step([_det(0.0, 0.0)], dt=0.1, ego_pose=_pose())
    assert t.n_tracks == 1
    # 4 frames sin detecciones → > max_age_frames → muere.
    for _ in range(4):
        t.step([], dt=0.1, ego_pose=_pose())
    assert t.n_tracks == 0


# ---------- min_hits gate --------------------------------------------------


def test_track_not_emitted_until_min_hits() -> None:
    """Con min_hits=3, las primeras dos detecciones de un track no se reportan."""
    t = MOTTracker(min_confidence_to_birth=0.5, min_hits_to_emit=3)
    out0 = t.step([_det(0.0, 0.0, t=0.0)], dt=0.1, ego_pose=_pose())
    assert out0 == []  # hits=1
    out1 = t.step([_det(0.05, 0.0, t=0.1)], dt=0.1, ego_pose=_pose())
    assert out1 == []  # hits=2
    out2 = t.step([_det(0.1, 0.0, t=0.2)], dt=0.1, ego_pose=_pose())
    assert len(out2) == 1  # hits=3


# ---------- reset ----------------------------------------------------------


def test_reset_clears_tracks_but_preserves_id_uniqueness() -> None:
    t = MOTTracker(min_confidence_to_birth=0.5, min_hits_to_emit=1)
    out0 = t.step([_det(0.0, 0.0)], dt=0.1, ego_pose=_pose())
    id0 = out0[0].track_id
    t.reset()
    assert t.n_tracks == 0
    out1 = t.step([_det(0.0, 0.0)], dt=0.1, ego_pose=_pose())
    id1 = out1[0].track_id
    assert id1 != id0  # IDs únicos cross-reset
