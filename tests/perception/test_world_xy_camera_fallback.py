"""Tests del helper ``_project_detection_to_world`` (plan TANDA 0.2).

ANTES del fix: la proyección al frame mundo se hacía sólo cuando la
fuente de distancia era LiDAR. Si LiDAR no llegaba (timeout, sector
ocluido), el TrackedObject del peatón quedaba con ``position_world_xy =
None`` y el scenario `PedestrianYield` lo descartaba silenciosamente.

DESPUÉS: la proyección depende sólo de tener una ``distance_cm`` válida
y la pose ego — la fuente (LiDAR vs cámara pinhole) ya no importa. La
cámara, con la calibración correcta (`_estimate_sign_distance_cm`), provee
una estimación razonable para objetos a nivel del piso (peatones, autos,
signs).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.types.pose import Pose2D, PoseEstimate
from src.hardware.camera.threads.threadLocalPerception import threadLocalPerception


def _make_thread_with_pose(ego_x=0.0, ego_y=0.0, ego_yaw=0.0):
    """Instanciar el thread mínimamente — sólo lo que toca el helper."""
    inst = threadLocalPerception.__new__(threadLocalPerception)
    pose = PoseEstimate(
        timestamp=0.0,
        fused_pose=Pose2D(x=ego_x, y=ego_y, yaw=ego_yaw),
        speed_mps=0.0,
    )
    inst.pose_estimate_buffer = MagicMock()
    inst.pose_estimate_buffer.read_latest.return_value = pose
    return inst


def test_lidar_distance_projects_to_world():
    """Caso baseline: distancia LiDAR + pose ego → world_xy delante del auto."""
    inst = _make_thread_with_pose(ego_x=0.0, ego_y=0.0, ego_yaw=0.0)
    # bbox en el centro de la imagen (x_center=0.5) → angle=0
    box = [0.4, 0.45, 0.9, 0.55]  # [y1, x1, y2, x2] normalizado
    result = inst._project_detection_to_world(box, distance_cm=200.0)  # 2m
    assert result is not None
    x, y = result
    # Ego en (0,0) mirando +x, target a 2m delante → world ≈ (2, 0)
    assert abs(x - 2.0) < 1e-3
    assert abs(y) < 1e-3


def test_camera_distance_projects_to_world_same_result():
    """Plan TANDA 0.2: la fuente no importa, sólo la distancia.

    Antes ``_project_lidar_detection_to_world`` se llamaba sólo con
    ``distance_source=="lidar"``. Ahora ``_project_detection_to_world`` no
    distingue — la prueba documenta el contrato y previene la regresión.
    """
    inst = _make_thread_with_pose(ego_x=0.0, ego_y=0.0, ego_yaw=0.0)
    box = [0.4, 0.45, 0.9, 0.55]
    # Mismo bbox, misma distancia, sólo "fuente conceptual" distinta.
    # El método ya no chequea fuente — debe dar idéntico al caso LiDAR.
    result_camera = inst._project_detection_to_world(box, distance_cm=200.0)
    result_lidar = inst._project_detection_to_world(box, distance_cm=200.0)
    assert result_camera == result_lidar
    assert result_camera is not None
    assert abs(result_camera[0] - 2.0) < 1e-3


def test_none_distance_returns_none():
    """Sin distancia confiable, no inventamos posición — el scenario debe
    saber que no hay observación accionable."""
    inst = _make_thread_with_pose()
    box = [0.4, 0.45, 0.9, 0.55]
    assert inst._project_detection_to_world(box, distance_cm=None) is None


def test_off_center_bbox_projects_to_side():
    """Bbox al borde derecho de la imagen → world target a la derecha del ego."""
    inst = _make_thread_with_pose(ego_x=0.0, ego_y=0.0, ego_yaw=0.0)
    # x_center=0.85 (cerca del borde derecho) → angle negativo → body_y negativo
    box = [0.4, 0.80, 0.9, 0.90]
    result = inst._project_detection_to_world(box, distance_cm=200.0)
    assert result is not None
    x, y = result
    # Con yaw=0 (mirando +x) y angle negativo, world_y debe ser negativo
    # (target a la "derecha" del ego en convención right-hand frame).
    assert y < 0.0


def test_ego_yaw_rotation_applies():
    """Con yaw=90°, target delante del auto = world_y positivo (auto mirando +y)."""
    inst = _make_thread_with_pose(ego_x=0.0, ego_y=0.0, ego_yaw=math.pi / 2)
    box = [0.4, 0.45, 0.9, 0.55]  # centro → angle≈0
    result = inst._project_detection_to_world(box, distance_cm=100.0)  # 1m
    assert result is not None
    x, y = result
    # Auto en origen mirando +y, target 1m delante → world ≈ (0, 1)
    assert abs(x) < 1e-3
    assert abs(y - 1.0) < 1e-3


def test_pose_unavailable_returns_none():
    """Sin pose válida no podemos transformar al frame mundo."""
    inst = threadLocalPerception.__new__(threadLocalPerception)
    inst.pose_estimate_buffer = MagicMock()
    inst.pose_estimate_buffer.read_latest.return_value = None
    box = [0.4, 0.45, 0.9, 0.55]
    assert inst._project_detection_to_world(box, distance_cm=200.0) is None
