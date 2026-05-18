"""Transforms 2D y constantes de frames del repositorio.

Single source of truth para las dos convenciones de yaw que conviven en
el código:

  * **Matemática (y-up)** — la canónica. Usada por `trajectory_builder`,
    `pose_estimator`, `behavior_planner`. Rotación 2D estándar:

        [world_x]   [cos(ψ) -sin(ψ)] [body_x]   [origin_x]
        [world_y] = [sin(ψ)  cos(ψ)] [body_y] + [origin_y]

  * **Imagen (y-down)** — heredada de `path_optimizer` que históricamente
    venía de coordenadas de imagen donde la fila Y crece hacia abajo. La
    transformación efectiva niega el sign de la componente lateral:

        world_x = origin_x + cos(ψ) * body_x + sin(ψ) * body_y_left
        world_y = origin_y + sin(ψ) * body_x - cos(ψ) * body_y_left

Mantenemos AMBAS convenciones explícitas para no romper signos en la
migración. Cada call-site importa la función adecuada para su frame.

Frames disponibles (constantes string):
  * ``FRAME_MAP`` — frame plano fijo del mundo (Lanelet2 OSM proyectado).
  * ``FRAME_BASE_LINK`` — chasis del auto.
  * ``FRAME_CAMERA`` — cámara montada (X forward, Y left, Z up).
  * ``FRAME_LIDAR`` — LiDAR montado, mismo eje que base_link.
"""

from __future__ import annotations

import math

import numpy as np

# Aliases re-exportados desde header.py para que call-sites importen los
# string-IDs desde un único lugar.
from src.core.messaging.header import (  # noqa: F401
    FRAME_BASE_LINK,
    FRAME_CAMERA,
    FRAME_LIDAR,
    FRAME_MAP,
)


def body_to_world(
    body_xy: np.ndarray | tuple[float, float] | list[float],
    *,
    origin_xy: np.ndarray | tuple[float, float],
    yaw_rad: float,
) -> np.ndarray:
    """Convención matemática (y-up). Reemplaza `_local_to_world`.

    Rota un punto en el frame del chasis a coordenadas del mapa usando la
    matriz de rotación 2D estándar.

    Args:
        body_xy: Punto ``(x_fwd, y_left)`` en el frame del chasis.
        origin_xy: Posición del origen del chasis en el frame del mapa.
        yaw_rad: Heading del chasis (rad, CCW positivo).

    Returns:
        ``np.ndarray`` shape (2,) con el punto en el frame del mapa.
    """
    cos_y = math.cos(float(yaw_rad))
    sin_y = math.sin(float(yaw_rad))
    bx = float(body_xy[0])
    by = float(body_xy[1])
    ox = float(origin_xy[0])
    oy = float(origin_xy[1])
    return np.array(
        [
            ox + cos_y * bx - sin_y * by,
            oy + sin_y * bx + cos_y * by,
        ],
        dtype=float,
    )


def world_to_body(
    world_xy: np.ndarray | tuple[float, float] | list[float],
    *,
    origin_xy: np.ndarray | tuple[float, float],
    yaw_rad: float,
) -> np.ndarray:
    """Inverso de `body_to_world` (convención matemática y-up)."""
    cos_y = math.cos(float(yaw_rad))
    sin_y = math.sin(float(yaw_rad))
    dx = float(world_xy[0]) - float(origin_xy[0])
    dy = float(world_xy[1]) - float(origin_xy[1])
    return np.array(
        [
            cos_y * dx + sin_y * dy,
            -sin_y * dx + cos_y * dy,
        ],
        dtype=float,
    )


def body_to_world_yflip(
    *,
    x_fwd_m: float,
    y_left_m: float,
    pose_x: float,
    pose_y: float,
    pose_yaw: float,
) -> np.ndarray:
    """Convención y-down heredada de `path_optimizer._body_to_world_xy`.

    El sign de la componente lateral viene negado respecto a la
    convención matemática porque históricamente los mapas raster del
    proyecto BFMC tenían fila Y creciendo hacia abajo.

    Mantener fórmula byte-a-byte idéntica para no romper signos en
    `path_optimizer` durante la migración.

    Args:
        x_fwd_m: Componente "forward" en frame chasis.
        y_left_m: Componente "left" en frame chasis (convención y-down).
        pose_x: Origen X en frame mundo.
        pose_y: Origen Y en frame mundo.
        pose_yaw: Heading (rad).

    Returns:
        ``np.ndarray`` shape (2,) en el frame mundo (y-down).
    """
    cos_y = math.cos(float(pose_yaw))
    sin_y = math.sin(float(pose_yaw))
    world_x = float(pose_x) + (cos_y * float(x_fwd_m)) + (sin_y * float(y_left_m))
    world_y = float(pose_y) + (sin_y * float(x_fwd_m)) - (cos_y * float(y_left_m))
    return np.array([world_x, world_y], dtype=float)


def camera_pixel_to_body(
    *,
    u_px: float,
    v_px: float,
    fy: float,
    cy: float,
    pitch_rad: float,
    height_m: float,
    fx: float | None = None,
    cx: float | None = None,
) -> tuple[float, float]:
    """Proyecta un pixel del frame cámara al frame base_link del chasis.

    Asume cámara montada a altura ``height_m`` sobre el suelo y mirando
    hacia adelante con pitch positivo hacia abajo. El pixel ``(u, v)``
    se intersecta con el plano del suelo.

    Args:
        u_px: Coordenada horizontal en píxeles (columnas).
        v_px: Coordenada vertical en píxeles (filas).
        fy: Distancia focal en píxeles vertical.
        cy: Centro óptico vertical en píxeles.
        pitch_rad: Pitch de la cámara (rad, positivo hacia abajo).
        height_m: Altura de la cámara sobre el suelo (m).
        fx: Focal horizontal (default = fy si None).
        cx: Centro óptico horizontal (default = image_center; sólo usado
            para componente lateral, NO afecta x_fwd).

    Returns:
        Tupla ``(x_fwd_m, y_left_m)`` en frame base_link.

    Notas:
        Si el rayo no intersecta el suelo (apunta por encima del horizonte
        cuando pitch~0), devuelve ``(+inf, 0.0)`` — caller debe filtrar.
    """
    if fx is None:
        fx = fy

    # Ángulo del rayo en el plano vertical (positivo hacia abajo).
    ray_pitch = math.atan2(float(v_px) - float(cy), float(fy)) + float(pitch_rad)
    if ray_pitch <= 1e-6:
        return float("inf"), 0.0

    x_fwd = float(height_m) / math.tan(ray_pitch)

    if cx is not None:
        ray_yaw = math.atan2(float(u_px) - float(cx), float(fx))
        y_left = -x_fwd * math.tan(ray_yaw)
    else:
        y_left = 0.0

    return x_fwd, y_left


__all__ = [
    "FRAME_MAP",
    "FRAME_BASE_LINK",
    "FRAME_CAMERA",
    "FRAME_LIDAR",
    "body_to_world",
    "world_to_body",
    "body_to_world_yflip",
    "camera_pixel_to_body",
]
