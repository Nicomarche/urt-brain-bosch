# src/behavior/trajectory_builder.py
#
# Función pura: dado el `LaneletMap`, la lanelet actual del ego y la
# velocidad target, construye un `target_path` (N+1, 3) de waypoints
# uniformemente espaciados por arc-length.
#
# ¿Por qué Frenet (proyección sobre centerline)?
#   El MPC que ejecuta este plan opera en frame inercial pero referencia
#   estados en el espacio (x, y, psi). La forma simple de generar
#   referencias para el horizonte: caminar por el grafo de lanelets,
#   acumular puntos de centerline, resamplear a paso constante de
#   `target_speed * dt`. La heading psi en cada sample se computa como
#   atan2 del segmento adyacente.
#
# Performance:
#   La función es O(K) donde K = puntos de centerline visitados. Para
#   horizonte N=20 a 1.5 m/s con dt=0.05 s eso son 1.5 m de path → 8-15
#   centerline points. Trivial.
#
# Decisiones:
#   - Si la lanelet actual no alcanza para cubrir el horizonte, seguimos
#     la PRIMERA sucesora. Si hay rama (intersección) el `route.next_lanelet_ids`
#     guía qué sucesora elegir; fallback es la primera del listado.
#   - Si en algún punto NO hay sucesor disponible (terminal), el path
#     se "estira" con la última posición + última psi para que el MPC
#     reciba un plan completo. El velocity_overlay decidirá si frenar.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.routing.lanelet.lanelet_map import LaneletMap


def build_target_path(
    lanelet_map: "LaneletMap",
    start_lanelet_id: str,
    start_xy: tuple[float, float],
    target_speed_mps: float,
    horizon_n: int,
    dt: float,
    next_lanelet_hint_ids: tuple[str, ...] = (),
) -> np.ndarray:
    """Construye target_path (N+1, 3) caminando por las lanelets sucesoras.

    Args:
      lanelet_map: HD map cargado.
      start_lanelet_id: lanelet donde está el ego actualmente (de
        `LaneletMap.at_pose`).
      start_xy: posición actual del ego (x, y) en frame del mapa. Usada
        para arrancar el muestreo desde el punto más cercano de la
        centerline (no desde el inicio de la lanelet).
      target_speed_mps: velocidad de referencia. Define el espaciado
        del muestreo: `step_arc = target_speed * dt`. Si == 0, usamos
        un default de 0.10 m para que no colapse el muestreo.
      horizon_n: cantidad de pasos de control. El path tiene N+1 filas.
      dt: duración del paso (s).
      next_lanelet_hint_ids: IDs de sucesoras preferidas (orden
        topológico desde la ruta planificada). Usadas para desempate
        cuando una lanelet tiene múltiples sucesoras.

    Returns:
      ndarray (N+1, 3) con (x, y, psi) en cada fila. psi en radianes.
    """
    if horizon_n <= 0:
        # Defensa: devolvemos un path mínimo válido (1 punto). El
        # MotionController debería marcar invalid pero no crashear.
        return np.array([[start_xy[0], start_xy[1], 0.0]], dtype=float)

    step_arc = max(0.10, float(target_speed_mps) * float(dt))
    total_arc_needed = step_arc * horizon_n

    # 1. Construimos la polyline de centerlines concatenadas, recortada
    #    al arc-length necesario.
    polyline = _walk_centerlines(
        lanelet_map=lanelet_map,
        start_lanelet_id=start_lanelet_id,
        start_xy=start_xy,
        max_arc_m=total_arc_needed * 1.5,  # cushion para que el resample no se quede corto
        next_lanelet_hint_ids=next_lanelet_hint_ids,
    )
    if polyline.shape[0] < 2:
        # No tenemos centerline utilizable — devolvemos N+1 copias del
        # punto actual con psi=0. El velocity_overlay verá que no avanza
        # y el planner debería marcar invalid.
        return np.tile(np.array([start_xy[0], start_xy[1], 0.0]), (horizon_n + 1, 1))

    # 2. Resampleamos a paso constante de step_arc.
    sampled_xy = _resample_polyline(polyline, step_arc=step_arc, n_samples=horizon_n + 1)

    # 3. Computamos psi en cada sample como atan2 del segmento adyacente.
    psi = _compute_headings(sampled_xy)

    return np.column_stack([sampled_xy, psi])


# ---------------------------------------------------------------------
# Caminata por sucesoras
# ---------------------------------------------------------------------


def _walk_centerlines(
    lanelet_map: "LaneletMap",
    start_lanelet_id: str,
    start_xy: tuple[float, float],
    max_arc_m: float,
    next_lanelet_hint_ids: tuple[str, ...],
) -> np.ndarray:
    """Concatena puntos de centerline desde `start_lanelet_id` hacia
    adelante hasta cubrir `max_arc_m` o agotar el grafo.

    El primer chunk arranca en el punto de la centerline más cercano a
    `start_xy` (proyección Frenet) para que el path esté centrado en
    el ego, no en el nodo upstream.
    """
    pieces: list[np.ndarray] = []
    accumulated_arc = 0.0
    current_id: str | None = start_lanelet_id
    hint_iter = iter(next_lanelet_hint_ids)

    # Subset hint a un set para descarte rápido al elegir sucesora.
    hint_set = set(next_lanelet_hint_ids)

    visited: set[str] = set()
    is_first = True

    while current_id is not None and accumulated_arc < max_arc_m:
        if current_id in visited:
            # Loop detectado — terminamos.
            break
        visited.add(current_id)

        ll = lanelet_map.get_lanelet(current_id)
        if ll is None:
            break

        cl = ll.centerline
        if cl.shape[0] < 2:
            current_id = _pick_next(ll, hint_set, hint_iter, lanelet_map)
            continue

        if is_first:
            # Recortamos la centerline al sub-arco que arranca cerca del ego.
            start_idx = _nearest_index(cl, start_xy)
            cl = cl[start_idx:]
            is_first = False

        if cl.shape[0] < 2:
            current_id = _pick_next(ll, hint_set, hint_iter, lanelet_map)
            continue

        pieces.append(cl)
        seg_lens = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        accumulated_arc += float(np.sum(seg_lens))

        if accumulated_arc >= max_arc_m:
            break
        current_id = _pick_next(ll, hint_set, hint_iter, lanelet_map)

    if not pieces:
        return np.empty((0, 2), dtype=float)
    # Eliminamos duplicados consecutivos en el join de chunks.
    out: list[np.ndarray] = []
    last_pt: np.ndarray | None = None
    for piece in pieces:
        if last_pt is not None and piece.shape[0] >= 1 and np.allclose(piece[0], last_pt):
            piece = piece[1:]
        if piece.shape[0] >= 1:
            out.append(piece)
            last_pt = piece[-1]
    if not out:
        return np.empty((0, 2), dtype=float)
    return np.concatenate(out, axis=0)


def _pick_next(
    lanelet,
    hint_set: set,
    hint_iter,
    lanelet_map: "LaneletMap",
) -> str | None:
    """Elige la sucesora preferida — primero las que están en el hint
    (la ruta planificada), después la primera del listado del grafo."""
    successors = lanelet.successor_ids
    if not successors:
        return None
    if hint_set:
        for s in successors:
            if s in hint_set:
                return s
    # Fallback: primer sucesor cualquiera.
    return successors[0]


def _nearest_index(cl: np.ndarray, target_xy: tuple[float, float]) -> int:
    """Devuelve el índice del punto de cl más cercano a target_xy."""
    diffs = cl - np.asarray(target_xy)
    sq = np.einsum("ij,ij->i", diffs, diffs)
    return int(np.argmin(sq))


# ---------------------------------------------------------------------
# Resampling y headings
# ---------------------------------------------------------------------


def _resample_polyline(
    polyline: np.ndarray,
    step_arc: float,
    n_samples: int,
) -> np.ndarray:
    """Genera `n_samples` puntos sobre la polyline a paso constante de arc-length.

    Si la polyline es más corta que `step_arc * n_samples`, los samples
    sobrantes se replican desde el último punto disponible (el plan
    queda "parado" al final, lo cual es correcto: el vehículo no tiene
    a dónde ir).
    """
    if polyline.shape[0] < 2:
        return np.tile(polyline[0], (n_samples, 1)) if polyline.shape[0] == 1 else np.zeros((n_samples, 2))

    seg_lens = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])

    samples = np.zeros((n_samples, 2), dtype=float)
    for k in range(n_samples):
        target = step_arc * k
        if target >= total:
            samples[k] = polyline[-1]
            continue
        # Buscar el segmento que contiene `target`.
        idx = int(np.searchsorted(cum, target, side="right") - 1)
        idx = max(0, min(idx, polyline.shape[0] - 2))
        seg_len = seg_lens[idx]
        if seg_len <= 1e-9:
            samples[k] = polyline[idx]
            continue
        t = (target - cum[idx]) / seg_len
        samples[k] = polyline[idx] + t * (polyline[idx + 1] - polyline[idx])
    return samples


def _compute_headings(samples_xy: np.ndarray) -> np.ndarray:
    """Heading (rad) en cada sample, derivado del segmento adyacente.

    Para los N-1 primeros puntos usamos el vector hacia el siguiente
    sample; para el último, replicamos el heading anterior. Esto evita
    discontinuidades en el último step del horizonte.
    """
    n = samples_xy.shape[0]
    if n < 2:
        return np.zeros(n)
    psi = np.zeros(n)
    diffs = np.diff(samples_xy, axis=0)
    psi[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
    psi[-1] = psi[-2]
    # Manejar segmentos degenerados (||diff|| ≈ 0): copiar de vecino.
    for i in range(n):
        if i < n - 1 and np.linalg.norm(diffs[i]) < 1e-6:
            # Buscar el siguiente diff válido.
            for j in range(i + 1, n - 1):
                if np.linalg.norm(diffs[j]) >= 1e-6:
                    psi[i] = math.atan2(diffs[j, 1], diffs[j, 0])
                    break
    return psi
