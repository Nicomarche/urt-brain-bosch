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

from src.utils.live_log import live_log

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
    polyline, walk_debug = _walk_centerlines(
        lanelet_map=lanelet_map,
        start_lanelet_id=start_lanelet_id,
        start_xy=start_xy,
        max_arc_m=total_arc_needed * 1.5,  # cushion para que el resample no se quede corto
        next_lanelet_hint_ids=next_lanelet_hint_ids,
    )
    if polyline.shape[0] < 2:
        live_log(
            "trajectory_builder", event="target_path_degenerate",
            start_lanelet_id=str(start_lanelet_id),
            hint_lanelet_ids=[str(item) for item in next_lanelet_hint_ids[:6]],
            lanelet_sequence=list(walk_debug.get("lanelet_sequence", [])),
            successor_choices=list(walk_debug.get("successor_choices", [])),
            termination_reason=str(walk_debug.get("termination_reason", "unknown")),
            target_speed_mps=float(target_speed_mps),
            horizon_n=int(horizon_n),
            dt=float(dt),
            polyline_points=int(polyline.shape[0]),
        )
        # No tenemos centerline utilizable — devolvemos N+1 copias del
        # punto actual con psi=0. El velocity_overlay verá que no avanza
        # y el planner debería marcar invalid.
        return np.tile(np.array([start_xy[0], start_xy[1], 0.0]), (horizon_n + 1, 1))

    # 2. Resampleamos a paso constante de step_arc.
    sampled_xy = _resample_polyline(polyline, step_arc=step_arc, n_samples=horizon_n + 1)

    # 3. Computamos psi en cada sample como atan2 del segmento adyacente.
    psi = _compute_headings(sampled_xy)

    target_path = np.column_stack([sampled_xy, psi])
    live_log(
        "trajectory_builder", event="target_path_built",
        start_lanelet_id=str(start_lanelet_id),
        hint_lanelet_ids=[str(item) for item in next_lanelet_hint_ids[:6]],
        lanelet_sequence=list(walk_debug.get("lanelet_sequence", [])),
        successor_choices=list(walk_debug.get("successor_choices", [])),
        termination_reason=str(walk_debug.get("termination_reason", "unknown")),
        target_speed_mps=float(target_speed_mps),
        step_arc_m=float(step_arc),
        total_arc_needed_m=float(total_arc_needed),
        polyline_points=int(polyline.shape[0]),
        path_points=int(target_path.shape[0]),
        first_ref=[
            round(float(target_path[0, 0]), 3),
            round(float(target_path[0, 1]), 3),
            round(float(target_path[0, 2]), 4),
        ],
        last_ref=[
            round(float(target_path[-1, 0]), 3),
            round(float(target_path[-1, 1]), 3),
            round(float(target_path[-1, 2]), 4),
        ],
    )
    return target_path


def build_target_path_from_route(
    route_waypoints,
    matched_idx: int,
    start_xy: tuple[float, float],
    start_yaw_rad: float,
    matched_xy: tuple[float, float] | None,
    target_speed_mps: float,
    horizon_n: int,
    dt: float,
) -> np.ndarray:
    """Construye una referencia cruda siguiendo la ruta activa.

    A diferencia de `build_target_path(...)`, esta variante toma la ruta
    global ya densificada por el RoutePlanner y arma una polyline local
    que:
      1. se engancha al corredor de la ruta activa a partir de `matched_idx`;
      2. interpola un conector corto y suave desde la pose fused actual
         hacia ese corredor para evitar saltos laterales bruscos.

    El resultado NO necesita tener shape (N+1, 3): el PathOptimizer lo
    volverá a suavizar y re-muestrear. Lo importante es que la geometría
    base siga la misma ruta que ve el operador en el map view.
    """
    pose_xy = np.asarray(start_xy, dtype=float)
    if horizon_n <= 0:
        return np.array([[pose_xy[0], pose_xy[1], float(start_yaw_rad)]], dtype=float)

    route = _coerce_route_waypoints(route_waypoints)
    if route.shape[0] == 0:
        return np.tile(np.array([pose_xy[0], pose_xy[1], float(start_yaw_rad)]), (horizon_n + 1, 1))

    idx0 = max(0, min(int(matched_idx), route.shape[0] - 1))
    route_tail = route[idx0:]
    route_tail_xy = np.asarray(route_tail[:, :2], dtype=float)
    matched_xy_arr = _coerce_xy(matched_xy, fallback_xy=route_tail_xy[0])
    route_corridor_xy = _prepend_xy(route_tail_xy, matched_xy_arr)
    # Importante: el yaw del corredor debe salir de la geometría real de la
    # ruta, no del micro-segmento matched->first_wp. Ese gap puede ser de
    # milímetros y flippear de +pi a 0 por ruido numérico, generando un
    # conector demasiado agresivo justo cuando el auto debería seguir
    # derecho. La referencia estable es la tangente del route_tail.
    route_start_yaw = _heading_with_fallback(route_tail_xy, idx=0, fallback=float(route_tail[0, 2]))

    step_arc = max(0.10, float(target_speed_mps) * float(dt))
    pose_gap_m = float(np.linalg.norm(pose_xy - matched_xy_arr))
    yaw_delta_rad = abs(_wrap_angle(route_start_yaw - float(start_yaw_rad)))
    bridge_goal_idx = _select_bridge_goal_idx(
        route_corridor_xy,
        step_arc=step_arc,
        pose_gap_m=pose_gap_m,
    )
    use_connector = (
        route_corridor_xy.shape[0] >= 2
        and bridge_goal_idx >= 1
        and (
            pose_gap_m > 0.02
            or yaw_delta_rad > math.radians(8.0)
        )
    )

    if use_connector:
        goal_xy = np.asarray(route_corridor_xy[bridge_goal_idx], dtype=float)
        goal_yaw = _heading_with_fallback(
            route_corridor_xy,
            idx=bridge_goal_idx,
            fallback=float(route_tail[min(max(bridge_goal_idx - 1, 0), route_tail.shape[0] - 1), 2]),
        )
        connector_samples = max(
            5,
            min(
                horizon_n + 2,
                int(math.ceil(_polyline_arc_length(route_corridor_xy[: bridge_goal_idx + 1]) / max(step_arc, 1e-6))) + 2,
            ),
        )
        connector_xy = _sample_cubic_connector(
            start_xy=pose_xy,
            start_yaw_rad=float(start_yaw_rad),
            goal_xy=goal_xy,
            goal_yaw_rad=goal_yaw,
            n_samples=connector_samples,
        )
        polyline_xy = connector_xy
        tail_after_goal = route_corridor_xy[bridge_goal_idx + 1 :]
        if tail_after_goal.shape[0] > 0:
            polyline_xy = np.vstack([polyline_xy, tail_after_goal])
        bridge_reason = "connector"
    else:
        polyline_xy = route_corridor_xy
        bridge_reason = "matched_route_only"

    if polyline_xy.shape[0] < 2:
        polyline_xy = np.vstack([pose_xy, matched_xy_arr])

    headings = _compute_headings(polyline_xy)
    target_path = np.column_stack([polyline_xy, headings])
    live_log(
        "trajectory_builder", event="route_target_path_built",
        matched_idx=int(idx0),
        route_waypoint_count=int(route.shape[0]),
        route_tail_count=int(route_tail.shape[0]),
        pose_gap_m=float(pose_gap_m),
        route_start_yaw_rad=float(route_start_yaw),
        start_yaw_rad=float(start_yaw_rad),
        yaw_delta_deg=float(math.degrees(yaw_delta_rad)),
        bridge_goal_idx=int(bridge_goal_idx),
        bridge_reason=str(bridge_reason),
        path_points=int(target_path.shape[0]),
        first_ref=[
            round(float(target_path[0, 0]), 3),
            round(float(target_path[0, 1]), 3),
            round(float(target_path[0, 2]), 4),
        ],
        last_ref=[
            round(float(target_path[-1, 0]), 3),
            round(float(target_path[-1, 1]), 3),
            round(float(target_path[-1, 2]), 4),
        ],
    )
    return target_path


# ---------------------------------------------------------------------
# Caminata por sucesoras
# ---------------------------------------------------------------------


def _walk_centerlines(
    lanelet_map: "LaneletMap",
    start_lanelet_id: str,
    start_xy: tuple[float, float],
    max_arc_m: float,
    next_lanelet_hint_ids: tuple[str, ...],
) -> tuple[np.ndarray, dict]:
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
    debug = {
        "lanelet_sequence": [],
        "successor_choices": [],
        "termination_reason": "unknown",
    }

    while current_id is not None and accumulated_arc < max_arc_m:
        if current_id in visited:
            # Loop detectado — terminamos.
            debug["termination_reason"] = "loop_detected"
            break
        visited.add(current_id)
        debug["lanelet_sequence"].append(str(current_id))

        ll = lanelet_map.get_lanelet(current_id)
        if ll is None:
            debug["termination_reason"] = "lanelet_missing"
            break

        cl = ll.centerline
        if cl.shape[0] < 2:
            current_id, pick_reason = _pick_next(ll, hint_set, hint_iter, lanelet_map)
            debug["successor_choices"].append(
                f"{str(debug['lanelet_sequence'][-1])}->{current_id or 'None'}:{pick_reason}"
            )
            if current_id is None:
                debug["termination_reason"] = "no_successor_after_short_centerline"
            continue

        if is_first:
            # Recortamos la centerline al sub-arco que arranca en la
            # proyección del ego sobre la lanelet actual. Elegir solo el
            # vértice más cercano hace que, si el ego viene lateralmente
            # desplazado cerca del final de una lanelet corta, el path salte
            # prematuramente al nodo terminal y pierda el resto del tramo
            # actual. Eso vuelve demasiado brusca la transición al siguiente
            # segmento y el controlador termina "cortando" el giro.
            cl = _trim_centerline_from_projection(cl, start_xy)
            is_first = False

        if cl.shape[0] < 2:
            current_id, pick_reason = _pick_next(ll, hint_set, hint_iter, lanelet_map)
            debug["successor_choices"].append(
                f"{str(debug['lanelet_sequence'][-1])}->{current_id or 'None'}:{pick_reason}"
            )
            if current_id is None:
                debug["termination_reason"] = "no_successor_after_projection_trim"
            continue

        pieces.append(cl)
        seg_lens = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        accumulated_arc += float(np.sum(seg_lens))

        if accumulated_arc >= max_arc_m:
            debug["termination_reason"] = "reached_max_arc"
            break
        current_id, pick_reason = _pick_next(ll, hint_set, hint_iter, lanelet_map)
        debug["successor_choices"].append(
            f"{str(debug['lanelet_sequence'][-1])}->{current_id or 'None'}:{pick_reason}"
        )
        if current_id is None:
            debug["termination_reason"] = "no_successor"

    if not pieces:
        if debug["termination_reason"] == "unknown":
            debug["termination_reason"] = "no_pieces"
        return np.empty((0, 2), dtype=float), debug
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
        if debug["termination_reason"] == "unknown":
            debug["termination_reason"] = "empty_after_join"
        return np.empty((0, 2), dtype=float), debug
    if debug["termination_reason"] == "unknown":
        debug["termination_reason"] = "joined_polyline"
    return np.concatenate(out, axis=0), debug


def _pick_next(
    lanelet,
    hint_set: set,
    hint_iter,
    lanelet_map: "LaneletMap",
) -> tuple[str | None, str]:
    """Elige la sucesora preferida usando el orden topológico del hint.

    `next_lanelet_hint_ids` llega ordenado desde la ruta activa. No alcanza
    con preguntar "¿está esta sucesora en el set?": en un loop o una ruta que
    vuelve a pasar por el mismo cruce, más de una rama puede aparecer en el
    hint y la correcta es la PRIMERA futura, no la primera del grafo.
    """
    successors = lanelet.successor_ids
    if not successors:
        return None, "no_successor"
    if hint_iter is not None:
        for hinted_lanelet_id in hint_iter:
            hinted_lanelet_id = str(hinted_lanelet_id or "")
            if hinted_lanelet_id in successors:
                return hinted_lanelet_id, "hint_order"
    if hint_set:
        for s in successors:
            if s in hint_set:
                return s, "hint_membership"
    # Fallback: primer sucesor cualquiera.
    return successors[0], "first_successor_fallback"


def _trim_centerline_from_projection(cl: np.ndarray, target_xy: tuple[float, float]) -> np.ndarray:
    """Recorta `cl` desde la proyección más cercana de `target_xy`.

    Devuelve una polyline que arranca en el punto proyectado sobre el
    segmento más cercano y conserva el resto de la centerline hacia
    adelante. Si la proyección cae exactamente en el último vértice,
    puede devolver una polyline de un solo punto para que el caller
    avance a la sucesora.
    """
    if cl.shape[0] < 2:
        return cl

    target = np.asarray(target_xy, dtype=float)
    best_seg_idx = 0
    best_t = 0.0
    best_proj = np.asarray(cl[0], dtype=float)
    best_dist_sq = float("inf")

    for seg_idx in range(cl.shape[0] - 1):
        p0 = np.asarray(cl[seg_idx], dtype=float)
        p1 = np.asarray(cl[seg_idx + 1], dtype=float)
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq <= 1e-12:
            t = 0.0
            proj = p0
        else:
            t = float(np.dot(target - p0, seg) / seg_len_sq)
            t = max(0.0, min(1.0, t))
            proj = p0 + t * seg
        dist_sq = float(np.dot(target - proj, target - proj))
        if dist_sq < best_dist_sq:
            best_seg_idx = seg_idx
            best_t = t
            best_proj = proj
            best_dist_sq = dist_sq

    out: list[np.ndarray] = [best_proj]
    next_vertices = cl[best_seg_idx + 1 :]
    if next_vertices.size == 0:
        return np.asarray(out, dtype=float)
    if np.allclose(best_proj, next_vertices[0]):
        next_vertices = next_vertices[1:]
    if next_vertices.size:
        out.extend(np.asarray(pt, dtype=float) for pt in next_vertices)
    return np.asarray(out, dtype=float)


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


def _coerce_route_waypoints(route_waypoints) -> np.ndarray:
    try:
        arr = np.asarray(route_waypoints, dtype=float)
    except Exception:
        return np.empty((0, 3), dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    if arr.shape[1] < 2:
        return np.empty((0, 3), dtype=float)
    if arr.shape[1] < 3:
        arr = np.column_stack([arr[:, :2], np.zeros(arr.shape[0], dtype=float)])
    return arr[:, :3]


def _coerce_xy(candidate_xy, fallback_xy: np.ndarray) -> np.ndarray:
    if candidate_xy is None:
        return np.asarray(fallback_xy, dtype=float)
    try:
        return np.asarray([float(candidate_xy[0]), float(candidate_xy[1])], dtype=float)
    except Exception:
        return np.asarray(fallback_xy, dtype=float)


def _prepend_xy(polyline_xy: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    if polyline_xy.shape[0] == 0:
        return np.asarray([point_xy], dtype=float)
    if np.linalg.norm(polyline_xy[0] - point_xy) <= 1e-6:
        return np.array(polyline_xy, copy=True)
    return np.vstack([np.asarray(point_xy, dtype=float), polyline_xy])


def _polyline_arc_length(polyline_xy: np.ndarray) -> float:
    if polyline_xy.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(polyline_xy, axis=0), axis=1)))


def _select_bridge_goal_idx(
    route_corridor_xy: np.ndarray,
    *,
    step_arc: float,
    pose_gap_m: float,
) -> int:
    if route_corridor_xy.shape[0] < 2:
        return 0
    seg_lens = np.linalg.norm(np.diff(route_corridor_xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    target_arc_m = min(
        max(5.0 * float(step_arc), 2.5 * float(pose_gap_m), 0.35),
        max(0.50, 8.0 * float(step_arc)),
    )
    idx = int(np.searchsorted(cum, target_arc_m, side="left"))
    idx = max(1, min(idx, route_corridor_xy.shape[0] - 1))
    return idx


def _sample_cubic_connector(
    *,
    start_xy: np.ndarray,
    start_yaw_rad: float,
    goal_xy: np.ndarray,
    goal_yaw_rad: float,
    n_samples: int,
) -> np.ndarray:
    n = max(2, int(n_samples))
    p0 = np.asarray(start_xy, dtype=float)
    p1 = np.asarray(goal_xy, dtype=float)
    chord_m = float(np.linalg.norm(p1 - p0))
    tangent_len_m = min(
        max(0.10, chord_m * 0.50),
        max(0.12, chord_m * 1.20),
    )
    m0 = np.array([math.cos(float(start_yaw_rad)), math.sin(float(start_yaw_rad))], dtype=float) * tangent_len_m
    m1 = np.array([math.cos(float(goal_yaw_rad)), math.sin(float(goal_yaw_rad))], dtype=float) * tangent_len_m

    ts = np.linspace(0.0, 1.0, n)
    out = np.zeros((n, 2), dtype=float)
    for idx, t in enumerate(ts):
        h00 = 2.0 * t ** 3 - 3.0 * t ** 2 + 1.0
        h10 = t ** 3 - 2.0 * t ** 2 + t
        h01 = -2.0 * t ** 3 + 3.0 * t ** 2
        h11 = t ** 3 - t ** 2
        out[idx] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    return out


def _heading_with_fallback(polyline_xy: np.ndarray, idx: int, fallback: float) -> float:
    if polyline_xy.shape[0] < 2:
        return float(fallback)
    idx = max(0, min(int(idx), polyline_xy.shape[0] - 1))
    if idx < polyline_xy.shape[0] - 1:
        diff = polyline_xy[idx + 1] - polyline_xy[idx]
        if np.linalg.norm(diff) >= 1e-9:
            return float(math.atan2(diff[1], diff[0]))
    if idx > 0:
        diff = polyline_xy[idx] - polyline_xy[idx - 1]
        if np.linalg.norm(diff) >= 1e-9:
            return float(math.atan2(diff[1], diff[0]))
    return float(fallback)


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))
