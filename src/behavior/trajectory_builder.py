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


_STRAIGHT_ROUTE_HEADING_TOL_DEG = 8.0
_STRAIGHT_HOLD_MAX_YAW_DELTA_DEG = 8.0
_STRAIGHT_HOLD_YAW_STABILITY_MARGIN_DEG = 1.0
_STRAIGHT_HOLD_PREFIX_M = 0.45
_STRAIGHT_HOLD_ROUTE_WINDOW_M = 0.60
_STRAIGHT_HOLD_EARLY_RECENTER_FORWARD_M = 0.45
_STRAIGHT_HOLD_EARLY_RECENTER_LATERAL_M = 0.005
_ROUTE_LOCAL_RECENTER_MAX_POSE_GAP_M = 0.12
_ROUTE_LOCAL_RECENTER_TAIL_M = 0.60
# Si el ego está casi sobre el corredor (gap pequeño) pero su yaw está muy
# desalineado con la dirección del corredor (>30°), el cubic connector
# genera S-curves apretadas que el MPC interpreta como reverse: pide
# v_opt < 0 y queda atascado en speed=0 con steering al máximo.
# Saltamos el connector y usamos matched_route_only para que el MPC
# alinee el yaw siguiendo el corredor crudo.
_CONNECTOR_NEAR_CORRIDOR_GAP_M = 0.04
_CONNECTOR_MAX_YAW_DELTA_DEG = 30.0
_VISUAL_PREFIX_MIN_FORWARD_M = 0.02
_VISUAL_PREFIX_NEAR_ORIGIN_M = 0.02
_VISUAL_PREFIX_MIN_TANGENT_X_M = -0.001


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
    *,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str]]:
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
        out = np.array([[pose_xy[0], pose_xy[1], float(start_yaw_rad)]], dtype=float)
        if return_metadata:
            return out, _bridge_metadata(bridge_mode="matched_route_only")
        return out

    route = _coerce_route_waypoints(route_waypoints)
    if route.shape[0] == 0:
        out = np.tile(np.array([pose_xy[0], pose_xy[1], float(start_yaw_rad)]), (horizon_n + 1, 1))
        if return_metadata:
            return out, _bridge_metadata(bridge_mode="matched_route_only")
        return out

    idx0 = max(0, min(int(matched_idx), route.shape[0] - 1))
    route_tail = route[idx0:]
    route_tail_xy = np.asarray(route_tail[:, :2], dtype=float)
    matched_xy_arr = _coerce_xy(matched_xy, fallback_xy=route_tail_xy[0])
    route_corridor_xy = _trim_route_corridor_from_matched_point(route_tail_xy, matched_xy_arr)
    # Importante: el yaw del corredor debe salir de la geometría real de la
    # ruta, no del micro-segmento matched->first_wp. Ese gap puede ser de
    # milímetros y flippear de +pi a 0 por ruido numérico, generando un
    # conector demasiado agresivo justo cuando el auto debería seguir
    # derecho. La referencia estable es la tangente del route_tail.
    route_start_yaw = _heading_with_fallback(route_tail_xy, idx=0, fallback=float(route_tail[0, 2]))

    step_arc = max(0.10, float(target_speed_mps) * float(dt))
    pose_gap_m = float(np.linalg.norm(pose_xy - matched_xy_arr))
    yaw_delta_rad = abs(_wrap_angle(route_start_yaw - float(start_yaw_rad)))
    yaw_delta_deg = float(math.degrees(yaw_delta_rad))
    early_recenter_snap = _requires_straight_hold_for_early_recenter(
        pose_xy=pose_xy,
        route_start_yaw_rad=float(route_start_yaw),
        matched_xy=matched_xy_arr,
        pose_gap_m=pose_gap_m,
        protected_prefix_m=_STRAIGHT_HOLD_PREFIX_M,
    )
    bridge_goal_idx = _select_bridge_goal_idx(
        route_corridor_xy,
        step_arc=step_arc,
        pose_gap_m=pose_gap_m,
    )
    bridge_meta = _bridge_metadata(bridge_mode="matched_route_only")
    route_is_straight = _is_route_locally_straight(
        route_tail_xy,
        route_start_yaw_rad=route_start_yaw,
        window_m=_STRAIGHT_HOLD_ROUTE_WINDOW_M,
        heading_tol_deg=_STRAIGHT_ROUTE_HEADING_TOL_DEG,
    )
    # En tramos rectos, una deriva chica del yaw estimado alrededor del
    # umbral puede hacer flip-flop frame a frame entre straight_hold y
    # connector aunque la geometria de la ruta no haya cambiado. Le damos
    # un margen extra chico para preferir la referencia recta, que es la
    # mas estable en este caso.
    straight_hold_yaw_limit_deg = (
        _STRAIGHT_HOLD_MAX_YAW_DELTA_DEG + _STRAIGHT_HOLD_YAW_STABILITY_MARGIN_DEG
        if route_is_straight
        else _STRAIGHT_HOLD_MAX_YAW_DELTA_DEG
    )
    local_recenter_only = (
        route_is_straight
        and (
            pose_gap_m > _ROUTE_LOCAL_RECENTER_MAX_POSE_GAP_M
            or yaw_delta_deg > straight_hold_yaw_limit_deg
        )
    )
    use_straight_hold = (
        route_is_straight
        and route_corridor_xy.shape[0] >= 2
        and (pose_gap_m > 0.02 or early_recenter_snap)
        and pose_gap_m <= _ROUTE_LOCAL_RECENTER_MAX_POSE_GAP_M
        and yaw_delta_rad <= math.radians(straight_hold_yaw_limit_deg)
    )
    yaw_severely_misaligned_near_corridor = (
        pose_gap_m < _CONNECTOR_NEAR_CORRIDOR_GAP_M
        and yaw_delta_rad > math.radians(_CONNECTOR_MAX_YAW_DELTA_DEG)
    )
    use_connector = (
        route_corridor_xy.shape[0] >= 2
        and bridge_goal_idx >= 1
        and (
            pose_gap_m > 0.02
            or yaw_delta_rad > math.radians(_STRAIGHT_HOLD_MAX_YAW_DELTA_DEG)
        )
        and not yaw_severely_misaligned_near_corridor
    )

    if use_straight_hold:
        straight_polyline_xy, straight_meta = _build_straight_hold_route_merge(
            pose_xy=pose_xy,
            start_yaw_rad=float(start_yaw_rad),
            route_start_yaw_rad=float(route_start_yaw),
            route_corridor_xy=route_corridor_xy,
            step_arc=step_arc,
            pose_gap_m=pose_gap_m,
            protected_prefix_m=_STRAIGHT_HOLD_PREFIX_M,
        )
        if straight_polyline_xy is not None:
            polyline_xy = straight_polyline_xy
            bridge_meta = straight_meta
            bridge_reason = "straight_hold"
            bridge_goal_idx = int(bridge_meta.get("route_goal_idx", bridge_goal_idx))
        else:
            use_straight_hold = False

    if not use_straight_hold and use_connector:
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
        # Cuando estamos en local_recenter_only (ego desviado lateralmente del
        # corredor recto) la combinación de pose_yaw distinto al heading-to-goal
        # genera una S-curve apretada en el cubic Hermite: el path va primero
        # en dirección de pose_yaw y luego se curva al goal_yaw. El MPC
        # interpreta esa S-curve como "ve atrás-izquierda" y emite v_opt<0.
        # Evitamos la S-curve usando heading-to-goal como tangente inicial:
        # el cubic se vuelve casi una línea recta + curvatura final, sin
        # rebote inicial.
        if local_recenter_only:
            chord_vec = goal_xy - pose_xy
            if float(np.linalg.norm(chord_vec)) > 1e-6:
                connector_start_yaw = float(math.atan2(chord_vec[1], chord_vec[0]))
            else:
                connector_start_yaw = float(start_yaw_rad)
        else:
            connector_start_yaw = float(start_yaw_rad)
        connector_xy = _sample_cubic_connector(
            start_xy=pose_xy,
            start_yaw_rad=connector_start_yaw,
            goal_xy=goal_xy,
            goal_yaw_rad=goal_yaw,
            n_samples=connector_samples,
        )
        polyline_xy = connector_xy
        tail_after_goal = route_corridor_xy[bridge_goal_idx + 1 :]
        if tail_after_goal.shape[0] > 0:
            if local_recenter_only:
                # En local_recenter_only la ruta se verificó "straight" en la
                # ventana _STRAIGHT_HOLD_ROUTE_WINDOW_M desde el corridor inicio.
                # Si el tail se extiende más allá de esa ventana (porque
                # bridge_goal está dentro y la cola añade _ROUTE_LOCAL_RECENTER_TAIL_M
                # encima), puede entrar en la curva del lanelet siguiente y
                # el MPC empieza a doblar antes de tiempo. Acotamos la cola
                # al espacio remanente de la ventana verificada.
                bridge_arc_m = float(
                    _polyline_arc_length(route_corridor_xy[: bridge_goal_idx + 1])
                )
                safe_tail_m = max(0.0, _STRAIGHT_HOLD_ROUTE_WINDOW_M - bridge_arc_m)
                tail_max_m = min(_ROUTE_LOCAL_RECENTER_TAIL_M, safe_tail_m)
                if tail_max_m <= 0.0:
                    tail_after_goal = tail_after_goal[:0]
                else:
                    tail_after_goal = _truncate_polyline_prefix(
                        tail_after_goal,
                        max_arc_m=tail_max_m,
                    )
            if tail_after_goal.shape[0] > 0:
                polyline_xy = np.vstack([polyline_xy, tail_after_goal])
        bridge_reason = "local_recenter_connector" if local_recenter_only else "connector"
        bridge_meta = _bridge_metadata(
            bridge_mode="connector",
            merge_start_idx=1,
            merge_end_idx=int(connector_xy.shape[0] - 1),
            route_goal_idx=int(bridge_goal_idx),
        )
    else:
        if not use_straight_hold:
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
        yaw_delta_deg=float(yaw_delta_deg),
        straight_hold_yaw_limit_deg=float(straight_hold_yaw_limit_deg),
        route_is_straight=bool(route_is_straight),
        local_recenter_only=bool(local_recenter_only),
        early_recenter_snap=bool(early_recenter_snap),
        bridge_goal_idx=int(bridge_goal_idx),
        bridge_reason=str(bridge_reason),
        bridge_mode=str(bridge_meta.get("bridge_mode", bridge_reason)),
        protected_prefix_m=float(bridge_meta.get("protected_prefix_m", 0.0) or 0.0),
        merge_start_idx=bridge_meta.get("merge_start_idx"),
        merge_end_idx=bridge_meta.get("merge_end_idx"),
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
    if return_metadata:
        return target_path, bridge_meta
    return target_path


def build_target_path_from_visual(
    center_waypoints_body,
    ego_pose,
    target_speed_mps: float,
    horizon_n: int,
    dt: float,
    *,
    connect_from_ego_pose: bool = True,
) -> np.ndarray:
    """Construye target_path (N+1, 3) en frame OSM/mapa a partir de waypoints visuales.

    Replica el paradigma de `urt-ref::LaneDetector::get_waypoints` →
    `state_refs` del solver acados: la percepción ya trae la curva del
    centro de carril en frame body (x adelante, y izquierda) y la
    transformamos rígidamente con la pose fusionada del ego.

    Convención de frames:
      - Body: x adelante del coche, y a la IZQUIERDA del piloto, ψ CCW+.
      - OSM/mapa (lo que consume el resto del pipeline): y crece HACIA
        EL SUR en pantalla, yaw es CW+ desde +x. Por eso al rotar el
        offset body→mundo hay que NEGAR la contribución de `y_left` en
        el eje y (la izquierda en body = norte en pantalla = -y en OSM).
        `motion_controller._osm_pose_to_controller_frame` se ocupa luego
        de espejar al frame estándar ENU del solver acados.

    Args:
      center_waypoints_body: iterable de `(x_fwd_m, y_left_m, psi_rad)`
        en frame body del vehículo. Ya viene con la curvatura del
        polinomio fitteado a las líneas detectadas (incluso en
        single_line, porque la línea ausente se sintetiza desplazando
        la opuesta — no se asume paralelismo).
      ego_pose: cualquier objeto con `.x`, `.y` y `.yaw` (Pose2D o
        PoseEstimate). El yaw está en convención OSM (CW+).
      target_speed_mps: velocidad de referencia para definir step_arc
        (`max(0.10, v*dt)`). Mismo criterio que `build_target_path`.
      horizon_n: cantidad de pasos de control. El path tiene N+1 filas.
      dt: duración del paso (s).
      connect_from_ego_pose: si True, prepende la pose actual cuando los
        waypoints visuales arrancan más adelante. Conviene en `two_line`
        para no dejar un gap en el path; en `single_line` puede crear una
        diagonal artificial que hace que el MPC "entre a doblar" demasiado
        pronto, así que el caller puede desactivarlo.

    Returns:
      ndarray (N+1, 3) con `(x, y, psi)` en frame OSM/mapa. Si el
      polinomio no cubre todo el horizonte, los samples sobrantes se
      extienden en línea recta con el yaw del último waypoint
      transformado (no se extrapola el polinomio fuera de su rango).
    """
    pose_x = float(getattr(ego_pose, "x", 0.0) or 0.0)
    pose_y = float(getattr(ego_pose, "y", 0.0) or 0.0)
    pose_yaw = float(getattr(ego_pose, "yaw", 0.0) or 0.0)
    if horizon_n <= 0:
        return np.array([[pose_x, pose_y, pose_yaw]], dtype=float)

    waypoints = [
        (float(w[0]), float(w[1]), float(w[2]))
        for w in (center_waypoints_body or ())
        if w is not None and len(w) >= 3
    ]
    if not waypoints:
        return np.tile(np.array([pose_x, pose_y, pose_yaw]), (horizon_n + 1, 1))
    waypoints, trimmed_prefix_points = _trim_nonforward_visual_prefix(waypoints)

    # Body (x_fwd, y_left) → OSM (x_east, y_south_screen):
    #   heading vector h_osm = (cos θ, sin θ)
    #   left vector   l_osm = (sin θ, -cos θ)  ← rota h 90° en sentido del
    #                                            piloto (norte de pantalla)
    cos_y = math.cos(pose_yaw)
    sin_y = math.sin(pose_yaw)
    world_xy = np.empty((len(waypoints), 2), dtype=float)
    for idx, (x_fwd, y_left, _psi) in enumerate(waypoints):
        world_xy[idx, 0] = pose_x + cos_y * x_fwd + sin_y * y_left
        world_xy[idx, 1] = pose_y + sin_y * x_fwd - cos_y * y_left

    pose_xy = np.array([pose_x, pose_y], dtype=float)
    first_gap_m = float(np.linalg.norm(world_xy[0] - pose_xy))
    if connect_from_ego_pose and first_gap_m > 1e-6:
        # Los waypoints visuales arrancan en el primer lookahead visible del
        # BEV, no necesariamente en la pose actual. En two-line conviene
        # cerrar ese hueco para no dejar un target_path "saltado". En
        # single-line, en cambio, esa unión puede generar una diagonal
        # sintética hacia el centro reconstruido y adelantar demasiado el
        # giro, por eso el caller puede desactivarla.
        world_xy = np.vstack([pose_xy, world_xy])

    if world_xy.shape[0] < 2:
        # Polinomio degenerado: un solo punto. Replicamos con yaw del ego para
        # que el MPC reciba un plan estático en lugar de crashear.
        return np.tile(np.array([pose_x, pose_y, pose_yaw]), (horizon_n + 1, 1))

    step_arc = max(0.10, float(target_speed_mps) * float(dt))
    total_required_m = step_arc * horizon_n
    seg_lens = np.linalg.norm(np.diff(world_xy, axis=0), axis=1)
    polyline_arc_m = float(np.sum(seg_lens))
    if polyline_arc_m < total_required_m:
        # Extender en línea recta con el heading del último segmento real
        # — NO seguir extrapolando el polinomio fuera del rango fitteado.
        last_diff = world_xy[-1] - world_xy[-2]
        last_norm = float(np.linalg.norm(last_diff))
        if last_norm > 1e-9:
            tail_dir = last_diff / last_norm
            extra_m = total_required_m - polyline_arc_m + step_arc
            extension = world_xy[-1] + tail_dir * extra_m
            world_xy = np.vstack([world_xy, extension])

    sampled_xy = _resample_polyline(world_xy, step_arc=step_arc, n_samples=horizon_n + 1)
    psi = _compute_headings(sampled_xy)
    target_path = np.column_stack([sampled_xy, psi])
    live_log(
        "trajectory_builder", event="visual_target_path_built",
        ego_pose=[round(pose_x, 3), round(pose_y, 3), round(pose_yaw, 4)],
        connected_from_ego_pose=bool(connect_from_ego_pose and first_gap_m > 1e-6),
        body_waypoint_count=int(len(waypoints)),
        trimmed_prefix_points=int(trimmed_prefix_points),
        polyline_arc_m=round(polyline_arc_m, 3),
        step_arc_m=round(step_arc, 4),
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


def blend_target_paths(
    path_visual: np.ndarray,
    path_route: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Mezcla convexamente dos `target_path` (N+1, 3) en el mismo frame.

    Args:
      path_visual: path armado desde los waypoints visuales
        (`build_target_path_from_visual`). Shape (M, 3) con `(x, y, psi)`
        en frame OSM/mapa.
      path_route: path armado desde la ruta densa/centerline
        (`build_target_path_from_route` / `build_target_path`). Mismo
        shape y frame que `path_visual`.
      alpha: peso del visual ∈ [0, 1]. 1 = visual puro, 0 = route puro.
        Valores intermedios producen blend convexo punto a punto.

    Convención de blend:
      - Posición (x, y): blend convexo lineal.
      - Heading (ψ): blend angle-aware con `atan2(α·sinψv+(1−α)·sinψr,
        α·cosψv+(1−α)·cosψr)` para evitar discontinuidades en ±π.

    Si los shapes difieren se truncan al mínimo común (típicamente coinciden
    porque ambos paths se arman con el mismo `horizon_n`). Si uno está vacío
    o degenerado, se devuelve el otro.
    """
    if path_visual is None and path_route is None:
        return np.zeros((1, 3), dtype=float)
    if path_visual is None or path_visual.size == 0:
        return np.asarray(path_route, dtype=float)
    if path_route is None or path_route.size == 0:
        return np.asarray(path_visual, dtype=float)

    a = float(max(0.0, min(1.0, alpha)))
    if a >= 1.0 - 1e-6:
        return np.asarray(path_visual, dtype=float)
    if a <= 1e-6:
        return np.asarray(path_route, dtype=float)

    pv = np.asarray(path_visual, dtype=float)
    pr = np.asarray(path_route, dtype=float)
    if pv.ndim != 2 or pv.shape[1] < 3 or pr.ndim != 2 or pr.shape[1] < 3:
        # Shape inesperado — preferimos el visual si parece sano, sino route.
        return pv if pv.ndim == 2 and pv.shape[1] >= 3 else pr

    n = int(min(pv.shape[0], pr.shape[0]))
    pv = pv[:n]
    pr = pr[:n]

    xy = a * pv[:, :2] + (1.0 - a) * pr[:, :2]

    psi_v = pv[:, 2]
    psi_r = pr[:, 2]
    sin_part = a * np.sin(psi_v) + (1.0 - a) * np.sin(psi_r)
    cos_part = a * np.cos(psi_v) + (1.0 - a) * np.cos(psi_r)
    psi = np.arctan2(sin_part, cos_part)

    return np.column_stack([xy, psi])


def _trim_nonforward_visual_prefix(
    waypoints: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], int]:
    """Drop a pathological visual prefix that starts lateral/backwards.

    En `waypoint_mode` el PathOptimizer puede re-muestrear el path visual a
    pasos muy chicos (~1.25 cm). Si conectamos la pose actual con un primer
    waypoint que está casi al costado o apenas atrás del auto, ese prefijo
    queda "micro-sampleado" por el MPC como una maniobra de reversa lateral.
    El síntoma en pista es exactamente el visto en logs: steer saturado y
    `v_opt_raw < 0`.

    Conservamos prefijos sanos:
      - cuando el primer waypoint ya está claramente adelante; o
      - cuando arranca prácticamente en el origen y la tangente local avanza.
    """
    if len(waypoints) < 2:
        return waypoints, 0

    body_xy = np.asarray([(float(item[0]), float(item[1])) for item in waypoints], dtype=float)
    if _visual_prefix_is_forward(body_xy):
        return waypoints, 0

    for idx in range(1, len(waypoints) - 1):
        tail_xy = body_xy[idx:]
        if _visual_prefix_is_forward(tail_xy):
            return waypoints[idx:], idx
    return waypoints, 0


def _visual_prefix_is_forward(body_xy: np.ndarray) -> bool:
    if body_xy.ndim != 2 or body_xy.shape[0] == 0:
        return True
    first = np.asarray(body_xy[0], dtype=float)
    if float(first[0]) >= _VISUAL_PREFIX_MIN_FORWARD_M:
        return True
    if body_xy.shape[0] < 2:
        return False
    first_gap_m = float(np.linalg.norm(first))
    tangent_x = float(body_xy[1, 0] - body_xy[0, 0])
    return (
        first_gap_m <= _VISUAL_PREFIX_NEAR_ORIGIN_M
        and tangent_x >= _VISUAL_PREFIX_MIN_TANGENT_X_M
    )


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
    last_appended_end: np.ndarray | None = None  # para orientar el SIGUIENTE chunk
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

        cl = np.asarray(ll.centerline, dtype=float)
        if cl.shape[0] < 2:
            current_id, pick_reason = _pick_next(ll, hint_set, hint_iter, lanelet_map)
            debug["successor_choices"].append(
                f"{str(debug['lanelet_sequence'][-1])}->{current_id or 'None'}:{pick_reason}"
            )
            if current_id is None:
                debug["termination_reason"] = "no_successor_after_short_centerline"
            continue

        # Orientación de la centerline: la dirección del parser puede estar
        # invertida respecto al flujo real de la ruta. Decidir según el
        # vecino (sucesor preferido) o el último chunk encolado.
        if is_first:
            # Para la PRIMERA lanelet, orientar mirando hacia el primer hint
            # (sucesor de la ruta activa) si está disponible.
            forward = _orient_against_hint(
                cl, lanelet_map, next_lanelet_hint_ids, ll.successor_ids
            )
            if not forward:
                cl = cl[::-1].copy()
            # Recortamos la centerline al sub-arco que arranca en la
            # proyección del ego sobre la lanelet actual.
            cl = _trim_centerline_from_projection(cl, start_xy)
            is_first = False
        else:
            # Para las siguientes, orientar pegándose al fin del chunk anterior.
            if last_appended_end is not None:
                d_start = float(np.linalg.norm(cl[0] - last_appended_end))
                d_end = float(np.linalg.norm(cl[-1] - last_appended_end))
                if d_end < d_start:
                    cl = cl[::-1].copy()

        if cl.shape[0] < 2:
            current_id, pick_reason = _pick_next(ll, hint_set, hint_iter, lanelet_map)
            debug["successor_choices"].append(
                f"{str(debug['lanelet_sequence'][-1])}->{current_id or 'None'}:{pick_reason}"
            )
            if current_id is None:
                debug["termination_reason"] = "no_successor_after_projection_trim"
            continue

        pieces.append(cl)
        last_appended_end = np.asarray(cl[-1], dtype=float)
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


def _orient_against_hint(
    cl: np.ndarray,
    lanelet_map: "LaneletMap",
    next_lanelet_hint_ids,
    successor_ids: tuple,
) -> bool:
    """Decide forward (True) or reverse (False) for the FIRST lanelet of the
    walk. Looks at the next hinted lanelet that's actually a successor and
    compares which endpoint of `cl` is closer to that neighbor's centerline."""
    if cl.shape[0] < 2:
        return True
    succ_set = set(successor_ids or [])
    chosen_next: str | None = None
    for hinted in next_lanelet_hint_ids:
        hid = str(hinted or "")
        if hid in succ_set:
            chosen_next = hid
            break
    if chosen_next is None and successor_ids:
        chosen_next = successor_ids[0]
    if chosen_next is None:
        return True
    neighbor = lanelet_map.get_lanelet(chosen_next)
    if neighbor is None or neighbor.centerline.shape[0] == 0:
        return True
    nb = np.asarray(neighbor.centerline, dtype=float)
    d_start = float(np.min(np.linalg.norm(nb - cl[0], axis=1)))
    d_end = float(np.min(np.linalg.norm(nb - cl[-1], axis=1)))
    # successor case: exit endpoint closer to neighbor → forward iff cl[-1]
    return d_end <= d_start


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


def _trim_route_corridor_from_matched_point(
    polyline_xy: np.ndarray,
    matched_xy: np.ndarray,
) -> np.ndarray:
    """Start the route corridor at the matched point without reversing first."""

    pts = np.asarray(polyline_xy, dtype=float)
    matched = np.asarray(matched_xy, dtype=float)
    if pts.shape[0] == 0:
        return matched.reshape(1, 2)
    if pts.shape[0] < 2:
        if np.linalg.norm(pts[0] - matched) <= 1e-6:
            return np.array(pts, copy=True)
        return np.vstack([matched, pts[0]])

    best_seg_idx = 0
    best_proj = np.asarray(pts[0], dtype=float)
    best_dist = math.inf
    for idx in range(pts.shape[0] - 1):
        p0 = np.asarray(pts[idx], dtype=float)
        p1 = np.asarray(pts[idx + 1], dtype=float)
        seg = p1 - p0
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 <= 1e-12:
            continue
        rel = matched - p0
        t = max(0.0, min(1.0, float(np.dot(rel, seg) / seg_len2)))
        proj = p0 + t * seg
        dist = float(np.linalg.norm(matched - proj))
        if dist < best_dist:
            best_dist = dist
            best_seg_idx = idx
            best_proj = proj

    tail = pts[best_seg_idx + 1 :]
    pieces: list[np.ndarray] = [np.asarray(best_proj, dtype=float)]
    if tail.shape[0] > 0:
        if np.linalg.norm(tail[0] - best_proj) > 1e-6:
            pieces.extend(tail)
        else:
            pieces.extend(tail[1:])
    if len(pieces) < 2:
        pieces.append(np.asarray(pts[-1], dtype=float))
    return np.vstack(pieces)


def _polyline_arc_length(polyline_xy: np.ndarray) -> float:
    if polyline_xy.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(polyline_xy, axis=0), axis=1)))


def _bridge_metadata(
    *,
    bridge_mode: str,
    protected_prefix_m: float = 0.0,
    merge_start_idx: int | None = None,
    merge_end_idx: int | None = None,
    route_goal_idx: int | None = None,
) -> dict[str, float | int | str]:
    meta: dict[str, float | int | str] = {
        "bridge_mode": str(bridge_mode),
        "protected_prefix_m": float(protected_prefix_m),
    }
    if merge_start_idx is not None:
        meta["merge_start_idx"] = int(merge_start_idx)
    if merge_end_idx is not None:
        meta["merge_end_idx"] = int(merge_end_idx)
    if route_goal_idx is not None:
        meta["route_goal_idx"] = int(route_goal_idx)
    return meta


def _is_route_locally_straight(
    route_tail_xy: np.ndarray,
    *,
    route_start_yaw_rad: float,
    window_m: float,
    heading_tol_deg: float,
) -> bool:
    if route_tail_xy.shape[0] < 2:
        return False
    end_idx = _arc_index_at_distance(route_tail_xy, distance_m=max(0.10, float(window_m)))
    end_heading = _heading_with_fallback(
        route_tail_xy,
        idx=max(0, min(end_idx, route_tail_xy.shape[0] - 1)),
        fallback=float(route_start_yaw_rad),
    )
    heading_delta_deg = abs(
        math.degrees(_wrap_angle(float(end_heading) - float(route_start_yaw_rad)))
    )
    return heading_delta_deg <= float(heading_tol_deg)


def _build_straight_hold_route_merge(
    *,
    pose_xy: np.ndarray,
    start_yaw_rad: float,
    route_start_yaw_rad: float,
    route_corridor_xy: np.ndarray,
    step_arc: float,
    pose_gap_m: float,
    protected_prefix_m: float,
) -> tuple[np.ndarray | None, dict[str, float | int | str]]:
    if route_corridor_xy.shape[0] < 2:
        return None, _bridge_metadata(bridge_mode="matched_route_only")

    merge_target_arc_m = max(
        float(protected_prefix_m) + max(0.20, 3.0 * float(pose_gap_m)),
        0.70,
    )
    route_goal_idx = _arc_index_at_distance(route_corridor_xy, distance_m=merge_target_arc_m)
    prefix_yaw_rad = _straight_hold_prefix_yaw(
        current_yaw_rad=float(start_yaw_rad),
        route_yaw_rad=float(route_start_yaw_rad),
    )
    prefix_local = np.array([float(protected_prefix_m), 0.0], dtype=float)
    prefix_samples = max(
        2,
        int(math.ceil(float(protected_prefix_m) / max(float(step_arc), 1e-6))),
    )

    local_goal = None
    goal_xy = None
    while route_goal_idx < route_corridor_xy.shape[0]:
        goal_xy = np.asarray(route_corridor_xy[route_goal_idx], dtype=float)
        local_goal = _world_to_local(goal_xy - pose_xy, yaw_rad=float(prefix_yaw_rad))
        if float(local_goal[0]) > float(protected_prefix_m) + 0.05:
            break
        route_goal_idx += 1

    if goal_xy is None or local_goal is None:
        return None, _bridge_metadata(bridge_mode="matched_route_only")
    if route_goal_idx >= route_corridor_xy.shape[0]:
        return None, _bridge_metadata(bridge_mode="matched_route_only")
    if float(local_goal[0]) <= float(protected_prefix_m) + 0.05:
        return None, _bridge_metadata(bridge_mode="matched_route_only")

    merge_dx = float(local_goal[0] - prefix_local[0])
    merge_samples = max(
        4,
        int(math.ceil(max(merge_dx, 0.05) / max(float(step_arc), 1e-6))) + 1,
    )

    local_points = [np.array([0.0, 0.0], dtype=float)]
    for idx in range(1, prefix_samples + 1):
        alpha = float(idx) / float(prefix_samples)
        local_points.append(
            np.array([float(prefix_local[0]) * alpha, 0.0], dtype=float)
        )
    merge_start_idx = len(local_points)
    for idx in range(1, merge_samples + 1):
        alpha = float(idx) / float(merge_samples)
        smooth_alpha = _smoothstep(alpha)
        x_val = float(prefix_local[0] + merge_dx * alpha)
        y_val = float(local_goal[1]) * smooth_alpha
        local_points.append(np.array([x_val, y_val], dtype=float))

    merge_xy = np.asarray(
        [_local_to_world(pt, origin_xy=pose_xy, yaw_rad=float(prefix_yaw_rad)) for pt in local_points],
        dtype=float,
    )
    polyline_xy = merge_xy
    tail_after_goal = route_corridor_xy[route_goal_idx + 1 :]
    if tail_after_goal.shape[0] > 0:
        polyline_xy = np.vstack([polyline_xy, tail_after_goal])

    return polyline_xy, _bridge_metadata(
        bridge_mode="straight_hold",
        protected_prefix_m=float(protected_prefix_m),
        merge_start_idx=int(merge_start_idx),
        merge_end_idx=int(len(local_points) - 1),
        route_goal_idx=int(route_goal_idx),
    )


def _requires_straight_hold_for_early_recenter(
    *,
    pose_xy: np.ndarray,
    route_start_yaw_rad: float,
    matched_xy: np.ndarray,
    pose_gap_m: float,
    protected_prefix_m: float,
) -> bool:
    if pose_gap_m <= 1e-6:
        return False
    local_goal = _world_to_local(
        np.asarray(matched_xy, dtype=float) - np.asarray(pose_xy, dtype=float),
        yaw_rad=float(route_start_yaw_rad),
    )
    local_forward_m = float(local_goal[0])
    local_lateral_m = float(local_goal[1])
    if local_forward_m < -0.02:
        return False
    return (
        local_forward_m <= max(
            float(protected_prefix_m),
            _STRAIGHT_HOLD_EARLY_RECENTER_FORWARD_M,
        )
        and abs(local_lateral_m) >= _STRAIGHT_HOLD_EARLY_RECENTER_LATERAL_M
    )


def _arc_index_at_distance(polyline_xy: np.ndarray, *, distance_m: float) -> int:
    if polyline_xy.shape[0] < 2:
        return 0
    seg_lens = np.linalg.norm(np.diff(polyline_xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    idx = int(np.searchsorted(cum, max(0.0, float(distance_m)), side="left"))
    return max(1, min(idx, polyline_xy.shape[0] - 1))


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


def _truncate_polyline_prefix(polyline_xy: np.ndarray, *, max_arc_m: float) -> np.ndarray:
    if polyline_xy.shape[0] < 2 or max_arc_m <= 0.0:
        return np.array(polyline_xy, copy=True)
    out: list[np.ndarray] = [np.asarray(polyline_xy[0], dtype=float)]
    accumulated_m = 0.0
    for idx in range(1, polyline_xy.shape[0]):
        prev = np.asarray(polyline_xy[idx - 1], dtype=float)
        curr = np.asarray(polyline_xy[idx], dtype=float)
        seg = curr - prev
        seg_len_m = float(np.linalg.norm(seg))
        if seg_len_m <= 1e-9:
            continue
        if accumulated_m + seg_len_m >= float(max_arc_m):
            remain_m = max(0.0, float(max_arc_m) - accumulated_m)
            alpha = remain_m / seg_len_m
            out.append(prev + alpha * seg)
            break
        out.append(curr)
        accumulated_m += seg_len_m
    if len(out) == 1:
        out.append(np.asarray(polyline_xy[1], dtype=float))
    return np.asarray(out, dtype=float)


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


def _world_to_local(delta_xy: np.ndarray, *, yaw_rad: float) -> np.ndarray:
    cos_y = math.cos(-float(yaw_rad))
    sin_y = math.sin(-float(yaw_rad))
    return np.array(
        [
            cos_y * float(delta_xy[0]) - sin_y * float(delta_xy[1]),
            sin_y * float(delta_xy[0]) + cos_y * float(delta_xy[1]),
        ],
        dtype=float,
    )


def _local_to_world(local_xy: np.ndarray, *, origin_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    # Wrapper sobre src.core.frames.body_to_world (single source of truth).
    # Mantenemos el nombre legacy para no romper imports internos del módulo.
    from src.core.frames import body_to_world as _bw
    return _bw(local_xy, origin_xy=origin_xy, yaw_rad=yaw_rad)


def _smoothstep(alpha: float) -> float:
    x = max(0.0, min(1.0, float(alpha)))
    return (3.0 * x * x) - (2.0 * x * x * x)


def _straight_hold_prefix_yaw(*, current_yaw_rad: float, route_yaw_rad: float) -> float:
    _ = current_yaw_rad
    return float(route_yaw_rad)
