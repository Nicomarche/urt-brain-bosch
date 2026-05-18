# src/perception/tracking/sort_tracker.py
#
# `MOTTracker` — SORT-light. Asocia `DetectedObject` frame-a-frame
# manteniendo IDs persistentes, predice velocidad, expira tracks viejos.
# Implementa `IObjectTracker` (DIP).
#
# Pipeline por step:
#   1. PREDICT: para cada track activo, llamar TrackState.predict(dt).
#   2. ASSIGN: matching detección↔track usando Hungarian. La función de
#      costo es la distancia Mahalanobis al cuadrado, gateada por un
#      umbral chi² (gate). Detecciones sin posición mundo se descartan.
#   3. UPDATE: tracks con match aplican TrackState.update(detection).
#   4. BIRTH: detecciones sin track asignado y con confidence ≥ threshold
#      crean tracks nuevos.
#   5. DEATH: tracks con `time_since_update > max_age_frames` se eliminan.
#   6. EMIT: convertir tracks confirmados (`hits >= min_hits`) a
#      `TrackedObject`. Tracks no confirmados se mantienen vivos pero
#      NO se reportan al BehaviorPlanner (reduce falsos positivos).
#
# Decisiones:
#   - Costo de asociación = Mahalanobis al cuadrado en la covarianza
#     posicional del track (HPH^T + R). Mahalanobis trata correctamente
#     la incertidumbre creciente cuando un track perdió detecciones.
#   - Gate por chi² 99% (2 DoF) = 9.21. Cualquier costo arriba ⇒ no se
#     considera match (se asigna +inf en la matriz Hungarian).
#   - Sin re-ID. Un peatón ocluido > max_age_frames se le asigna ID nuevo
#     al volver. Esto es aceptable para BFMC.
#
# Performance:
#   ~10 detecciones × ~10 tracks por frame ⇒ matriz Hungarian 10x10 ⇒
#   trivial. Total < 0.5 ms por step en Jetson Nano.

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.core.types.perception import DetectedObject, TrackedObject
from src.core.types.pose import PoseEstimate
from src.perception.tracking.kalman_track import TrackState

# Chi² 2 DoF al 99% — matches arriba de este costo se rechazan.
_GATE_CHI2_99 = 9.21


class MOTTracker:
    """SORT-light tracker. Stateful entre llamadas a `step`.

    El uso previsto es UNA instancia por proceso. El thread de tracker
    llama `step` por frame y publica los `TrackedObject` resultantes.
    """

    def __init__(
        self,
        *,
        min_confidence_to_birth: float = 0.35,
        min_hits_to_emit: int = 3,
        max_age_frames: int = 8,
        max_age_per_class: dict[int, int] | None = None,
        gate_mahalanobis_sq: float = _GATE_CHI2_99,
        process_noise_std: float = 0.5,
        measure_noise_std: float = 0.20,
        reid_window_frames: int = 20,
        reid_max_distance_m: float = 0.6,
    ) -> None:
        """SORT tracker con max_age por clase y re-ID heurístico.

        Plan C3 cambios:
          * ``max_age_per_class``: dict {class_id → max_age_frames}. Si la
            clase no está en el dict, fallback a ``max_age_frames``. Default
            PEDESTRIAN=20 (≈1s a 20Hz), CAR=15, signs=8.
          * ``reid_window_frames``: frames después de la muerte de un track
            durante los cuales se considera re-asignar el ID viejo si una
            detección nueva tiene la misma clase y posición cercana.
          * ``reid_max_distance_m``: distancia máxima para considerar re-ID.
        """
        # Configuración de filtros / asociación.
        self._min_confidence_to_birth = float(min_confidence_to_birth)
        self._min_hits_to_emit = int(min_hits_to_emit)
        self._max_age_frames = int(max_age_frames)
        self._max_age_per_class: dict[int, int] = dict(max_age_per_class or {})
        self._gate_mahalanobis_sq = float(gate_mahalanobis_sq)
        self._process_noise_std = float(process_noise_std)
        self._measure_noise_std = float(measure_noise_std)
        self._reid_window_frames = int(reid_window_frames)
        self._reid_max_distance_m = float(reid_max_distance_m)

        # Estado runtime.
        self._tracks: dict[int, TrackState] = {}
        self._next_track_id: int = 1
        # Plan C3.2: tracks recién muertos, candidatos a re-ID. Mapa
        # track_id → (class_id, last_xy, frames_since_death).
        self._dead_tracks: dict[int, tuple[int, tuple[float, float], int]] = {}

    # ----------------------------------------------------------------
    # API IObjectTracker
    # ----------------------------------------------------------------
    def step(
        self,
        detections: list[DetectedObject],
        dt: float,
        ego_pose: PoseEstimate,
    ) -> list[TrackedObject]:
        """Avanza un tick y devuelve tracks confirmados.

        `ego_pose` no se usa hoy (las detecciones ya vienen en mundo) pero
        se acepta porque la interfaz IObjectTracker lo exige y futuras
        variantes pueden necesitarlo (transformar BEV→mundo aquí).
        """
        # Filtrar detecciones sin posición mundo. La proyección 2D-BEV
        # vive aguas arriba (calibration.py); si falla, el detector ya
        # emitió None.
        valid_dets: list[tuple[int, DetectedObject]] = [
            (i, d) for i, d in enumerate(detections)
            if d.position_world_xy is not None
        ]

        # 1. Predict.
        for tr in self._tracks.values():
            tr.predict(dt)

        # 2. Assign.
        track_ids = list(self._tracks.keys())
        det_indices = [i for i, _ in valid_dets]

        if track_ids and det_indices:
            cost = self._build_cost_matrix([self._tracks[tid] for tid in track_ids],
                                           [d for _, d in valid_dets])
            row_idx, col_idx = linear_sum_assignment(cost)

            assigned_track_set: set[int] = set()
            assigned_det_set: set[int] = set()
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] >= self._gate_mahalanobis_sq:
                    continue  # match descartado por gate
                tid = track_ids[r]
                det_pos_in_valid = c
                det = valid_dets[det_pos_in_valid][1]
                self._tracks[tid].update(det)
                assigned_track_set.add(tid)
                assigned_det_set.add(valid_dets[det_pos_in_valid][0])
        else:
            assigned_track_set = set()
            assigned_det_set = set()

        # 3. Birth: detecciones no asignadas con confidence suficiente.
        # Plan C3.2: antes de crear track nuevo, intentar re-ID con un track
        # recién muerto de la misma clase y posición cercana.
        for global_idx, det in valid_dets:
            if global_idx in assigned_det_set:
                continue
            if det.confidence < self._min_confidence_to_birth:
                continue
            reused_id = self._try_reid(det)
            if reused_id is not None:
                self._revive_track(reused_id, det)
            else:
                self._birth_track(det)

        # 4. Death: expirar tracks viejos. Plan C3.1: max_age depende de la
        # clase — peatones y autos sobreviven oclusiones más largas que signs.
        dead_ids: list[int] = []
        for tid, tr in self._tracks.items():
            max_age = self._max_age_per_class.get(int(tr.class_id), self._max_age_frames)
            if tr.is_stale(max_age):
                dead_ids.append(tid)
        for tid in dead_ids:
            tr = self._tracks.pop(tid)
            # Guardar en buffer de re-ID si tiene posición conocida.
            try:
                xy = tr.position()
                self._dead_tracks[tid] = (int(tr.class_id), (float(xy[0]), float(xy[1])), 0)
            except Exception:
                pass

        # Avanzar la edad de los dead tracks y expirar los muy viejos
        # (más allá de reid_window_frames).
        for tid in list(self._dead_tracks.keys()):
            cls, xy, ago = self._dead_tracks[tid]
            ago += 1
            if ago > self._reid_window_frames:
                del self._dead_tracks[tid]
            else:
                self._dead_tracks[tid] = (cls, xy, ago)

        # 5. Emit confirmados.
        return [
            self._track_to_output(tr)
            for tr in self._tracks.values()
            if tr.is_confirmed(self._min_hits_to_emit)
        ]

    def reset(self) -> None:
        """Limpia todos los tracks. Útil al cambiar de mapa o teleportar."""
        self._tracks.clear()
        # No reseteamos `_next_track_id` para que IDs sigan siendo únicos
        # incluso después de reset. Esto facilita debug en logs continuos.

    # ----------------------------------------------------------------
    # Helpers internos
    # ----------------------------------------------------------------
    def _build_cost_matrix(
        self,
        tracks: list[TrackState],
        detections: list[DetectedObject],
    ) -> np.ndarray:
        """Distancia Mahalanobis al cuadrado entre cada (track, detection).

        Costo "alto" (gate + 1) para pares que no deben matchearse.
        """
        n_t = len(tracks)
        n_d = len(detections)
        cost = np.full((n_t, n_d), self._gate_mahalanobis_sq + 1.0, dtype=float)

        for i, tr in enumerate(tracks):
            mu = np.array(tr.position(), dtype=float)
            S = tr.position_covariance() + np.eye(2) * (self._measure_noise_std ** 2)
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue
            for j, det in enumerate(detections):
                z = np.asarray(det.position_world_xy, dtype=float)
                diff = z - mu
                # d² = diff^T S_inv diff
                d2 = float(diff @ S_inv @ diff)

                # Pequeña penalty por mismatch de clase: si las clases no
                # coinciden, multiplicar por 4 para que en el límite del
                # gate prefiera no matchear. NO infinito porque el detector
                # a veces clasifica mal (peatón ↔ ciclista) frame a frame.
                if det.class_id != tr.class_id:
                    d2 *= 4.0

                cost[i, j] = d2
        return cost

    def _birth_track(self, det: DetectedObject) -> None:
        tr = TrackState(
            track_id=self._next_track_id,
            initial_xy=det.position_world_xy,  # type: ignore[arg-type]
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            timestamp=det.timestamp,
            process_noise_std=self._process_noise_std,
            measure_noise_std=self._measure_noise_std,
        )
        self._tracks[self._next_track_id] = tr
        self._next_track_id += 1

    def _try_reid(self, det: DetectedObject) -> int | None:
        """Plan C3.2: re-asignar ID de un track muerto si distance < gate.

        Heurística simple: misma clase + distancia euclidiana al last_xy
        < ``reid_max_distance_m``. NO embedding deep, NO IoU — sólo
        proximidad espacial. Suficiente para mitigar ID churn por
        oclusiones cortas.
        """
        if not self._dead_tracks or det.position_world_xy is None:
            return None
        det_xy = (float(det.position_world_xy[0]), float(det.position_world_xy[1]))
        best_id: int | None = None
        best_dist = self._reid_max_distance_m
        for tid, (cls, xy, _ago) in self._dead_tracks.items():
            if int(cls) != int(det.class_id):
                continue
            dist = ((det_xy[0] - xy[0]) ** 2 + (det_xy[1] - xy[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        if best_id is not None:
            del self._dead_tracks[best_id]
        return best_id

    def _revive_track(self, track_id: int, det: DetectedObject) -> None:
        """Reasigna un track_id viejo a una detección nueva (re-ID)."""
        tr = TrackState(
            track_id=int(track_id),
            initial_xy=det.position_world_xy,  # type: ignore[arg-type]
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            timestamp=det.timestamp,
            process_noise_std=self._process_noise_std,
            measure_noise_std=self._measure_noise_std,
        )
        self._tracks[int(track_id)] = tr

    def _track_to_output(self, tr: TrackState) -> TrackedObject:
        x, y = tr.position()
        vx, vy = tr.velocity()
        return TrackedObject(
            timestamp=tr.last_timestamp,
            track_id=tr.track_id,
            class_id=tr.class_id,
            class_name=tr.class_name,
            confidence=tr.confidence,
            position_world_xy=(x, y),
            velocity_world_xy=(vx, vy),
            age_frames=tr.age,
        )

    # ----------------------------------------------------------------
    # Inspeccion / debugging
    # ----------------------------------------------------------------
    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(self._tracks.keys())

    @property
    def n_tracks(self) -> int:
        return len(self._tracks)

    def all_tracks(self) -> Iterable[TrackState]:
        """Iterador sobre todos los tracks (incluido los no confirmados).

        El consumidor principal (BehaviorPlanner) ve sólo confirmados,
        pero el dashboard/debug puede querer todos para visualización.
        """
        return self._tracks.values()
