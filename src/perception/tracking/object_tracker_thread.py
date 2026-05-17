# src/perception/tracking/object_tracker_thread.py
#
# Thread que ejecuta el `MOTTracker` a la tasa de detecciones.
#
# Pipeline:
#   detection_buffer (DetectedObject[]) ──► MOTTracker.step ──► tracked_buffer
#                                                              ──► TRACKED_OBJECTS_MSG
#
# El detector (YOLO) escribe en `detection_buffer` cada vez que procesa
# un frame. Este thread lee el último snapshot, llama `step()`, y
# publica el resultado.
#
# Diseño:
#   - Thread DELGADO: cero lógica de tracking acá. Toda la decisión vive
#     en `MOTTracker` y `TrackState`. SRP.
#   - Sigue el patrón establecido por `threadBehaviorPlanner` y
#     `threadPoseEstimator`: buffers tipados para el data path crítico,
#     IPC msg para snapshot al dashboard.
#   - dt: derivado de timestamps reales (no del clock del thread). Esto
#     mantiene el KF preciso aún si el detector tiene jitter.

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.core.bus.topics import TRACKED_OBJECTS_MSG
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.types.perception import DetectedObject, TrackedObject
from src.core.types.pose import PoseEstimate
from src.templates.threadwithstop import ThreadWithStop

if TYPE_CHECKING:
    from src.core.messaging.buffers import LatestValueBuffer
    from src.perception.tracking.sort_tracker import MOTTracker


class threadObjectTracker(ThreadWithStop):
    """Ejecuta MOTTracker.step a partir del detection_buffer."""

    def __init__(
        self,
        queuesList,
        tracker: "MOTTracker",
        detection_buffer: "LatestValueBuffer",
        pose_estimate_buffer: "LatestValueBuffer",
        tracked_objects_buffer: "LatestValueBuffer",
        *,
        pause_s: float = 0.05,
        logging=None,
        debugging: bool = False,
    ) -> None:
        super().__init__(pause=float(pause_s))
        self.queuesList = queuesList
        self._tracker = tracker
        self._detection_buf = detection_buffer
        self._pose_buf = pose_estimate_buffer
        self._tracked_buf = tracked_objects_buffer
        self.logging = logging
        self.debugging = bool(debugging)

        self._sender = messageHandlerSender(queuesList, TRACKED_OBJECTS_MSG)
        self._last_processed_seq: int = -1
        self._last_step_t: float | None = None

    # ----------------------------------------------------------------
    # Loop principal
    # ----------------------------------------------------------------
    def thread_work(self) -> None:
        # Detecciones más recientes. Si no hay nuevas (mismo seq) saltamos
        # — el KF no avanza si no hay tick de detector.
        det_value, det_ts, det_seq = self._detection_buf.read_latest(with_metadata=True)
        if det_seq == self._last_processed_seq or det_value is None:
            return
        self._last_processed_seq = det_seq

        detections = _coerce_detection_list(det_value)
        if detections is None:
            # Payload con tipo inesperado — log y salir, no romper el thread.
            if self.logging is not None:
                self.logging.warning(
                    "threadObjectTracker: detection buffer payload no es list[DetectedObject]"
                )
            return

        pose = self._pose_buf.read_latest()
        if not isinstance(pose, PoseEstimate):
            # Sin pose, no proyectamos. El tracker igual puede correr si las
            # detecciones ya vienen con position_world_xy populated.
            pose = PoseEstimate()

        now = time.monotonic()
        dt = (now - self._last_step_t) if self._last_step_t is not None else 0.05
        self._last_step_t = now

        try:
            tracked = self._tracker.step(detections, dt, pose)
        except Exception:
            if self.logging is not None:
                self.logging.exception("MOTTracker.step crashed")
            return

        # Persistir como tupla (frozen-friendly) para el BehaviorPlanner.
        tracked_tuple: tuple[TrackedObject, ...] = tuple(tracked)
        self._tracked_buf.write(tracked_tuple, timestamp=det_ts)

        # Publicar snapshot al dashboard.
        try:
            self._sender.send(_serialize_tracked(tracked_tuple, det_ts))
        except Exception:
            if self.logging is not None:
                self.logging.exception("failed to publish TRACKED_OBJECTS_MSG")


def _coerce_detection_list(payload: Any) -> list[DetectedObject] | None:
    """Acepta list/tuple homogéneos de DetectedObject. Caso contrario None."""
    if isinstance(payload, (list, tuple)):
        if all(isinstance(d, DetectedObject) for d in payload):
            return list(payload)
    return None


def _serialize_tracked(
    objects: tuple[TrackedObject, ...],
    timestamp: float,
) -> dict[str, Any]:
    """Serialización mínima para TRACKED_OBJECTS_MSG (cola IPC)."""
    return {
        "timestamp": float(timestamp),
        "objects": [
            {
                "track_id": int(o.track_id),
                "kind": str(o.class_name),
                "x": float(o.position_world_xy[0]),
                "y": float(o.position_world_xy[1]),
                "vx": float(o.velocity_world_xy[0]),
                "vy": float(o.velocity_world_xy[1]),
                "confidence": float(o.confidence),
                "age_frames": int(o.age_frames),
            }
            for o in objects
        ],
    }
