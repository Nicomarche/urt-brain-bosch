# src/core/interfaces/tracker.py
#
# Contrato del tracker multi-objeto. Su responsabilidad: asociar detecciones
# crudas frame-a-frame, mantener IDs persistentes, y predecir velocidad por
# track. Output: lista de `TrackedObject` con ID estable que el
# BehaviorPlanner consume para generar slowdowns y yields.
#
# Implementación en Fase 5: SORT-light (Kalman constant-velocity en plano
# BEV + Hungarian via scipy.optimize.linear_sum_assignment). Sin re-ID
# (deep_sort_realtime) porque trae torch y eso no entra en Jetson Nano.
#
# Los tipos `DetectedObject` y `TrackedObject` los define
# `src/core/types/perception.py` en Fase 5; por ahora la firma usa `object`
# para evitar acoplar Phase 1 a Phase 5.

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.types.pose import PoseEstimate


@runtime_checkable
class IObjectTracker(Protocol):
    """Asocia detecciones a tracks con IDs persistentes."""

    def step(
        self,
        detections: list[object],
        dt: float,
        ego_pose: PoseEstimate,
    ) -> list[object]:
        """Avanza un tick del tracker.

        `detections` son objetos `DetectedObject` recién emitidos por el
        detector (Fase 5). `ego_pose` se usa para proyectar las posiciones
        a coordenadas mundo (BEV-mundo) si el tracker mantiene tracks en
        marco mundo. Retorno: `list[TrackedObject]` con IDs estables y
        velocidad estimada.
        """
        ...

    def reset(self) -> None:
        """Limpia todos los tracks vivos (rara vez necesario)."""
        ...
