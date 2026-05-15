# src/behavior/context.py
#
# `PlanningContext` — bundle inmutable de TODO lo que un escenario y el
# planner necesitan para decidir un tick. Pasarlo por valor como
# parámetro evita que cada scenario tenga que conocer la firma de los
# 6 inputs distintos.
#
# Vive en `src/behavior/` (no en `core/types/`) porque referencia tipos
# concretos de varias capas (perception.LaneObservation,
# routing.RouteContext, lanelet_map.LaneletMap). `core/types/` debe
# permanecer libre de dependencias hacia capas concretas (DIP).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.types.perception import LaneObservation, StoplineObservation, TrackedObject
from src.core.types.lidar import LidarObstacle, LidarScan
from src.core.types.pose import PoseEstimate
from src.core.types.routing import RouteContext

if TYPE_CHECKING:
    from src.routing.lanelet.lanelet_map import LaneletMap


@dataclass(frozen=True)
class PlanningContext:
    """Snapshot completo de inputs que un scenario consume en `is_active(ctx)` y `plan(ctx)`.

    Invariantes:
      - `now_s`: timestamp UNIX del tick — debe ser el mismo para todos
        los inputs (el planner_thread se encarga del alineamiento).
      - `dt`: duración nominal del paso de control (s). Debe coincidir
        con el dt del MPC que va a ejecutar el plan.
      - `horizon_n`: longitud del plan en steps. `target_path` resultante
        tendrá shape (horizon_n + 1, 3).
      - `nominal_speed_mps`: velocidad target en lane_keep "limpio". Sirve
        de baseline; los scenarios pueden bajarla, nunca subirla más allá
        de `max_speed_mps`.
      - `max_speed_mps`: hard cap absoluto. Aplicado al final por el
        velocity_overlay; ningún scenario puede emitir velocidades arriba.

    Notas de diseño:
      - `lane_observation` y `stopline_observation` son Optional porque la
        cámara puede estar en blind_mode o no haber recibido frames aún.
      - `tracked_objects` es tupla (frozen) para que el contexto sea
        compartible entre scenarios sin riesgo de mutación.
      - `lanelet_map` se carga una vez al startup y se inyecta. None
        durante bootstrap antes de que el RoutePlanner termine init.
    """

    now_s: float
    dt: float
    horizon_n: int
    nominal_speed_mps: float
    max_speed_mps: float
    pose: PoseEstimate
    route: RouteContext
    lane_observation: LaneObservation | None = None
    stopline_observation: StoplineObservation | None = None
    tracked_objects: tuple[TrackedObject, ...] = ()
    lidar_scan: LidarScan | None = None
    lidar_obstacles: tuple[LidarObstacle, ...] = ()
    lanelet_map: "LaneletMap | None" = None
    # Hints opcionales del subsistema de signos. Cada record describe un
    # sign reciente (kind, distance_m, side). El BehaviorPlanner usa esto
    # para activar scenarios "intersection" / "stop" / "parking" antes de
    # que la regulatory map del lanelet aparezca en regulatory_ahead.
    sign_hints: tuple[dict, ...] = field(default_factory=tuple)
