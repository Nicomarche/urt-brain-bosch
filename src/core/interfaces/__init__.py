# src/core/interfaces/
#
# Protocols (PEP 544 / structural typing) que definen los contratos públicos
# de los componentes intercambiables del stack. Aplica DIP: los threads
# consumen el Protocol, no la implementación concreta.
#
# Ejemplo:
#   pose_estimator_thread.py recibe `localizer: ILocalizer`, no `EKF7`. Esto
#   permite swappear el filtro (EKF7 ↔ EKF15 ↔ mock para tests) sin tocar
#   el thread.
#
#   ILocalizer       — fusiona sensores en PoseEstimate. Impl: EKF7 (Fase 2).
#   IBehaviorPlanner — produce BehaviorOutput por tick. Impl: BehaviorPlanner (Fase 4).
#   IScenario        — un escenario activable; lo usan los planners (Fase 4).
#   IMotionController — convierte BehaviorOutput → MotorCommand. Impl: AcadosMPC (Fase 6).
#   IObjectTracker   — asocia detections → tracks con IDs. Impl: SortTracker (Fase 5).

from src.core.interfaces.localizer import ILocalizer
from src.core.interfaces.controller import IMotionController
from src.core.interfaces.tracker import IObjectTracker

__all__ = [
    "ILocalizer",
    "IMotionController",
    "IObjectTracker",
]
