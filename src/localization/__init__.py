# src/localization/
#
# Localización del vehículo: fusiona sensores (GPS, IMU, encoder, lane
# visual, landmarks semánticos) y publica una `PoseEstimate` por tick.
#
# Sub-paquetes y módulos:
#
#   ekf/                       — filtro de estado (Fase 2):
#       state_filter.py        EKF7 — `[px, py, ψ, vx, vy, ω, a]`.
#       sensor_models.py       Jacobianos H + builders de R por sensor.
#       coordinate_frame.py    Transformaciones GPS↔ENU local con pyproj.
#
#   pose_estimator_thread.py   Thread `threadPoseEstimator`. Subscribe a
#                              IMU/GPS/encoder/lane/landmark, llama al
#                              EKF y publica PoseEstimate. **Es el único
#                              publisher** de pose en el sistema.
#
#   dead_reckoning.py          Estimador puramente cinemático (steer +
#                              speed → desplazamiento). Hoy es la fuente
#                              principal; se vuelve sensor más del EKF7
#                              cuando aterriza la Fase 2.
#
#   relocalization_thread.py   Cascada que decide cuándo emitir updates
#                              de landmarks (semantic sign match, BEV
#                              stopline). En Fase 2 se separa en
#                              relocalization.py (lógica pura) +
#                              route_context_thread.py.
#
# DIP: el thread depende del Protocol `ILocalizer` (en
# `src/core/interfaces/localizer.py`), no del `EKF7` concreto. Así los
# tests pueden inyectar mocks sin heredar.
