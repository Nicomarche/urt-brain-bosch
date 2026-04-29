# src/localization/ekf/
#
# Implementación del filtro de Kalman extendido. Vacío hasta Fase 2.
#
# Convención de estado (7-state planar):
#   x = [px, py, ψ, vx, vy, ω, a]
#   px, py    — posición ENU local (m).
#   ψ         — yaw (rad), wrapped en [-π, π).
#   vx, vy    — velocidad en marco vehículo (m/s).
#   ω         — velocidad angular (rad/s).
#   a         — aceleración longitudinal (m/s²).
#
# Decisión: 7-state en vez del 15-state de Autoware. BFMC es plano (sin
# pitch/roll relevantes), y el plano vehículo es despreciable a baja
# velocidad. Menos estado = menos sintonización.
#
#   state_filter.py     EKF7 que implementa `ILocalizer`.
#   sensor_models.py    Jacobianos H + R per-sensor (GPS, IMU, encoder,
#                        lane_normal, landmark).
#   coordinate_frame.py GPS WGS84 → ENU local vía
#                        `pyproj.Transformer.from_crs`.
