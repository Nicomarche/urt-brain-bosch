# src/perception/tracking/
#
# Multi-Object Tracker (MOT). Mantiene IDs persistentes para cada
# `DetectedObject` no-señal (peatón, otro vehículo) frame a frame y
# estima velocidad por track. Sin re-ID, sin embeddings deep — solo
# Kalman constant-velocity + asignación Hungarian.
#
# Inspirado en `autoware_universe/perception/multi_object_tracker`,
# adaptado a Jetson Nano (sin torch).
#
#   kalman_track.py            — `TrackState` con KF [x, y, vx, vy].
#   sort_tracker.py            — `MOTTracker(IObjectTracker)`. Hungarian
#                                 vía `scipy.optimize.linear_sum_assignment`.
#   object_tracker_thread.py   — thread que conecta detector → tracker
#                                 → buffer compartido `TrackedObjectsMsg`.
#
# Implementación: Fase 5 del plan (`/Users/luciogarcia/.claude/plans/...`).
# El consumidor principal es el `BehaviorPlanner`: scenarios como Crosswalk
# o LaneKeep usan la velocidad del peatón rastreado para decidir slowdown.
