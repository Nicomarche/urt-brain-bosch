# src/perception/lane/
#
# Componentes de detección y observación del carril.
#
#   yolo_lane_detector.py    — inferencia del modelo YOLO TensorRT (Fase 5).
#                               Saca máscara binaria + bounding-boxes de
#                               líneas; sin lógica de control.
#   lane_fitting.py          — ajuste polinomial / spline a los pixeles
#                               de la máscara, con coordenadas BEV (Fase 5).
#   lane_observer_thread.py  — thread que lee snapshots visuales y emite
#                               LaneObservation + StoplineObservation
#                               tipadas hacia el pose estimator y el
#                               BehaviorPlanner.
#
# Producción única:
#   - El thread observador es el ÚNICO publisher de LaneObservation. El
#     consumidor (PoseEstimator) hace `update_lane_normal(...)` con esos
#     valores. Si querés agregar otra fuente de datos visual, ponela acá
#     y el consumidor no se entera.
