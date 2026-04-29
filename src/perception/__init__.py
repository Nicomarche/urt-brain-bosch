# src/perception/
#
# Pipeline de percepción visual del vehículo. Toma frames crudos de la
# cámara y produce observaciones tipadas que el resto del stack consume:
#
#   LaneObservation       — offset lateral + heading_error al carril.
#   StoplineObservation   — distancia + visibilidad de la línea de stop.
#   SignDetection         — clase + bounding-box de señales de tráfico.
#   DetectedObject        — bounding-box BEV de objetos no-señal (peatón,
#                            otro vehículo). Llega al MOT en Fase 5.
#   TrackedObject         — DetectedObject con ID estable + velocidad.
#
# Sub-paquetes:
#   camera/      — captura + calibración + transformación BEV.
#   lane/        — detector YOLO + lane-fitting + thread observador.
#   signs/       — clasificador de señales + mapeo Sign → ScenarioHint
#                  (NO ejecuta acciones; eso lo hace el BehaviorPlanner).
#   stopline/    — detector de líneas blancas en BEV.
#   tracking/    — MOT (multi-object tracker) tipo SORT.
#
# Composición en `src/perception/pipeline.py` (Fase 4-5): la
# `PerceptionPipeline` orquesta los detectors y publica las observaciones
# en buffers compartidos; los threads que las leen son los `*_thread.py`.
#
# Estado actual:
#   - El thread observador de carril vive ya en `lane/lane_observer_thread.py`.
#   - El resto se va a llenar a medida que las fases 4 y 5 avanzan; el
#     monolito legacy `src/hardware/camera/threads/threadLineFollowing.py`
#     se desarma e introduce sus piezas útiles en estos sub-paquetes hasta
#     que termina borrado por completo.
