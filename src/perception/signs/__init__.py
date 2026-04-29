# src/perception/signs/
#
# Detección y clasificación de señales de tráfico.
#
#   sign_classifier.py   — post-proceso de las detecciones YOLO de signs:
#                           filtra por clase, aplica threshold de
#                           confianza, decide si la detección es estable
#                           (>N frames consecutivos antes de emitir).
#   sign_actions.py      — mapeo Sign → ScenarioHint. **Cero ejecución.**
#                           En el sistema legacy `signActions.py`
#                           ejecutaba directamente cambios de speed; en
#                           la nueva arquitectura sólo emite hints que el
#                           BehaviorPlanner consume para activar
#                           Crosswalk/Stop/Highway scenarios. La decisión
#                           final de velocidad sigue siendo del planner.
#
# El archivo legacy `src/hardware/camera/threads/signActions.py`
# permanece operativo hasta que el BehaviorPlanner (Fase 4) lo reemplace.
# Cuando esa fase aterrice, se borra del runtime y todas sus constantes
# (BASE_SPEED, LOW_SPEED, etc.) pasan a config + scenarios.
