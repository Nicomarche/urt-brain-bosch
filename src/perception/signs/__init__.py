# src/perception/signs/
#
# Detección y clasificación de señales de tráfico (post-Phase 6).
#
#   sign_classifier.py — helpers PUROS de mapeo Sign → canonical name +
#                        predicado is_actionable_sign + action_group_of.
#                        Es lo único que sobrevive del antiguo
#                        `src/hardware/camera/threads/signActions.py`,
#                        que fue borrado cuando el `motor_command_dispatcher`
#                        pasó a ser el único writer de motores.
#
# La detección concreta (inferencia YOLO, post-proceso, publicación de
# `SignDetected`) sigue viviendo en `threadLocalPerception` por ahora.
# La decisión de qué hacer ante un sign vive en:
#   - `ManeuverManager.observe_sign(...)` para overrides de speed/stop.
#   - los `IScenario` del `BehaviorPlanner` (Crosswalk, Highway,
#     Intersection, Parking) para producir el `BehaviorOutput` final.
