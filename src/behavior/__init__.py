# src/behavior/
#
# Capa de **decisión de comportamiento**. Es el cerebro táctico del
# vehículo: decide qué hacer ahora dado pose + ruta + percepción +
# objetos trackeados, y emite un `BehaviorOutput` (target_path +
# speed_profile + stop_required) que el `MotionController` ejecuta.
#
# Inspirado en los planners de Autoware
# (`autoware_universe/planning/behavior_*`), simplificado al stack BFMC.
#
# ESTRUCTURA SOLID:
#
#   planner.py          BehaviorPlanner — orquestador. Itera scenarios
#                        por prioridad, delega al primero activo, aplica
#                        velocity_overlay. NO contiene lógica de scenario.
#
#   scenarios/          Un escenario por archivo. Cada uno implementa
#                        IScenario (is_active + plan). Agregar uno
#                        nuevo no requiere tocar el planner (OCP).
#       base.py         BaseScenario — clase abstracta + helpers comunes
#                        para construir target_path/speed_profile.
#       lane_keep.py    LaneKeep — fallback default. Siempre activo si
#                        ningún scenario más específico lo está.
#       intersection.py Intersection — desacelera al entrar, frena en
#                        stopline si corresponde, cruza a velocidad baja.
#       crosswalk.py    Crosswalk — frena si peatón cerca, baja velocidad
#                        en zona crosswalk.
#       parking.py      Parking — maniobra de estacionamiento (placeholder
#                        Fase 4; se completa en BFMC final).
#       highway.py      Highway — aumenta velocidad máxima en tramos
#                        flagged ATTR_HIGHWAY_LEFT/RIGHT.
#       roundabout.py   Roundabout — desacelera, yield en entrada.
#
#   trajectory_builder.py  Construye target_path desde la centerline de
#                          la lanelet actual + sucesoras (Frenet projection
#                          al frame del vehículo). Función pura.
#
#   velocity_overlay.py    Aplica caps de velocidad por regulatory_elements
#                          (stopline → 0, crosswalk → slowdown,
#                          speed_limit → cap). Función pura.
#
#   context.py             PlanningContext — bundle inmutable de inputs
#                          que recibe el planner y los scenarios.
#
#   planner_thread.py      Thread que subscribe a topics, llama al planner,
#                          publica BehaviorOutputMsg. Sin lógica.
#
# UNA SOLA FUENTE DE VERDAD:
#   La velocidad del vehículo sale exclusivamente de
#   `BehaviorOutput.speed_profile[0]`. Cualquier "override" (sign action,
#   semáforo, slowdown) se modela como scenario o velocity_overlay rule;
#   no hay parches imperativos en otros threads.
