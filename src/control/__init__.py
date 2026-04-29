# src/control/
#
# Motion control + safety. Convierte la decisión del BehaviorPlanner
# (`BehaviorOutput`: target_path + speed_profile) en `MotorCommand`
# (steering + speed para el firmware Nucleo STM32).
#
# Decisión arquitectónica clave: **single source of truth**.
#   - `motion_controller.py` (Acados MPC) es el ÚNICO componente que
#     decide steering en runtime de competencia. Stanley se elimina.
#   - El controller NO decide speed: lo recibe del BehaviorPlanner y
#     lo ejecuta. Cualquier override (sign action, stopline) es un
#     escenario en behavior/, no un parche imperativo aquí.
#
# Módulos:
#   motion_controller.py          MPC acoplado (steering + velocity en
#                                  un único QP no-lineal vía Acados).
#                                  Implementa `IMotionController`.
#   _acados_solver_gen.py         CLI para regenerar el solver C
#                                  (build-time; no se importa en runtime).
#   motor_command_dispatcher.py   Dispatch puro: toma `MotorCommand` y
#                                  lo emite por la cola Critical hacia
#                                  threadWrite (UART). Sin lógica de
#                                  decisión. Cuando arriba en Phase 6,
#                                  el archivo se vacía aún más.
#   navigation_control_policy.py  Política legacy que mapea
#                                  RouteContext → speed/steer de
#                                  recovery. Se elimina cuando el
#                                  BehaviorPlanner cubre los mismos
#                                  modos en Phase 4.
#   safety_gate.py                (Phase 6) watchdog + failsafe:
#                                  `BehaviorOutput` stale > 200 ms
#                                  → MotorCommand(0, 0).
#
# Dependencia inversa (DIP): este paquete consume `BehaviorOutput`
# (definido en `src/core/types/behavior.py` en Fase 4); el behavior
# planner NO importa nada de control.
