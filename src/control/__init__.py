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
#   - `motor_command_dispatcher.py` es el ÚNICO writer de SpeedMotor /
#     SteerMotor en runtime de competencia.
#
# Módulos (post-Fase 6):
#   motion_controller.py          `MotionController` (IMotionController)
#                                  + `AcadosMPC` (low-level solver wrapper).
#                                  Traduce BehaviorOutput → MotorCommand.
#   safety_gate.py                Watchdog: BehaviorOutput stale > 500 ms
#                                  o PoseEstimate stale > 300 ms →
#                                  MotorCommand(0, 0). Stateless excepto
#                                  por last_reason (diagnóstico).
#   controller_thread.py          `threadMotionController`. Lee
#                                  behavior_output_buffer + pose buffer
#                                  → llama MotionController.compute() →
#                                  escribe motor_command_buffer.
#   motor_command_dispatcher.py   `threadMotorCommandDispatcher`. Lee
#                                  motor_command_buffer → safety_gate →
#                                  SpeedMotor + SteerMotor (firmware).
#                                  Único writer de motor en runtime.
#   _acados_solver_gen.py         CLI para regenerar el solver C
#                                  (build-time; no se importa en runtime).
#
# Dependencia inversa (DIP): este paquete consume `BehaviorOutput`
# (definido en `src/core/types/behavior.py`) y produce `MotorCommand`
# (definido en `src/core/types/control.py`). El BehaviorPlanner NO
# importa nada de control.
