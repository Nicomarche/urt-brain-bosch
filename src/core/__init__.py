# src/core/
#
# Núcleo del stack URT Brain post-port (BFMC 2026). Contiene:
#
#   - types/      dataclasses inmutables que viajan entre componentes
#                 (Pose, RouteContext, BehaviorOutput, etc.). Son la "lingua
#                 franca" tipada del sistema; ningún componente comparte
#                 estado mutable, solo intercambia estos objetos.
#
#   - interfaces/ Protocols (PEP 544) que definen los contratos públicos
#                 de los componentes intercambiables del stack: localizador
#                 (EKF7), planner de comportamiento, controller de motion,
#                 tracker de objetos. Aplica DIP (Dependency Inversion):
#                 los threads consumen el Protocol, no la implementación.
#
#   - messaging/  Bus de mensajes IPC priorizados (Critical/Warning/General/
#                 Config/Log) y buffers thread-safe que conectan los 4 procesos
#                 (camera, serial, dashboard, gateway). Migrado desde
#                 src/utils/messages y src/hardware/pipeline/sharedBuffers.
#
# Quien lee este paquete primero: empezar por types/pose.py para entender el
# dato más usado del sistema, después interfaces/ para ver los contratos, y
# por último messaging/allMessages.py para entender cómo se enrutan.
