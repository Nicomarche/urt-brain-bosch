# src/behavior/scenarios/
#
# Cada archivo define UN escenario de manejo (LaneKeep, Intersection,
# Crosswalk, Parking, Highway, Roundabout) implementando la interfaz
# `IScenario` (`is_active(ctx)`, `plan(ctx)`, `name`, `priority`).
#
# El `BehaviorPlanner` mantiene una lista de scenarios ordenados por
# `priority` (descendente). Itera por el listado y delega `plan(ctx)`
# al primero que dice `is_active(ctx) == True`.
#
# Histeresis: cada scenario es responsable de evitar oscilaciones cuando
# la pose está cerca del borde de su zona activa. Se hace típicamente
# con thresholds asimétricos (`enter_threshold` < `exit_threshold`).
#
# Agregar un escenario nuevo = crear un archivo + agregarlo a la lista
# en el `BehaviorPlanner`. NO requiere editar el planner ni el thread
# (Open/Closed Principle).
