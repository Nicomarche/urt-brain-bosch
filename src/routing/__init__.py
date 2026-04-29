# src/routing/
#
# Mapa HD + planificación de ruta global. Toma el GraphML de pista BFMC
# y la pose actual, y produce un `RouteContext` con waypoints + lanelet
# activo + elementos regulatorios próximos.
#
# Sub-paquetes y módulos:
#
#   lanelet/                       — capa "Lanelet HD map" estilo Autoware
#                                     (envuelve el GraphML existente).
#       from_graphml.py            Parser GraphML → `TrackGraph` (hoy es
#                                   también el grafo de ruteo; en Fase 3
#                                   se separa de la capa Lanelet).
#       semantics.py               `track_semantics.json` loader.
#       lanelet_map.py             (Fase 3) `LaneletMap` consultable:
#                                   `at_pose()`, `successors_of()`,
#                                   `regulatory_within()`.
#       queries.py                 (Fase 3) KDTree spatial queries.
#
#   route_planner.py               `PathManager`: Dijkstra + selección de
#                                   waypoint actual + lookahead. **Único
#                                   publisher** de ruta global.
#
#   navigation_planner_thread.py   `threadNavigationPlanner`: thread que
#                                   subscribe a comandos de navegación
#                                   (Location messages) y reconfigura
#                                   `PathManager` cuando cambia destino.
#
#   visualizer.py                  `TrackVisualizer`: overlay del grafo +
#                                   pose + ruta para el dashboard.
#                                   Lee, no escribe — pura UI.
