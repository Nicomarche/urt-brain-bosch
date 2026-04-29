# src/routing/lanelet/
#
# Capa "Lanelet HD map" — terminología tomada de Autoware
# (`autoware_lanelet2_extension`). En BFMC NO usamos `lanelet2` como
# librería (tiene bindings C++ que no compilan en Jetson Nano); este
# paquete es una reimplementación liviana en Python idiomático que cubre
# los queries que el `BehaviorPlanner` necesita.
#
#   from_graphml.py    Parser del GraphML de pista BFMC + clase
#                       `TrackGraph` con Dijkstra (legacy unificado;
#                       en Fase 3 se splittea: parser separado del grafo
#                       de ruteo).
#   semantics.py       Loader de `track_semantics.json` (atributos de
#                       nodo: ATTR_STOPLINE=7, ATTR_INTERSECTION=2,
#                       ATTR_HIGHWAY_LEFT/RIGHT, etc.).
#   lanelet_map.py     (Fase 3) `LaneletMap` consultable.
#   queries.py         (Fase 3) `LaneletKDTreeIndex` para `at_pose()`
#                       en O(log N).
#
# Pitfall conocido (apuntado en plan): los tramos highway necesitan
# centerlines offseteadas (lanelet izquierda y derecha). Hoy el grafo
# tiene una sola línea por tramo highway; en Fase 3 se generan las
# lanelets desplazadas usando `_LANE_HALF_WIDTH_M`.
