# src/routing/lanelet/
#
# Capa "Lanelet HD map" — terminología tomada de Autoware
# (`autoware_lanelet2_extension`). En BFMC NO usamos `lanelet2` como
# librería (tiene bindings C++ que no compilan en Jetson Nano); este
# paquete es una reimplementación liviana en Python idiomático que cubre
# los queries que el `BehaviorPlanner` necesita.
#
#   from_graphml.py    Parser del GraphML de pista BFMC + clase
#                       `TrackGraph` con Dijkstra. La factory
#                       `lanelet_map.from_track_graph()` lo envuelve.
#   semantics.py       Loader de `track_semantics.json` (atributos de
#                       nodo: ATTR_STOPLINE=7, ATTR_INTERSECTION=2,
#                       ATTR_HIGHWAY_LEFT/RIGHT, etc.).
#   lanelet_map.py     `LaneletMap` consultable: `at_pose(x, y)`,
#                       `successors_of(id)`, `regulatory_within(id, m)`.
#                       Dataclasses `Lanelet` + factory `from_track_graph`.
#   queries.py         `LaneletKDTreeIndex` — index espacial sobre
#                       puntos de centerline para `at_pose()` en O(log N).
#
# Pitfall conocido — tramos highway: hoy el grafo tiene UNA línea por
# tramo highway con `ATTR_HIGHWAY_LEFT/RIGHT`. La lanelet hereda ese
# atributo crudo (sin desplazamiento geométrico). El consumidor lo
# interpreta como "estoy en un carril de highway", suficiente para
# decisiones de comportamiento. Si BFMC 2026 exige carriles
# físicamente separados, agregar el offset en `_densify_segment`.
