# src/routing/lanelet/
#
# Capa "Lanelet HD map" — terminología tomada de Autoware
# (`autoware_lanelet2_extension`). En BFMC NO usamos `lanelet2` como
# librería (tiene bindings C++ que no compilan en Jetson Nano); este
# paquete es una reimplementación liviana en Python idiomático que cubre
# los queries que el `BehaviorPlanner` necesita.
#
#   from_osm.py        Loader liviano de lanelet2 `.osm` -> `LaneletMap`.
#   osm_router.py      Route handler OSM-only y path denso.
#   attributes.py      Constantes compartidas de atributos de lanelet.
#   lanelet_map.py     `LaneletMap` consultable: `at_pose(x, y)`,
#                       `successors_of(id)`, `regulatory_within(id, m)`.
#                       Dataclasses `Lanelet` + helpers de compatibilidad.
#   queries.py         `LaneletKDTreeIndex` — index espacial sobre
#                       puntos de centerline para `at_pose()` en O(log N).
#
# Pitfall conocido — tramos highway: si el OSM tiene UNA línea por
# tramo highway con `ATTR_HIGHWAY_LEFT/RIGHT`, la lanelet hereda ese
# atributo crudo (sin desplazamiento geométrico). El consumidor lo
# interpreta como "estoy en un carril de highway", suficiente para
# decisiones de comportamiento. Si BFMC 2026 exige carriles
# físicamente separados, agregar el offset en `_densify_segment`.
