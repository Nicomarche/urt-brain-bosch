# src/routing/lanelet/queries.py
#
# `LaneletKDTreeIndex` — index espacial sobre los puntos de centerline de
# todas las lanelets de un `LaneletMap`. Resuelve `at_pose(x, y)` en
# O(log N) usando `scipy.spatial.cKDTree`.
#
# ¿Por qué un módulo separado de `lanelet_map.py`?
#   SRP: `LaneletMap` se encarga de la API pública (queries topológicos,
#   regulator lookup); `LaneletKDTreeIndex` se encarga del índice espacial.
#   Mañana cambiar a un grid uniforme o R-tree no toca el contrato del
#   `LaneletMap`. Además rompe el ciclo de imports: `lanelet_map.py`
#   importa `Lanelet` (definido ahí), y `queries.py` toma Lanelets como
#   input para construir el árbol — evita la dependencia circular.
#
# Implementación:
#   - Aplanamos todos los puntos (x, y) de todas las centerlines en un
#     único array (M, 2) y mantenemos un array paralelo `lanelet_ids`
#     que mapea cada punto al ID de su lanelet.
#   - `cKDTree.query(point)` devuelve el índice del punto más cercano,
#     y de ahí leemos el lanelet_id correspondiente.
#
# Performance:
#   - Construcción O(M log M).
#   - Query O(log M).
#   - Memoria O(M).
#   En BFMC con ~150 lanelets * 5 puntos cada una = 750 puntos. Trivial.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.routing.lanelet.lanelet_map import Lanelet


# Importación condicional: scipy es dependencia runtime obligatoria del
# repo; importamos en runtime para no romper si alguien testea pieces
# individuales sin scipy instalado (no debería pasar — se documenta el
# requisito acá).
try:
    from scipy.spatial import cKDTree

    _SCIPY_OK = True
except ImportError:  # pragma: no cover — fail rápido si no está instalado
    _SCIPY_OK = False


class LaneletKDTreeIndex:
    """Index KDTree sobre puntos de centerline para `at_pose`.

    Uso típico:
        index = LaneletKDTreeIndex.from_lanelets(lanelets_dict)
        ll_id = index.query_lanelet(x=12.5, y=-3.7)

    Si ningún punto de centerline está dentro de `max_distance_m`,
    `query_lanelet(...)` devuelve None.
    """

    def __init__(
        self,
        points: np.ndarray,
        lanelet_ids: tuple[str, ...],
    ) -> None:
        if not _SCIPY_OK:
            raise ImportError(
                "scipy.spatial.cKDTree no está disponible. Instalá scipy "
                "antes de usar LaneletKDTreeIndex."
            )
        if points.shape[0] != len(lanelet_ids):
            raise ValueError(
                f"Tamaños inconsistentes: points={points.shape}, "
                f"lanelet_ids={len(lanelet_ids)}"
            )
        self._points = points
        self._lanelet_ids = lanelet_ids
        # Si está vacío, dejamos el tree en None y query devuelve None.
        self._tree = cKDTree(points) if points.shape[0] > 0 else None

    # --------------------------------------------------------------
    # Constructores
    # --------------------------------------------------------------
    @classmethod
    def from_lanelets(cls, lanelets: dict[str, "Lanelet"]) -> "LaneletKDTreeIndex":
        """Construye el index aplanando todas las centerlines.

        Cada punto (x, y) de la centerline de la lanelet L produce una
        entrada en el array; el `lanelet_ids[i]` paralelo guarda el ID
        de L. Edge case: lanelets con centerline de < 2 puntos se
        ignoran (no contribuyen al index).
        """
        chunks: list[np.ndarray] = []
        ids: list[str] = []
        for ll_id, ll in lanelets.items():
            cl = ll.centerline
            if cl is None or cl.shape[0] == 0:
                continue
            chunks.append(cl)
            ids.extend([ll_id] * cl.shape[0])
        if not chunks:
            return cls(points=np.empty((0, 2)), lanelet_ids=())
        points = np.concatenate(chunks, axis=0)
        return cls(points=points, lanelet_ids=tuple(ids))

    # --------------------------------------------------------------
    # Queries
    # --------------------------------------------------------------
    def query_lanelet(
        self,
        x: float,
        y: float,
        max_distance_m: float | None = None,
    ) -> str | None:
        """Devuelve el lanelet_id del punto de centerline más cercano.

        - `max_distance_m`: si se especifica y la distancia al punto más
          cercano la excede, devuelve None. Útil para detectar "ego está
          fuera del mapa".
        """
        if self._tree is None:
            return None
        dist, idx = self._tree.query([float(x), float(y)], k=1)
        if math.isinf(float(dist)) or idx >= len(self._lanelet_ids):
            return None
        if max_distance_m is not None and float(dist) > float(max_distance_m):
            return None
        return self._lanelet_ids[int(idx)]

    def query_distance(self, x: float, y: float) -> float:
        """Distancia al punto de centerline más cercano (m).

        Devuelve `inf` si el index está vacío. No discrimina por lanelet —
        sirve para sanity checks tipo "qué tan lejos del mapa estoy".
        """
        if self._tree is None:
            return math.inf
        dist, _ = self._tree.query([float(x), float(y)], k=1)
        return float(dist)

    # --------------------------------------------------------------
    # Diagnóstico
    # --------------------------------------------------------------
    @property
    def num_points(self) -> int:
        return int(self._points.shape[0])
