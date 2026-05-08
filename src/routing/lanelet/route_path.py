from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RoutePath:
    node_ids: list[str]
    waypoints: np.ndarray
    wp_node_attrs: np.ndarray
    wp_node_ids: list[str]
    closed_loop: bool = False
    route_id: str | None = None
    source: str = "lanelet"
    wp_edge_ids: list[str] = field(default_factory=list)
    wp_semantic_ids: list[str | None] = field(default_factory=list)
    wp_semantic_types: list[str] = field(default_factory=list)
    wp_zone_ids: list[list[str]] = field(default_factory=list)
    wp_zone_types: list[list[str]] = field(default_factory=list)
    route_events: list[dict] = field(default_factory=list)
    map_metadata: dict = field(default_factory=dict)
    available_destinations: list[dict] = field(default_factory=list)

    def preview_points(self, max_points: int = 140) -> list[dict[str, float]]:
        if self.waypoints.size == 0:
            return []
        total = int(self.waypoints.shape[0])
        step = max(1, total // max(1, int(max_points)))
        preview = self.waypoints[::step, :2]
        if total > 0 and (len(preview) == 0 or not np.allclose(preview[-1], self.waypoints[-1, :2])):
            preview = np.vstack([preview, self.waypoints[-1, :2]])
        return [
            {"x": round(float(x), 4), "y": round(float(y), 4)}
            for x, y in preview
        ]

    def destination_point(self) -> dict[str, float] | None:
        if self.waypoints.size == 0:
            return None
        x, y, _ = self.waypoints[-1]
        return {"x": round(float(x), 4), "y": round(float(y), 4)}
