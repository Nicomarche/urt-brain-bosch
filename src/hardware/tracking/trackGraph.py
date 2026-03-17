"""
Track graph loader and waypoint generator.

Loads a GraphML file, builds an ordered traversal of nodes starting from
``is_start=true``, and interpolates a dense waypoint array using a cubic
spline — equivalent to SplineUtils.cpp in the reference repo.

Node attribute encoding (new_attribute field):
    0 = NORMAL
    1 = CROSSWALK
    2 = INTERSECTION
    3 = ONEWAY
    4 = HIGHWAY_LEFT
    5 = HIGHWAY_RIGHT
    6 = ROUNDABOUT
    7 = STOPLINE
    8 = DOTTED
    9 = DOTTED_CROSSWALK
"""

import math
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict

try:
    from scipy.interpolate import CubicSpline
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

ATTR_NORMAL = 0
ATTR_CROSSWALK = 1
ATTR_INTERSECTION = 2
ATTR_ONEWAY = 3
ATTR_HIGHWAY_LEFT = 4
ATTR_HIGHWAY_RIGHT = 5
ATTR_ROUNDABOUT = 6
ATTR_STOPLINE = 7

# Attributes treated as "precision zones" where waypoint guidance replaces vision
WAYPOINT_MODE_ATTRS = {ATTR_INTERSECTION, ATTR_STOPLINE}


class TrackNode:
    __slots__ = ("node_id", "x", "y", "attribute", "is_start")

    def __init__(self, node_id, x, y, attribute=ATTR_NORMAL, is_start=False):
        self.node_id = node_id
        self.x = float(x)
        self.y = float(y)
        self.attribute = int(attribute)
        self.is_start = bool(is_start)


class TrackGraph:
    """Loads a GraphML track file and provides navigation utilities."""

    GRAPHML_NS = "http://graphml.graphdrawing.org/graphml"

    def __init__(self, graphml_path: str, step_m: float = 0.05):
        """
        Args:
            graphml_path: Path to the GraphML file.
            step_m:       Spline interpolation step in metres (default 5 cm).
        """
        self.step_m = float(step_m)
        self.nodes: dict[str, TrackNode] = {}
        self.adj: dict[str, list[str]] = defaultdict(list)

        # Filled after build():
        self.ordered_nodes: list[TrackNode] = []   # DFS-ordered graph nodes
        self.waypoints: np.ndarray = np.empty((0, 3))  # (N, 3) → x, y, psi
        self.wp_node_attrs: np.ndarray = np.empty(0, dtype=int)  # attribute per wp

        self._load(graphml_path)
        self._build_path()
        self._interpolate()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self, path: str) -> None:
        tree = ET.parse(path)
        root = tree.getroot()
        ns = self.GRAPHML_NS

        # Discover key IDs for x, y, new_attribute, is_start
        key_map = {}
        for key in root.findall(f"{{{ns}}}key"):
            name = key.get("attr.name", "")
            kid = key.get("id", "")
            key_map[name] = kid

        x_key = key_map.get("x", "d0")
        y_key = key_map.get("y", "d1")
        attr_key = key_map.get("new_attribute", "d2")
        start_key = key_map.get("is_start", "d3")

        graph = root.find(f"{{{ns}}}graph")
        if graph is None:
            raise ValueError(f"No <graph> element found in {path}")

        for node_el in graph.findall(f"{{{ns}}}node"):
            nid = node_el.get("id")
            data = {d.get("key"): d.text for d in node_el.findall(f"{{{ns}}}data")}
            x = float(data.get(x_key, 0.0))
            y = float(data.get(y_key, 0.0))
            attr = int(data.get(attr_key, 0) or 0)
            is_start = str(data.get(start_key, "false")).lower() == "true"
            self.nodes[nid] = TrackNode(nid, x, y, attr, is_start)

        for edge_el in graph.findall(f"{{{ns}}}edge"):
            src = edge_el.get("source")
            tgt = edge_el.get("target")
            if src and tgt:
                self.adj[src].append(tgt)

    # ------------------------------------------------------------------
    # Path building (DFS from start node)
    # ------------------------------------------------------------------
    def _build_path(self) -> None:
        # Find start node
        start_id = None
        for nid, node in self.nodes.items():
            if node.is_start:
                start_id = nid
                break
        if start_id is None:
            # Fall back to the node with the smallest numeric id
            start_id = min(self.nodes.keys(), key=lambda k: int(k))

        # Walk following edges; stop when we'd revisit or dead-end
        visited = set()
        path = []
        current = start_id
        while current is not None and current not in visited:
            visited.add(current)
            node = self.nodes.get(current)
            if node is None:
                break
            path.append(node)
            successors = self.adj.get(current, [])
            # Pick first unvisited successor; if all visited, pick first anyway
            # (allows the loop to "re-enter" for the final segment)
            next_node = None
            for s in successors:
                if s not in visited:
                    next_node = s
                    break
            current = next_node

        self.ordered_nodes = path

    # ------------------------------------------------------------------
    # Spline interpolation (like SplineUtils.cpp)
    # ------------------------------------------------------------------
    def _interpolate(self) -> None:
        if len(self.ordered_nodes) < 2:
            # Single or no nodes: one waypoint at the start position
            if self.ordered_nodes:
                n = self.ordered_nodes[0]
                self.waypoints = np.array([[n.x, n.y, 0.0]])
                self.wp_node_attrs = np.array([n.attribute], dtype=int)
            return

        xs = np.array([n.x for n in self.ordered_nodes])
        ys = np.array([n.y for n in self.ordered_nodes])
        attrs = np.array([n.attribute for n in self.ordered_nodes], dtype=int)

        # Arc-length parameterisation
        dists = np.hypot(np.diff(xs), np.diff(ys))
        dists = np.maximum(dists, 1e-6)
        t = np.concatenate([[0.0], np.cumsum(dists)])
        total_len = t[-1]

        if _SCIPY_OK:
            cs_x = CubicSpline(t, xs)
            cs_y = CubicSpline(t, ys)
            n_pts = max(2, int(total_len / self.step_m) + 1)
            t_dense = np.linspace(0.0, total_len, n_pts)
            wx = cs_x(t_dense)
            wy = cs_y(t_dense)
            dx = cs_x(t_dense, 1)
            dy = cs_y(t_dense, 1)
        else:
            # Linear fallback
            n_pts = max(2, int(total_len / self.step_m) + 1)
            t_dense = np.linspace(0.0, total_len, n_pts)
            wx = np.interp(t_dense, t, xs)
            wy = np.interp(t_dense, t, ys)
            dx = np.gradient(wx, t_dense)
            dy = np.gradient(wy, t_dense)

        psi = np.arctan2(dy, dx)
        self.waypoints = np.column_stack([wx, wy, psi])

        # Map each dense waypoint to the attribute of its nearest graph node
        wp_attrs = np.zeros(len(t_dense), dtype=int)
        for i, td in enumerate(t_dense):
            idx = int(np.searchsorted(t, td, side="right")) - 1
            idx = max(0, min(idx, len(attrs) - 1))
            wp_attrs[i] = attrs[idx]
        self.wp_node_attrs = wp_attrs

    # ------------------------------------------------------------------
    # Navigation utilities
    # ------------------------------------------------------------------
    def find_closest_waypoint(self, x: float, y: float, search_start: int = 0,
                               search_window: int = 60) -> int:
        """Return the index of the waypoint closest to (x, y).

        Searches within ±search_window around search_start to handle
        path wrap-around efficiently.
        """
        n = len(self.waypoints)
        if n == 0:
            return 0
        lo = max(0, search_start - 5)
        hi = min(n, search_start + search_window)
        sub = self.waypoints[lo:hi, :2]
        dists = np.hypot(sub[:, 0] - x, sub[:, 1] - y)
        return int(lo + np.argmin(dists))

    def find_waypoint_ahead(self, x: float, y: float, current_idx: int,
                             lookahead_m: float = 0.30) -> int:
        """Advance current_idx until we reach a waypoint ≥ lookahead_m ahead."""
        n = len(self.waypoints)
        if n == 0:
            return 0
        idx = current_idx % n
        # Skip waypoints that are already behind the car (within lookahead)
        for _ in range(n):
            wp = self.waypoints[idx % n]
            dist = math.hypot(wp[0] - x, wp[1] - y)
            if dist >= lookahead_m:
                break
            idx = (idx + 1) % n
        return idx % n

    def compute_tracking_error(self, x: float, y: float, yaw: float,
                                wp_idx: int) -> tuple[float, float]:
        """Compute lateral (crosstrack) and heading errors vs. waypoint wp_idx.

        Returns:
            (error_m, heading_rad) — same convention as LateralMPC.compute()
                error_m  > 0 → car is to the right of the path centre
                heading_rad > 0 → car is heading left relative to path tangent
        """
        if len(self.waypoints) == 0:
            return 0.0, 0.0
        idx = wp_idx % len(self.waypoints)
        xr, yr, psi_r = self.waypoints[idx]
        dx = x - xr
        dy = y - yr
        # Crosstrack: signed lateral distance from path tangent
        error_m = -dx * math.sin(psi_r) + dy * math.cos(psi_r)
        # Heading error: wrap to [-pi, pi]
        heading_rad = yaw - psi_r
        while heading_rad > math.pi:
            heading_rad -= 2.0 * math.pi
        while heading_rad < -math.pi:
            heading_rad += 2.0 * math.pi
        return float(error_m), float(heading_rad)

    def is_precision_zone(self, wp_idx: int, lookahead_pts: int = 8) -> bool:
        """Return True if any of the next lookahead_pts waypoints is in a
        STOPLINE or INTERSECTION zone — triggers waypoint-mode control."""
        n = len(self.waypoints)
        if n == 0:
            return False
        for i in range(lookahead_pts):
            attr = self.wp_node_attrs[(wp_idx + i) % n]
            if attr in WAYPOINT_MODE_ATTRS:
                return True
        return False

    def get_curvature(self, wp_idx: int) -> float:
        """Return signed path curvature kappa at waypoint wp_idx (1/m).

        Positive = left curve (counterclockwise), Negative = right curve.
        Estimated via central finite differences of the tangent angle psi
        over the two adjacent waypoints, divided by 2 * step_m.
        """
        n = len(self.waypoints)
        if n < 3:
            return 0.0
        idx = wp_idx % n
        i_next = (idx + 1) % n
        i_prev = (idx - 1) % n
        dpsi = float(self.waypoints[i_next][2]) - float(self.waypoints[i_prev][2])
        # Wrap difference to [-pi, pi]
        while dpsi > math.pi:
            dpsi -= 2.0 * math.pi
        while dpsi < -math.pi:
            dpsi += 2.0 * math.pi
        return dpsi / max(2.0 * self.step_m, 1e-6)

    def get_start_pose(self) -> tuple[float, float, float]:
        """Return (x, y, yaw) of the start waypoint."""
        if len(self.waypoints) == 0:
            if self.ordered_nodes:
                n = self.ordered_nodes[0]
                return n.x, n.y, 0.0
            return 0.0, 0.0, 0.0
        wp = self.waypoints[0]
        return float(wp[0]), float(wp[1]), float(wp[2])
