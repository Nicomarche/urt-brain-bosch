from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
from pyproj import CRS, Transformer

from src.core.types.routing import RegulatoryElement
from src.routing.lanelet.attributes import (
    ATTR_CROSSWALK,
    ATTR_INTERSECTION,
    ATTR_NORMAL,
    ATTR_ROUNDABOUT,
    ATTR_STOPLINE,
)
from src.routing.lanelet.lanelet_map import Lanelet, LaneletMap

_BFMC_LANE_WIDTH_M = 0.35
_TOPOLOGY_MATCH_TOL_M = 0.03
_PARTIAL_BRANCH_CENTER_GAP_M = 0.75 * _BFMC_LANE_WIDTH_M
_PARTIAL_FALLBACK_CENTER_GAP_M = 1.20 * _BFMC_LANE_WIDTH_M
_PARTIAL_FALLBACK_MARGIN_M = 0.12
_ZERO_MATCH_CENTER_GAP_M = 0.50 * _BFMC_LANE_WIDTH_M
_ZERO_MATCH_MARGIN_M = 0.06


@dataclass(frozen=True)
class OsmNode:
    node_id: str
    lat: float
    lon: float
    x: float
    y: float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OsmWay:
    way_id: str
    node_ids: tuple[str, ...]
    tags: dict[str, str]


@dataclass(frozen=True)
class OsmRelation:
    relation_id: str
    members: tuple[tuple[str, str, str], ...]  # (type, ref, role)
    tags: dict[str, str]


@dataclass(frozen=True)
class ParsedOsmMap:
    nodes: dict[str, OsmNode]
    ways: dict[str, OsmWay]
    relations: dict[str, OsmRelation]


def load_lanelet2_osm(path: str, *, step_m: float = 0.50) -> LaneletMap:
    """Carga un mapa Lanelet2 OSM y lo convierte al `LaneletMap` del planner.

    Soporta el subconjunto que necesitamos para el behavior planner:
      - relations `type=lanelet` con bounds `left` y `right`;
      - relations `type=regulatory_element` enlazadas desde la lanelet;
      - proyección geográfica WGS84 -> plano local en metros.

    No depende de los bindings C++ de lanelet2, así que corre igual en macOS
    y Jetson. La idea es que el planner use esta capa como backend geométrico
    único y no tenga que conocer detalles del formato fuente.
    """
    parsed = parse_lanelet2_osm(path)
    if not parsed.nodes:
        return LaneletMap(lanelets={}, regulators={}, kdtree_index=None, map_metadata={})

    regulators = _build_regulators(
        relations=parsed.relations,
        ways=parsed.ways,
        nodes=parsed.nodes,
    )
    lanelets = _build_lanelets(
        relations=parsed.relations,
        ways=parsed.ways,
        nodes=parsed.nodes,
        regulators=regulators,
        step_m=step_m,
    )

    from src.routing.lanelet.queries import LaneletKDTreeIndex

    kdtree = LaneletKDTreeIndex.from_lanelets(lanelets) if lanelets else None
    return LaneletMap(
        lanelets=lanelets,
        regulators=regulators,
        kdtree_index=kdtree,
        map_metadata=_build_map_metadata(parsed=parsed),
    )


def parse_lanelet2_osm(path: str) -> ParsedOsmMap:
    root = ET.parse(path).getroot()
    raw_nodes: list[tuple[str, float, float, float | None, float | None, dict[str, str]]] = []
    projected_raw_nodes: list[tuple[str, float, float]] = []
    for node_el in root.findall("node"):
        node_id = str(node_el.get("id") or "").strip()
        if not node_id:
            continue
        lat = _safe_float(node_el.get("lat"))
        lon = _safe_float(node_el.get("lon"))
        tags = _load_tags(node_el)
        local_x = _first_float(tags, "local_x", "urt:local_x")
        local_y = _first_float(tags, "local_y", "urt:local_y")
        raw_nodes.append((node_id, lat, lon, local_x, local_y, tags))
        if local_x is None or local_y is None:
            projected_raw_nodes.append((node_id, lat, lon))

    transformer = (
        _build_local_transformer(projected_raw_nodes)
        if projected_raw_nodes and _has_nonzero_latlon(projected_raw_nodes)
        else None
    )

    nodes: dict[str, OsmNode] = {}
    for node_id, lat, lon, local_x, local_y, tags in raw_nodes:
        if local_x is not None and local_y is not None:
            x = float(local_x)
            y = float(local_y)
        elif transformer is not None:
            x, y = transformer.transform(float(lon), float(lat))
            x = float(x)
            y = float(y)
        else:
            x = float(lon)
            y = float(lat)
        nodes[node_id] = OsmNode(
            node_id=node_id,
            lat=float(lat),
            lon=float(lon),
            x=float(x),
            y=float(y),
            tags=dict(tags),
        )
    nodes = _flip_nodes_y(nodes)

    ways: dict[str, OsmWay] = {}
    for way_el in root.findall("way"):
        way_id = str(way_el.get("id") or "").strip()
        if not way_id:
            continue
        node_ids = tuple(
            str(nd.get("ref") or "").strip()
            for nd in way_el.findall("nd")
            if str(nd.get("ref") or "").strip()
        )
        ways[way_id] = OsmWay(way_id=way_id, node_ids=node_ids, tags=_load_tags(way_el))

    relations: dict[str, OsmRelation] = {}
    for rel_el in root.findall("relation"):
        rel_id = str(rel_el.get("id") or "").strip()
        if not rel_id:
            continue
        members = tuple(
            (
                str(member.get("type") or ""),
                str(member.get("ref") or ""),
                str(member.get("role") or ""),
            )
            for member in rel_el.findall("member")
        )
        relations[rel_id] = OsmRelation(
            relation_id=rel_id,
            members=members,
            tags=_load_tags(rel_el),
        )
    return ParsedOsmMap(nodes=nodes, ways=ways, relations=relations)


def _build_local_transformer(raw_nodes: list[tuple[str, float, float]]) -> Transformer:
    lat0 = sum(lat for _, lat, _ in raw_nodes) / float(len(raw_nodes))
    lon0 = sum(lon for _, _, lon in raw_nodes) / float(len(raw_nodes))
    src = CRS.from_epsg(4326)
    dst = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(src, dst, always_xy=True)


def _flip_nodes_y(nodes: dict[str, OsmNode]) -> dict[str, OsmNode]:
    """Mirror the imported OSM vertically into the simulator/dashboard frame.

    The BFMC OSM was authored with the opposite Y orientation from the track
    texture and the simulator's `brain_map`/LoCSys projection. Flipping here
    keeps a single frame for:
      - lanelet rendering,
      - ego localisation,
      - lanelet lookup for the planner,
      - simulator ground-truth fixes.
    """
    if not nodes:
        return nodes
    ys = [float(node.y) for node in nodes.values()]
    y_min = min(ys)
    y_max = max(ys)
    y_mid_sum = float(y_min + y_max)
    flipped: dict[str, OsmNode] = {}
    for node_id, node in nodes.items():
        flipped[node_id] = OsmNode(
            node_id=node.node_id,
            lat=float(node.lat),
            lon=float(node.lon),
            x=float(node.x),
            y=float(y_mid_sum - float(node.y)),
            tags=dict(node.tags),
        )
    return flipped


def _build_map_metadata(*, parsed: ParsedOsmMap) -> dict:
    xs = [float(node.x) for node in parsed.nodes.values()]
    ys = [float(node.y) for node in parsed.nodes.values()]
    if not xs or not ys:
        return {}

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    width_m = max(0.001, float(x_max - x_min))
    height_m = max(0.001, float(y_max - y_min))
    meters_per_pixel = 0.01
    image_width_px = max(1, int(math.ceil(width_m / meters_per_pixel)))
    image_height_px = max(1, int(math.ceil(height_m / meters_per_pixel)))

    metadata = {
        "coordinate_origin": "bottom_left",
        "y_axis_inverted": False,
        "width_m": round(width_m, 5),
        "height_m": round(height_m, 5),
        "world_bounds": {
            "x_min": round(x_min, 5),
            "x_max": round(x_max, 5),
            "y_min": round(y_min, 5),
            "y_max": round(y_max, 5),
        },
        "meters_per_pixel": meters_per_pixel,
        "pixels_per_meter": round(1.0 / meters_per_pixel, 6),
        "image_width_px": image_width_px,
        "image_height_px": image_height_px,
        "source": "lanelet2_osm",
    }
    return metadata


def _load_tags(element: ET.Element) -> dict[str, str]:
    return {
        str(tag.get("k") or ""): str(tag.get("v") or "")
        for tag in element.findall("tag")
        if tag.get("k") is not None
    }


def _safe_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _first_float(tags: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key not in tags:
            continue
        try:
            return float(tags[key])
        except (TypeError, ValueError):
            continue
    return None


def _has_nonzero_latlon(raw_nodes: list[tuple[str, float, float]]) -> bool:
    for _node_id, lat, lon in raw_nodes:
        if abs(float(lat)) > 1e-12 or abs(float(lon)) > 1e-12:
            return True
    return False


def _build_regulators(
    *,
    relations: dict[str, OsmRelation],
    ways: dict[str, OsmWay],
    nodes: dict[str, OsmNode],
) -> dict[str, RegulatoryElement]:
    regulators: dict[str, RegulatoryElement] = {}
    for rel in relations.values():
        if str(rel.tags.get("type", "")).lower() != "regulatory_element":
            continue
        kind = str(rel.tags.get("subtype") or rel.tags.get("regulatory_element") or "regulatory_element").lower()
        anchor = _relation_anchor_xy(rel, ways=ways, nodes=nodes)
        if anchor is None:
            continue
        regulators[rel.relation_id] = RegulatoryElement(
            element_id=rel.relation_id,
            kind=kind,
            position_xy=anchor,
            anchor_node_id=None,
            label=str(rel.tags.get("name") or rel.relation_id),
            data=dict(rel.tags),
        )
    return regulators


def _build_lanelets(
    *,
    relations: dict[str, OsmRelation],
    ways: dict[str, OsmWay],
    nodes: dict[str, OsmNode],
    regulators: dict[str, RegulatoryElement],
    step_m: float,
) -> dict[str, Lanelet]:
    lanelet_specs: list[dict[str, object]] = []
    for rel in relations.values():
        if str(rel.tags.get("type", "")).lower() != "lanelet":
            continue
        left_way_id = _member_ref(rel, role="left", member_type="way")
        right_way_id = _member_ref(rel, role="right", member_type="way")
        if left_way_id is None or right_way_id is None:
            continue
        left_way = ways.get(left_way_id)
        right_way = ways.get(right_way_id)
        if left_way is None or right_way is None:
            continue
        left_node_ids = tuple(left_way.node_ids)
        right_node_ids = _align_boundary_node_ids(left_node_ids, right_way.node_ids, nodes)
        left = _way_polyline(ways.get(left_way_id), nodes)
        right = _way_polyline(ways.get(right_way_id), nodes)
        if left.shape[0] < 2 or right.shape[0] < 2:
            continue
        right = _align_boundary_direction(left, right)
        centerline = _centerline_from_bounds(left, right, step_m=step_m)
        if centerline.shape[0] < 2:
            continue

        attr = _attribute_from_lanelet_tags(rel.tags)
        regulatory_ids = tuple(
            ref
            for member_type, ref, role in rel.members
            if member_type == "relation" and role == "regulatory_element" and ref in regulators
        )
        lanelet_specs.append(
            {
                "lanelet_id": str(rel.relation_id),
                "centerline": centerline,
                "attr": int(attr),
                "regulatory_ids": regulatory_ids,
                "left_boundary": np.asarray(left, dtype=float),
                "right_boundary": np.asarray(right, dtype=float),
                "left_node_ids": left_node_ids,
                "right_node_ids": right_node_ids,
            }
        )

        if attr in (ATTR_CROSSWALK, ATTR_STOPLINE):
            synth_id = f"implicit-{rel.relation_id}"
            if synth_id not in regulators:
                regulators[synth_id] = RegulatoryElement(
                    element_id=synth_id,
                    kind="crosswalk" if attr == ATTR_CROSSWALK else "stopline",
                    position_xy=(float(centerline[0, 0]), float(centerline[0, 1])),
                    anchor_node_id=None,
                    label=str(rel.tags.get("subtype") or rel.relation_id),
                    data={"source": "lanelet_subtype", **dict(rel.tags)},
                )
            regulatory_ids = tuple(list(regulatory_ids) + [synth_id])
            lanelet_specs[-1]["regulatory_ids"] = regulatory_ids

    lanelets: dict[str, Lanelet] = {}
    boundary_specs: dict[str, dict[str, tuple[float, float]]] = {}
    for spec in lanelet_specs:
        lanelet_id = str(spec["lanelet_id"])
        centerline = np.asarray(spec["centerline"], dtype=float)
        attr = int(spec["attr"])
        regulatory_ids = tuple(spec["regulatory_ids"])
        left_boundary = np.asarray(spec["left_boundary"], dtype=float)
        right_boundary = np.asarray(spec["right_boundary"], dtype=float)
        length = float(np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1)))
        lanelets[lanelet_id] = Lanelet(
            lanelet_id=str(lanelet_id),
            source_node_id=f"{lanelet_id}:start",
            target_node_id=f"{lanelet_id}:end",
            centerline=centerline,
            length_m=length,
            attribute=int(attr),
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            predecessor_ids=(),
            successor_ids=(),
            regulatory_element_ids=tuple(regulatory_ids),
        )
        boundary_specs[lanelet_id] = _boundary_endpoint_spec(
            tuple(spec["left_node_ids"]),
            tuple(spec["right_node_ids"]),
            nodes,
        )

    successor_candidates: dict[str, list[tuple[int, float, float, str]]] = {
        lanelet_id: []
        for lanelet_id in lanelets
    }
    predecessor_candidates: dict[str, list[tuple[int, float, float, str]]] = {
        lanelet_id: []
        for lanelet_id in lanelets
    }
    for lanelet_id in lanelets:
        boundary = boundary_specs.get(lanelet_id)
        if boundary is None:
            continue
        for other_id in lanelets:
            if other_id == lanelet_id:
                continue
            other_boundary = boundary_specs.get(other_id)
            if other_boundary is None:
                continue
            successor_candidate = _boundary_connection_candidate(boundary, other_boundary)
            if successor_candidate is not None:
                successor_candidates[lanelet_id].append((*successor_candidate, other_id))
            predecessor_candidate = _boundary_connection_candidate(other_boundary, boundary)
            if predecessor_candidate is not None:
                predecessor_candidates[lanelet_id].append((*predecessor_candidate, other_id))

    enriched: dict[str, Lanelet] = {}
    for lanelet_id, lanelet in lanelets.items():
        successors = _select_connection_ids(successor_candidates.get(lanelet_id, []))
        predecessors = _select_connection_ids(predecessor_candidates.get(lanelet_id, []))
        enriched[lanelet_id] = Lanelet(
            lanelet_id=lanelet.lanelet_id,
            source_node_id=lanelet.source_node_id,
            target_node_id=lanelet.target_node_id,
            centerline=lanelet.centerline,
            length_m=lanelet.length_m,
            attribute=lanelet.attribute,
            left_boundary=lanelet.left_boundary,
            right_boundary=lanelet.right_boundary,
            predecessor_ids=tuple(predecessors),
            successor_ids=tuple(successors),
            regulatory_element_ids=lanelet.regulatory_element_ids,
        )
    return enriched


def _member_ref(rel: OsmRelation, *, role: str, member_type: str) -> str | None:
    for m_type, ref, m_role in rel.members:
        if m_type == member_type and m_role == role:
            return ref
    return None


def _way_polyline(way: OsmWay | None, nodes: dict[str, OsmNode]) -> np.ndarray:
    if way is None:
        return np.empty((0, 2), dtype=float)
    pts = []
    for node_id in way.node_ids:
        node = nodes.get(node_id)
        if node is None:
            continue
        pts.append((node.x, node.y))
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)


def _align_boundary_direction(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if _should_reverse_boundary(right=right, left=left):
        return right[::-1]
    return right


def _align_boundary_node_ids(
    left_node_ids: tuple[str, ...],
    right_node_ids: tuple[str, ...],
    nodes: dict[str, OsmNode],
) -> tuple[str, ...]:
    if len(left_node_ids) < 2 or len(right_node_ids) < 2:
        return tuple(right_node_ids)
    left = np.asarray(
        [(nodes[node_id].x, nodes[node_id].y) for node_id in left_node_ids if node_id in nodes],
        dtype=float,
    )
    right = np.asarray(
        [(nodes[node_id].x, nodes[node_id].y) for node_id in right_node_ids if node_id in nodes],
        dtype=float,
    )
    if left.shape[0] < 2 or right.shape[0] < 2:
        return tuple(right_node_ids)
    if _should_reverse_boundary(right=right, left=left):
        return tuple(reversed(right_node_ids))
    return tuple(right_node_ids)


def _should_reverse_boundary(*, left: np.ndarray, right: np.ndarray) -> bool:
    if right.shape[0] < 2:
        return False
    forward_score = float(np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[-1] - right[-1]))
    reverse_score = float(np.linalg.norm(left[0] - right[-1]) + np.linalg.norm(left[-1] - right[0]))
    return reverse_score < forward_score


def _boundary_endpoint_spec(
    left_node_ids: tuple[str, ...],
    right_node_ids: tuple[str, ...],
    nodes: dict[str, OsmNode],
) -> dict[str, tuple[float, float]]:
    def _xy(node_id: str) -> tuple[float, float]:
        node = nodes[node_id]
        return float(node.x), float(node.y)

    left_start_xy = _xy(left_node_ids[0])
    left_end_xy = _xy(left_node_ids[-1])
    right_start_xy = _xy(right_node_ids[0])
    right_end_xy = _xy(right_node_ids[-1])

    return {
        "left_start_xy": left_start_xy,
        "left_end_xy": left_end_xy,
        "right_start_xy": right_start_xy,
        "right_end_xy": right_end_xy,
        "start_center_xy": (
            (left_start_xy[0] + right_start_xy[0]) * 0.5,
            (left_start_xy[1] + right_start_xy[1]) * 0.5,
        ),
        "end_center_xy": (
            (left_end_xy[0] + right_end_xy[0]) * 0.5,
            (left_end_xy[1] + right_end_xy[1]) * 0.5,
        ),
    }


def _boundary_connection_candidate(
    current: dict[str, tuple[float, float]],
    candidate: dict[str, tuple[float, float]],
    *,
    match_tol_m: float = _TOPOLOGY_MATCH_TOL_M,
) -> tuple[int, float, float] | None:
    # The imported OSM is mirrored in Y to match the simulator/dashboard frame,
    # which flips the handedness of the lanelet geometry. Because of that, a
    # true successor can align either left->left/right->right or crossed.
    direct_pairs = (
        math.dist(current["left_end_xy"], candidate["left_start_xy"]),
        math.dist(current["right_end_xy"], candidate["right_start_xy"]),
    )
    crossed_pairs = (
        math.dist(current["left_end_xy"], candidate["right_start_xy"]),
        math.dist(current["right_end_xy"], candidate["left_start_xy"]),
    )
    direct_matches = sum(gap <= float(match_tol_m) for gap in direct_pairs)
    crossed_matches = sum(gap <= float(match_tol_m) for gap in crossed_pairs)

    if (direct_matches, -max(direct_pairs)) >= (crossed_matches, -max(crossed_pairs)):
        match_count = int(direct_matches)
        endpoint_gap = float(max(direct_pairs))
    else:
        match_count = int(crossed_matches)
        endpoint_gap = float(max(crossed_pairs))

    center_gap = math.dist(current["end_center_xy"], candidate["start_center_xy"])
    return match_count, float(center_gap), endpoint_gap


def _select_connection_ids(
    scored_candidates: list[tuple[int, float, float, str]],
) -> list[str]:
    if not scored_candidates:
        return []

    ordered = sorted(
        (
            (
                int(match_count),
                float(center_gap),
                float(endpoint_gap),
                str(lanelet_id),
            )
            for match_count, center_gap, endpoint_gap, lanelet_id in scored_candidates
        ),
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    )
    exact = [item for item in ordered if item[0] >= 2]
    partial = [
        item
        for item in ordered
        if item[0] == 1 and item[1] <= float(_PARTIAL_FALLBACK_CENTER_GAP_M)
    ]
    zero_gap = [
        item
        for item in ordered
        if item[0] == 0
        and item[1] <= float(_ZERO_MATCH_CENTER_GAP_M)
        and item[2] <= float(_ZERO_MATCH_CENTER_GAP_M)
    ]

    if exact:
        keep = list(exact)
        keep.extend(
            item
            for item in partial
            if item[1] <= float(_PARTIAL_BRANCH_CENTER_GAP_M)
        )
    elif partial:
        best_center_gap = partial[0][1]
        keep = [
            item
            for item in partial
            if item[1]
            <= min(
                float(_PARTIAL_FALLBACK_CENTER_GAP_M),
                best_center_gap + float(_PARTIAL_FALLBACK_MARGIN_M),
            )
        ]
    elif zero_gap:
        best_center_gap = zero_gap[0][1]
        keep = [
            item
            for item in zero_gap
            if item[1]
            <= min(
                float(_ZERO_MATCH_CENTER_GAP_M),
                best_center_gap + float(_ZERO_MATCH_MARGIN_M),
            )
        ]
    else:
        keep = []

    result: list[str] = []
    seen: set[str] = set()
    for _match_count, _center_gap, _endpoint_gap, lanelet_id in keep:
        if lanelet_id in seen:
            continue
        seen.add(lanelet_id)
        result.append(lanelet_id)
    return result


def _centerline_from_bounds(left: np.ndarray, right: np.ndarray, *, step_m: float) -> np.ndarray:
    n = max(2, left.shape[0], right.shape[0])
    left_rs = _resample_polyline(left, n=n)
    right_rs = _resample_polyline(right, n=n)
    center = 0.5 * (left_rs + right_rs)
    return _densify_polyline(center, step_m=max(0.10, float(step_m)))


def _resample_polyline(polyline: np.ndarray, *, n: int) -> np.ndarray:
    if polyline.shape[0] < 2:
        return np.tile(polyline[0], (n, 1)) if polyline.shape[0] == 1 else np.zeros((n, 2), dtype=float)
    seg_lens = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    samples = np.zeros((n, 2), dtype=float)
    for idx, target in enumerate(np.linspace(0.0, total, num=n)):
        seg_idx = int(np.searchsorted(cum, target, side="right") - 1)
        seg_idx = max(0, min(seg_idx, polyline.shape[0] - 2))
        seg_len = float(seg_lens[seg_idx])
        if seg_len <= 1e-9:
            samples[idx] = polyline[seg_idx]
            continue
        t = (target - float(cum[seg_idx])) / seg_len
        samples[idx] = polyline[seg_idx] + t * (polyline[seg_idx + 1] - polyline[seg_idx])
    return samples


def _densify_polyline(polyline: np.ndarray, *, step_m: float) -> np.ndarray:
    if polyline.shape[0] < 2:
        return polyline
    out = [polyline[0]]
    for idx in range(polyline.shape[0] - 1):
        p0 = polyline[idx]
        p1 = polyline[idx + 1]
        dist = float(np.linalg.norm(p1 - p0))
        if dist <= 1e-9:
            continue
        steps = max(1, int(math.ceil(dist / max(step_m, 1e-3))))
        for step in range(1, steps + 1):
            t = float(step) / float(steps)
            out.append(p0 + t * (p1 - p0))
    return np.asarray(out, dtype=float)


def _attribute_from_lanelet_tags(tags: dict[str, str]) -> int:
    subtype = str(tags.get("subtype") or "").lower()
    location = str(tags.get("location") or "").lower()
    turn_direction = str(tags.get("turn_direction") or "").lower()
    if subtype == "crosswalk":
        return ATTR_CROSSWALK
    if subtype in {"stop_line", "stopline"}:
        return ATTR_STOPLINE
    if subtype == "roundabout" or location == "roundabout":
        return ATTR_ROUNDABOUT
    if location == "intersection" or turn_direction in {"left", "right", "straight"}:
        return ATTR_INTERSECTION
    return ATTR_NORMAL


def _relation_anchor_xy(
    rel: OsmRelation,
    *,
    ways: dict[str, OsmWay],
    nodes: dict[str, OsmNode],
) -> tuple[float, float] | None:
    for member_type, ref, _role in rel.members:
        if member_type == "node" and ref in nodes:
            node = nodes[ref]
            return float(node.x), float(node.y)
        if member_type == "way":
            poly = _way_polyline(ways.get(ref), nodes)
            if poly.shape[0] > 0:
                mid = poly[poly.shape[0] // 2]
                return float(mid[0]), float(mid[1])
    return None


def _heading_of(polyline: np.ndarray) -> float:
    if polyline.shape[0] < 2:
        return 0.0
    p0 = polyline[0]
    p1 = polyline[min(1, polyline.shape[0] - 1)]
    return math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))
