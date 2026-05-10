from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from statistics import median

from src.routing.lanelet.osm_router import OsmRouteGraph
from src.routing.lanelet.from_osm import load_lanelet2_osm
from src.routing.route_planner import PathManager


_MINI_LANELET_OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="pytest">
  <node id="1" lat="46.000010" lon="23.000000" />
  <node id="2" lat="46.000010" lon="23.000020" />
  <node id="3" lat="45.999990" lon="23.000000" />
  <node id="4" lat="45.999990" lon="23.000020" />
  <node id="5" lat="46.000010" lon="23.000040" />
  <node id="6" lat="45.999990" lon="23.000040" />

  <way id="10"><nd ref="1"/><nd ref="2"/></way>
  <way id="20"><nd ref="3"/><nd ref="4"/></way>
  <way id="11"><nd ref="2"/><nd ref="5"/></way>
  <way id="21"><nd ref="4"/><nd ref="6"/></way>

  <relation id="100">
    <member type="way" ref="10" role="left"/>
    <member type="way" ref="20" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>

  <relation id="200">
    <member type="way" ref="11" role="left"/>
    <member type="way" ref="21" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="location" v="intersection"/>
    <tag k="turn_direction" v="left"/>
  </relation>
</osm>
"""


_LOCAL_XY_LANELET_OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="pytest">
  <node id="1" lat="0.0" lon="0.0">
    <tag k="local_x" v="1.0" />
    <tag k="local_y" v="2.0" />
  </node>
  <node id="2" lat="0.0" lon="0.0">
    <tag k="local_x" v="2.0" />
    <tag k="local_y" v="2.0" />
  </node>
  <node id="3" lat="0.0" lon="0.0">
    <tag k="local_x" v="1.0" />
    <tag k="local_y" v="1.0" />
  </node>
  <node id="4" lat="0.0" lon="0.0">
    <tag k="local_x" v="2.0" />
    <tag k="local_y" v="1.0" />
  </node>

  <way id="10"><nd ref="1"/><nd ref="2"/></way>
  <way id="20"><nd ref="3"/><nd ref="4"/></way>

  <relation id="100">
    <member type="way" ref="10" role="left"/>
    <member type="way" ref="20" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>
</osm>
"""


_LOCAL_XY_CHAIN_OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="pytest">
  <node id="1" lat="0.0" lon="0.0"><tag k="local_x" v="1.0" /><tag k="local_y" v="2.0" /></node>
  <node id="2" lat="0.0" lon="0.0"><tag k="local_x" v="2.0" /><tag k="local_y" v="2.0" /></node>
  <node id="3" lat="0.0" lon="0.0"><tag k="local_x" v="1.0" /><tag k="local_y" v="1.0" /></node>
  <node id="4" lat="0.0" lon="0.0"><tag k="local_x" v="2.0" /><tag k="local_y" v="1.0" /></node>
  <node id="5" lat="0.0" lon="0.0"><tag k="local_x" v="3.0" /><tag k="local_y" v="2.0" /></node>
  <node id="6" lat="0.0" lon="0.0"><tag k="local_x" v="3.0" /><tag k="local_y" v="1.0" /></node>

  <way id="10"><nd ref="1"/><nd ref="2"/></way>
  <way id="20"><nd ref="3"/><nd ref="4"/></way>
  <way id="11"><nd ref="2"/><nd ref="5"/></way>
  <way id="21"><nd ref="4"/><nd ref="6"/></way>

  <relation id="100">
    <member type="way" ref="10" role="left"/>
    <member type="way" ref="20" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>

  <relation id="200">
    <member type="way" ref="11" role="left"/>
    <member type="way" ref="21" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>
</osm>
"""


_LOCAL_XY_OPPOSITE_OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="pytest">
  <node id="1" lat="0.0" lon="0.0"><tag k="local_x" v="1.0" /><tag k="local_y" v="2.1" /></node>
  <node id="2" lat="0.0" lon="0.0"><tag k="local_x" v="2.0" /><tag k="local_y" v="2.1" /></node>
  <node id="3" lat="0.0" lon="0.0"><tag k="local_x" v="1.0" /><tag k="local_y" v="1.9" /></node>
  <node id="4" lat="0.0" lon="0.0"><tag k="local_x" v="2.0" /><tag k="local_y" v="1.9" /></node>
  <node id="5" lat="0.0" lon="0.0"><tag k="local_x" v="2.0" /><tag k="local_y" v="1.1" /></node>
  <node id="6" lat="0.0" lon="0.0"><tag k="local_x" v="1.0" /><tag k="local_y" v="1.1" /></node>
  <node id="7" lat="0.0" lon="0.0"><tag k="local_x" v="2.0" /><tag k="local_y" v="0.9" /></node>
  <node id="8" lat="0.0" lon="0.0"><tag k="local_x" v="1.0" /><tag k="local_y" v="0.9" /></node>

  <way id="10"><nd ref="1"/><nd ref="2"/></way>
  <way id="20"><nd ref="3"/><nd ref="4"/></way>
  <way id="11"><nd ref="5"/><nd ref="6"/></way>
  <way id="21"><nd ref="7"/><nd ref="8"/></way>

  <relation id="100">
    <member type="way" ref="10" role="left"/>
    <member type="way" ref="20" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>

  <relation id="200">
    <member type="way" ref="11" role="left"/>
    <member type="way" ref="21" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>
</osm>
"""


_LOCAL_XY_UNREACHABLE_DEST_FALLBACK_OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="pytest">
  <node id="1" lat="0.0" lon="0.0"><tag k="local_x" v="1.0" /><tag k="local_y" v="2.0" /></node>
  <node id="2" lat="0.0" lon="0.0"><tag k="local_x" v="2.0" /><tag k="local_y" v="2.0" /></node>
  <node id="3" lat="0.0" lon="0.0"><tag k="local_x" v="1.0" /><tag k="local_y" v="1.0" /></node>
  <node id="4" lat="0.0" lon="0.0"><tag k="local_x" v="2.0" /><tag k="local_y" v="1.0" /></node>
  <node id="5" lat="0.0" lon="0.0"><tag k="local_x" v="3.0" /><tag k="local_y" v="2.0" /></node>
  <node id="6" lat="0.0" lon="0.0"><tag k="local_x" v="3.0" /><tag k="local_y" v="1.0" /></node>
  <node id="7" lat="0.0" lon="0.0"><tag k="local_x" v="2.1" /><tag k="local_y" v="1.8" /></node>
  <node id="8" lat="0.0" lon="0.0"><tag k="local_x" v="3.1" /><tag k="local_y" v="1.8" /></node>
  <node id="9" lat="0.0" lon="0.0"><tag k="local_x" v="2.1" /><tag k="local_y" v="1.4" /></node>
  <node id="10" lat="0.0" lon="0.0"><tag k="local_x" v="3.1" /><tag k="local_y" v="1.4" /></node>

  <way id="10"><nd ref="1"/><nd ref="2"/></way>
  <way id="20"><nd ref="3"/><nd ref="4"/></way>
  <way id="11"><nd ref="2"/><nd ref="5"/></way>
  <way id="21"><nd ref="4"/><nd ref="6"/></way>
  <way id="30"><nd ref="7"/><nd ref="8"/></way>
  <way id="40"><nd ref="9"/><nd ref="10"/></way>

  <relation id="100">
    <member type="way" ref="10" role="left"/>
    <member type="way" ref="20" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>

  <relation id="200">
    <member type="way" ref="11" role="left"/>
    <member type="way" ref="21" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>

  <relation id="300">
    <member type="way" ref="30" role="left"/>
    <member type="way" ref="40" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>
</osm>
"""


_LOCAL_XY_OVERLAPPED_DEST_OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="pytest">
  <node id="1" lat="0.0" lon="0.0"><tag k="local_x" v="0.0" /><tag k="local_y" v="2.0" /></node>
  <node id="2" lat="0.0" lon="0.0"><tag k="local_x" v="1.0" /><tag k="local_y" v="2.0" /></node>
  <node id="3" lat="0.0" lon="0.0"><tag k="local_x" v="0.0" /><tag k="local_y" v="1.0" /></node>
  <node id="4" lat="0.0" lon="0.0"><tag k="local_x" v="1.0" /><tag k="local_y" v="1.0" /></node>
  <node id="5" lat="0.0" lon="0.0"><tag k="local_x" v="2.0" /><tag k="local_y" v="2.0" /></node>
  <node id="6" lat="0.0" lon="0.0"><tag k="local_x" v="2.0" /><tag k="local_y" v="1.0" /></node>

  <way id="10"><nd ref="1"/><nd ref="2"/></way>
  <way id="20"><nd ref="3"/><nd ref="4"/></way>
  <way id="11"><nd ref="2"/><nd ref="5"/></way>
  <way id="21"><nd ref="4"/><nd ref="6"/></way>
  <way id="30"><nd ref="6"/><nd ref="4"/></way>
  <way id="40"><nd ref="5"/><nd ref="2"/></way>

  <relation id="100">
    <member type="way" ref="10" role="left"/>
    <member type="way" ref="20" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>

  <relation id="200">
    <member type="way" ref="11" role="left"/>
    <member type="way" ref="21" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>

  <relation id="300">
    <member type="way" ref="30" role="left"/>
    <member type="way" ref="40" role="right"/>
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
  </relation>
</osm>
"""


_REPO_PROBLEM_LANELET_IDS = tuple(dict.fromkeys(
    str(item)
    for item in (
        457, 338, 695, 762, 770, 756,
        466, 682, 308, 323, 688, 313,
        783, 841, 735, 776, 849, 445,
        601, 662, 814, 791, 872, 435, 1351, 800, 877,
        1054, 701, 348, 1073, 1077, 1069,
        1097, 1043, 1060, 1086, 745, 1030, 835, 855,
        865, 921, 935, 903, 938, 823, 888, 891,
        645, 882, 946, 580,
        1010, 611, 1024, 714, 725, 1020,
        621, 927, 915, 953,
    )
))


def _reachable_lanelets_from(lanelet_map, start_id: str) -> set[str]:
    seen = {str(start_id)}
    queue = deque([str(start_id)])
    while queue:
        lanelet_id = queue.popleft()
        for successor_id in lanelet_map.successors_of(lanelet_id):
            if successor_id in seen:
                continue
            seen.add(successor_id)
            queue.append(successor_id)
    return seen


def test_load_lanelet2_osm_builds_lanelets_and_topology(tmp_path: Path) -> None:
    osm_path = tmp_path / "mini_lanelet.osm"
    osm_path.write_text(_MINI_LANELET_OSM, encoding="utf-8")

    lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.20)

    assert len(lanelet_map) == 2
    assert lanelet_map.get_lanelet("100") is not None
    assert lanelet_map.get_lanelet("200") is not None
    assert "200" in lanelet_map.successors_of("100")
    assert lanelet_map.get_lanelet("200").attribute != 0


def test_load_lanelet2_osm_prefers_local_xy_tags(tmp_path: Path) -> None:
    osm_path = tmp_path / "local_xy_lanelet.osm"
    osm_path.write_text(_LOCAL_XY_LANELET_OSM, encoding="utf-8")

    lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.20)

    lanelet = lanelet_map.get_lanelet("100")
    assert lanelet is not None
    start = lanelet.centerline[0]
    end = lanelet.centerline[-1]
    assert abs(float(start[0]) - 1.0) < 0.25
    assert abs(float(start[1]) - 1.5) < 0.25
    assert abs(float(end[0]) - 2.0) < 0.25
    assert abs(float(end[1]) - 1.5) < 0.25
    assert lanelet_map.at_pose(1.5, 1.5, max_distance_m=0.6) == "100"


def test_load_lanelet2_osm_preserves_boundary_handedness_after_y_flip(tmp_path: Path) -> None:
    osm_path = tmp_path / "local_xy_lanelet.osm"
    osm_path.write_text(_LOCAL_XY_LANELET_OSM, encoding="utf-8")

    lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.20)
    lanelet = lanelet_map.get_lanelet("100")

    assert lanelet is not None
    left_offsets = [lanelet.project_arclength(float(x), float(y))[1] for x, y in lanelet.left_boundary]
    right_offsets = [lanelet.project_arclength(float(x), float(y))[1] for x, y in lanelet.right_boundary]
    assert median(left_offsets) > 0.0
    assert median(right_offsets) < 0.0


def test_load_lanelet2_osm_keeps_each_lanelet_boundary_geometry(tmp_path: Path) -> None:
    osm_path = tmp_path / "chain_lanelets.osm"
    osm_path.write_text(_LOCAL_XY_CHAIN_OSM, encoding="utf-8")

    lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.10)
    lanelet_100 = lanelet_map.get_lanelet("100")
    lanelet_200 = lanelet_map.get_lanelet("200")

    assert lanelet_100 is not None
    assert lanelet_200 is not None
    assert float(lanelet_100.left_boundary[:, 0].mean()) < 1.75
    assert float(lanelet_200.left_boundary[:, 0].mean()) > 2.25
    assert float(lanelet_100.right_boundary[:, 0].mean()) < 1.75
    assert float(lanelet_200.right_boundary[:, 0].mean()) > 2.25


def test_router_uses_yaw_to_avoid_selecting_opposite_direction_lanelet(tmp_path: Path) -> None:
    osm_path = tmp_path / "opposite_lanelets.osm"
    osm_path.write_text(_LOCAL_XY_OPPOSITE_OSM, encoding="utf-8")

    router = OsmRouteGraph(str(osm_path), step_m=0.10, start_lanelet_id="100")

    assert router.find_nearest_node(1.5, 1.5, yaw_rad=0.0) == "100"
    assert router.find_nearest_node(1.5, 1.5, yaw_rad=3.141592653589793) == "200"


def test_router_trims_first_lanelet_to_current_pose(tmp_path: Path) -> None:
    osm_path = tmp_path / "chain_lanelets.osm"
    osm_path.write_text(_LOCAL_XY_CHAIN_OSM, encoding="utf-8")

    router = OsmRouteGraph(str(osm_path), step_m=0.10, start_lanelet_id="100")
    route = router.go_to({"x": 1.6, "y": 1.5, "yaw_rad": 0.0}, {"lanelet_id": "200"})

    assert route.node_ids[:2] == ["100", "200"]
    assert float(route.waypoints[0][0]) > 1.45


def test_path_manager_starts_without_implicit_route(tmp_path: Path) -> None:
    osm_path = tmp_path / "chain_lanelets.osm"
    osm_path.write_text(_LOCAL_XY_CHAIN_OSM, encoding="utf-8")

    router = OsmRouteGraph(str(osm_path), step_m=0.10, start_lanelet_id="100")
    manager = PathManager(router)

    assert manager.route_active is False
    assert manager.active_route is None
    assert manager.destination_node_id is None


def test_path_manager_clear_command_drops_active_route(tmp_path: Path) -> None:
    osm_path = tmp_path / "chain_lanelets.osm"
    osm_path.write_text(_LOCAL_XY_CHAIN_OSM, encoding="utf-8")

    router = OsmRouteGraph(str(osm_path), step_m=0.10, start_lanelet_id="100")
    manager = PathManager(router)

    ok = manager.handle_command(
        {"x": 2.8, "y": 1.5},
        current_pose={"x": 1.2, "y": 1.5, "yaw_rad": 0.0},
    )
    assert ok is True
    assert manager.route_active is True
    assert manager.active_route is not None

    cleared = manager.handle_command({"mode": "clear"})
    assert cleared is True
    assert manager.route_active is False
    assert manager.active_route is None
    assert manager.destination_node_id is None


def test_path_manager_preserves_explicit_lanelet_id_for_direct_xy_commands(tmp_path: Path) -> None:
    osm_path = tmp_path / "chain_lanelets.osm"
    osm_path.write_text(_LOCAL_XY_CHAIN_OSM, encoding="utf-8")

    router = OsmRouteGraph(str(osm_path), step_m=0.10, start_lanelet_id="100")
    manager = PathManager(router)

    ok = manager.handle_command(
        {"x": 2.8, "y": 1.5, "lanelet_id": "200"},
        current_pose={"x": 1.2, "y": 1.5, "yaw_rad": 0.0},
    )

    assert ok is True
    assert manager.route_active is True
    assert manager.destination_node_id == "200"
    assert manager.destination_specs == [{"x": 2.8, "y": 1.5, "lanelet_id": "200"}]
    assert manager.active_route is not None
    assert manager.active_route.node_ids[-1] == "200"


def test_router_falls_back_to_nearest_reachable_lanelet_when_explicit_destination_is_disconnected(
    tmp_path: Path,
) -> None:
    osm_path = tmp_path / "unreachable_dest_fallback.osm"
    osm_path.write_text(_LOCAL_XY_UNREACHABLE_DEST_FALLBACK_OSM, encoding="utf-8")

    router = OsmRouteGraph(str(osm_path), step_m=0.10, start_lanelet_id="100")
    route = router.go_to(
        {"x": 1.2, "y": 1.5, "yaw_rad": 0.0},
        {"x": 2.6, "y": 1.6, "lanelet_id": "300"},
    )

    assert route.source == "go_to_reachable_fallback"
    assert route.node_ids == ["100", "200"]


def test_router_prefers_short_reachable_xy_lanelet_over_overlapped_wrong_direction(
    tmp_path: Path,
) -> None:
    osm_path = tmp_path / "overlapped_destination.osm"
    osm_path.write_text(_LOCAL_XY_OVERLAPPED_DEST_OSM, encoding="utf-8")

    router = OsmRouteGraph(str(osm_path), step_m=0.10, start_lanelet_id="100")
    route = router.go_to(
        {"lanelet_id": "100"},
        {"x": 1.6, "y": 1.5, "lanelet_id": "300"},
    )

    assert route.source == "go_to_reachable_xy"
    assert route.node_ids == ["100", "200"]


def test_load_lanelet2_osm_exposes_map_metadata_from_osm_bounds(tmp_path: Path) -> None:
    osm_path = tmp_path / "local_xy_lanelet.osm"
    osm_path.write_text(_LOCAL_XY_LANELET_OSM, encoding="utf-8")

    lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.20)
    metadata = lanelet_map.get_map_metadata()

    assert metadata["source"] == "lanelet2_osm"
    assert metadata["y_axis_inverted"] is False
    assert metadata["world_bounds"]["x_min"] == 1.0
    assert metadata["world_bounds"]["x_max"] == 2.0
    assert metadata["world_bounds"]["y_min"] == 1.0
    assert metadata["world_bounds"]["y_max"] == 2.0
    assert metadata["width_m"] == 1.0
    assert metadata["height_m"] == 1.0


def test_repo_sim_map_uses_contiguous_successors_for_reference_loop() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"

    lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.05)
    assert lanelet_map.successors_of("50") == ("55",)
    assert lanelet_map.successors_of("55") == ("60",)
    assert lanelet_map.successors_of("60") == ("70",)
    assert lanelet_map.successors_of("70") == ("75",)
    assert lanelet_map.successors_of("75") == ("85",)
    assert "120" in lanelet_map.successors_of("90")
    assert "125" in lanelet_map.successors_of("120")
    assert "138" in lanelet_map.successors_of("125")
    assert "125" in lanelet_map.predecessors_of("138")

    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="50")
    assert router.reference_node_ids[:4] == ["50", "55", "60", "70"]


def test_repo_sim_map_topology_is_reciprocal() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.05)

    for lanelet_id in lanelet_map.lanelet_ids:
        for successor_id in lanelet_map.successors_of(lanelet_id):
            assert lanelet_id in lanelet_map.predecessors_of(successor_id)
        for predecessor_id in lanelet_map.predecessors_of(lanelet_id):
            assert lanelet_id in lanelet_map.successors_of(predecessor_id)


def test_repo_sim_problem_lanelets_are_reachable_from_reference_start() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.05)
    reachable = _reachable_lanelets_from(lanelet_map, "50")

    missing = [lanelet_id for lanelet_id in _REPO_PROBLEM_LANELET_IDS if lanelet_id not in lanelet_map.lanelet_ids]
    unreachable = [
        lanelet_id
        for lanelet_id in _REPO_PROBLEM_LANELET_IDS
        if lanelet_id in lanelet_map.lanelet_ids and lanelet_id not in reachable
    ]

    assert missing == ["1351"]
    assert not unreachable


def test_repo_sim_problem_lanelets_route_without_fallback() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="50")

    for lanelet_id in _REPO_PROBLEM_LANELET_IDS:
        if lanelet_id not in router.lanelet_map.lanelet_ids:
            continue
        route = router.go_to({"lanelet_id": "50"}, {"lanelet_id": lanelet_id})
        assert route.source == "go_to"
        assert route.node_ids
        assert route.node_ids[-1] == lanelet_id


def test_repo_sim_dense_routes_bridge_small_lanelet_gaps() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="50")

    route = router.go_to({"lanelet_id": "50"}, {"lanelet_id": "466"})
    gaps = [
        math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))
        for a, b in zip(route.waypoints, route.waypoints[1:])
    ]

    assert route.source == "go_to"
    assert route.node_ids[-1] == "466"
    assert max(gaps) <= 0.12


def test_repo_sim_map_boundaries_are_left_positive_after_y_flip() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.05)

    for lanelet_id in ("1187", "221", "524", "526", "282", "298", "303"):
        lanelet = lanelet_map.get_lanelet(lanelet_id)
        assert lanelet is not None
        left_step = max(1, lanelet.left_boundary.shape[0] // 8)
        right_step = max(1, lanelet.right_boundary.shape[0] // 8)
        left_offsets = [
            lanelet.project_arclength(float(x), float(y))[1]
            for x, y in lanelet.left_boundary[::left_step]
        ]
        right_offsets = [
            lanelet.project_arclength(float(x), float(y))[1]
            for x, y in lanelet.right_boundary[::right_step]
        ]
        assert median(left_offsets) > 0.0
        assert median(right_offsets) < 0.0


def test_repo_sim_map_can_route_to_lanelet_138_without_fallback() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"

    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="50")
    lanelet = router.lanelet_map.get_lanelet("138")
    assert lanelet is not None
    mid = lanelet.centerline[lanelet.centerline.shape[0] // 2]

    route = router.go_to(
        {"lanelet_id": "50"},
        {"x": float(mid[0]), "y": float(mid[1]), "lanelet_id": "138"},
    )

    assert route.source == "go_to"
    assert route.node_ids == ["50", "55", "60", "70", "75", "85", "90", "120", "125", "138"]
