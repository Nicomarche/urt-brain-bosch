from __future__ import annotations

from pathlib import Path

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


def test_load_lanelet2_osm_exposes_map_metadata_from_osm_bounds(tmp_path: Path) -> None:
    osm_path = tmp_path / "local_xy_lanelet.osm"
    osm_path.write_text(_LOCAL_XY_LANELET_OSM, encoding="utf-8")

    lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.20)
    metadata = lanelet_map.get_map_metadata()

    assert metadata["source"] == "lanelet2_osm"
    assert metadata["y_axis_inverted"] is True
    assert metadata["world_bounds"]["x_min"] == 1.0
    assert metadata["world_bounds"]["x_max"] == 2.0
    assert metadata["world_bounds"]["y_min"] == 1.0
    assert metadata["world_bounds"]["y_max"] == 2.0
    assert metadata["width_m"] == 1.0
    assert metadata["height_m"] == 1.0


def test_repo_sim_map_uses_contiguous_successors_for_reference_loop() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"

    lanelet_map = load_lanelet2_osm(str(osm_path), step_m=0.05)
    assert lanelet_map.successors_of("9") == ("22",)
    assert lanelet_map.successors_of("52") == ("1945",)
    assert lanelet_map.successors_of("1765") == ("443",)
    assert "186" in lanelet_map.successors_of("1945")

    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="9")
    assert router.reference_node_ids[:4] == ["9", "22", "33", "1945"]
