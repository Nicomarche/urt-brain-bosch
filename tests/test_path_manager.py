import os
import unittest

from src.hardware.tracking.pathManager import PathManager
from src.hardware.tracking.trackGraph import TrackGraph


GRAPHML_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Track GraphML File.graphml')
)


class PathManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = TrackGraph(GRAPHML_PATH, step_m=0.10)
        cls.reference_ids = [node_id for node_id in cls.graph.reference_node_ids if node_id in cls.graph.nodes]
        cls.start_x, cls.start_y, cls.start_yaw = cls.graph.get_start_pose()

    def setUp(self):
        self.manager = PathManager(self.graph)

    def _update(self):
        return self.manager.update(
            x=self.start_x,
            y=self.start_y,
            yaw=self.start_yaw,
            speed_mps=0.15,
            min_lookahead_m=0.15,
            lookahead_time_s=0.6,
            max_lookahead_m=0.8,
            precision_lookahead_m=0.10,
            lookahead_pts=8,
            search_window=18,
            distance_weight=1.0,
            heading_weight=0.35,
        )

    def test_reset_route_exposes_reference_route_status(self):
        update = self._update()

        self.assertTrue(update.route_active)
        self.assertEqual(update.route_source, "reference")
        self.assertIsNotNone(update.current_node_id)
        self.assertGreater(len(update.route_points), 1)

        status = self.manager.build_navigation_status(update)
        self.assertTrue(status["route_active"])
        self.assertIn("route_points", status)
        self.assertGreater(len(status["route_points"]), 1)
        self.assertIn("map_metadata", status)
        self.assertIn("available_destinations", status)

    def test_handle_go_to_multiple_command_sets_explicit_destination(self):
        mid_id = self.reference_ids[min(8, len(self.reference_ids) - 1)]
        end_id = self.reference_ids[min(18, len(self.reference_ids) - 1)]

        ok = self.manager.handle_command(
            {
                "mode": "go_to_multiple",
                "destinations": [{"node_id": mid_id}, {"node_id": end_id}],
            },
            current_pose={"x": self.start_x, "y": self.start_y},
        )

        self.assertTrue(ok)
        update = self._update()

        self.assertTrue(update.route_active)
        self.assertEqual(update.destination_node_id, end_id)
        self.assertEqual(update.route_source, "go_to_multiple")
        self.assertIsNotNone(update.current_node_id)
        self.assertGreaterEqual(update.remaining_distance_m, 0.0)
        self.assertIsInstance(update.route_queue, list)


if __name__ == "__main__":
    unittest.main()
