import os
import unittest

from src.hardware.tracking.trackGraph import TrackGraph


GRAPHML_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Track GraphML File.graphml')
)


class TrackGraphPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = TrackGraph(GRAPHML_PATH, step_m=0.10)
        cls.reference_ids = [node_id for node_id in cls.graph.reference_node_ids if node_id in cls.graph.nodes]

    def test_go_to_builds_dense_path_for_reference_destination(self):
        start_id = self.reference_ids[0]
        dest_id = self.reference_ids[min(12, len(self.reference_ids) - 1)]

        route = self.graph.go_to(start_id, {"node_id": dest_id})

        self.assertGreaterEqual(len(route.node_ids), 2)
        self.assertEqual(route.node_ids[0], start_id)
        self.assertEqual(route.node_ids[-1], dest_id)
        self.assertGreater(route.waypoints.shape[0], len(route.node_ids))
        self.assertFalse(route.closed_loop)
        self.assertEqual(route.source, "go_to")
        self.assertIsNotNone(route.destination_point())
        self.assertGreater(len(route.preview_points()), 1)

    def test_go_to_multiple_ends_at_last_destination(self):
        start_id = self.reference_ids[0]
        mid_id = self.reference_ids[min(8, len(self.reference_ids) - 1)]
        end_id = self.reference_ids[min(18, len(self.reference_ids) - 1)]

        route = self.graph.go_to_multiple(start_id, [mid_id, {"node_id": end_id}])

        self.assertGreaterEqual(len(route.node_ids), 3)
        self.assertEqual(route.node_ids[0], start_id)
        self.assertEqual(route.node_ids[-1], end_id)
        self.assertEqual(route.source, "go_to_multiple")
        self.assertGreater(route.waypoints.shape[0], len(route.node_ids))

    def test_named_destination_and_map_metadata_are_available(self):
        route = self.graph.go_to(self.reference_ids[0], {"destination_id": "start"})

        self.assertGreater(route.waypoints.shape[0], 0)
        self.assertIn("meters_per_pixel", route.map_metadata)
        self.assertGreater(len(route.available_destinations), 0)
        self.assertEqual(len(route.wp_semantic_types), route.waypoints.shape[0])


if __name__ == "__main__":
    unittest.main()
