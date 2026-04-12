import os
import unittest

from src.hardware.pipeline.sharedTypes import Pose2D
from src.hardware.tracking.threadNavigationPlanner import threadNavigationPlanner
from src.hardware.tracking.pathManager import PathManager
from src.hardware.tracking.trackGraph import TrackGraph


GRAPHML_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Track GraphML File.graphml')
)


class NavigationPlannerThreadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = TrackGraph(GRAPHML_PATH, step_m=0.10)
        cls.manager = PathManager(cls.graph)
        cls.start_x, cls.start_y, cls.start_yaw = cls.graph.get_start_pose()

    def test_build_route_context_keeps_matched_pose_and_semantics(self):
        update = self.manager.update(
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

        route_context = threadNavigationPlanner._build_route_context(update, timestamp=55.0)

        self.assertTrue(route_context.route_active)
        self.assertIsInstance(route_context.matched_pose, Pose2D)
        self.assertAlmostEqual(route_context.matched_pose.x, update.matched_x, places=4)
        self.assertAlmostEqual(route_context.matched_pose.yaw, update.matched_yaw, places=4)
        self.assertEqual(route_context.route_source, update.route_source)
        self.assertEqual(route_context.next_semantic_type, update.next_semantic_type)


if __name__ == "__main__":
    unittest.main()
