import math
import unittest

from src.hardware.camera.threads.threadLineFollowing import StanleyController


class StanleyControllerTests(unittest.TestCase):
    def test_uses_paper_denominator_with_softening_plus_speed(self):
        controller = StanleyController(k=0.04, k_soft=3.0, max_steering=25)

        result = controller.compute(
            crosstrack_error=100.0,
            heading_error=0.0,
            speed=15.0,
        )

        expected = math.degrees(math.atan2(4.0, 18.0))
        self.assertAlmostEqual(result, expected, places=4)

    def test_clamps_with_speed_floor_at_low_speed(self):
        controller = StanleyController(k=0.25, k_soft=1.0, max_steering=25)

        result = controller.compute(
            crosstrack_error=500.0,
            heading_error=0.0,
            speed=0.0,
        )

        self.assertEqual(result, 25)

    def test_applies_steady_state_heading_and_steer_damping_terms(self):
        controller = StanleyController(k=0.0, k_soft=1.0, k_d_steer=0.5, max_steering=25)

        result = controller.compute(
            crosstrack_error=0.0,
            heading_error=0.2,
            speed=10.0,
            steady_state_heading=0.1,
            steer_damping_delta=0.04,
        )

        expected = math.degrees(0.2 - 0.1 + 0.5 * 0.04)
        self.assertAlmostEqual(result, expected, places=4)


if __name__ == "__main__":
    unittest.main()
