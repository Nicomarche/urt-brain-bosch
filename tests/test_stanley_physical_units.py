import time
import unittest

from src.hardware.camera.threads.threadLineFollowing import threadLineFollowing


class StanleyPhysicalUnitsTests(unittest.TestCase):
    def _make_controller_stub(self):
        controller = threadLineFollowing.__new__(threadLineFollowing)
        controller.lane_width_cm = 35.0
        controller._last_px_per_cm = None
        controller._last_local_ai_lane_width_px = None
        controller.stanley_speed_to_mps = 0.01
        controller.stanley_measured_speed_to_mps = 0.001
        controller.stanley_use_measured_speed = True
        controller.stanley_measured_speed_timeout = 0.4
        controller._measured_speed = 0.0
        controller._last_speed_time = None
        return controller

    def test_convert_error_px_to_m_uses_lane_width_scale(self):
        controller = self._make_controller_stub()

        error_m, px_per_cm = controller._convert_error_px_to_m(
            error_px=30.0,
            lane_width_px=210.0,
            img_w=640,
        )

        self.assertAlmostEqual(px_per_cm, 6.0, places=4)
        self.assertAlmostEqual(error_m, 0.05, places=4)

    def test_resolve_speed_prefers_fresh_measured_speed(self):
        controller = self._make_controller_stub()
        controller._measured_speed = 150.0  # 15.0 cm/s -> 0.15 m/s
        controller._last_speed_time = time.time()

        speed_mps, source = controller._resolve_stanley_speed_mps(
            speed_value=20.0,
            speed_cap=20.0,
        )

        self.assertEqual(source, "measured")
        self.assertAlmostEqual(speed_mps, 0.15, places=4)

    def test_resolve_speed_falls_back_to_command_when_stale(self):
        controller = self._make_controller_stub()
        controller._measured_speed = 150.0
        controller._last_speed_time = time.time() - 1.0  # stale

        speed_mps, source = controller._resolve_stanley_speed_mps(
            speed_value=18.0,
            speed_cap=18.0,
        )

        self.assertEqual(source, "command")
        self.assertAlmostEqual(speed_mps, 0.18, places=4)


if __name__ == "__main__":
    unittest.main()
