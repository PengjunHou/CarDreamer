"""Offline tests for the web-monitor concurrency fix (latest-frame slot).

No CARLA needed. The monitor starts a daemon Flask thread on carla_port+7000;
each test uses a distinct high port so the background bind cannot collide.
"""

import time
import types
import unittest

from car_dreamer.toolkit.monitor.monitor import EnvMonitorOpenCV


def _stub_config(port):
    return types.SimpleNamespace(
        world=types.SimpleNamespace(carla_port=port),
        display=types.SimpleNamespace(enable=False, render_keys=[]),
    )


class TestMonitorLatestFrame(unittest.TestCase):
    def test_render_keeps_only_latest(self):
        m = EnvMonitorOpenCV(_stub_config(13581))
        for i in range(1000):
            m.render({"birdeye_wpt": i}, {"step": i})
        # Unbounded queues are gone: only the newest frame is retained.
        self.assertIsNotNone(m._latest)
        _, info = m._latest
        self.assertEqual(info["step"], 999)
        self.assertTrue(m._new_frame.is_set())

    def test_render_pairs_obs_info(self):
        m = EnvMonitorOpenCV(_stub_config(13582))
        obs, info = {"k": 1}, {"v": 2}
        m.render(obs, info)
        self.assertEqual(m._latest, (obs, info))

    def test_stop_returns_immediately(self):
        # Regression: stop() used to join the never-returning Flask server and
        # hang shutdown (crashed evals became port/GPU-holding zombies).
        m = EnvMonitorOpenCV(_stub_config(13583))
        t0 = time.time()
        m.stop()
        self.assertLess(time.time() - t0, 0.5)


if __name__ == "__main__":
    unittest.main()
