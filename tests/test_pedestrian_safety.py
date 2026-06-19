import importlib.util
import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "car_dreamer" / "toolkit" / "pedestrian_safety.py"
sys.path.insert(0, str(REPO_ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location("pedestrian_safety", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SAFETY = _load_module()


class Vec:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class PedestrianHazardGeometryTest(unittest.TestCase):
    def _hazard(self, walker_location, vehicle_velocity=None, walker_velocity=None):
        return SAFETY.evaluate_pedestrian_hazard(
            Vec(0.0, 0.0),
            Vec(1.0, 0.0),
            vehicle_velocity or Vec(0.0, 0.0),
            walker_location,
            walker_velocity or Vec(0.0, 0.0),
            max_distance_m=12.0,
            front_angle_deg=50.0,
            ttc_threshold_s=2.0,
            brake_distance_m=5.0,
        )

    def test_front_close_pedestrian_is_hazard(self):
        result = self._hazard(Vec(4.0, 0.0))
        self.assertIsNotNone(result)
        distance_m, angle_deg, ttc_s = result
        self.assertAlmostEqual(distance_m, 4.0)
        self.assertAlmostEqual(angle_deg, 0.0)
        self.assertTrue(math.isinf(ttc_s))

    def test_side_rear_pedestrian_is_not_hazard(self):
        self.assertIsNone(self._hazard(Vec(-2.0, 2.0)))

    def test_ttc_threshold_detects_farther_front_pedestrian(self):
        result = self._hazard(Vec(10.0, 0.0), vehicle_velocity=Vec(5.0, 0.0))
        self.assertIsNotNone(result)
        _, _, ttc_s = result
        self.assertAlmostEqual(ttc_s, 2.0)


if __name__ == "__main__":
    unittest.main()
