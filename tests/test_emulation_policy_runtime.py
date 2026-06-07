import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = REPO_ROOT / "car_dreamer" / "toolkit" / "policy"


def _ensure_pkg(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
        return
    module.__path__ = [str(path)]


def _load_policy_module():
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", REPO_ROOT / "car_dreamer" / "toolkit")
    full_name = "car_dreamer.toolkit.policy"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, POLICY_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


POLICY = _load_policy_module()


class EmulationPolicyRuntimeTest(unittest.TestCase):
    def test_policies_produce_valid_action_fields(self):
        vehicles = [
            POLICY.VehicleInfo(vehicle_id=101, collaboration_score=0.9, distance_m=8.0),
            POLICY.VehicleInfo(vehicle_id=202, collaboration_score=0.6, distance_m=12.0),
            POLICY.VehicleInfo(vehicle_id=303, collaboration_score=0.4, distance_m=5.0),
            POLICY.VehicleInfo(vehicle_id=404, collaboration_score=0.2, distance_m=18.0),
        ]

        p1 = POLICY.get_policy("P1")(vehicles)
        self.assertTrue(all(value == 0.0 for value in p1.alpha.values()))
        self.assertTrue(all(value == 0.0 for value in p1.nu.values()))
        self.assertTrue(all(value == 0.0 for value in p1.bandwidth.values()))

        for policy_id in ["P2", "P3", "P4", "P5", "P6", "P7", "P8"]:
            action = POLICY.get_policy(policy_id)(vehicles)
            self.assertEqual(set(action.alpha.keys()), {101, 202, 303, 404})
            self.assertEqual(set(action.nu.keys()), {101, 202, 303, 404})
            self.assertEqual(set(action.bandwidth.keys()), {101, 202, 303, 404})
            active_ids = action.active_ids()
            self.assertGreater(len(active_ids), 0)
            self.assertAlmostEqual(sum(action.bandwidth[vid] for vid in active_ids), 1.0, places=6)
            for vid in active_ids:
                self.assertIn(action.nu[vid], {POLICY.NU_LOW, POLICY.NU_MID, POLICY.NU_HIGH})
                self.assertGreater(action.bandwidth[vid], 0.0)


if __name__ == "__main__":
    unittest.main()
