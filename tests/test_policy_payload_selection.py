import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_ROOT = REPO_ROOT / "car_dreamer" / "toolkit"
EMULATION_ROOT = TOOLKIT_ROOT / "emulation"
COMM_ROOT = TOOLKIT_ROOT / "communication"


def _ensure_pkg(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
        return
    module.__path__ = [str(path)]


def _load_module(full_name: str, path: Path):
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_emulation_module(module_name: str):
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", TOOLKIT_ROOT)
    _ensure_pkg("car_dreamer.toolkit.emulation", EMULATION_ROOT)
    return _load_module(
        f"car_dreamer.toolkit.emulation.{module_name}",
        EMULATION_ROOT / f"{module_name}.py",
    )


def _load_comm_module(module_name: str):
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", TOOLKIT_ROOT)
    _ensure_pkg("car_dreamer.toolkit.communication", COMM_ROOT)
    return _load_module(
        f"car_dreamer.toolkit.communication.{module_name}",
        COMM_ROOT / f"{module_name}.py",
    )


POLICY = _load_emulation_module("policy")
PAYLOADS = _load_comm_module("payloads")


class PolicyPayloadSelectionTest(unittest.TestCase):
    def test_rule_based_policy_selector_covers_key_rules(self):
        registry = POLICY.build_default_policy_registry()
        selector = POLICY.RuleBasedPolicySelector()

        self.assertEqual(
            selector(POLICY.SceneSummary(num_candidates=0), registry=registry).policy_id,
            "P1",
        )
        self.assertEqual(
            selector(POLICY.SceneSummary(num_candidates=2, avg_link_latency_s=0.3), registry=registry).policy_id,
            "P2",
        )
        self.assertEqual(
            selector(POLICY.SceneSummary(num_candidates=2, min_distance_m=6.0), registry=registry).policy_id,
            "P6",
        )
        self.assertEqual(
            selector(
                POLICY.SceneSummary(num_candidates=2, max_sender_collab=0.7, min_distance_m=15.0),
                registry=registry,
            ).policy_id,
            "P5",
        )
        self.assertEqual(
            selector(
                POLICY.SceneSummary(num_candidates=4, mean_sender_collab=0.1, min_distance_m=20.0),
                registry=registry,
            ).policy_id,
            "P7",
        )
        self.assertEqual(
            selector(
                POLICY.SceneSummary(num_candidates=2, max_sender_collab=0.2, mean_sender_collab=0.3, min_distance_m=20.0),
                registry=registry,
            ).policy_id,
            "P3",
        )

    def test_policy_registry_supports_runtime_registration(self):
        class CustomPolicy(POLICY.FixedCollaborationPolicy):
            policy_id = "PX"

            def __call__(self, vehicles, rng=None):
                del rng
                ids = [vehicle.vehicle_id for vehicle in vehicles]
                return POLICY.CollaborationAction(
                    alpha={vehicle_id: 1.0 for vehicle_id in ids},
                    nu={vehicle_id: POLICY.NU_LOW for vehicle_id in ids},
                    bandwidth={vehicle_id: 1.0 / len(ids) if ids else 0.0 for vehicle_id in ids},
                )

        registry = POLICY.build_default_policy_registry()
        registry.register(CustomPolicy())
        self.assertIn("PX", registry.list_policy_ids())
        action = registry.get("PX")([POLICY.VehicleInfo(vehicle_id=5, sender_collab=0.2, distance_m=8.0)])
        self.assertEqual(action.alpha[5], 1.0)

    def test_collaboration_action_normalizes_only_active_bandwidth(self):
        action = POLICY.CollaborationAction(
            alpha={1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0},
            nu={1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0},
            bandwidth={1: 0.5, 2: 0.5, 3: 0.5, 4: 0.4},
        )

        normalized = action.normalized_bandwidth()

        self.assertAlmostEqual(normalized.bandwidth[1], 1.0 / 3.0)
        self.assertAlmostEqual(normalized.bandwidth[2], 1.0 / 3.0)
        self.assertAlmostEqual(normalized.bandwidth[3], 1.0 / 3.0)
        self.assertEqual(normalized.bandwidth[4], 0.0)

        partial = POLICY.CollaborationAction(
            alpha={1: 1.0, 2: 1.0, 3: 0.0},
            nu={1: 1.0, 2: 1.0, 3: 0.0},
            bandwidth={1: 0.5, 2: 0.2, 3: 0.9},
        ).normalized_bandwidth()
        self.assertEqual(partial.bandwidth[1], 0.5)
        self.assertEqual(partial.bandwidth[2], 0.2)
        self.assertEqual(partial.bandwidth[3], 0.0)

    def test_payload_encoders_and_selector(self):
        registry = PAYLOADS.build_default_payload_registry()
        image = np.full((8, 8, 3), fill_value=120, dtype=np.uint8)
        obs = {"camera": image, "message": "car on right"}

        tokens_encoding = registry.get("tokens").encode(
            sender=None,
            obs=obs,
            feature_size=16,
            image_proc_fn=lambda img, feature_size: "scene ahead clear",
        )
        self.assertEqual(tokens_encoding.payload_type, "tokens")
        self.assertGreater(tokens_encoding.data_nbytes, 0)
        self.assertIn("scene ahead clear", tokens_encoding.text)

        images_encoding = registry.get("images").encode(
            sender=None,
            obs=obs,
            feature_size=16,
            jpeg_quality=70,
        )
        self.assertEqual(images_encoding.payload_type, "images")
        self.assertGreater(images_encoding.data_nbytes, 0)

        decoded = PAYLOADS.decode_payload_dict(images_encoding.to_payload_dict())
        self.assertEqual(decoded["payload_type"], "images")
        self.assertIsInstance(decoded["image"], Image.Image)

        selector = PAYLOADS.RuleBasedPayloadSelector()
        low_link = selector(
            sender_id=1,
            bandwidth=0.8,
            distance_m=5.0,
            latest_comm_stats={"comm_feasible": 0.0, "link_rate_bps": 1.0e6},
            registry=registry,
            enabled_types=["images", "tokens"],
        )
        self.assertEqual(low_link.payload_type, "tokens")
        high_link = selector(
            sender_id=1,
            bandwidth=0.6,
            distance_m=6.0,
            latest_comm_stats={"comm_feasible": 1.0, "link_rate_bps": 8.0e6},
            registry=registry,
            enabled_types=["images", "tokens"],
        )
        self.assertEqual(high_link.payload_type, "images")
        self.assertEqual(PAYLOADS.payload_type_to_one_hot("images").tolist(), [0.0, 0.0, 1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
