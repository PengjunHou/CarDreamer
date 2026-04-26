import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EMULATION_ROOT = REPO_ROOT / "car_dreamer" / "toolkit" / "emulation"


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_module(module_name: str):
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", REPO_ROOT / "car_dreamer" / "toolkit")
    _ensure_pkg("car_dreamer.toolkit.communication", REPO_ROOT / "car_dreamer" / "toolkit" / "communication")
    _ensure_pkg("car_dreamer.toolkit.emulation", EMULATION_ROOT)
    full_name = f"car_dreamer.toolkit.emulation.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, EMULATION_ROOT / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SCHEMA = _load_module("schema")
QUERIES = _load_module("queries")
FEATURES = _load_module("features")
SYNTHETIC = _load_module("synthetic")
ADAPTER = _load_module("adapter_vlm")
DATASET = _load_module("dataset")
MODEL = _load_module("model")


class EmulationGraphGRUTest(unittest.TestCase):
    def test_synthetic_episode_roundtrip_and_feature_separation(self):
        episode = SYNTHETIC.generate_synthetic_canonical_episode(
            scene_type="lane_change",
            num_steps=12,
            num_vehicles=4,
            seed=7,
        )
        SCHEMA.validate_episode_record(episode)

        payload = SCHEMA.episode_to_dict(episode)
        reloaded = SCHEMA.episode_from_dict(payload)
        SCHEMA.validate_episode_record(reloaded)

        vehicle = reloaded.steps[0].candidate_vehicles[0]
        node_feature = FEATURES.pack_vehicle_node_state(vehicle)
        self.assertEqual(node_feature.shape[0], 36)
        self.assertEqual(len(vehicle.query_task_relevance), len(reloaded.steps[0].queries))
        self.assertEqual(len(vehicle.shared_summary_raw), 8)
        self.assertEqual(len(vehicle.shared_summary_semantic), 8)
        self.assertEqual(len(vehicle.intent_summary), 4)
        shared_summary = FEATURES.pack_vehicle_shared_state_features(vehicle)
        self.assertEqual(shared_summary.shape[0], 21)
        self.assertEqual(vehicle.payload_type, "tokens")

    def test_adapter_builds_canonical_episode_from_real_vlm_log(self):
        episode = ADAPTER.adapt_vlm_records_to_canonical_episode(
            REPO_ROOT / "data" / "vlm_records_terminated_step_96.json",
            scene_type="right_turn",
        )
        SCHEMA.validate_episode_record(episode)

        self.assertEqual(len(episode.steps), 96)
        self.assertEqual([query.query_id for query in episode.steps[0].queries], [
            "clg_left_rear_vehicle",
            "clg_right_rear_vehicle",
            "clg_right_front_vehicle",
            "clg_left_front_vehicle",
        ])
        self.assertEqual(len(episode.steps[0].candidate_vehicles), 0)
        self.assertEqual([vehicle.vehicle_id for vehicle in episode.steps[1].candidate_vehicles], [873, 876])

        sender = episode.steps[1].candidate_vehicles[0]
        self.assertEqual(len(sender.shared_summary_raw), 8)
        self.assertEqual(len(sender.shared_summary_semantic), 8)
        self.assertEqual(len(sender.intent_summary), 4)
        self.assertEqual(set(sender.query_task_relevance.keys()), {
            "clg_left_rear_vehicle",
            "clg_right_rear_vehicle",
            "clg_right_front_vehicle",
            "clg_left_front_vehicle",
        })

    def test_dataset_outputs_query_conditioned_targets(self):
        episode = SYNTHETIC.generate_synthetic_canonical_episode(
            scene_type="right_turn",
            num_steps=16,
            num_vehicles=3,
            seed=3,
        )
        dataset = DATASET.CanonicalEmulationDataset([episode], history_len=8, horizon=5)
        sample = dataset[9]

        self.assertEqual(sample["node_features"].shape, (8, 3, 36))
        self.assertEqual(sample["state_node_features"].shape, (8, 3, 28))
        self.assertEqual(sample["action_features"].shape, (8, 3, 8))
        self.assertEqual(sample["target_raw_state"].shape, (5, 3, 5))
        self.assertEqual(sample["target_shared_state"].shape, (5, 3, 21))
        self.assertEqual(sample["task_relevance"].shape, (8, 3, 6))
        self.assertEqual(sample["target_sender_collab"].shape, (5, 3, 6))
        self.assertEqual(sample["target_sender_gain"].shape, (5, 3, 6))
        self.assertEqual(sample["target_ego_sc"].shape, (5, 6))
        self.assertEqual(sample["future_node_mask"].shape, (5, 3))
        self.assertEqual(sample["query_mask"].sum(), 6.0)
        self.assertEqual(sample["node_mask"].shape, (8, 3))
        self.assertTrue(np.any(sample["task_relevance"] > 0.0))

    @unittest.skipUnless(MODEL.torch_is_available(), "PyTorch is not installed in this environment.")
    def test_model_forward_and_loss(self):
        import torch

        episode = SYNTHETIC.generate_synthetic_canonical_episode(
            scene_type="left_turn",
            num_steps=14,
            num_vehicles=4,
            seed=11,
        )
        dataset = DATASET.CanonicalEmulationDataset([episode], history_len=4, horizon=3)
        sample = dataset[6]
        config = MODEL.GraphGRUEmulationConfig(
            node_dim=sample["state_node_features"].shape[-1],
            query_dim=sample["query_features"].shape[-1],
            history_len=4,
            horizon=3,
            hidden_dim=32,
            raw_state_dim=sample["target_raw_state"].shape[-1],
            shared_state_dim=sample["target_shared_state"].shape[-1],
        )
        net = MODEL.GraphGRUEmulationModel(config)
        outputs = net(sample)

        self.assertEqual(tuple(outputs["sender_collab"].shape), (1, 3, 4, 6))
        self.assertEqual(tuple(outputs["sender_gain"].shape), (1, 3, 4, 6))
        self.assertEqual(tuple(outputs["ego_sc"].shape), (1, 3, 6))

        diff = torch.abs(outputs["sender_collab"][0, :, 0, 0] - outputs["sender_collab"][0, :, 0, 1]).sum()
        self.assertGreater(float(diff), 1e-8)

        losses = MODEL.compute_emulation_loss(outputs, sample)
        self.assertTrue(torch.isfinite(losses["loss"]))

    @unittest.skipUnless(MODEL.torch_is_available(), "PyTorch is not installed in this environment.")
    def test_compact_summary_shared_state_loss_smoke(self):
        import copy
        import torch

        episode = SYNTHETIC.generate_synthetic_canonical_episode(
            scene_type="right_turn",
            num_steps=12,
            num_vehicles=3,
            seed=17,
        )
        history_vehicle = episode.steps[2].candidate_vehicles[0]
        history_vehicle.shared_summary_raw = [0.0] * len(history_vehicle.shared_summary_raw)
        history_vehicle.shared_summary_semantic = [0.0] * len(history_vehicle.shared_summary_semantic)
        history_vehicle.shared_confidence = 0.0
        history_vehicle.component_valid_mask["shared_summary_raw"] = False
        history_vehicle.component_valid_mask["shared_summary_semantic"] = False
        history_vehicle.component_valid_mask["shared_confidence"] = False

        future_vehicle = episode.steps[3].candidate_vehicles[0]
        future_vehicle.shared_summary_raw = [0.0] * len(future_vehicle.shared_summary_raw)
        future_vehicle.shared_summary_semantic = [0.0] * len(future_vehicle.shared_summary_semantic)
        future_vehicle.shared_confidence = 0.0
        future_vehicle.component_valid_mask["shared_summary_raw"] = False
        future_vehicle.component_valid_mask["shared_summary_semantic"] = False
        future_vehicle.component_valid_mask["shared_confidence"] = False

        dataset = DATASET.CanonicalEmulationDataset([episode], history_len=4, horizon=3)
        sample = dataset[2]
        self.assertEqual(float(sample["future_node_mask"][0, 0]), 1.0)

        config = MODEL.GraphGRUEmulationConfig(
            node_dim=sample["state_node_features"].shape[-1],
            query_dim=sample["query_features"].shape[-1],
            history_len=4,
            horizon=3,
            hidden_dim=32,
            raw_state_dim=sample["target_raw_state"].shape[-1],
            shared_state_dim=sample["target_shared_state"].shape[-1],
        )
        net = MODEL.GraphGRUEmulationModel(config)
        outputs = net(sample)
        losses = MODEL.compute_emulation_loss(outputs, sample)
        self.assertTrue(torch.isfinite(losses["shared_state_loss"]))


if __name__ == "__main__":
    unittest.main()
