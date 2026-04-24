import importlib.util
import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path


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


TRAINING = _load_module("training")
SCHEMA = _load_module("schema")
SYNTHETIC = _load_module("synthetic")
POLICY = _load_module("policy")
POLICY_GEN = _load_module("policy_data_generator")


class EmulationTrainingTest(unittest.TestCase):
    def test_parse_episode_source_spec(self):
        source = TRAINING.parse_episode_source_spec(
            "data/vlm_records_terminated_step_96.json::right_turn::0.2",
            default_scene_type="lane_change",
            default_dt=0.1,
        )
        self.assertEqual(source.path, "data/vlm_records_terminated_step_96.json")
        self.assertEqual(source.scene_type, "right_turn")
        self.assertAlmostEqual(source.dt, 0.2)

        defaulted = TRAINING.parse_episode_source_spec(
            "data/vlm_records_terminated_step_96.json",
            default_scene_type="left_turn",
            default_dt=0.15,
        )
        self.assertEqual(defaulted.scene_type, "left_turn")
        self.assertAlmostEqual(defaulted.dt, 0.15)

    def test_split_episode_indices(self):
        train_only, val_none = TRAINING.split_episode_indices(1, val_ratio=0.2, seed=0)
        self.assertEqual(train_only, [0])
        self.assertEqual(val_none, [])

        train_idx, val_idx = TRAINING.split_episode_indices(5, val_ratio=0.4, seed=123)
        self.assertEqual(sorted(train_idx + val_idx), [0, 1, 2, 3, 4])
        self.assertGreaterEqual(len(val_idx), 1)
        self.assertGreaterEqual(len(train_idx), 1)

    def test_old_vlm_logs_raise_for_strict_shared_latent_mode(self):
        real_source = TRAINING.EpisodeSource(
            path=str(REPO_ROOT / "data" / "vlm_records_terminated_step_96.json"),
            scene_type="right_turn",
            dt=0.1,
        )
        episodes = TRAINING.load_episodes_from_sources([real_source])
        SCHEMA.validate_episode_record(episodes[0])

        config = TRAINING.EmulationTrainingConfig(history_len=4, horizon=3, val_ratio=0.5, seed=1)
        with self.assertRaisesRegex(ValueError, "shared_latent"):
            TRAINING.build_dataset_splits(episodes, config)

    def test_build_dataset_splits_from_strict_shared_latent_episodes(self):
        episodes = TRAINING.load_episodes_from_sources(
            [],
            synthetic_episodes=2,
            synthetic_scene_type="lane_change",
            synthetic_num_steps=10,
            synthetic_num_vehicles=2,
            seed=5,
        )
        self.assertEqual(len(episodes), 2)
        for episode in episodes:
            SCHEMA.validate_episode_record(episode)

        config = TRAINING.EmulationTrainingConfig(history_len=4, horizon=3, val_ratio=0.5, seed=1)
        train_dataset, val_dataset, train_ids, val_ids = TRAINING.build_dataset_splits(episodes, config)
        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(val_dataset)
        self.assertEqual(sorted(train_ids + val_ids), [0, 1])
        self.assertEqual(train_dataset.max_nodes, val_dataset.max_nodes)
        self.assertEqual(train_dataset.max_queries, val_dataset.max_queries)

    def test_load_episode_from_canonical_json(self):
        episode = SYNTHETIC.generate_synthetic_canonical_episode(num_steps=8, num_vehicles=2, seed=9)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "canonical_episode.json"
            path.write_text(
                __import__("json").dumps(SCHEMA.episode_to_dict(episode), ensure_ascii=False),
                encoding="utf-8",
            )
            loaded = TRAINING.load_episode_from_path(path)
        SCHEMA.validate_episode_record(loaded)
        self.assertEqual(len(loaded.steps), 8)
        self.assertEqual(len(loaded.steps[0].candidate_vehicles), 2)

    def test_unseen_policy_split_uses_policy_ids(self):
        episodes = POLICY_GEN.generate_policy_dataset(
            episodes_per_policy=1,
            num_steps=8,
            num_vehicles=3,
            seed=7,
            policy_ids=["P1", "P5", "P7"],
        )
        config = TRAINING.EmulationTrainingConfig(
            history_len=4,
            horizon=2,
            eval_mode="unseen",
            unseen_policy_ids=["P5", "P7"],
        )
        train_dataset, val_dataset, train_ids, val_ids = TRAINING.build_dataset_splits(episodes, config)
        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(val_dataset)
        self.assertEqual(len(train_ids), 1)
        self.assertEqual(len(val_ids), 2)
        self.assertEqual(episodes[train_ids[0]].policy_id, "P1")
        self.assertEqual(sorted(episodes[idx].policy_id for idx in val_ids), ["P5", "P7"])

    def test_unseen_policy_split_uses_mixed_episode_metadata(self):
        base_episode = SYNTHETIC.generate_synthetic_canonical_episode(num_steps=8, num_vehicles=3, seed=13)
        mixed_episode = copy.deepcopy(base_episode)
        mixed_episode.policy_id = "mixed"
        mixed_episode.metadata = {
            **dict(mixed_episode.metadata or {}),
            "policy_ids_used": ["P1", "P7"],
            "payload_types_used": ["images", "tokens"],
        }
        config = TRAINING.EmulationTrainingConfig(
            history_len=4,
            horizon=2,
            eval_mode="unseen",
            unseen_policy_ids=["P7"],
        )
        _, val_dataset, train_ids, val_ids = TRAINING.build_dataset_splits([base_episode, mixed_episode], config)
        self.assertIsNotNone(val_dataset)
        self.assertEqual(train_ids, [0])
        self.assertEqual(val_ids, [1])

    @unittest.skipIf(TRAINING.torch_is_available(), "This guard test only applies when torch is unavailable.")
    def test_fit_emulation_model_requires_torch(self):
        config = TRAINING.EmulationTrainingConfig(
            synthetic_episodes=1,
            max_epochs=1,
            save_dir=str(REPO_ROOT / "tmp_emulation_training_test"),
        )
        with self.assertRaises(ImportError):
            TRAINING.fit_emulation_model(config)

    @unittest.skipUnless(TRAINING.torch_is_available(), "PyTorch is required for the training smoke test.")
    def test_fit_emulation_model_smoke_with_all_policy_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episodes = POLICY_GEN.generate_policy_dataset(
                episodes_per_policy=1,
                num_steps=10,
                num_vehicles=3,
                seed=11,
                policy_ids=POLICY.list_policy_ids(),
            )
            config = TRAINING.EmulationTrainingConfig(
                history_len=4,
                horizon=2,
                batch_size=4,
                max_epochs=1,
                hidden_dim=16,
                num_graph_layers=1,
                save_dir=tmpdir,
                eval_mode="unseen",
                unseen_policy_ids=["P7", "P8"],
                report_every=0,
            )
            summary = TRAINING.fit_emulation_model(config, episodes=episodes)
            self.assertTrue(Path(summary["best_checkpoint"]).exists())
            self.assertEqual(summary["num_episodes"], len(POLICY.list_policy_ids()))

    @unittest.skipUnless(TRAINING.torch_is_available(), "PyTorch is required for the training smoke test.")
    def test_train_one_epoch_with_missing_shared_latent_masks(self):
        episode = SYNTHETIC.generate_synthetic_canonical_episode(
            scene_type="right_turn",
            num_steps=12,
            num_vehicles=3,
            seed=23,
        )
        shared_dim = len(episode.steps[3].candidate_vehicles[0].shared_latent)
        episode.steps[2].candidate_vehicles[0].shared_latent = [0.0] * shared_dim
        episode.steps[2].candidate_vehicles[0].component_valid_mask["shared_latent"] = False
        episode.steps[3].candidate_vehicles[0].shared_latent = [0.0] * shared_dim
        episode.steps[3].candidate_vehicles[0].component_valid_mask["shared_latent"] = False

        config = TRAINING.EmulationTrainingConfig(
            history_len=4,
            horizon=2,
            batch_size=2,
            hidden_dim=16,
            num_graph_layers=1,
            report_every=0,
            device="cpu",
        )
        train_dataset, _, _, _ = TRAINING.build_dataset_splits([episode], config)
        self.assertIsNotNone(train_dataset)
        loader, _ = TRAINING.build_dataloaders(train_dataset, None, config)
        batch = next(iter(loader))
        self.assertIn("history_shared_latent_mask", batch)
        self.assertIn("future_shared_latent_mask", batch)

        model, _ = TRAINING.make_model_from_dataset(train_dataset, config)
        optimizer = TRAINING.AdamW(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        metrics, _ = TRAINING.train_one_epoch(
            model,
            loader,
            optimizer,
            device="cpu",
            config=config,
        )
        self.assertIn("shared_state_loss", metrics)
        self.assertGreaterEqual(float(metrics["shared_state_loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
