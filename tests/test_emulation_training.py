import importlib.util
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

    def test_load_sources_and_build_dataset_splits(self):
        real_source = TRAINING.EpisodeSource(
            path=str(REPO_ROOT / "data" / "vlm_records_terminated_step_96.json"),
            scene_type="right_turn",
            dt=0.1,
        )
        episodes = TRAINING.load_episodes_from_sources(
            [real_source],
            synthetic_episodes=1,
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

    @unittest.skipIf(TRAINING.torch_is_available(), "This guard test only applies when torch is unavailable.")
    def test_fit_emulation_model_requires_torch(self):
        config = TRAINING.EmulationTrainingConfig(
            synthetic_episodes=1,
            max_epochs=1,
            save_dir=str(REPO_ROOT / "tmp_emulation_training_test"),
        )
        with self.assertRaises(ImportError):
            TRAINING.fit_emulation_model(config)


if __name__ == "__main__":
    unittest.main()
