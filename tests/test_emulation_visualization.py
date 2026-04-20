import importlib.util
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import sys
import types

from PIL import Image


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


SYNTHETIC = _load_module("synthetic")
DATASET = _load_module("dataset")
MODEL = _load_module("model")
VISUALIZATION = _load_module("visualization")


class EmulationVisualizationTest(unittest.TestCase):
    def test_ground_truth_render_outputs_png_sequence_and_gif(self):
        episode = SYNTHETIC.generate_synthetic_canonical_episode(
            scene_type="right_turn",
            num_steps=8,
            num_vehicles=4,
            seed=3,
        )
        query_id = episode.steps[0].queries[0].query_id

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = VISUALIZATION.render_ground_truth_topology_sequences(
                episode,
                tmpdir,
                metrics=["sender_collab"],
                queries=[query_id],
                step_start=0,
                step_end=2,
                gif_duration_ms=80,
                canvas_size=(320, 320),
            )

            sequence_dir = Path(tmpdir) / "gt" / "sender_collab" / query_id
            png_paths = [
                sequence_dir / "step_000.png",
                sequence_dir / "step_001.png",
                sequence_dir / "step_002.png",
            ]
            gif_path = sequence_dir / "sequence.gif"

            self.assertEqual(summary["frame_counts"]["sender_collab"][query_id], 3)
            for path in png_paths:
                self.assertTrue(path.exists(), str(path))
                with Image.open(path) as image:
                    self.assertEqual(image.size, (320, 320))
            self.assertTrue(gif_path.exists(), str(gif_path))

    def test_prediction_compare_requires_torch_when_torch_is_missing(self):
        if MODEL.torch_is_available():
            self.skipTest("This environment has PyTorch installed; the missing-torch guard is not applicable.")

        episode = SYNTHETIC.generate_synthetic_canonical_episode(
            scene_type="right_turn",
            num_steps=8,
            num_vehicles=3,
            seed=4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ImportError):
                VISUALIZATION.render_prediction_comparison_sequences(
                    episode,
                    Path(tmpdir) / "missing_checkpoint.pt",
                    tmpdir,
                    metrics=["sender_gain"],
                    queries=[episode.steps[0].queries[0].query_id],
                    canvas_size=(320, 320),
                )

    @unittest.skipUnless(MODEL.torch_is_available(), "PyTorch is not installed in this environment.")
    def test_prediction_compare_outputs_png_sequence_and_gif(self):
        import torch

        episode = SYNTHETIC.generate_synthetic_canonical_episode(
            scene_type="right_turn",
            num_steps=10,
            num_vehicles=3,
            seed=5,
        )
        dataset = DATASET.CanonicalEmulationDataset([episode], history_len=4, horizon=3)
        sample = dataset[0]
        model_config = MODEL.GraphGRUEmulationConfig(
            node_dim=int(sample["node_features"].shape[-1]),
            query_dim=int(sample["query_features"].shape[-1]),
            edge_attr_dim=int(sample["edge_attr"].shape[-1]),
            history_len=4,
            horizon=3,
            hidden_dim=32,
            num_graph_layers=2,
            raw_state_dim=int(sample["target_raw_state"].shape[-1]),
            shared_state_dim=int(sample["target_shared_state"].shape[-1]),
        )
        model = MODEL.GraphGRUEmulationModel(model_config)
        query_id = episode.steps[0].queries[0].query_id

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(model_config),
                    "train_config": {
                        "history_len": 4,
                        "horizon": 3,
                        "hidden_dim": 32,
                        "num_graph_layers": 2,
                        "dropout": 0.0,
                    },
                },
                checkpoint_path,
            )

            summary = VISUALIZATION.render_prediction_comparison_sequences(
                episode,
                checkpoint_path,
                tmpdir,
                metrics=["sender_gain"],
                queries=[query_id],
                step_start=1,
                step_end=2,
                horizon=2,
                device="cpu",
                gif_duration_ms=80,
                canvas_size=(240, 240),
            )

            sequence_dir = Path(tmpdir) / "compare" / "sender_gain" / query_id
            self.assertTrue((sequence_dir / "anchor_001_h01.png").exists())
            self.assertTrue((sequence_dir / "anchor_001_h02.png").exists())
            self.assertTrue((sequence_dir / "anchor_002_h01.png").exists())
            self.assertTrue((sequence_dir / "anchor_002_h02.png").exists())
            self.assertTrue((sequence_dir / "sequence.gif").exists())
            self.assertEqual(summary["frame_counts"]["sender_gain"][query_id], 4)


if __name__ == "__main__":
    unittest.main()
