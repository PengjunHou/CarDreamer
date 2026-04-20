import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EMULATION_ROOT = REPO_ROOT / "car_dreamer" / "toolkit" / "emulation"


def _ensure_pkg(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
        return
    module.__path__ = [str(path)]


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


COLLECTOR = _load_module("carla_rollout_collector")


class CarlaRolloutCollectorTest(unittest.TestCase):
    def test_build_policy_rollout_argv_includes_policy_and_dump_overrides(self):
        config = COLLECTOR.CARLARolloutCollectorConfig(
            task_name="carla_group_right_turn_auto",
            output_dir="data/emulation",
            episodes_per_policy=1,
            speed_preset="fast_episode",
            task_argv=["--env.vlm.eval_period=2"],
        )
        argv = COLLECTOR.build_policy_rollout_argv(
            config,
            policy_id="P6",
            episode_index=3,
            policy_dir="data/emulation/P6",
        )
        self.assertIn("--env.policy_id=P6", argv)
        self.assertIn("--env.scene_id=P6_scene_0003", argv)
        self.assertIn("--env.emulation_dump_dir=data/emulation/P6", argv)
        self.assertIn("--env.dump_emulation_records_on_episode_end=True", argv)
        self.assertIn("--env.dump_vlm_records_on_episode_end=False", argv)
        self.assertIn("--env.policy_mode=fixed", argv)
        self.assertIn("--env.payload.selector_id=default", argv)

    def test_build_policy_rollout_argv_supports_adaptive_mode(self):
        config = COLLECTOR.CARLARolloutCollectorConfig(
            output_dir="data/emulation",
            policy_mode="adaptive",
            policy_selector_id="default",
            payload_enabled_types=["images", "tokens"],
        )
        argv = COLLECTOR.build_policy_rollout_argv(
            config,
            policy_id="adaptive",
            episode_index=2,
            policy_dir="data/emulation/adaptive",
        )
        self.assertIn("--env.policy_mode=adaptive", argv)
        self.assertIn("--env.scene_id=adaptive_scene_0002", argv)
        self.assertIn("--env.payload.enabled_types=[images,tokens]", argv)

    def test_collect_policy_rollouts_groups_outputs_by_policy(self):
        created = []

        class DummyActionSpace:
            def sample(self):
                return 0

        class DummyEnv:
            action_space = DummyActionSpace()

            def __init__(self, policy_id: str, episode_index: int, out_dir: str):
                self.policy_id = policy_id
                self.episode_index = episode_index
                self.out_dir = Path(out_dir)
                self.step_count = 0

            def reset(self, seed=None):
                self.step_count = 0
                return {}, {}

            def step(self, action):
                del action
                self.step_count += 1
                path = self.out_dir / f"{self.policy_id}_ep_{self.episode_index:04d}.json"
                path.write_text("{}", encoding="utf-8")
                info = {"emulation_dump_path": str(path)}
                return {}, 0.0, True, False, info

            def close(self):
                return None

        def fake_task_factory(task_name, argv):
            self.assertEqual(task_name, "carla_group_right_turn_auto")
            policy_id = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--env.policy_id="))
            scene_id = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--env.scene_id="))
            out_dir = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--env.emulation_dump_dir="))
            episode_index = int(scene_id.rsplit("_", 1)[1])
            created.append((policy_id, scene_id))
            return DummyEnv(policy_id, episode_index, out_dir), {}

        with tempfile.TemporaryDirectory() as tmpdir:
            config = COLLECTOR.CARLARolloutCollectorConfig(
                output_dir=tmpdir,
                policy_ids=["P1", "P8"],
                episodes_per_policy=2,
                max_steps=4,
            )
            saved = COLLECTOR.collect_policy_rollouts(config, task_factory=fake_task_factory)

        self.assertEqual(sorted(saved.keys()), ["P1", "P8"])
        self.assertEqual(len(saved["P1"]), 2)
        self.assertEqual(len(saved["P8"]), 2)
        self.assertIn(("P1", "P1_scene_0000"), created)
        self.assertIn(("P8", "P8_scene_0001"), created)

    def test_rollout_single_episode_forces_dump_when_max_steps_reached(self):
        class DummyActionSpace:
            def sample(self):
                return 0

        class DummyBaseEnv:
            def __init__(self, out_dir: str):
                self._emulation_dump_dir = out_dir
                self._time_step = 0
                self._emulation_episode_dumped = False

            def dump_emulation_episode(self, path: str) -> None:
                Path(path).write_text('{"forced": true}', encoding="utf-8")

        class DummyEnv:
            action_space = DummyActionSpace()

            def __init__(self, out_dir: str):
                self._base = DummyBaseEnv(out_dir)
                self.unwrapped = self._base

            def reset(self, seed=None):
                return {}, {}

            def step(self, action):
                del action
                self._base._time_step += 1
                return {}, 0.0, False, False, {}

        with tempfile.TemporaryDirectory() as tmpdir:
            env = DummyEnv(tmpdir)
            dump_path = COLLECTOR.rollout_single_episode(
                env,
                seed=0,
                max_steps=2,
                force_dump_on_max_steps=True,
            )
            self.assertTrue(Path(dump_path).exists())
            self.assertIn("emulation_episode_max_steps_step_2.json", dump_path)


if __name__ == "__main__":
    unittest.main()
