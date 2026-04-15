import importlib.util
import json
import logging
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_ROOT = REPO_ROOT / "car_dreamer" / "toolkit"
VLM_ROOT = TOOLKIT_ROOT / "vlm"


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


def _install_torch_and_transformers_stubs() -> None:
    torch_stub = sys.modules.get("torch")
    if torch_stub is None:
        torch_stub = types.ModuleType("torch")
        sys.modules["torch"] = torch_stub

    if not hasattr(torch_stub, "Tensor"):
        class _Tensor:
            pass

        torch_stub.Tensor = _Tensor
    if not hasattr(torch_stub, "bfloat16"):
        torch_stub.bfloat16 = object()
    if not hasattr(torch_stub, "float32"):
        torch_stub.float32 = object()

    if not hasattr(torch_stub, "cuda"):
        class _Cuda:
            @staticmethod
            def is_available():
                return False

        torch_stub.cuda = _Cuda()

    if not hasattr(torch_stub, "no_grad"):
        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        torch_stub.no_grad = lambda: _NoGrad()

    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")

        class _AutoProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError("transformers stub should not be used in unit tests")

        class _Qwen:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError("transformers stub should not be used in unit tests")

        transformers_stub.AutoProcessor = _AutoProcessor
        transformers_stub.Qwen2_5_VLForConditionalGeneration = _Qwen
        sys.modules["transformers"] = transformers_stub


def _install_runtime_logging_stub() -> None:
    runtime_logging = types.ModuleType("runtime_logging")
    runtime_logging.get_runtime_logger = logging.getLogger
    runtime_logging.get_runtime_logging_config = lambda: {"step_debug_interval": 1000}
    runtime_logging.should_log_periodic = lambda *args, **kwargs: False
    sys.modules["runtime_logging"] = runtime_logging


def _install_query_mapper_stub() -> None:
    ego_mapper = types.ModuleType("car_dreamer.toolkit.vlm.ego_query_direction_mapper")

    class _Converted:
        def __init__(self, question_id: str):
            self.query = f"converted query: {question_id}"
            self.positive = f"converted positive: {question_id}"
            self.negative = f"converted negative: {question_id}"

    ego_mapper.compute_query_direction_from_observer = (
        lambda ego_pose, observer_pose, question_id: _Converted(question_id)
    )
    sys.modules["car_dreamer.toolkit.vlm.ego_query_direction_mapper"] = ego_mapper


def _load_vlm_module(module_name: str):
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", TOOLKIT_ROOT)
    _ensure_pkg("car_dreamer.toolkit.vlm", VLM_ROOT)
    return _load_module(
        f"car_dreamer.toolkit.vlm.{module_name}",
        VLM_ROOT / f"{module_name}.py",
    )


def _load_prompts_module():
    _install_torch_and_transformers_stubs()
    return _load_vlm_module("right_turn_auto_prompts")


def _load_scoring_module():
    _install_torch_and_transformers_stubs()
    _install_runtime_logging_stub()
    _install_query_mapper_stub()
    _load_vlm_module("right_turn_auto_prompts")
    _load_vlm_module("right_turn_auto_context")
    return _load_vlm_module("right_turn_auto_scoring")


class RightTurnAutoVLMOptimizationTest(unittest.TestCase):
    def test_scene_description_uses_dedicated_budget_and_cache(self):
        prompts = _load_prompts_module()

        class Dummy(prompts.RightTurnAutoVLMPromptMixin):
            def __init__(self):
                self._vlm_enable_step_cache = True
                self._vlm_scene_description_max_new_tokens = 96
                self._time_step = 7
                self._vlm_step_cache = {}
                self.calls = []

            def _run_qwen_generation(self, prompt, image=None, max_new_tokens=None):
                self.calls.append(
                    {
                        "prompt": str(prompt),
                        "image": image,
                        "max_new_tokens": int(max_new_tokens),
                    }
                )
                return "Front: clear"

        dummy = Dummy()
        img_np = np.zeros((8, 8, 3), dtype=np.uint8)
        first = dummy._compute_single_image_description_from_array(
            img_np,
            token_size=3072,
            cache_key=("scene_description", 101),
        )
        second = dummy._compute_single_image_description_from_array(
            img_np,
            token_size=3072,
            cache_key=("scene_description", 101),
        )

        self.assertEqual(first, "Front: clear")
        self.assertEqual(second, "Front: clear")
        self.assertEqual(len(dummy.calls), 1)
        self.assertEqual(dummy.calls[0]["max_new_tokens"], 96)

    def test_parse_multi_query_scores(self):
        prompts = _load_prompts_module()
        parser = prompts.RightTurnAutoVLMPromptMixin()
        result = parser._parse_multi_query_scores(
            json.dumps(
                {
                    "results": {
                        "clg_front_vehicle": {
                            "answer": "positive",
                            "visibility_status": "visible",
                            "question_answerability": "answerable",
                            "support_strength": "strong",
                            "reason": "A vehicle is directly ahead.",
                        },
                        "clg_rear_vehicle": {
                            "answer": "insufficient",
                            "visibility_status": "not_visible",
                            "question_answerability": "not_answerable",
                            "support_strength": "none",
                            "reason": "Rear region is not visible.",
                        },
                    }
                }
            ),
            [
                {"id": "clg_front_vehicle"},
                {"id": "clg_rear_vehicle"},
            ],
        )
        self.assertEqual(set(result.keys()), {"clg_front_vehicle", "clg_rear_vehicle"})
        self.assertEqual(result["clg_front_vehicle"]["answer"], "positive")
        self.assertGreater(result["clg_front_vehicle"]["confidence"], 0.2)
        self.assertEqual(result["clg_rear_vehicle"]["question_answerability"], "not_answerable")

    def test_multi_query_evaluation_reduces_generation_calls(self):
        scoring = _load_scoring_module()

        class _Transform:
            def __init__(self, x: float, y: float, yaw: float):
                self.location = types.SimpleNamespace(x=x, y=y)
                self.rotation = types.SimpleNamespace(yaw=yaw)

        class _Velocity:
            def __init__(self, x: float, y: float):
                self.x = x
                self.y = y

        class _Actor:
            def __init__(self, actor_id: int, x: float, y: float, yaw: float):
                self.id = actor_id
                self._transform = _Transform(x, y, yaw)
                self._velocity = _Velocity(0.0, 0.0)

            def get_transform(self):
                return self._transform

            def get_velocity(self):
                return self._velocity

        class Dummy(scoring.RightTurnAutoVLMScoringMixin):
            def __init__(self):
                self._config = types.SimpleNamespace(
                    world=types.SimpleNamespace(fixed_delta_seconds=0.1)
                )
                self._time_step = 11
                self._vlm_enabled = True
                self._vlm_enable_multi_query_scoring = True
                self._vlm_enable_step_cache = True
                self._vlm_step_cache = {}
                self._vlm_shared_source = "received_feat"
                self._vlm_received_window_s = 2.0
                self._vlm_score_max_new_tokens = 32
                self._vlm_scene_description_max_new_tokens = 16
                self._vlm_do_sample = False
                self._vlm_temperature = 0.0
                self._vlm_top_p = 0.9
                self._vlm_model = object()
                self._vlm_processor = object()
                self._vlm_records = []
                self._vlm_last_eval = {}
                self._vlm_ego_conf_weight = 1.0
                self._vlm_default_shared_conf_weight = 1.0
                self._vlm_shared_conf_weights = {}
                self._vlm_importance_distance_tau = 1.0
                self._vlm_importance_region_weight = 1.0
                self._vlm_importance_facing_weight = 1.0
                self._vlm_importance_distance_weight = 1.0
                self._vlm_importance_ego_bias = 0.0
                self._vlm_sc_beta = 1.0
                self._vlm_sensor_fov_deg = 120.0
                self.feature_size = 3072
                self._emulation_episode_steps = []
                self._emulation_step_counter = 0
                self._vlm_questions = [
                    {
                        "id": "clg_front_vehicle",
                        "type": "clg",
                        "query": "front?",
                        "positive": "front yes",
                        "negative": "front no",
                    },
                    {
                        "id": "clg_rear_vehicle",
                        "type": "clg",
                        "query": "rear?",
                        "positive": "rear yes",
                        "negative": "rear no",
                    },
                ]
                self.ego = _Actor(1, 0.0, 0.0, 0.0)
                self.calls = []

            def _run_qwen_generation(self, prompt, image=None, max_new_tokens=None):
                self.calls.append(
                    {
                        "prompt": str(prompt),
                        "image": bool(image is not None),
                        "max_new_tokens": int(max_new_tokens or 0),
                    }
                )
                if "Describe only safety-relevant facts" in str(prompt):
                    return "Front: visible\nLeft-front: visible\nRight-front: visible\nRear: not_visible\nLeft-rear: not_visible\nRight-rear: not_visible"
                results = {}
                for question_cfg in self._vlm_questions:
                    question_id = str(question_cfg["id"])
                    if "rear" in question_id:
                        answer = "insufficient"
                        visibility = "not_visible"
                        answerability = "not_answerable"
                        strength = "none"
                        reason = "Rear is not visible."
                    else:
                        answer = "positive"
                        visibility = "visible"
                        answerability = "answerable"
                        strength = "moderate"
                        reason = "A vehicle is visible."
                    results[question_id] = {
                        "answer": answer,
                        "visibility_status": visibility,
                        "question_answerability": answerability,
                        "support_strength": strength,
                        "reason": reason,
                    }
                return json.dumps({"results": results})

            def _get_ego_and_shared_images_info(self):
                ego_image = Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8))
                shared_infos = [
                    {
                        "sender_id": 2,
                        "sensor_name": "cam0",
                        "pose": {"x": 8.0, "y": 0.0, "yaw": 0.0},
                        "vel": {"vx": 0.0, "vy": 0.0},
                        "received_age_s": 0.1,
                        "deliver_step": self._time_step,
                        "created_step": self._time_step,
                        "image": None,
                        "img_emb": None,
                        "scene_description": "Front: visible",
                        "text": "observer_message: vehicle ahead",
                    }
                ]
                shared_meta = {
                    "shared_source": "received_feat",
                    "num_candidate_msgs": 1,
                    "num_selected_shared_images": 1,
                    "selected_sender_ids": [2],
                    "window_s": 2.0,
                    "sampling_strategy": "uniform",
                }
                return ego_image, [], shared_infos, shared_meta

            def _maybe_record_emulation_step(self, eval_result, shared_infos, shared_meta):
                self._recorded_eval_result = dict(eval_result)

        dummy = Dummy()
        result = dummy._evaluate_vlm_questions()

        self.assertEqual(len(dummy.calls), 3)
        self.assertEqual(len(result["questions"]), 2)
        self.assertEqual(len(dummy._vlm_records), 2)
        self.assertIn("clg_front_vehicle", result["questions"])
        self.assertIn("clg_rear_vehicle", result["questions"])
        self.assertIn("per_sensor_scores", result["questions"]["clg_front_vehicle"])
        self.assertIn("sender_importance_positive", result["questions"]["clg_front_vehicle"])
        self.assertIn("ego_only", result["questions"]["clg_front_vehicle"])
        self.assertIn("confidence_gain", result["questions"]["clg_front_vehicle"])


if __name__ == "__main__":
    unittest.main()
