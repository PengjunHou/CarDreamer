import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = REPO_ROOT / 'car_dreamer' / 'toolkit' / 'vlm' / 'right_turn_auto_prompts.py'
SCORING_PATH = REPO_ROOT / 'car_dreamer' / 'toolkit' / 'vlm' / 'right_turn_auto_scoring.py'


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_prompts_module():
    if 'torch' not in sys.modules:
        torch_stub = types.ModuleType('torch')
        torch_stub.Tensor = object
        torch_stub.bfloat16 = object()
        torch_stub.float32 = object()

        class _Cuda:
            @staticmethod
            def is_available():
                return False

        torch_stub.cuda = _Cuda()
        sys.modules['torch'] = torch_stub

    if 'transformers' not in sys.modules:
        transformers_stub = types.ModuleType('transformers')

        class _AutoProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError('transformers stub should not be used in parser-only tests')

        class _Qwen:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError('transformers stub should not be used in parser-only tests')

        transformers_stub.AutoProcessor = _AutoProcessor
        transformers_stub.Qwen2_5_VLForConditionalGeneration = _Qwen
        sys.modules['transformers'] = transformers_stub

    return _load_module('test_right_turn_auto_prompts_module', PROMPTS_PATH)


def _load_scoring_module():
    for package_name in ['car_dreamer', 'car_dreamer.toolkit', 'car_dreamer.toolkit.vlm']:
        module = sys.modules.get(package_name)
        if module is None:
            module = types.ModuleType(package_name)
            module.__path__ = []
            sys.modules[package_name] = module

    runtime_logging = types.ModuleType('runtime_logging')
    runtime_logging.get_runtime_logger = logging.getLogger
    sys.modules['runtime_logging'] = runtime_logging

    ego_mapper = types.ModuleType('car_dreamer.toolkit.vlm.ego_query_direction_mapper')

    class _Converted:
        def __init__(self, query, positive, negative):
            self.query = query
            self.positive = positive
            self.negative = negative

    ego_mapper.compute_query_direction_from_observer = lambda ego_pose, observer_pose, question_id: _Converted(
        query=f'converted:{question_id}',
        positive=f'converted positive:{question_id}',
        negative=f'converted negative:{question_id}',
    )
    sys.modules['car_dreamer.toolkit.vlm.ego_query_direction_mapper'] = ego_mapper

    context_module = types.ModuleType('car_dreamer.toolkit.vlm.right_turn_auto_context')

    class RightTurnAutoVLMContextMixin:  # pragma: no cover - test stub
        pass

    context_module.RightTurnAutoVLMContextMixin = RightTurnAutoVLMContextMixin
    sys.modules['car_dreamer.toolkit.vlm.right_turn_auto_context'] = context_module

    return _load_module('car_dreamer.toolkit.vlm.right_turn_auto_scoring', SCORING_PATH)


class VLMUnknownScoringTest(unittest.TestCase):
    def test_parse_positive_visual_answer_keeps_positive_signal(self):
        prompts = _load_prompts_module()
        parser = prompts.RightTurnAutoVLMPromptMixin()
        result = parser._parse_language_scores(
            '{"answer": "positive", "visibility_status": "visible", '
            '"question_answerability": "answerable", "support_strength": "moderate", '
            '"reason": "A vehicle is clearly visible in the target region."}'
        )
        self.assertEqual(result['answer'], 'positive')
        self.assertEqual(result['visibility_status'], 'visible')
        self.assertEqual(result['question_answerability'], 'answerable')
        self.assertGreater(result['positive_score'], 0.4)
        self.assertLess(result['unknown_score'], 0.1)
        self.assertGreater(result['confidence'], 0.2)

    def test_parse_insufficient_not_visible_does_not_create_fake_confidence(self):
        prompts = _load_prompts_module()
        parser = prompts.RightTurnAutoVLMPromptMixin()
        result = parser._parse_language_scores(
            '{"answer": "insufficient", "visibility_status": "not_visible", '
            '"question_answerability": "not_answerable", "support_strength": "none", '
            '"reason": "The queried region is outside the camera view."}'
        )
        self.assertEqual(result['answer'], 'uncertain')
        self.assertEqual(result['visibility_status'], 'not_visible')
        self.assertEqual(result['question_answerability'], 'not_answerable')
        self.assertEqual(result['evidence'], 0.0)
        self.assertEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['unknown_score'], 0.25)

    def test_rear_question_skips_ego_forward_camera(self):
        scoring = _load_scoring_module()

        class Dummy(scoring.RightTurnAutoVLMScoringMixin):
            pass

        dummy = Dummy()
        use_sensor, reason = dummy._should_use_sensor_for_question(
            {'id': 'clg_left_rear_vehicle'},
            {'is_ego': True},
            {'question_answerability': 'answerable'},
        )
        self.assertFalse(use_sensor)
        self.assertIn('rear', reason)

    def test_weighted_aggregate_prefers_shared_positive_when_it_is_only_usable_signal(self):
        scoring = _load_scoring_module()

        class Dummy(scoring.RightTurnAutoVLMScoringMixin):
            def __init__(self):
                self._vlm_ego_conf_weight = 1.0
                self._vlm_default_shared_conf_weight = 1.0
                self._vlm_shared_conf_weights = {}

        dummy = Dummy()
        sensor_mean_scores = {
            '204:cam0': {
                'sender_id': 204,
                'is_ego': False,
                'positive_score': 0.7,
                'negative_score': 0.0,
                'unknown_score': 0.2,
                'belief': 0.7,
                'evidence': 0.7,
                'received_age_s_mean': 0.0,
                'answerability_score': 1.0,
                'answer': 'positive',
                'visibility_status': 'visible',
                'question_answerability': 'answerable',
            }
        }
        importance_maps = {'per_sensor': {'204:cam0': {'importance_weight': 1.0}}}
        result = dummy._weighted_confidence_aggregate(sensor_mean_scores, importance_maps)
        self.assertEqual(result['answer'], 'positive')
        self.assertGreater(result['confidence'], 0.1)
        self.assertGreater(result['positive_score'], result['negative_score'])


if __name__ == '__main__':
    unittest.main()
