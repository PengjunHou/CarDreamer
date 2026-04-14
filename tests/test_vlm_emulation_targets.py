import math
import importlib.util
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "car_dreamer" / "toolkit" / "vlm" / "right_turn_auto_emulation.py"
SPEC = importlib.util.spec_from_file_location("right_turn_auto_emulation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

build_emulation_rollout_examples = MODULE.build_emulation_rollout_examples
build_emulation_step_summaries = MODULE.build_emulation_step_summaries
infer_question_order = MODULE.infer_question_order
infer_sender_order = MODULE.infer_sender_order
pack_step_features = MODULE.pack_step_features


EGO_ID = 100


def _make_record(
    step,
    question_id,
    shared_sender_id,
    *,
    ego_conf,
    fused_conf,
    gain,
    importance,
    shared_gain,
    age_s,
    distance_m,
):
    raw_shared_conf = 0.0 if importance <= 0 else shared_gain / importance
    return {
        "step": step,
        "question_id": question_id,
        "question_type": "clg",
        "num_candidate_msgs": 2,
        "num_selected_shared_images": 1,
        "selected_sender_ids": [shared_sender_id],
        "ego_only": {"confidence": ego_conf},
        "ego_plus_shared": {"confidence": fused_conf},
        "confidence_gain": gain,
        "per_sensor_scores": [
            {
                "sender_id": EGO_ID,
                "sensor_name": "cam0",
                "is_ego": True,
                "confidence": ego_conf,
                "importance_positive": 0.8,
                "importance_negative": 0.8,
                "received_age_s_mean": 0.0,
                "distance_m": 0.0,
                "region_alignment": 1.0,
                "facing_alignment": 1.0,
                "distance_alignment": 1.0,
            },
            {
                "sender_id": shared_sender_id,
                "sensor_name": "cam0",
                "is_ego": False,
                "confidence": raw_shared_conf,
                "importance_positive": importance,
                "importance_negative": importance,
                "received_age_s_mean": age_s,
                "distance_m": distance_m,
                "region_alignment": 0.7,
                "facing_alignment": 0.8,
                "distance_alignment": 0.9,
            },
        ],
        "aggregated_details": {
            "per_sender": [
                {"sender_id": EGO_ID, "weighted_confidence": ego_conf},
                {"sender_id": shared_sender_id, "weighted_confidence": shared_gain},
            ]
        },
    }


class RightTurnAutoEmulationTargetsTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            _make_record(
                0,
                "clg_right_front_vehicle",
                200,
                ego_conf=0.10,
                fused_conf=0.16,
                gain=0.06,
                importance=0.50,
                shared_gain=0.06,
                age_s=0.20,
                distance_m=12.0,
            ),
            _make_record(
                0,
                "clg_left_front_vehicle",
                300,
                ego_conf=0.20,
                fused_conf=0.24,
                gain=0.04,
                importance=0.25,
                shared_gain=0.04,
                age_s=0.00,
                distance_m=8.0,
            ),
            _make_record(
                1,
                "clg_right_front_vehicle",
                200,
                ego_conf=0.30,
                fused_conf=0.35,
                gain=0.05,
                importance=0.40,
                shared_gain=0.05,
                age_s=0.10,
                distance_m=10.0,
            ),
            _make_record(
                1,
                "clg_left_front_vehicle",
                300,
                ego_conf=0.15,
                fused_conf=0.21,
                gain=0.06,
                importance=0.60,
                shared_gain=0.06,
                age_s=0.30,
                distance_m=6.0,
            ),
        ]

    def test_infer_orders(self):
        self.assertEqual(
            infer_question_order(self.records),
            ["clg_right_front_vehicle", "clg_left_front_vehicle"],
        )
        self.assertEqual(infer_sender_order(self.records), [200, 300])

    def test_build_step_summaries(self):
        summaries = build_emulation_step_summaries(self.records)
        self.assertEqual(len(summaries), 2)

        step0 = summaries[0]
        np.testing.assert_allclose(step0["ego_sc"], np.array([0.10, 0.20], dtype=np.float32))
        np.testing.assert_allclose(step0["fused_sc"], np.array([0.16, 0.24], dtype=np.float32))
        np.testing.assert_allclose(step0["gain"], np.array([0.06, 0.04], dtype=np.float32))

        expected_collab_200 = 0.50 * math.exp(-0.20)
        expected_collab_300 = 0.25 * math.exp(-0.00)
        self.assertAlmostEqual(float(step0["sender_collab"][0, 0]), expected_collab_200, places=6)
        self.assertAlmostEqual(float(step0["sender_collab"][1, 1]), expected_collab_300, places=6)
        self.assertAlmostEqual(float(step0["sender_gain"][0, 0]), 0.06, places=6)
        self.assertAlmostEqual(float(step0["sender_gain"][1, 1]), 0.04, places=6)
        self.assertEqual(float(step0["sender_available"][0, 0]), 1.0)
        self.assertEqual(float(step0["sender_available"][1, 0]), 0.0)
        self.assertEqual(float(step0["sender_selected"][0, 0]), 1.0)

    def test_pack_step_features_shape(self):
        summaries = build_emulation_step_summaries(self.records)
        features = pack_step_features(summaries[0])
        question_count = len(summaries[0]["question_ids"])
        sender_count = len(summaries[0]["sender_ids"])
        expected_dim = (5 * question_count) + (9 * sender_count * question_count)
        self.assertEqual(features.shape, (expected_dim,))

    def test_build_rollout_examples(self):
        summaries = build_emulation_step_summaries(self.records)
        examples = build_emulation_rollout_examples(summaries, horizon=2)
        self.assertEqual(len(examples), 2)

        first = examples[0]
        np.testing.assert_allclose(first["future_mask"], np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(
            first["future_ego_sc"][0],
            np.array([0.35, 0.21], dtype=np.float32),
        )
        np.testing.assert_allclose(
            first["future_gain"][0],
            np.array([0.05, 0.06], dtype=np.float32),
        )
        self.assertAlmostEqual(float(first["future_sender_gain"][0, 0, 0]), 0.05, places=6)
        self.assertAlmostEqual(float(first["future_sender_gain"][0, 1, 1]), 0.06, places=6)


if __name__ == "__main__":
    unittest.main()
