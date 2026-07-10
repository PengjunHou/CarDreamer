import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from car_dreamer.v2v_comm_mixin import V2VCommMixin
from car_dreamer.toolkit.wam import (
    ObjectState,
    build_coop_request,
    build_placeholder_policy,
    predict_notable_motion,
    select_notable_objects,
)


def make_object(actor_id, x, y, *, visible=False, collaborator_ids=()):
    return ObjectState(
        actor_id=actor_id,
        actor_type="vehicle.test",
        object_class="vehicle",
        x=float(x),
        y=float(y),
        z=0.0,
        vx=1.0,
        vy=0.0,
        yaw=0.0,
        length=4.0,
        width=2.0,
        height=1.5,
        visible_to_ego=visible,
        visible_to_collaborators=tuple(collaborator_ids),
    )


class WAMRuntimeTest(unittest.TestCase):
    def test_select_notable_objects_keeps_route_near_top_k(self):
        route = [(0.0, 0.0), (20.0, 0.0)]
        objects = [
            make_object(1, 2.0, 1.0),
            make_object(2, 4.0, 3.0),
            make_object(3, 6.0, 12.0),
            make_object(4, 8.0, 0.5),
        ]

        records = select_notable_objects(
            objects,
            route,
            notable_distance_m=10.0,
            max_notable_objects=2,
        )

        self.assertEqual([record.object_state.actor_id for record in records], [4, 1])
        self.assertTrue(all(record.notable for record in records))

    def test_visibility_drives_motion_uncertainty(self):
        records = select_notable_objects(
            [
                make_object(1, 2.0, 0.0, visible=True),
                make_object(2, 3.0, 0.0, visible=False, collaborator_ids=(10,)),
            ],
            [(0.0, 0.0), (10.0, 0.0)],
            notable_distance_m=10.0,
            max_notable_objects=3,
        )

        predictions = predict_notable_motion(
            records,
            dt=0.1,
            horizon_steps=3,
            visible_uncertainty=0.2,
            invisible_uncertainty=2.0,
        )

        self.assertAlmostEqual(predictions[1].uncertainty_score, 0.2)
        self.assertAlmostEqual(predictions[2].uncertainty_score, 2.0)
        self.assertEqual(len(predictions[1].future_xy), 3)

    def test_request_triggers_when_uncertainty_exceeds_threshold(self):
        records = select_notable_objects(
            [make_object(7, 1.0, 0.0, visible=False, collaborator_ids=(11,))],
            [(0.0, 0.0), (10.0, 0.0)],
            notable_distance_m=10.0,
            max_notable_objects=3,
        )
        predictions = predict_notable_motion(
            records,
            dt=0.1,
            horizon_steps=2,
            visible_uncertainty=0.2,
            invisible_uncertainty=2.0,
        )

        request = build_coop_request(
            ego_id=100,
            step=5,
            predictions=predictions,
            uncertainty_threshold=1.0,
        )

        self.assertIsNotNone(request)
        self.assertEqual(request.high_uncertainty_object_ids, (7,))

    def test_placeholder_policy_selects_all_candidates_on_request(self):
        records = select_notable_objects(
            [make_object(7, 1.0, 0.0, visible=False, collaborator_ids=(11,))],
            [(0.0, 0.0), (10.0, 0.0)],
            notable_distance_m=10.0,
            max_notable_objects=3,
        )
        predictions = predict_notable_motion(
            records,
            dt=0.1,
            horizon_steps=2,
            visible_uncertainty=0.2,
            invisible_uncertainty=2.0,
        )
        request = build_coop_request(
            ego_id=100,
            step=5,
            predictions=predictions,
            uncertainty_threshold=1.0,
        )

        policy = build_placeholder_policy(
            request=request,
            candidate_vehicle_ids={21, 20},
            bandwidth_ratio=0.5,
            frequency_steps=1,
            default_modality="objlist",
        )

        self.assertEqual(policy.selected_vehicle_ids, (20, 21))
        self.assertEqual(policy.modality_by_vehicle, {20: "objlist", 21: "objlist"})
        self.assertAlmostEqual(policy.bandwidth_by_vehicle[20], 0.5)
        self.assertAlmostEqual(policy.bandwidth_by_vehicle[21], 0.5)

    def test_placeholder_policy_is_empty_without_request(self):
        policy = build_placeholder_policy(
            request=None,
            candidate_vehicle_ids={20, 21},
            bandwidth_ratio=1.0,
            frequency_steps=1,
            default_modality="objlist",
        )

        self.assertEqual(policy.selected_vehicle_ids, ())
        self.assertEqual(policy.modality_by_vehicle, {})

    def test_checkpoint_predictor_uses_slot_rebuilt_window_to_request_coop(self):
        class DummyGraph:
            def __init__(self, step):
                self.step = step

            def clone(self):
                return self

            def to(self, device):
                del device
                return self

        class DummyModel:
            def __call__(self, window):
                self.window_steps = [graph.step for graph in window]
                return {
                    "object_node_ids": torch.tensor([7, 8], dtype=torch.long),
                    "notable_prob": torch.tensor([1.0, 0.1], dtype=torch.float32),
                    "traj_mu": torch.zeros((2, 3, 2), dtype=torch.float32),
                    "traj_log_var": torch.zeros((2, 3, 2), dtype=torch.float32),
                }

        mixin = object.__new__(V2VCommMixin)
        model = DummyModel()
        mixin._wam_predictor_history_window = 1
        mixin._wam_predictor_sample_period_steps = 2
        mixin._wam_slot_state_history = {10: {"step": 10}, 12: {"step": 12}}
        mixin._comm_config = SimpleNamespace(prediction_window_steps=20, allow_cross_policy_messages=False)
        mixin._wam_active_policy_by_step = {12: 1}
        mixin._wam_received_message_cache = {}
        mixin._ensure_comm_process = lambda: SimpleNamespace(active_policy_id=1)
        mixin._build_wam_graph_for_stage1_slot = lambda state, messages, prediction_step: DummyGraph(state["step"])
        mixin._wam_predictor_device_resolved = torch.device("cpu")
        mixin._wam_predictor_uncertainty_source = "notable_weighted_trace"
        mixin._wam_uncertainty_threshold = 0.5
        mixin.ego = SimpleNamespace(id=100)
        mixin._load_wam_predictor = lambda: model

        mixin._predict_wam_with_checkpoint(step=12)

        self.assertEqual(model.window_steps, [10, 12])
        self.assertIsNotNone(mixin._wam_coop_request)
        self.assertEqual(mixin._wam_coop_request.high_uncertainty_object_ids, (7,))
        self.assertIn(7, mixin._wam_motion_predictions)
        self.assertNotIn(8, mixin._wam_coop_request.high_uncertainty_object_ids)

    def test_checkpoint_slot_message_filter_matches_training_semantics(self):
        def msg(msg_id, *, sender_id, t_sense, t_recv, policy_id=1):
            return SimpleNamespace(
                msg_id=msg_id,
                sender_id=sender_id,
                t_sense=t_sense,
                t_recv=t_recv,
                policy_id=policy_id,
            )

        mixin = object.__new__(V2VCommMixin)
        mixin._comm_config = SimpleNamespace(prediction_window_steps=20, allow_cross_policy_messages=False)
        mixin._wam_active_policy_by_step = {20: 1}
        mixin._wam_received_message_cache = {
            1: msg(1, sender_id=2, t_sense=12, t_recv=18),
            2: msg(2, sender_id=3, t_sense=14, t_recv=22),
            3: msg(3, sender_id=4, t_sense=12, t_recv=19),
            4: msg(4, sender_id=5, t_sense=16, t_recv=17, policy_id=2),
            5: msg(5, sender_id=6, t_sense=-2, t_recv=0),
        }
        mixin._ensure_comm_process = lambda: SimpleNamespace(active_policy_id=1)

        self.assertEqual([m.sender_id for m in mixin._messages_for_checkpoint_slot(12, 20)], [2, 4])
        self.assertEqual(mixin._messages_for_checkpoint_slot(14, 20), [])
        self.assertEqual(mixin._messages_for_checkpoint_slot(16, 20), [])

    def test_random_duration_sampler_builds_valid_comm_policy(self):
        mixin = object.__new__(V2VCommMixin)
        mixin._wam_policy_rng = np.random.default_rng(0)
        mixin._wam_random_policy_local_prob = 0.0
        mixin._wam_random_policy_counts = ("all",)
        mixin._wam_random_policy_modalities = (("bev",),)
        mixin._wam_random_policy_bandwidth_ratios = (0.5,)
        mixin._wam_random_policy_duration_grid = (5,)
        mixin._comm_config = SimpleNamespace(policy_duration_steps=5)
        mixin._comm_policy_counter = 0
        mixin.ego = SimpleNamespace(id=100)

        policy = mixin._sample_random_comm_policy(step=10, candidates=[3, 1, 2])

        self.assertEqual(policy.start_step, 10)
        self.assertEqual(policy.duration_steps, 5)
        self.assertEqual(policy.selected_collaborators, (1, 2, 3))
        self.assertEqual(policy.modalities_by_vehicle, {1: ("bev",), 2: ("bev",), 3: ("bev",)})
        # Shared spectrum: sampled ratio 0.5 split across |S|=3 members -> 0.5/3 each (Σ B_m = 0.5).
        self.assertEqual(policy.bandwidth_by_vehicle, {1: 0.5 / 3, 2: 0.5 / 3, 3: 0.5 / 3})
        self.assertEqual(policy.reason, "random_duration")

    def test_random_duration_policy_lifecycle_respects_td(self):
        class DummyProc:
            def __init__(self):
                self.policy = None
                self.set_count = 0

            def set_policy(self, policy, step):
                del step
                self.policy = policy
                self.set_count += 1

        mixin = object.__new__(V2VCommMixin)
        proc = DummyProc()
        mixin._ensure_comm_process = lambda: proc
        mixin._wam_enabled = True
        mixin._wam_policy_sampler_mode = "random_duration"
        mixin._wam_random_policy_respect_request = False
        mixin._wam_policy_rng = np.random.default_rng(0)
        mixin._wam_random_policy_local_prob = 1.0
        mixin._wam_random_policy_counts = ("all",)
        mixin._wam_random_policy_modalities = (("objlist",),)
        mixin._wam_random_policy_bandwidth_ratios = (1.0,)
        mixin._wam_random_policy_duration_grid = (5,)
        mixin._comm_config = SimpleNamespace(policy_duration_steps=5, sensor_period_steps=1)
        mixin._comm_policy_counter = 0
        mixin._wam_coop_request = None
        mixin.coop_participant_ids = {1, 2}
        mixin.selected_collaborators = set()
        mixin.ego = SimpleNamespace(id=100)

        mixin._update_policy_lifecycle(step=0)
        first_policy_id = proc.policy.policy_id
        mixin._update_policy_lifecycle(step=1)
        mixin._update_policy_lifecycle(step=4)
        self.assertEqual(proc.set_count, 1)
        self.assertEqual(proc.policy.policy_id, first_policy_id)

        mixin._update_policy_lifecycle(step=5)
        self.assertEqual(proc.set_count, 2)
        self.assertNotEqual(proc.policy.policy_id, first_policy_id)


if __name__ == "__main__":
    unittest.main()
