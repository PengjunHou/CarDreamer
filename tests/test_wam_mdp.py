import math
import unittest

import torch

from car_dreamer.toolkit.wam import (
    ActionSpec,
    MDPConfig,
    RolloutContext,
    SubAction,
    WorldActionScorer,
    decode_subaction,
    encode_subaction,
    eq12_motion_uncertainty,
    subaction_cost,
    subaction_from_metadata,
)
from car_dreamer.toolkit.wam.graph import VehicleNodeInput
from car_dreamer.toolkit.wam.runtime import ObjectState


def _vehicle(actor_id, x, y, *, is_ego=False, slot=0, vx=0.0, vy=0.0):
    return VehicleNodeInput(
        actor_id=actor_id, is_ego=is_ego, agent_slot=slot,
        x=float(x), y=float(y), z=0.0, vx=float(vx), vy=float(vy), yaw=0.0,
    )


def _object(actor_id, x, y, *, visible_ego=False, collab_ids=()):
    return ObjectState(
        actor_id=actor_id, actor_type="vehicle.test", object_class="vehicle",
        x=float(x), y=float(y), z=0.0, vx=1.0, vy=0.0, yaw=0.0,
        length=4.0, width=2.0, height=1.5,
        visible_to_ego=visible_ego, visible_to_collaborators=tuple(collab_ids),
    )


def _context():
    ego = _vehicle(100, 0.0, 0.0, is_ego=True, vx=3.0)
    collab = _vehicle(7, 20.0, 0.0, slot=1)
    obj = _object(50, 22.0, 1.0, visible_ego=False, collab_ids=(7,))
    return RolloutContext(
        ego=ego, collaborators=(collab,), objects=(obj,),
        route_xy=((0.0, 0.0), (40.0, 0.0)), notable_ids=(50,),
        link_rate_fn=lambda mid, dist, ratio: 1.0e5 * max(float(ratio), 1e-3),
        ego_v0=3.0,
    )


class ActionCodecTest(unittest.TestCase):
    def setUp(self):
        # framing A (default): per-step (S,B,D), fixed step_slots
        self.specA = ActionSpec(max_members=8, bandwidth_grid=(0.2, 0.5, 0.8, 1.0),
                                modalities=("objlist", "bev"), step_slots=2)
        # framing B: (S,B,D,n) with an explicit duration factor
        self.specB = ActionSpec(max_members=8, bandwidth_grid=(0.2, 0.5, 0.8, 1.0),
                                modalities=("objlist", "bev"), duration_grid=(10, 20, 30, 40, 50),
                                use_duration=True)
        self.cands = [7, 3, 9]

    def test_dims(self):
        self.assertEqual(self.specA.dims, (9, 4, 2))         # (M+1, |B|, |D|)
        self.assertEqual(self.specB.dims, (9, 4, 2, 5))      # + |n|

    def test_framingA_cooperative_roundtrip_fixed_duration(self):
        sub = SubAction(selected=(3,), bandwidth_by_vehicle={3: 0.5},
                        modality_by_vehicle={3: "bev"}, duration_slots=40)
        idx = encode_subaction(sub, self.cands, self.specA)
        self.assertEqual(idx, (2, 1, 1))  # 3-tuple, no n
        back = decode_subaction(idx, self.cands, self.specA)
        self.assertEqual(back.selected, (3,))
        self.assertAlmostEqual(back.bandwidth_by_vehicle[3], 0.5)
        self.assertEqual(back.modality_by_vehicle[3], "bev")
        self.assertEqual(back.duration_slots, 2)  # fixed step_slots (n emerges by aggregation)

    def test_framingA_local_only(self):
        idx = encode_subaction(SubAction(selected=(), duration_slots=30), self.cands, self.specA)
        self.assertEqual(idx, (0, 0, 0))
        self.assertTrue(decode_subaction(idx, self.cands, self.specA).is_local_only)

    def test_framingB_cooperative_roundtrip_with_duration(self):
        sub = SubAction(selected=(3,), bandwidth_by_vehicle={3: 0.5},
                        modality_by_vehicle={3: "bev"}, duration_slots=40)
        idx = encode_subaction(sub, self.cands, self.specB)
        self.assertEqual(idx, (2, 1, 1, 3))  # 4-tuple incl. n (40 -> grid index 3)
        back = decode_subaction(idx, self.cands, self.specB)
        self.assertEqual(back.selected, (3,))
        self.assertEqual(back.duration_slots, 40)

    def test_bandwidth_snaps_to_nearest_grid(self):
        sub = SubAction(selected=(7,), bandwidth_by_vehicle={7: 0.47},
                        modality_by_vehicle={7: "objlist"}, duration_slots=10)
        idx = encode_subaction(sub, self.cands, self.specA)
        self.assertEqual(idx[1], 1)  # 0.47 -> nearest 0.5

    def test_member_not_in_candidates_falls_back_to_local(self):
        sub = SubAction(selected=(999,), bandwidth_by_vehicle={999: 1.0},
                        modality_by_vehicle={999: "bev"}, duration_slots=20)
        idx = encode_subaction(sub, self.cands, self.specA)
        self.assertEqual(idx[0], 0)


class Eq12UncertaintyTest(unittest.TestCase):
    def test_equal_weight_over_notable_then_saturate(self):
        # two objects; only object 50 is notable. log_var chosen so TrΣ = exp(lv_x)+exp(lv_y).
        log_var = torch.tensor(
            [[[0.0, 0.0]], [[math.log(2.0), math.log(2.0)]]], dtype=torch.float32
        )  # [Q=2, H=1, 2]
        ids = torch.tensor([50, 51], dtype=torch.long)
        u = eq12_motion_uncertainty(log_var, ids, notable_ids=(50,), sigma0=4.0)
        # only obj 50: TrΣ = 1+1 = 2 -> U = 1 - exp(-2/4)
        self.assertAlmostEqual(u, 1.0 - math.exp(-2.0 / 4.0), places=6)

    def test_two_notable_equal_weight_mean(self):
        log_var = torch.zeros((2, 1, 2), dtype=torch.float32)  # TrΣ = 2 each
        ids = torch.tensor([50, 51], dtype=torch.long)
        u = eq12_motion_uncertainty(log_var, ids, notable_ids=(50, 51), sigma0=4.0)
        self.assertAlmostEqual(u, 1.0 - math.exp(-2.0 / 4.0), places=6)  # mean(2,2)=2

    def test_no_notable_returns_zero(self):
        log_var = torch.zeros((1, 1, 2), dtype=torch.float32)
        ids = torch.tensor([50], dtype=torch.long)
        self.assertEqual(eq12_motion_uncertainty(log_var, ids, notable_ids=(999,), sigma0=4.0), 0.0)

    def test_saturates_to_one_for_large_uncertainty(self):
        log_var = torch.full((1, 1, 2), 8.0, dtype=torch.float32)  # huge variance
        ids = torch.tensor([50], dtype=torch.long)
        u = eq12_motion_uncertainty(log_var, ids, notable_ids=(50,), sigma0=1.0)
        self.assertGreater(u, 0.99)
        self.assertLessEqual(u, 1.0)


class MetadataParserTest(unittest.TestCase):
    """Parse the recorder's ``active_subaction`` dict (as produced by the sub-action recording)."""

    def test_cooperative_metadata(self):
        meta = {"policy_id": 0, "reason": "random_duration", "S": [116],
                "D": {116: ["objlist", "bev"]}, "B": {116: 0.8}, "n": 50}
        sub = subaction_from_metadata(meta, [110, 113, 116, 119], duration_slots=2)
        self.assertEqual(sub.selected, (116,))
        self.assertAlmostEqual(sub.bandwidth_by_vehicle[116], 0.8)
        self.assertEqual(sub.modality_by_vehicle[116], "bev")   # multi-modality collapses to bev
        self.assertEqual(sub.duration_slots, 2)                 # per-step duration, not recorded n=50

    def test_none_and_empty_are_local_only(self):
        self.assertTrue(subaction_from_metadata(None, [1, 2], duration_slots=2).is_local_only)
        self.assertTrue(subaction_from_metadata({"S": []}, [1, 2], duration_slots=2).is_local_only)

    def test_member_absent_from_candidates_is_local_only(self):
        meta = {"S": [999], "B": {999: 1.0}, "D": {999: ["objlist"]}, "n": 10}
        self.assertTrue(subaction_from_metadata(meta, [1, 2, 3], duration_slots=2).is_local_only)

    def test_objlist_only_modality(self):
        meta = {"S": [3], "B": {3: 0.5}, "D": {3: ["objlist"]}, "n": 20}
        sub = subaction_from_metadata(meta, [3], duration_slots=5)
        self.assertEqual(sub.modality_by_vehicle[3], "objlist")


class SubActionRewardTest(unittest.TestCase):
    def setUp(self):
        # rule fallback scorer (no checkpoint) keeps the reward wiring CARLA/model free.
        self.scorer = WorldActionScorer(perception_model=None)
        self.cfg = MDPConfig(lam=1.0, c0=0.5, budget_bandwidth=0.4)
        self.ctx = _context()

    def test_reward_is_negative_p2_cost(self):
        sub = SubAction(selected=(7,), bandwidth_by_vehicle={7: 0.5},
                        modality_by_vehicle={7: "bev"}, duration_slots=20)
        reward, roll, br = subaction_cost(self.ctx, sub, self.scorer, self.cfg, z=1.0)
        self.assertAlmostEqual(reward, -br.total, places=9)
        self.assertEqual(len(roll.per_slot_uncertainty), 20)
        # cooperation allocates bandwidth -> positive bandwidth rate under z>0
        self.assertGreater(br.bandwidth_rate, 0.0)

    def test_local_only_has_zero_bandwidth_and_netload(self):
        sub = SubAction(selected=(), duration_slots=20)
        reward, roll, br = subaction_cost(self.ctx, sub, self.scorer, self.cfg, z=5.0)
        self.assertEqual(br.bandwidth_rate, 0.0)
        self.assertEqual(br.net_load_rate, 0.0)
        self.assertAlmostEqual(reward, -br.total, places=9)


if __name__ == "__main__":
    unittest.main()
