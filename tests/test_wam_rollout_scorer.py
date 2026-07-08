"""Unit tests for the world-action rollout scorer (paper Sec III)."""

import math
import unittest

import torch

from car_dreamer.toolkit.wam import (
    ActionChunk,
    GraphBuildSpec,
    ObjectState,
    RolloutContext,
    SubAction,
    VehicleNodeInput,
    WAMPerceptionConfig,
    WAMPerceptionModel,
    WorldActionScorer,
    local_only_chunk,
)

ROUTE = tuple((float(x), 0.0) for x in range(0, 51, 5))


def perc_model():
    torch.manual_seed(0)
    cfg = WAMPerceptionConfig(route_waypoints=2, hidden_dim=32, num_layers=2, num_heads=4,
                              temporal_hidden_dim=32, head_hidden_dim=32, traj_samples=3)
    return WAMPerceptionModel(cfg)


def make_context():
    ego = VehicleNodeInput(actor_id=1, is_ego=True, agent_slot=0, x=0.0, y=0.0, z=0.0,
                           vx=2.0, vy=0.0, yaw=0.0, route_xy=((5.0, 0.0), (10.0, 0.0)))
    member = VehicleNodeInput(actor_id=7, is_ego=False, agent_slot=1, x=18.0, y=0.0, z=0.0,
                              vx=0.0, vy=0.0, yaw=180.0, route_xy=())
    objects = (
        ObjectState(actor_id=100, actor_type="vehicle.x", object_class="vehicle", x=12.0, y=1.0, z=0.0,
                    vx=0.0, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
                    visible_to_ego=False, visible_to_collaborators=(7,)),
        ObjectState(actor_id=101, actor_type="vehicle.x", object_class="vehicle", x=6.0, y=0.5, z=0.0,
                    vx=0.0, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
                    visible_to_ego=True, visible_to_collaborators=()),
    )
    return RolloutContext(
        ego=ego, collaborators=(member,), objects=objects, route_xy=ROUTE, notable_ids=(100, 101),
        link_rate_fn=lambda mid, dist, ratio: 1.0e6 * max(float(ratio), 0.05),
        graph_spec=GraphBuildSpec(route_waypoints=2), ego_v0=2.0, dt_seconds=0.1, sensor_period_steps=1,
    )


def coop_chunk(member=7, bw=0.5, dur=4):
    return ActionChunk((SubAction(selected=(member,), bandwidth_by_vehicle={member: bw},
                                  modality_by_vehicle={member: "bev"}, duration_slots=dur),))


class RolloutScorerModelTest(unittest.TestCase):
    def setUp(self):
        self.scorer = WorldActionScorer(perception_model=perc_model(), history_window=3, alpha=0.5)
        self.ctx = make_context()

    def test_shapes_and_ranges(self):
        res = self.scorer.score_chunk(self.ctx, coop_chunk(dur=4))
        self.assertEqual(len(res.per_slot_uncertainty), 4)
        self.assertTrue(all(math.isfinite(u) for u in res.per_slot_uncertainty))
        self.assertTrue(all(0.0 <= u <= 1.0 for u in res.per_slot_uncertainty))
        self.assertEqual(res.per_slot_bandwidth, [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(len(res.ego_speed_series), 4)
        # cooperative member accrues predicted load + service
        self.assertGreater(res.per_member_predicted_load_bits.get(7, 0.0), 0.0)
        self.assertGreater(res.per_member_predicted_service_bits.get(7, 0.0), 0.0)

    def test_local_only_zero_load_and_bandwidth(self):
        res = self.scorer.score_chunk(self.ctx, local_only_chunk(n_slots=4))
        self.assertEqual(len(res.per_slot_uncertainty), 4)
        self.assertEqual(res.per_slot_bandwidth, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(res.per_member_predicted_load_bits, {})
        self.assertEqual(res.per_member_predicted_service_bits, {})

    def test_j2_chunk_bandwidth_profile(self):
        chunk = ActionChunk((
            SubAction(selected=(7,), bandwidth_by_vehicle={7: 0.8}, modality_by_vehicle={7: "bev"}, duration_slots=2),
            SubAction(duration_slots=3),  # local
        ))
        res = self.scorer.score_chunk(self.ctx, chunk)
        self.assertEqual(len(res.per_slot_uncertainty), 5)
        self.assertEqual(res.per_slot_bandwidth, [0.8, 0.8, 0.0, 0.0, 0.0])


class RolloutScorerRuleFallbackTest(unittest.TestCase):
    def setUp(self):
        self.scorer = WorldActionScorer(perception_model=None, alpha=0.5)  # rule U_phi
        self.ctx = make_context()

    def test_rule_fallback_runs(self):
        res = self.scorer.score_chunk(self.ctx, coop_chunk(dur=3))
        self.assertEqual(len(res.per_slot_uncertainty), 3)
        self.assertTrue(all(math.isfinite(u) and 0.0 <= u <= 1.0 for u in res.per_slot_uncertainty))

    def test_coverage_never_increases_with_a_member(self):
        ego_pose = self.ctx.ego_pose
        member = self.ctx.collaborators[0]
        u_with = self.scorer._coverage_uncertainty(self.ctx, ego_pose, [member], [1.0])
        u_without = self.scorer._coverage_uncertainty(self.ctx, ego_pose, [], [])
        self.assertLessEqual(u_with, u_without + 1e-9)  # adding an observer can only cover more


if __name__ == "__main__":
    unittest.main()
