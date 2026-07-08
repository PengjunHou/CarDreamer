"""Unit tests for the Lyapunov scheduler: (P2) selection, no-degradation, event-driven epoch (Sec IV)."""

import math
import unittest

from car_dreamer.toolkit.wam import (
    GraphBuildSpec,
    LyapunovScheduler,
    ObjectState,
    ReferenceTrajectory,
    RolloutContext,
    SchedulerConfig,
    VehicleNodeInput,
    WorldActionScorer,
    candidate_chunks,
    next_epoch_step,
    select_action_chunk,
)

ROUTE = tuple((float(x), 0.0) for x in range(0, 61, 5))


def make_context():
    ego = VehicleNodeInput(actor_id=1, is_ego=True, agent_slot=0, x=0.0, y=0.0, z=0.0,
                           vx=2.0, vy=0.0, yaw=0.0, route_xy=((5.0, 0.0), (10.0, 0.0)))
    member = VehicleNodeInput(actor_id=7, is_ego=False, agent_slot=1, x=25.0, y=0.0, z=0.0,
                              vx=0.0, vy=0.0, yaw=180.0, route_xy=())
    objects = (
        ObjectState(actor_id=100, actor_type="vehicle.x", object_class="vehicle", x=22.0, y=1.0, z=0.0,
                    vx=0.0, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
                    visible_to_ego=False, visible_to_collaborators=(7,)),
    )
    return RolloutContext(
        ego=ego, collaborators=(member,), objects=objects, route_xy=ROUTE, notable_ids=(100,),
        link_rate_fn=lambda mid, dist, ratio: 5.0e5 * max(float(ratio), 0.05),
        graph_spec=GraphBuildSpec(route_waypoints=2), ego_v0=2.0, dt_seconds=0.1, sensor_period_steps=1,
    )


def small_cfg(**kw):
    base = dict(lam=1.0, c0=0.5, budget_bandwidth=0.5, F_max_slots=10, n_min_slots=5,
                bandwidth_grid=(0.5, 1.0), duration_grid=(5,), j_max=1,
                eps_gap=0.1, t_min_slots=2, T_a_slots=3, ts_seconds=0.1)
    base.update(kw)
    return SchedulerConfig(**base)


def rule_scorer():
    return WorldActionScorer(perception_model=None, alpha=0.5)


class CandidateTest(unittest.TestCase):
    def test_includes_local_only_and_feasible(self):
        chunks = candidate_chunks(make_context(), small_cfg())
        self.assertTrue(any(all(sa.is_local_only for sa in c.sub_actions) for c in chunks))  # local-only present
        for c in chunks:
            self.assertLessEqual(c.horizon_slots, 10)
            self.assertTrue(all(sa.duration_slots >= 5 for sa in c.sub_actions))
            self.assertTrue(all(len(sa.selected) <= 1 for sa in c.sub_actions))


class SelectionTest(unittest.TestCase):
    def test_no_degradation(self):
        ctx, cfg, scorer = make_context(), small_cfg(), rule_scorer()
        sched = LyapunovScheduler(cfg, scorer, request_vehicle_id=1)
        chunk, br, _ = select_action_chunk(ctx, scorer=scorer, lyap=sched.lyap, cfg=cfg)
        # local-only baseline cost under the same (zero) queue state
        from car_dreamer.toolkit.wam import action_cost_rate, local_only_chunk
        local = local_only_chunk(n_slots=cfg.F_max_slots, n_min=cfg.n_min_slots)
        roll = scorer.score_chunk(ctx, local)
        local_br = action_cost_rate(
            horizon_slots=local.horizon_slots, lam=cfg.lam, c0=cfg.c0,
            per_slot_uncertainty=roll.per_slot_uncertainty, per_slot_bandwidth=roll.per_slot_bandwidth,
            z=sched.lyap.z.value, link_backlogs=sched.lyap.backlogs(),
            per_member_arrival_bits=roll.per_member_predicted_load_bits,
            per_member_service_bits=roll.per_member_predicted_service_bits,
        )
        self.assertLessEqual(br.total, local_br.total + 1e-9)  # installed never worse than local-only

    def test_high_price_selects_local_only(self):
        ctx, cfg, scorer = make_context(), small_cfg(), rule_scorer()
        sched = LyapunovScheduler(cfg, scorer, request_vehicle_id=1)
        sched.lyap.z.value = 1.0e6  # spectrum extremely expensive -> cooperation priced out
        chunk, _, _ = select_action_chunk(ctx, scorer=scorer, lyap=sched.lyap, cfg=cfg)
        self.assertTrue(all(sa.is_local_only for sa in chunk.sub_actions))


class EpochTest(unittest.TestCase):
    def test_reference_and_next_epoch(self):
        ref = ReferenceTrajectory(install_step=10, horizon_slots=6, per_slot_uncertainty=(0.2,) * 6)
        self.assertAlmostEqual(ref.expected_at(12), 0.2)
        self.assertIsNone(ref.expected_at(99))
        cfg = small_cfg(t_min_slots=2, T_a_slots=1, eps_gap=0.1)
        # no mismatch -> scheduled end
        self.assertEqual(next_epoch_step(ref, realized_uncertainty_fn=lambda s: 0.2, cfg=cfg), 16)
        # large deviation -> early epoch, strictly after install+T_min
        early = next_epoch_step(ref, realized_uncertainty_fn=lambda s: 0.9, cfg=cfg)
        self.assertLess(early, 16)
        self.assertGreater(early, ref.install_step + cfg.t_min_slots)


class RunOfflineTest(unittest.TestCase):
    def test_bounded_queues_and_rows(self):
        cfg, scorer = small_cfg(), rule_scorer()
        sched = LyapunovScheduler(cfg, scorer, request_vehicle_id=1)
        contexts = [make_context() for _ in range(25)]
        rows, summary = sched.run_offline(contexts)
        self.assertEqual(len(rows), 25)
        self.assertGreaterEqual(summary["epochs"], 2)          # F_max=10 over 25 steps
        self.assertTrue(math.isfinite(summary["z_final"]))
        self.assertTrue(math.isfinite(summary["mean_backlog"]))
        self.assertTrue(all(math.isfinite(r["z"]) for r in rows))
        self.assertIn("u_c", summary)

    def test_mismatch_triggers_replans(self):
        cfg, scorer = small_cfg(), rule_scorer()
        sched = LyapunovScheduler(cfg, scorer, request_vehicle_id=1)
        contexts = [make_context() for _ in range(20)]
        # realized U far from any reference -> event-driven re-plans fire after T_min
        _, summary = sched.run_offline(contexts, realized_uncertainty_fn=lambda s: 5.0)
        self.assertGreater(summary["replans"], 0)


if __name__ == "__main__":
    unittest.main()
