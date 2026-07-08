"""Unit tests for the Lyapunov queues + per-frame cost-rate (paper Sec I.D / Sec IV)."""

import unittest

from car_dreamer.toolkit.wam import (
    BudgetVirtualQueue,
    CostRateBreakdown,
    LinkQueue,
    LyapunovConfig,
    LyapunovState,
    action_cost_rate,
)


class LinkQueueTest(unittest.TestCase):
    def test_recurrence_and_clamp(self):
        q = LinkQueue()
        # arrival 100, no service -> 100
        self.assertAlmostEqual(q.update(service_bits=0.0, arrival_bits=100.0), 100.0)
        # service 30, arrival 0 -> 70
        self.assertAlmostEqual(q.update(service_bits=30.0, arrival_bits=0.0), 70.0)
        # over-drain clamps at 0 before adding arrival
        self.assertAlmostEqual(q.update(service_bits=1000.0, arrival_bits=5.0), 5.0)


class BudgetVirtualQueueTest(unittest.TestCase):
    def test_grows_over_budget_shrinks_under(self):
        z = BudgetVirtualQueue()
        self.assertAlmostEqual(z.update(allocated=1.0, budget=0.5), 0.5)   # over budget -> grow
        self.assertAlmostEqual(z.update(allocated=1.0, budget=0.5), 1.0)
        self.assertAlmostEqual(z.update(allocated=0.0, budget=0.5), 0.5)   # under budget -> drain
        self.assertAlmostEqual(z.update(allocated=0.0, budget=0.5), 0.0)
        self.assertAlmostEqual(z.update(allocated=0.0, budget=0.5), 0.0)   # clamp at 0


class LyapunovStateTest(unittest.TestCase):
    def test_advance_slot_and_snapshot(self):
        st = LyapunovState(LyapunovConfig(budget_bandwidth=0.5))
        st.advance_slot(
            per_member_service_bits={7: 0.0},
            per_member_arrival_bits={7: 40.0},
            allocated_bandwidth=1.0,
        )
        self.assertAlmostEqual(st.link_backlog(7), 40.0)
        self.assertAlmostEqual(st.z.value, 0.5)
        st.advance_slot(
            per_member_service_bits={7: 10.0},
            per_member_arrival_bits={7: 0.0},
            allocated_bandwidth=0.0,
        )
        self.assertAlmostEqual(st.link_backlog(7), 30.0)
        self.assertAlmostEqual(st.z.value, 0.0)  # 0.5 + (0 - 0.5)
        snap = st.snapshot()
        self.assertEqual(snap["slots"], 2.0)
        self.assertAlmostEqual(snap["mean_backlog"], (40.0 + 30.0) / 2)
        self.assertAlmostEqual(snap["time_avg_bandwidth"], (1.0 + 0.0) / 2)
        self.assertAlmostEqual(snap["z_over_t"], 0.0 / 2)


class CostRateTest(unittest.TestCase):
    def test_numeric_terms_and_F_normalization(self):
        br = action_cost_rate(
            horizon_slots=4,
            lam=2.0,
            c0=0.5,
            per_slot_uncertainty=[0.1, 0.2, 0.3, 0.4],   # sum = 1.0
            per_slot_bandwidth=[0.5, 0.5, 0.0, 0.0],     # sum = 1.0
            z=3.0,
            link_backlogs={7: 10.0},
            per_member_arrival_bits={7: 8.0},
            per_member_service_bits={7: 5.0},
        )
        self.assertAlmostEqual(br.planning_rate, 2.0 * 0.5 / 4)       # Λc0/F
        self.assertAlmostEqual(br.uncertainty_rate, 2.0 * 1.0 / 4)    # (Λ/F)ΣŨ
        self.assertAlmostEqual(br.bandwidth_rate, 3.0 * 1.0 / 4)      # (Z/F)ΣB
        self.assertAlmostEqual(br.net_load_rate, 10.0 * (8.0 - 5.0) / 4)  # (1/F)ΣQ(L̂-R̂Ts)
        self.assertAlmostEqual(
            br.total, br.planning_rate + br.uncertainty_rate + br.bandwidth_rate + br.net_load_rate
        )

    def test_local_only_zeroes_bandwidth_and_load(self):
        br = action_cost_rate(
            horizon_slots=5, lam=1.0, c0=0.5,
            per_slot_uncertainty=[0.4] * 5,
            per_slot_bandwidth=[0.0] * 5,     # no bandwidth used
            z=100.0,                          # even with a large price, bandwidth term stays 0
            link_backlogs={}, per_member_arrival_bits={}, per_member_service_bits={},
        )
        self.assertEqual(br.bandwidth_rate, 0.0)
        self.assertEqual(br.net_load_rate, 0.0)
        self.assertAlmostEqual(br.uncertainty_rate, 0.4)
        self.assertAlmostEqual(br.planning_rate, 0.5 / 5)

    def test_monotone_in_price_and_backlog(self):
        def rate(z, q):
            return action_cost_rate(
                horizon_slots=4, lam=1.0, c0=0.0,
                per_slot_uncertainty=[0.0] * 4, per_slot_bandwidth=[1.0] * 4,
                z=z, link_backlogs={7: q},
                per_member_arrival_bits={7: 10.0}, per_member_service_bits={7: 2.0},
            ).total
        self.assertGreater(rate(5.0, 1.0), rate(1.0, 1.0))   # increasing in Z
        self.assertGreater(rate(1.0, 5.0), rate(1.0, 1.0))   # increasing in Q_m


if __name__ == "__main__":
    unittest.main()
