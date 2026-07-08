"""Unit tests for the risk-aware speed controller (paper Sec I.E, eqs 29-32)."""

import unittest

from car_dreamer.toolkit.wam import (
    RiskControlConfig,
    TrackedObject,
    advance_pose_along_route,
    object_future_positions,
    point_at_arclength,
    risk_of_accel,
    select_accel,
    cumulative_lengths,
)

STRAIGHT = [(0.0, 0.0), (100.0, 0.0)]


class GeometryTest(unittest.TestCase):
    def test_advance_pose_arclength_monotone(self):
        pos = advance_pose_along_route([(0.0, 0.0), (10.0, 0.0)], 0.0, [1.0, 1.0, 1.0], dt=1.0)
        self.assertEqual([round(p[0], 3) for p in pos], [1.0, 2.0, 3.0])
        self.assertTrue(all(a[0] <= b[0] for a, b in zip(pos, pos[1:])))  # monotone along route

    def test_object_future_positions(self):
        pos = object_future_positions(TrackedObject(xy0=(0.0, 0.0), vxy=(1.0, 0.0)), horizon_steps=3, dt=1.0)
        self.assertEqual([round(p[0], 3) for p in pos], [1.0, 2.0, 3.0])

    def test_point_at_arclength_clamps(self):
        cum = cumulative_lengths(STRAIGHT)
        self.assertEqual(point_at_arclength(STRAIGHT, cum, -5.0), (0.0, 0.0))    # before start
        self.assertEqual(point_at_arclength(STRAIGHT, cum, 1e9), (100.0, 0.0))  # past end


class ControlTest(unittest.TestCase):
    def test_clear_road_accelerates_to_max(self):
        cfg = RiskControlConfig()
        d = select_accel(ego_v0=2.0, route_xy=STRAIGHT, ego_arc0=0.0, objects=[], cfg=cfg)
        self.assertEqual(d.u_star, max(cfg.accel_set))  # no risk -> full accel

    def test_dead_ahead_object_brakes(self):
        cfg = RiskControlConfig()
        clear = select_accel(ego_v0=5.0, route_xy=STRAIGHT, ego_arc0=0.0, objects=[], cfg=cfg)
        obj = TrackedObject(xy0=(8.0, 0.0), vxy=(0.0, 0.0), radius=1.5, trace_sigma=0.0, weight=3.0)
        blocked = select_accel(ego_v0=5.0, route_xy=STRAIGHT, ego_arc0=0.0, objects=[obj], cfg=cfg)
        self.assertLessEqual(blocked.u_star, 0.0)            # brake / hold
        self.assertLess(blocked.u_star, clear.u_star)        # more conservative than clear road

    def test_risk_increases_with_uncertainty_trace(self):
        cfg = RiskControlConfig()
        near = dict(xy0=(5.0, 0.0), vxy=(0.0, 0.0), radius=1.0, weight=1.0)
        r_low, _ = risk_of_accel(0.0, ego_v0=3.0, route_xy=STRAIGHT, ego_arc0=0.0,
                                 objects=[TrackedObject(trace_sigma=0.0, **near)], cfg=cfg)
        r_high, _ = risk_of_accel(0.0, ego_v0=3.0, route_xy=STRAIGHT, ego_arc0=0.0,
                                  objects=[TrackedObject(trace_sigma=9.0, **near)], cfg=cfg)
        self.assertGreater(r_high, r_low)  # larger TrΣ -> larger r^eff -> higher risk (eq 30)

    def test_waccel_holds_when_at_vmax(self):
        cfg = RiskControlConfig(v_max=8.0, w_accel=0.5)
        # already at v_max, no objects: accel gives no speed gain (clipped) but is penalized -> u*=0
        d = select_accel(ego_v0=8.0, route_xy=STRAIGHT, ego_arc0=0.0, objects=[], cfg=cfg)
        self.assertEqual(d.u_star, 0.0)
        self.assertTrue(all(v <= cfg.v_max + 1e-9 for v in d.speed_series))  # clipping holds


if __name__ == "__main__":
    unittest.main()
