import unittest

import numpy as np

from car_dreamer.toolkit.wam import BevSpec, CoverageConfig, build_coverage_raster, coverage_metrics


SPEC = BevSpec(size=32, range_m=40.0)
CFG = CoverageConfig(
    past_route_distance_m=10.0,
    future_route_distance_m=30.0,
    corridor_width_m=8.0,
    coverage_distance_scale_m=20.0,
    route_risk_distance_scale_m=20.0,
    u_prior=1.0,
)


def cell_for(x, y, spec=SPEC):
    ppm = spec.pixels_per_meter
    c0 = spec.size / 2.0
    row = int(round(c0 - x * ppm - 0.5))
    col = int(round(c0 - y * ppm - 0.5))
    return max(0, min(spec.size - 1, row)), max(0, min(spec.size - 1, col))


def raster(*, route=((30.0, 0.0),), past=((-10.0, 0.0),), collabs=(), polygons=None, ego_fov=120.0):
    out, metrics = build_coverage_raster(
        ego_pose=(0.0, 0.0, 0.0),
        route_xy=route,
        past_route_xy=past,
        ego_observer=(1, 0.0, 0.0, 0.0),
        collaborator_observers=collabs,
        actor_polygons=polygons or {},
        ego_fov=ego_fov,
        ego_sight_range=35.0,
        collaborator_fov=120.0,
        collaborator_sight_range=35.0,
        config=CFG,
        spec=SPEC,
    )
    return out, metrics


class CoverageRasterTest(unittest.TestCase):
    def test_route_corridor_covers_past_and_future(self):
        cov, _ = raster()
        route_mask = cov[0]
        self.assertGreater(route_mask[cell_for(20.0, 0.0)], 0.5)
        self.assertGreater(route_mask[cell_for(-6.0, 0.0)], 0.5)
        self.assertLess(route_mask[cell_for(-18.0, 0.0)], 0.5)

    def test_distance_quality_decreases_with_distance(self):
        cov, _ = raster(ego_fov=180.0)
        quality = cov[4]
        self.assertGreater(float(quality[cell_for(5.0, 0.0)]), float(quality[cell_for(25.0, 0.0)]))

    def test_occluded_cell_has_zero_quality(self):
        obstacle = {
            10: [(4.0, -2.0), (6.0, -2.0), (6.0, 2.0), (4.0, 2.0)],
        }
        cov, _ = raster(polygons=obstacle)
        self.assertEqual(float(cov[4][cell_for(12.0, 0.0)]), 0.0)

    def test_collaborator_quality_improves_coverage(self):
        ego_only, ego_metrics = raster(ego_fov=60.0)
        collab, collab_metrics = raster(ego_fov=60.0, collabs=((2, 22.0, 0.0, 0.0),))
        self.assertGreater(float(collab[3][cell_for(27.0, 0.0)]), 0.0)
        self.assertGreater(float(collab[4][cell_for(27.0, 0.0)]), float(ego_only[4][cell_for(27.0, 0.0)]))
        self.assertLess(collab_metrics["coverage_uncertainty"], ego_metrics["coverage_uncertainty"])

    def test_metrics_are_finite_for_empty_route(self):
        cov, _ = build_coverage_raster(
            ego_pose=(0.0, 0.0, 0.0),
            route_xy=(),
            past_route_xy=(),
            ego_observer=(1, 0.0, 0.0, 0.0),
            collaborator_observers=(),
            actor_polygons={},
            ego_fov=120.0,
            ego_sight_range=35.0,
            collaborator_fov=120.0,
            collaborator_sight_range=35.0,
            config=CFG,
            spec=SPEC,
        )
        metrics = coverage_metrics(cov)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))


if __name__ == "__main__":
    unittest.main()
