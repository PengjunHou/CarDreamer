import unittest

from car_dreamer.toolkit.wam import (
    TrajectoryTargetBuffer,
    build_trajectory_targets,
)


class BuildTrajectoryTargetsTest(unittest.TestCase):
    def test_identity_frame_when_ego_at_origin_zero_yaw(self):
        target, mask = build_trajectory_targets(
            object_node_ids=[5],
            ego_pose=(0.0, 0.0, 0.0),
            future_positions=[{5: (1.0, 0.0)}, {5: (2.0, 0.0)}],
        )
        self.assertEqual(target.shape, (1, 2, 2))
        self.assertEqual(mask.shape, (1, 2))
        self.assertAlmostEqual(target[0, 0, 0], 1.0, places=5)
        self.assertAlmostEqual(target[0, 1, 0], 2.0, places=5)
        self.assertTrue((mask == 1.0).all())

    def test_ego_frame_rotation(self):
        # ego heading +90deg; a world point straight ahead (along +x) maps to ego-frame (0, -1).
        target, mask = build_trajectory_targets(
            object_node_ids=[5],
            ego_pose=(0.0, 0.0, 90.0),
            future_positions=[{5: (1.0, 0.0)}],
        )
        self.assertAlmostEqual(target[0, 0, 0], 0.0, places=5)
        self.assertAlmostEqual(target[0, 0, 1], -1.0, places=5)
        self.assertEqual(mask[0, 0], 1.0)

    def test_missing_actor_is_masked(self):
        target, mask = build_trajectory_targets(
            object_node_ids=[6],
            ego_pose=(0.0, 0.0, 0.0),
            future_positions=[{5: (1.0, 0.0)}, {}],
        )
        self.assertTrue((mask == 0.0).all())
        self.assertTrue((target == 0.0).all())


class TrajectoryTargetBufferTest(unittest.TestCase):
    def test_emits_after_horizon(self):
        buf = TrajectoryTargetBuffer(fixed_dt=1.0, horizon_s=2.0, samples=2)
        self.assertEqual(buf.step_offsets, (1, 2))
        self.assertEqual(buf.horizon_steps, 2)

        buf.observe(0, {5: (0.0, 0.0)})
        buf.register(0, [5], (0.0, 0.0, 0.0))
        self.assertEqual(buf.flush_ready(0), [])

        buf.observe(1, {5: (1.0, 0.0)})
        self.assertEqual(buf.flush_ready(1), [])

        buf.observe(2, {5: (2.0, 0.0)})
        ready = buf.flush_ready(2)
        self.assertEqual(len(ready), 1)
        step, target, mask = ready[0]
        self.assertEqual(step, 0)
        self.assertEqual(target.shape, (1, 2, 2))
        self.assertAlmostEqual(target[0, 0, 0], 1.0, places=5)  # position at step 0+1
        self.assertAlmostEqual(target[0, 1, 0], 2.0, places=5)  # position at step 0+2
        self.assertTrue((mask == 1.0).all())

    def test_flush_all_masks_unavailable_futures(self):
        buf = TrajectoryTargetBuffer(fixed_dt=1.0, horizon_s=2.0, samples=2)
        buf.observe(0, {5: (0.0, 0.0)})
        buf.observe(1, {5: (1.0, 0.0)})  # step 0+2 never observed
        buf.register(0, [5], (0.0, 0.0, 0.0))
        ready = buf.flush_all()
        self.assertEqual(len(ready), 1)
        _, target, mask = ready[0]
        self.assertEqual(mask[0, 0], 1.0)  # offset 1 available
        self.assertEqual(mask[0, 1], 0.0)  # offset 2 missing


if __name__ == "__main__":
    unittest.main()
