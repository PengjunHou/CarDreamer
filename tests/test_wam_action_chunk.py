"""Unit tests for the action-chunk data model + CommPolicy segmentation (paper Sec I.B)."""

import unittest

from car_dreamer.toolkit.wam import (
    ActionChunk,
    SubAction,
    action_chunk_to_comm_segments,
    enumerate_candidate_chunks,
    local_only_chunk,
    sub_action_to_comm_policy,
)


def coop(m, b, d):
    return SubAction(selected=(m,), bandwidth_by_vehicle={m: b}, modality_by_vehicle={m: "bev"}, duration_slots=d)


class SubActionTest(unittest.TestCase):
    def test_local_only_and_bandwidth(self):
        self.assertTrue(SubAction().is_local_only)
        self.assertEqual(SubAction().total_bandwidth(), 0.0)
        s = coop(7, 0.5, 10)
        self.assertFalse(s.is_local_only)
        self.assertAlmostEqual(s.total_bandwidth(), 0.5)


class ActionChunkTest(unittest.TestCase):
    def test_horizon_offsets_and_lookup(self):
        chunk = ActionChunk((coop(7, 0.5, 3), coop(7, 1.0, 2), SubAction(duration_slots=4)))
        self.assertEqual(chunk.num_subepochs, 3)
        self.assertEqual(chunk.horizon_slots, 9)
        self.assertEqual(chunk.subepoch_start_offsets(), (0, 3, 5))
        # slot -> (j, sub)
        self.assertEqual(chunk.sub_action_at(0)[0], 0)
        self.assertEqual(chunk.sub_action_at(2)[0], 0)
        self.assertEqual(chunk.sub_action_at(3)[0], 1)  # boundary -> next sub-action
        self.assertEqual(chunk.sub_action_at(5)[0], 2)
        self.assertEqual(chunk.sub_action_at(100)[0], 2)  # clamp to last
        self.assertEqual(chunk.per_subepoch_bandwidth(), (0.5, 1.0, 0.0))

    def test_j1_special_case(self):
        chunk = ActionChunk((coop(3, 0.8, 6),))
        self.assertEqual(chunk.num_subepochs, 1)
        self.assertEqual(chunk.horizon_slots, 6)
        self.assertEqual(chunk.subepoch_start_offsets(), (0,))

    def test_empty_chunk_lookup_raises(self):
        with self.assertRaises(ValueError):
            ActionChunk(()).sub_action_at(0)


class LocalOnlyChunkTest(unittest.TestCase):
    def test_local_only(self):
        chunk = local_only_chunk(n_slots=8, n_min=2)
        self.assertEqual(chunk.num_subepochs, 1)
        self.assertEqual(chunk.horizon_slots, 8)
        self.assertTrue(chunk.sub_actions[0].is_local_only)
        # n_min floor
        self.assertEqual(local_only_chunk(n_slots=1, n_min=5).horizon_slots, 5)


class EnumerateTest(unittest.TestCase):
    def test_j1_singles_plus_local(self):
        chunks = enumerate_candidate_chunks(
            [7], bandwidth_grid=[0.5, 1.0], duration_grid=[5, 10], j_max=1, modality="bev"
        )
        coop_chunks = [c for c in chunks if not c.sub_actions[0].is_local_only]
        local_chunks = [c for c in chunks if all(s.is_local_only for s in c.sub_actions)]
        self.assertGreaterEqual(len(local_chunks), 1)  # always includes local-only
        self.assertEqual(len(coop_chunks), 4)          # 2 bw x 2 dur
        for c in coop_chunks:
            self.assertEqual(len(c.sub_actions[0].selected), 1)          # |S| <= 1
            self.assertEqual(c.sub_actions[0].modality_by_vehicle[7], "bev")

    def test_fmax_prunes_long_chunks(self):
        chunks = enumerate_candidate_chunks(
            [7], bandwidth_grid=[1.0], duration_grid=[5, 10], j_max=2, modality="bev", f_max=12
        )
        self.assertTrue(all(c.horizon_slots <= 12 for c in chunks))
        # J=2 with two duration-10 subs (=20) must be pruned; a 5+5 (=10) survives
        self.assertTrue(any(c.num_subepochs == 2 for c in chunks))

    def test_bad_modality(self):
        with self.assertRaises(ValueError):
            enumerate_candidate_chunks([1], modality="lidar")


class SegmentationTest(unittest.TestCase):
    def test_sub_action_to_comm_policy_cooperative(self):
        p = sub_action_to_comm_policy(coop(7, 0.5, 10), policy_id=3, request_vehicle_id=1, start_step=100)
        self.assertEqual(p.policy_id, 3)
        self.assertEqual(p.start_step, 100)
        self.assertEqual(p.duration_steps, 10)
        self.assertEqual(p.selected_collaborators, (7,))
        self.assertEqual(p.modalities_by_vehicle[7], ("bev",))
        self.assertAlmostEqual(p.bandwidth_by_vehicle[7], 0.5)
        self.assertFalse(p.is_local_only)

    def test_sub_action_to_comm_policy_local(self):
        p = sub_action_to_comm_policy(SubAction(duration_slots=6), policy_id=4, request_vehicle_id=1, start_step=50)
        self.assertTrue(p.is_local_only)
        self.assertEqual(p.selected_collaborators, ())
        self.assertEqual(p.duration_steps, 6)

    def test_chunk_to_segments_tiles_the_horizon(self):
        chunk = ActionChunk((coop(7, 0.5, 3), SubAction(duration_slots=4), coop(9, 1.0, 2)))
        segs = action_chunk_to_comm_segments(chunk, first_policy_id=10, request_vehicle_id=1, start_step=200)
        self.assertEqual(len(segs), 3)
        self.assertEqual([s.policy_id for s in segs], [10, 11, 12])
        self.assertEqual([s.start_step for s in segs], [200, 203, 207])   # consecutive, no gap
        self.assertEqual([s.duration_steps for s in segs], [3, 4, 2])
        self.assertEqual(segs[-1].end_step, 200 + chunk.horizon_slots)     # tiles [200, 200+F)
        self.assertTrue(segs[1].is_local_only)


if __name__ == "__main__":
    unittest.main()
