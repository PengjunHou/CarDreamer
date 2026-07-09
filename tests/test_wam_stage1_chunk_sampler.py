"""Offline tests for paper-faithful a=(S,B,D,n) sub-action sampling + action/comm recording.

Covers (no CARLA server, run with ``python -m unittest``):
* n (duration) sampled from ``random_policy.duration_grid`` for coop AND local-only sub-actions;
* |S| <= 1 under ``collaborator_counts=("0","1")``;
* policy lifecycle switches exactly at the sampled expiry (variable n);
* :class:`WAMStage1DataRecorder` metadata: ``active_subaction`` / ``slot_subactions`` across an
  action switch boundary, ``slot_comm_stats`` alignment, and backward compatibility;
* :meth:`CommunicationProcess.comm_stats` queue/link primitives (V2X (P2) inputs).
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from car_dreamer.v2v_comm_mixin import V2VCommMixin
from car_dreamer.toolkit.communication.process import (
    CommConfig,
    CommPolicy,
    CommunicationProcess,
    SenseSnapshot,
    make_local_policy,
)
from car_dreamer.toolkit.wam import (
    GraphBuildSpec,
    ObjectState,
    ObservationNodeInput,
    VehicleNodeInput,
    WAMPolicy,
    WAMStage1DataRecorder,
    build_wam_hetero_graph,
    subaction_from_policy,
)

DURATION_GRID = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
REQUEST_ID = 1
COLLAB_ID = 2
ROUTE_WAYPOINTS = 2


def make_sampler_mixin(*, counts=("1",), local_prob=0.0, duration_grid=DURATION_GRID, seed=0):
    mixin = object.__new__(V2VCommMixin)
    mixin._wam_policy_rng = np.random.default_rng(seed)
    mixin._wam_random_policy_local_prob = local_prob
    mixin._wam_random_policy_counts = counts
    mixin._wam_random_policy_modalities = (("objlist",), ("bev",))
    mixin._wam_random_policy_bandwidth_ratios = (0.2, 0.5, 0.8, 1.0)
    mixin._wam_random_policy_duration_grid = tuple(duration_grid)
    mixin._comm_config = SimpleNamespace(policy_duration_steps=5, sensor_period_steps=1)
    mixin._comm_policy_counter = 0
    mixin.ego = SimpleNamespace(id=100)
    return mixin


class DummyProc:
    def __init__(self):
        self.policy = None
        self.installed = []

    def set_policy(self, policy, step):
        del step
        self.policy = policy
        self.installed.append(policy)


def make_lifecycle_mixin(*, duration_grid, respect_request=False, seed=0):
    mixin = make_sampler_mixin(counts=("1",), local_prob=0.0, duration_grid=duration_grid, seed=seed)
    proc = DummyProc()
    mixin._ensure_comm_process = lambda: proc
    mixin._wam_enabled = True
    mixin._wam_policy_sampler_mode = "random_duration"
    mixin._wam_random_policy_respect_request = respect_request
    mixin._wam_coop_request = None
    mixin.coop_participant_ids = {2, 3}
    mixin.selected_collaborators = set()
    return mixin, proc


class DurationSamplingTest(unittest.TestCase):
    def test_duration_sampled_from_grid(self):
        mixin = make_sampler_mixin(counts=("1",))
        seen = set()
        for _ in range(200):
            policy = mixin._sample_random_comm_policy(step=0, candidates=[2, 3])
            self.assertIn(policy.duration_steps, DURATION_GRID)
            seen.add(policy.duration_steps)
        self.assertGreaterEqual(len(seen), 3)

    def test_local_policies_sample_duration_from_grid(self):
        # local via the local_prob mixture
        mixin = make_sampler_mixin(counts=("1",), local_prob=1.0)
        for _ in range(50):
            policy = mixin._sample_random_comm_policy(step=0, candidates=[2, 3])
            self.assertTrue(policy.is_local_only)
            self.assertIn(policy.duration_steps, DURATION_GRID)
        # local via the "0" cell of collaborator_counts
        mixin = make_sampler_mixin(counts=("0",))
        for _ in range(50):
            policy = mixin._sample_random_comm_policy(step=0, candidates=[2, 3])
            self.assertTrue(policy.is_local_only)
            self.assertIn(policy.duration_steps, DURATION_GRID)
        # local via the respect_request lifecycle branch (no pending request)
        mixin, proc = make_lifecycle_mixin(duration_grid=DURATION_GRID, respect_request=True)
        mixin._update_policy_lifecycle(step=0)
        self.assertTrue(proc.policy.is_local_only)
        self.assertIn(proc.policy.duration_steps, DURATION_GRID)

    def test_selection_at_most_one_collaborator(self):
        mixin = make_sampler_mixin(counts=("0", "1"))
        sizes = set()
        for _ in range(200):
            policy = mixin._sample_random_comm_policy(step=0, candidates=[1, 2, 3])
            k = len(policy.selected_collaborators)
            self.assertLessEqual(k, 1)
            sizes.add(k)
            self.assertEqual(set(policy.modalities_by_vehicle), set(policy.selected_collaborators))
            self.assertEqual(set(policy.bandwidth_by_vehicle), set(policy.selected_collaborators))
        self.assertEqual(sizes, {0, 1})

    def test_policy_switches_at_expiry(self):
        mixin, proc = make_lifecycle_mixin(duration_grid=(3,))
        for step in range(9):
            mixin._update_policy_lifecycle(step=step)
        self.assertEqual(len(proc.installed), 3)
        self.assertEqual([p.start_step for p in proc.installed], [0, 3, 6])
        # variable n: every install happens exactly at the previous policy's sampled expiry
        mixin, proc = make_lifecycle_mixin(duration_grid=(3, 5), seed=1)
        for step in range(40):
            mixin._update_policy_lifecycle(step=step)
        self.assertGreaterEqual(len(proc.installed), 2)
        for prev, nxt in zip(proc.installed, proc.installed[1:]):
            self.assertEqual(nxt.start_step, prev.start_step + prev.duration_steps)
            self.assertIn(prev.duration_steps, (3, 5))


def graph_with_objects(objs):
    ego = VehicleNodeInput(actor_id=1, is_ego=True, agent_slot=0, x=0.0, y=0.0, z=0.0, vx=1.0, vy=0.0,
                           yaw=0.0, route_xy=((5.0, 0.0), (10.0, 0.0)))
    objects = [
        ObjectState(actor_id=i, actor_type="vehicle.x", object_class="vehicle", x=x, y=y, z=0.0,
                    vx=0.5, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5, visible_to_ego=True)
        for (i, x, y) in objs
    ]
    obs = [ObservationNodeInput(vehicle_id=1, modality="objlist",
                                observed_object_ids=tuple(i for (i, _, _) in objs))]
    policy = WAMPolicy(selected_vehicle_ids=(), modality_by_vehicle={}, bandwidth_by_vehicle={},
                       frequency_steps=5, reason="t")
    return build_wam_hetero_graph(ego=ego, collaborators=[], objects=objects, observations=obs,
                                  policy=policy, spec=GraphBuildSpec(route_waypoints=ROUTE_WAYPOINTS),
                                  notable_ids={objs[0][0]} if objs else set())


def fake_policy(policy_id, *, start_step, n, selected=(), bandwidth=0.5, modality="objlist"):
    return SimpleNamespace(
        policy_id=policy_id,
        reason="random_duration" if selected else "local_only",
        selected_collaborators=tuple(selected),
        modalities_by_vehicle={int(v): (modality,) for v in selected},
        bandwidth_by_vehicle={int(v): float(bandwidth) for v in selected},
        duration_steps=n,
        start_step=start_step,
    )


class RecorderSubactionMetadataTest(unittest.TestCase):
    def test_metadata_subactions_across_switch_boundary(self):
        pol_a = fake_policy(1, start_step=0, n=3, selected=(2,))
        pol_b = fake_policy(2, start_step=3, n=10)
        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1DataRecorder(tmp, fixed_dt=0.1, horizon_s=0.1, samples=1, history_window=4)
            for step in range(5):
                active = pol_a if step < 3 else pol_b
                rec.observe(step, {100: (8.0 + 0.1 * step, 1.0)})
                rec.observe_messages(step, (), active_policy_id=active.policy_id, active_policy=active)
                rec.register(step, graph=graph_with_objects([(100, 8.0 + 0.1 * step, 1.0)]),
                             ego_pose=(0.0, 0.0, 0.0))
            rec.flush_all()
            sample = torch.load(sorted(Path(tmp).glob("*.pt"))[-1], weights_only=False)
            meta = sample["metadata"]
            self.assertEqual(meta["window_steps"], [0, 1, 2, 3, 4])
            subs = meta["slot_subactions"]
            self.assertEqual([s["policy_id"] for s in subs], [1, 1, 1, 2, 2])
            self.assertEqual([s["slot_offset"] for s in subs], [0, 1, 2, 0, 1])
            # exact a=(S,B,D,n) round-trip
            self.assertEqual(subs[0]["S"], [2])
            self.assertEqual(subs[0]["D"], {2: ["objlist"]})
            self.assertEqual(subs[0]["B"], {2: 0.5})
            self.assertEqual(subs[0]["n"], 3)
            self.assertEqual(subs[0]["start_step"], 0)
            self.assertEqual(subs[3]["S"], [])
            self.assertEqual(subs[3]["n"], 10)
            active = meta["active_subaction"]
            self.assertEqual(active["policy_id"], 2)
            self.assertEqual(active["slot_offset"], 1)
            self.assertEqual(active, subs[-1])

    def test_subaction_from_policy_serializes_comm_policy(self):
        policy = CommPolicy(
            policy_id=7, request_vehicle_id=REQUEST_ID, start_step=12, duration_steps=30,
            selected_collaborators=(COLLAB_ID,), modalities_by_vehicle={COLLAB_ID: ("bev",)},
            bandwidth_by_vehicle={COLLAB_ID: 0.8}, reason="random_duration",
        )
        sub = subaction_from_policy(policy, at_step=15)
        self.assertEqual(sub, {
            "policy_id": 7, "reason": "random_duration", "S": [COLLAB_ID],
            "D": {COLLAB_ID: ["bev"]}, "B": {COLLAB_ID: 0.8}, "n": 30,
            "start_step": 12, "slot_offset": 3,
        })

    def test_recorder_slot_comm_stats_aligned_and_backward_compatible(self):
        stats = {COLLAB_ID: {"arrival_bits": 8000.0, "rate_bps": 80000.0, "service_bits": 8000.0,
                             "backlog_bits": 8000.0, "queue_busy_s": 0.1}}
        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1DataRecorder(tmp, fixed_dt=0.1, horizon_s=0.1, samples=1, history_window=2)
            for step in range(3):
                rec.observe(step, {100: (8.0, 1.0)})
                if step == 1:
                    rec.observe_messages(step, (), active_policy_id=1, comm_stats=stats)
                else:
                    # legacy call without the new kwargs must keep working
                    rec.observe_messages(step, (), active_policy_id=1)
                rec.register(step, graph=graph_with_objects([(100, 8.0, 1.0)]), ego_pose=(0.0, 0.0, 0.0))
            rec.flush_all()
            sample = torch.load(sorted(Path(tmp).glob("*.pt"))[-1], weights_only=False)
            meta = sample["metadata"]
            self.assertEqual(meta["window_steps"], [0, 1, 2])
            self.assertEqual(meta["slot_comm_stats"], [{}, stats, {}])
            self.assertEqual(meta["slot_subactions"], [None, None, None])
            self.assertIsNone(meta["active_subaction"])
            # training-side consumers never read metadata; the sample still loads
            from car_dreamer.toolkit.wam import WAMStage1Dataset

            ds = WAMStage1Dataset([sample])
            self.assertIn("active_subaction", ds[0]["metadata"])


def _fixed_rate(rate_bps):
    return lambda sender_id, distance_m, bandwidth_ratio: float(rate_bps)


class CommStatsTest(unittest.TestCase):
    def _proc_with_message(self, rate_bps=80000.0, payload=1000):
        cfg = CommConfig(dt=0.1, sensor_period_steps=1, proc_delay_s=0.0, policy_duration_steps=50)
        proc = CommunicationProcess(cfg, REQUEST_ID)
        proc.set_policy(CommPolicy(
            policy_id=0, request_vehicle_id=REQUEST_ID, start_step=0, duration_steps=50,
            selected_collaborators=(COLLAB_ID,), modalities_by_vehicle={COLLAB_ID: ("objlist",)},
            bandwidth_by_vehicle={COLLAB_ID: 0.5}, reason="test",
        ), step=0)
        snap = SenseSnapshot(sender_id=COLLAB_ID, distance_m=10.0, payload_size=payload,
                             modalities=("objlist",), data={})
        emitted = proc.generate(0, {COLLAB_ID: snap}, _fixed_rate(rate_bps))
        self.assertEqual(len(emitted), 1)
        return proc, emitted[0]

    def test_comm_stats_primitives_plausible(self):
        proc, msg = self._proc_with_message()
        stats = proc.comm_stats(0, link_rate_bps=_fixed_rate(80000.0), distance_by_sender={COLLAB_ID: 10.0})
        s = stats[COLLAB_ID]
        self.assertEqual(s["arrival_bits"], 8000.0)     # L_m = 8 * payload
        self.assertEqual(s["backlog_bits"], 8000.0)     # still in flight
        self.assertEqual(s["rate_bps"], 80000.0)
        self.assertAlmostEqual(s["service_bits"], 8000.0)  # R_m * dt
        self.assertGreater(s["queue_busy_s"], 0.0)      # tx_delay = 8000/80000 = 0.1s
        # after delivery: backlog/busy drop to 0, arrival at the sense step is still recoverable
        proc.deliver(int(msg.t_recv))
        stats = proc.comm_stats(int(msg.t_recv), link_rate_bps=_fixed_rate(80000.0),
                                distance_by_sender={COLLAB_ID: 10.0})
        self.assertEqual(stats[COLLAB_ID]["backlog_bits"], 0.0)
        self.assertEqual(stats[COLLAB_ID]["arrival_bits"], 0.0)  # nothing sensed at t_recv
        stats_at_sense = proc.comm_stats(0, link_rate_bps=_fixed_rate(80000.0),
                                         distance_by_sender={COLLAB_ID: 10.0})
        self.assertEqual(stats_at_sense[COLLAB_ID]["arrival_bits"], 8000.0)

    def test_comm_stats_rate_fallback_and_local_only(self):
        proc, msg = self._proc_with_message(rate_bps=40000.0)
        self.assertEqual(float(msg.rate_bps), 40000.0)  # stored at generation
        stats = proc.comm_stats(0)  # no link_rate_bps/distance -> newest-message fallback
        self.assertEqual(stats[COLLAB_ID]["rate_bps"], 40000.0)
        self.assertAlmostEqual(stats[COLLAB_ID]["service_bits"], 4000.0)
        proc.set_policy(make_local_policy(policy_id=1, request_vehicle_id=REQUEST_ID,
                                          start_step=5, duration_steps=10), step=5)
        self.assertEqual(proc.comm_stats(5), {})


if __name__ == "__main__":
    unittest.main()
