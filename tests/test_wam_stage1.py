import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from car_dreamer.toolkit.communication.process import CommConfig
from car_dreamer.toolkit.wam import (
    GraphBuildSpec,
    MODALITY_TO_ID,
    OBS_OBJ,
    OBSERVATION,
    ObjectState,
    ObservationNodeInput,
    VehicleNodeInput,
    WAMFlowMatchingConfig,
    WAMGraphModelConfig,
    WAMPerceptionConfig,
    WAMPerceptionModel,
    WAMPolicy,
    WAMStage1CommunicationPolicyDataRecorder,
    WAMStage1Config,
    WAMStage1DataRecorder,
    WAMStage1Dataset,
    WAMStage1PolicyDataRecorder,
    WAMStage1Trainer,
    WAMUnifiedWorldModel,
    build_stage1_policy_graph,
    build_wam_hetero_graph,
    collate_stage1_samples,
    enumerate_stage1_policies,
    evaluate_stage1_uncertainty_rows,
    init_encoder_from_stage1,
    make_stage1_policy_metadata,
    make_stage1_sample,
    perception_metrics,
    policy_key,
    trajectory_ade_fde,
    union_object_ids,
    valid_object_ids,
    visible_object_ids_by_vehicle,
)

ROUTE_WAYPOINTS = 2


def perc_config(**overrides):
    base = dict(route_waypoints=ROUTE_WAYPOINTS, hidden_dim=32, num_layers=2, num_heads=4,
                temporal_hidden_dim=32, head_hidden_dim=32, traj_samples=3)
    base.update(overrides)
    return WAMPerceptionConfig(**base)


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
    notable = {objs[0][0]} if objs else set()
    return build_wam_hetero_graph(ego=ego, collaborators=[], objects=objects, observations=obs,
                                  policy=policy, spec=GraphBuildSpec(route_waypoints=ROUTE_WAYPOINTS),
                                  notable_ids=notable)


def synthetic_sample(cfg, objs=((100, 8.0, 1.0), (101, 5.0, 2.0)), window_len=3):
    objs = list(objs)
    window = [graph_with_objects([(i, x + 0.1 * k, y) for (i, x, y) in objs]) for k in range(window_len)]
    ids = valid_object_ids(window[-1])
    q = len(ids)
    target = torch.randn(q, cfg.traj_samples, 2)
    valid = torch.ones(q, cfg.traj_samples)
    return make_stage1_sample(window, target, valid, ids)


def synthetic_dataset(cfg, n=6):
    return WAMStage1Dataset([synthetic_sample(cfg, objs=[(100, 8.0 + 0.1 * i, 1.0), (101, 5.0, 2.0)])
                             for i in range(n)])


def policy_scene_inputs():
    ego = VehicleNodeInput(actor_id=1, is_ego=True, agent_slot=0, x=0.0, y=0.0, z=0.0, vx=1.0, vy=0.0,
                           yaw=0.0, route_xy=((5.0, 0.0), (10.0, 0.0)))
    collab2 = VehicleNodeInput(actor_id=2, is_ego=False, agent_slot=1, x=10.0, y=0.0, z=0.0, vx=0.0, vy=0.0,
                               yaw=0.0)
    collab3 = VehicleNodeInput(actor_id=3, is_ego=False, agent_slot=2, x=-5.0, y=4.0, z=0.0, vx=0.0, vy=0.0,
                               yaw=90.0)
    objects = [
        ObjectState(actor_id=100, actor_type="vehicle.x", object_class="vehicle", x=8.0, y=1.0, z=0.0,
                    vx=0.5, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5, visible_to_ego=True),
        ObjectState(actor_id=101, actor_type="vehicle.x", object_class="vehicle", x=12.0, y=2.0, z=0.0,
                    vx=0.0, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
                    visible_to_ego=False, visible_to_collaborators=(2,)),
        ObjectState(actor_id=102, actor_type="vehicle.x", object_class="vehicle", x=-7.0, y=7.0, z=0.0,
                    vx=0.0, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
                    visible_to_ego=False, visible_to_collaborators=(3,)),
    ]
    return ego, [collab2, collab3], objects


class MetricsTest(unittest.TestCase):
    def test_perception_metrics(self):
        m = perception_metrics(torch.tensor([0.9, 0.2, 0.8, 0.1]), torch.tensor([1.0, 0.0, 1.0, 1.0]))
        self.assertAlmostEqual(m["precision"], 1.0, places=4)   # 2 predicted, both correct
        self.assertAlmostEqual(m["recall"], 2.0 / 3.0, places=4)  # 3 positives, 2 found

    def test_trajectory_ade_fde(self):
        mu = torch.zeros(2, 3, 2)
        target = torch.ones(2, 3, 2)
        valid = torch.ones(2, 3)
        out = trajectory_ade_fde(mu, target, valid_mask=valid)
        self.assertAlmostEqual(out["ade"], float(np.sqrt(2.0)), places=4)
        self.assertAlmostEqual(out["fde"], float(np.sqrt(2.0)), places=4)


class RecorderTest(unittest.TestCase):
    def test_records_window_samples_with_aligned_target(self):
        cfg = perc_config()
        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1DataRecorder(tmp, fixed_dt=0.1, horizon_s=0.3, samples=cfg.traj_samples,
                                        history_window=2)
            horizon_steps = rec.horizon_steps
            for step in range(0, horizon_steps + 4):
                # object 100 marches +0.5m/x; object 101 static
                rec.observe(step, {100: (8.0 + 0.5 * step, 1.0), 101: (5.0, 2.0)})
                if step <= 2:
                    rec.register(step, graph=graph_with_objects([(100, 8.0 + 0.5 * step, 1.0), (101, 5.0, 2.0)]),
                                 ego_pose=(0.0, 0.0, 0.0))
                rec.flush_ready(step)
            written = rec.flush_all()
            self.assertTrue(rec.written >= 3)
            sample = torch.load(written[0] if written else sorted(Path(tmp).glob("*.pt"))[0], weights_only=False)
            self.assertIn("window", sample)
            self.assertLessEqual(len(sample["window"]), 3)  # capped at history_window+1
            q = len(sample["object_node_ids"])
            self.assertEqual(tuple(sample["target_xy"].shape), (q, cfg.traj_samples, 2))
            self.assertEqual(tuple(sample["valid"].shape), (q, cfg.traj_samples))

    def test_missing_future_masked(self):
        cfg = perc_config()
        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1DataRecorder(tmp, fixed_dt=0.1, horizon_s=0.3, samples=cfg.traj_samples,
                                        history_window=1)
            # observe only step 0 positions, then register step 0 but never observe the futures
            rec.observe(0, {100: (8.0, 1.0)})
            rec.register(0, graph=graph_with_objects([(100, 8.0, 1.0)]), ego_pose=(0.0, 0.0, 0.0))
            rec.flush_all()
            sample = torch.load(sorted(Path(tmp).glob("*.pt"))[0], weights_only=False)
            self.assertEqual(float(sample["valid"].sum()), 0.0)  # no future positions available

    def test_sample_period_aligns_history_and_future_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1DataRecorder(
                tmp,
                fixed_dt=0.1,
                horizon_s=1.0,
                samples=5,
                history_window=10,
                sample_period_s=0.2,
            )
            self.assertEqual(rec.sample_period_steps, 2)
            self.assertEqual(rec.step_offsets, (2, 4, 6, 8, 10))
            registered_steps = []
            last_record_step = 24
            for step in range(0, last_record_step + rec.horizon_steps + 1):
                rec.observe(step, {100: (8.0 + 0.1 * step, 1.0)})
                if step <= last_record_step and rec.should_register_step(step):
                    registered_steps.append(step)
                    rec.register(
                        step,
                        graph=graph_with_objects([(100, 8.0 + 0.1 * step, 1.0)]),
                        ego_pose=(0.0, 0.0, 0.0),
                    )
                rec.flush_ready(step)
            rec.flush_all()

            self.assertTrue(registered_steps)
            self.assertTrue(all(step % 2 == 0 for step in registered_steps))
            sample = torch.load(sorted(Path(tmp).glob("*.pt"))[-1], weights_only=False)
            self.assertEqual(len(sample["window"]), 11)
            self.assertEqual(sample["metadata"]["sample_period_steps"], 2)
            window_steps = sample["metadata"]["window_steps"]
            self.assertEqual(len(window_steps), 11)
            self.assertTrue(all((b - a) == 2 for a, b in zip(window_steps, window_steps[1:])))
            self.assertEqual(tuple(sample["target_xy"].shape[1:]), (5, 2))

    def test_slot_rebuild_uses_only_received_matching_sense_time_messages(self):
        def msg(msg_id, *, sender_id, t_sense, t_recv, policy_id=1):
            return SimpleNamespace(
                msg_id=msg_id,
                policy_id=policy_id,
                sender_id=sender_id,
                receiver_id=1,
                modalities=("objlist",),
                payload_size=1,
                data={},
                t_sense=t_sense,
                t_recv=t_recv,
                total_latency=0.0,
            )

        def graph_builder(state, messages, prediction_step):
            del messages, prediction_step
            step = int(state["step"])
            return graph_with_objects([(100, float(step), 1.0)])

        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1DataRecorder(
                tmp,
                fixed_dt=0.1,
                horizon_s=0.2,
                samples=1,
                history_window=4,
                sample_period_s=0.2,
                graph_builder=graph_builder,
                receive_window_steps=20,
                allow_cross_policy_messages=False,
            )
            for step in (12, 14, 16, 18, 20):
                rec.observe(step, {100: (float(step), 1.0)})
                rec.register_slot(step, state={"step": step, "ego_pose": (0.0, 0.0, 0.0)})
            rec.observe_messages(
                20,
                (
                    msg(1, sender_id=2, t_sense=12, t_recv=18),  # included for slot 12
                    msg(2, sender_id=3, t_sense=14, t_recv=22),  # not received by prediction step 20
                    msg(3, sender_id=4, t_sense=12, t_recv=19),  # must not be reused for slot 16
                    msg(4, sender_id=5, t_sense=-2, t_recv=0),   # outside receive window
                ),
                active_policy_id=1,
            )
            rec.observe(22, {100: (22.0, 1.0)})
            rec.flush_ready(22)
            sample = torch.load(sorted(Path(tmp).glob("*.pt"))[-1], weights_only=False)
            self.assertEqual(sample["metadata"]["window_steps"], [12, 14, 16, 18, 20])
            self.assertEqual(sample["metadata"]["slot_message_counts"], [2, 0, 0, 0, 0])
            self.assertEqual(sample["metadata"]["slot_selected_vehicle_ids"], [[2, 4], [], [], [], []])

    def test_union_object_set_and_t_time_perception_labels(self):
        # Last slot (prediction step t) is ego-only with object 100; collaborator-only object 101 only
        # appears in the earlier (already-received) slots. The recorded target/label set must be the
        # union {100, 101}, and labels must reflect each object's status *at t* (101 -> invisible).
        def graph_builder(state, messages, prediction_step):
            del messages, prediction_step
            return graph_with_objects(list(state["objs"]))

        def last_live_states():
            return (
                ObjectState(actor_id=100, actor_type="vehicle.x", object_class="vehicle", x=2.0, y=1.0,
                            z=0.0, vx=0.5, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
                            visible_to_ego=True),
                ObjectState(actor_id=101, actor_type="vehicle.x", object_class="vehicle", x=5.0, y=2.0,
                            z=0.0, vx=0.0, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
                            visible_to_ego=False, visible_to_collaborators=(2,)),
            )

        objs_by_step = {
            0: [(100, 2.0, 1.0), (101, 5.0, 2.0)],   # cooperative (received) frame
            1: [(100, 2.0, 1.0), (101, 5.0, 2.0)],
            2: [(100, 2.0, 1.0)],                     # ego-only last frame
        }
        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1DataRecorder(
                tmp, fixed_dt=0.1, horizon_s=0.2, samples=1, history_window=2,
                sample_period_s=0.1, graph_builder=graph_builder, receive_window_steps=20,
            )
            for step in (0, 1, 2):
                rec.observe(step, {100: (2.0, 1.0), 101: (5.0, 2.0)})
                rec.register_slot(step, state={
                    "step": step,
                    "ego_pose": (0.0, 0.0, 0.0),
                    "objs": objs_by_step[step],
                    "live_states": last_live_states(),
                    "notable_ids": (101,),  # the invisible collaborator object is the notable one at t
                })
            rec.observe(4, {100: (2.0, 1.0), 101: (5.0, 2.0)})
            rec.flush_all()
            sample = torch.load(sorted(Path(tmp).glob("*.pt"))[-1], weights_only=False)

            self.assertEqual(sample["object_node_ids"], [100, 101])  # union, sorted
            self.assertIn("perception_labels", sample)
            labels = sample["perception_labels"]
            row = {oid: i for i, oid in enumerate(sample["object_node_ids"])}
            # object 101 (collaborator-only, absent from last frame) is still supervised at t:
            self.assertEqual(float(labels["invisible"][row[101]]), 1.0)
            self.assertEqual(float(labels["visible"][row[101]]), 0.0)
            self.assertEqual(float(labels["notable"][row[101]]), 1.0)
            # object 100 is ego-visible and not notable at t:
            self.assertEqual(float(labels["visible"][row[100]]), 1.0)
            self.assertEqual(float(labels["invisible"][row[100]]), 0.0)
            self.assertEqual(float(labels["notable"][row[100]]), 0.0)

    def test_slot_recorder_writes_coverage_history_when_builder_is_present(self):
        def graph_builder(state, messages, prediction_step):
            del messages, prediction_step
            step = int(state["step"])
            return graph_with_objects([(100, float(step), 1.0)])

        def coverage_builder(state, messages, prediction_step):
            del messages, prediction_step
            return np.full((6, 4, 4), float(state["step"]), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1DataRecorder(
                tmp,
                fixed_dt=0.1,
                horizon_s=0.2,
                samples=1,
                history_window=2,
                sample_period_s=0.1,
                graph_builder=graph_builder,
                coverage_builder=coverage_builder,
                receive_window_steps=20,
            )
            for step in (0, 1, 2):
                rec.observe(step, {100: (float(step), 1.0)})
                rec.register_slot(step, state={"step": step, "ego_pose": (0.0, 0.0, 0.0)})
            rec.observe(4, {100: (4.0, 1.0)})
            rec.flush_all()
            sample = torch.load(sorted(Path(tmp).glob("*.pt"))[-1], weights_only=False)
            self.assertIn("coverage_history", sample)
            self.assertEqual(tuple(sample["coverage_history"].shape), (3, 6, 4, 4))


class PolicyAugmentedStage1Test(unittest.TestCase):
    def test_policy_enumeration_covers_fixed_family(self):
        policies = enumerate_stage1_policies([3, 2], bandwidth_ratio=1.0, frequency_steps=5)
        types = [name for name, _ in policies]
        self.assertEqual(types.count("ego_only"), 1)
        self.assertEqual(types.count("single_candidate_objlist"), 2)
        self.assertEqual(types.count("single_candidate_bev"), 2)
        self.assertEqual(types.count("all_candidates_objlist"), 1)
        self.assertEqual(types.count("all_candidates_bev"), 1)
        all_obj = next(policy for name, policy in policies if name == "all_candidates_objlist")
        self.assertEqual(all_obj.selected_vehicle_ids, (2, 3))
        self.assertTrue(all(v == "objlist" for v in all_obj.modality_by_vehicle.values()))
        self.assertTrue(all(v == 1.0 for v in all_obj.bandwidth_by_vehicle.values()))

    def test_policy_graphs_change_with_selection_and_modality(self):
        ego, collaborators, objects = policy_scene_inputs()
        spec = GraphBuildSpec(route_waypoints=ROUTE_WAYPOINTS, max_object_nodes=8)
        ego_only = WAMPolicy((), {}, {}, frequency_steps=5, reason="t")
        all_objlist = WAMPolicy((2, 3), {2: "objlist", 3: "objlist"}, {2: 1.0, 3: 1.0}, 5, "t")
        all_bev = WAMPolicy((2, 3), {2: "bev", 3: "bev"}, {2: 1.0, 3: 1.0}, 5, "t")

        graph_ego = build_stage1_policy_graph(ego=ego, collaborators=collaborators, objects=objects,
                                              policy=ego_only, spec=spec, notable_ids={101})
        graph_obj = build_stage1_policy_graph(ego=ego, collaborators=collaborators, objects=objects,
                                              policy=all_objlist, spec=spec, notable_ids={101})
        graph_bev = build_stage1_policy_graph(ego=ego, collaborators=collaborators, objects=objects,
                                              policy=all_bev, spec=spec, notable_ids={101})

        self.assertGreater(int(graph_obj[OBSERVATION].node_mask.sum()), int(graph_ego[OBSERVATION].node_mask.sum()))
        self.assertGreater(int(graph_obj[OBS_OBJ].edge_index.shape[1]), int(graph_ego[OBS_OBJ].edge_index.shape[1]))
        self.assertEqual(int(graph_bev[OBS_OBJ].edge_index.shape[1]), int(graph_obj[OBS_OBJ].edge_index.shape[1]))
        self.assertTrue(hasattr(graph_bev[OBSERVATION], "bev_raster"))
        self.assertIn(MODALITY_TO_ID["bev"], graph_bev[OBSERVATION].modality_id.tolist())

    def test_policy_sample_metadata_and_old_sample_compatibility(self):
        cfg = perc_config()
        old = synthetic_sample(cfg)
        self.assertNotIn("metadata", old)

        ego, collaborators, objects = policy_scene_inputs()
        policy = WAMPolicy((2,), {2: "objlist"}, {2: 1.0}, 5, "t")
        graph = build_stage1_policy_graph(ego=ego, collaborators=collaborators, objects=objects,
                                          policy=policy, spec=GraphBuildSpec(route_waypoints=ROUTE_WAYPOINTS))
        ids = valid_object_ids(graph)
        metadata = make_stage1_policy_metadata(
            step=7,
            episode_id=0,
            policy_type="single_candidate_objlist",
            policy=policy,
            candidate_vehicle_ids=[2, 3],
            notable_object_ids=[101],
            visible_ids_by_vehicle=visible_object_ids_by_vehicle(1, [2, 3], objects),
            ego_pose=(0.0, 0.0, 0.0),
            fixed_dt=0.1,
        )
        sample = make_stage1_sample([graph], torch.zeros(len(ids), cfg.traj_samples, 2),
                                    torch.ones(len(ids), cfg.traj_samples), ids, metadata=metadata)
        self.assertIn("metadata", sample)
        self.assertEqual(sample["metadata"]["policy_type"], "single_candidate_objlist")
        self.assertEqual(sample["metadata"]["policy"]["selected_vehicle_ids"], [2])

        model = WAMPerceptionModel(cfg)
        trainer = WAMStage1Trainer(model, WAMStage1Config(lr=1e-2, batch_size=2, max_steps=1,
                                                          log_interval=0, ckpt_interval=0,
                                                          ckpt_dir=tempfile.mkdtemp()))
        losses = trainer.loss_on_batch(collate_stage1_samples([old, sample]))
        self.assertTrue(bool(torch.isfinite(losses["total"])))

    def test_policy_recorder_writes_manifest_and_samples(self):
        cfg = perc_config()
        ego, collaborators, objects = policy_scene_inputs()
        policy = WAMPolicy((2,), {2: "objlist"}, {2: 1.0}, 5, "t")
        graph = build_stage1_policy_graph(ego=ego, collaborators=collaborators, objects=objects,
                                          policy=policy, spec=GraphBuildSpec(route_waypoints=ROUTE_WAYPOINTS))
        metadata = make_stage1_policy_metadata(
            step=0,
            episode_id=0,
            policy_type="single_candidate_objlist",
            policy=policy,
            candidate_vehicle_ids=[2, 3],
            notable_object_ids=[101],
            visible_ids_by_vehicle=visible_object_ids_by_vehicle(1, [2, 3], objects),
            ego_pose=(0.0, 0.0, 0.0),
            fixed_dt=0.1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1PolicyDataRecorder(tmp, fixed_dt=0.1, horizon_s=0.2, samples=cfg.traj_samples,
                                              history_window=1, manifest={"task": "unit"})
            for step in range(4):
                rec.observe(step, {100: (8.0 + step, 1.0), 101: (12.0, 2.0), 102: (-7.0, 7.0)})
            rec.register(0, key=policy_key("single_candidate_objlist", policy), graph=graph,
                         ego_pose=(0.0, 0.0, 0.0), metadata=metadata)
            rec.flush_all()
            manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["sample_count"], 1)
            files = sorted(Path(tmp).glob("sample_*.pt"))
            self.assertEqual(len(files), 1)
            sample = torch.load(files[0], weights_only=False)
            self.assertEqual(sample["metadata"]["policy_type"], "single_candidate_objlist")

    def test_policy_recorder_union_object_set_and_t_time_labels(self):
        # Policy-augmented recording must also predict/supervise the window union and carry t-time GT
        # labels: collaborator-only object 101 (ego can't see it) must appear and be labelled invisible.
        cfg = perc_config()
        ego, collaborators, objects = policy_scene_inputs()
        policy = WAMPolicy((2, 3), {2: "objlist", 3: "objlist"}, {2: 1.0, 3: 1.0}, 5, "t")
        graph = build_stage1_policy_graph(ego=ego, collaborators=collaborators, objects=objects,
                                          policy=policy, spec=GraphBuildSpec(route_waypoints=ROUTE_WAYPOINTS,
                                                                             max_object_nodes=8),
                                          notable_ids={101})
        metadata = make_stage1_policy_metadata(
            step=0, episode_id=0, policy_type="all_candidates_objlist", policy=policy,
            candidate_vehicle_ids=[2, 3], notable_object_ids=[101],
            visible_ids_by_vehicle=visible_object_ids_by_vehicle(1, [2, 3], objects),
            ego_pose=(0.0, 0.0, 0.0), fixed_dt=0.1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1PolicyDataRecorder(tmp, fixed_dt=0.1, horizon_s=0.2, samples=cfg.traj_samples,
                                              history_window=1, manifest={"task": "unit"})
            for step in range(4):
                rec.observe(step, {100: (8.0 + step, 1.0), 101: (12.0, 2.0), 102: (-7.0, 7.0)})
            rec.register(0, key=policy_key("all_candidates_objlist", policy), graph=graph,
                         ego_pose=(0.0, 0.0, 0.0), metadata=metadata,
                         live_states=objects, notable_ids=(101,))
            rec.flush_all()
            sample = torch.load(sorted(Path(tmp).glob("sample_*.pt"))[0], weights_only=False)
            self.assertEqual(sample["object_node_ids"], union_object_ids(sample["window"]))
            self.assertIn(101, sample["object_node_ids"])
            self.assertIn("perception_labels", sample)
            row = {oid: i for i, oid in enumerate(sample["object_node_ids"])}
            self.assertEqual(float(sample["perception_labels"]["invisible"][row[101]]), 1.0)
            self.assertEqual(float(sample["perception_labels"]["visible"][row[101]]), 0.0)
            self.assertEqual(float(sample["perception_labels"]["notable"][row[101]]), 1.0)
            self.assertEqual(float(sample["perception_labels"]["visible"][row[100]]), 1.0)

    def test_policy_recorder_without_live_states_has_no_labels(self):
        # Back-compat: omitting live_states leaves perception_labels off -> trainer uses node labels.
        cfg = perc_config()
        ego, collaborators, objects = policy_scene_inputs()
        policy = WAMPolicy((2,), {2: "objlist"}, {2: 1.0}, 5, "t")
        graph = build_stage1_policy_graph(ego=ego, collaborators=collaborators, objects=objects,
                                          policy=policy, spec=GraphBuildSpec(route_waypoints=ROUTE_WAYPOINTS))
        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1PolicyDataRecorder(tmp, fixed_dt=0.1, horizon_s=0.2, samples=cfg.traj_samples,
                                              history_window=1, manifest={"task": "unit"})
            for step in range(4):
                rec.observe(step, {100: (8.0 + step, 1.0), 101: (12.0, 2.0)})
            rec.register(0, key=policy_key("single_candidate_objlist", policy), graph=graph,
                         ego_pose=(0.0, 0.0, 0.0), metadata={"policy_type": "single_candidate_objlist"})
            rec.flush_all()
            sample = torch.load(sorted(Path(tmp).glob("sample_*.pt"))[0], weights_only=False)
            self.assertNotIn("perception_labels", sample)

    def test_comm_recorder_union_object_set_and_t_time_labels(self):
        # Communication replay path: under latency the last frame is ego-only, so the collaborator-only
        # object 101 only appears in earlier slots. _emit must still emit the union and t-time labels.
        cfg = perc_config()
        ego, collaborators, _ = policy_scene_inputs()
        live = (
            ObjectState(actor_id=100, actor_type="vehicle.x", object_class="vehicle", x=2.0, y=1.0,
                        z=0.0, vx=0.5, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
                        visible_to_ego=True),
            ObjectState(actor_id=101, actor_type="vehicle.x", object_class="vehicle", x=5.0, y=2.0,
                        z=0.0, vx=0.0, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
                        visible_to_ego=False, visible_to_collaborators=(2,)),
        )
        base_state = {"ego": ego, "collaborators": collaborators, "live_states": live,
                      "notable_ids": (101,), "ego_pose": (0.0, 0.0, 0.0)}

        def graph_builder(state, messages, prediction_step):
            del messages, prediction_step
            return graph_with_objects(list(state["objs"]))

        def slot(objs):
            s = dict(base_state)
            s["objs"] = objs
            return s

        comm = CommConfig(dt=0.1, sensor_period_steps=1, proc_delay_s=0.0, policy_duration_steps=10)
        with tempfile.TemporaryDirectory() as tmp:
            rec = WAMStage1CommunicationPolicyDataRecorder(
                tmp, fixed_dt=0.1, comm_config=comm, link_rate_bps=lambda *a, **k: 1e9,
                graph_builder=graph_builder, horizon_s=0.2, samples=cfg.traj_samples,
                history_window=2, manifest={"task": "unit"},
            )
            rec._ensure_runtimes(base_state, start_step=0)
            key = list(rec._runtimes)[0]
            payload = {
                "key": key,
                "source_window": [slot([(100, 2.0, 1.0), (101, 5.0, 2.0)]),   # cooperative
                                  slot([(100, 2.0, 1.0), (101, 5.0, 2.0)]),
                                  slot([(100, 2.0, 1.0)])],                    # ego-only @ t
                "window_steps": [0, 1, 2],
                "ego_pose": (0.0, 0.0, 0.0),
                "metadata": {"step": 2, "episode_id": 0, "policy_type": "x"},
            }
            path = rec._emit(2, payload)
            sample = torch.load(path, weights_only=False)
            self.assertEqual(sample["object_node_ids"], [100, 101])  # union, not last-frame ego-only
            self.assertIn("perception_labels", sample)
            row = {oid: i for i, oid in enumerate(sample["object_node_ids"])}
            self.assertEqual(float(sample["perception_labels"]["invisible"][row[101]]), 1.0)
            self.assertEqual(float(sample["perception_labels"]["visible"][row[100]]), 1.0)
            self.assertEqual(float(sample["perception_labels"]["notable"][row[101]]), 1.0)

    def test_evaluate_emits_notable_only_metrics(self):
        cfg = perc_config()
        sample = synthetic_sample(cfg, objs=[(100, 8.0, 1.0), (101, 5.0, 2.0)])
        ids = sample["object_node_ids"]
        n = len(ids)
        sample["perception_labels"] = {
            "notable": torch.tensor([1.0] + [0.0] * (n - 1)),
            "visible": torch.ones(n),
            "invisible": torch.zeros(n),
        }
        sample["metadata"] = make_stage1_policy_metadata(
            step=3, episode_id=0, policy_type="ego_only", policy=WAMPolicy((), {}, {}, 5, "t"),
            candidate_vehicle_ids=[], notable_object_ids=[ids[0]], visible_ids_by_vehicle={1: [ids[0]]},
            ego_pose=(0.0, 0.0, 0.0), fixed_dt=0.1,
        )
        rows = evaluate_stage1_uncertainty_rows(WAMPerceptionModel(cfg), [sample], device="cpu")
        row = rows[0]
        for key in ("motion_uncertainty_notable", "total_uncertainty_notable", "ade_notable", "fde_notable"):
            self.assertIn(key, row)
            self.assertTrue(np.isfinite(row[key]))

    def test_evaluate_normalized_uncertainty_in_unit_interval(self):
        cfg = perc_config()
        sample = synthetic_sample(cfg, objs=[(100, 8.0, 1.0), (101, 5.0, 2.0)])
        sample["metadata"] = make_stage1_policy_metadata(
            step=3, episode_id=0, policy_type="ego_only", policy=WAMPolicy((), {}, {}, 5, "t"),
            candidate_vehicle_ids=[], notable_object_ids=[100, 101], visible_ids_by_vehicle={1: [100]},
            ego_pose=(0.0, 0.0, 0.0), fixed_dt=0.1,
        )
        rows = evaluate_stage1_uncertainty_rows(WAMPerceptionModel(cfg), [sample], device="cpu",
                                                sigma_scale=4.0, alpha=0.5)
        row = rows[0]
        for key in ("motion_uncertainty_norm_notable", "total_uncertainty_norm_notable"):
            self.assertIn(key, row)
            self.assertGreaterEqual(float(row[key]), 0.0)
            self.assertLessEqual(float(row[key]), 1.0)
        self.assertEqual(float(row["sigma_scale"]), 4.0)

    def test_uncertainty_rows_are_csv_ready_and_finite(self):
        cfg = perc_config()
        policy = WAMPolicy((), {}, {}, 5, "t")
        sample = synthetic_sample(cfg)
        sample["metadata"] = make_stage1_policy_metadata(
            step=3,
            episode_id=1,
            policy_type="ego_only",
            policy=policy,
            candidate_vehicle_ids=[],
            notable_object_ids=[100],
            visible_ids_by_vehicle={1: [100]},
            ego_pose=(0.0, 0.0, 0.0),
            fixed_dt=0.1,
        )
        model = WAMPerceptionModel(cfg)
        rows = evaluate_stage1_uncertainty_rows(model, [sample], device="cpu")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for key in ("step", "episode_id", "policy_type", "selected_vehicle_ids", "modality_by_vehicle",
                    "notable_object_ids", "uncertainty", "ade", "fde"):
            self.assertIn(key, row)
        self.assertTrue(np.isfinite(row["uncertainty"]))
        self.assertTrue(np.isfinite(row["ade"]))
        self.assertTrue(np.isfinite(row["fde"]))


class DatasetTrainerTest(unittest.TestCase):
    def _model_and_cfg(self, **overrides):
        model = WAMPerceptionModel(perc_config())
        base = dict(lr=1e-2, batch_size=4, max_steps=40, log_interval=0, ckpt_interval=0,
                    ckpt_dir=tempfile.mkdtemp())
        base.update(overrides)
        return model, WAMStage1Config(**base)

    def test_collate_keeps_window_list(self):
        cfg = perc_config()
        batch = collate_stage1_samples([synthetic_sample(cfg), synthetic_sample(cfg)])
        self.assertEqual(len(batch["samples"]), 2)
        self.assertIn("window", batch["samples"][0])

    def test_one_batch_loss_finite(self):
        model, cfg = self._model_and_cfg()
        trainer = WAMStage1Trainer(model, cfg)
        batch = collate_stage1_samples([synthetic_dataset(perc_config(), 4)[i] for i in range(4)])
        losses = trainer.loss_on_batch(batch)
        self.assertTrue(bool(torch.isfinite(losses["total"])))
        self.assertIn("perception", losses)
        self.assertIn("traj", losses)

    def test_trainer_uses_recorded_t_time_labels(self):
        # A sample whose recorded perception_labels disagree with the graph-node labels: the trainer
        # must supervise against the recorded (t-time) labels, aligned by node_id, not the node labels.
        cfg = perc_config()
        sample = synthetic_sample(cfg, objs=[(100, 8.0, 1.0), (101, 5.0, 2.0)])
        ids = sample["object_node_ids"]
        # force notable=1 for *both* objects via the recorded labels (graph marks only the first notable)
        sample["perception_labels"] = {
            "notable": torch.ones(len(ids)),
            "visible": torch.ones(len(ids)),
            "invisible": torch.zeros(len(ids)),
        }
        model, _ = self._model_and_cfg()
        trainer = WAMStage1Trainer(model, WAMStage1Config(ckpt_dir=tempfile.mkdtemp(), log_interval=0,
                                                          ckpt_interval=0))
        model.eval()
        with torch.no_grad():
            out = model([g for g in sample["window"]])
        gt = trainer._gt_labels(out, sample)
        aligned_ids = [int(v) for v in out["object_node_ids"].tolist()]
        for i, oid in enumerate(aligned_ids):
            self.assertEqual(float(gt["notable"][i]), 1.0)  # recorded label wins for every object
        losses = trainer.loss_on_batch(collate_stage1_samples([sample]))
        self.assertTrue(bool(torch.isfinite(losses["total"])))

    def test_trainer_falls_back_to_node_labels_without_recorded_labels(self):
        # Back-compat: an older sample with no perception_labels still trains via the graph-node labels.
        cfg = perc_config()
        sample = synthetic_sample(cfg)
        self.assertNotIn("perception_labels", sample)
        model, _ = self._model_and_cfg()
        trainer = WAMStage1Trainer(model, WAMStage1Config(ckpt_dir=tempfile.mkdtemp(), log_interval=0,
                                                          ckpt_interval=0))
        losses = trainer.loss_on_batch(collate_stage1_samples([sample]))
        self.assertTrue(bool(torch.isfinite(losses["total"])))

    def test_overfit_decreases_loss(self):
        torch.manual_seed(0)
        model, cfg = self._model_and_cfg(max_steps=60)
        trainer = WAMStage1Trainer(model, cfg)
        dataset = synthetic_dataset(perc_config(), 4)
        batch = collate_stage1_samples([dataset[i] for i in range(len(dataset))])
        first = float(trainer.loss_on_batch(batch)["total"].detach())
        trainer.train(dataset)
        model.eval()
        last = float(trainer.loss_on_batch(batch)["total"].detach())
        self.assertLess(last, first)

    def test_checkpoint_roundtrip(self):
        torch.manual_seed(0)
        model, cfg = self._model_and_cfg(max_steps=10)
        trainer = WAMStage1Trainer(model, cfg)
        dataset = synthetic_dataset(perc_config(), 4)
        trainer.train(dataset)
        path = trainer.save_checkpoint()

        batch = collate_stage1_samples([dataset[i] for i in range(len(dataset))])
        trainer.model.eval()
        loss_before = float(trainer.loss_on_batch(batch)["total"].detach())

        trainer2 = WAMStage1Trainer(WAMPerceptionModel(perc_config()), cfg)
        trainer2.load_checkpoint(path)
        self.assertEqual(trainer2.step, trainer.step)
        trainer2.model.eval()
        loss_after = float(trainer2.loss_on_batch(batch)["total"].detach())
        self.assertAlmostEqual(loss_before, loss_after, places=4)

    def test_evaluate_returns_metrics(self):
        model, cfg = self._model_and_cfg(max_steps=2)
        trainer = WAMStage1Trainer(model, cfg)
        metrics = trainer.evaluate(synthetic_dataset(perc_config(), 4))
        for key in ("val_notable_f1", "val_invisible_recall", "val_ade", "val_fde", "val_mean_uncertainty"):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))


class WarmStartTest(unittest.TestCase):
    def test_init_stage2_encoder_from_stage1(self):
        torch.manual_seed(0)
        model = WAMPerceptionModel(perc_config())
        trainer = WAMStage1Trainer(model, WAMStage1Config(lr=1e-2, batch_size=4, max_steps=5,
                                                          log_interval=0, ckpt_interval=0,
                                                          ckpt_dir=tempfile.mkdtemp()))
        trainer.train(synthetic_dataset(perc_config(), 4))
        path = trainer.save_checkpoint()

        graph_cfg = WAMGraphModelConfig(route_waypoints=ROUTE_WAYPOINTS, hidden_dim=32, num_layers=2, num_heads=4)
        flow_cfg = WAMFlowMatchingConfig(hidden_dim=32, num_layers=2, num_heads=4, time_embed_dim=16,
                                         max_members=3, horizon=4, num_register_tokens=2, bev_latent_dim=32)
        stage2 = WAMUnifiedWorldModel(graph_cfg, flow_cfg)
        missing, unexpected = init_encoder_from_stage1(stage2, path)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        # the Stage-2 graph encoder now equals the trained Stage-1 encoder
        s1 = model.graph_net.state_dict()
        s2 = stage2.context_encoder.graph_net.state_dict()
        self.assertEqual(set(s1), set(s2))
        self.assertTrue(all(torch.equal(s1[k], s2[k]) for k in s1))


if __name__ == "__main__":
    unittest.main()
