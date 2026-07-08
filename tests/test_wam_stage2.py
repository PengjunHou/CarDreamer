import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from car_dreamer.toolkit.wam import (
    BEV_NUM_CHANNELS,
    BevSpec,
    GraphBuildSpec,
    ObjectState,
    ObservationNodeInput,
    VehicleNodeInput,
    WAMFlowDataRecorder,
    WAMFlowDataset,
    WAMFlowMatchingConfig,
    WAMGraphModelConfig,
    WAMPolicy,
    WAMStage2Config,
    WAMStage2Trainer,
    WAMUnifiedWorldModel,
    build_wam_hetero_graph,
    collate_flow_samples,
    decode_chunk_to_wampolicy,
    load_wam_uwm,
    make_flow_sample,
    rasterize_bev,
)

ROUTE_WAYPOINTS = 2
TEST_BEV = BevSpec(size=16, range_m=30.0)


def small_flow_config(**overrides):
    base = dict(
        hidden_dim=32, num_layers=2, num_heads=4, ff_dim=64, time_embed_dim=16,
        max_members=3, horizon=4, num_register_tokens=2, bev_latent_dim=32,
    )
    base.update(overrides)
    return WAMFlowMatchingConfig(**base)


def graph_cfg():
    # bev_size/channels must match the TEST_BEV rasters so the decoder output lines up with the targets.
    return WAMGraphModelConfig(route_waypoints=ROUTE_WAYPOINTS, hidden_dim=32, num_layers=2, num_heads=4,
                               bev_channels=BEV_NUM_CHANNELS, bev_size=TEST_BEV.size)


def make_objects(objs):
    return [
        ObjectState(actor_id=i, actor_type="vehicle.x", object_class="vehicle", x=x, y=y, z=0.0,
                    vx=0.5, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5, visible_to_ego=True)
        for (i, x, y) in objs
    ]


def graph_with_objects(objs):
    ego = VehicleNodeInput(actor_id=1, is_ego=True, agent_slot=0, x=0.0, y=0.0, z=0.0, vx=1.0, vy=0.0,
                           yaw=0.0, route_xy=((5.0, 0.0), (10.0, 0.0)))
    objects = make_objects(objs)
    obs = [ObservationNodeInput(vehicle_id=1, modality="objlist",
                                observed_object_ids=tuple(i for (i, _, _) in objs))]
    policy = WAMPolicy(selected_vehicle_ids=(), modality_by_vehicle={}, bandwidth_by_vehicle={},
                       frequency_steps=5, reason="t")
    notable = {objs[0][0]} if objs else set()
    return build_wam_hetero_graph(ego=ego, collaborators=[], objects=objects, observations=obs,
                                  policy=policy, spec=GraphBuildSpec(route_waypoints=ROUTE_WAYPOINTS),
                                  notable_ids=notable)


def synthetic_sample(cfg, objs=((100, 8.0, 1.0), (101, 5.0, 2.0)), history_window=0):
    objs = list(objs)
    graph = graph_with_objects(objs)
    policy_chunk = torch.randn(cfg.horizon, cfg.max_members, cfg.policy_width)
    member_mask = torch.zeros(cfg.max_members)
    member_mask[0] = 1.0
    policy_step_mask = torch.ones(cfg.horizon)
    # build real visibility-aware rasters for history/future (ego at origin)
    raster = rasterize_bev((0.0, 0.0, 0.0), make_objects(objs), route_xy=[(5, 0), (10, 0)], spec=TEST_BEV)
    bev_future = torch.from_numpy(np.stack([raster] * cfg.horizon)).to(torch.uint8)
    bev_history = torch.from_numpy(np.stack([raster] * (history_window + 1))).to(torch.uint8)
    return make_flow_sample(
        graph, policy_chunk, member_mask, policy_step_mask, notable_object_ids=[objs[0][0]],
        bev_history=bev_history, bev_future=bev_future, bev_spec=TEST_BEV, history_window=history_window,
    )


def synthetic_dataset(cfg, n=8):
    return WAMFlowDataset([synthetic_sample(cfg, objs=[(100, 8.0 + 0.1 * i, 1.0), (101, 5.0, 2.0)])
                           for i in range(n)])


class MakeFlowSampleTest(unittest.TestCase):
    def test_sample_schema_and_defaults(self):
        cfg = small_flow_config()
        sample = synthetic_sample(cfg)
        c, s = BEV_NUM_CHANNELS, TEST_BEV.size
        self.assertEqual(sample["policy_chunk"].shape, (cfg.horizon, cfg.max_members, cfg.policy_width))
        self.assertEqual(sample["bev_future"].shape, (cfg.horizon, c, s, s))
        self.assertEqual(sample["bev_history"].shape, (1, c, s, s))  # history_window=0 -> 1
        self.assertEqual(sample["bev_future"].dtype, torch.uint8)
        self.assertEqual(len(sample["vehicle_graphs"]), 1)
        self.assertEqual(sample["notable_object_ids"], [100])


class DatasetCollateTest(unittest.TestCase):
    def test_collate_shapes_and_padding(self):
        cfg = small_flow_config()
        c, s = BEV_NUM_CHANNELS, TEST_BEV.size
        samples = [synthetic_sample(cfg, objs=[(100, 8.0, 1.0), (101, 5.0, 2.0)]),
                   synthetic_sample(cfg, objs=[(100, 8.0, 1.0)])]
        batch = collate_flow_samples(samples, cfg)
        self.assertEqual(batch["policy_chunk"].shape, (2, cfg.horizon, cfg.max_members, cfg.policy_width))
        self.assertEqual(batch["bev_future"].shape, (2, cfg.horizon, c, s, s))
        self.assertEqual(batch["bev_history"].shape, (2, 1, c, s, s))
        self.assertEqual(batch["bev_step_mask"].shape, (2, cfg.horizon))
        self.assertEqual(len(batch["samples"]), 2)


class TrainerTest(unittest.TestCase):
    def _model_and_cfg(self, **overrides):
        model = WAMUnifiedWorldModel(graph_cfg(), small_flow_config())
        base = dict(lr=1e-2, batch_size=4, max_steps=40, log_interval=0, ckpt_interval=0,
                    ckpt_dir=tempfile.mkdtemp())
        base.update(overrides)
        return model, WAMStage2Config(**base), small_flow_config()

    def test_one_batch_loss_finite(self):
        model, cfg, flow_cfg = self._model_and_cfg()
        trainer = WAMStage2Trainer(model, cfg)
        batch = collate_flow_samples([synthetic_dataset(flow_cfg, 4)[i] for i in range(4)], flow_cfg)
        losses = trainer.loss_on_batch(batch)
        self.assertTrue(bool(torch.isfinite(losses["total"])))
        self.assertIn("policy", losses)
        self.assertIn("bev", losses)
        self.assertIn("bev_recon", losses)  # BEV reconstruction term present

    def test_overfit_decreases_loss(self):
        torch.manual_seed(0)
        model, cfg, flow_cfg = self._model_and_cfg(max_steps=60)
        trainer = WAMStage2Trainer(model, cfg)
        dataset = synthetic_dataset(flow_cfg, 4)
        batch = collate_flow_samples([dataset[i] for i in range(len(dataset))], flow_cfg)
        torch.manual_seed(1)
        first = float(trainer.loss_on_batch(batch)["total"].detach())
        trainer.train(dataset)
        torch.manual_seed(1)
        last = float(trainer.loss_on_batch(batch)["total"].detach())
        self.assertLess(last, first)

    def test_checkpoint_roundtrip(self):
        torch.manual_seed(0)
        model, cfg, flow_cfg = self._model_and_cfg(max_steps=10)
        trainer = WAMStage2Trainer(model, cfg)
        dataset = synthetic_dataset(flow_cfg, 4)
        trainer.train(dataset)
        path = trainer.save_checkpoint()

        batch = collate_flow_samples([dataset[i] for i in range(len(dataset))], flow_cfg)
        trainer.model.eval()
        torch.manual_seed(7)
        loss_before = float(trainer.loss_on_batch(batch)["total"].detach())

        trainer2 = WAMStage2Trainer(WAMUnifiedWorldModel(graph_cfg(), flow_cfg), cfg)
        trainer2.load_checkpoint(path)
        self.assertEqual(trainer2.step, trainer.step)
        trainer2.model.eval()
        torch.manual_seed(7)
        loss_after = float(trainer2.loss_on_batch(batch)["total"].detach())
        self.assertAlmostEqual(loss_before, loss_after, places=4)

    def test_freeze_encoder_toggles_grads(self):
        model, cfg, flow_cfg = self._model_and_cfg(freeze_encoder=True)
        trainer = WAMStage2Trainer(model, cfg)
        enc_params = list(model.context_encoder.parameters())
        self.assertTrue(all(not p.requires_grad for p in enc_params))
        batch = collate_flow_samples([synthetic_dataset(flow_cfg, 4)[i] for i in range(4)], flow_cfg)
        trainer.loss_on_batch(batch)["total"].backward()
        self.assertTrue(all(p.grad is None for p in enc_params))

        model2, cfg2, flow_cfg2 = self._model_and_cfg(freeze_encoder=False)
        trainer2 = WAMStage2Trainer(model2, cfg2)
        batch2 = collate_flow_samples([synthetic_dataset(flow_cfg2, 4)[i] for i in range(4)], flow_cfg2)
        trainer2.loss_on_batch(batch2)["total"].backward()
        self.assertTrue(any(p.grad is not None for p in model2.context_encoder.graph_net.parameters()))


class RecorderTest(unittest.TestCase):
    def _policy(self, selected=()):
        return WAMPolicy(selected_vehicle_ids=tuple(selected),
                         modality_by_vehicle={v: "objlist" for v in selected},
                         bandwidth_by_vehicle={v: 1.0 for v in selected},
                         frequency_steps=5, reason="t")

    def test_records_samples_with_policy_chunk_and_bev(self):
        cfg = small_flow_config()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = WAMFlowDataRecorder(tmp, samples=cfg.horizon, max_members=cfg.max_members,
                                           num_formats=cfg.num_formats, bev_spec=TEST_BEV)
            horizon_steps = recorder.horizon_steps  # H (future BEV needs t+1..t+H)
            for step in range(0, horizon_steps + 3):
                recorder.observe_policy(step, self._policy(selected=(7,) if step >= 1 else ()))
                # request vehicle moves forward; rasterize its visible objects each step
                recorder.observe_bev(step, (0.0, 0.0, 0.0), make_objects([(100, 8.0, 1.0)]),
                                     route_xy=[(5, 0), (10, 0)])
                if step <= 2:
                    graph = graph_with_objects([(100, 8.0 + 0.1 * step, 1.0)])
                    recorder.register(step, graph=graph, candidate_ids=[7], notable_object_ids=[100])
                recorder.flush_ready(step)
            written = recorder.flush_all()
            self.assertTrue(recorder.written >= 3)
            first_path = written[0] if written else sorted(Path(tmp).glob("*.pt"))[0]
            sample = torch.load(first_path, weights_only=False)
            c, s = BEV_NUM_CHANNELS, TEST_BEV.size
            self.assertEqual(sample["policy_chunk"].shape, (cfg.horizon, cfg.max_members, cfg.policy_width))
            self.assertEqual(sample["bev_future"].shape, (cfg.horizon, c, s, s))
            self.assertEqual(sample["bev_history"].shape, (1, c, s, s))
            self.assertEqual(sample["bev_future"].dtype, torch.uint8)
            # a recorded future raster is non-empty (the visible object was rasterized)
            self.assertGreater(int(sample["bev_future"].sum()), 0)
            self.assertEqual(float(sample["policy_chunk"][1, 0, 0]), 1.0)

    def test_dataset_loads_recorded_dir(self):
        cfg = small_flow_config()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = WAMFlowDataRecorder(tmp, samples=cfg.horizon, max_members=cfg.max_members,
                                           num_formats=cfg.num_formats, bev_spec=TEST_BEV)
            for step in range(0, recorder.horizon_steps + 5):
                recorder.observe_policy(step, self._policy())
                recorder.observe_bev(step, (0.0, 0.0, 0.0), make_objects([(100, 8.0, 1.0)]))
                if step <= 3:
                    graph = graph_with_objects([(100, 8.0 + 0.1 * step, 1.0)])
                    recorder.register(step, graph=graph, candidate_ids=[], notable_object_ids=[100])
                recorder.flush_ready(step)
            recorder.flush_all()
            dataset = WAMFlowDataset(tmp)
            self.assertTrue(len(dataset) >= 4)
            batch = collate_flow_samples([dataset[i] for i in range(len(dataset))], cfg)
            self.assertEqual(batch["bev_future"].shape[2], BEV_NUM_CHANNELS)


class EndToEndTest(unittest.TestCase):
    def test_train_runs_and_checkpoints(self):
        torch.manual_seed(0)
        flow_cfg = small_flow_config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = WAMStage2Config(lr=5e-3, batch_size=4, max_steps=20, log_interval=0,
                                  ckpt_interval=0, ckpt_dir=tmp)
            trainer = WAMStage2Trainer(WAMUnifiedWorldModel(graph_cfg(), flow_cfg), cfg)
            result = trainer.train(synthetic_dataset(flow_cfg, 8))
            self.assertTrue(np.isfinite(result["final_loss"]))
            self.assertEqual(int(result["steps"]), 20)
            self.assertTrue(sorted(Path(tmp).glob("*.pt")))


class Stage2InferenceUtilsTest(unittest.TestCase):
    """load_wam_uwm round-trip + decode_chunk_to_wampolicy slot mapping."""

    def test_load_wam_uwm_roundtrip(self):
        gcfg, fcfg = graph_cfg(), small_flow_config()
        model = WAMUnifiedWorldModel(gcfg, fcfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "uwm.pt"
            torch.save({"model": model.state_dict(), "graph_config": gcfg, "flow_config": fcfg}, path)
            loaded = load_wam_uwm(path, device="cpu")
        self.assertEqual(int(loaded.flow.config.horizon), int(fcfg.horizon))
        s1, s2 = model.state_dict(), loaded.state_dict()
        self.assertEqual(set(s1), set(s2))
        self.assertTrue(all(torch.equal(s1[k], s2[k]) for k in s1))
        self.assertFalse(loaded.training)  # returned in eval mode

    def test_decode_chunk_to_wampolicy(self):
        F = 2
        P = F + 3
        chunk = torch.zeros(4, 3, P)  # [H, M, P]
        # member0 (id 11): selected, fmt=bev(idx1), bw=0.7 ; member1: not selected ; member2 (id 13): objlist
        chunk[0, 0, 0] = 6.0; chunk[0, 0, 1 + 1] = 6.0; chunk[0, 0, P - 1] = 0.7
        chunk[0, 1, 0] = -6.0
        chunk[0, 2, 0] = 6.0; chunk[0, 2, 1 + 0] = 6.0; chunk[0, 2, P - 1] = 0.4
        pol = decode_chunk_to_wampolicy(chunk, [11, 12, 13], num_formats=F, step=0)
        self.assertEqual(set(pol.selected_vehicle_ids), {11, 13})
        self.assertEqual(pol.modality_by_vehicle[11], "bev")
        self.assertEqual(pol.modality_by_vehicle[13], "objlist")
        self.assertAlmostEqual(pol.bandwidth_by_vehicle[11], 0.7, places=4)
        # slots beyond candidate_ids are ignored
        self.assertTrue(set(pol.selected_vehicle_ids) <= {11, 12, 13})


if __name__ == "__main__":
    unittest.main()
