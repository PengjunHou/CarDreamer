import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from car_dreamer.toolkit.wam import (
    GraphBuildSpec,
    ObjectState,
    ObservationNodeInput,
    VehicleNodeInput,
    WAMFlowMatchingConfig,
    WAMGraphModelConfig,
    WAMPerceptionConfig,
    WAMPerceptionModel,
    WAMPolicy,
    WAMStage1Config,
    WAMStage1DataRecorder,
    WAMStage1Dataset,
    WAMStage1Trainer,
    WAMUnifiedWorldModel,
    build_wam_hetero_graph,
    collate_stage1_samples,
    init_encoder_from_stage1,
    make_stage1_sample,
    perception_metrics,
    trajectory_ade_fde,
    valid_object_ids,
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
