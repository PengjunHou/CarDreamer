import unittest

import torch

from car_dreamer.toolkit.wam import (
    GraphBuildSpec,
    ObjectState,
    ObservationNodeInput,
    SinusoidalTimeEmbedding,
    VehicleNodeInput,
    WAMBSContextEncoder,
    WAMFlowMatchingConfig,
    WAMFlowMatchingUWM,
    WAMGraphContextPool,
    WAMGraphModelConfig,
    WAMHeteroGraphNet,
    WAMPolicy,
    WAMUnifiedWorldModel,
    build_wam_hetero_graph,
    decode_policy_vector,
    encode_policy,
    encode_policy_chunk,
    flow_matching_loss,
    interpolate,
    pad_condition_tokens,
    sample_training_batch,
)
from car_dreamer.toolkit.wam.flow_matching import (
    _T_BEV_HIST,
    _T_CTX_GRAPH,
    _T_REQUEST,
    _T_TASK,
)

# graph vehicle-node dim depends on route_waypoints; the graph spec and graph model config must agree.
ROUTE_WAYPOINTS = 2


def small_flow_config(**overrides):
    base = dict(
        hidden_dim=32, num_layers=2, num_heads=4, ff_dim=64, time_embed_dim=16,
        max_members=3, horizon=4, num_register_tokens=2, bev_latent_dim=32,
    )
    base.update(overrides)
    return WAMFlowMatchingConfig(**base)


def graph_with_objects(objs, *, ego_xy=(0.0, 0.0)):
    """Ego-only graph whose object nodes are exactly ``objs`` (mirrors test_wam_heads)."""
    ego = VehicleNodeInput(
        actor_id=1, is_ego=True, agent_slot=0, x=ego_xy[0], y=ego_xy[1], z=0.0, vx=1.0, vy=0.0,
        yaw=0.0, route_xy=((5.0, 0.0), (10.0, 0.0)),
    )
    objects = [
        ObjectState(actor_id=i, actor_type="vehicle.x", object_class="vehicle", x=x, y=y, z=0.0,
                    vx=0.5, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5, visible_to_ego=True)
        for (i, x, y) in objs
    ]
    observations = [
        ObservationNodeInput(vehicle_id=1, modality="objlist",
                             observed_object_ids=tuple(i for (i, _, _) in objs),
                             payload_bytes=100.0, latency_s=0.0, freshness=1.0)
    ]
    policy = WAMPolicy(selected_vehicle_ids=(), modality_by_vehicle={}, bandwidth_by_vehicle={},
                       frequency_steps=5, reason="test")
    notable_ids = {objs[0][0]} if objs else set()
    return build_wam_hetero_graph(ego=ego, collaborators=[], objects=objects,
                                  observations=observations, policy=policy,
                                  spec=GraphBuildSpec(route_waypoints=ROUTE_WAYPOINTS),
                                  notable_ids=notable_ids)


def synthetic_cond(cfg, b=2, tc=3):
    cond = torch.randn(b, tc, cfg.hidden_dim)
    tids = torch.randint(0, 4, (b, tc))  # condition-token types (graph/request/task/bev)
    mask = torch.ones(b, tc)
    return cond, tids, mask


class TimeEmbeddingAndInterpolateTest(unittest.TestCase):
    def test_time_embedding_shape_and_finite(self):
        emb = SinusoidalTimeEmbedding(16, 32)
        out = emb(torch.tensor([0.0, 0.5, 1.0]))
        self.assertEqual(out.shape, (3, 32))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_interpolate_endpoints_and_velocity(self):
        x0 = torch.randn(2, 3, 4)
        x1 = torch.randn(2, 3, 4)
        self.assertTrue(torch.allclose(interpolate(x0, x1, torch.zeros(2)), x0))
        self.assertTrue(torch.allclose(interpolate(x0, x1, torch.ones(2)), x1))
        s = torch.tensor([0.3, 0.7])
        xs = interpolate(x0, x1, s)
        recovered = (x1 - xs) / (1.0 - s).view(2, 1, 1)
        self.assertTrue(torch.allclose(recovered, x1 - x0, atol=1e-5))


class GraphContextPoolTest(unittest.TestCase):
    def test_pool_shape_and_convex_combination(self):
        pool = WAMGraphContextPool(3)
        node_dict = {"a": torch.tensor([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]])}
        z = pool(node_dict)
        self.assertEqual(z.shape, (3,))
        lo = node_dict["a"].min(dim=0).values
        hi = node_dict["a"].max(dim=0).values
        self.assertTrue(bool((z >= lo - 1e-5).all()) and bool((z <= hi + 1e-5).all()))

    def test_pool_excludes_masked_nodes(self):
        pool = WAMGraphContextPool(3)
        node_dict = {"a": torch.tensor([[1.0, 1.0, 1.0], [100.0, 100.0, 100.0]])}
        mask_dict = {"a": torch.tensor([1.0, 0.0])}
        z = pool(node_dict, mask_dict)
        self.assertTrue(torch.allclose(z, node_dict["a"][0]))


class FlowMatchingForwardTest(unittest.TestCase):
    def _noised(self, cfg, b=2):
        return (
            torch.randn(b, cfg.horizon, cfg.max_members, cfg.policy_width),
            torch.randn(b, cfg.horizon, cfg.bev_latent_dim),
            torch.rand(b),
            torch.rand(b),
        )

    def test_forward_shapes_and_finite(self):
        cfg = small_flow_config()
        model = WAMFlowMatchingUWM(cfg).eval()
        cond, tids, mask = synthetic_cond(cfg)
        pol, bev, s_pi, s_z = self._noised(cfg)
        with torch.no_grad():
            out = model(cond, tids, mask, pol, bev, s_pi, s_z)
        self.assertEqual(out["u_pi"].shape, (2, cfg.horizon, cfg.max_members, cfg.policy_width))
        self.assertEqual(out["u_bev"].shape, (2, cfg.horizon, cfg.bev_latent_dim))
        for v in out.values():
            self.assertTrue(bool(torch.isfinite(v).all()))

    def test_key_padding_mask_does_not_nan(self):
        # everything padded except the always-valid register tokens -> still finite.
        cfg = small_flow_config()
        model = WAMFlowMatchingUWM(cfg).eval()
        cond, tids, _ = synthetic_cond(cfg)
        pol, bev, s_pi, s_z = self._noised(cfg)
        with torch.no_grad():
            out = model(
                cond, tids, torch.zeros(2, cond.shape[1]), pol, bev, s_pi, s_z,
                member_mask=torch.zeros(2, cfg.max_members),
                policy_step_mask=torch.zeros(2, cfg.horizon),
                bev_step_mask=torch.zeros(2, cfg.horizon),
            )
        for v in out.values():
            self.assertTrue(bool(torch.isfinite(v).all()))

    def test_enable_bev_false_drops_bev(self):
        cfg = small_flow_config(enable_bev=False)
        model = WAMFlowMatchingUWM(cfg).eval()
        cond, tids, mask = synthetic_cond(cfg)
        pol, _, s_pi, s_z = self._noised(cfg)
        with torch.no_grad():
            out = model(cond, tids, mask, pol, None, s_pi, s_z)
        self.assertIsNone(out["u_bev"])
        self.assertEqual(out["u_pi"].shape, (2, cfg.horizon, cfg.max_members, cfg.policy_width))


class FlowMatchingLossTest(unittest.TestCase):
    def test_zero_when_pred_equals_target(self):
        cfg = small_flow_config()
        b = 2
        target = {
            "u_pi": torch.randn(b, cfg.horizon, cfg.max_members, cfg.policy_width),
            "u_bev": torch.randn(b, cfg.horizon, cfg.bev_latent_dim),
        }
        pred = {k: v.clone() for k, v in target.items()}
        self.assertAlmostEqual(float(flow_matching_loss(pred, target)["total"]), 0.0, places=6)

    def test_masked_members_excluded(self):
        cfg = small_flow_config()
        b = 1
        target = {"u_pi": torch.zeros(b, cfg.horizon, cfg.max_members, cfg.policy_width), "u_bev": None}
        pred = {"u_pi": torch.full((b, cfg.horizon, cfg.max_members, cfg.policy_width), 99.0), "u_bev": None}
        losses = flow_matching_loss(
            pred, target, policy_mask=torch.zeros(b, cfg.horizon, cfg.max_members)
        )
        self.assertAlmostEqual(float(losses["policy"]), 0.0, places=6)
        self.assertAlmostEqual(float(losses["bev"]), 0.0, places=6)

    def test_overfit_decreases_loss(self):
        torch.manual_seed(0)
        cfg = small_flow_config()
        model = WAMFlowMatchingUWM(cfg).train()
        b = 4
        cond, tids, mask = synthetic_cond(cfg, b=b)
        pol1 = torch.randn(b, cfg.horizon, cfg.max_members, cfg.policy_width)
        bev1 = torch.randn(b, cfg.horizon, cfg.bev_latent_dim)
        batch = sample_training_batch(pol1, bev1, enable_bev=True)

        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        first = last = None
        for step in range(60):
            opt.zero_grad()
            pred = model(cond, tids, mask, batch["policy_s"], batch["bev_s"], batch["s_pi"], batch["s_z"])
            loss = flow_matching_loss(pred, batch)["total"]
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss.detach())
            last = float(loss.detach())
        self.assertLess(last, first)


class InferenceModesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = small_flow_config()
        self.model = WAMFlowMatchingUWM(self.cfg).eval()
        self.cond, self.tids, self.mask = synthetic_cond(self.cfg)

    def test_forward_rollout_shapes(self):
        policy = torch.randn(2, self.cfg.horizon, self.cfg.max_members, self.cfg.policy_width)
        bev = self.model.rollout_future_bev(self.cond, self.tids, self.mask, policy, n_steps=4)
        self.assertEqual(bev.shape, (2, self.cfg.horizon, self.cfg.bev_latent_dim))
        self.assertTrue(bool(torch.isfinite(bev).all()))

    def test_policy_proposal_shapes_and_decode(self):
        cand = self.model.propose_policies(self.cond, self.tids, self.mask, n_candidates=5, n_steps=4)
        self.assertEqual(cand.shape, (2, 5, self.cfg.horizon, self.cfg.max_members, self.cfg.policy_width))
        dec = decode_policy_vector(cand, num_formats=self.cfg.num_formats)
        self.assertTrue(bool((dec["sel"] >= 0).all()) and bool((dec["sel"] <= 1).all()))
        self.assertEqual(dec["fmt"].shape[-1], self.cfg.num_formats)

    def test_inverse_policy_search_shapes(self):
        bev = torch.randn(2, self.cfg.horizon, self.cfg.bev_latent_dim)
        pi = self.model.inverse_policy_search(self.cond, self.tids, self.mask, bev, n_steps=4)
        self.assertEqual(pi.shape, (2, self.cfg.horizon, self.cfg.max_members, self.cfg.policy_width))
        self.assertTrue(bool(torch.isfinite(pi).all()))

    def test_joint_generate_shapes(self):
        out = self.model.joint_generate(self.cond, self.tids, self.mask, n_steps=4)
        self.assertEqual(out["policy"].shape, (2, self.cfg.horizon, self.cfg.max_members, self.cfg.policy_width))
        self.assertEqual(out["bev"].shape, (2, self.cfg.horizon, self.cfg.bev_latent_dim))


class PolicyCodecTest(unittest.TestCase):
    def test_encode_layout(self):
        policy = WAMPolicy(selected_vehicle_ids=(20,), modality_by_vehicle={20: "bev"},
                           bandwidth_by_vehicle={20: 3.5}, frequency_steps=4, reason="t")
        vec = encode_policy(policy, candidate_ids=[20, 21], max_members=3, num_formats=2)
        self.assertEqual(vec.shape, (3, 5))  # P = sel(1)+fmt(2)+freq(1)+bw(1)
        self.assertEqual(float(vec[0, 0]), 1.0)
        self.assertEqual(float(vec[0, 1 + 1]), 1.0)
        self.assertEqual(float(vec[0, 3]), 4.0)
        self.assertEqual(float(vec[0, 4]), 3.5)
        self.assertTrue(bool((vec[2] == 0).all()))

    def test_encode_chunk_shape(self):
        p = WAMPolicy(selected_vehicle_ids=(20,), modality_by_vehicle={20: "objlist"},
                      bandwidth_by_vehicle={20: 1.0}, frequency_steps=5, reason="t")
        chunk = encode_policy_chunk([p, p, p], candidate_ids=[20, 21], max_members=3, num_formats=2)
        self.assertEqual(chunk.shape, (3, 3, 5))  # [H, M, P]
        self.assertEqual(float(chunk[0, 0, 0]), 1.0)


class BSContextEncoderTest(unittest.TestCase):
    def _encoder(self, d=32):
        graph_cfg = WAMGraphModelConfig(route_waypoints=ROUTE_WAYPOINTS, hidden_dim=d, num_layers=2, num_heads=4)
        return WAMBSContextEncoder(graph_cfg, bev_latent_dim=d), d

    def test_condition_token_assembly(self):
        enc, d = self._encoder()
        g = graph_with_objects([(100, 8.0, 1.0), (101, 5.0, 2.0)])
        cond, tids = enc([g, g], request_index=0, notable_object_ids=[100], bev_history=torch.zeros(1, d))
        # 2 graph tokens + 1 request marker + 1 notable task token (100) + 1 bev-history = 5
        self.assertEqual(cond.shape, (5, d))
        ids = tids.tolist()
        self.assertEqual(ids.count(_T_CTX_GRAPH), 2)
        self.assertEqual(ids.count(_T_REQUEST), 1)
        self.assertEqual(ids.count(_T_TASK), 1)
        self.assertEqual(ids.count(_T_BEV_HIST), 1)
        self.assertTrue(bool(torch.isfinite(cond).all()))

    def test_pad_condition_tokens(self):
        a = torch.randn(2, 8)
        b = torch.randn(4, 8)
        ta = torch.zeros(2, dtype=torch.long)
        tb = torch.ones(4, dtype=torch.long)
        cond, tids, mask = pad_condition_tokens([a, b], [ta, tb], hidden_dim=8)
        self.assertEqual(cond.shape, (2, 4, 8))
        self.assertEqual(mask[0].tolist(), [1, 1, 0, 0])
        self.assertEqual(mask[1].tolist(), [1, 1, 1, 1])


class UnifiedWorldModelTest(unittest.TestCase):
    def _model(self):
        graph_cfg = WAMGraphModelConfig(route_waypoints=ROUTE_WAYPOINTS, hidden_dim=32, num_layers=2, num_heads=4)
        flow_cfg = small_flow_config(hidden_dim=32, bev_latent_dim=32)
        return WAMUnifiedWorldModel(graph_cfg, flow_cfg), flow_cfg

    def _sample(self, model, d, objs):
        g = graph_with_objects(objs)
        return {"vehicle_graphs": [g], "request_index": 0,
                "notable_object_ids": [objs[0][0]], "bev_history": torch.zeros(1, d)}

    def test_hidden_dim_mismatch_raises(self):
        graph_cfg = WAMGraphModelConfig(route_waypoints=ROUTE_WAYPOINTS, hidden_dim=64)
        flow_cfg = small_flow_config(hidden_dim=32)
        with self.assertRaises(ValueError):
            WAMUnifiedWorldModel(graph_cfg, flow_cfg)

    def test_condition_batch_and_training_step(self):
        model, cfg = self._model()
        samples = [self._sample(model, cfg.hidden_dim, [(100, 8.0, 1.0), (101, 5.0, 2.0)]) for _ in range(2)]
        cond, tids, mask = model.condition_tokens_batch(samples)
        self.assertEqual(cond.shape[0], 2)
        self.assertEqual(cond.shape[-1], cfg.hidden_dim)

        b = 2
        policy_1 = torch.randn(b, cfg.horizon, cfg.max_members, cfg.policy_width)
        bev_1 = torch.zeros(b, cfg.horizon, cfg.bev_latent_dim)
        member_mask = torch.ones(b, cfg.max_members)
        step_mask = torch.ones(b, cfg.horizon)
        losses = model.training_step(
            cond, tids, mask, policy_1, bev_1,
            member_mask=member_mask, policy_step_mask=step_mask, bev_step_mask=step_mask,
            policy_loss_mask=step_mask.unsqueeze(-1) * member_mask.unsqueeze(1),
        )
        self.assertTrue(bool(torch.isfinite(losses["total"])))
        losses["total"].backward()
        grads = [p.grad for p in model.context_encoder.graph_net.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)

    def test_end_to_end_rollout(self):
        model, cfg = self._model()
        samples = [self._sample(model, cfg.hidden_dim, [(100, 8.0, 1.0)])]
        cond, tids, mask = model.condition_tokens_batch(samples)
        policy = torch.randn(1, cfg.horizon, cfg.max_members, cfg.policy_width)
        bev = model.flow.rollout_future_bev(cond, tids, mask, policy, n_steps=3)
        self.assertEqual(bev.shape, (1, cfg.horizon, cfg.bev_latent_dim))
        self.assertTrue(bool(torch.isfinite(bev).all()))


if __name__ == "__main__":
    unittest.main()
