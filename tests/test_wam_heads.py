import unittest

import torch

from car_dreamer.toolkit.wam import (
    GraphBuildSpec,
    ObjectState,
    ObservationNodeInput,
    VehicleNodeInput,
    WAMPerceptionConfig,
    WAMPerceptionModel,
    WAMPolicy,
    align_object_history,
    build_wam_hetero_graph,
    gaussian_trajectory_nll,
    perception_loss,
    policy_uncertainty,
)

# The vehicle-node state dim depends on route_waypoints, so the graph spec and the model config
# must agree on it (in the env both come from the same config key).
ROUTE_WAYPOINTS = 2


def graph_with_objects(objs, *, ego_xy=(0.0, 0.0)):
    """Ego-only graph whose object nodes are exactly ``objs`` (all ego-visible)."""
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


def make_model(traj_samples=4):
    return WAMPerceptionModel(
        WAMPerceptionConfig(route_waypoints=ROUTE_WAYPOINTS, hidden_dim=32, num_layers=2, num_heads=4,
                            temporal_hidden_dim=32, head_hidden_dim=32, traj_samples=traj_samples)
    ).eval()


class AlignObjectHistoryTest(unittest.TestCase):
    def test_alignment_and_presence_mask(self):
        d = 3
        # window of 3 steps; node ids per step (object 99 only present at the last step)
        embeddings = [
            torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),   # ids 10, 20
            torch.tensor([[3.0, 3.0, 3.0]]),                     # id 10
            torch.tensor([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]),   # ids 10, 99
        ]
        node_ids = [torch.tensor([10, 20]), torch.tensor([10]), torch.tensor([10, 99])]
        query = torch.tensor([10, 99])
        seq, mask = align_object_history(embeddings, node_ids, query)
        self.assertEqual(seq.shape, (2, 3, d))
        # object 10 present at all 3 steps
        self.assertTrue(torch.equal(mask[0], torch.tensor([1.0, 1.0, 1.0])))
        # object 99 only at the last step
        self.assertTrue(torch.equal(mask[1], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.allclose(seq[0, 1], torch.tensor([3.0, 3.0, 3.0])))
        self.assertTrue(torch.allclose(seq[1, 0], torch.zeros(d)))  # absent -> zero
        self.assertTrue(torch.allclose(seq[1, 2], torch.tensor([5.0, 5.0, 5.0])))


class WAMPerceptionModelTest(unittest.TestCase):
    def test_forward_shapes_and_finiteness(self):
        model = make_model(traj_samples=4)
        window = [
            graph_with_objects([(100, 8.0, 1.0), (101, 5.0, 2.0), (102, 3.0, 0.5)]),
            graph_with_objects([(100, 8.5, 1.0), (101, 5.5, 2.0), (102, 3.5, 0.5)]),
            graph_with_objects([(100, 9.0, 1.0), (101, 6.0, 2.0), (102, 4.0, 0.5)]),
        ]
        with torch.no_grad():
            out = model(window)
        q = out["object_node_ids"].shape[0]
        self.assertEqual(q, 3)
        self.assertEqual(out["notable_logits"].shape, (3,))
        self.assertEqual(out["traj_mu"].shape, (3, 4, 2))
        self.assertEqual(out["traj_log_var"].shape, (3, 4, 2))
        self.assertTrue(bool(torch.isfinite(out["z_object"]).all()))
        self.assertTrue(bool(torch.isfinite(out["traj_mu"]).all()))
        self.assertEqual(set(out["labels"].keys()), {"notable", "visible", "invisible"})

    def test_single_graph_window(self):
        model = make_model()
        with torch.no_grad():
            out = model([graph_with_objects([(100, 8.0, 1.0)])])
        self.assertEqual(out["object_node_ids"].shape[0], 1)
        self.assertTrue(bool(torch.isfinite(out["traj_mu"]).all()))

    def test_perception_loss_on_model_output(self):
        model = make_model()
        window = [graph_with_objects([(100, 8.0, 1.0), (101, 5.0, 2.0)])]
        with torch.no_grad():
            out = model(window)
        losses = perception_loss(out["perception_logits"], out["labels"])
        self.assertIn("total", losses)
        self.assertTrue(bool(torch.isfinite(losses["total"])))


class LossAndUncertaintyTest(unittest.TestCase):
    def test_trajectory_nll_decreases_when_mu_matches_target(self):
        target = torch.zeros(2, 3, 2)
        log_var = torch.zeros(2, 3, 2)
        notable = torch.ones(2)
        valid = torch.ones(2, 3)
        nll_match = gaussian_trajectory_nll(target.clone(), log_var, target, notable_weight=notable, valid_mask=valid)
        nll_far = gaussian_trajectory_nll(target + 5.0, log_var, target, notable_weight=notable, valid_mask=valid)
        self.assertTrue(bool(torch.isfinite(nll_match)))
        self.assertLess(float(nll_match), float(nll_far))

    def test_trajectory_nll_respects_valid_mask(self):
        mu = torch.zeros(1, 2, 2)
        target = torch.zeros(1, 2, 2)
        target[0, 1] = 100.0  # huge error, but masked out
        log_var = torch.zeros(1, 2, 2)
        notable = torch.ones(1)
        valid = torch.tensor([[1.0, 0.0]])
        nll = gaussian_trajectory_nll(mu, log_var, target, notable_weight=notable, valid_mask=valid)
        self.assertAlmostEqual(float(nll), 0.0, places=5)

    def test_policy_uncertainty_matches_manual_formula(self):
        notable_prob = torch.tensor([1.0, 0.0])
        log_var = torch.zeros(2, 2, 2)  # trace = exp(0)+exp(0) = 2 per step
        u = policy_uncertainty(notable_prob, log_var)
        # only the first object has weight 1; its trace is 2 everywhere -> U = 2.0
        self.assertAlmostEqual(float(u), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
