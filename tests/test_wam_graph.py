import unittest

import torch

from car_dreamer.toolkit.wam import (
    OBJECT,
    OBJECT_STATE_DIM,
    OBS_OBJ,
    OBS_SCALAR_DIM,
    OBSERVATION,
    VEH_OBS,
    VEH_VEH,
    VEHICLE,
    GraphBuildSpec,
    ObjectState,
    ObservationNodeInput,
    VehicleNodeInput,
    WAMGraphModelConfig,
    WAMHeteroGraphNet,
    WAMPolicy,
    build_wam_hetero_graph,
    detection_confidence,
    fuse_injected_objects,
    hetero_graph_stats,
    vehicle_state_dim,
)


def make_object(actor_id, x, y, *, object_class="vehicle"):
    return ObjectState(
        actor_id=actor_id,
        actor_type=f"{object_class}.test",
        object_class=object_class,
        x=float(x),
        y=float(y),
        z=0.0,
        vx=1.0,
        vy=0.0,
        yaw=0.0,
        length=4.0,
        width=2.0,
        height=1.5,
    )


def make_scene(*, selected):
    """ego sees objects 100,102; collaborator 2 sees object 101 (invisible to ego)."""
    ego = VehicleNodeInput(
        actor_id=1, is_ego=True, agent_slot=0, x=0.0, y=0.0, z=0.0, vx=2.0, vy=0.0, yaw=0.0,
        route_xy=((5.0, 0.0), (10.0, 0.0), (15.0, 0.0)),
    )
    collab = VehicleNodeInput(
        actor_id=2, is_ego=False, agent_slot=1, x=10.0, y=5.0, z=0.0, vx=1.0, vy=0.0, yaw=90.0,
    )
    objects = [make_object(100, 8.0, 1.0), make_object(101, 12.0, 6.0), make_object(102, 3.0, 0.5, object_class="pedestrian")]
    observations = [
        ObservationNodeInput(
            vehicle_id=1, modality="objlist", observed_object_ids=(100, 102), latency_s=0.0, freshness=1.0,
            payload_bytes=200.0, det_confidence_by_object={100: 0.92, 102: 0.55},
        ),
        ObservationNodeInput(vehicle_id=2, modality="objlist", observed_object_ids=(101,), latency_s=0.03, freshness=0.8, payload_bytes=80.0),
    ]
    policy = WAMPolicy(
        selected_vehicle_ids=tuple(selected),
        modality_by_vehicle={vid: "objlist" for vid in selected},
        bandwidth_by_vehicle={vid: 1.0 for vid in selected},
        frequency_steps=5,
        reason="test",
    )
    return ego, collab, objects, observations, policy


class WAMGraphConstructionTest(unittest.TestCase):
    def test_policy_conditioned_node_and_edge_counts(self):
        ego, collab, objects, observations, policy = make_scene(selected=(2,))
        data = build_wam_hetero_graph(
            ego=ego, collaborators=[collab], objects=objects, observations=observations,
            policy=policy, spec=GraphBuildSpec(route_waypoints=6, max_object_nodes=32), notable_ids={101},
            latency_by_vehicle={2: 0.042},
        )

        self.assertEqual(data.metadata()[0], [VEHICLE, OBJECT, OBSERVATION])
        self.assertEqual(set(data.metadata()[1]), {VEH_OBS, OBS_OBJ, VEH_VEH})

        stats = hetero_graph_stats(data)
        self.assertEqual(stats["wam_graph_num_vehicle_nodes"], 2)  # ego + 1 selected
        self.assertEqual(stats["wam_graph_num_object_nodes"], 3)  # union observed (100,101,102)
        self.assertEqual(stats["wam_graph_num_observation_nodes"], 2)
        self.assertEqual(stats["wam_graph_num_veh_obs_edges"], 2)
        self.assertEqual(stats["wam_graph_num_obs_obj_edges"], 3)  # ego 2 + collaborator 1
        self.assertEqual(stats["wam_graph_num_veh_veh_edges"], 1)

    def test_edge_attributes(self):
        # obs_obj carries det_confidence; veh_veh carries policy latency; veh_obs stays structural.
        ego, collab, objects, observations, policy = make_scene(selected=(2,))
        data = build_wam_hetero_graph(
            ego=ego, collaborators=[collab], objects=objects, observations=observations,
            policy=policy, spec=GraphBuildSpec(route_waypoints=6, max_object_nodes=32),
            latency_by_vehicle={2: 0.042},
        )
        # veh_obs: structural only (no edge_attr).
        self.assertFalse(hasattr(data[VEH_OBS], "edge_attr"))
        # obs_obj: one det_confidence per (obs, object) edge in [0, 1]; ego's 100/102 use the scene values.
        oo_attr = data[OBS_OBJ].edge_attr
        self.assertEqual(tuple(oo_attr.shape), (3, 1))
        oo_src, oo_dst = data[OBS_OBJ].edge_index
        node_id = data[OBJECT].node_id.tolist()
        attr_by_obj = {node_id[int(d)]: float(oo_attr[i, 0]) for i, d in enumerate(oo_dst.tolist())}
        self.assertAlmostEqual(attr_by_obj[100], 0.92, places=5)
        self.assertAlmostEqual(attr_by_obj[102], 0.55, places=5)
        self.assertAlmostEqual(attr_by_obj[101], 1.0, places=5)  # collaborator default 1.0
        # veh_veh: one latency (seconds) per coop edge.
        vv_attr = data[VEH_VEH].edge_attr
        self.assertEqual(tuple(vv_attr.shape), (1, 1))
        self.assertAlmostEqual(float(vv_attr[0, 0]), 0.042, places=5)

    def test_veh_veh_edge_attr_empty_when_no_coop(self):
        ego, collab, objects, observations, policy = make_scene(selected=())
        data = build_wam_hetero_graph(
            ego=ego, collaborators=[collab], objects=objects, observations=observations, policy=policy,
        )
        self.assertEqual(tuple(data[VEH_VEH].edge_attr.shape), (0, 1))

    def test_state_vector_dims(self):
        ego, collab, objects, observations, policy = make_scene(selected=(2,))
        data = build_wam_hetero_graph(
            ego=ego, collaborators=[collab], objects=objects, observations=observations,
            policy=policy, spec=GraphBuildSpec(route_waypoints=6),
        )
        self.assertEqual(data[VEHICLE].x.shape[1], vehicle_state_dim(6))
        self.assertEqual(data[OBJECT].x.shape[1], OBJECT_STATE_DIM)
        self.assertEqual(data[OBSERVATION].x.shape[1], OBS_SCALAR_DIM)

    def test_invisible_flag_marks_collaborator_only_object(self):
        ego, collab, objects, observations, policy = make_scene(selected=(2,))
        data = build_wam_hetero_graph(
            ego=ego, collaborators=[collab], objects=objects, observations=observations, policy=policy,
        )
        node_id = data[OBJECT].node_id.tolist()
        invisible = data[OBJECT].invisible.tolist()
        idx_101 = node_id.index(101)
        # 101 is seen only by the collaborator -> invisible to ego.
        self.assertEqual(invisible[idx_101], 1.0)
        self.assertEqual(sum(invisible), 1.0)
        # ego-visible objects are not flagged invisible.
        self.assertEqual(data[OBJECT].visible.sum().item(), 2.0)

    def test_unselected_member_is_excluded(self):
        # Same scene, but the policy selects nobody -> ego-only graph.
        ego, collab, objects, observations, policy = make_scene(selected=())
        data = build_wam_hetero_graph(
            ego=ego, collaborators=[collab], objects=objects, observations=observations, policy=policy,
        )
        stats = hetero_graph_stats(data)
        self.assertEqual(stats["wam_graph_num_vehicle_nodes"], 1)
        self.assertEqual(stats["wam_graph_num_veh_veh_edges"], 0)
        self.assertEqual(stats["wam_graph_num_observation_nodes"], 1)  # ego's own observation
        self.assertEqual(stats["wam_graph_num_object_nodes"], 2)  # only ego-visible 100,102


class WAMGraphModelTest(unittest.TestCase):
    def _net(self):
        return WAMHeteroGraphNet(
            WAMGraphModelConfig(route_waypoints=6, hidden_dim=64, num_layers=2, num_heads=4)
        ).eval()

    def test_forward_shapes_and_finiteness(self):
        ego, collab, objects, observations, policy = make_scene(selected=(2,))
        data = build_wam_hetero_graph(
            ego=ego, collaborators=[collab], objects=objects, observations=observations, policy=policy,
        )
        net = self._net()
        with torch.no_grad():
            H = net(data)
        self.assertEqual(set(H.keys()), {VEHICLE, OBJECT, OBSERVATION})
        self.assertEqual(H[VEHICLE].shape, (2, 64))
        self.assertEqual(H[OBJECT].shape, (3, 64))
        self.assertEqual(H[OBSERVATION].shape, (2, 64))
        for tensor in H.values():
            self.assertTrue(bool(torch.isfinite(tensor).all()))

    def test_forward_on_ego_only_graph(self):
        ego, collab, objects, observations, policy = make_scene(selected=())
        data = build_wam_hetero_graph(
            ego=ego, collaborators=[collab], objects=objects, observations=observations, policy=policy,
        )
        net = self._net()
        with torch.no_grad():
            H = net(data)
        self.assertEqual(H[VEHICLE].shape, (1, 64))
        self.assertEqual(H[OBSERVATION].shape, (1, 64))
        self.assertTrue(all(bool(torch.isfinite(t).all()) for t in H.values()))

    def test_bev_modality_node_builds_and_runs(self):
        ego, collab, objects, observations, _ = make_scene(selected=(2,))
        # collaborator shares BEV instead of objlist (placeholder z^bev).
        bev_obs = ObservationNodeInput(vehicle_id=2, modality="bev", observed_object_ids=(), payload_bytes=131072.0, latency_s=0.05, freshness=0.6)
        observations = [observations[0], bev_obs]
        policy = WAMPolicy(selected_vehicle_ids=(2,), modality_by_vehicle={2: "bev"}, bandwidth_by_vehicle={2: 1.0}, frequency_steps=5, reason="bev")
        data = build_wam_hetero_graph(
            ego=ego, collaborators=[collab], objects=objects, observations=observations, policy=policy,
        )
        stats = hetero_graph_stats(data)
        # bev observation has no obs_obj edges in v1; ego still contributes its 2.
        self.assertEqual(stats["wam_graph_num_obs_obj_edges"], 2)
        net = WAMHeteroGraphNet(WAMGraphModelConfig(route_waypoints=6, hidden_dim=64, num_layers=2, num_heads=4, bev_channels=8, bev_size=32)).eval()
        with torch.no_grad():
            H = net(data)
        self.assertTrue(all(bool(torch.isfinite(t).all()) for t in H.values()))


class InjectionFusionTest(unittest.TestCase):
    def _obj(self, oid, x, y, *, vte=True, vtc=()):
        s = make_object(oid, x, y)
        return ObjectState(
            actor_id=oid, actor_type=s.actor_type, object_class=s.object_class, x=x, y=y, z=0.0,
            vx=1.0, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
            visible_to_ego=vte, visible_to_collaborators=vtc,
        )

    def test_fuse_prefers_ego_and_highest_confidence(self):
        ego_vis = [self._obj(100, 5, 0)]
        dets = [
            (self._obj(101, 20, 0, vte=False, vtc=(2,)), 0.4),
            (self._obj(101, 20, 0, vte=False, vtc=(3,)), 0.85),  # higher conf wins
            (self._obj(100, 5, 0), 0.2),                          # ego already has it
        ]
        fused = fuse_injected_objects(ego_vis, dets)
        self.assertEqual({int(s.actor_id) for s in fused.object_states}, {100, 101})
        self.assertEqual(fused.ego_visible_ids, {100})
        self.assertAlmostEqual(fused.det_confidence_by_object[101], 0.85)
        self.assertEqual(fused.det_confidence_by_object[100], 1.0)

    def test_detection_confidence_decreases_with_distance(self):
        near = detection_confidence((0.0, 0.0), make_object(1, 1.0, 0.0))
        far = detection_confidence((0.0, 0.0), make_object(1, 50.0, 0.0))
        self.assertGreater(near, far)
        self.assertLessEqual(near, 1.0)

    def test_object_visibility_override_sets_labels(self):
        ego = VehicleNodeInput(actor_id=1, is_ego=True, agent_slot=0, x=0, y=0, z=0, vx=1, vy=0, yaw=0,
                               route_xy=((5.0, 0.0),))
        objs = [make_object(100, 8, 1), make_object(101, 12, 6)]
        # single ego observation referencing both; visibility comes from the override, not the obs.
        obs = ObservationNodeInput(vehicle_id=1, modality="objlist", observed_object_ids=(100, 101),
                                   det_confidence_by_object={100: 1.0, 101: 0.7})
        policy = WAMPolicy((), {}, {}, frequency_steps=5, reason="inject")
        data = build_wam_hetero_graph(ego=ego, collaborators=[], objects=objs, observations=[obs],
                                      policy=policy, object_visibility={100: True, 101: False})
        by_id = {int(i): (v, iv) for i, v, iv in zip(
            data[OBJECT].node_id.tolist(), data[OBJECT].visible.tolist(), data[OBJECT].invisible.tolist()) if int(i) >= 0}
        self.assertEqual(by_id[100], (1.0, 0.0))
        self.assertEqual(by_id[101], (0.0, 1.0))  # injected -> invisible even though ego obs references it
        self.assertEqual(int(data[VEH_VEH].edge_index.shape[1]), 0)


if __name__ == "__main__":
    unittest.main()
