import unittest

from car_dreamer.toolkit.wam import (
    ObjectState,
    build_coop_request,
    build_placeholder_policy,
    predict_notable_motion,
    select_notable_objects,
)


def make_object(actor_id, x, y, *, visible=False, collaborator_ids=()):
    return ObjectState(
        actor_id=actor_id,
        actor_type="vehicle.test",
        object_class="vehicle",
        x=float(x),
        y=float(y),
        z=0.0,
        vx=1.0,
        vy=0.0,
        yaw=0.0,
        length=4.0,
        width=2.0,
        height=1.5,
        visible_to_ego=visible,
        visible_to_collaborators=tuple(collaborator_ids),
    )


class WAMRuntimeTest(unittest.TestCase):
    def test_select_notable_objects_keeps_route_near_top_k(self):
        route = [(0.0, 0.0), (20.0, 0.0)]
        objects = [
            make_object(1, 2.0, 1.0),
            make_object(2, 4.0, 3.0),
            make_object(3, 6.0, 12.0),
            make_object(4, 8.0, 0.5),
        ]

        records = select_notable_objects(
            objects,
            route,
            notable_distance_m=10.0,
            max_notable_objects=2,
        )

        self.assertEqual([record.object_state.actor_id for record in records], [4, 1])
        self.assertTrue(all(record.notable for record in records))

    def test_visibility_drives_motion_uncertainty(self):
        records = select_notable_objects(
            [
                make_object(1, 2.0, 0.0, visible=True),
                make_object(2, 3.0, 0.0, visible=False, collaborator_ids=(10,)),
            ],
            [(0.0, 0.0), (10.0, 0.0)],
            notable_distance_m=10.0,
            max_notable_objects=3,
        )

        predictions = predict_notable_motion(
            records,
            dt=0.1,
            horizon_steps=3,
            visible_uncertainty=0.2,
            invisible_uncertainty=2.0,
        )

        self.assertAlmostEqual(predictions[1].uncertainty_score, 0.2)
        self.assertAlmostEqual(predictions[2].uncertainty_score, 2.0)
        self.assertEqual(len(predictions[1].future_xy), 3)

    def test_request_triggers_when_uncertainty_exceeds_threshold(self):
        records = select_notable_objects(
            [make_object(7, 1.0, 0.0, visible=False, collaborator_ids=(11,))],
            [(0.0, 0.0), (10.0, 0.0)],
            notable_distance_m=10.0,
            max_notable_objects=3,
        )
        predictions = predict_notable_motion(
            records,
            dt=0.1,
            horizon_steps=2,
            visible_uncertainty=0.2,
            invisible_uncertainty=2.0,
        )

        request = build_coop_request(
            ego_id=100,
            step=5,
            predictions=predictions,
            uncertainty_threshold=1.0,
        )

        self.assertIsNotNone(request)
        self.assertEqual(request.high_uncertainty_object_ids, (7,))

    def test_placeholder_policy_selects_all_candidates_on_request(self):
        records = select_notable_objects(
            [make_object(7, 1.0, 0.0, visible=False, collaborator_ids=(11,))],
            [(0.0, 0.0), (10.0, 0.0)],
            notable_distance_m=10.0,
            max_notable_objects=3,
        )
        predictions = predict_notable_motion(
            records,
            dt=0.1,
            horizon_steps=2,
            visible_uncertainty=0.2,
            invisible_uncertainty=2.0,
        )
        request = build_coop_request(
            ego_id=100,
            step=5,
            predictions=predictions,
            uncertainty_threshold=1.0,
        )

        policy = build_placeholder_policy(
            request=request,
            candidate_vehicle_ids={21, 20},
            uplink_bps=6000000.0,
            frequency_steps=1,
            default_modality="objlist",
        )

        self.assertEqual(policy.selected_vehicle_ids, (20, 21))
        self.assertEqual(policy.modality_by_vehicle, {20: "objlist", 21: "objlist"})
        self.assertAlmostEqual(policy.bandwidth_by_vehicle[20], 3000000.0)

    def test_placeholder_policy_is_empty_without_request(self):
        policy = build_placeholder_policy(
            request=None,
            candidate_vehicle_ids={20, 21},
            uplink_bps=6000000.0,
            frequency_steps=1,
            default_modality="objlist",
        )

        self.assertEqual(policy.selected_vehicle_ids, ())
        self.assertEqual(policy.modality_by_vehicle, {})


if __name__ == "__main__":
    unittest.main()
