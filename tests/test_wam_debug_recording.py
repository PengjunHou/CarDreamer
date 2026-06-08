import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from car_dreamer.toolkit.wam import (
    ActorSnapshot,
    MotionPredictionRecord,
    NotableObjectRecord,
    ObjectState,
    WAMNotableDebugRecorder,
    carla_actor_bbox_footprint,
    future_sample_offsets,
    future_sample_step_offsets,
    predicted_future_waypoints,
)
from car_dreamer.toolkit.wam.visualization import render_wam_bev_record


def make_object(actor_id=7, x=0.0, y=0.0, vx=2.0, vy=0.0):
    return ObjectState(
        actor_id=actor_id,
        actor_type="vehicle.test",
        object_class="vehicle",
        x=float(x),
        y=float(y),
        z=0.0,
        vx=float(vx),
        vy=float(vy),
        yaw=0.0,
        length=4.0,
        width=2.0,
        height=1.5,
        bbox=(),
        visible_to_ego=True,
    )


def make_notable(obj):
    return NotableObjectRecord(
        object_state=obj,
        notable=True,
        visible=True,
        invisible=False,
        occluding=False,
        route_distance=1.0,
    )


def make_snapshot(actor_id, x, y=0.0, *, object_class="vehicle"):
    return ActorSnapshot(
        actor_id=int(actor_id),
        actor_type=f"{object_class}.test",
        object_class=object_class,
        position=(float(x), float(y), 0.0),
        velocity=(2.0, 0.0, 0.0),
        yaw=0.0,
    )


class WAMDebugRecordingTest(unittest.TestCase):
    def test_carla_actor_bbox_footprint_returns_four_bev_corners(self):
        class Vec:
            def __init__(self, x, y=0.0, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        class Rotation:
            yaw = 0.0

        class Transform:
            location = Vec(10.0, 20.0, 0.0)
            rotation = Rotation()

        class BBox:
            location = Vec(1.0, 0.0, 0.0)
            extent = Vec(2.0, 1.0, 1.0)

        class Actor:
            bounding_box = BBox()

            def get_transform(self):
                return Transform()

        footprint = carla_actor_bbox_footprint(Actor())

        self.assertEqual(len(footprint), 4)
        self.assertEqual(
            footprint,
            (
                (9.0, 19.0),
                (13.0, 19.0),
                (13.0, 21.0),
                (9.0, 21.0),
            ),
        )

    def test_three_second_six_waypoint_sampling(self):
        self.assertEqual(future_sample_offsets(3.0, 6), (0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
        self.assertEqual(future_sample_step_offsets(0.1, 3.0, 6), (5, 10, 15, 20, 25, 30))

    def test_pending_record_flushes_when_future_history_is_ready(self):
        recorder = WAMNotableDebugRecorder(fixed_dt=0.1, horizon_s=3.0, future_samples=6)
        obj = make_object(actor_id=7, x=0.0, vx=2.0)
        pred = MotionPredictionRecord(
            actor_id=7,
            future_xy=(),
            covariance_diag=(),
            uncertainty_score=0.2,
        )
        ego = make_snapshot(100, -1.0)

        ready = []
        for step in range(31):
            ready.extend(
                recorder.observe(
                    step=step,
                    time_s=step * 0.1,
                    ego=ego,
                    actors=[make_snapshot(7, step * 0.2)],
                    notable_records=[make_notable(obj)] if step == 0 else [],
                    predictions={7: pred} if step == 0 else {},
                    wam={"coop_triggered": False},
                    include_record=step == 0,
                )
            )

        self.assertEqual(len(ready), 1)
        record = ready[0]
        self.assertEqual(record["step"], 0)
        gt = record["notable_objects"][0]["ground_truth"]["future_waypoints"]
        self.assertEqual([item["available"] for item in gt], [True] * 6)
        self.assertAlmostEqual(gt[0]["position"][0], 1.0)
        self.assertAlmostEqual(gt[-1]["position"][0], 6.0)

    def test_disappeared_actor_future_waypoints_are_unavailable(self):
        recorder = WAMNotableDebugRecorder(fixed_dt=0.1, horizon_s=0.5, future_samples=1)
        obj = make_object(actor_id=7, x=0.0, vx=2.0)

        recorder.observe(
            step=0,
            time_s=0.0,
            ego=make_snapshot(100, -1.0),
            actors=[make_snapshot(7, 0.0)],
            notable_records=[make_notable(obj)],
            predictions={},
            wam={},
            include_record=True,
        )
        ready = []
        for step in range(1, 6):
            ready.extend(
                recorder.observe(
                    step=step,
                    time_s=step * 0.1,
                    ego=make_snapshot(100, -1.0),
                    actors=[],
                    notable_records=[],
                    predictions={},
                    wam={},
                    include_record=False,
                )
            )

        self.assertEqual(len(ready), 1)
        gt = ready[0]["notable_objects"][0]["ground_truth"]["future_waypoints"]
        self.assertEqual(gt, [{"dt": 0.5, "position": None, "available": False}])

    def test_predicted_waypoints_follow_constant_velocity(self):
        obj = make_object(actor_id=7, x=1.0, y=-2.0, vx=2.0, vy=1.0)

        waypoints = predicted_future_waypoints(
            obj,
            future_offsets_s=(0.5, 1.0, 1.5),
            uncertainty=0.2,
        )

        self.assertEqual([item["position"] for item in waypoints], [[2.0, -1.5, 0.0], [3.0, -1.0, 0.0], [4.0, -0.5, 0.0]])
        self.assertEqual([item["uncertainty"] for item in waypoints], [0.2, 0.2, 0.2])

    def test_bev_renderer_handles_bbox_convex_hull_center(self):
        class FakeMapRenderer:
            _surface = np.zeros((120, 120, 3), dtype=np.uint8)
            _scale = 1.0
            _pixels_per_meter = 2.0
            _world_offset_in_meter = (-30.0, -30.0)

        record = {
            "step": 0,
            "ego": {
                "id": 100,
                "type": "vehicle",
                "actor_type": "vehicle.ego",
                "position": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "yaw": 0.0,
                "bbox": [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]],
            },
            "wam": {},
            "notable_objects": [],
        }

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "frame.png"
            render_wam_bev_record(
                map_renderer=FakeMapRenderer(),
                record=record,
                output_path=output_path,
                bev_range_m=20.0,
                image_size_px=128,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertEqual(cv2.imread(str(output_path)).shape, (128, 128, 3))


if __name__ == "__main__":
    unittest.main()
