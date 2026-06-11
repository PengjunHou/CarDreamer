import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from car_dreamer.toolkit.wam import (
    BEV_CHANNEL_NAMES,
    BEV_NUM_CHANNELS,
    BevSpec,
    GraphBuildSpec,
    ObjectState,
    ObservationNodeInput,
    VehicleNodeInput,
    WAMBevDecoder,
    WAMGraphModelConfig,
    WAMHeteroGraphNet,
    WAMPolicy,
    bev_iou,
    bev_reconstruction_loss,
    build_wam_hetero_graph,
    rasterize_bev,
)
from car_dreamer.toolkit.wam.bev import _CH_EGO, _CH_ROUTE
from car_dreamer.toolkit.wam.graph import CLASS_TO_ID
from car_dreamer.toolkit.wam.visualization import render_bev_raster_rgb, render_vehicle_centric_wam_bev

SPEC = BevSpec(size=32, range_m=20.0)


def obj(i, x, y, cls="vehicle", yaw=0.0):
    return ObjectState(actor_id=i, actor_type="t", object_class=cls, x=x, y=y, z=0.0,
                       vx=0.0, vy=0.0, yaw=yaw, length=4.0, width=2.0, height=1.5)


class RasterizerTest(unittest.TestCase):
    def test_channel_layout(self):
        self.assertEqual(BEV_NUM_CHANNELS, 7)
        self.assertEqual(BEV_CHANNEL_NAMES, ("vehicle", "pedestrian", "bicycle", "other", "ego", "route", "drivable"))

    def test_shape_and_dtype(self):
        r = rasterize_bev((0.0, 0.0, 0.0), [obj(10, 8.0, 0.0)], spec=SPEC)
        self.assertEqual(r.shape, (BEV_NUM_CHANNELS, SPEC.size, SPEC.size))
        self.assertEqual(r.dtype, np.uint8)

    def test_visibility_exclusion(self):
        # only the passed (visible) object is drawn; an absent object leaves the scene empty there.
        r_with = rasterize_bev((0.0, 0.0, 0.0), [obj(10, 8.0, 0.0)], spec=SPEC)
        r_without = rasterize_bev((0.0, 0.0, 0.0), [], spec=SPEC)
        self.assertGreater(int(r_with[CLASS_TO_ID["vehicle"]].sum()), 0)
        self.assertEqual(int(r_without[CLASS_TO_ID["vehicle"]].sum()), 0)

    def test_ego_centered(self):
        r = rasterize_bev((0.0, 0.0, 0.0), [], spec=SPEC)
        mid = SPEC.size // 2
        self.assertGreater(int(r[_CH_EGO].sum()), 0)
        self.assertTrue(bool(r[_CH_EGO][mid, mid] > 0))

    def test_class_routing(self):
        r = rasterize_bev((0.0, 0.0, 0.0), [obj(10, 6.0, 0.0, cls="pedestrian")], spec=SPEC)
        self.assertGreater(int(r[CLASS_TO_ID["pedestrian"]].sum()), 0)
        self.assertEqual(int(r[CLASS_TO_ID["vehicle"]].sum()), 0)

    def test_heading_up_rotation(self):
        # same world object (east of ego); heading-up means it appears "ahead" (above center) when the
        # ego faces +x, and "to the side" (≈center row) when the ego faces +y.
        mid = SPEC.size / 2.0
        r0 = rasterize_bev((0.0, 0.0, 0.0), [obj(10, 8.0, 0.0)], spec=SPEC)  # facing +x
        r90 = rasterize_bev((0.0, 0.0, 90.0), [obj(10, 8.0, 0.0)], spec=SPEC)  # facing +y
        rows0 = np.where(r0[CLASS_TO_ID["vehicle"]] > 0)[0]
        rows90, cols90 = np.where(r90[CLASS_TO_ID["vehicle"]] > 0)
        self.assertLess(rows0.max(), mid)            # ahead -> above center
        self.assertTrue(abs(float(rows90.mean()) - mid) < SPEC.size * 0.25)  # to the side -> ≈ center row
        self.assertGreater(cols90.mean(), mid)       # and offset in column

    def test_route_channel(self):
        r = rasterize_bev((0.0, 0.0, 0.0), [], route_xy=[(0, 0), (10, 0), (20, 0)], spec=SPEC)
        self.assertGreater(int(r[_CH_ROUTE].sum()), 0)

    def test_per_vehicle_visibility_filters_different_objects(self):
        ego_pose = (0.0, 0.0, 0.0)
        candidate_pose = (0.0, 0.0, 0.0)
        ego_only = obj(10, 8.0, 0.0)
        candidate_only = obj(20, 0.0, 8.0)
        ego_only = replace(ego_only, visible_to_ego=True, visible_to_collaborators=())
        candidate_only = replace(candidate_only, visible_to_ego=False, visible_to_collaborators=(2,))
        objects = [ego_only, candidate_only]

        ego_visible = [s for s in objects if s.visible_to_ego]
        candidate_visible = [s for s in objects if 2 in s.visible_to_collaborators]
        ego_raster = rasterize_bev(ego_pose, ego_visible, spec=SPEC)
        candidate_raster = rasterize_bev(candidate_pose, candidate_visible, spec=SPEC)

        vehicle_ch = CLASS_TO_ID["vehicle"]
        self.assertGreater(int(ego_raster[vehicle_ch].sum()), 0)
        self.assertGreater(int(candidate_raster[vehicle_ch].sum()), 0)
        self.assertFalse(bool(np.array_equal(ego_raster[vehicle_ch], candidate_raster[vehicle_ch])))


class BevVisualizationTest(unittest.TestCase):
    def test_render_bev_raster_rgb_shape_dtype_and_ego(self):
        raster = rasterize_bev((0.0, 0.0, 0.0), [], spec=SPEC)
        rgb = render_bev_raster_rgb(raster)

        self.assertEqual(rgb.shape, (SPEC.size, SPEC.size, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertGreater(int(rgb.sum()), 0)
        self.assertGreater(int(raster[_CH_EGO].sum()), 0)

    def test_render_vehicle_centric_wam_bev_writes_textless_png(self):
        class FakeMapRenderer:
            _scale = 1.0
            _pixels_per_meter = 2.0
            _world_offset_in_meter = (-32.0, -32.0)

            def __init__(self):
                self._surface = np.zeros((128, 128, 3), dtype=np.uint8)
                self._surface[:, :] = (45, 45, 45)

        vehicle = {"position": (0.0, 0.0, 0.0), "yaw": 0.0, "bbox": ((2, 1), (2, -1), (-2, -1), (-2, 1))}
        visible = [
            {"position": (8.0, 0.0, 0.0), "yaw": 0.0, "bbox": ((10, 1), (10, -1), (6, -1), (6, 1))}
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "vehicle_1" / "bev_000000.png"
            render_vehicle_centric_wam_bev(
                map_renderer=FakeMapRenderer(),
                vehicle=vehicle,
                visible_objects=visible,
                output_path=path,
                image_size_px=64,
                bev_range_m=32.0,
                ego_offset_m=0.0,
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


class DecoderTest(unittest.TestCase):
    def test_decoder_shape(self):
        dec = WAMBevDecoder(latent_dim=16, channels=BEV_NUM_CHANNELS, size=32)
        out = dec(torch.randn(3, 16))
        self.assertEqual(out.shape, (3, BEV_NUM_CHANNELS, 32, 32))

    def test_recon_loss_and_iou(self):
        target = torch.zeros(2, BEV_NUM_CHANNELS, 8, 8)
        target[:, 0, 2:5, 2:5] = 1.0
        big = (target * 2 - 1) * 30.0  # logits that match the target after sigmoid
        self.assertLess(float(bev_reconstruction_loss(big, target)), 1e-2)
        self.assertAlmostEqual(bev_iou(target, target), 1.0, places=4)

    def test_autoencoder_overfit(self):
        torch.manual_seed(0)
        net = WAMHeteroGraphNet(WAMGraphModelConfig(route_waypoints=2, hidden_dim=32, num_layers=2,
                                                    num_heads=4, bev_channels=BEV_NUM_CHANNELS, bev_size=16))
        dec = WAMBevDecoder(latent_dim=32, channels=BEV_NUM_CHANNELS, size=16)
        spec = BevSpec(size=16, range_m=20.0)
        raster = torch.from_numpy(rasterize_bev((0.0, 0.0, 0.0), [obj(10, 6.0, 0.0)],
                                                route_xy=[(0, 0), (8, 0)], spec=spec)).float().unsqueeze(0)
        opt = torch.optim.Adam(list(net.embedding.bev_encoder.parameters()) + list(dec.parameters()), lr=1e-2)
        first = last = None
        first_iou = last_iou = None
        for step in range(80):
            opt.zero_grad()
            logits = dec(net.encode_bev(raster))
            loss = bev_reconstruction_loss(logits, raster)
            loss.backward()
            opt.step()
            iou = bev_iou(torch.sigmoid(logits.detach()), raster)
            if step == 0:
                first, first_iou = float(loss.detach()), iou
            last, last_iou = float(loss.detach()), iou
        self.assertLess(last, first)
        self.assertGreater(last_iou, first_iou)


class GraphBevNodeTest(unittest.TestCase):
    def _graph(self, raster):
        ego = VehicleNodeInput(actor_id=1, is_ego=True, agent_slot=0, x=0, y=0, z=0, vx=1, vy=0, yaw=0,
                               route_xy=((1, 0), (2, 0)))
        collab = VehicleNodeInput(actor_id=2, is_ego=False, agent_slot=1, x=5, y=0, z=0, vx=1, vy=0, yaw=0)
        objs = [obj(10, 8.0, 0.0)]
        bev_obs = ObservationNodeInput(vehicle_id=2, modality="bev", observed_object_ids=(), bev_raster=raster)
        ego_obs = ObservationNodeInput(vehicle_id=1, modality="objlist", observed_object_ids=(10,))
        pol = WAMPolicy((2,), {2: "bev"}, {2: 1e6}, 5, "t")
        return build_wam_hetero_graph(ego=ego, collaborators=[collab], objects=objs,
                                      observations=[ego_obs, bev_obs], policy=pol,
                                      spec=GraphBuildSpec(route_waypoints=2), notable_ids={10},
                                      latency_by_vehicle={2: 0.01})

    def test_real_raster_changes_zbev(self):
        spec = BevSpec(size=16, range_m=20.0)
        real = rasterize_bev((5.0, 0.0, 0.0), [obj(10, 8.0, 0.0)], spec=spec)
        zero = np.zeros_like(real)
        net = WAMHeteroGraphNet(WAMGraphModelConfig(route_waypoints=2, hidden_dim=32, num_layers=2,
                                                    num_heads=4, bev_channels=BEV_NUM_CHANNELS, bev_size=16)).eval()
        with torch.no_grad():
            h_real = net(self._graph(real))["observation"]
            h_zero = net(self._graph(zero))["observation"]
        # the BEV observation embedding depends on the raster (real != zero placeholder)
        self.assertFalse(bool(torch.allclose(h_real, h_zero)))
        self.assertTrue(bool(torch.isfinite(h_real).all()))


if __name__ == "__main__":
    unittest.main()
