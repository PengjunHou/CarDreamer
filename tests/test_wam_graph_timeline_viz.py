"""Offline tests for the per-step WAM cooperative-graph timeline visualization.

Builds synthetic ``HeteroData`` graphs (same fixture style as test_wam_graph) and exercises the
record extraction, layered layout, matplotlib rendering, and PNG/GIF/HTML/JSONL writers. CARLA-free.
"""

import json
import tempfile
import unittest
from pathlib import Path

from car_dreamer.toolkit.wam import (
    GraphBuildSpec,
    ObjectState,
    ObservationNodeInput,
    VehicleNodeInput,
    WAMPolicy,
    build_wam_hetero_graph,
    group_records_to_frames,
    hetero_graph_to_record,
    layout_layered,
    load_records_jsonl,
    render_graph_matplotlib,
    write_graph_frames_png,
    write_graph_timeline_gif,
    write_graph_timeline_html,
)
from car_dreamer.toolkit.wam.graph_timeline_viz import append_record_jsonl


def _object(actor_id, x, y, *, object_class="vehicle"):
    return ObjectState(
        actor_id=actor_id, actor_type=f"{object_class}.test", object_class=object_class,
        x=float(x), y=float(y), z=0.0, vx=1.0, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
    )


def _graph(*, selected, modalities=None, latency=None):
    """ego(1) sees 100,102; collaborator 2 sees 101 (invisible to ego); optional collaborator 3."""
    modalities = modalities or {vid: "objlist" for vid in selected}
    latency = latency or {vid: 0.04 for vid in selected}
    ego = VehicleNodeInput(actor_id=1, is_ego=True, agent_slot=0, x=0.0, y=0.0, z=0.0, vx=2.0, vy=0.0, yaw=0.0,
                           route_xy=((5.0, 0.0), (10.0, 0.0)))
    collab2 = VehicleNodeInput(actor_id=2, is_ego=False, agent_slot=1, x=10.0, y=5.0, z=0.0, vx=1.0, vy=0.0, yaw=90.0)
    collab3 = VehicleNodeInput(actor_id=3, is_ego=False, agent_slot=2, x=-10.0, y=5.0, z=0.0, vx=1.0, vy=0.0, yaw=90.0)
    objects = [_object(100, 8.0, 1.0), _object(101, 12.0, 6.0), _object(102, 3.0, 0.5, object_class="pedestrian")]
    observations = [
        ObservationNodeInput(vehicle_id=1, modality="objlist", observed_object_ids=(100, 102),
                             latency_s=0.0, freshness=1.0, payload_bytes=200.0,
                             det_confidence_by_object={100: 0.9, 102: 0.5}),
        ObservationNodeInput(vehicle_id=2, modality=modalities.get(2, "objlist"), observed_object_ids=(101,),
                             latency_s=latency.get(2, 0.04), freshness=0.8, payload_bytes=80.0),
        ObservationNodeInput(vehicle_id=3, modality=modalities.get(3, "bev"), observed_object_ids=(100,),
                             latency_s=latency.get(3, 0.06), freshness=0.7, payload_bytes=131072.0),
    ]
    policy = WAMPolicy(
        selected_vehicle_ids=tuple(selected),
        modality_by_vehicle={vid: modalities.get(vid, "objlist") for vid in selected},
        bandwidth_by_vehicle={vid: 3e6 for vid in selected}, frequency_steps=5, reason="test",
    )
    return build_wam_hetero_graph(
        ego=ego, collaborators=[collab2, collab3], objects=objects, observations=observations,
        policy=policy, spec=GraphBuildSpec(route_waypoints=6, max_object_nodes=32),
        notable_ids={101}, latency_by_vehicle=latency,
    )


class RecordExtractionTest(unittest.TestCase):
    def test_ego_only_graph_is_local(self):
        rec = hetero_graph_to_record(_graph(selected=()), step=0, policy_label="ego_only")
        self.assertEqual(rec["counts"]["vehicles"], 1)
        self.assertTrue(rec["vehicles"][0]["is_ego"])
        self.assertEqual(rec["edges"]["veh_veh"], [])
        self.assertFalse(rec["is_v2v"])

    def test_v2v_graph_captures_modalities_and_latency(self):
        rec = hetero_graph_to_record(
            _graph(selected=(2, 3), modalities={2: "objlist", 3: "bev"}, latency={2: 0.04, 3: 0.06}),
            step=3, policy_label="all_candidates", policy_id=7,
        )
        self.assertEqual(rec["counts"]["vehicles"], 3)
        self.assertTrue(rec["is_v2v"])
        self.assertEqual(rec["policy_id"], 7)
        mods = sorted(o["modality"] for o in rec["observations"])
        self.assertEqual(mods, ["bev", "objlist", "objlist"])
        # veh_veh edges carry the measured latency L_M for each collaborator
        latencies = sorted(round(e["attr"], 3) for e in rec["edges"]["veh_veh"])
        self.assertEqual(latencies, [0.04, 0.06])
        # obs_obj edges carry detection confidence
        self.assertTrue(all("attr" in e for e in rec["edges"]["obs_obj"]))


class LayoutTest(unittest.TestCase):
    def test_rows_are_ordered_and_ego_centered(self):
        rec = hetero_graph_to_record(_graph(selected=(2, 3)), step=0, policy_label="p")
        pos = layout_layered(rec)
        veh_y = [pos[("vehicle", v["idx"])][1] for v in rec["vehicles"]]
        obs_y = [pos[("observation", o["idx"])][1] for o in rec["observations"]]
        obj_y = [pos[("object", o["idx"])][1] for o in rec["objects"] if o["valid"]]
        self.assertTrue(min(veh_y) > max(obs_y) > max(obj_y))  # vehicle row above obs row above object row
        ego_idx = next(v["idx"] for v in rec["vehicles"] if v["is_ego"])
        self.assertAlmostEqual(pos[("vehicle", ego_idx)][0], 0.0)  # ego centered


class RenderAndWriteTest(unittest.TestCase):
    def _records(self):
        return [
            hetero_graph_to_record(_graph(selected=()), step=0, policy_label="active"),
            hetero_graph_to_record(_graph(selected=(2,)), step=1, policy_label="active"),
            hetero_graph_to_record(_graph(selected=(2, 3), modalities={2: "objlist", 3: "bev"}), step=2, policy_label="active"),
        ]

    def test_render_returns_figure(self):
        import matplotlib

        fig = render_graph_matplotlib(self._records()[2])
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertGreater(len(fig.axes[0].patches), 0)
        matplotlib.pyplot.close(fig)

    def test_group_records_to_frames(self):
        frames = group_records_to_frames(self._records())
        self.assertEqual([f["step"] for f in frames], [0, 1, 2])
        self.assertEqual(len(frames[0]["panels"]), 1)

    def test_counterfactual_frame_has_multiple_panels(self):
        recs = [
            hetero_graph_to_record(_graph(selected=()), step=0, policy_label="ego_only"),
            hetero_graph_to_record(_graph(selected=(2,)), step=0, policy_label="single_objlist"),
        ]
        frames = group_records_to_frames(recs)
        self.assertEqual(len(frames), 1)
        self.assertEqual(set(frames[0]["panels"]), {"ego_only", "single_objlist"})

    def test_episode_aware_grouping_keeps_episodes_separate(self):
        r0 = hetero_graph_to_record(_graph(selected=(2,)), step=5, policy_label="coop[2]")
        r0["extra"] = {"episode": 0}
        r1 = hetero_graph_to_record(_graph(selected=(2,)), step=5, policy_label="coop[2]")
        r1["extra"] = {"episode": 1}
        frames = group_records_to_frames([r0, r1])
        self.assertEqual(len(frames), 2)  # same step number, different episodes -> distinct frames
        self.assertEqual([f["episode"] for f in frames], [0, 1])

    def test_node_labels_and_notable_fill(self):
        import matplotlib
        import matplotlib.colors as mcolors
        from matplotlib.patches import Circle

        from car_dreamer.toolkit.wam.graph_timeline_viz import COLOR_OBJECT, COLOR_OBJECT_UNIMPORTANT

        rec = hetero_graph_to_record(_graph(selected=(2,)), step=0, policy_label="p")  # obj101 notable
        fig = render_graph_matplotlib(rec)
        ax = fig.axes[0]
        texts = {t.get_text() for t in ax.texts}
        self.assertIn("ego", texts)
        self.assertIn("V1", texts)  # collaborator label uses V (not "vehicle")
        self.assertTrue(any(t.startswith("O") for t in texts))  # object label uses O (not "obj")
        self.assertFalse(any(t.startswith("obj") for t in texts))

        def rgba(c):
            return tuple(round(v, 3) for v in mcolors.to_rgba(c))

        circle_colors = {rgba(p.get_facecolor()) for p in ax.patches if isinstance(p, Circle)}
        self.assertIn(rgba(COLOR_OBJECT), circle_colors)  # notable object -> red
        self.assertIn(rgba(COLOR_OBJECT_UNIMPORTANT), circle_colors)  # non-notable -> gray
        matplotlib.pyplot.close(fig)

    def test_writers_produce_files(self):
        records = self._records()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pngs = write_graph_frames_png(records, d / "frames")
            self.assertEqual(len(pngs), 3)
            self.assertTrue(all(p.stat().st_size > 0 for p in pngs))

            gif = write_graph_timeline_gif(records, d / "timeline.gif", fps=2)
            self.assertTrue(gif.exists() and gif.stat().st_size > 0)

            html = write_graph_timeline_html(records, d / "timeline.html")
            text = html.read_text(encoding="utf-8")
            self.assertGreater(html.stat().st_size, 0)
            self.assertIn('type="range"', text)  # the time-step slider
            self.assertIn("FRAMES", text)         # embedded frame array

    def test_record_has_positions_and_bev_panel(self):
        import matplotlib

        rec = hetero_graph_to_record(_graph(selected=(2, 3)), step=0, policy_label="p")
        # ego-frame positions extracted into the record (for the BEV panel)
        self.assertIn("x", rec["vehicles"][0])
        self.assertIn("y", rec["vehicles"][0])
        self.assertIn("x", rec["objects"][0])
        self.assertTrue(any(abs(o.get("x", 0.0)) + abs(o.get("y", 0.0)) > 0 for o in rec["objects"]))

        fig = render_graph_matplotlib(rec)  # topology (left) + BEV (right)
        self.assertEqual(len(fig.axes), 2)
        bev_titles = [ax.get_title() for ax in fig.axes if "BEV" in ax.get_title()]
        self.assertTrue(bev_titles)
        matplotlib.pyplot.close(fig)

        fig2 = render_graph_matplotlib(rec, with_bev=False)  # topology only
        self.assertEqual(len(fig2.axes), 1)
        matplotlib.pyplot.close(fig2)

    def test_birdeye_image_used_when_available(self):
        import matplotlib
        import cv2
        import numpy as np

        from car_dreamer.toolkit.wam import BevOptions
        from car_dreamer.toolkit.wam.graph_timeline_viz import _resolve_birdeye_path

        rec = hetero_graph_to_record(_graph(selected=(2,)), step=7, policy_label="p")
        ego_id = next(v["node_id"] for v in rec["vehicles"] if v["is_ego"])
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            vdir = d / f"vehicle_{ego_id}"
            vdir.mkdir(parents=True)
            cv2.imwrite(str(vdir / "birdeye_000007.png"), np.zeros((128, 128, 3), np.uint8))
            self.assertIsNotNone(_resolve_birdeye_path(rec, d))  # resolved by ego id + step
            fig = render_graph_matplotlib(rec, bev=BevOptions(birdeye_dir=d))
            self.assertTrue(any(len(ax.images) > 0 for ax in fig.axes))  # birdeye imshow'd
            # the policy graph nodes are overlaid (V/O labels) on top of the birdeye
            bev_ax = next(ax for ax in fig.axes if ax.images)
            texts = {t.get_text() for t in bev_ax.texts}
            self.assertIn("ego", texts)
            self.assertTrue(any(t.startswith("O") for t in texts))
            matplotlib.pyplot.close(fig)
        self.assertIsNone(_resolve_birdeye_path(rec, None))  # no dir -> scatter fallback

    def test_html_uses_independent_per_policy_images(self):
        recs = [
            hetero_graph_to_record(_graph(selected=()), step=0, policy_label="ego_only"),
            hetero_graph_to_record(_graph(selected=(2,)), step=0, policy_label="single[2]"),
        ]
        with tempfile.TemporaryDirectory() as d:
            html = write_graph_timeline_html(recs, Path(d) / "t.html")
            text = html.read_text(encoding="utf-8")
            self.assertIn('id="panels"', text)  # container for separate images
            self.assertIn("for (const b64 of FRAMES", text)  # each policy rendered as its own <img>

    def test_jsonl_roundtrip(self):
        records = self._records()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "timeline.jsonl"
            for rec in records:
                append_record_jsonl(rec, path)
            loaded = load_records_jsonl(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[2]["counts"]["vehicles"], 3)
            # loaded plain dicts still render
            write_graph_frames_png(loaded, Path(d) / "frames2")


if __name__ == "__main__":
    unittest.main()
