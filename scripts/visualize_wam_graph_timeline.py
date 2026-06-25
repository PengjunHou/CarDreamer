#!/usr/bin/env python3
"""Render a recorded WAM cooperative-graph timeline (JSONL) to HTML / PNG / GIF.

CARLA-free. Reads the JSONL produced by ``scripts/record_wam_graph_timeline.py`` (actual per-step
graph under the active policy) and writes:

  * an interactive HTML with a time-step slider (``--html``),
  * one PNG per step (``--png-dir``),
  * an animated GIF (``--gif``).

Records that share a step are laid out as separate panels. Use ``--policies`` to restrict which
policy labels are shown.

Examples
--------
    python scripts/visualize_wam_graph_timeline.py --jsonl outputs/graph_timeline.jsonl \
        --html outputs/graph_timeline.html --gif outputs/graph_timeline.gif --png-dir outputs/frames

By default the BEV panel reuses the exact data/birdeye_frames look (road/lane colours, green vehicles,
blue ego) on a fixed per-episode crop of the recorded CARLA map: the background stays put while the
ego box moves across it. Use ``--bev-mode auto --birdeye-dir data/birdeye_frames`` to instead use the
dumped CARLA birdeye images as the background when available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _policy_matches(record, keep) -> bool:
    """Match a record against --policies tokens.

    Labels can carry an actor-id suffix (e.g. ``coop[551,554]``), so a clean family name is matched
    against the record's ``extra.policy_type`` AND the label's prefix (text before ``[``), not just
    the full label.
    """
    label = str(record.get("policy_label", ""))
    base = label.split("[", 1)[0]
    policy_type = str((record.get("extra") or {}).get("policy_type") or "")
    return bool(keep & {label, base, policy_type})


def _merge_uncertainty_csv(records: List[dict], csv_path: Path) -> None:
    """Inject per-step motion/coverage/total uncertainty from a CSV into ``record.extra.uncertainty``.

    Matches on ``(episode_index, step)`` -> ``(extra.episode, step)`` so the timeline's bottom panel
    can show uncertainty even when the JSONL itself does not carry it (e.g. the graph timeline written
    by compare_wam_fixed_policies.py, whose uncertainty lives in a sibling CSV).
    """
    import csv as _csv

    by_key = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                key = (int(float(row["episode_index"])), int(float(row["step"])))
            except (KeyError, ValueError):
                continue
            by_key[key] = {
                "motion_uncertainty": float(row.get("motion_uncertainty", 0.0) or 0.0),
                "coverage_uncertainty": float(row.get("coverage_uncertainty", 0.0) or 0.0),
                "total_uncertainty": float(row.get("total_uncertainty", 0.0) or 0.0),
            }
    matched = 0
    for rec in records:
        extra = rec.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            rec["extra"] = extra
        key = (int(extra.get("episode", 0)), int(rec.get("step", 0)))
        if key in by_key:
            extra["uncertainty"] = by_key[key]
            matched += 1
    print(f"merged uncertainty from {csv_path}: {matched}/{len(records)} records matched", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a recorded WAM graph timeline (JSONL).")
    parser.add_argument("--jsonl", required=True, help="input timeline JSONL")
    parser.add_argument("--html", default=None, help="write interactive HTML to this path")
    parser.add_argument("--gif", default=None, help="write animated GIF to this path")
    parser.add_argument("--png-dir", default=None, help="write one PNG per step into this directory")
    parser.add_argument("--policies", default=None,
                        help="comma-separated policy family names or labels to keep. Family names "
                             "match actor-id-suffixed labels too. Default: all.")
    parser.add_argument("--birdeye-dir", default=None,
                        help="directory of real CARLA birdeye dumps (data/birdeye_frames). When given, "
                             "the BEV panel uses <dir>/vehicle_<egoid>/birdeye_<step>.png as background "
                             "and overlays the policy graph's vehicles/objects in auto/birdeye-dir mode.")
    parser.add_argument("--bev-mode", choices=("generated", "auto", "birdeye-dir", "scatter"), default="generated",
                        help="BEV panel source. generated draws a fixed episode-level birdeye-style canvas "
                             "without reading birdeye dumps.")
    parser.add_argument("--bev-frame", choices=("map", "birdeye", "world", "episode_start"), default="map",
                        help="generated BEV frame. map = fixed crop of the recorded map with birdeye "
                             "colours and a moving ego (default); birdeye = ego-centric warp; "
                             "world/episode_start are debugging matplotlib views.")
    parser.add_argument("--bev-margin-m", type=float, default=10.0,
                        help="margin around episode-level generated BEV bounds (default 10m).")
    parser.add_argument("--hide-candidates", action="store_true", default=False,
                        help="do not draw non-selected cooperative candidate vehicles in generated BEV.")
    parser.add_argument("--bev-obs-range", type=float, default=64.0,
                        help="birdeye obs_range in meters (matches the recorded birdeye; default 64).")
    parser.add_argument("--bev-ego-offset", type=float, default=12.0,
                        help="birdeye ego_offset in meters (matches the recorded birdeye; default 12).")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--title", default="WAM cooperative graph timeline")
    parser.add_argument("--no-uncertainty", dest="uncertainty", action="store_false", default=True,
                        help="hide the bottom motion/coverage/total uncertainty-vs-step panel")
    parser.add_argument("--uncertainty-csv", default=None,
                        help="merge per-step motion/coverage/total uncertainty from this CSV "
                             "(columns episode_index, step, *_uncertainty) into the records, e.g. the "
                             "fixed_policy_uncertainty.csv from compare_wam_fixed_policies.py")
    parser.add_argument("--unc-metric", choices=("raw", "norm"), default="raw",
                        help="bottom panel: raw motion/coverage/total, or the [0,1]-saturated *_norm")
    parser.add_argument("--unc-sigma-scale", type=float, default=4.0,
                        help="tau for norm motion = 1-exp(-motion/tau) when *_norm is not in the data")
    parser.add_argument("--unc-alpha", type=float, default=0.5,
                        help="convex weight motion vs coverage for norm total when not in the data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from car_dreamer.toolkit.wam import (
        BevOptions,
        load_records_jsonl,
        write_graph_frames_png,
        write_graph_timeline_gif,
        write_graph_timeline_html,
    )

    records = load_records_jsonl(args.jsonl)
    if args.policies:
        keep = {p.strip() for p in args.policies.split(",") if p.strip()}
        records = [r for r in records if _policy_matches(r, keep)]
    if not records:
        raise SystemExit(f"no records to render from {args.jsonl} (after --policies filter)")

    if args.uncertainty_csv:
        _merge_uncertainty_csv(records, Path(args.uncertainty_csv))

    if not (args.html or args.gif or args.png_dir):
        args.html = str(Path(args.jsonl).with_suffix(".html"))  # sensible default

    bev = BevOptions(
        birdeye_dir=args.birdeye_dir,
        obs_range=args.bev_obs_range,
        ego_offset=args.bev_ego_offset,
        mode=args.bev_mode,
        frame=args.bev_frame,
        margin_m=args.bev_margin_m,
        show_candidates=not bool(args.hide_candidates),
    )
    unc = bool(args.uncertainty)
    unc_kw = dict(show_uncertainty=unc, unc_mode=args.unc_metric,
                  unc_sigma_scale=args.unc_sigma_scale, unc_alpha=args.unc_alpha)
    if args.html:
        out = write_graph_timeline_html(records, args.html, title=args.title, fps=args.fps, dpi=args.dpi, bev=bev,
                                        **unc_kw)
        print(f"HTML  -> {out}", flush=True)
    if args.gif:
        out = write_graph_timeline_gif(records, args.gif, fps=args.fps, dpi=args.dpi, bev=bev, **unc_kw)
        print(f"GIF   -> {out}", flush=True)
    if args.png_dir:
        paths = write_graph_frames_png(records, args.png_dir, dpi=args.dpi, bev=bev, **unc_kw)
        print(f"PNG   -> {len(paths)} frames in {args.png_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
