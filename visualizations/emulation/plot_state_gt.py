"""Plot ground-truth time-series for any registered state without needing a trained model.

Mirrors ``plot_state_prediction.py`` but only renders the GT panel by reading
``vehicle.shared_summary_semantic[k]`` (and friends) directly from the episode
JSON.

Usage (from repo root):

    conda run -n cardreamer_gnn python visualizations/emulation/plot_state_gt.py \\
        --episode data/emulation_fixed_2026059/P3/right_turn_episode_000004.json \\
        --state positive_score_mean \\
        --out-dir logdir/state_gt_viz

    # per-(vehicle, query) state needs a query id from the episode:
    conda run -n cardreamer_gnn python visualizations/emulation/plot_state_gt.py \\
        --episode data/emulation_fixed_2026059/P3/right_turn_episode_000004.json \\
        --state sender_collab --query-id clg_front_vehicle
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visualizations._common import load_training_module  # noqa: E402


@dataclass
class GTSpec:
    gt_fn: Callable[[Any, Optional[str]], float]
    label: str
    needs_query: bool = False


def _semantic_idx_gt(idx: int) -> Callable[[Any, Optional[str]], float]:
    def _fn(vehicle: Any, qid: Optional[str]) -> float:
        vec = list(vehicle.shared_summary_semantic) or []
        return float(vec[idx]) if len(vec) > idx else float("nan")
    return _fn


def _build_state_registry() -> Dict[str, GTSpec]:
    reg: Dict[str, GTSpec] = {
        "complementarity": GTSpec(
            gt_fn=lambda v, qid: float(v.complementarity),
            label="complementarity",
        ),
        "accessibility": GTSpec(
            gt_fn=lambda v, qid: float(v.accessibility),
            label="accessibility",
        ),
    }
    semantic_states = {
        "positive_score_mean": (0, "mean(positive_score)"),
        "negative_score_mean": (1, "mean(negative_score)"),
        "unknown_score_mean":  (2, "mean(unknown_score)"),
        "confidence_mean":     (3, "mean(confidence)"),
        "belief_mean":         (4, "mean(belief)"),
        "evidence_mean":       (5, "mean(evidence)"),
        "answerability_mean":  (6, "mean(answerability_score)"),
        "visibility_mean":     (7, "mean(visibility_score)"),
    }
    for name, (idx, label) in semantic_states.items():
        reg[name] = GTSpec(gt_fn=_semantic_idx_gt(idx), label=label)
    raw_states = {
        "delta_pos_x": (0, lambda v, qid: float(v.delta_pos[0])),
        "delta_pos_y": (1, lambda v, qid: float(v.delta_pos[1])),
        "delta_vel_x": (2, lambda v, qid: float(v.delta_vel[0])),
        "delta_vel_y": (3, lambda v, qid: float(v.delta_vel[1])),
        "delta_yaw":   (4, lambda v, qid: float(v.delta_yaw)),
    }
    for name, (idx, gt_fn) in raw_states.items():
        reg[name] = GTSpec(gt_fn=gt_fn, label=name)
    reg["sender_collab"] = GTSpec(
        gt_fn=lambda v, qid: float(v.sender_collab.get(qid, 0.0)),
        label="sender_collab",
        needs_query=True,
    )
    reg["sender_gain"] = GTSpec(
        gt_fn=lambda v, qid: float(v.sender_gain.get(qid, 0.0)),
        label="sender_gain",
        needs_query=True,
    )
    return reg


STATE_REGISTRY = _build_state_registry()


def _render(
    *,
    steps_arr: np.ndarray,
    gt_arr: np.ndarray,
    slot_to_vid: Dict[int, int],
    spec: GTSpec,
    qid: Optional[str],
    state_name: str,
    episode: Any,
    out_dir: Path,
    max_vehicles: int,
) -> None:
    num_nodes = gt_arr.shape[1]
    active_slots = [
        slot for slot in range(num_nodes) if not np.all(np.isnan(gt_arr[:, slot]))
    ]
    if max_vehicles and len(active_slots) > max_vehicles:
        coverage = [int(np.sum(~np.isnan(gt_arr[:, slot]))) for slot in active_slots]
        active_slots = [s for _, s in sorted(zip(coverage, active_slots), reverse=True)][
            :max_vehicles
        ]
    if not active_slots:
        suffix = f" [{qid}]" if qid is not None else ""
        print(f"  skip: no vehicles with GT for state={state_name}{suffix}")
        return

    metric_label = f"{spec.label}{(' [' + qid + ']') if qid is not None else ''}"
    fname_state = state_name if qid is None else f"{state_name}__{qid}"

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 1, figsize=(13, 4.2))
    for i, slot in enumerate(active_slots):
        ax.plot(
            steps_arr,
            gt_arr[:, slot],
            marker="o",
            ms=3,
            linewidth=1.4,
            color=cmap(i % 10),
            label=f"vehicle {slot_to_vid[slot]}",
        )
    fig.suptitle(
        f"{metric_label}  —  episode {episode.episode_id}, policy "
        f"{episode.policy_id or 'n/a'}  (GT only, no model)",
        fontsize=12,
    )
    ax.set_xlabel("time step")
    ax.set_ylabel(f"{metric_label} (GT)")
    ax.grid(alpha=0.25)
    ax.legend(
        ncol=min(len(active_slots), 4),
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    fig_path = out_dir / f"{fname_state}_gt_only_{episode.episode_id}.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    import csv
    csv_path = out_dir / f"{fname_state}_gt_only_{episode.episode_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["step"] + [f"gt_v{slot_to_vid[s]}" for s in active_slots]
        writer.writerow(header)
        for row_idx, t in enumerate(steps_arr):
            row: List[Any] = [int(t)]
            row.extend(gt_arr[row_idx, slot] for slot in active_slots)
            writer.writerow(row)
    print(f"Saved CSV:    {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GT (no model required) for any registered state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episode", help="Single canonical episode JSON file.")
    parser.add_argument(
        "--state",
        default="positive_score_mean",
        choices=sorted(STATE_REGISTRY.keys()),
    )
    parser.add_argument(
        "--query-id",
        default=None,
        help="For per-(vehicle, query) states. Pass 'all' to render every query in the episode.",
    )
    parser.add_argument("--out-dir", default="logdir/state_gt_viz")
    parser.add_argument("--scene-type", default="right_turn")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-vehicles-per-figure", type=int, default=8)
    parser.add_argument(
        "--list-states", action="store_true", help="Print available --state values and exit."
    )
    args = parser.parse_args()

    if args.list_states:
        for name, spec in sorted(STATE_REGISTRY.items()):
            tag = " (needs --query-id)" if spec.needs_query else ""
            print(f"  {name:<24s}  {spec.label}{tag}")
        return
    if not args.episode:
        parser.error("--episode is required (unless --list-states).")

    spec = STATE_REGISTRY[args.state]
    training = load_training_module()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = training.parse_episode_source_spec(
        args.episode, default_scene_type=args.scene_type, default_dt=args.dt
    )
    episodes = training.load_episodes_from_sources([source])
    if not episodes:
        raise SystemExit(f"No episodes loaded from {args.episode}")
    episode = episodes[0]

    # Resolve queries to render.
    episode_qids: List[str] = []
    for step in episode.steps:
        for vehicle in step.candidate_vehicles:
            for qid in getattr(vehicle, "sender_collab", {}) or {}:
                if qid not in episode_qids:
                    episode_qids.append(qid)
            for qid in getattr(vehicle, "sender_gain", {}) or {}:
                if qid not in episode_qids:
                    episode_qids.append(qid)

    if spec.needs_query:
        if args.query_id and args.query_id != "all":
            if episode_qids and args.query_id not in episode_qids:
                raise SystemExit(
                    f"--query-id {args.query_id!r} not in episode queries: {episode_qids}"
                )
            target_qids: List[Optional[str]] = [args.query_id]
        else:
            target_qids = list(episode_qids) or [None]
    else:
        target_qids = [None]

    # Stable slot assignment based on first-appearance order.
    vid_to_slot: Dict[int, int] = {}
    slot_to_vid: Dict[int, int] = {}
    for step in episode.steps:
        for vehicle in step.candidate_vehicles:
            vid = int(vehicle.vehicle_id)
            if vid not in vid_to_slot:
                slot = len(vid_to_slot)
                vid_to_slot[vid] = slot
                slot_to_vid[slot] = vid
    num_nodes = len(vid_to_slot)
    if num_nodes == 0:
        raise SystemExit("No candidate_vehicles found in the episode.")

    num_steps = len(episode.steps)
    steps_arr = np.arange(num_steps, dtype=int)

    print(
        f"Episode: id={episode.episode_id} policy={episode.policy_id} "
        f"steps={num_steps}  vehicles={num_nodes}  state={args.state}"
        + (f"  queries={target_qids}" if spec.needs_query else "")
    )

    for qid in target_qids:
        gt_arr = np.full((num_steps, num_nodes), np.nan, dtype=np.float64)
        for step_idx, step in enumerate(episode.steps):
            for vehicle in step.candidate_vehicles:
                slot = vid_to_slot.get(int(vehicle.vehicle_id))
                if slot is None:
                    continue
                try:
                    gt_arr[step_idx, slot] = float(spec.gt_fn(vehicle, qid))
                except Exception:
                    gt_arr[step_idx, slot] = float("nan")
        _render(
            steps_arr=steps_arr,
            gt_arr=gt_arr,
            slot_to_vid=slot_to_vid,
            spec=spec,
            qid=qid,
            state_name=args.state,
            episode=episode,
            out_dir=out_dir,
            max_vehicles=args.max_vehicles_per_figure,
        )


if __name__ == "__main__":
    main()
