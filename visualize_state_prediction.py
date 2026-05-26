"""Visualize per-vehicle world-model predictions vs ground-truth, for any
exposed state. Choose which state to plot via --state.

For every step `s` of the chosen episode where the trained world model has a
full `history_len` window of past observations and at least one future step
exists, we:

  1. Build the corresponding sample using the canonical emulation dataset.
  2. Run the model in inference mode.
  3. Read the offset-0 prediction (the predicted "next step", i.e. s + 1).
  4. Read the actual value at step `s + 1` from the episode JSON.

We then plot, for each candidate vehicle, two stacked line charts:

    Top    — ground-truth values over time
    Bottom — predicted next-step values over time

Available states (--state):

    complementarity         derived_complementarity head
    accessibility           derived_accessibility head
    positive_score_mean     shared_state[..., 8]   GT = shared_summary_semantic[0]
    positive_score_max      shared_state[..., 14]  GT = shared_summary_semantic[6]
    negative_score_mean     shared_state[..., 9]   GT = shared_summary_semantic[1]
    negative_score_max      shared_state[..., 15]  GT = shared_summary_semantic[7]
    unknown_score_mean      shared_state[..., 10]  GT = shared_summary_semantic[2]
    confidence_mean         shared_state[..., 11]  GT = shared_summary_semantic[3]
    belief_mean             shared_state[..., 12]  GT = shared_summary_semantic[4]
    evidence_mean           shared_state[..., 13]  GT = shared_summary_semantic[5]
    delta_pos_x             raw_state[..., 0]      GT = vehicle.delta_pos[0]
    delta_pos_y             raw_state[..., 1]      GT = vehicle.delta_pos[1]
    delta_vel_x             raw_state[..., 2]      GT = vehicle.delta_vel[0]
    delta_vel_y             raw_state[..., 3]      GT = vehicle.delta_vel[1]
    delta_yaw               raw_state[..., 4]      GT = vehicle.delta_yaw
    sender_collab           sender_collab[..., qidx]  needs --query-id
    sender_gain             sender_gain[..., qidx]    needs --query-id

Run --list-states to print the registry without running anything.

Usage:
    conda run -n cardreamer_gnn python visualize_state_prediction.py \
        --checkpoint logdir/emulation/fixed_20260430/checkpoint_best.pt \
        --episode data/emulation_fixed_20260430/P3/emulation_episode_terminated_step_101.json \
        --state complementarity \
        --out-dir logdir/state_pred_viz

    # per-(vehicle, query) state needs a query id from the episode:
    conda run -n cardreamer_gnn python visualize_state_prediction.py \
        --checkpoint logdir/emulation/fixed_20260430/checkpoint_best.pt \
        --episode data/emulation_fixed_20260430/P3/emulation_episode_terminated_step_101.json \
        --state sender_collab --query-id clg_front_vehicle
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
EMULATION_ROOT = REPO_ROOT / "car_dreamer" / "toolkit" / "emulation"


# --------------------------------------------------------------------------- #
# State registry
# --------------------------------------------------------------------------- #


@dataclass
class StateSpec:
    """Describe how to extract one scalar-per-vehicle quantity from the model
    predictions and the ground-truth episode."""

    pred_fn: Callable[[Dict[str, Any], Optional[int]], np.ndarray]
    gt_fn: Callable[[Any, Optional[str]], float]
    label: str
    needs_query: bool = False


def _shared_dim(idx: int) -> Callable[[Dict[str, Any], Optional[int]], np.ndarray]:
    return lambda preds, qidx: preds["shared_state"][0, 0, :, idx].detach().cpu().numpy()


def _semantic_idx_gt(idx: int) -> Callable[[Any, Optional[str]], float]:
    def _fn(vehicle: Any, qid: Optional[str]) -> float:
        vec = list(vehicle.shared_summary_semantic) or []
        return float(vec[idx]) if len(vec) > idx else float("nan")
    return _fn


def _raw_dim(idx: int) -> Callable[[Dict[str, Any], Optional[int]], np.ndarray]:
    return lambda preds, qidx: preds["raw_state"][0, 0, :, idx].detach().cpu().numpy()


def _build_state_registry() -> Dict[str, StateSpec]:
    reg: Dict[str, StateSpec] = {}

    reg["complementarity"] = StateSpec(
        pred_fn=lambda preds, qidx: preds["derived_complementarity"][0, 0].detach().cpu().numpy(),
        gt_fn=lambda v, qid: float(v.complementarity),
        label="complementarity",
    )
    reg["accessibility"] = StateSpec(
        pred_fn=lambda preds, qidx: preds["derived_accessibility"][0, 0].detach().cpu().numpy(),
        gt_fn=lambda v, qid: float(v.accessibility),
        label="accessibility",
    )

    # shared_summary_semantic indices: [0]=mean(pos), [1]=mean(neg), [2]=mean(unk),
    # [3]=mean(conf), [4]=mean(belief), [5]=mean(evidence), [6]=max(pos), [7]=max(neg)
    # The 20-dim shared_state target is raw_summary(8) | semantic(8) | intent(4),
    # so semantic index k lives at shared_state index 8 + k.
    semantic_states = {
        "positive_score_mean": (0, "mean(positive_score)"),
        "negative_score_mean": (1, "mean(negative_score)"),
        "unknown_score_mean":  (2, "mean(unknown_score)"),
        "confidence_mean":     (3, "mean(confidence)"),
        "belief_mean":         (4, "mean(belief)"),
        "evidence_mean":       (5, "mean(evidence)"),
        "positive_score_max":  (6, "max(positive_score)"),
        "negative_score_max":  (7, "max(negative_score)"),
    }
    for name, (sem_idx, label) in semantic_states.items():
        reg[name] = StateSpec(
            pred_fn=_shared_dim(8 + sem_idx),
            gt_fn=_semantic_idx_gt(sem_idx),
            label=label,
        )

    # raw_state target dims (delta_pos x/y, delta_vel x/y, delta_yaw)
    raw_states = {
        "delta_pos_x": (0, lambda v, qid: float(v.delta_pos[0])),
        "delta_pos_y": (1, lambda v, qid: float(v.delta_pos[1])),
        "delta_vel_x": (2, lambda v, qid: float(v.delta_vel[0])),
        "delta_vel_y": (3, lambda v, qid: float(v.delta_vel[1])),
        "delta_yaw":   (4, lambda v, qid: float(v.delta_yaw)),
    }
    for name, (idx, gt_fn) in raw_states.items():
        reg[name] = StateSpec(pred_fn=_raw_dim(idx), gt_fn=gt_fn, label=name)

    # per-(vehicle, query) heads — need a query id resolved to qidx
    reg["sender_collab"] = StateSpec(
        pred_fn=lambda preds, qidx: preds["sender_collab"][0, 0, :, qidx].detach().cpu().numpy(),
        gt_fn=lambda v, qid: float(v.sender_collab.get(qid, 0.0)),
        label="sender_collab",
        needs_query=True,
    )
    reg["sender_gain"] = StateSpec(
        pred_fn=lambda preds, qidx: preds["sender_gain"][0, 0, :, qidx].detach().cpu().numpy(),
        gt_fn=lambda v, qid: float(v.sender_gain.get(qid, 0.0)),
        label="sender_gain",
        needs_query=True,
    )

    return reg


STATE_REGISTRY = _build_state_registry()


# --------------------------------------------------------------------------- #
# Module loading
# --------------------------------------------------------------------------- #


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load(full_name: str, file_name: str):
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, EMULATION_ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_emulation_modules():
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", REPO_ROOT / "car_dreamer" / "toolkit")
    _ensure_pkg("car_dreamer.toolkit.emulation", EMULATION_ROOT)
    _load("car_dreamer.toolkit.emulation.schema", "schema.py")
    _load("car_dreamer.toolkit.emulation.features", "features.py")
    _load("car_dreamer.toolkit.emulation.queries", "queries.py")
    dataset_mod = _load("car_dreamer.toolkit.emulation.dataset", "dataset.py")
    training = _load("car_dreamer.toolkit.emulation.training", "training.py")
    model_mod = _load("car_dreamer.toolkit.emulation.model", "model.py")
    return dataset_mod, training, model_mod


def _to_batch_tensor(sample: Dict[str, Any], torch):
    tensor_keys = (
        "node_features", "state_node_features", "action_features",
        "vehicle_exogenous_features", "ego_state_features", "step_exogenous_features",
        "component_valid_mask", "node_mask", "edge_index", "edge_attr", "edge_mask",
        "query_features", "query_mask", "task_relevance",
        "future_mask", "future_node_mask", "future_action_features",
        "target_raw_state", "target_shared_state", "target_sender_collab",
        "target_sender_gain", "target_ego_sc", "history_mask",
    )
    return {k: torch.as_tensor(sample[k]).unsqueeze(0) for k in tensor_keys if k in sample}


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #


def _smooth_columns_nan(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean over axis 0, NaN-aware. Each output position is
    the mean of finite values inside a window of size ``window`` centered on
    that row; positions with no finite values stay NaN."""
    if window <= 1 or arr.size == 0:
        return arr
    half = window // 2
    n = arr.shape[0]
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = arr[lo:hi]
        for c in range(arr.shape[1]):
            col = chunk[:, c]
            finite = col[np.isfinite(col)]
            if finite.size > 0:
                out[i, c] = float(finite.mean())
    return out


# --------------------------------------------------------------------------- #
# Animation
# --------------------------------------------------------------------------- #


def _save_growing_lines_gif(
    pred_steps_arr: np.ndarray,
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    active_slots: List[int],
    slot_to_vid: Dict[int, int],
    *,
    metric_label: str,
    episode_id: str,
    policy_id: str,
    out_path: Path,
    fps: float,
) -> None:
    """Save a GIF that progressively reveals GT (top) and predicted (bottom)
    line plots one timestep at a time."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)

    gt_lines = []
    pred_lines = []
    for i, slot in enumerate(active_slots):
        color = cmap(i % 10)
        label = f"Collaborator {i + 1}"
        (ln_gt,) = axes[0].plot([], [], marker="o", ms=3, linewidth=1.4,
                                color=color, label=label)
        (ln_pr,) = axes[1].plot([], [], marker="x", ms=4, linewidth=1.4,
                                color=color, label=label)
        gt_lines.append(ln_gt)
        pred_lines.append(ln_pr)

    cursor_top = axes[0].axvline(pred_steps_arr[0], color="0.4",
                                 linestyle="--", linewidth=0.8)
    cursor_bot = axes[1].axvline(pred_steps_arr[0], color="0.4",
                                 linestyle="--", linewidth=0.8)

    # Lock axes so they do not jump frame-to-frame.
    x_pad = max(0.5, 0.02 * (pred_steps_arr[-1] - pred_steps_arr[0] + 1))
    axes[1].set_xlim(pred_steps_arr[0] - x_pad, pred_steps_arr[-1] + x_pad)

    def _ylim(arr: np.ndarray) -> tuple:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return (-1.0, 1.0)
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if lo == hi:
            pad = 0.5 if lo == 0.0 else 0.05 * abs(lo)
            return (lo - pad, hi + pad)
        pad = 0.05 * (hi - lo)
        return (lo - pad, hi + pad)

    axes[0].set_ylim(*_ylim(gt_arr))
    axes[1].set_ylim(*_ylim(pred_arr))

    suptitle = fig.suptitle("", fontsize=14)
    axes[0].set_title("Ground truth", fontsize=14, loc="left")
    axes[1].set_title("Predicted next step (model offset = +1)", fontsize=14, loc="left")
    axes[0].set_ylabel(f"{metric_label} (GT)", fontsize=14)
    axes[1].set_ylabel(f"{metric_label} (pred)", fontsize=14)
    axes[1].set_xlabel("time step (target step = predicted step)", fontsize=14)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(
        ncol=min(len(active_slots), 4),
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))

    def _update(frame: int):
        upto = frame + 1
        x = pred_steps_arr[:upto]
        for line, slot in zip(gt_lines, active_slots):
            line.set_data(x, gt_arr[:upto, slot])
        for line, slot in zip(pred_lines, active_slots):
            line.set_data(x, pred_arr[:upto, slot])
        cursor_top.set_xdata([pred_steps_arr[frame], pred_steps_arr[frame]])
        cursor_bot.set_xdata([pred_steps_arr[frame], pred_steps_arr[frame]])
        suptitle.set_text(
            f"{metric_label}  —  episode {episode_id}, policy {policy_id}  "
            f"—  step {pred_steps_arr[frame]}"
        )
        return (*gt_lines, *pred_lines, cursor_top, cursor_bot, suptitle)

    anim = FuncAnimation(
        fig, _update, frames=len(pred_steps_arr), interval=int(1000.0 / max(fps, 0.1)),
        blit=False,
    )
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _render_query(
    *,
    pred_steps_arr: np.ndarray,
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    num_nodes: int,
    slot_to_vid: Dict[int, int],
    spec: StateSpec,
    qid: Optional[str],
    state_name: str,
    episode: Any,
    out_dir: Path,
    max_vehicles: int,
    make_gif: bool,
    gif_fps: float,
    gt_smooth_window: int = 1,
) -> None:
    """Produce PNG (+optional GIF) + CSV for a single (state, query) pair."""
    active_slots = [
        slot for slot in range(num_nodes)
        if not np.all(np.isnan(gt_arr[:, slot])) and not np.all(np.isnan(pred_arr[:, slot]))
    ]
    if max_vehicles and len(active_slots) > max_vehicles:
        coverage = [np.sum(~np.isnan(gt_arr[:, slot])) for slot in active_slots]
        active_slots = [s for _, s in sorted(zip(coverage, active_slots), reverse=True)][
            :max_vehicles
        ]
    if not active_slots:
        suffix = f" [{qid}]" if qid is not None else ""
        print(f"  skip: no active vehicles for state={state_name}{suffix}")
        return

    metric_label = f"{spec.label}{(' [' + qid + ']') if qid is not None else ''}"
    fname_state = state_name if qid is None else f"{state_name}__{qid}"

    gt_arr_plot = _smooth_columns_nan(gt_arr, gt_smooth_window)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    for i, slot in enumerate(active_slots):
        color = cmap(i % 10)
        label = f"Collaborator {i + 1}"
        axes[0].plot(pred_steps_arr, gt_arr_plot[:, slot], marker="o", ms=3,
                     linewidth=1.4, color=color, label=label)
        axes[1].plot(pred_steps_arr, pred_arr[:, slot], marker="x", ms=4,
                     linewidth=1.4, color=color, label=label)
    fig.suptitle(
        f"{metric_label}  —  episode {episode.episode_id}, policy "
        f"{episode.policy_id or 'n/a'}",
        fontsize=14,
    )
    axes[0].set_title("Ground truth", fontsize=14, loc="left")
    axes[0].set_ylabel(f"{metric_label} (GT)", fontsize=14)
    axes[0].grid(alpha=0.25)
    axes[0].legend(
        ncol=min(len(active_slots), 4),
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    axes[1].set_title("Predicted next step (model offset = +1)", fontsize=14, loc="left")
    axes[1].set_ylabel(f"{metric_label} (pred)", fontsize=14)
    axes[1].set_xlabel("time step (target step = predicted step)", fontsize=14)
    axes[1].grid(alpha=0.25)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    fig_path = out_dir / f"{fname_state}_gt_vs_pred_{episode.episode_id}.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    if make_gif:
        gif_path = out_dir / f"{fname_state}_gt_vs_pred_{episode.episode_id}.gif"
        _save_growing_lines_gif(
            pred_steps_arr,
            gt_arr,
            pred_arr,
            active_slots,
            slot_to_vid,
            metric_label=metric_label,
            episode_id=episode.episode_id,
            policy_id=episode.policy_id or "n/a",
            out_path=gif_path,
            fps=gif_fps,
        )
        print(f"Saved GIF:    {gif_path}")

    import csv
    csv_path = out_dir / f"{fname_state}_gt_vs_pred_{episode.episode_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["step"] + [f"gt_v{slot_to_vid[s]}" for s in active_slots] + [
            f"pred_v{slot_to_vid[s]}" for s in active_slots
        ]
        writer.writerow(header)
        for row_idx, t in enumerate(pred_steps_arr):
            row: List[Any] = [int(t)]
            row.extend(gt_arr[row_idx, slot] for slot in active_slots)
            row.extend(pred_arr[row_idx, slot] for slot in active_slots)
            writer.writerow(row)
    print(f"Saved CSV:    {csv_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize per-vehicle world-model predictions vs GT for any state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", help="Trained .pt checkpoint.")
    parser.add_argument("--episode", help="Single canonical episode JSON file.")
    parser.add_argument("--state", default="complementarity",
                        choices=sorted(STATE_REGISTRY.keys()),
                        help="Which state to plot.")
    parser.add_argument("--query-id", default=None,
                        help="For per-(vehicle, query) states. Omit (or pass 'all') "
                             "to render every query found in the episode.")
    parser.add_argument("--out-dir", default="logdir/state_pred_viz")
    parser.add_argument("--scene-type", default="right_turn")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-vehicles-per-figure", type=int, default=8)
    parser.add_argument("--gif", action="store_true",
                        help="Also save an animated GIF that grows over time.")
    parser.add_argument("--gif-fps", type=float, default=4.0,
                        help="Playback speed for --gif (frames per second).")
    parser.add_argument("--gt-smooth-window", type=int, default=1,
                        help="Centered moving-average window for GT only (>=2 to smooth).")
    parser.add_argument("--list-states", action="store_true",
                        help="Print available --state values and exit.")
    args = parser.parse_args()

    if args.list_states:
        for name, spec in sorted(STATE_REGISTRY.items()):
            tag = " (needs --query-id)" if spec.needs_query else ""
            print(f"  {name:<24s}  {spec.label}{tag}")
        return

    if not args.checkpoint or not args.episode:
        parser.error("--checkpoint and --episode are required (unless --list-states).")

    spec = STATE_REGISTRY[args.state]

    torch = sys.modules.get("torch") or __import__("torch")
    dataset_mod, training, model_mod = _load_emulation_modules()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = training.resolve_device(args.device)
    print(f"Device: {device}")

    # --- Load checkpoint ---------------------------------------------------
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = model_mod.GraphGRUEmulationConfig(**checkpoint["model_config"])
    model = model_mod.GraphGRUEmulationModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    history_len = int(model_cfg.history_len)
    horizon = int(model_cfg.horizon)
    print(f"Loaded checkpoint epoch={checkpoint.get('epoch')} "
          f"best_metric={checkpoint.get('best_metric')}")

    # --- Load episode + dataset --------------------------------------------
    source = training.parse_episode_source_spec(
        args.episode, default_scene_type=args.scene_type, default_dt=args.dt
    )
    episodes = training.load_episodes_from_sources([source])
    if not episodes:
        raise SystemExit(f"No episodes loaded from {args.episode}")
    episode = episodes[0]
    dataset = dataset_mod.CanonicalEmulationDataset(
        [episode], history_len=history_len, horizon=horizon
    )

    node_ids = dataset.episode_indices[0].node_ids
    slot_to_vid = {slot: int(vid) for slot, vid in enumerate(node_ids)}
    vid_to_slot = {int(vid): slot for slot, vid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    # Resolve which queries to render. For per-vehicle states this is just [None];
    # for per-(vehicle, query) states, omitting --query-id (or passing "all") fans
    # out across every query in the episode.
    episode_qids = list(dataset.episode_indices[0].query_ids)
    if spec.needs_query:
        if args.query_id and args.query_id != "all":
            if args.query_id not in episode_qids:
                raise SystemExit(
                    f"--query-id {args.query_id!r} not in episode queries: {episode_qids}"
                )
            target_qids: List[Optional[str]] = [args.query_id]
        else:
            target_qids = list(episode_qids)
    else:
        target_qids = [None]

    qid_to_qidx: Dict[Optional[str], Optional[int]] = {
        qid: (episode_qids.index(qid) if qid is not None else None) for qid in target_qids
    }

    print(f"Episode: id={episode.episode_id} policy={episode.policy_id} "
          f"steps={len(episode.steps)}  state={args.state}"
          + (f"  queries={target_qids}" if spec.needs_query else ""))

    # --- Run model on every step with full history + a real next step ------
    # For per-query states we collect a separate (T, V) matrix per query,
    # but only run the model once per timestep.
    pred_steps: List[int] = []
    gt_matrix: Dict[Optional[str], List[np.ndarray]] = {qid: [] for qid in target_qids}
    pred_matrix: Dict[Optional[str], List[np.ndarray]] = {qid: [] for qid in target_qids}

    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            sample = dataset[sample_idx]
            end_step = int(sample["step"])
            future_step_index = end_step + 1
            if future_step_index >= len(episode.steps):
                continue
            if int(np.asarray(sample["history_mask"]).sum()) < history_len:
                continue

            batch = _to_batch_tensor(sample, torch)
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            preds = model(batch)
            node_mask_now = np.asarray(sample["node_mask"])[-1]
            future_step = episode.steps[future_step_index]
            pred_steps.append(future_step_index)

            for qid in target_qids:
                qidx = qid_to_qidx[qid]
                pred_per_node = spec.pred_fn(preds, qidx)  # shape (num_nodes,)

                gt_row = np.full(num_nodes, np.nan, dtype=np.float64)
                for vehicle in future_step.candidate_vehicles:
                    slot = vid_to_slot.get(int(vehicle.vehicle_id))
                    if slot is None:
                        continue
                    gt_row[slot] = spec.gt_fn(vehicle, qid)

                pred_row = np.full(num_nodes, np.nan, dtype=np.float64)
                for slot in range(num_nodes):
                    if node_mask_now[slot] > 0.5:
                        pred_row[slot] = float(pred_per_node[slot])

                gt_matrix[qid].append(gt_row)
                pred_matrix[qid].append(pred_row)

    if not pred_steps:
        raise SystemExit("Episode is too short — no step has a full history window AND a next step.")

    pred_steps_arr = np.asarray(pred_steps)
    print(f"Generated predictions for {len(pred_steps_arr)} timesteps "
          f"(steps {pred_steps_arr[0]}..{pred_steps_arr[-1]}).")

    for qid in target_qids:
        gt_arr = np.asarray(gt_matrix[qid])
        pred_arr = np.asarray(pred_matrix[qid])
        _render_query(
            pred_steps_arr=pred_steps_arr,
            gt_arr=gt_arr,
            pred_arr=pred_arr,
            num_nodes=num_nodes,
            slot_to_vid=slot_to_vid,
            spec=spec,
            qid=qid,
            state_name=args.state,
            episode=episode,
            out_dir=out_dir,
            max_vehicles=args.max_vehicles_per_figure,
            make_gif=bool(args.gif),
            gif_fps=float(args.gif_fps),
            gt_smooth_window=int(args.gt_smooth_window),
        )


if __name__ == "__main__":
    main()
