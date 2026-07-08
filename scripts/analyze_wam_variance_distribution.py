#!/usr/bin/env python
"""Offline histogram of the per-object predicted motion variance ``TrΣ_o``.

This is exactly the quantity the online WAM uncertainty sums over objects
(``v2v_comm_mixin._predict_wam_with_checkpoint``)::

    trace       = exp(traj_log_var).sum(-1)   # [Q, H]  σ²_x + σ²_y per future step
    TrΣ_o       = trace.mean(-1)              # [Q]     mean over the horizon (m²)
    motion_unc  = (w * TrΣ).sum() / (w.sum() + mass_floor)   # w = notable_prob (or soft-gate)

The concern: a few objects get a huge ``TrΣ_o`` (the head saturates toward the
``log_var_max`` clamp), and their ``notable_prob`` weight is not small enough to cancel
them in the weighted average. This script reproduces that variance distribution **without
CARLA** -- it runs a Stage-1 checkpoint over recorded graph windows (the same windows the
online predictor builds), so the distribution matches the live env -- and plots it so you
can pick a discard threshold.

Usage (offline, cardreamer_gnn env)::

    conda run -n cardreamer_gnn python scripts/analyze_wam_variance_distribution.py \
        --checkpoint outputs/wam_stage1/stage1_step1000.pt \
        --data-dir data/wam_stage1_policy \
        --out outputs/wam_stage1_uncertainty/variance_distribution.png
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from car_dreamer.toolkit.wam import WAMPerceptionModel
from car_dreamer.toolkit.wam.heads import per_object_trace


def load_model(ckpt_path: str, device: torch.device) -> WAMPerceptionModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("perception_config") if isinstance(ckpt, dict) else None
    if cfg is None:
        raise ValueError(f"{ckpt_path} has no 'perception_config'; cannot rebuild the model")
    model = WAMPerceptionModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def collect(model, files, device, limit):
    """Return (traces, notable_probs): flat per-object TrΣ_o and the model's notable_prob."""
    traces, probs = [], []
    n_samples = 0
    for i, f in enumerate(files):
        if limit and i >= limit:
            break
        sample = torch.load(f, weights_only=False)
        window = [g.to(device) for g in sample["window"]]
        out = model(window)
        if int(out["object_node_ids"].numel()) == 0:
            continue
        traces.append(per_object_trace(out["traj_log_var"]).cpu().numpy())
        probs.append(out["notable_prob"].cpu().numpy())
        n_samples += 1
        if (i + 1) % 500 == 0:
            print(f"  ...processed {i + 1} files", flush=True)
    if not traces:
        return np.empty(0), np.empty(0), 0
    return np.concatenate(traces), np.concatenate(probs), n_samples


def print_stats(traces: np.ndarray, probs: np.ndarray, log_var_max: float | None) -> None:
    n = traces.size
    total = float(traces.sum())
    print("\n================ per-object TrΣ_o (m²) ================")
    print(f"objects          : {n}")
    print(f"min / mean / max : {traces.min():.4f} / {traces.mean():.3f} / {traces.max():.1f}")
    print(f"std              : {traces.std():.3f}")
    if log_var_max is not None:
        sat = float(np.exp(log_var_max)) * 2.0  # both coords at the clamp
        print(f"clamp ceiling    : TrΣ_max ≈ 2·exp(log_var_max={log_var_max:g}) = {sat:.0f} m²"
              f"   (objects within 1% of it: {(traces >= 0.99 * sat).mean() * 100:.2f}%)")
    print("\npercentiles (TrΣ_o):")
    for p in (50, 75, 90, 95, 99, 99.9):
        print(f"  p{p:<5} = {np.percentile(traces, p):10.3f} m²")

    print("\ntail concentration (sorted by TrΣ desc):")
    print("   top-k% objects |  TrΣ threshold |  share of total Σ-mass")
    order = np.sort(traces)[::-1]
    csum = np.cumsum(order)
    for k in (1, 2, 5, 10, 25):
        m = max(1, int(round(k / 100.0 * n)))
        thr = order[m - 1]
        mass = csum[m - 1] / total * 100.0
        print(f"   {k:>5}%         | >= {thr:10.1f}  |  {mass:6.1f}%")

    # The actual concern, as text: are the big-variance objects un-suppressed by notable_prob?
    if probs.size == traces.size and traces.size:
        hi = traces >= np.percentile(traces, 99)
        print("\nnotable_prob of the top-1% highest-variance objects:")
        print(f"  median notable_prob = {np.median(probs[hi]):.3f}   mean = {probs[hi].mean():.3f}"
              f"   (overall median = {np.median(probs):.3f})")
        print("  -> if this is not ~0, notable_prob does NOT cancel them in the weighted sum.")
    print("======================================================\n")


def plot(traces: np.ndarray, out_path: Path, bins: int, title: str) -> None:
    p90, p95, p99 = (np.percentile(traces, q) for q in (90, 95, 99))
    total = float(traces.sum())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) linear histogram -- the bulk near 0 + the thin heavy tail.
    ax = axes[0]
    ax.hist(traces, bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.3)
    for v, c, lab in ((p90, "#F58518", "p90"), (p95, "#E45756", "p95"), (p99, "#B279A2", "p99")):
        ax.axvline(v, color=c, ls="--", lw=1.4, label=f"{lab}={v:.0f}")
    ax.set_xlabel("TrΣ_o  (predicted variance, m²)")
    ax.set_ylabel("frequency (# objects)")
    ax.set_title("linear")
    ax.legend(fontsize=8)

    # (2) log-x histogram -- makes the heavy tail visible.
    ax = axes[1]
    pos = traces[traces > 0]
    lo = max(pos.min(), 1e-4)
    log_bins = np.logspace(np.log10(lo), np.log10(traces.max() + 1e-9), bins)
    ax.hist(traces, bins=log_bins, color="#54A24B", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    for v, c, lab in ((p90, "#F58518", "p90"), (p95, "#E45756", "p95"), (p99, "#B279A2", "p99")):
        ax.axvline(v, color=c, ls="--", lw=1.4, label=f"{lab}={v:.0f}")
    ax.set_xlabel("TrΣ_o  (m², log scale)")
    ax.set_ylabel("frequency (# objects)")
    ax.set_title("log-x")
    ax.legend(fontsize=8)

    # (3) cumulative: fraction of objects vs fraction of total Σ-mass below a threshold.
    ax = axes[2]
    s = np.sort(traces)
    frac_obj = np.arange(1, s.size + 1) / s.size
    frac_mass = np.cumsum(s) / total
    ax.plot(s, frac_obj, color="#4C78A8", label="fraction of objects ≤ x")
    ax.plot(s, frac_mass, color="#E45756", label="fraction of Σ-mass ≤ x")
    ax.set_xscale("log")
    for v, c, lab in ((p95, "#E45756", "p95"), (p99, "#B279A2", "p99")):
        ax.axvline(v, color=c, ls="--", lw=1.0)
    ax.set_xlabel("discard-threshold TrΣ_o  (m², log scale)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("CDF: pick threshold → coverage")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="center right")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"saved figure -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="outputs/wam_stage1/stage1_step1000.pt")
    ap.add_argument("--data-dir", default="data/wam_stage1_policy")
    ap.add_argument("--out", default="outputs/wam_stage1_uncertainty/variance_distribution.png")
    ap.add_argument("--limit", type=int, default=0, help="max #samples to process (0 = all)")
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-npy", default="", help="optional path to dump the flat TrΣ array (.npy)")
    args = ap.parse_args()

    device = torch.device(args.device)
    files = sorted(glob.glob(str(Path(args.data_dir) / "*.pt")))
    if not files:
        raise FileNotFoundError(f"no .pt under {args.data_dir}")
    print(f"checkpoint : {args.checkpoint}")
    print(f"data-dir   : {args.data_dir}  ({len(files)} files"
          f"{', limited to ' + str(args.limit) if args.limit else ''})")

    model, cfg = load_model(args.checkpoint, device)
    traces, probs, n_samples = collect(model, files, device, args.limit)
    if traces.size == 0:
        print("no objects collected; nothing to plot")
        return
    print(f"collected {traces.size} objects over {n_samples} non-empty samples")

    if args.save_npy:
        np.save(args.save_npy, traces)
        print(f"saved raw traces -> {args.save_npy}")

    print_stats(traces, probs, getattr(cfg, "log_var_max", None))
    plot(traces, Path(args.out), args.bins,
         title=f"WAM per-object motion variance  ·  {Path(args.checkpoint).name}  ·  {args.data_dir}")


if __name__ == "__main__":
    main()
