#!/usr/bin/env python
"""Back-to-back ("mirror") plots of variance vs the notable weighting.

Two figures, both with x = per-object motion variance ``TrΣ_o`` (m²):

  Figure A  top (y>0): frequency histogram of TrΣ_o
            bottom (y<0, mirrored): per-bin mean ``notable_prob``, plus the soft-gated
            weight ``w = σ(k·(notable_prob − 0.5))`` for one or two ``gate_k`` values.
            -> "for objects at this variance, how notable does the model think they are,
               and how much weight survives the gate?"

  Figure B  top (y>0): frequency histogram of TrΣ_o
            bottom (y<0, mirrored): per-bin mean of ``notable_prob · TrΣ_o`` (each object's
            contribution to the online motion-uncertainty *numerator*), plus the gated
            version ``w · TrΣ_o`` for comparison.
            -> "which variance range actually drives Σ(w·TrΣ)?"

The bottom panels are aggregated per variance bin (same bins as the histogram) as the mean
over the objects in that bin, then drawn downward (the y-axis is inverted) so it hangs below
the shared x-axis.

Runs a Stage-1 checkpoint over recorded graph windows (no CARLA); caches the per-object
``(TrΣ, notable_prob)`` to an ``.npz`` so re-styling is instant.

Usage (offline, cardreamer_gnn env)::

    conda run -n cardreamer_gnn python scripts/analyze_wam_variance_weight_mirror.py
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


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("perception_config") if isinstance(ckpt, dict) else None
    if cfg is None:
        raise ValueError(f"{ckpt_path} has no 'perception_config'")
    model = WAMPerceptionModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def collect(model, files, device, limit):
    """Flat per-object (TrΣ_o, notable_prob) over all samples."""
    traces, probs = [], []
    for i, f in enumerate(files):
        if limit and i >= limit:
            break
        sample = torch.load(f, weights_only=False)
        out = model([g.to(device) for g in sample["window"]])
        if int(out["object_node_ids"].numel()) == 0:
            continue
        traces.append(per_object_trace(out["traj_log_var"]).cpu().numpy())
        probs.append(out["notable_prob"].cpu().numpy())
        if (i + 1) % 500 == 0:
            print(f"  ...processed {i + 1} files", flush=True)
    if not traces:
        return np.empty(0), np.empty(0)
    return np.concatenate(traces), np.concatenate(probs)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bin_mean(values, idx, counts, nbins):
    s = np.bincount(idx, weights=values, minlength=nbins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / counts
    m[counts == 0] = np.nan
    return m


def _hist_top(ax, centers, counts, width):
    ax.bar(centers, counts, width=width, color="#4C78A8", edgecolor="white", linewidth=0.2)
    ax.set_ylabel("frequency (# objects)")
    ax.margins(x=0.01)
    ax.axhline(0, color="black", lw=0.8)


def _finish(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"saved -> {out_path}")
    plt.close(fig)


def figure_A(centers, counts, width, mean_p, gates, out_path, title):
    fig, (axt, axb) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 7),
        gridspec_kw=dict(height_ratios=[2, 1.4], hspace=0.04),
    )
    _hist_top(axt, centers, counts, width)
    axt.set_title(title, fontsize=11)

    axb.fill_between(centers, mean_p, color="#E45756", alpha=0.30, step="mid")
    axb.plot(centers, mean_p, color="#E45756", lw=1.6, label="mean notable_prob (raw weight)")
    for (k, m), col in zip(gates, ("#54A24B", "#9D755D")):
        axb.plot(centers, m, color=col, lw=1.6, ls="--", label=f"mean gated weight  w=σ({k}·(p−0.5))")
    axb.set_ylim(0, 1.02)
    axb.invert_yaxis()  # hang below the shared x-axis
    axb.set_ylabel("notable_prob / weight\n(below x-axis)")
    axb.set_xlabel("TrΣ_o  (predicted variance, m²)")
    axb.axhline(0, color="black", lw=0.8)
    axb.legend(fontsize=8, loc="lower right")
    _finish(fig, out_path)


def figure_B(centers, counts, width, mean_pT, gate_lines, out_path, title):
    fig, (axt, axb) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 7),
        gridspec_kw=dict(height_ratios=[2, 1.4], hspace=0.04),
    )
    _hist_top(axt, centers, counts, width)
    axt.set_title(title, fontsize=11)

    axb.bar(centers, mean_pT, width=width, color="#B279A2", edgecolor="white", linewidth=0.2,
            label="mean  notable_prob × TrΣ  (raw numerator)")
    stack = [mean_pT[~np.isnan(mean_pT)]]
    for (k, m), col in zip(gate_lines, ("#54A24B", "#E45756", "#9D755D")):
        axb.plot(centers, m, color=col, lw=1.8, ls="--",
                 label=f"mean  w × TrΣ   (gated, k={k})")
        stack.append(m[~np.isnan(m)])
    top = np.nanmax(np.concatenate(stack)) * 1.1
    axb.set_ylim(0, max(top, 1e-6))
    axb.invert_yaxis()
    axb.set_ylabel("notable_prob × TrΣ  (m²)\n(below x-axis)")
    axb.set_xlabel("TrΣ_o  (predicted variance, m²)")
    axb.axhline(0, color="black", lw=0.8)
    axb.legend(fontsize=8, loc="lower right")
    _finish(fig, out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="outputs/wam_stage1/stage1_step1000.pt")
    ap.add_argument("--data-dir", default="data/wam_stage1_policy")
    ap.add_argument("--cache", default="outputs/wam_stage1_uncertainty/variance_probs.npz")
    ap.add_argument("--recompute", action="store_true", help="ignore the cache and re-run the model")
    ap.add_argument("--out-a", default="outputs/wam_stage1_uncertainty/variance_vs_weight.png")
    ap.add_argument("--out-b", default="outputs/wam_stage1_uncertainty/variance_vs_prob_times_var.png")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--gate-k", type=float, default=8.0, help="online default soft-gate sharpness")
    ap.add_argument("--gate-k2", type=float, default=16.0, help="second gate_k to overlay (0 = skip)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cache = Path(args.cache)
    if cache.exists() and not args.recompute:
        d = np.load(cache)
        traces, probs = d["traces"], d["probs"]
        print(f"loaded cache {cache}  ({traces.size} objects)")
    else:
        files = sorted(glob.glob(str(Path(args.data_dir) / "*.pt")))
        if not files:
            raise FileNotFoundError(f"no .pt under {args.data_dir}")
        print(f"running {args.checkpoint} over {len(files)} samples...")
        model = load_model(args.checkpoint, torch.device(args.device))
        traces, probs = collect(model, files, torch.device(args.device), args.limit)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, traces=traces, probs=probs)
        print(f"collected {traces.size} objects -> cached {cache}")

    if traces.size == 0:
        print("nothing collected")
        return

    # shared linear bins over [0, max]
    vmax = float(traces.max())
    edges = np.linspace(0.0, vmax, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1] - edges[0]) * 0.95
    idx = np.clip(np.digitize(traces, edges) - 1, 0, args.bins - 1)
    counts = np.bincount(idx, minlength=args.bins).astype(float)

    mean_p = bin_mean(probs, idx, counts, args.bins)
    w_k = sigmoid(args.gate_k * (probs - 0.5))
    gates = [(args.gate_k, bin_mean(w_k, idx, counts, args.bins))]
    if args.gate_k2 and args.gate_k2 > 0:
        w_k2 = sigmoid(args.gate_k2 * (probs - 0.5))
        gates.append((args.gate_k2, bin_mean(w_k2, idx, counts, args.bins)))

    mean_pT = bin_mean(probs * traces, idx, counts, args.bins)
    gate_lines = [(args.gate_k, bin_mean(w_k * traces, idx, counts, args.bins))]
    if args.gate_k2 and args.gate_k2 > 0:
        gate_lines.append((args.gate_k2, bin_mean(w_k2 * traces, idx, counts, args.bins)))

    tag = f"{Path(args.checkpoint).name} · {args.data_dir} · {traces.size} objects"
    figure_A(centers, counts, width, mean_p, gates, args.out_a,
             title=f"Variance vs notable weight   ({tag})")
    figure_B(centers, counts, width, mean_pT, gate_lines, args.out_b,
             title=f"Variance vs notable_prob × variance   ({tag})")


if __name__ == "__main__":
    main()
