#!/usr/bin/env python
"""Does the notable classifier suppress high-variance NON-notable objects? (soft-gating diagnostic)

Hypothesis (why the soft-gated online motion-uncertainty rises under cooperation): the weight
``w = notable_prob`` fails to zero out objects that have large predicted variance ``TrΣ_o`` but are
**not** ground-truth notable, so their variance leaks into the weighted numerator ``Σ_o w_o·TrΣ_o``.

This runs a Stage-1 checkpoint over recorded windows and, per object, collects ``(TrΣ_o, notable_prob,
gt_notable)`` (GT from the sample's ``perception_labels['notable']``, aligned by object id). Then plots,
per variance bin:

  top    : TrΣ_o histogram, stacked GT-notable vs GT-non-notable;
  middle : mean ``notable_prob`` for GT-notable vs GT-non-notable (+ gated weight for non-notable)
           -> if the non-notable line stays high at high TrΣ, the gate is NOT suppressing them;
  bottom : mean numerator contribution ``notable_prob·TrΣ_o`` split GT-notable vs GT-non-notable
           -> how much of the (inflating) uncertainty numerator comes from non-notable objects.

Offline (cardreamer_gnn env), no CARLA. Usage::

    python scripts/analyze_wam_notable_gating.py --checkpoint outputs/wam_stage1_chunk_v3/stage1_step2000.pt \
        --data-dir data/wam_stage1_policy_eval_v3 --device cuda
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


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WAMPerceptionModel(ck["perception_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def collect(model, files, device, limit):
    traces, probs, gts = [], [], []
    for i, f in enumerate(files):
        if limit and i >= limit:
            break
        s = torch.load(f, weights_only=False)
        out = model([g.to(device) for g in s["window"]])
        ids = out["object_node_ids"]
        if int(ids.numel()) == 0:
            continue
        gt_map = {int(o): float(v) for o, v in
                  zip(s["object_node_ids"], s["perception_labels"]["notable"].tolist())}
        ids_np = ids.cpu().numpy()
        traces.append(per_object_trace(out["traj_log_var"]).cpu().numpy())
        probs.append(out["notable_prob"].cpu().numpy())
        gts.append(np.array([gt_map.get(int(o), 0.0) for o in ids_np], dtype=np.float32))
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1} files", flush=True)
    if not traces:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.concatenate(traces), np.concatenate(probs), np.concatenate(gts)


def bin_mean(values, idx, nbins, counts):
    s = np.bincount(idx, weights=values, minlength=nbins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / counts
    m[counts == 0] = np.nan
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="outputs/wam_stage1_chunk_v3/stage1_step2000.pt")
    ap.add_argument("--data-dir", default="data/wam_stage1_policy_eval_v3")
    ap.add_argument("--cache", default="outputs/wam_notable_gating/tr_prob_gt.npz")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out", default="outputs/wam_notable_gating/notable_gating.png")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--gate-k", type=float, default=8.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cache = Path(args.cache)
    if cache.exists() and not args.recompute:
        d = np.load(cache)
        tr, prob, gt = d["tr"], d["prob"], d["gt"]
        print(f"loaded cache {cache} ({tr.size} objects)")
    else:
        files = sorted(glob.glob(str(Path(args.data_dir) / "*.pt")))
        if not files:
            raise FileNotFoundError(f"no .pt under {args.data_dir}")
        print(f"running {args.checkpoint} over {len(files)} samples (limit={args.limit})...")
        tr, prob, gt = collect(load_model(args.checkpoint, torch.device(args.device)), files,
                               torch.device(args.device), args.limit)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, tr=tr, prob=prob, gt=gt)
        print(f"collected {tr.size} objects -> cached {cache}")

    if tr.size == 0:
        print("nothing collected"); return

    notable = gt > 0.5
    nonnot = ~notable
    print(f"\n[summary] {tr.size} objects, notable={int(notable.sum())} ({100*notable.mean():.1f}%)")
    # high-variance objects: top quartile of TrΣ
    hi = tr >= np.quantile(tr, 0.75)
    print(f"[high-TrΣ top25%] fraction notable={100*notable[hi].mean():.1f}%  "
          f"mean notable_prob: notable={prob[hi & notable].mean() if (hi&notable).any() else float('nan'):.3f} "
          f"non-notable={prob[hi & nonnot].mean() if (hi&nonnot).any() else float('nan'):.3f}")
    num_total = float((prob * tr).sum())
    num_nonnot = float((prob[nonnot] * tr[nonnot]).sum())
    print(f"[numerator Σ(prob·TrΣ)] total={num_total:.0f}  from non-notable={num_nonnot:.0f} "
          f"({100*num_nonnot/max(num_total,1e-9):.1f}%)  <-- leakage")
    wk = 1.0 / (1.0 + np.exp(-args.gate_k * (prob - 0.5)))
    num_g = float((wk * tr).sum()); num_g_nonnot = float((wk[nonnot] * tr[nonnot]).sum())
    print(f"[gated k={args.gate_k} Σ(w·TrΣ)] total={num_g:.0f}  from non-notable={num_g_nonnot:.0f} "
          f"({100*num_g_nonnot/max(num_g,1e-9):.1f}%)")

    vmax = float(tr.max())
    edges = np.linspace(0.0, vmax, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1] - edges[0]) * 0.95
    idx = np.clip(np.digitize(tr, edges) - 1, 0, args.bins - 1)
    counts = np.bincount(idx, minlength=args.bins).astype(float)
    counts_n = np.bincount(idx[notable], minlength=args.bins).astype(float)
    counts_x = np.bincount(idx[nonnot], minlength=args.bins).astype(float)

    p_notable = bin_mean(np.where(notable, prob, 0.0), idx, args.bins, counts_n)
    p_nonnot = bin_mean(np.where(nonnot, prob, 0.0), idx, args.bins, counts_x)
    w_nonnot = bin_mean(np.where(nonnot, wk, 0.0), idx, args.bins, counts_x)
    contrib_notable = bin_mean(np.where(notable, prob * tr, 0.0), idx, args.bins, counts)
    contrib_nonnot = bin_mean(np.where(nonnot, prob * tr, 0.0), idx, args.bins, counts)

    fig, (a0, a1, a2) = plt.subplots(3, 1, sharex=True, figsize=(12, 10),
                                     gridspec_kw=dict(height_ratios=[1.5, 1.2, 1.2], hspace=0.06))
    a0.bar(centers, counts_x, width=width, color="#B279A2", label="GT non-notable", edgecolor="white", lw=0.2)
    a0.bar(centers, counts_n, width=width, bottom=counts_x, color="#4C78A8", label="GT notable", edgecolor="white", lw=0.2)
    a0.set_ylabel("frequency (# objects)"); a0.legend(fontsize=9, loc="upper right")
    a0.set_title(f"Notable-gating diagnostic  ({Path(args.checkpoint).name} · {args.data_dir} · {tr.size} objects)", fontsize=11)

    a1.plot(centers, p_notable, color="#4C78A8", lw=2, label="mean notable_prob | GT notable")
    a1.plot(centers, p_nonnot, color="#E45756", lw=2, label="mean notable_prob | GT NON-notable  ← leak if high")
    a1.plot(centers, w_nonnot, color="#54A24B", lw=1.6, ls="--", label=f"gated w=σ({args.gate_k}(p−.5)) | non-notable")
    a1.set_ylim(0, 1.02); a1.set_ylabel("mean notable_prob"); a1.legend(fontsize=8, loc="upper right"); a1.grid(alpha=0.3)

    a2.plot(centers, contrib_notable, color="#4C78A8", lw=2, label="mean prob·TrΣ | GT notable")
    a2.plot(centers, contrib_nonnot, color="#E45756", lw=2, label="mean prob·TrΣ | GT NON-notable  ← inflation")
    a2.fill_between(centers, contrib_nonnot, color="#E45756", alpha=0.2, step="mid")
    a2.set_ylabel("mean notable_prob × TrΣ (m²)"); a2.set_xlabel("TrΣ_o  (predicted variance, m²)")
    a2.legend(fontsize=8, loc="upper right"); a2.grid(alpha=0.3)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
