#!/usr/bin/env python
"""Direct test of the cooperation-uncertainty paradox on the NORMALIZED motion uncertainty.

For each recorded scene (``source_window``), builds two counterfactual graphs -- **ego-only** (no
collaborators selected) and **cooperative** (all collaborators selected) -- runs the Stage-1 model on
each, and computes the normalized motion uncertainty two ways:

  * **soft**  : ``U = 1 - exp(-[ Σ_o prob_o·TrΣ_o / (Σ_o prob_o + mass_floor) ] / σ0)``  (notable_prob-weighted)
  * **eq-12** : ``U = 1 - exp(-[ mean_{o∈notable} TrΣ_o ] / σ0)``                          (equal-weight over the
                geometric notable set -- the paper eq(11)-(12))

Then compares ``ΔU = U(coop) - U(ego)`` per scene. The hypothesis: cooperation reveals more (mostly
non-notable, some high-variance) objects, so the **soft** ΔU is >= 0 (paradox -- U rises/flat under
cooperation), while the **eq-12** ΔU is < 0 (cooperation correctly *reduces* the notable-object
uncertainty). Uses only scenes with a non-empty notable set where cooperation actually reveals objects.

Offline (cardreamer_gnn env), no CARLA. Usage::

    python scripts/analyze_wam_coop_uncertainty.py --data-dir data/wam_stage1_chunk_v3 \
        --checkpoint outputs/wam_stage1_chunk_v3/stage1_step2000.pt --device cuda
"""

from __future__ import annotations

import argparse
import glob
import math
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

from car_dreamer.toolkit.wam import WAMPerceptionModel, WAMPolicy, eq12_motion_uncertainty
from car_dreamer.toolkit.wam.graph import GraphBuildSpec
from car_dreamer.toolkit.wam.heads import per_object_trace, policy_uncertainty
from car_dreamer.toolkit.wam.stage1_policy import build_stage1_policy_graph


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = WAMPerceptionModel(ck["perception_config"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def scene_uncertainty(model, ego, collaborators, objects, notable_ids, spec, sigma0, device, modality):
    policy = WAMPolicy(
        selected_vehicle_ids=tuple(int(c.actor_id) for c in collaborators),
        modality_by_vehicle={int(c.actor_id): modality for c in collaborators},
        bandwidth_by_vehicle={int(c.actor_id): 1.0 for c in collaborators},
        frequency_steps=1, reason="diag",
    )
    g = build_stage1_policy_graph(
        ego=ego, collaborators=list(collaborators), objects=list(objects),
        policy=policy, spec=spec, notable_ids=tuple(notable_ids),
    ).to(device)
    out = model([g])
    n_obj = int(out["object_node_ids"].numel())
    if n_obj == 0:
        return None
    soft_raw = float(policy_uncertainty(out["notable_prob"], out["traj_log_var"]))  # mass_floor=0, gate_k=None
    soft_norm = 1.0 - math.exp(-soft_raw / max(sigma0, 1e-6))
    eq12_norm = eq12_motion_uncertainty(out["traj_log_var"], out["object_node_ids"], notable_ids, sigma0=sigma0)
    # imputed: aggregate over the FULL geometric notable set; a notable object NOT present in this graph
    # (unobserved -> blind spot) counts as maximally uncertain (per-object saturated u=1).
    tr = per_object_trace(out["traj_log_var"]).cpu().numpy()
    id2tr = {int(o): float(t) for o, t in zip(out["object_node_ids"].cpu().numpy().tolist(), tr)}
    us = []
    for nid in notable_ids:
        t = id2tr.get(int(nid))
        us.append(1.0 if t is None else (1.0 - math.exp(-t / max(sigma0, 1e-6))))
    imputed_norm = float(sum(us) / len(us)) if us else 0.0
    return soft_norm, eq12_norm, imputed_norm, n_obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="outputs/wam_stage1_chunk_v3/stage1_step2000.pt")
    ap.add_argument("--data-dir", default="data/wam_stage1_chunk_v3")
    ap.add_argument("--out", default="outputs/wam_notable_gating/coop_uncertainty.png")
    ap.add_argument("--sigma0", type=float, default=4.0)
    ap.add_argument("--modality", default="objlist")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--route-waypoints", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    spec = GraphBuildSpec(route_waypoints=args.route_waypoints)
    files = sorted(glob.glob(str(Path(args.data_dir) / "*.pt")))
    if args.limit:
        files = files[: args.limit]

    rows = []  # (ego_soft, coop_soft, ego_eq12, coop_eq12)
    for i, f in enumerate(files):
        d = torch.load(f, weights_only=False)
        sw = d["source_window"][-1]
        notable_ids = tuple(sw.get("notable_ids", ()))
        collabs = tuple(sw["collaborators"])
        objects = tuple(sw["live_states"])
        if not notable_ids or not collabs:
            continue
        # cooperation must actually reveal something ego can't see
        ego_vis = {int(o.actor_id) for o in objects if bool(o.visible_to_ego)}
        collab_ids = {int(c.actor_id) for c in collabs}
        revealed = {int(o.actor_id) for o in objects
                    if (set(int(c) for c in o.visible_to_collaborators) & collab_ids) and not bool(o.visible_to_ego)}
        if not revealed:
            continue
        ego = scene_uncertainty(model, sw["ego"], (), objects, notable_ids, spec, args.sigma0, device, args.modality)
        coop = scene_uncertainty(model, sw["ego"], collabs, objects, notable_ids, spec, args.sigma0, device, args.modality)
        if ego is None or coop is None:
            continue
        rows.append((ego[0], coop[0], ego[1], coop[1], ego[2], coop[2]))
        if (i + 1) % 300 == 0:
            print(f"  ...{i + 1} files, {len(rows)} usable", flush=True)

    if not rows:
        print("no usable scenes (need notable set + cooperation revealing hidden objects)")
        return
    arr = np.array(rows)  # [N,6]: ego/coop x (soft, eq12, imputed)
    d_soft = arr[:, 1] - arr[:, 0]
    d_eq12 = arr[:, 3] - arr[:, 2]
    d_imp = arr[:, 5] - arr[:, 4]
    print(f"\n[N={len(rows)} scenes; cooperation reveals ego-hidden objects]")
    print(f"  soft    U: ego={arr[:,0].mean():.3f} coop={arr[:,1].mean():.3f}  ΔU={d_soft.mean():+.4f}  "
          f"(down in {100*(d_soft<0).mean():.0f}% of scenes)")
    print(f"  eq-12   U: ego={arr[:,2].mean():.3f} coop={arr[:,3].mean():.3f}  ΔU={d_eq12.mean():+.4f}  "
          f"(down in {100*(d_eq12<0).mean():.0f}% of scenes)")
    print(f"  imputed U: ego={arr[:,4].mean():.3f} coop={arr[:,5].mean():.3f}  ΔU={d_imp.mean():+.4f}  "
          f"(down in {100*(d_imp<0).mean():.0f}% of scenes)  <-- hidden notable = max uncertain")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5))
    lo = min(d_soft.min(), d_eq12.min(), d_imp.min())
    hi = max(d_soft.max(), d_eq12.max(), d_imp.max())
    bins = np.linspace(lo, hi, 41)
    a0.hist(d_soft, bins=bins, alpha=0.55, color="#E45756", label=f"soft-gated (present)  ({d_soft.mean():+.3f})")
    a0.hist(d_eq12, bins=bins, alpha=0.55, color="#F58518", label=f"eq-12 notable (present)  ({d_eq12.mean():+.3f})")
    a0.hist(d_imp, bins=bins, alpha=0.55, color="#54A24B", label=f"imputed full-notable  ({d_imp.mean():+.3f})")
    a0.axvline(0, color="black", lw=1)
    a0.set_xlabel("ΔU^mot(norm) = coop − ego-only"); a0.set_ylabel("# scenes")
    a0.set_title("Cooperation effect on normalized motion U\n(<0 = cooperation reduces U, the correct sign)")
    a0.legend(fontsize=8)

    a1.scatter(arr[:, 4], arr[:, 5], s=6, alpha=0.4, color="#54A24B", label="imputed full-notable")
    lim = [0, max(arr[:, 4:6].max(), 1e-3) * 1.05]
    a1.plot(lim, lim, "k--", lw=1, label="coop = ego (no change)")
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("U^mot(norm) ego-only"); a1.set_ylabel("U^mot(norm) cooperative")
    a1.set_title("imputed measure: below diagonal = cooperation helps"); a1.legend(fontsize=9)

    fig.suptitle(f"Cooperation vs normalized motion uncertainty  ({Path(args.checkpoint).name} · {args.data_dir} · N={len(rows)})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
