#!/usr/bin/env python3
"""Phase-1 Stage-2 inference verification (CARLA-free).

Loads a trained UWM checkpoint and, on recorded flow samples, checks:
  * policy proposal: ``propose_policies`` vs the recorded GT policy chunk (slot-level selection match +
    L2 over active members/steps);
  * future-BEV prediction: ``rollout_future_bev`` (given the GT policy) -> ``decode_bev`` -> IoU /
    reconstruction loss vs the recorded ``bev_future``, compared against a copy-last-history baseline.

Recorded samples do NOT store candidate vehicle ids, so policy proposals are checked at the member-slot
level (not named to vehicles); that is sufficient to judge "are proposals reasonable + is BEV accurate".

Example:
    python scripts/eval_wam_stage2_inference.py \
        --checkpoint outputs/wam_stage2_multi/stage2_step2000.pt --data-dir data/wam_flow_multi --limit 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify a trained Stage-2 UWM by inference on recorded flow samples.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-dir", type=Path, default=Path("data/wam_flow_multi"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="max samples to evaluate")
    p.add_argument("--n-candidates", type=int, default=4)
    p.add_argument("--sel-threshold", type=float, default=0.5)
    p.add_argument("--n-steps", type=int, default=None, help="ODE integration steps (default = config)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import functools

    import torch

    from car_dreamer.toolkit.wam import (
        WAMFlowDataset,
        bev_iou,
        bev_reconstruction_loss,
        collate_flow_samples,
        decode_policy_vector,
        load_wam_uwm,
    )

    device = torch.device(args.device)
    model = load_wam_uwm(args.checkpoint, device=device)
    cfg = model.flow.config
    enable_bev = bool(cfg.enable_bev)
    print(f"[eval] loaded UWM hidden={cfg.hidden_dim} horizon={cfg.horizon} max_members={cfg.max_members} "
          f"num_formats={cfg.num_formats} enable_bev={enable_bev}", flush=True)

    ds = WAMFlowDataset(args.data_dir)
    n_total = len(ds) if args.limit is None else min(len(ds), int(args.limit))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, list(range(n_total))),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=functools.partial(collate_flow_samples, flow_config=cfg),
    )

    # accumulators
    n_samples = 0
    sel_match_num = 0.0; sel_active = 0.0
    pol_l2_sum = 0.0; pol_l2_cnt = 0.0
    iou_sum = 0.0; recon_sum = 0.0; iou_base_sum = 0.0; bev_batches = 0
    examples = []

    with torch.no_grad():
        for batch in loader:
            samples = [
                {"vehicle_graphs": [g.to(device) for g in s["vehicle_graphs"]],
                 "request_index": int(s.get("request_index", 0)),
                 "notable_object_ids": s.get("notable_object_ids", ())}
                for s in batch["samples"]
            ]
            bev_hist = batch["bev_history"].to(device).float()
            if enable_bev:
                bsz, kp1 = bev_hist.shape[0], bev_hist.shape[1]
                hist_lat = model.encode_bev(bev_hist.reshape(bsz * kp1, *bev_hist.shape[2:])).reshape(bsz, kp1, -1)
                for i, s in enumerate(samples):
                    s["bev_history"] = hist_lat[i]
            cond, tids, mask = model.condition_tokens_batch(samples)

            member_mask = batch["member_mask"].to(device)          # [B,M]
            policy_step_mask = batch["policy_step_mask"].to(device)  # [B,H]
            bev_step_mask = batch["bev_step_mask"].to(device)        # [B,H]
            gt_policy = batch["policy_chunk"].to(device)            # [B,H,M,P]
            B = gt_policy.shape[0]; n_samples += B

            # ---- policy proposal vs GT (slot-level) ----
            cand = model.flow.propose_policies(
                cond, tids, mask, n_candidates=int(args.n_candidates),
                member_mask=member_mask, policy_step_mask=policy_step_mask, n_steps=args.n_steps,
            )  # [B, K, H, M, P]
            prop0 = cand[:, 0]                                       # [B,H,M,P] first candidate
            active = (policy_step_mask.unsqueeze(-1) * member_mask.unsqueeze(1))  # [B,H,M]
            gt_sel = (gt_policy[..., 0] > 0.5).float()              # GT selection (0/1)
            pred_sel = (torch.sigmoid(prop0[..., 0]) > float(args.sel_threshold)).float()
            sel_match_num += float((((pred_sel == gt_sel).float()) * active).sum())
            sel_active += float(active.sum())
            # L2 over the full per-member policy vector on active slots
            diff2 = ((prop0 - gt_policy) ** 2).sum(-1) * active     # [B,H,M]
            pol_l2_sum += float(diff2.sum()); pol_l2_cnt += float(active.sum())

            # ---- future BEV prediction (given GT policy) ----
            if enable_bev:
                bev_pred = model.flow.rollout_future_bev(
                    cond, tids, mask, gt_policy,
                    member_mask=member_mask, policy_step_mask=policy_step_mask,
                    bev_step_mask=bev_step_mask, n_steps=args.n_steps,
                )  # [B,H,Dz]
                Bx, H = bev_pred.shape[0], bev_pred.shape[1]
                logits = model.decode_bev(bev_pred.reshape(Bx * H, -1))          # [B*H,C,S,S]
                gt_fut = batch["bev_future"].to(device).float().reshape(Bx * H, *batch["bev_future"].shape[2:])
                m = bev_step_mask.reshape(Bx * H)
                iou_sum += float(bev_iou(torch.sigmoid(logits), gt_fut)) * Bx     # bev_iou wants probabilities
                recon_sum += float(bev_reconstruction_loss(logits, gt_fut, mask=m)) * Bx  # BCE-with-logits
                # baseline: predict the last history frame's occupancy (0/1) for every future step
                last_hist = batch["bev_history"][:, -1].to(device).float()        # [B,C,S,S] (0/1 occupancy)
                base = last_hist.unsqueeze(1).expand(-1, H, -1, -1, -1).reshape(Bx * H, *last_hist.shape[1:])
                iou_base_sum += float(bev_iou(base, gt_fut)) * Bx
                bev_batches += Bx

            if len(examples) < 3:
                dec = decode_policy_vector(prop0[0, 0].cpu(), num_formats=cfg.num_formats)
                examples.append({
                    "active_members": int(member_mask[0].sum()),
                    "gt_sel_step0": [int(x) for x in gt_sel[0, 0].cpu().tolist()],
                    "pred_sel_step0": [int(x) for x in pred_sel[0, 0].cpu().tolist()],
                    "pred_bw_step0": [round(float(x), 2) for x in dec["bw"].tolist()],
                })

    print(f"\n[eval] samples={n_samples}")
    print(f"  policy selection match (active slots): {100.0 * sel_match_num / max(sel_active,1):.1f}%")
    print(f"  policy chunk L2 / active slot:         {pol_l2_sum / max(pol_l2_cnt,1):.4f}")
    if bev_batches:
        print(f"  future-BEV IoU (model vs GT):          {iou_sum / bev_batches:.4f}")
        print(f"  future-BEV IoU (copy-last baseline):   {iou_base_sum / bev_batches:.4f}")
        print(f"  future-BEV reconstruction loss:        {recon_sum / bev_batches:.4f}")
    print("\n[eval] examples (step 0, first sample of first 3 batches):")
    for e in examples:
        print("  ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
