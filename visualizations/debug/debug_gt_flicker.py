"""Diagnose VLM-derived flicker by reading the runtime vlm_records JSON.

Unlike emulation_episode_*.json (which only stores aggregated shared_summary_*),
vlm_records_*.json stores raw per-sensor scores including evaluation_mode,
received_window_s, sampling_strategy, etc. This is the right file to inspect.

Usage (from repo root):
    python visualizations/debug/debug_gt_flicker.py \
        --records data/emulation_fixed_20260509/P4/vlm_records_terminated_step_<N>.json \
        --vehicle-id 224 --steps 5:25
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_step_range(spec: str) -> Tuple[int, int]:
    if ":" in spec:
        a, b = spec.split(":", 1)
        return (int(a) if a else 0, int(b) if b else 10**9)
    s = int(spec)
    return s, s + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="Path to vlm_records_*.json")
    parser.add_argument("--vehicle-id", type=int, required=True)
    parser.add_argument("--steps", default="0:30")
    parser.add_argument("--query-id", default=None)
    args = parser.parse_args()

    records: List[Dict[str, Any]] = json.loads(Path(args.records).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("Expected vlm_records JSON to be a list of records.")
    s0, s1 = _parse_step_range(args.steps)

    by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        step = int(rec.get("step", -1))
        if s0 <= step < s1:
            by_step[step].append(rec)

    if not by_step:
        print(f"No records in step range [{s0}, {s1}).")
        return

    # Print one summary header per file.
    sample = next(iter(by_step.values()))[0]
    print(
        f"file: {args.records}\n"
        f"window_s={sample.get('received_window_s')} "
        f"sampling_strategy={sample.get('sampling_strategy')} "
        f"shared_source={sample.get('shared_source')}\n"
        f"vehicle: {args.vehicle_id}   step range: [{s0}, {min(s1, max(by_step)+1)})\n"
    )

    age_weighted = 0
    cached_count = 0
    other_modes: Dict[str, int] = defaultdict(int)
    total_records = 0
    has_aggregated = 0
    flick_pos: Dict[int, float] = {}  # step -> mean(pos) across queries

    for step in sorted(by_step.keys()):
        recs = by_step[step]
        per_query_lines: List[str] = []
        sum_pos, n_q = 0.0, 0
        for rec in recs:
            qid = str(rec.get("question_id", ""))
            if args.query_id and qid != args.query_id:
                continue
            sensors = [
                s
                for s in (rec.get("per_sensor_scores") or [])
                if int(s.get("sender_id", -1)) == int(args.vehicle_id)
                and not bool(s.get("is_ego", False))
            ]
            if not sensors:
                continue
            for s in sensors:
                total_records += 1
                mode = str(s.get("evaluation_mode", ""))
                if "_age_weighted" in mode:
                    age_weighted += 1
                if "cached" in mode:
                    cached_count += 1
                else:
                    other_modes[mode] += 1
                pos = float(s.get("positive_score", float("nan")))
                neg = float(s.get("negative_score", float("nan")))
                ans_v = float(s.get("answerability_score", float("nan")))
                ans_label = s.get("question_answerability", "")
                vis_v = float(s.get("visibility_score", float("nan")))
                vis_label = s.get("visibility_status", "")
                used = bool(s.get("used_for_aggregation", True))
                num_images = int(s.get("num_images", 1))
                desc = str(s.get("scene_description", ""))
                k_hint = max(num_images, desc.count("[latest]") + desc.count("[age="))
                support = str(s.get("support_strength", ""))
                answer = str(s.get("answer", ""))
                per_query_lines.append(
                    f"      q={qid:<28s} answer={answer:<12s} support={support:<8s} "
                    f"pos={pos:.3f} neg={neg:.3f} ans={ans_v:.2f}({ans_label}) "
                    f"vis={vis_v:.2f}({vis_label}) used={used} K~{k_hint}"
                )
                per_query_lines.append(f"        mode={mode}")
            if not args.query_id and sensors:
                sum_pos += float(sensors[0].get("positive_score", 0.0))
                n_q += 1

        gt_pos = (sum_pos / n_q) if n_q else float("nan")
        flick_pos[step] = gt_pos
        print(f"step {step:>3}  GT pos≈mean({n_q} queries)={gt_pos:.3f}")
        for ln in per_query_lines:
            print(ln)
        if not per_query_lines:
            print("      (no per-sender records for this vehicle)")

    # Per-step delta to highlight where it jumps.
    print()
    sorted_steps = sorted(flick_pos.keys())
    big_jumps = []
    for prev, cur in zip(sorted_steps[:-1], sorted_steps[1:]):
        d = abs(flick_pos[cur] - flick_pos[prev])
        if d > 0.2:
            big_jumps.append((prev, cur, flick_pos[prev], flick_pos[cur], d))
    if big_jumps:
        print(f"Big jumps (|Δ| > 0.2) in GT pos≈mean:")
        for prev, cur, p0, p1, d in big_jumps:
            print(f"  step {prev}->{cur}:  {p0:.3f} -> {p1:.3f}  (Δ={d:+.3f})")
    else:
        print("No big jumps (>0.2) in GT pos≈mean within range.")

    # Diagnostics summary.
    print()
    print(f"Total per-sender records scanned: {total_records}")
    print(f"  age_weighted (new path):        {age_weighted}")
    print(f"  cache hits within new path:     {cached_count}")
    if other_modes:
        print(f"  other evaluation_modes:")
        for m, c in sorted(other_modes.items(), key=lambda kv: -kv[1]):
            print(f"    {m:<60s} {c}")

    if total_records and age_weighted == 0:
        print(
            "\n⚠  No records used the new age-weighted path. "
            "Most likely the data was collected before the code change took effect, "
            "or per_sender_unmerged_infos wasn't populated and the run fell back to merge."
        )
    elif total_records and age_weighted < total_records:
        print(
            f"\n⚠  Mixed: only {age_weighted}/{total_records} records used the new path. "
            "Inspect the others' modes."
        )
    elif total_records:
        print("\n✓  All records went through the new age-weighted path.")


if __name__ == "__main__":
    main()
