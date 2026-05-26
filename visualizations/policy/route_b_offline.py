"""Offline Route-B metrics computation and visualization.

Reads per-policy episode dumps in ``<data-dir>/P*/`` and computes the same
metrics the runtime EpisodeMetrics produces, plus a re-applied
ConfidenceTracker trace for visualization. Writes CSV summaries and PNG
figures to ``--out-dir``.

Usage:
    python scripts/route_b_offline_analysis.py \
        --data-dir data/emulation_fixed_20260511 \
        --out-dir data/route_b_analysis

Required fields per dump:
    emulation_episode_*.json: steps[].ego_state.velocity_xy / pose_xy,
                              steps[].candidate_vehicles[*].delta_pos/delta_vel,
                              steps[].candidate_vehicles[*].communication_stats.latest_payload_bytes,
                              dt, policy_id
    vlm_records_*.json:       per record: step, question_id, ego_plus_shared.confidence

The script does NOT require carla; only stdlib + numpy + matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Re-implemented helpers (kept self-contained so the script runs without
# importing the carla-coupled package).
# ---------------------------------------------------------------------------


def smooth_confidence_trace(
    raw_per_step: List[Optional[float]],
    ema_alpha: float = 0.15,
    max_delta_per_step: float = 0.05,
    c_min: float = 0.1,
    c_max: float = 1.0,
    c_init: float = 0.5,
) -> List[float]:
    """Apply the same EMA + slew + clamp logic as ConfidenceTracker."""
    out: List[float] = []
    c_smooth = c_init
    c_out = c_init
    last_raw = c_init
    for raw in raw_per_step:
        if raw is not None and math.isfinite(raw):
            last_raw = float(raw)
        c_smooth = (1.0 - ema_alpha) * c_smooth + ema_alpha * last_raw
        delta = c_smooth - c_out
        if delta > max_delta_per_step:
            c_out += max_delta_per_step
        elif delta < -max_delta_per_step:
            c_out -= max_delta_per_step
        else:
            c_out = c_smooth
        if c_out < c_min:
            c_out = c_min
        elif c_out > c_max:
            c_out = c_max
        out.append(c_out)
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def find_policy_dirs(data_dir: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if not os.path.isdir(full):
            continue
        if not re.match(r"^P\d+$", entry):
            continue
        out.append((entry, full))
    return out


def find_episode_files(policy_dir: str) -> List[Tuple[str, Optional[str]]]:
    """Return list of (episode_json, vlm_records_json or None) per episode."""
    files = sorted(os.listdir(policy_dir))
    episode_files = [f for f in files if f.startswith("emulation_episode_")]
    out: List[Tuple[str, Optional[str]]] = []
    for ep in episode_files:
        # Try to find the matching vlm_records file by step number.
        m = re.search(r"step_(\d+)\.json$", ep)
        step_suffix = m.group(0) if m else None
        vlm_match = None
        if step_suffix:
            for f in files:
                if f.startswith("vlm_records_") and f.endswith(step_suffix):
                    vlm_match = f
                    break
        if vlm_match is None:
            for f in files:
                if f.startswith("vlm_records_") and f.endswith(".json") and "live" not in f:
                    vlm_match = f
                    break
        out.append(
            (
                os.path.join(policy_dir, ep),
                os.path.join(policy_dir, vlm_match) if vlm_match else None,
            )
        )
    return out


def build_confidence_per_step(vlm_records_path: Optional[str]) -> Dict[int, float]:
    """Return mapping step -> mean ego_plus_shared.confidence across questions."""
    if not vlm_records_path or not os.path.exists(vlm_records_path):
        return {}
    with open(vlm_records_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        return {}
    bucket: Dict[int, List[float]] = defaultdict(list)
    for rec in records:
        if not isinstance(rec, dict):
            continue
        step = rec.get("step")
        if step is None:
            continue
        block = rec.get("ego_plus_shared", {})
        if not isinstance(block, dict):
            continue
        c = block.get("confidence")
        try:
            c = float(c)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(c):
            continue
        bucket[int(step)].append(c)
    return {s: float(np.mean(v)) for s, v in bucket.items() if v}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_episode_metrics(
    episode_json_path: str,
    vlm_records_path: Optional[str],
    npc_conflict_max_dist_m: float = 30.0,
    npc_conflict_fov_deg: float = 180.0,
    ema_alpha: float = 0.15,
    max_delta_per_step: float = 0.05,
    c_min: float = 0.1,
    c_max: float = 1.0,
    c_init: float = 0.5,
) -> Dict[str, float]:
    with open(episode_json_path, "r", encoding="utf-8") as f:
        ep = json.load(f)

    dt = float(ep.get("dt", 0.1))
    steps = ep.get("steps", []) or []
    T = len(steps)
    if T == 0:
        return {"num_steps": 0}

    # Speed
    speeds = []
    for s in steps:
        vxy = s.get("ego_state", {}).get("velocity_xy", [0.0, 0.0])
        try:
            vx, vy = float(vxy[0]), float(vxy[1])
        except (TypeError, ValueError, IndexError):
            vx, vy = 0.0, 0.0
        speeds.append(math.hypot(vx, vy))
    speeds_arr = np.asarray(speeds, dtype=np.float64)
    mean_speed = float(speeds_arr.mean())
    std_speed = float(speeds_arr.std())

    accs = np.diff(speeds_arr) / dt
    mean_abs_acc = float(np.mean(np.abs(accs))) if accs.size > 0 else 0.0
    max_abs_acc = float(np.max(np.abs(accs))) if accs.size > 0 else 0.0

    jerks = np.diff(accs) / dt
    mean_abs_jerk = float(np.mean(np.abs(jerks))) if jerks.size > 0 else 0.0
    max_abs_jerk = float(np.max(np.abs(jerks))) if jerks.size > 0 else 0.0

    # Conflict-distance + headway + violation-rate stats.
    # For every NPC in ego's forward FOV that is actively closing in:
    #   - dist                = geometric distance
    #   - headway             = dist / ego_speed (speed-normalized following time)
    #   - dist  < D_THRESH    counts toward "danger-distance" steps
    #   - ttc   < T_THRESH    counts toward "danger-TTC" steps
    # Violation rates use step count as denominator (a step counts at most once
    # even if multiple NPCs trigger).
    half_fov = npc_conflict_fov_deg / 2.0
    DIST_VIOLATION_THRESHOLD = 5.0  # meters
    TTC_VIOLATION_THRESHOLD = 1.5  # seconds
    conflict_distances: List[float] = []
    headway_times: List[float] = []
    dist_violation_steps = 0
    ttc_violation_steps = 0
    for i, s in enumerate(steps):
        ego_speed = speeds[i]
        cands = s.get("candidate_vehicles", []) or []
        step_has_close_npc = False
        step_has_low_ttc = False
        for c in cands:
            try:
                dpx, dpy = float(c["delta_pos"][0]), float(c["delta_pos"][1])
                dvx, dvy = float(c["delta_vel"][0]), float(c["delta_vel"][1])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            dist = math.hypot(dpx, dpy)
            if dist < 1e-6 or dist > npc_conflict_max_dist_m:
                continue
            bearing = math.degrees(math.atan2(dpy, dpx))
            if abs(bearing) > half_fov:
                continue
            n_x, n_y = dpx / dist, dpy / dist
            v_rel_closing = -(dvx * n_x + dvy * n_y)
            if v_rel_closing <= 0.1:
                continue
            conflict_distances.append(dist)
            if dist < DIST_VIOLATION_THRESHOLD:
                step_has_close_npc = True
            ttc = dist / v_rel_closing
            if ttc < TTC_VIOLATION_THRESHOLD:
                step_has_low_ttc = True
            if ego_speed > 0.1:
                headway_times.append(dist / ego_speed)
        if step_has_close_npc:
            dist_violation_steps += 1
        if step_has_low_ttc:
            ttc_violation_steps += 1

    if conflict_distances:
        dist_arr = np.asarray(conflict_distances, dtype=np.float64)
        mean_conflict_distance = float(dist_arr.mean())
        p05_conflict_distance = float(np.percentile(dist_arr, 5))
        p10_conflict_distance = float(np.percentile(dist_arr, 10))
        p25_conflict_distance = float(np.percentile(dist_arr, 25))
        min_conflict_distance = float(dist_arr.min())
    else:
        mean_conflict_distance = -1.0
        p05_conflict_distance = -1.0
        p10_conflict_distance = -1.0
        p25_conflict_distance = -1.0
        min_conflict_distance = -1.0

    if headway_times:
        hw_arr = np.asarray(headway_times, dtype=np.float64)
        mean_headway_s = float(hw_arr.mean())
        p10_headway_s = float(np.percentile(hw_arr, 10))
    else:
        mean_headway_s = -1.0
        p10_headway_s = -1.0

    dist_violation_rate = dist_violation_steps / T
    ttc_violation_rate = ttc_violation_steps / T

    # Bandwidth: sum across steps and senders of latest_payload_bytes.
    total_bytes = 0.0
    for s in steps:
        cands = s.get("candidate_vehicles", []) or []
        for c in cands:
            cs = c.get("communication_stats", {})
            try:
                total_bytes += float(cs.get("latest_payload_bytes", 0.0))
            except (TypeError, ValueError):
                pass
    avg_bytes_per_step = total_bytes / T

    # Confidence trace (replay tracker offline using vlm_records).
    conf_per_step_raw_map = build_confidence_per_step(vlm_records_path)
    conf_raw_seq: List[Optional[float]] = []
    for s in steps:
        st = int(s.get("step", -1))
        conf_raw_seq.append(conf_per_step_raw_map.get(st))
    conf_smoothed = smooth_confidence_trace(
        conf_raw_seq,
        ema_alpha=ema_alpha,
        max_delta_per_step=max_delta_per_step,
        c_min=c_min,
        c_max=c_max,
        c_init=c_init,
    )
    raw_for_mean = [v for v in conf_raw_seq if v is not None]
    mean_confidence_raw = float(np.mean(raw_for_mean)) if raw_for_mean else 0.0
    mean_confidence_smoothed = float(np.mean(conf_smoothed)) if conf_smoothed else 0.0

    completion_time = T * dt  # offline: file ends at termination, take total length
    return {
        "num_steps": float(T),
        "dt": float(dt),
        "mean_speed_mps": mean_speed,
        "mean_speed_kmh": mean_speed * 3.6,
        "speed_std_mps": std_speed,
        "mean_abs_acc_mps2": mean_abs_acc,
        "max_abs_acc_mps2": max_abs_acc,
        "mean_abs_jerk_mps3": mean_abs_jerk,
        "max_abs_jerk_mps3": max_abs_jerk,
        "mean_conflict_distance_m": mean_conflict_distance,
        "p05_conflict_distance_m": p05_conflict_distance,
        "p10_conflict_distance_m": p10_conflict_distance,
        "p25_conflict_distance_m": p25_conflict_distance,
        "min_conflict_distance_m": min_conflict_distance,
        "mean_headway_s": mean_headway_s,
        "p10_headway_s": p10_headway_s,
        "dist_violation_rate": dist_violation_rate,
        "ttc_violation_rate": ttc_violation_rate,
        "total_bandwidth_bytes": total_bytes,
        "avg_bandwidth_bytes_per_step": avg_bytes_per_step,
        "completion_time_s": completion_time,
        "mean_confidence_raw": mean_confidence_raw,
        "mean_confidence_smoothed": mean_confidence_smoothed,
    }, {
        "speeds": speeds,
        "conf_raw": conf_raw_seq,
        "conf_smoothed": conf_smoothed,
        "dt": dt,
    }


# ---------------------------------------------------------------------------
# Aggregation + IO
# ---------------------------------------------------------------------------


def aggregate_by_policy(
    rows: List[Dict[str, float]],
    metric_keys: List[str],
) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    out: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    by_policy: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for r in rows:
        by_policy[r["policy_id"]].append(r)
    for pid, eps in by_policy.items():
        agg: Dict[str, Tuple[float, float, int]] = {}
        for k in metric_keys:
            values = [float(e[k]) for e in eps if k in e]
            if not values:
                agg[k] = (float("nan"), float("nan"), 0)
                continue
            arr = np.asarray(values, dtype=np.float64)
            agg[k] = (float(arr.mean()), float(arr.std()), int(arr.size))
        out[pid] = agg
    return out


def write_csv(path: str, header: List[str], rows: List[List]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_metric_bars(
    agg: Dict[str, Dict[str, Tuple[float, float, int]]],
    metric_keys: List[str],
    out_dir: str,
) -> List[str]:
    """Write one PNG per metric into ``out_dir``. Returns list of paths."""
    os.makedirs(out_dir, exist_ok=True)
    policies = sorted(agg.keys(), key=lambda p: int(p[1:]) if p[1:].isdigit() else 0)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(policies))]
    saved: List[str] = []
    for k in metric_keys:
        means = [agg[p][k][0] for p in policies]
        stds = [agg[p][k][1] for p in policies]
        x = np.arange(len(policies))
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        for rect, mean in zip(bars, means):
            ax.annotate(
                f"{mean:.3g}",
                xy=(rect.get_x() + rect.get_width() / 2, mean),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(policies)
        ax.set_xlabel("policy", fontsize=14)
        if k == "mean_speed_kmh":
            ax.set_ylabel("Mean Speed (km/h)", fontsize=14)
        elif k == "completion_time_s":
            ax.set_ylabel("completion time (s)", fontsize=14)
        elif k == "mean_conflict_distance_m":
            ax.set_ylabel("Mean Conflict Distance (m)", fontsize=14)
        elif k == "p05_conflict_distance_m":
            ax.set_ylabel("P5 Conflict Distance (m)", fontsize=14)
        elif k == "p10_conflict_distance_m":
            ax.set_ylabel("P10 Conflict Distance (m)", fontsize=14)
        elif k == "p25_conflict_distance_m":
            ax.set_ylabel("P25 Conflict Distance (m)", fontsize=14)
        elif k == "min_conflict_distance_m":
            ax.set_ylabel("Min Conflict Distance (m)", fontsize=14)
        elif k == "mean_headway_s":
            ax.set_ylabel("Mean Headway (s)", fontsize=14)
        elif k == "p10_headway_s":
            ax.set_ylabel("P10 Headway (s)", fontsize=14)
        elif k == "dist_violation_rate":
            ax.set_ylabel("Distance Violation Rate", fontsize=14)
        elif k == "ttc_violation_rate":
            ax.set_ylabel("TTC Violation Rate", fontsize=14)
        elif k == "total_bandwidth_bytes":
            ax.set_ylabel("total bandwidth (bytes)", fontsize=14)
        elif k == "mean_confidence_raw":
            ax.set_ylabel("Mean Semantic Confidence", fontsize=14)
        elif k == "mean_confidence_smoothed":
            ax.set_ylabel("Mean Semantic Confidence", fontsize=14)
        # ax.set_ylabel(k, fontsize=14)
        # ax.set_title(k, fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"bar_{k}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        saved.append(out_path)
    return saved


def plot_pareto(
    agg: Dict[str, Dict[str, Tuple[float, float, int]]],
    x_key: str,
    y_key: str,
    out_path: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    log_x: bool = True,
) -> None:
    policies = sorted(agg.keys(), key=lambda p: int(p[1:]) if p[1:].isdigit() else 0)
    xs = [agg[p][x_key][0] for p in policies]
    ys = [agg[p][y_key][0] for p in policies]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(xs, ys, s=80, color="darkorange")
    for p, x, y in zip(policies, xs, ys):
        ax.annotate(p, (x, y), textcoords="offset points", xytext=(6, 4))
    if log_x:
        ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel(x_label or x_key)
    ax.set_ylabel(y_label or y_key)
    ax.set_title(f"{y_key}  vs  {x_key}")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confidence_traces(
    traces_by_policy: Dict[str, List[List[float]]],
    out_path: str,
) -> None:
    policies = sorted(
        traces_by_policy.keys(), key=lambda p: int(p[1:]) if p[1:].isdigit() else 0
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, p in enumerate(policies):
        traces = traces_by_policy[p]
        if not traces:
            continue
        # average trace length: pad/truncate to first trace's length (1 ep per policy here)
        first = traces[0]
        ax.plot(first, color=cmap(i % 10), label=p, linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("c_smoothed")
    ax.set_title("Smoothed confidence trace per policy")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_speed_traces(
    speed_by_policy: Dict[str, List[List[float]]],
    out_path: str,
) -> None:
    policies = sorted(
        speed_by_policy.keys(), key=lambda p: int(p[1:]) if p[1:].isdigit() else 0
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, p in enumerate(policies):
        traces = speed_by_policy[p]
        if not traces:
            continue
        first = np.asarray(traces[0]) * 3.6  # km/h
        ax.plot(first, color=cmap(i % 10), label=p, linewidth=1.2)
    ax.set_xlabel("step", fontsize=14)
    ax.set_ylabel("speed (km/h)", fontsize=14)
    ax.set_title("Ego speed trace per policy", fontsize=14)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="root with P*/ subdirs")
    parser.add_argument(
        "--out-dir", required=True, help="output dir for csv + figures"
    )
    parser.add_argument("--ema-alpha", type=float, default=0.15)
    parser.add_argument("--max-delta-per-step", type=float, default=0.05)
    parser.add_argument("--c-min", type=float, default=0.1)
    parser.add_argument("--c-max", type=float, default=1.0)
    parser.add_argument("--c-init", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"data-dir not found: {args.data_dir}", file=sys.stderr)
        return 2
    os.makedirs(args.out_dir, exist_ok=True)

    metric_keys = [
        "mean_speed_kmh",
        "speed_std_mps",
        "completion_time_s",
        "mean_abs_acc_mps2",
        "max_abs_acc_mps2",
        "mean_abs_jerk_mps3",
        "max_abs_jerk_mps3",
        "mean_conflict_distance_m",
        "p05_conflict_distance_m",
        "p10_conflict_distance_m",
        "p25_conflict_distance_m",
        "min_conflict_distance_m",
        "mean_headway_s",
        "p10_headway_s",
        "dist_violation_rate",
        "ttc_violation_rate",
        "total_bandwidth_bytes",
        "avg_bandwidth_bytes_per_step",
        "mean_confidence_raw",
        "mean_confidence_smoothed",
    ]

    rows: List[Dict[str, float]] = []
    speed_traces: Dict[str, List[List[float]]] = defaultdict(list)
    conf_traces: Dict[str, List[List[float]]] = defaultdict(list)

    for pid, pdir in find_policy_dirs(args.data_dir):
        episodes = find_episode_files(pdir)
        if not episodes:
            print(f"  [{pid}] no episode files; skipping")
            continue
        for ep_path, vlm_path in episodes:
            try:
                m, traces = compute_episode_metrics(
                    ep_path,
                    vlm_path,
                    ema_alpha=args.ema_alpha,
                    max_delta_per_step=args.max_delta_per_step,
                    c_min=args.c_min,
                    c_max=args.c_max,
                    c_init=args.c_init,
                )
            except Exception as exc:
                print(f"  [{pid}] FAILED on {os.path.basename(ep_path)}: {exc}")
                continue
            row = {"policy_id": pid, "episode_file": os.path.basename(ep_path), **m}
            rows.append(row)
            speed_traces[pid].append(traces["speeds"])
            conf_traces[pid].append(traces["conf_smoothed"])
            print(
                f"  [{pid}] {os.path.basename(ep_path)} "
                f"steps={int(m.get('num_steps',0))} "
                f"speed={m.get('mean_speed_kmh',0):.2f}km/h "
                f"conf={m.get('mean_confidence_raw',0):.3f} "
                f"bw={m.get('total_bandwidth_bytes',0):.0f}B "
                f"mean_dist={m.get('mean_conflict_distance_m',-1):.2f}m "
                f"p10_hw={m.get('p10_headway_s',-1):.2f}s "
                f"ttc_viol={m.get('ttc_violation_rate',0):.2%}"
            )

    if not rows:
        print("no episodes processed; abort", file=sys.stderr)
        return 1

    # Per-episode CSV
    per_ep_path = os.path.join(args.out_dir, "metrics_per_episode.csv")
    header = ["policy_id", "episode_file"] + metric_keys
    write_csv(
        per_ep_path,
        header,
        [[r["policy_id"], r["episode_file"]] + [r.get(k, "") for k in metric_keys] for r in rows],
    )

    # Per-policy aggregate
    agg = aggregate_by_policy(rows, metric_keys)
    by_policy_path = os.path.join(args.out_dir, "metrics_by_policy.csv")
    by_policy_header = ["policy_id"] + sum(
        [[f"{k}_mean", f"{k}_std", f"{k}_n"] for k in metric_keys], []
    )
    by_policy_rows = []
    for p in sorted(agg.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        row = [p]
        for k in metric_keys:
            mean, std, n = agg[p][k]
            row += [f"{mean:.6g}", f"{std:.6g}", n]
        by_policy_rows.append(row)
    write_csv(by_policy_path, by_policy_header, by_policy_rows)

    # Plots — one PNG per metric under bars/
    saved_bar_paths = plot_metric_bars(
        agg,
        metric_keys,
        os.path.join(args.out_dir, "bars"),
    )
    for p in saved_bar_paths:
        print(f"  bar : {p}")
    plot_pareto(
        agg,
        x_key="total_bandwidth_bytes",
        y_key="mean_speed_kmh",
        out_path=os.path.join(args.out_dir, "pareto_bw_speed.png"),
        x_label="total bandwidth (bytes, log)",
        y_label="mean speed (km/h)",
    )
    plot_pareto(
        agg,
        x_key="total_bandwidth_bytes",
        y_key="completion_time_s",
        out_path=os.path.join(args.out_dir, "pareto_bw_completion.png"),
        x_label="total bandwidth (bytes, log)",
        y_label="completion time (s)",
    )
    plot_confidence_traces(
        conf_traces, os.path.join(args.out_dir, "confidence_trace.png")
    )
    plot_speed_traces(speed_traces, os.path.join(args.out_dir, "speed_trace.png"))

    print()
    print(f"per-episode csv : {per_ep_path}")
    print(f"per-policy csv  : {by_policy_path}")
    print(f"figures saved to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
