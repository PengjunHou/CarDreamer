# Visualizations

All offline visualization scripts for CarDreamer live under this directory.
They are organized by **what they read** rather than by file age.

> The deep rendering module
> [`car_dreamer/toolkit/emulation/visualization.py`](../car_dreamer/toolkit/emulation/visualization.py)
> is **not** moved here — it is part of the emulation toolkit and is imported
> by the package's `__init__.py`. The CLI wrapper
> [`emulation/render_topology.py`](emulation/render_topology.py) calls into it.

## Layout

```
visualizations/
├── README.md                    ← this file
├── _common/                     ← shared helpers (no CLI; importable)
│   ├── repo_path.py             — REPO_ROOT / EMULATION_ROOT constants
│   └── emulation_loader.py      — dodge-the-carla-import module loader
├── emulation/                   ← world-model / graph-GRU visualizations
│   ├── render_topology.py
│   ├── plot_state_gt.py
│   └── plot_state_prediction.py
├── vlm/                         ← VLM-records (per-sensor) visualizations
│   └── plot_vlm_records.py
├── policy/                      ← cross-policy comparisons (P1..P8)
│   ├── compare_info_gain.py
│   └── route_b_offline.py
└── debug/                       ← diagnostic CLIs (printed output, not plots)
    └── debug_gt_flicker.py
```

All scripts are runnable from the **repo root** and use the
`cardreamer_gnn` conda environment.

---

## Quick reference — what produces what

| Script | Reads | Produces |
|---|---|---|
| `emulation/render_topology.py` | `emulation_episode_*.json` (+ optional `.pt` checkpoint) | Topology / region / world-region / prediction-comparison **GIFs + PNGs** |
| `emulation/plot_state_gt.py` | `emulation_episode_*.json` | One **PNG + CSV** per state, GT-only line plot per vehicle |
| `emulation/plot_state_prediction.py` | `emulation_episode_*.json` + `.pt` checkpoint | Two-panel **PNG (GT vs prediction)** per state + CSV, optional animated GIF |
| `vlm/plot_vlm_records.py` | `vlm_records_*.json[l]` | Per-question confidence bar / time-series PNGs + per-sensor alignment PNGs + 2 CSVs |
| `policy/compare_info_gain.py` | `<root>/P*/vlm_records_*.json` | Grouped bar chart across P1..P8 + 3 CSVs |
| `policy/route_b_offline.py` | `<data-dir>/P*/emulation_episode_*.json` + matching `vlm_records_*.json` | Per-metric bar PNGs, Pareto plots, confidence/speed trace plots + 2 CSVs |
| `debug/debug_gt_flicker.py` | `vlm_records_*.json` | Stdout diagnostics only (no plots) |

---

## Scripts in detail

### `emulation/render_topology.py`
Thin CLI wrapper around
[`car_dreamer.toolkit.emulation.visualization.main()`](../car_dreamer/toolkit/emulation/visualization.py).
Renders **topology / region / world-region / prediction-comparison** sequences
as GIFs and PNGs from canonical episode JSON.

```bash
# Single episode
python visualizations/emulation/render_topology.py \
    --episode data/emulation_fixed_20260430/P3/emulation_episode_terminated_step_101.json \
    --output-dir logdir/topology_viz

# Batch across policies
python visualizations/emulation/render_topology.py \
    --policy-ids P1 P2 P3 \
    --policy-root data/emulation_fixed_20260430 \
    --output-dir logdir/topology_viz

# With model predictions overlaid
python visualizations/emulation/render_topology.py \
    --episode data/.../emulation_episode_*.json \
    --checkpoint logdir/emulation/fixed_20260430/checkpoint_best.pt \
    --output-dir logdir/topology_viz

# Other views
... --view regions          # ego-centric region overview
... --view world-regions    # world-frame region overview
```

See `--help` for the full flag set (`--queries`, `--step-start/--step-end`,
`--gif-duration-ms`, `--canvas-size`, etc.).

### `emulation/plot_state_gt.py`
Plot **GT-only** time-series of any registered state, without needing a
trained model. Reads `vehicle.shared_summary_semantic[k]` (or
`vehicle.delta_pos`/`delta_vel`/`delta_yaw`/`sender_*`) directly from the
episode JSON.

```bash
# List all available --state names
python visualizations/emulation/plot_state_gt.py --list-states

# Plot mean positive_score over time, one line per vehicle
python visualizations/emulation/plot_state_gt.py \
    --episode data/emulation_fixed_2026059/P3/right_turn_episode_000004.json \
    --state positive_score_mean \
    --out-dir logdir/state_gt_viz

# Per-(vehicle, query) state — needs a --query-id (or "all")
python visualizations/emulation/plot_state_gt.py \
    --episode ... --state sender_collab --query-id clg_front_vehicle
```

### `emulation/plot_state_prediction.py`
Plot per-vehicle **world-model predictions vs GT** for any state. Requires a
trained `.pt` checkpoint. For every step `s` with a full `history_len` window
plus a real `s+1`, runs the model and reads its offset-0 prediction.

```bash
# List available --state names
python visualizations/emulation/plot_state_prediction.py --list-states

# Two-panel PNG (top=GT, bottom=pred) + CSV
python visualizations/emulation/plot_state_prediction.py \
    --checkpoint logdir/emulation/fixed_20260430/checkpoint_best.pt \
    --episode data/emulation_fixed_20260430/P3/emulation_episode_terminated_step_101.json \
    --state complementarity \
    --out-dir logdir/state_pred_viz

# Also emit a growing-line animated GIF
... --gif --gif-fps 4

# Smooth the GT trace with a centered moving-average window
... --gt-smooth-window 7

# Per-(vehicle, query) state
... --state sender_collab --query-id clg_front_vehicle
```

> Note on the semantic state registry: `plot_state_gt.py` maps semantic
> indices 6/7 to **answerability / visibility**, while
> `plot_state_prediction.py` maps them to **max(positive) / max(negative)**.
> This reflects an existing divergence in the source data layout — registries
> are intentionally kept separate.

### `vlm/plot_vlm_records.py`
Visualize a single VLM-records dump (per-sensor scores, alignment, ego-only
vs ego-plus-collaborator confidence). Outputs include:

- `confidence_table.csv`, `sensor_alignment_table.csv`
- `avg_confidence_comparison_by_question.png` — bar chart
- `timeseries_confidence_<qid>.png`, `timeseries_confidence_gain_<qid>.png`
- `timeseries_{facing,region,distance,fov}_alignment_<qid>.png`
- `timeseries_confidence_contribution_<qid>.png`

```bash
python visualizations/vlm/plot_vlm_records.py \
    --input data/emulation_fixed_20260430/P3/vlm_records_terminated_step_101.json \
    --output_dir logdir/vlm_viz
```

### `policy/compare_info_gain.py`
Cross-policy comparison: for each `P*` subdir and each `question_id`,
average `confidence_gain` across all records (ignores the live log).
Produces three CSVs and two bar charts.

```bash
python visualizations/policy/compare_info_gain.py \
    --root data/emulation_fixed_20260430 \
    --out-dir logdir/policy_info_gain

# Restrict to a subset of policies
... --policy-ids P1 P3 P5
```

### `policy/route_b_offline.py`
Offline Route-B metric computation + visualization. Recomputes the runtime
EpisodeMetrics (speed, conflict-distance, headway, violations, bandwidth) and
replays the `ConfidenceTracker` (EMA + slew + clamp). Writes per-episode and
per-policy CSVs plus bar / Pareto / trace PNGs.

```bash
python visualizations/policy/route_b_offline.py \
    --data-dir data/emulation_fixed_20260511 \
    --out-dir data/route_b_analysis

# Tweak the confidence-tracker replay parameters
... --ema-alpha 0.1 --max-delta-per-step 0.03 --c-init 0.5
```

### `debug/debug_gt_flicker.py`
Diagnose VLM-derived GT flicker by printing per-step per-sensor diagnostics
(evaluation mode, age weighting, support strength, big-jump detection).
**Stdout only — no plots.**

```bash
python visualizations/debug/debug_gt_flicker.py \
    --records data/emulation_fixed_20260509/P4/vlm_records_terminated_step_220.json \
    --vehicle-id 224 --steps 5:25
```

---

## Shared helpers (`_common/`)

The three emulation scripts can't do
`from car_dreamer.toolkit.emulation import training` directly — importing
`car_dreamer`'s top-level `__init__` drags in `carla`. Instead they:

1. Insert the repo root onto `sys.path` (so `visualizations._common` resolves).
2. Call `_common.load_*_module()` to manually `exec_module` only the
   emulation submodules they need.

If you add a new script under `emulation/` that also needs the toolkit, do
the same:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visualizations._common import load_training_module        # GT-only
from visualizations._common import load_emulation_modules      # needs model
from visualizations._common import load_visualization_module   # render_topology
```

Scripts under `vlm/`, `policy/`, and `debug/` don't touch the emulation
toolkit, so they don't need any of this — they just read JSON / JSONL and
plot.

---

## Conda environment

All scripts assume `cardreamer_gnn`:

```bash
conda run -n cardreamer_gnn python visualizations/.../<script>.py ...
# or
conda activate cardreamer_gnn
python visualizations/.../<script>.py ...
```

`emulation/render_topology.py` and `emulation/plot_state_prediction.py`
need the full env (numpy, matplotlib, torch, PIL). The others need
numpy + matplotlib (+ pandas for `vlm/plot_vlm_records.py`).
