# EPU × collaboration strategy × communication latency

Model-free study of how **which vehicles share intention** and **how stale that intention is**
affect the ego's perception uncertainty (EPU) in CarDreamer / CARLA 0.9.16, Town03 right-turn,
zero-shot `right_turn_hard.ckpt`. Analysis is offline and does **not** modify `car_dreamer` source.

**Report:** `EPU_latency_collaboration_report.docx` (setup, results, analysis; figures embedded).
Source: `EPU_latency_collaboration_report.html`.

## Headline result
EPU (entropy `U_self` × freshness `trust(k)=exp(-k·0.1/τ)`, τ=1s), simple env, ×10⁻³:

| strategy | k0 | k10 | mean | collision | success | speed |
|---|---|---|---|---|---|---|
| all | 0.0 | 18.9 | 10.2 | 0.00 | 0.93 | 2.97 |
| nearest-2 | 3.1 | 25.9 | 17.3 | 0.06 | 0.86 | 2.91 |
| nearest-1 | 33.6 | 32.7 | 32.9 | 0.23 | 0.68 | 2.28 |
| random-1 | 35.0 | 36.1 | 36.0 | 0.05 | 0.80 | 2.10 |

Strategies that share the key conflict vehicle (all, nearest-2) have low EPU that **rises
monotonically with latency**; strategies that miss it (nearest-1, random-1) stay flat-high. EPU
ordering matches the driving metrics.

## Layout
- `scripts/` — analysis (no CARLA sim; needs a running CARLA/Town03 only for `U_self` map queries):
  - `epu_freshness_table.py` — main metric (freshness `trust(k)`).
  - `epu_entropy_table.py` — entropy `U_self` + old position-residual γ baseline.
  - `ecpg_from_geometry.py` — relevance / geometry helpers.
  - `epu_decompose.py` — shared-vs-unshared EPU decomposition.
  - `driving_metrics_vs_latency.py` — collision/success/speed from eval `metrics.jsonl`.
  - `plot_epu_vs_latency.py`, `figs_for_doc.py` — figures.
- `results/` — `simple_freshness/` (main), `simple_gamma_epu_table.csv` (γ baseline),
  `rich_freshness/` + `entropy_gamma_rich/` (richer-env comparison), `driving_{simple,rich}.csv`.
- `figures/` — `fig1_epu_vs_latency_freshness.png`, `fig2_driving_metrics.png`, `fig3_formula_comparison.png`.

## Reproduce
Recorded geometry (inputs) live in `logdir/ecpg_geom` (simple env, 44 conditions) and
`logdir/group_epu_geom` (rich env). With a CARLA/Town03 up on port 2010:

```bash
python scripts/epu_freshness_table.py --geom logdir/ecpg_geom --port 2010 --tau 1.0 --out results/simple_freshness
python scripts/driving_metrics_vs_latency.py --geom logdir/ecpg_geom --out results/driving_simple.csv
python scripts/figs_for_doc.py
```
