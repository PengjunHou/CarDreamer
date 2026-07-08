# WAM Lyapunov-Guided World-Action Policy Search — Implementation Guide

This document explains the implementation of the V2X paper (`docs/V2X_Paper.pdf`) core contribution:
**action chunks `a=(S,B,D,n)` + a Lyapunov-guided per-frame policy search (P2)** with a world-action rollout
scorer, a risk-aware speed controller, per-link + virtual queues, and event-driven re-planning. It covers
the module map, the paper-equation↔code mapping, the config, how to run every script, and the verification.

**Scope decisions (from the design session):**
- Follow the **paper body text** (Sec I–IV); the figures are from an older draft and are ignored — there is
  **no VLM**. The operational metric is the geometric `U = αU^mot + (1−α)U^cov` (already in the repo).
- **Reuse Stage-1 only** as `U_φ`. The **Stage-2 UWM is NOT used** (its old `π=(S,B,f,d)` flow-matching is
  stale w.r.t. `a=(S,B,D,n)`). Candidate chunks come from a **heuristic enumerator** — the v1 stand-in for the
  Sec III generative proposer `W_θ` (a learnable `W_θ` is a documented future extension).
- **CARLA-free offline + unit tests are the validation.** The online `policy_sampler_mode="lyapunov"` wiring
  is written but **not live-verified** (CARLA may hang).

---

## Module map — `car_dreamer/toolkit/wam/`

| Module | Paper | Responsibility |
|---|---|---|
| `action_chunk.py` | Sec I.B | `SubAction`/`ActionChunk` data model (`a=(S,B,D,n)`, chunk `F(a)=Σn≤F_max`); `local_only_chunk`; `enumerate_candidate_chunks` (heuristic candidate generator); `action_chunk_to_comm_segments` (expand a chunk into installable `CommPolicy` segments). |
| `lyapunov.py` | Sec I.D, IV.A, eq 27/35/42 | `LinkQueue` (`Q_m`), `BudgetVirtualQueue` (`Z`), `LyapunovState`; `action_cost_rate` → `CostRateBreakdown` (the (P2) ratio objective, eq 42). Pure floats. |
| `risk_controller.py` | Sec I.E, eq 29–32 | `select_accel` — risk-aware longitudinal control: speed profile, uncertainty-inflated avoidance radius `r^eff=r_o+κ√TrΣ`, Gaussian risk, `argmax ω_ν ν̄ − ω_r Risk − ω_u|u|`. |
| `rollout_scorer.py` | Sec III | `WorldActionScorer.score_chunk` — rolls the chunk out slot-by-slot, producing the per-slot `Ũ` trajectory (via Stage-1 `U_φ` or a rule fallback) + predicted comm load. |
| `lyapunov_scheduler.py` | Sec IV, Alg 1, eq 8/41 | `select_action_chunk` ((P2) argmin, no-degradation), `ReferenceTrajectory` + `next_epoch_step` (event-driven epoch), `LyapunovScheduler` (Algorithm-1 driver + `run_offline`). |

All are pure/CARLA-free, CPU-runnable, and exported from `car_dreamer.toolkit.wam`.

## Paper equation ↔ code

| Paper | Code |
|---|---|
| `a=(S,B,D,n)` (eq 1), chunk (eq 3–7) | `action_chunk.SubAction` / `ActionChunk` |
| `Q_m(t+1)=max{Q_m−R·Ts,0}+1[m∈S]L` (eq 27) | `lyapunov.LinkQueue.update` |
| `Z(t+1)=max{Z+(ΣB−B̄_bgt),0}` (eq 35) | `lyapunov.BudgetVirtualQueue.update` |
| `U=αU^mot+(1−α)U^cov` (eq 16) | `rollout_scorer.WorldActionScorer.score_chunk` (uses `heads.policy_uncertainty` + `coverage.coverage_metrics`) |
| speed/risk control (eq 29–32) | `risk_controller.select_accel` |
| (P2) ratio program (eq 41), cost-rate (eq 42) | `lyapunov.action_cost_rate`, `lyapunov_scheduler.select_action_chunk` |
| event-driven epoch (eq 8) | `lyapunov_scheduler.next_epoch_step` / `LyapunovScheduler.is_decision_epoch` |
| no-degradation (Sec IV.C) | local-only always in `candidate_chunks` ⇒ `select_action_chunk` argmin ≤ local |
| budget compliance `Z(T)/T→0`, `[O(1/Λ),O(Λ)]` (Prop 1) | `LyapunovState.snapshot`, the analysis scripts |

## Data flow (one decision epoch)

```
C^BS (RolloutContext)
  └─ candidate_chunks: heuristic {member×bandwidth×duration, J≤j_max} + local-only  (feasibility-filtered)
       └─ for each candidate: WorldActionScorer.score_chunk
            per slot τ:  build policy graph → U_φ → U^mot ; coverage → U^cov ; Ũ_τ=αU^mot+(1−α)U^cov
                         risk controller picks u*, advances ego ; accumulate L̂, R̂·Ts (payload units)
       └─ action_cost_rate(Ũ, bandwidth, Z, Q_m, L̂, R̂·Ts) → CostRateBreakdown
  └─ argmin total → install chunk as CommPolicy segments ; store reference trajectory Ũ_{v,t|tk}
between epochs: each slot updates Q_m (eq 27) and Z (eq 35); every Ta compare realized U vs Ũ →
  |U−Ũ|>ε_gap after T_min triggers an early re-plan (eq 8)
```

**Units note (`bit_scale`):** payloads/service/backlog are normalized to *payload units* (one bev payload ≈
1.0) so the net-load term is commensurate with the ratio-scale bandwidth term and the `[0,1]` uncertainty
term — the paper's `δ_max` role. Without it the raw-bit queue term dominates the (P2) objective (see the
problems log P1).

## Config — `env.wam.lyapunov.*` (`car_dreamer/configs/common.yaml`)

Set `env.wam.policy_sampler_mode: lyapunov` and tune the `lyapunov:` block: `lam` (Λ), `c0`,
`budget_bandwidth_ratio` (B̄_bgt), `F_max_slots`, `n_min_slots`, `B_max_ratio`, `bandwidth_grid`,
`duration_grid`, `j_max`, `eps_gap`, `t_min_slots`, `T_a_slots`, `alpha`, `predictor_checkpoint` (Stage-1
`U_φ`; `null` → rule fallback). `U_φ` is taken from the existing `predictor_mode=checkpoint` /
`predictor_checkpoint` when set.

## How to run (CARLA-free; env `cardreamer_gnn`)

**Offline driver (functional):**
```
python scripts/run_wam_lyapunov_offline.py --synthetic --steps 200 --budget 0.4 --out-dir outputs/wam_lyapunov
# with a Stage-1 U_φ checkpoint (recommended for the Λ trade-off curve):
python scripts/run_wam_lyapunov_offline.py --stage1-ckpt <stage1.pt> --synthetic --steps 300 --out-dir outputs/wam_lyapunov
```
Writes `lyapunov_trace.csv` / `.jsonl` (per-slot chunk, `Ũ` vs realized `U`, `Q_m`, `Z`, cost terms) and
`summary.json` (time-avg `U_c`, `B̄`, `Z(T)/T`, mean backlog, epochs/replans). `--bev-size` sets the coverage
raster resolution (coarse = faster).

**Analysis / argument (each reproduces a paper claim; all take `--steps`/`--budget`/`--stage1-ckpt`):**
```
python scripts/analyze_wam_lyapunov_tradeoff.py --steps 150         # Prop 1: utility ~O(1/Λ), backlog ~O(Λ)
python scripts/analyze_wam_lyapunov_budget.py   --steps 300 --budget 0.3   # Z(T)/T→0, B̄≤B̄_bgt, backlog bounded
python scripts/analyze_wam_lyapunov_price.py    --steps 300         # adaptive shadow price Z(t)
python scripts/analyze_wam_cost_rate_decomposition.py --steps 200   # eq-42 cost-rate decomposition
python scripts/compare_wam_lyapunov_vs_local.py --steps 200         # no-degradation (asserts selected ≤ local)
python scripts/ablate_wam_lyapunov.py --steps 120                   # grid Λ×B̄_bgt (or --axis {lam,budget,c0,alpha,f_max,j})
```
Each writes a CSV + PNG under `--out-dir` (default `outputs/wam_lyapunov_analysis`).

**Online (written, not live-verified):** set `env.wam.policy_sampler_mode: lyapunov` on a V2V task. The BS
plans a chunk each decision epoch and installs it as `CommPolicy` segments; the sampler falls back to
local-only on any error. Do not rely on this path until validated against a running CARLA.

## Verification

```
conda run -n cardreamer_gnn python -m unittest discover -s tests -p "test_wam_*.py"   # 178 baseline + new
```
New test modules: `test_wam_action_chunk`, `test_wam_lyapunov`, `test_wam_risk_controller`,
`test_wam_rollout_scorer`, `test_wam_lyapunov_scheduler`, `test_wam_lyapunov_offline_smoke`.
Import guard for the online wiring: `python -c "import car_dreamer.v2v_comm_mixin"`.

## Caveats & future work
- **Λ trade-off needs a Stage-1 checkpoint.** The rule fallback's `U^mot` is *collaborator-blind* (reads only
  `visible_to_ego`), so it cancels out of the (P2) argmin and the Λ utility/backlog curve is muted. Budget
  compliance, no-degradation, and the adaptive price `Z(t)` are all clearly shown regardless (problems log P3).
- **Coverage rasterization** is the offline cost driver; use a coarse `--bev-size` for long runs / sweeps.
- **Learnable `W_θ`** matching `a=(S,B,D,n)` (Sec III) would replace the heuristic enumerator — needs a fresh
  data pipeline over the new action structure; not built here.
- The **online path is unverified** (no CARLA); treat it as scaffolding pending a live run.
