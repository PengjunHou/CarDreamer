# V2V Communication Latency Model

This document describes the simplified V2V communication model used to compute the
end-to-end member-to-ego transmission latency, lists every configurable parameter
and where to set it, and gives a reference table of latencies under different
combinations of payload size, distance, bandwidth, and number of selected members.

The model is implemented in [`car_dreamer/toolkit/communication/comm.py`](../car_dreamer/toolkit/communication/comm.py) (class `SimpleWirelessLatency`) and consumed by every CARLA env subclass that performs
V2V messaging.

---

## 1. Model

The link latency from a sender to a receiver is computed as a **two-segment** model:

```
latency  =  t_overhead  +  t_tx  +  jitter

t_overhead  =  overhead_base_s  +  overhead_per_kb_s  *  payload_KB
t_tx        =  8 * payload_bytes  /  shannon_bps
shannon_bps =  B_alloc  *  log2(1 + SNR_linear)

SNR_dB      =  Pt_dBm  -  PL(d, fc)  -  N0_dBm(B_alloc)  -  margin_db
PL(d, fc)   =  intercept  +  beta_d * log10(d_m)  +  gamma_fc * log10(fc_GHz)
N0_dBm(B)   =  -174 + 10*log10(B)  +  NF
```

### Intent of each term

| Term | Captures |
|---|---|
| `t_overhead` | Application-layer + MAC-layer overhead that is **insensitive to the channel** (encoding, serialization, decoding, sidelink resource selection). One fixed cost + one size-linear cost. |
| `t_tx`       | Channel-dependent transmission time, driven by SNR and bandwidth. The only term where distance and bandwidth allocation enter. |
| `PL(d, fc)`  | 3GPP TR 37.885 V2V path-loss formula. Selectable via `pathloss_model`. |
| `margin_db`  | A single lumped knob covering shadow fading, vehicle blockage, RF implementation losses. Optionally sampled per call as `N(margin_db, margin_sigma_db^2)` for stochastic shadow fading. |
| `jitter`     | Optional uniform jitter `U(-jitter_s, +jitter_s)` for non-determinism. |

### Path-loss models (3GPP TR 37.885)

`pathloss_model` selects one of:

| Name | `intercept` | `beta_d` | `gamma_fc` | When to use |
|---|---:|---:|---:|---|
| `urban_los`   | 38.77 | 16.7 | 18.2 | Urban LOS / NLOSv (default) |
| `urban_nlos`  | 36.85 | 30.0 | 18.9 | Urban NLOS (building-blocked) |
| `highway_los` | 32.40 | 20.0 | 20.0 | Highway / open-road LOS |

`PL(d, fc) = intercept + beta_d * log10(d_m) + gamma_fc * log10(fc_GHz)` — used directly in the SNR computation above.

### Notes on the model boundaries

- **Bandwidth allocation across N members** is handled **upstream** in the policy:
  `_scale_net_resource_for_bandwidth` in [`right_turn_auto_runtime.py`](../car_dreamer/right_turn_auto_runtime.py) multiplies the configured total `bandwidth_hz` by the policy's per-member `bandwidth` action. The `NetResource` passed into `analyze_transmission` therefore already represents that member's allocated share.
- **Number of members N** thus enters via bandwidth share — no separate `N` parameter exists in the model.
- **Drops**: when `required_load_bps > shannon_bps`, the analysis returns `feasible = False`. The runtime drops the message if `drop_on_capacity_exceeded` is true.
- **What this model omits on purpose**: HARQ retransmissions, MAC-layer queueing, LOS/NLOS state machine, fast fading. Add these only if needed; see §5 for guidance.

---

## 2. Configuration parameters

All parameters live under `communication:` in [`car_dreamer/configs/common.yaml`](../car_dreamer/configs/common.yaml). Defaults shown:

```yaml
communication:
  group_update_period: 20         # how often to recompute groups (steps)
  comm_period: 1                  # how often to trigger intra-group communication (steps)
  bandwidth_hz: 20000000.0        # total shared wireless bandwidth (Hz); policy splits this across selected members
  overhead_base_s: 0.030          # fixed app/MAC overhead per link (s)
  overhead_per_kb_s: 0.0015       # size-linear overhead (s/KB)
  pathloss_model: urban_los       # urban_los | urban_nlos | highway_los
  margin_db: 10.0                 # lumped fading + blockage + implementation margin (dB)
  margin_sigma_db: 0.0            # if > 0, sample margin per call as N(margin_db, sigma^2)
  jitter_s: 0.0                   # uniform jitter ±jitter_s added to latency (s)
  overhead_bytes: 64              # protocol overhead added to payload bytes for tx-time accounting
  drop_on_capacity_exceeded: false
  log_dropped_messages: true
```

### Parameter reference

| Parameter | Type | Default | Where it enters | Effect |
|---|---|---|---|---|
| `bandwidth_hz`         | Hz       | `2e7`   | `B_total`; policy splits per member | Bigger pool → larger per-member share → smaller `t_tx` |
| `overhead_base_s`      | s        | `0.030` | `t_overhead` (constant)            | Sets the latency floor regardless of size or channel |
| `overhead_per_kb_s`    | s/KB     | `0.0015`| `t_overhead` (size-linear)         | Slope of latency vs payload (encode/decode cost) |
| `pathloss_model`       | enum     | `urban_los` | Selects `(intercept, beta_d, gamma_fc)` | Controls how steeply distance hurts SNR |
| `margin_db`            | dB       | `10.0`  | Subtracted from SNR                | Single knob for "real-world losses" — bigger → SNR-limited regimes |
| `margin_sigma_db`      | dB       | `0.0`   | Per-call Gaussian on margin        | > 0 enables stochastic shadow fading |
| `jitter_s`             | s        | `0.0`   | Uniform `±jitter_s` added          | For non-deterministic latency draws |
| `overhead_bytes`       | bytes    | `64`    | Added to `payload_bytes`           | Constant header/footer for tx-time |
| `drop_on_capacity_exceeded` | bool | `false`| Drop if required > shannon         | Hard outage when over-subscribed |
| `log_dropped_messages` | bool     | `true`  | Logging                            | — |

### Per-member RF defaults

These come from `NetResource` defaults in [`comm.py`](../car_dreamer/toolkit/communication/comm.py) (overridable per vehicle, but usually left at defaults):

| Field | Default | Used in |
|---|---:|---|
| `tx_power_dbm`    | `20.0` (= 100 mW)  | `Pt` in SNR |
| `noise_figure_db` | `9.0`              | `NF` in N0 |
| `carrier_freq_hz` | `5.9e9`            | `fc` in path loss |

---

## 3. Reference latency tables

All numbers below use the **default** configuration above (`urban_los`, `margin_db = 10`, `Pt = 20 dBm`, `fc = 5.9 GHz`, `overhead = 30 ms + 1.5 ms/KB`). All values are **milliseconds**.

### Table 1 — payload size × distance, `B_alloc = 10 MHz` (urban LOS)

| d \ payload | 1 KB | 5 KB | 10 KB | 50 KB | 100 KB |
|---|---:|---:|---:|---:|---:|
| **20 m**  | 31.6 | 37.9 | 45.8 | 109.0 | 188.1 |
| **50 m**  | 31.6 | 38.0 | 46.0 | 110.2 | 190.3 |
| **100 m** | 31.6 | 38.2 | 46.3 | 111.5 | 193.1 |
| **200 m** | 31.7 | 38.4 | 46.8 | 113.8 | 197.7 |
| **400 m** | 31.8 | 38.8 | 47.7 | 118.3 | 206.5 |

Under LOS, distance effect is mild (≤ 10 ms across 20 m → 400 m). Latency is dominated by payload size (linear).

### Table 2 — bandwidth × payload, `d = 50 m` (urban LOS)

| B_alloc \ payload | 1 KB | 5 KB | 10 KB | 50 KB | 100 KB |
|---|---:|---:|---:|---:|---:|
| **1 MHz**  | 32.2 | 41.1 | 52.3 | 141.4 | 252.9 |
| **5 MHz**  | 31.7 | 38.4 | 46.8 | 114.2 | 198.4 |
| **10 MHz** | 31.6 | 38.0 | 46.0 | 110.2 | 190.3 |
| **20 MHz** | 31.6 | 37.8 | 45.6 | 108.0 | 185.9 |
| **40 MHz** | 31.5 | 37.7 | 45.3 | 106.7 | 183.4 |

Bandwidth matters mostly when payload is large **and** B is small. At 100 KB, going 1 MHz → 40 MHz saves ~70 ms; at 1 KB the difference is sub-millisecond.

### Table 3 — number of members N (B_total = 20 MHz, equal split, d = 50 m)

Each of N selected members gets `B_alloc = 20 / N` MHz.

| N \ payload | 1 KB | 5 KB | 10 KB | 50 KB | 100 KB |
|---|---:|---:|---:|---:|---:|
| **N=1**  | 31.6 | 37.8 | 45.6 | 108.0 | 185.9 |
| **N=2**  | 31.6 | 38.0 | 46.0 | 110.2 | 190.3 |
| **N=3**  | 31.6 | 38.2 | 46.4 | 112.2 | 194.4 |
| **N=5**  | 31.7 | 38.6 | 47.2 | 116.1 | 202.2 |
| **N=10** | 31.9 | 39.5 | 49.0 | 125.0 | 220.0 |

In LOS, splitting bandwidth across more members has only a mild effect (10× more members → +30 ms at 100 KB) because shannon rate stays well above the required rate.

### Table 4 — distance × N, fixed `payload = 10 KB`, `B_total = 20 MHz`

| d \ N | 1 | 2 | 3 | 5 | 10 |
|---|---:|---:|---:|---:|---:|
| **20 m**  | 45.4 | 45.8 | 46.1 | 46.8 | 48.3 |
| **50 m**  | 45.6 | 46.0 | 46.4 | 47.2 | 49.0 |
| **100 m** | 45.8 | 46.3 | 46.8 | 47.7 | 49.8 |
| **200 m** | 46.1 | 46.8 | 47.4 | 48.5 | 50.9 |
| **400 m** | 46.8 | 47.7 | 48.4 | 49.8 | 52.8 |

Typical perception sharing (~10 KB) under LOS sits in 45–55 ms across all reasonable d, N combinations.

### Table 5 — same sweep but with `pathloss_model = urban_nlos`, `B_alloc = 10 MHz`

| d \ payload | 1 KB | 5 KB | 10 KB | 50 KB | 100 KB |
|---|---:|---:|---:|---:|---:|
| **20 m**  | 31.7 | 38.3 | 46.7 | 113.4 | 196.8 |
| **50 m**  | 32.0 | 40.2 | 50.5 | 132.3 | 234.7 |
| **100 m** | 34.3 | 51.3 | 72.6 | 243.2 | 456.4 |
| **200 m** | 51.7 | 138.5 | 247.0 | **1115.0** | **2199.9** |
| **400 m** | 191.1 | 835.6 | 1641.2 | **8086** | **16142** |

NLOS uses `30·log10(d)`. Past ~200 m the SNR collapses and tx-time explodes — the natural model of "blocked / out-of-range" members.

### Reference SNR / Shannon rate (B = 10 MHz, urban LOS)

| d (m) | SNR (dB) | shannon (Mbps) |
|---:|---:|---:|
| 20  | 30.5 | 101.2 |
| 50  | 23.8 |  79.2 |
| 100 | 18.8 |  62.6 |
| 200 | 13.8 |  46.4 |
| 400 |  8.7 |  30.9 |

---

## 4. Sensitivity summary

| Variable | Effect under default LOS config | Becomes significant when |
|---|---|---|
| **payload size** | **Strong** — dominant linear term | Always; main driver of latency |
| **distance**     | **Weak** (< 10 ms in LOS up to 400 m) | Switching to `urban_nlos`, or beyond ~200 m |
| **bandwidth (per-member)** | **Weak–medium** | Large payload (> 50 KB) **and** small B (< 5 MHz) |
| **N members**    | **Weak** | Combined with NLOS or large payload, where t_tx already matters |

---

## 5. Suggested experiment recipes

| Goal | Suggested override |
|---|---|
| Observe size effect (default) | Sweep `payload_size` ∈ {1, 5, 10, 50, 100} KB; nothing else to change |
| Make distance effect visible  | Set `pathloss_model: urban_nlos`, or raise `margin_db` to 20–25 dB |
| Make bandwidth / N effect visible | Use payload ≥ 50 KB **or** lower `bandwidth_hz` (total pool) to 5 MHz |
| Add stochastic spread for stats / P95 | Set `margin_sigma_db: 3.0` (log-normal shadow fading); optionally `jitter_s: 0.001` |
| Hard outage at the cell edge   | Set `drop_on_capacity_exceeded: true`; combine with `urban_nlos` |

---

## 6. Where things live

| Concern | File / location |
|---|---|
| Model implementation | [`car_dreamer/toolkit/communication/comm.py`](../car_dreamer/toolkit/communication/comm.py) — `SimpleWirelessLatency`, `_PATHLOSS_COEFS`, `NetResource`, `LinkCapacityAnalysis` |
| Default config       | [`car_dreamer/configs/common.yaml`](../car_dreamer/configs/common.yaml) — `communication:` block |
| Bandwidth-per-member scaling | [`car_dreamer/right_turn_auto_runtime.py`](../car_dreamer/right_turn_auto_runtime.py) — `_scale_net_resource_for_bandwidth` |
| Per-link logging     | [`car_dreamer/right_turn_auto_runtime.py`](../car_dreamer/right_turn_auto_runtime.py) — `_record_comm_link_analysis` |
| Tests                | [`tests/test_communication_capacity.py`](../tests/test_communication_capacity.py) |

To change a parameter for a single experiment, override the corresponding key under `communication:` in your run config (the `common.yaml` values act as the global default).
