#!/usr/bin/env bash
# Bandwidth sweep for the WAM Phase-3 online comparison table.
#
# The physical link bandwidth (policy_bandwidth_hz) is a pure ONLINE knob: it gates which
# collaborator messages arrive in time (tx/queue delay) -> which objects get injected into the
# ego graph -> what U_phi sees, and it enters the Lyapunov scheduler cost model. It does NOT enter
# Stage-1 training, so we reuse ONE v3 checkpoint for every bandwidth (re-collecting/retraining per
# bandwidth would confound the comparison with a different model).
#
# ego-only (local_only) never communicates, so it is bandwidth-invariant: we run it ONCE and reuse
# it as the shared baseline for every bandwidth row (its 6 MHz run also doubles as an empirical
# invariance sanity vs the existing 20 kHz ego-only ~0.606).
#
# CARLA on this box crashes past ~2300 accumulated steps, and 8 bandwidths x 3 rounds x <=200 steps
# would exceed that in one instance, so we restart CARLA fresh before every bandwidth (each gets its
# own <=600-step budget) and retry once on failure.
set -u

PY=/home/peh324/apps/miniconda3/envs/cardreamer_gnn/bin/python
CKPT=outputs/wam_stage1_chunk_v3/stage1_step2000.pt
OUT=outputs/wam_bandwidth_sweep
ROUNDS=3
STEPS=200
CANDS=60
BEV=24
PORT=2000
CARLA=/home/peh324/carla_simulator/CarlaUE4.sh
RUN_TIMEOUT=5400   # 90 min hard cap per bandwidth run (guards CARLA reset hangs)
# high -> low bandwidth (Hz)
BWS="6000000 3000000 1000000 500000 100000 50000 20000 10000"

mkdir -p "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

carla_up(){ ss -ltn 2>/dev/null | grep -q ":$PORT " ; }

restart_carla(){
  log "restarting CARLA on :$PORT ..."
  pkill -9 -f "CarlaUE4[-]Linux-Shipping" 2>/dev/null
  sleep 6
  nohup "$CARLA" -RenderOffScreen -carla-port=$PORT >/tmp/carla_sweep.log 2>&1 &
  local i
  for i in $(seq 1 90); do sleep 2; carla_up && break; done
  sleep 10  # let the map finish loading
  if carla_up; then log "CARLA up"; else log "CARLA FAILED to come up"; fi
}

run_mode(){  # $1=mode  $2=out-subdir  $3=bandwidth_hz
  local mode="$1" dir="$2" hz="$3"
  if [ -f "$dir/$mode/summary.json" ]; then log "$dir/$mode already done, skip"; return 0; fi
  local attempt
  for attempt in 1 2; do
    restart_carla
    log "=== $mode bw=$hz attempt=$attempt -> $dir ==="
    timeout $RUN_TIMEOUT $PY -u scripts/run_wam_lyapunov_online_episode.py \
      --mode "$mode" --rounds $ROUNDS --max-episode-steps $STEPS \
      --checkpoint "$CKPT" --out-dir "$dir" \
      --env.communication.policy_bandwidth_hz=$hz \
      --env.wam.graph.bev_size=$BEV \
      --env.wam.lyapunov.max_score_candidates=$CANDS \
      >> "$dir.log" 2>&1
    if [ -f "$dir/$mode/summary.json" ]; then log "$mode bw=$hz OK"; return 0; fi
    log "$mode bw=$hz attempt=$attempt FAILED (no summary.json)"
  done
  log "$mode bw=$hz GAVE UP after 2 attempts"
  return 1
}

log "SWEEP START (rounds=$ROUNDS steps<=$STEPS cands=$CANDS)"

# Shared, bandwidth-invariant ego-only baseline (nominal 6 MHz; doubles as invariance sanity).
run_mode local_only "$OUT/ego_only" 6000000

# Lyapunov side: one online run per bandwidth.
for HZ in $BWS; do
  run_mode lyapunov "$OUT/bw_$HZ" "$HZ"
done

log "SWEEP DONE"
