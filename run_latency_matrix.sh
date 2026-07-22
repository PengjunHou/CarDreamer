#!/bin/bash
# Intention-channel latency experiment (plan v2): full-observability background
# vehicles (hard.ckpt, in-distribution), only the intention lines go through
# designated selection + communication latency. 41 cells; yesterday's v1 cells
# are seeded as the k=0 row. Low collision rate keeps TM-bug exposure minimal;
# the retry/pooling safety net from run_coop_matrix.sh is kept as insurance.
set -u
cd "$(dirname "$0")"

export CARLA_ROOT=${CARLA_ROOT:-/home/peh324/carla_simulator}
CKPT=./checkpoints/CarDreamer_checkpoints/right_turn_hard.ckpt
PORT=2000
STALL_SEC=150
# Max wall-clock before first metrics line (CARLA connect + JAX compile + first
# episodes normally < 3min; beyond this = a startup RPC hang the file-based
# stall check cannot see because metrics.jsonl does not exist yet)
STARTUP_MAX_SEC=300
OUT=./logdir/eval/latency
EPISODES_TARGET=110

DESIG="--env.observation.birdeye_wpt.waypoint_obs designated"
STEPS="--dreamerv3.run.steps 2e4"

ensure_carla() {
    if ! pgrep -f "CarlaUE4-Linux.*-carla-port=$PORT" > /dev/null; then
        echo "[lat] launching CARLA..."
        nohup "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -carla-port=$PORT -benchmark -fps=10 > carla_latency.log 2>&1 &
        until nc -z localhost $PORT 2>/dev/null; do sleep 2; done
        sleep 10
    fi
}

kill_carla() {
    pkill -9 -f "CarlaUE4.*-carla-port=$PORT" 2>/dev/null
    sleep 3
}

run_eval() {
    local tag=$1; shift
    local logdir=$OUT/$tag
    if [ -f "$logdir/DONE" ]; then
        echo "[lat] $tag already done, skipping"
        return
    fi
    local attempt
    for attempt in 1 2 3 4 5 6 7 8; do
        ensure_carla
        echo "[lat] start $tag attempt$attempt $(date '+%H:%M:%S')"
        PYTHONPATH="$(pwd)" conda run -n cardreamer python -X faulthandler -u dreamerv3/eval.py \
            --env.world.carla_port $PORT \
            --dreamerv3.jax.policy_devices 0 \
            --dreamerv3.run.from_checkpoint "$CKPT" \
            --task carla_right_turn_hard \
            --dreamerv3.logdir "$logdir" \
            "$@" >> "$OUT/${tag}.log" 2>&1 &
        local pid=$!
        local start_ts=$(date +%s)

        while kill -0 $pid 2>/dev/null; do
            sleep 30
            local now=$(date +%s)
            local m="$logdir/metrics.jsonl"
            if [ -f "$m" ]; then
                # mid-run hang: metrics file exists but hasn't advanced
                local age=$(( now - $(stat -c %Y "$m") ))
                if [ "$age" -gt "$STALL_SEC" ]; then
                    echo "[lat] $tag STALLED (${age}s no metrics update), killing run + CARLA"
                    pkill -9 -f "latency/$tag" 2>/dev/null
                    kill_carla
                    break
                fi
            elif [ $(( now - start_ts )) -gt $STARTUP_MAX_SEC ]; then
                # startup hang: no metrics file at all after too long (CARLA
                # RPC hang during connect/load/first-reset -- the file-based
                # stall check above can never fire because the file is absent)
                echo "[lat] $tag STARTUP-HANG ($(( now - start_ts ))s, no metrics file), killing run + CARLA"
                pkill -9 -f "latency/$tag" 2>/dev/null
                kill_carla
                break
            fi
        done
        wait $pid 2>/dev/null

        local episodes=0
        if [ -f "$logdir/metrics.jsonl" ]; then
            episodes=$(grep -c '"episode/score"' "$logdir/metrics.jsonl")
        fi
        if [ "${episodes:-0}" -ge $EPISODES_TARGET ]; then
            touch "$logdir/DONE"
            echo "[lat] done $tag $(date '+%H:%M:%S') episodes=$episodes attempts=$attempt"
            return
        fi
        echo "[lat] $tag attempt$attempt ended (accumulated episodes=${episodes:-0}/$EPISODES_TARGET), restarting CARLA and retrying"
        kill_carla
    done
    echo "[lat] FAILED $tag after 8 attempts"
}

mkdir -p "$OUT"

# --- smoke cells first: validate pipeline + latency semantics ---
run_eval nearest2_k0 $DESIG $STEPS --env.intention_sharing.rule nearest --env.intention_sharing.num 2 --env.intention_sharing.latency_steps 0
run_eval all_k1 $DESIG $STEPS --env.intention_sharing.rule all --env.intention_sharing.latency_steps 1

# --- main matrix ---
for k in 1 2 3 4 5 6 7 8 9 10; do
    run_eval all_k$k $DESIG $STEPS --env.intention_sharing.rule all --env.intention_sharing.latency_steps $k
    run_eval nearest1_k$k $DESIG $STEPS --env.intention_sharing.rule nearest --env.intention_sharing.num 1 --env.intention_sharing.latency_steps $k
    run_eval nearest2_k$k $DESIG $STEPS --env.intention_sharing.rule nearest --env.intention_sharing.num 2 --env.intention_sharing.latency_steps $k
    run_eval random1_k$k $DESIG $STEPS --env.intention_sharing.rule random --env.intention_sharing.num 1 --env.intention_sharing.latency_steps $k
done

echo "[lat] ALL DONE"
