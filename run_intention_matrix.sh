#!/bin/bash
# Intention-sharing ablation matrix: eval official right_turn_hard.ckpt
# under different designated-sharing rules. Each run has a stall watchdog
# (original CarDreamer occasionally hangs on a CARLA RPC at episode reset).
set -u
cd "$(dirname "$0")"

export CARLA_ROOT=${CARLA_ROOT:-/home/peh324/carla_simulator}
CKPT=./checkpoints/CarDreamer_checkpoints/right_turn_hard.ckpt
PORT=2000
STALL_SEC=300

ensure_carla() {
    if ! pgrep -f "CarlaUE4-Linux.*-carla-port=$PORT" > /dev/null; then
        echo "[matrix] launching CARLA..."
        nohup "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -carla-port=$PORT -benchmark -fps=10 > carla_matrix.log 2>&1 &
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
    local logdir=./logdir/eval/matrix_$tag
    if [ -f "$logdir/DONE" ]; then
        echo "[matrix] $tag already done, skipping"
        return
    fi
    ensure_carla
    echo "[matrix] start $tag $(date '+%H:%M:%S')"
    PYTHONPATH="$(pwd)" conda run -n cardreamer python -u dreamerv3/eval.py \
        --env.world.carla_port $PORT \
        --dreamerv3.jax.policy_devices 0 \
        --dreamerv3.run.from_checkpoint "$CKPT" \
        --task carla_right_turn_hard \
        --dreamerv3.logdir "$logdir" \
        "$@" > "eval_matrix_$tag.log" 2>&1 &
    local pid=$!

    while kill -0 $pid 2>/dev/null; do
        sleep 30
        local m="$logdir/metrics.jsonl"
        if [ -f "$m" ]; then
            local age=$(( $(date +%s) - $(stat -c %Y "$m") ))
            if [ "$age" -gt "$STALL_SEC" ]; then
                echo "[matrix] $tag STALLED (${age}s since last metric), killing run + CARLA"
                pkill -9 -f "matrix_$tag" 2>/dev/null
                kill_carla
                break
            fi
        fi
    done
    wait $pid 2>/dev/null
    touch "$logdir/DONE"
    echo "[matrix] done $tag $(date '+%H:%M:%S')"
}

DESIG="--env.observation.birdeye_wpt.waypoint_obs designated"

run_eval all      $DESIG --env.intention_sharing.rule all
run_eval none     $DESIG --env.intention_sharing.rule none
run_eval nearest1 $DESIG --env.intention_sharing.rule nearest --env.intention_sharing.num 1
run_eval random1  $DESIG --env.intention_sharing.rule random  --env.intention_sharing.num 1

echo "[matrix] ALL DONE"
