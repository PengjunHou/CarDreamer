#!/bin/bash
# Coop-perception experiment: communication latency x collaborator selection.
# Zero-shot eval of right_turn_sfov.ckpt on carla_right_turn_hard with the
# coop BEV (local FOV layer + delayed comm layer). 47 cells, 2e4 steps each.
# Smoke cells (all_k0, none, all_k5) run first; DONE markers allow safe re-runs.
set -u
cd "$(dirname "$0")"

export CARLA_ROOT=${CARLA_ROOT:-/home/peh324/carla_simulator}
CKPT_SFOV=./checkpoints/CarDreamer_checkpoints/right_turn_sfov.ckpt
CKPT_FOV=./checkpoints/CarDreamer_checkpoints/right_turn_fov.ckpt
PORT=2000
STALL_SEC=300
OUT=./logdir/eval/coop
# A cell is complete once this many episodes have accumulated across attempts
# (a clean 2e4-step run yields ~115-140 episodes)
EPISODES_TARGET=110

COOP_OBS="--env.observation.birdeye_wpt.observability fov \
--env.observation.birdeye_wpt.waypoint_obs designated \
--env.observation.birdeye_wpt.entities roadmap,waypoints,background_waypoints,ego_vehicle,background_vehicles,comm_vehicles"
STEPS="--dreamerv3.run.steps 2e4"

ensure_carla() {
    if ! pgrep -f "CarlaUE4-Linux.*-carla-port=$PORT" > /dev/null; then
        echo "[coop] launching CARLA..."
        nohup "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -carla-port=$PORT -benchmark -fps=10 > carla_coop.log 2>&1 &
        until nc -z localhost $PORT 2>/dev/null; do sleep 2; done
        sleep 10
    fi
}

kill_carla() {
    pkill -9 -f "CarlaUE4.*-carla-port=$PORT" 2>/dev/null
    sleep 3
}

run_eval() {
    local tag=$1 ckpt=$2; shift 2
    local logdir=$OUT/$tag
    if [ -f "$logdir/DONE" ]; then
        echo "[coop] $tag already done, skipping"
        return
    fi
    # Up to 8 attempts per cell: the CARLA 0.9.16 TM client can segfault
    # (upstream bug, probabilistic per collision-reset); metrics.jsonl appends
    # across attempts and same-config episodes pool statistically, so a cell is
    # complete once the ACCUMULATED episode count reaches the target.
    local attempt
    for attempt in 1 2 3 4 5 6 7 8; do
        ensure_carla
        echo "[coop] start $tag attempt$attempt $(date '+%H:%M:%S')"
        PYTHONPATH="$(pwd)" conda run -n cardreamer python -X faulthandler -u dreamerv3/eval.py \
            --env.world.carla_port $PORT \
            --dreamerv3.jax.policy_devices 0 \
            --dreamerv3.run.from_checkpoint "$ckpt" \
            --task carla_right_turn_hard \
            --dreamerv3.logdir "$logdir" \
            "$@" >> "$OUT/${tag}.log" 2>&1 &
        local pid=$!

        while kill -0 $pid 2>/dev/null; do
            sleep 30
            local m="$logdir/metrics.jsonl"
            if [ -f "$m" ]; then
                local age=$(( $(date +%s) - $(stat -c %Y "$m") ))
                if [ "$age" -gt "$STALL_SEC" ]; then
                    echo "[coop] $tag STALLED (${age}s), killing run + CARLA"
                    pkill -9 -f "coop/$tag" 2>/dev/null
                    kill_carla
                    break
                fi
            fi
        done
        wait $pid 2>/dev/null

        local episodes=0
        if [ -f "$logdir/metrics.jsonl" ]; then
            episodes=$(grep -c '"episode/score"' "$logdir/metrics.jsonl")
        fi
        if [ "${episodes:-0}" -ge $EPISODES_TARGET ]; then
            touch "$logdir/DONE"
            echo "[coop] done $tag $(date '+%H:%M:%S') episodes=$episodes attempts=$attempt"
            return
        fi
        echo "[coop] $tag attempt$attempt ended (accumulated episodes=${episodes:-0}/$EPISODES_TARGET), restarting CARLA and retrying"
        kill_carla
    done
    echo "[coop] FAILED $tag after 8 attempts"
}

mkdir -p "$OUT"

# --- smoke cells first ---
run_eval all_k0 $CKPT_SFOV $COOP_OBS $STEPS --env.intention_sharing.rule all --env.intention_sharing.latency_steps 0
run_eval none $CKPT_SFOV $COOP_OBS $STEPS --env.intention_sharing.rule none
run_eval all_k5 $CKPT_SFOV $COOP_OBS $STEPS --env.intention_sharing.rule all --env.intention_sharing.latency_steps 5

# --- reference cells ---
run_eval ref_sfov_native $CKPT_SFOV $STEPS --env.observation.birdeye_wpt.observability recursive_fov
run_eval ref_fov_native $CKPT_FOV $STEPS --env.observation.birdeye_wpt.observability fov

# --- main matrix ---
for k in 0 1 2 3 4 5 6 7 8 9 10; do
    run_eval all_k$k $CKPT_SFOV $COOP_OBS $STEPS --env.intention_sharing.rule all --env.intention_sharing.latency_steps $k
    run_eval nearest1_k$k $CKPT_SFOV $COOP_OBS $STEPS --env.intention_sharing.rule nearest --env.intention_sharing.num 1 --env.intention_sharing.latency_steps $k
    run_eval nearest2_k$k $CKPT_SFOV $COOP_OBS $STEPS --env.intention_sharing.rule nearest --env.intention_sharing.num 2 --env.intention_sharing.latency_steps $k
    run_eval random1_k$k $CKPT_SFOV $COOP_OBS $STEPS --env.intention_sharing.rule random --env.intention_sharing.num 1 --env.intention_sharing.latency_steps $k
done

echo "[coop] ALL DONE"
