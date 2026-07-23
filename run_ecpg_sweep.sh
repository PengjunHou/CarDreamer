#!/bin/bash
# ECPG geometry-recording sweep: 4 rules x 11 latencies = 44 configs.
# Each config records ~600 steps of per-frame geometry (~15-20 episodes) via
# CARDREAMER_RECORD_GEOMETRY, with the web monitor disabled (avoids the port
# 9000 conflict) and hang/crash watchdogs. Fresh file per attempt.
set -u
cd "$(dirname "$0")"

export CARLA_ROOT=${CARLA_ROOT:-/home/peh324/carla_simulator}
CKPT=./checkpoints/CarDreamer_checkpoints/right_turn_hard.ckpt
PORT=2000
STALL_SEC=150
STARTUP_MAX_SEC=300
STEPS=1000
MIN_FRAMES=600
GEOM=./logdir/ecpg_geom

ensure_carla() {
    if ! pgrep -f "CarlaUE4-Linux.*-carla-port=$PORT" > /dev/null; then
        echo "[ecpg] launching CARLA..."
        nohup "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -carla-port=$PORT -benchmark -fps=10 > carla_ecpg.log 2>&1 &
        until nc -z localhost $PORT 2>/dev/null; do sleep 2; done
        sleep 10
    fi
}
kill_carla() { pkill -9 -f "CarlaUE4.*-carla-port=$PORT" 2>/dev/null; sleep 3; }

run_cfg() {
    local tag=$1 rule=$2 num=$3 k=$4
    local geom="$GEOM/${tag}.jsonl"
    if [ -f "$GEOM/${tag}.DONE" ]; then echo "[ecpg] $tag done, skip"; return; fi
    local attempt
    for attempt in 1 2 3 4; do
        # Fresh CARLA per config: a killed eval leaves dangling sensor streams
        # that poison the next eval (ego stalls). A clean restart avoids it.
        kill_carla
        ensure_carla
        rm -f "$geom"
        echo "[ecpg] start $tag attempt$attempt $(date '+%H:%M:%S')"
        CARDREAMER_RECORD_GEOMETRY="$(pwd)/${geom#./}" \
        PYTHONPATH="$(pwd)" conda run -n cardreamer python -X faulthandler -u dreamerv3/eval.py \
            --env.world.carla_port $PORT --dreamerv3.jax.policy_devices 0 \
            --dreamerv3.run.from_checkpoint "$CKPT" --task carla_right_turn_hard \
            --dreamerv3.logdir "$GEOM/run_$tag" \
            --env.observation.birdeye_wpt.waypoint_obs designated \
            --env.intention_sharing.rule $rule --env.intention_sharing.num $num \
            --env.intention_sharing.latency_steps $k \
            --dreamerv3.run.steps $STEPS >> "$GEOM/${tag}.log" 2>&1 &
        local pid=$! start_ts=$(date +%s)
        while kill -0 $pid 2>/dev/null; do
            sleep 20
            local now=$(date +%s)
            if [ -f "$geom" ]; then
                local age=$(( now - $(stat -c %Y "$geom") ))
                [ "$age" -gt "$STALL_SEC" ] && { echo "[ecpg] $tag STALLED"; pkill -9 -f "run_$tag" 2>/dev/null; kill_carla; break; }
            elif [ $(( now - start_ts )) -gt $STARTUP_MAX_SEC ]; then
                echo "[ecpg] $tag STARTUP-HANG"; pkill -9 -f "run_$tag" 2>/dev/null; kill_carla; break
            fi
        done
        wait $pid 2>/dev/null
        local frames=$(wc -l < "$geom" 2>/dev/null || echo 0)
        if [ "${frames:-0}" -ge $MIN_FRAMES ]; then
            touch "$GEOM/${tag}.DONE"
            echo "[ecpg] done $tag frames=$frames attempts=$attempt"
            return
        fi
        echo "[ecpg] $tag attempt$attempt short (frames=${frames:-0}), retry"; kill_carla
    done
    echo "[ecpg] FAILED $tag"
}

mkdir -p "$GEOM"
for k in 0 1 2 3 4 5 6 7 8 9 10; do
    run_cfg all_k$k      all     1 $k
    run_cfg nearest1_k$k nearest 1 $k
    run_cfg nearest2_k$k nearest 2 $k
    run_cfg random1_k$k  random  1 $k
done
touch "$GEOM/SWEEP_DONE"
echo "[ecpg] SWEEP COMPLETE"
