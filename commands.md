### 跑完整 1 个 episode（到 terminated/truncated）就结束
python scripts/run_env.py --episodes 1

### 跑 3 个完整 episode
python scripts/run_env.py --episodes 3

### 配合 autopilot 看标准 task 完整跑一遍
python scripts/run_env.py --task carla_roundabout --autopilot --episodes 1

## Mode A
### 1) 录制:跑一个 run,逐步把真实 _wam_graph + 激活 policy 落到 JSONL(需 CARLA)
###    加 --wide-bev 顺手 dump 一张 100m ego-centered birdeye(让所有 object 进图)
python scripts/record_wam_graph_timeline.py \
    --task carla_group_right_turn_auto --carla-port 2000 \
    --steps 300 --out outputs/graph_timeline.jsonl --wide-bev

### 2) 离线渲染(无需 CARLA,可在 cardreamer_gnn 里跑)
###    用 100m birdeye 当 BEV 底图 -> --bev-obs-range 100 --bev-ego-offset 50
python scripts/visualize_wam_graph_timeline.py \
    --jsonl outputs/graph_timeline.jsonl \
    --birdeye-dir data/birdeye_frames --bev-obs-range 100 --bev-ego-offset 50 \
    --html outputs/graph_timeline.html \
    --gif  outputs/graph_timeline.gif \
    --png-dir outputs/graph_frames

## Mode B
### 1) 录制:每步对固定 policy 家族各建一张图,写进 timeline JSONL(需 CARLA)
###    --graph-timeline-jsonl 写 timeline(且只录单 episode);--wide-bev dump 100m birdeye
python scripts/record_wam_stage1_policy_data.py \
    --task carla_group_right_turn_auto --carla-port 2000 \
    --steps 200 --out-dir data/wam_stage1_policy \
    --graph-timeline-jsonl outputs/graph_cf.jsonl --wide-bev

### 2) 离线渲染;policy 家族每步会有很多个,建议用 --policies 只挑几个对比
python scripts/visualize_wam_graph_timeline.py \
    --jsonl outputs/graph_cf.jsonl \
    --policies ego_only,all_candidates_objlist,all_candidates_bev \
    --birdeye-dir data/birdeye_frames --bev-obs-range 100 --bev-ego-offset 50 \
    --html outputs/graph_cf.html \
    --gif  outputs/graph_cf.gif
