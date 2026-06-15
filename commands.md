### 跑完整 1 个 episode（到 terminated/truncated）就结束
python scripts/run_env.py --episodes 1

### 跑 3 个完整 episode
python scripts/run_env.py --episodes 3

### 配合 autopilot 看标准 task 完整跑一遍
python scripts/run_env.py --task carla_roundabout --autopilot --episodes 1

## Mode A
### 1) 录制:跑一个 run,逐步把真实 _wam_graph + 激活 policy 落到 JSONL(需 CARLA)
python scripts/record_wam_graph_timeline.py \
    --task carla_group_right_turn_auto --carla-port 2000 \
    --steps 300 --out outputs/graph_timeline.jsonl

### 2) 离线渲染(无需 CARLA,可在 cardreamer_gnn 里跑)
python scripts/visualize_wam_graph_timeline.py \
    --jsonl outputs/graph_timeline.jsonl \
    --html outputs/graph_timeline.html \
    --gif  outputs/graph_timeline.gif \
    --png-dir outputs/graph_frames

## Mode B
### 1) 录制:每步对固定 policy 家族各建一张图,写进 timeline JSONL(需 CARLA)
###    --graph-timeline-jsonl 是新加的开关;它同时还会照常产出 stage1 训练样本到 --out-dir
python scripts/record_wam_stage1_policy_data.py \
    --task carla_group_right_turn_auto --carla-port 2000 \
    --steps 200 --out-dir data/wam_stage1_policy \
    --graph-timeline-jsonl outputs/graph_cf.jsonl

### 2) 离线渲染;policy 家族每步会有很多个,建议用 --policies 只挑几个对比
python scripts/visualize_wam_graph_timeline.py \
    --jsonl outputs/graph_cf.jsonl \
    --policies ego_only,all_candidates_objlist,all_candidates_bev \
    --html outputs/graph_cf.html \
    --gif  outputs/graph_cf.gif
