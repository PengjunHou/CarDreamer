### 跑完整 1 个 episode（到 terminated/truncated）就结束
python scripts/run_env.py --episodes 1

### 跑 3 个完整 episode
python scripts/run_env.py --episodes 3

### 配合 autopilot 看标准 task 完整跑一遍
python scripts/run_env.py --task carla_roundabout --autopilot --episodes 1
