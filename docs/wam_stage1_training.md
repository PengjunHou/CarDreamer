# WAM Stage-1 训练 Pipeline 说明（对照 `docs/WAM/WAM Design.pdf` §16.1）

本文聚焦 **Stage 1：encoder + deterministic heads 预训练**。逐项说明
**设计文档怎么写 → 代码怎么实现 → 有哪些简化 → 后续怎么完善**。

> 前置：§5/§9 编码器、§10 temporal、§11 heads、§15.1/§15.2 loss 见 [docs/wam_implementation.md](docs/wam_implementation.md)；
> 本次只做 §16.1 的「训练那一段」，不含 §16.2 Stage 2 与 §16.3 Stage 3。

---

## 0. 设计目标（§16.1）

预训练 graph encoder + temporal encoder + Notable Head（§11.1）+ Gaussian 轨迹头（§11.2），让模型：
- 学到 task-aware 环境理解（notable / visible / invisible 识别）；
- 输出可信的预测不确定度 **`U^π`**（§11.2）—— 它是 §17 policy search 的 reward 核心。

**为什么先做 Stage 1**：`U^π` 来自 Gaussian 轨迹头；不训练它，policy reward 无意义，§17 / §16.3 闭环无法评测。
而 Stage 1 **不缺 GT**：perception 标签已挂在图 object node 上，轨迹真值由现成的 `TrajectoryTargetBuffer` 逻辑产出。

采用**离线两段式**：live env 录制 window 样本（需 CARLA）→ 离线训练（不需 CARLA）。与 Stage 2 同构、可复跑、可单测。

---

## 1. 整体数据流

```
 live env (CARLA)                          offline (no CARLA)
 ┌───────────────────────────┐            ┌────────────────────────────────────────┐
 │ 每步: sim._wam_graph        │  .pt 样本  │ WAMStage1Dataset(dir)                    │
 │       全场 actor 世界坐标    │ ────────▶ │   └─ collate_stage1_samples → batch      │
 │ WAMStage1DataRecorder       │           │ WAMStage1Trainer.train():                │
 │   observe + register(滑窗)   │           │   per window: WAMPerceptionModel.forward │
 │   flush_ready → torch.save  │           │   perception_loss + gaussian_traj_nll    │
 └───────────────────────────┘            │   backward + Adam → checkpoint           │
   scripts/record_wam_stage1_data.py      └────────────────────────────────────────┘
                                             scripts/train_wam_stage1.py
                                             ─▶ Stage-2 warm-start: init_encoder_from_stage1
```

涉及文件：
- [car_dreamer/toolkit/wam/stage1.py](car_dreamer/toolkit/wam/stage1.py)：config / dataset / collate / trainer / 目标对齐 / warm-start / 配置读取。
- [car_dreamer/toolkit/wam/stage1_recorder.py](car_dreamer/toolkit/wam/stage1_recorder.py)：`WAMStage1DataRecorder`（滑窗录制）+ `valid_object_ids`。
- [scripts/record_wam_stage1_data.py](scripts/record_wam_stage1_data.py)、[scripts/train_wam_stage1.py](scripts/train_wam_stage1.py)。
- [tests/test_wam_stage1.py](tests/test_wam_stage1.py)：10 个离线单测。
- 配置：`env.wam.stage1.*` + 复用 `env.wam.graph.*`（[tasks.yaml](car_dreamer/configs/tasks.yaml)）。

---

## 2. 逐组件：设计 → 代码

### 2.1 训练样本（一条 = 一个时间步 t 的图窗口）

| 字段 | 形状 | 来源 / 含义 |
| --- | --- | --- |
| `window` | `[HeteroData × (K+1)]` | 最近 `K+1` 步的 policy 条件图（oldest→newest）；仅用于 `WAMPerceptionModel` |
| `target_xy` | `[Q, H, 2]` | 最后一张图有效 object 的 GT 未来 ego 帧 xy（`build_trajectory_targets`） |
| `valid` | `[Q, H]` | 该未来步是否有真值（给 NLL 逐步 mask） |
| `object_node_ids` | `[Q]` | 最后一张图的有效 object ids（`valid_object_ids`，与模型 query 对齐用） |

> perception 标签（notable/visible/invisible）已挂在 `window[-1]` 的 object node 上（建图时写入），
> `WAMPerceptionModel.forward` 内部直接取，**无需单独记录**。occluding 无标签 → `λ_occ=0`。

### 2.2 GT 未来轨迹真值（复用 §15.2 思路）

`build_trajectory_targets(object_node_ids, ego_pose, future_positions)` → `target_xy [Q,H,2]`（t 时刻 ego 帧）+ `valid [Q,H]`。
未来步偏移用 `future_sample_step_offsets(fixed_dt, horizon_s, traj_samples)`，与轨迹头的 `H=traj_samples` 一致。
`valid_object_ids(graph)` 用 `node_id≥0 & node_mask>0.5`（节点顺序）—— 与 `WAMPerceptionModel.forward` 的 query 筛选**完全一致**，保证真值行序与模型输出 `object_node_ids` 对齐（trainer 再按 id 兜底重排）。

### 2.3 Loss（§15.1 + §15.2）

逐 window：
- `out = WAMPerceptionModel(window)` → `perception_logits`、`traj_mu/traj_log_var [Q,H,2]`、`object_node_ids`、`labels`。
- `perception_loss(out["perception_logits"], out["labels"], weights={notable:1, visible:λ_vis, invisible:λ_inv, occluding:0})`（BCE）。
- `gaussian_trajectory_nll(traj_mu, traj_log_var, target_xy, notable_weight=labels["notable"], valid_mask=valid)`（notable 加权 NLL）。
- `L = λ_perc·perc + λ_traj·nll`。

### 2.4 训练循环（`WAMStage1Trainer`）

每个 batch：逐样本算 `L`、batch 内求和取平均（样本 `Q` 不同、`WAMPerceptionModel` 一次吃一个 window）→ backward + `clip_grad_norm_` + Adam。
附：日志 `[wam-stage1] step=… L=… perc=… traj=…`、定期 `save_checkpoint`、`load_checkpoint`、`evaluate`（§20.1/§20.2：notable F1 / invisible recall / ADE / FDE / mean `U^π`，复用 `perception_metrics` / `trajectory_ade_fde` / `policy_uncertainty`）。

### 2.5 录制器（`WAMStage1DataRecorder`，仿 `TrajectoryTargetBuffer`）

- 维护最近 `K+1` 张图的 `deque`；`observe(step, snapshots)` 记全场 actor 世界 xy；`register(step, graph, ego_pose)` 把图滑入窗口并登记样本骨架（窗口快照 + `valid_object_ids`）。
- `flush_ready(step)`：horizon 到期的样本，调 `build_trajectory_targets` 生成真值，`torch.save` 成 `sample_*.pt`。
- CARLA 抽取在 `scripts/record_wam_stage1_data.py`（`sim._wam_graph` / `snapshot_from_carla_actor` / ego pose），录制器本体纯逻辑、可单测。

### 2.6 Stage-2 warm-start（`init_encoder_from_stage1`）

Stage-1 与 Stage-2 的图编码器都是同 config 的 `WAMHeteroGraphNet`。`init_encoder_from_stage1(stage2_model, ckpt)` 取 Stage-1 state_dict 里 `graph_net.*` 子状态，装进 `WAMUnifiedWorldModel.context_encoder.graph_net`（键完全匹配，返回 `(missing, unexpected)` 应为空）。脚本 `train_wam_stage2.py --init-from-stage1 <ckpt>`（可叠加 `--freeze-encoder`）。

### 2.7 配置（`wam_stage1_configs_from_env`）

读 `env.wam.graph.*`（共享 `hidden_dim`/`route_waypoints`/layers/heads）+ `env.wam.stage1.*`（λ/temporal/head/traj/history_window）→ `(WAMPerceptionConfig, WAMStage1Config)`。

---

## 3. 是否存在简化（及影响）

| 简化点 | 现状 | 为什么 | 影响 |
| --- | --- | --- | --- |
| occluding（§8.3） | 未实现，occ 标签恒 0、`λ_occ=0` | 无遮挡几何/标签 | 不训 occ 头 |
| 可见性来源 | GT + FOV/遮挡几何，非真实 detector | 同全局简化 | 标签是 oracle，非检测器输出 |
| window 编码 | 逐样本 Python 循环（非 batch 向量化） | temporal 须在单 window 内对齐 | 小 batch 够用；大 batch 偏慢 |
| 每步一张图 | `window` 用 request 图（N=1/步） | per-vehicle 图属 Stage-2 侧 deferred | 与 Stage 2 的图源一致 |
| 早期窗口 | episode 开头 window 长度 < K+1 | 历史不足 | 模型对任意 L≥1 鲁棒，影响小 |

> 一致性约束：训练用 `route_waypoints`/`hidden_dim` 必须与录制图一致；`traj_samples` 必须等于 recorder 的 `samples`。

---

## 4. 后续完善逻辑

- **真实 detector 标签**：把 GT+FOV 的可见性换成真实检测器输出（含 `s_det`），perception 头才学到真检测分布。
- **接入 §16.3 Stage 3**：用 Stage-1 checkpoint 在完整 pipeline 里算 `U^π` → `propose_policies`(§13.2) reranking → 选 policy → 评测 uncertainty reduction / collision / speed。
- **多任务联合（可选 B 路）**：把 §15.1/§15.2 作为 Stage-2 的辅助 loss 一起优化（当前是独立预训练 + warm-start）。
- **工程化**：window 批量向量化、数据分片、归一化、train/val/test 切分、LR schedule。

---

## 5. 如何验证

**离线（不需 CARLA）**
```bash
conda run -n cardreamer_gnn python -m unittest tests.test_wam_stage1 -v
conda run -n cardreamer_gnn python -m unittest discover -s tests -p "test_*.py"   # 全套
```
`test_wam_stage1` 覆盖：`perception_metrics`/`trajectory_ade_fde` 正确性、recorder 纯逻辑（窗口对齐 + 缺失 mask）、
dataset/collate、trainer 单 batch 有限 + **过拟合下降** + checkpoint 还原 + `evaluate` 指标、**warm-start 把 Stage-2 encoder 权重对齐 Stage-1**。

**在线（录制需 CARLA；训练不需）**
```bash
python scripts/record_wam_stage1_data.py --task carla_group_right_turn_auto --carla-port 2000 --steps 400 --out-dir data/wam_stage1
python scripts/train_wam_stage1.py --data-dir data/wam_stage1 --task carla_group_right_turn_auto --steps 2000
# 用 Stage-1 encoder warm-start Stage 2：
python scripts/train_wam_stage2.py --data-dir data/wam_flow --task carla_group_right_turn_auto \
  --init-from-stage1 outputs/wam_stage1/stage1_step2000.pt --freeze-encoder
```
期望：`data/wam_stage1/sample_*.pt` 出现；训练日志 `L` 随 step 下降、`val_*` 指标可读；`outputs/wam_stage1/stage1_step*.pt` 写出且可被 `load_checkpoint` / `init_encoder_from_stage1` 复用。

---

## 6. 关键文件 / 配置一览

- 代码：`stage1.py`（trainer/dataset/collate/目标对齐/warm-start/配置）、`stage1_recorder.py`（滑窗录制）、`heads.py`（新增 `perception_metrics` / `trajectory_ade_fde`）。
- 脚本：`record_wam_stage1_data.py`（录制，需 CARLA）、`train_wam_stage1.py`（训练，不需 CARLA）、`train_wam_stage2.py --init-from-stage1`（warm-start）。
- 配置：`env.wam.stage1.{lr,batch_size,steps,lambda_perc,lambda_traj,lambda_vis,lambda_inv,grad_clip,log_interval,ckpt_interval,history_window,temporal_hidden_dim,head_hidden_dim,traj_horizon_s,traj_samples,fixed_dt}`；`hidden_dim`/`route_waypoints` 复用 `env.wam.graph.*`。
- 导出：见 `car_dreamer/toolkit/wam/__init__.py`（`WAMStage1Trainer` / `WAMStage1Config` / `WAMStage1Dataset` / `WAMStage1DataRecorder` / `make_stage1_sample` / `collate_stage1_samples` / `wam_stage1_configs_from_env` / `init_encoder_from_stage1` / `valid_object_ids` / `perception_metrics` / `trajectory_ade_fde`）。
