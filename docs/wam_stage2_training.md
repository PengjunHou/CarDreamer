# WAM Stage-2 训练 Pipeline 说明（对照 `docs/WAM/WAM Design.pdf` §16.2）

本文聚焦**本次修改**：Graph Flow-Matching UWM 的 Stage-2 训练 pipeline。逐项说明
**设计文档怎么写的 → 对应代码怎么实现的 → 存在哪些简化 → 后续完善逻辑应该怎样**。

> 前置：§12–§13（flow-matching 模型 + 推理）见 [docs/wam_implementation.md](docs/wam_implementation.md) 的对应小节；
> 本次只做 §16.2 的「训练那一段」，不含 §16.1 Stage 1 与 §16.3 Stage 3。

---

## 0. 设计目标（§16.2）

设计文档 §16.2 的 Stage 2 要训练 **BS-centric Diffusion UWM**（Design Update §7-§8），学习

```
p_θ(π_{q,t:t+H-1}, z^bev_{q,t+1:t+H} | C^BS_{q,t-K:t})
```

- 条件（clean）：`C^BS` = 所有车 perception graph token + request token + driving-task(notable) token + request BEV-history token
- 噪声变量：future policy chunk `π_{q,t:t+H-1}`（解耦时间 `s_π`）、future request BEV latent `z^bev_{q,t+1:t+H}`（`s_z`），外加 register tokens
- 输出：`û_π, û_z`；目标：`L = w_π‖û_π-u_π‖² + w_z‖û_z-u_z‖²`
- **object-state X 不再是生成变量**（faithful 更新）—— 它只活在 driving-task 条件 token 里

本次实现采用**离线两段式**：先用 live env 录制样本到磁盘（需 CARLA），再离线训练（不需 CARLA）。
这样训练与仿真解耦、可复跑、可单测，和已有 `record_wam_notable_debug.py` 的录制范式一致。

---

## 1. 整体数据流

```
 live env (CARLA)                          offline (no CARLA)
 ┌───────────────────────────┐            ┌────────────────────────────────────────┐
 │ 每步: sim._wam_graph        │  .pt 样本  │ WAMFlowDataset(dir)                      │
 │       sim._wam_policy       │ ────────▶ │   └─ collate_flow_samples → batch        │
 │       sim._wam_notable      │           │ WAMStage2Trainer.train():                │
 │ WAMFlowDataRecorder         │           │   condition_tokens_batch → C^BS [B,T,d]  │
 │   observe_policy + register │           │   sample_training_batch (s_π,s_z)        │
 │   flush_ready → torch.save  │           │   flow.forward → flow_matching_loss(§8)  │
 └───────────────────────────┘            │   backward + Adam → checkpoint           │
   scripts/record_wam_flow_data.py        └────────────────────────────────────────┘
                                             scripts/train_wam_stage2.py
```

涉及文件：
- [car_dreamer/toolkit/wam/flow_recorder.py](car_dreamer/toolkit/wam/flow_recorder.py)：`WAMFlowDataRecorder`（录制）。
- [car_dreamer/toolkit/wam/stage2.py](car_dreamer/toolkit/wam/stage2.py)：目标构造 / dataset / collate / trainer / 配置读取。
- [scripts/record_wam_flow_data.py](scripts/record_wam_flow_data.py)、[scripts/train_wam_stage2.py](scripts/train_wam_stage2.py)。
- [tests/test_wam_stage2.py](tests/test_wam_stage2.py)：10 个离线单测。
- 配置：`env.wam.flow.*` + `env.wam.stage2.*`（[tasks.yaml](car_dreamer/configs/tasks.yaml)）。

---

## 2. 逐组件：设计 → 代码

### 2.1 训练样本（一条 = 一个时间步 t）

env 每步本就建好 policy 条件图（`v2v_comm_mixin.py` 的 `self._wam_graph`），把它与「horizon 之后才知道的未来真值」配对即一条样本。

| 字段 | 形状 | 来源 / 含义 |
| --- | --- | --- |
| `vehicle_graphs` | `[HeteroData]` | per-vehicle perception 图列表；v1 = `[request_graph]`（per-vehicle local 图 deferred） |
| `request_index` | `int` | `vehicle_graphs` 中 request 车的下标 |
| `notable_object_ids` | `[K_n]` | request 车 notable object ids（driving-task token 选行用） |
| `policy_chunk` (π₁) | `[H, M, P]` | `encode_policy_chunk(policy(t..t+H-1), candidate_ids, M, F)`，`P=sel(1)+fmt(F)+freq(1)+bw(1)` |
| `member_mask` | `[M]` | candidate 槽位是否存在 |
| `policy_step_mask` | `[H]` | 该未来步 policy 是否可得（给 loss 逐步 mask） |
| `bev_history` | `[K+1, Dz]` | request BEV-latent 当前+历史（零占位，condition token） |
| `bev_future` (z₁) | `[H, Dz]` | 未来 request BEV latent 真值（零占位） |
| `bev_step_mask` | `[H]` | 未来 BEV 步是否监督 |

> condition tokens 在 `WAMUnifiedWorldModel.condition_tokens_batch` 内逐样本由 `WAMBSContextEncoder` 编码、
> 再 `pad_condition_tokens` 右 pad + mask；policy/BEV 张量则在 `collate_flow_samples` 里 pad/stack 到 `[B,H,M/Dz]`。

### 2.2 未来 policy chunk 真值 `π₁`（`encode_policy_chunk`）

policy chunk 槽位 `h` = `t+h` 步的 policy，按**样本 t 时刻的 candidate 顺序**逐步 `encode_policy` 成 `[M,P]`，
堆叠成 `[H,M,P]`；`policy_step_mask[h]=1` 当 `t+h` 的 policy 已记录（缺则 mask）。占位策略下 chunk 近似常量
（与 §17 policy search 落地后改善）。future BEV `bev_future` 为零占位（degenerate 目标），`bev_step_mask` 全 1。

### 2.3 解耦 diffusion time 采样与 loss（§8 / §15.3）

复用 `flow_matching.py`：
- `sample_training_batch(policy_1, bev_1)`：抽 `x0~N(0,I)`、`s_π,s_z~U(0,1)`，给插值输入 `π_{s_π}/z^bev_{s_z}` 与目标速度 `u_π=π_1-π_0`、`u_z=z_1-z_0`。
- `flow_matching_loss`：`L = w_π·‖û_π-u_π‖² + w_z·‖û_z-u_z‖²`（masked mean；`policy_mask [B,H,M]`、`bev_mask [B,H]`），返回 `{total, policy, bev}`。

### 2.4 训练循环（`WAMStage2Trainer`，对应 §16.2）

每个 batch：
1. `cond, type_ids, mask = model.condition_tokens_batch(samples)`（逐样本 `WAMBSContextEncoder` 编码 + `pad_condition_tokens`；梯度回传 encoder）。
2. `model.training_step(cond, type_ids, mask, policy_1, bev_1, member_mask, policy_step_mask, bev_step_mask, policy_loss_mask)`：内部 `sample_training_batch`（§8 采样 `s_π/s_z` + 噪声）→ `flow.forward` → `flow_matching_loss`。
3. `backward` → `clip_grad_norm_` → `Adam.step()`。

附：日志（`[wam-stage2] step=… L=… policy=… bev=…`）、定期 `save_checkpoint`、可选 `evaluate(val)`、`load_checkpoint`。
**graph encoder 与 diffusion 联合端到端训练**（Stage 1 未训，故 encoder 靠 `L` 学；`freeze_encoder` 冻结 `context_encoder`）。

### 2.5 录制器（`WAMFlowDataRecorder`）

- `observe_policy(step, policy)`：每步记 policy（assemble future chunk 用）。
- `register(step, graph, candidate_ids, notable_object_ids)`：登记 request 车样本骨架（v1 `vehicle_graphs=[graph]`）。
- `flush_ready(step)`：horizon（=H-1）到期的样本，调 `encode_policy_chunk` 生成 `policy_chunk`，组装 `torch.save` 成 `sample_*.pt`。
- CARLA 相关抽取在 `scripts/record_wam_flow_data.py` 里（`sim._wam_graph` / `sim._wam_policy` / `sim._wam_notable_records`），
  录制器本体是纯逻辑、可单测。

### 2.6 配置（`wam_configs_from_env`）

读 `env.wam.graph.*`（共享 `hidden_dim`/`route_waypoints`）、`env.wam.flow.*`、`env.wam.stage2.*` →
`(WAMGraphModelConfig, WAMFlowMatchingConfig, WAMStage2Config)`。`flow.bev_latent_dim` 默认取 `hidden_dim`。

---

## 3. 是否存在简化（及影响）

| 简化点 | 现状 | 为什么 | 影响 |
| --- | --- | --- | --- |
| `bev_future`/`bev_history` | **零占位**（仍纳入 `L`） | 没有真实 per-vehicle BEV latent / `E_bev` | BEV 半边学退化目标（去噪向 0）；占算力但不破坏 policy 学习 |
| `vehicle_graphs` | `[request_graph]`（N=1） | per-vehicle local 图装配 deferred（需 per-vehicle route/notable） | BS 全局上下文暂只有请求车一张图 |
| 生成变量 | 仅 policy chunk + future BEV（X 不生成） | faithful 更新 §7（X→condition token） | object 未来由 driving-task 条件 token 间接表达 |
| 图编码 | 逐图 Python 循环（非 PyG `Batch`） | context pool 必须**单图内**池化 | 小 batch 够用；大 batch 偏慢 |
| policy 真值 `π_1` | 占位策略（选全部候选）chunk | §17 policy search 未做 | 学到的是该（低多样性）策略分布 |
| Stage 1 | 未训 → encoder 由 `L` 联合训练 | §16.1 训练循环未做 | 无预训练 warm-start / 辅助监督 |
| ODE 推理 | 定步长 Euler（`n_inference_steps`） | 第一版 | 采样精度有限 |
| 数据管线 | 单目录 `.pt`，全量内存 `DataLoader` | 第一版规模小 | 不支持分片/大规模/归一化 |

> 一致性约束：训练用的 `route_waypoints`/`hidden_dim` 必须与录制样本里 graph 一致；`flow.horizon` 必须等于 recorder 的 `samples`（=H）。

---

## 4. 后续完善逻辑（按优先级）

下面是把这条 pipeline 从「能跑」推进到「能用/能发论文」的完善路线。

### 4.1 真实未来真值（最高价值）
- **真实 `z^bev`（核心）**：实现 per-vehicle BEV semantic map 提取 + §5.3 真实 `E_bev`（替换零占位 CNN），把未来 H 步的 request-vehicle `z^bev` 填进 `bev_future`、当前+历史填进 `bev_history`；`L` 的 BEV 项从「对 0」变成「对真值」。**这一步打通后，BS-centric UWM 生成的 future observation 那半才真正有意义**（当前是退化目标）。
- **per-vehicle local 图（核心）**：把样本的 `vehicle_graphs` 从 `[request_graph]`（N=1）扩展为覆盖范围内每辆车的 local/V2V 图（复用 `is_fov_visible` 逐车判可见性）；需要给每辆车一条 route 以做 per-vehicle notable（目前只有 ego 有 route）。这一步打通后 BS 全局上下文才名副其实。
- 配套 §14 **BEV decoder** + §15.4 **BEV 重建 loss**（`CE(B̂_sem, B_sem^gt)` + 可选 latent L2），用于可视化与评测 future BEV rollout 质量。

### 4.2 policy 数据多样性（与 §17 联动）
- 现在 `π_1` 来自「选全部候选」的占位策略，分布单一。等 §17 heuristic 候选枚举 + `U^π` reranking 落地后，录制时**随机/多策略**采样 policy（selected 子集、modality、bandwidth、frequency 枚举），让 flow 学到真正的 `p(π | G^e)`。
- 也可在录制阶段对同一帧图**多策略增广**（一图多 policy 样本），提升 policy 分支的覆盖。

### 4.3 训练工程化
- **批量向量化**：把逐图编码换成 PyG `Batch.from_data_list` + 分段 attention pooling（按 batch index 池化），消除 Python 循环。
- **数据管线规模化**：样本分片目录、`num_workers>0`、train/val/test 切分、特征归一化（位置/速度的均值方差，flow 对尺度敏感）、多 episode / 多 task 混合。
- **训练稳定性**：`w_pi/w_o` 调参（policy 与 obs 量纲不同）、LR schedule / warmup、grad-clip 已有、可加 EMA、混合精度、断点续训（checkpoint 已存 optimizer/step）。

### 4.4 接入完整 pipeline（Stage 1 / Stage 3）
- **§16.1 Stage 1**：先用真值标签预训练 graph encoder + Gaussian 轨迹头（perception 头等真 detector 标签再训），把权重作为 Stage 2 的 warm-start；或保持联合训练但加 §15.1/§15.2 作为辅助 loss（多任务）。
- **§16.3 Stage 3**：用训练好的 checkpoint，串 `propose_policies`(§13.2) → `U^π` reranking(§11.2 `policy_uncertainty`) → 选 policy → 重建图 → 评测 uncertainty reduction / collision rate / average speed。
- **推理 ODE 升级**：Euler → Heun/RK4 或 few-step distillation，提升采样质量/速度。

### 4.5 评测指标
- future BEV rollout 质量：`rollout_future_bev` 的 `Ẑ^bev` vs GT 的 latent MSE、解码后 BEV 重建 IoU/mIoU（需真实 BEV + §14）。
- policy 生成质量：`propose_policies` 的候选 chunk 经 `U^π` 排序后，top-k 相对随机/全选的 uncertainty 下降。
- 注：object 未来轨迹的 ADE/FDE 属 **deterministic Gaussian 轨迹头**（§11.2，Stage 1），不再由 diffusion UWM 生成（X 已是 condition token）。

---

## 5. 如何验证

**离线（不需 CARLA）**
```bash
conda run -n cardreamer_gnn python -m unittest tests.test_wam_stage2 -v
conda run -n cardreamer_gnn python -m unittest discover -s tests -p "test_*.py"   # 全套
```
`test_wam_stage2` 覆盖：`build_object_state_target`（位置=GT/速度=有限差分/static 保持/缺失 mask）、
`WAMFlowDataset`+collate 形状与 pad、trainer 单 batch 有限、**过拟合时 `L_FM` 下降**、
**checkpoint 存取还原 loss**、`freeze_encoder` 切换 encoder 梯度、`WAMFlowDataRecorder` 纯逻辑、离线端到端训练落 checkpoint。

**在线（录制需 CARLA；训练不需）**
```bash
python scripts/record_wam_flow_data.py --task carla_group_right_turn_auto --carla-port 2000 --steps 400 --out-dir data/wam_flow
python scripts/train_wam_stage2.py --data-dir data/wam_flow --task carla_group_right_turn_auto --steps 2000
```
期望：`data/wam_flow/sample_*.pt` 出现；训练日志 `L_FM` 随 step 下降；`outputs/wam_stage2/stage2_step*.pt` 写出且可被
`WAMStage2Trainer.load_checkpoint` 还原。

---

## 6. 关键文件 / 配置一览

- 代码：`stage2.py`（trainer/dataset/collate/sample）、`flow_recorder.py`（录制）、`flow_matching.py`（`WAMBSContextEncoder` / `WAMFlowMatchingUWM` / `pad_condition_tokens` / `encode_policy_chunk`）。
- 脚本：`record_wam_flow_data.py`（录制，需 CARLA）、`train_wam_stage2.py`（训练，不需 CARLA）。
- 配置：`env.wam.flow.{num_layers,num_heads,time_embed_dim,max_members,num_formats,horizon,history_window,num_register_tokens,enable_bev,n_inference_steps}`、
  `env.wam.stage2.{lr,batch_size,steps,w_policy,w_bev,grad_clip,log_interval,ckpt_interval}`；`hidden_dim`/`route_waypoints` 复用 `env.wam.graph.*`。
- 导出：见 `car_dreamer/toolkit/wam/__init__.py`（`WAMStage2Trainer` / `WAMStage2Config` / `WAMFlowDataset` / `WAMFlowDataRecorder` / `WAMBSContextEncoder` / `collate_flow_samples` / `wam_configs_from_env` / `make_flow_sample` / `encode_policy_chunk`）。
