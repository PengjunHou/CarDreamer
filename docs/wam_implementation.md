# WAM 实现说明（对照 `docs/WAM/WAM Design.pdf`）

本文记录 **Graph Flow-Matching Unified World Model (WAM)** 当前已落地的实现，逐节对照设计文档说明：
**对应代码在哪、做了哪些简化、还有哪些没实现、输出是什么、有无文档/日志、以及如何验证**。

> 适用提交范围：rule-based runtime（步骤 1–3）、policy-conditioned 异构图 + 初始 embedding +
> **edge-attribute-aware 编码器**（§4–§7、§5、§9 + Edge Representation Update）、temporal encoder +
> deterministic task heads（§10、§11）、轨迹真值抽取（§15.2）、**BS-centric Diffusion UWM**
> + 多种 flow 推理模式（Design Update §2–§8、§13）、Stage-2 离线训练（§16.2）。
>
> **本轮更新（两份设计更新文档）**：
> 1. **Graph Edge Representation Update** —— obs_obj 边带 `det_confidence`、veh_veh 边带 policy-conditioned
>    `latency`，编码器在 attention 中消费 edge attribute（`α_ij = softmax(Q·K + φ(a_ij))`）；`s_det` 从
>    object node 移到 obs_obj 边，object state 由 12 维降为 11 维；`coop` 边更名 `veh_veh`。
> 2. **BS-centric UWM** —— UWM 改为以 **BS 全局 condition tokens**（所有车 perception graph token +
>    request token + notable/driving-task token + request BEV-history token，全部 clean 不加噪）为条件，
>    联合去噪 **future policy chunk** `π_{q,t:t+H-1}` 与 **future request BEV latent** `z^bev_{q,t+1:t+H}`
>    （解耦 diffusion time `s_π`/`s_z` + register tokens）；object-state X 不再作为生成变量。

---

## 0. 总体进度

设计文档的整体因果链：
`Collaboration Policy → Policy-conditioned Cooperative Graph → Graph Unified World Model → Future Object States / BEV → Waypoints`。

| 设计章节 | 内容 | 状态 |
| --- | --- | --- |
| §5.2 / §8 | notable object 定义与选择 | ✅ 已实现（rule-based） |
| §8.1/§8.2 | visible / invisible notable | ✅ 已实现 |
| §8.3 | occluding object | ❌ 未实现（占位 False） |
| §11.2（规则版） | notable 运动预测 + 不确定度 | ✅ 已实现（匀速 + 固定不确定度） |
| §13/§17 | Base Station 协同 policy | 🟡 placeholder（选全部候选） |
| §4–§7 | policy-conditioned 异构图构造 | ✅ 已实现 |
| Edge Update | obs_obj `det_confidence` + veh_veh `latency` 边属性；`s_det` 移出 object node（12→11） | ✅ 已实现（`det_confidence` 恒 1） |
| §5.3 BEV | per-vehicle visibility-aware `B^sem` 栅格 + `E_bev` 真实输入 | ✅ 已实现（drivable 通道暂空） |
| §5 | 三类 node 初始 embedding | ✅ 已实现 |
| §9 | edge-attribute-aware graph transformer（TransformerConv+HeteroConv） | ✅ 已实现 |
| §10 | Temporal Encoder | ✅ 已实现（GRU，仅 object 节点） |
| §11.1 | Notable Object Head（+ vis/inv/occ） | ✅ 已实现 + Stage 1 训练（occ 标签恒 0） |
| §11.2 | Gaussian Trajectory Head（学习版 + `U^π`） | ✅ 已实现 + Stage 1 训练 |
| §15.1/§15.2/§11.2 | perception loss / 轨迹 NLL / U^π | ✅ 函数已实现 |
| §15.2 | GT 未来轨迹真值抽取 | ✅ 已实现 |
| Update §2–§6 | BS condition tokens（per-vehicle graph token + request + driving-task + BEV-history） | ✅ 已实现（`WAMBSContextEncoder`；BEV-history 真实栅格编码） |
| Update §7–§8 | BS-centric Diffusion UWM（future policy chunk + future BEV latent，解耦 `s_π`/`s_z` + register） | ✅ 模型已实现（未训练；X 不再生成） |
| §13.1–§13.3 | flow 推理（rollout BEV / policy proposal / inverse + joint） | ✅ 已实现（Euler 积分） |
| §15.3 | flow matching loss（policy + BEV，masked） | ✅ 函数已实现 |
| §14 | BEV decoder `D_bev` | ✅ 已实现（`WAMBevDecoder`） |
| §15.4 | BEV 重建 loss `CE(B̂^sem, B^sem)` | ✅ 已实现（`bev_reconstruction_loss`，Stage-2 接入） |
| §16.1 | Stage 1：graph encoder + heads 预训练 + `U^π` | ✅ 已实现（recorder + dataset + trainer + checkpoint + Stage-2 warm-start） |
| §16.2 | Stage 2：BS-centric Diffusion UWM 训练 | ✅ 已实现（recorder + dataset + trainer + checkpoint） |
| §16.3 | Stage 3：policy search + planner 验证 | ❌ 未实现 |
| §17 | policy search（按 U^π reranking） | ❌ 未实现（当前选全部候选） |

---

## 1. 代码结构

```
car_dreamer/toolkit/wam/
├── runtime.py            # 步骤 1-3：notable / 运动预测 / coop request / placeholder policy
├── graph.py              # §4-§7：policy-conditioned 异构图构造（HeteroData）
├── graph_model.py        # §5 初始 embedding + §9 HGT 编码器
├── heads.py              # §10 temporal encoder + §11 task heads + §15/§11.2 loss/reward
├── flow_matching.py      # §12 Graph Flow-Matching UWM + §13 推理模式 + §15.3 loss
├── stage2.py             # §16.2 Stage-2 训练：dataset / collate / trainer / 目标构造
├── flow_recorder.py      # §16.2 数据录制：把 live env 的 (graph, GT future, policy) 写成 .pt
├── targets.py            # §15.2 GT 未来轨迹真值抽取
├── debug_recording.py    # rule-based notable/预测 的调试记录器（JSONL）
├── visualization.py      # 调试 BEV 渲染
└── __init__.py           # 统一导出

car_dreamer/v2v_comm_mixin.py   # 把 runtime + 图构造接入仿真环境（每步执行）
car_dreamer/configs/tasks.yaml  # carla_group_right_turn_auto 的 env.wam.* 配置
scripts/check_wam_graph.py          # 在线验证：打印每步图统计 / H_t 形状
scripts/record_wam_notable_debug.py # 在线记录 notable + 预测 vs 真值（JSONL + BEV 帧）
scripts/record_wam_flow_data.py     # §16.2 在线录制 Stage-2 训练样本（.pt，需 CARLA）
scripts/train_wam_stage2.py         # §16.2 离线训练 flow-matching UWM（不需 CARLA）
tests/test_wam_*.py                 # 离线单测（不需要 CARLA）
```

---

## 2. 逐节对照

### 步骤 1–3：环境理解与协同决策（rule-based）

文件：[car_dreamer/toolkit/wam/runtime.py](car_dreamer/toolkit/wam/runtime.py)，集成在 [car_dreamer/v2v_comm_mixin.py](car_dreamer/v2v_comm_mixin.py)。

**(1) Notable Object（§8 / §5.2）** — `select_notable_objects`
- 设计：物体到 ego planned route 的最小距离 `< d_notable=10m` 即为 notable；最多保留最近 `K_notable=3` 个；区分 visible（ego 可见）/ invisible（ego 不可见但协作者可见）。
- 实现：`distance_to_route` 计算点到 route 折线最小距离；`visible = obj.visible_to_ego`；`invisible = (not visible) and obj.visible_to_collaborators`；按 route 距离排序取前 K。
- 可见性来源：[v2v_comm_mixin.py](car_dreamer/v2v_comm_mixin.py) 的 `_wam_build_object_states` 用 [is_fov_visible](car_dreamer/toolkit/observer/handlers/utils.py)（**遮挡感知**的 FOV+range 多边形判定）算 `visible_to_ego` 和 `visible_to_collaborators`（仅统计「参与协同的候选车」）。
- **简化**：用 ground-truth 物体 + FOV/遮挡几何判定「能否看到」，**不是真实 detector**；`s_det` 检测置信度恒 1。
- **未实现**：§8.3 occluding object —— `NotableObjectRecord.occluding` 恒为 `False`。

**(2) 运动预测 + 不确定度（§11.2 的规则替身）** — `predict_notable_motion`
- 设计：预测每个 notable object 未来 H 步的位置分布，提供 reliable predictive uncertainty。
- 实现：**匀速外推** `x_{t+τ}=x+v·τ`，对角协方差为**固定常数**：visible→`0.2`，invisible→`2.0`（来自 config）。
- **简化**：不是学习版高斯轨迹头，不确定度是规则常数（用于触发协同，不是真实预测误差）。真正学习版见下面 §11.2。

**(3) 协同请求 + Base Station policy（§8.2 / §13 / §17）** — `build_coop_request`、`build_placeholder_policy`
- 设计：若 notable 不确定度 > 阈值 → 向 Base Station 请求协同；Base 据此生成 `π_t=(S_t,B_t,f_t,d_t)`。
- 实现：任一预测不确定度 `> uncertainty_threshold(=1.0)` → 生成 `CoopRequest`；`build_placeholder_policy` 在有请求时**选中全部候选协作车**，模态用默认 `objlist`，带宽 `B_t` 均分上行、频率 `f_t = comm_period`。
- **简化 / placeholder**：`S_t` 选全部候选（非按价值挑选）；`B_t`/`f_t` 为均分/固定占位；**没有** §17 的按 `U^π` reranking。
- policy 驱动通信：`_select_collaborators` 返回 `policy.selected_vehicle_ids ∩ 候选集`，从而 V2V 通信只在 ego + 选中协作车之间发生（触发式协同）。

环境集成（每步）：[v2v_comm_mixin.py](car_dreamer/v2v_comm_mixin.py) `_update_wam_runtime_state` 依次跑
`_wam_build_object_states → select_notable_objects → predict_notable_motion → build_coop_request → build_placeholder_policy → _build_wam_graph`，结果缓存在
`self._wam_object_states / _wam_notable_records / _wam_motion_predictions / _wam_coop_request / _wam_policy / _wam_graph`。

WAM 运行参数（默认值，可在 `env.wam.*` 覆盖）：`notable_distance_m=10`、`max_notable_objects=3`、
`reference_waypoint_count=6`、`prediction_horizon_steps=6`、`uncertainty_threshold=1.0`、
`visible_uncertainty=0.2`、`invisible_uncertainty=2.0`、`default_modality=objlist`、local/collaborator 的 FOV/range。

---

### §4–§7：Policy-conditioned 异构图构造

文件：[car_dreamer/toolkit/wam/graph.py](car_dreamer/toolkit/wam/graph.py)；环境侧组装在 `v2v_comm_mixin._build_wam_graph`。

- **§4 三类 node**：`vehicle` / `object` / `observation`。modality（`objlist`/`bev`）作为 observation 节点上的**特征 + type id**，而不是独立 node type —— 这样 §9 的三类 edge relation 完全对应。第一版 `M_0={objlist,bev}`。
- **§5.1 vehicle node state**：`[x,y,z, vx,vy, cos_yaw, sin_yaw, q_comm, q_comp, route(2·K)]`，维度 `9+2·route_waypoints`。
- **§5.2 object node state**：`[x,y,z, vx,vy, cos_yaw, sin_yaw, l, w, h, Δt]`，维度 **11**（Edge Update：`s_det` 移到 obs_obj 边）；object class 用单独 embedding（不在数值向量内）。
- **§5.3 observation node state**：modality id + 标量 `[payload_kb, latency_s, freshness, quality, sample_age_s]`（维度 5）。模态特征 `z^r` 在 embedding 模块里算（见 §5）。
- **§6 三类 edge**（Edge Representation Update）：`(vehicle, veh_obs, observation)`（结构、无属性）、`(observation, obs_obj, object)`（`edge_attr=[det_confidence]`）、`(vehicle, veh_veh, vehicle)`（`edge_attr=[latency_s]`，policy-conditioned；仅协同时存在）。`COOP` 常量更名为 `VEH_VEH`，方向仍是 collaborator m → request/ego。`EDGE_ATTR_DIMS` 声明各关系的属性维度。
  - obs_obj `det_confidence` 来自 `ObservationNodeInput.det_confidence_by_object`（缺省 1.0，ground-truth 感知；真实 detector 后续填）。
  - veh_veh `latency_s` 由 `_build_wam_graph` 复用 `SimpleWirelessLatency.compute_latency_s`（与 observation node `latency_s` 同一计算）汇成 `latency_by_vehicle` 传入。
- **§7 policy-conditioned 装配规则**（`build_wam_hetero_graph`）：
  - ego 永远在图里，自带 `objlist` observation（`L=0, c_fresh=1`），并连到其可见 object；
  - 对每个 `m ∈ policy.selected_vehicle_ids`：加 vehicle 节点 + `coop` 边 `m→ego`；对其激活模态加 observation 节点 + `veh_obs` 边；`objlist` 再连 `m` 可见的 object（`obs_obj`）；
  - object 节点 = ego ∪ 选中协作车所见的并集（按到 ego 距离截断 `max_object_nodes`）；
  - 无协同时退化为合法的 **ego-only 图**。
- 观测标量来源：`payload_bytes` 由物体数估算，`latency_s` 由 `SimpleWirelessLatency.compute_latency_s` 解析计算（ego 自身=0），`freshness = exp(-γ·L)`（§5.3 的 `c_fresh`）。

**简化**：
- 坐标统一转 **ego 帧**、yaw 用 **cos/sin**（设计写的是原始 `p`/`yaw`；这里为可学习性做了改进，是等价信息）。
- `s_det=1.0`、`Δt=0`、observation `quality=1.0`（ground-truth 感知，无检测噪声）。
- latency/freshness 是**按 policy 解析计算**，不是从真实投递的消息队列回填。
- 空 object 类型 pad 到 ≥1 个带 `node_mask` 的 dummy（满足 HGTConv 对每类 ≥1 节点的要求）。

**未实现**：
- §6.2 BEV-to-object 边：`bev` observation 节点在第一版**不连** `obs_obj`。
- 真实 per-vehicle BEV 语义图 `B^sem`：已实现（visibility-aware 栅格 + `E_bev`），详见 [docs/wam_bev.md](docs/wam_bev.md)。

---

### §5：三类 node 的初始 embedding

文件：[car_dreamer/toolkit/wam/graph_model.py](car_dreamer/toolkit/wam/graph_model.py) → `WAMHeteroGraphEmbedding`。

- 每类 node：`h^0 = MLP_type(state) + e^type + e^agent + e^time (+ e^class for object)`。
  - `MLP_veh / MLP_obj / MLP_obs` 分别编码三类 state；
  - `type` embedding 区分 vehicle / object / observation(objlist) / observation(bev)（共 4 个 type id）；
  - `agent` embedding 按 slot 索引（ego=0，协作车=1…Mmax）；`time` embedding（单步=0）；`class` embedding（object 类别）。
- **objlist 模态特征**（§5.3 `z^objlist = AttentionPooling({MLP_obj(x^obj)})`）：用 `obs_obj` 边，把 observation 节点的 object 邻居的 `MLP_obj` 输出做 **attention pooling**（`softmax` over 边 + scatter）。
- **bev 模态特征** `z^bev = E_bev(B^sem)`：`B^sem` 现为**真实 visibility-aware 栅格**（`rasterize_bev`，只含该车可见物体 + ego + route + drivable 通道），过共享 `E_bev`（CNN）得 `z^bev`；详见 [docs/wam_bev.md](docs/wam_bev.md)。

**简化 / 未实现**：`z^bev` 是占位（零图过 CNN）；`time` 只有单步（K 步时序在 §10 处理）；img/text 模态未做。

---

### §9：Heterogeneous Graph Transformer

文件：[car_dreamer/toolkit/wam/graph_model.py](car_dreamer/toolkit/wam/graph_model.py) → `WAMHeteroGraphEncoder` / `WAMHeteroGraphNet`。

- **Edge Representation Update**：由 `HGTConv` 换成 **per-relation `TransformerConv(edge_dim=...)` 包进 `HeteroConv`**（PyG 2.7.0），在 attention 中消费 edge attribute：`α_ij = softmax(Q_i·(K_j + φ(a_ij)))`，其中 `φ` 是 TransformerConv 自带的 `lin_edge`。obs_obj 用 `det_confidence`、veh_veh 用 `latency` 作 `edge_dim`；veh_obs 无属性（`edge_dim=None`）。默认 hidden=256、layers=3、heads=8（要求 hidden 可被 heads 整除）+ 残差 + per-type LayerNorm。
- 无入边的 node type（如无协同时的 vehicle）自动 carry-forward。
- 输出 `H_t = {h^veh, h^obs, h^obj}`（每类 `[N, hidden]`）。
- `WAMHeteroGraphNet.forward(HeteroData) → H_t`：embedding → 构造 `edge_attr_dict`（仅带属性的关系）→ encoder。

**简化**：换 conv 后保留 relation-specific 投影（`HeteroConv` 每关系独立 conv）；`φ` 用 TransformerConv 内置线性边投影（设计写的显式 `edge_encoder` 由它实现，未额外加 MLP）。

---

### §10：Temporal Encoder

文件：[car_dreamer/toolkit/wam/heads.py](car_dreamer/toolkit/wam/heads.py) → `WAMTemporalEncoder` + `align_object_history`。

- 设计链：`x^obj → h^{obj,0} → h^obj → z_o → task heads`。`z_{i,t}=TemporalEncoder(h_{i,t-K},…,h_{i,t})`，第一版用 GRU。
- 实现：`WAMPerceptionModel.forward(window)` 对窗口里每个图跑 `WAMHeteroGraphNet → H_τ`；`align_object_history` 按 `node_id` 把同一 object 跨窗口对齐成 `[Q, L, d]` + presence mask；`WAMTemporalEncoder`（GRU，附 presence bit）聚合，取当前步输出为 `z_{o,t}`。
- 参考了已有的 [GRUTemporalEncoder](dreamerv3/embodied/policy/coop_gnn_policy.py#L335) 风格（但这里跨时间聚合图上下文 embedding，而非 token 序列）。
- 对 object 出现/消失、`K=0`（单图窗口）都鲁棒。

**简化 / 未实现**：
- 只对 **object 节点**做时序编码（task head 只需要 object）；vehicle/observation 节点的时序编码机制相同但未做。
- 输入是「图窗口序列」（训练侧从 replay 取），**环境里不维护这个窗口**。

---

### §11：Deterministic Task Heads

文件：[car_dreamer/toolkit/wam/heads.py](car_dreamer/toolkit/wam/heads.py)。

- **§11.1 Notable Object Head** `NotableObjectHead`：输入 `z_{o,t}`，输出 `notable` logit + 辅助 `vis/inv/occ` logit（共享 trunk → 4 个输出）。
- **§11.2 Gaussian Trajectory Head** `GaussianTrajectoryHead`：输入 `z_{o,t}`，输出未来 H 步的 `(μ, logσ²)`（`[Q,H,2]`），对角协方差，`logσ²` 做 clamp。
- 这是**学习版**的轨迹头（区别于 runtime 里 rule-based 的匀速预测）。

**简化 / 未实现**：
- `occ` 头有输出，但 §8.3 occluding 未实现 → **occ 标签恒 0**，perception loss 里 `λ_occ` 默认 0。
- 头**已实现但未训练**（没有 Stage-1 训练循环，权重是随机初始化的）。

---

### §15 / §11.2：Loss 与 policy reward

文件：[car_dreamer/toolkit/wam/heads.py](car_dreamer/toolkit/wam/heads.py)（函数）。

- `perception_loss`（§15.1）：notable/vis/inv（+可选 occ）的 BCE，加权求和。
- `gaussian_trajectory_nll`（§15.2）：notable 加权的高斯 NLL，按 `valid_mask` 屏蔽缺失未来。
  - **简化**：设计写的是带 `1/2` 的**求和**；实现默认用 **加权平均**（`reduction="mean"`，更稳、与 batch 规模无关），可传 `reduction="sum"` 切回。
- `policy_uncertainty`（§11.2 `U^π_e(t)`）：notable 加权的预测 `Tr(Σ)` 均值，用于给协同 policy 打分（reward）。

---

### §15.2：GT 未来轨迹真值抽取

文件：[car_dreamer/toolkit/wam/targets.py](car_dreamer/toolkit/wam/targets.py)。

- `build_trajectory_targets(...)`：纯函数（torch-free）。把每个 object 的未来世界位置转到 **t 时刻 ego 帧**（与轨迹头预测同帧），返回 `target_xy [Q,H,2]` + `valid_mask [Q,H]`（actor 在某未来步缺席则 mask=0）。
- `TrajectoryTargetBuffer`：滚动历史 + 延迟发射。`observe(step, snapshots)` 记历史，`register(step, object_node_ids, ego_pose)` 登记某步的图，`flush_ready(current_step)` 在 horizon 过后吐出该步对齐好的 target。复用 [debug_recording.py](car_dreamer/toolkit/wam/debug_recording.py) 的 `ActorSnapshot` / `future_sample_step_offsets` / 延迟 flush 模式。
- 在 recording/replay 循环里：把 `WAMPerceptionModel` 输出的 `object_node_ids` 传给 `register`，即可得到与 `traj_mu` 行对齐的真值。

**简化 / 未实现**：把它接进真实 replay/训练数据管线（§16）尚未做；live 抽取的 stepping 模式可复用 `scripts/record_wam_notable_debug.py`。

---

### Update §2–§8：BS-centric Diffusion Unified World Model

文件：[car_dreamer/toolkit/wam/flow_matching.py](car_dreamer/toolkit/wam/flow_matching.py)。

设计修正（Update §1）：UWM 在 **Base Station**，以「policy 确定前就能构建」的 BS 全局上下文为条件，去噪未来 policy 与
未来 BEV latent：`p_θ(π_{q,t:t+H-1}, z^bev_{q,t+1:t+H} | C^BS_{q,t-K:t})`。

- **§2–§6 BS condition tokens**（`WAMBSContextEncoder`，clean 不加噪）：
  - 对每辆覆盖车的 perception graph（local / V2V）跑 §9 edge-aware 编码器 + `WAMGraphContextPool` → 车级 token `g^cp_v`；
  - request token `g^req_q = g^cp_q + e^req`（`e^req` 即 type embedding `_T_REQUEST`，把 request 车标出来）；
  - driving-task tokens `T^task = {h^obj_o : o ∈ notable(q)}`：取 **request graph 编码后的 object embedding** 中 notable 行（notable 集合即驾驶任务，§4）；
  - request BEV-latent tokens `{z^bev_{q,τ}}`（当前 + 历史，§5）—— 由 `E_bev(真实 B^sem 栅格)` 编码得到（不再零占位）。
  - 返回 `(cond_tokens [T_c,d], type_ids [T_c])`；批处理由 `pad_condition_tokens` 右 pad + mask。
- **§7–§8 生成变量与 transformer**（`WAMFlowMatchingUWM`）：
  - noised tokens = **policy chunk** `[B,H,M,P]`（H·M 个 token，带 step/member/type/`e(s_π)`）+ **future BEV** `[B,H,Dz]`（H 个 token，带 step/type/`e(s_z)`）+ **register tokens**；条件 token 拼前面、不加噪。
  - 解耦 diffusion time：`interpolate`、`s_π`/`s_z` 独立；`sample_training_batch(policy_1, bev_1)` 给插值输入 + 目标速度 `u_π=π_1-π_0`、`u_z=z_1-z_0`。
  - 过 `nn.TransformerEncoder`（`key_padding_mask` 屏蔽 pad 成员/步/条件），输出 `û_π=Head_π`、`û_z=Head_bev`。
  - **object-state X 不再是生成变量**（Q2 faithful）—— 它只活在 driving-task 条件 token 里。
- **Loss**：`flow_matching_loss` = `w_π·‖û_π-u_π‖² + w_z·‖û_z-u_z‖²`（masked mean；policy_mask `[B,H,M]`、bev_mask `[B,H]`）。
- **整体组合**：`WAMUnifiedWorldModel` = `WAMBSContextEncoder` + `WAMFlowMatchingUWM`，提供
  `condition_tokens(_batch)(samples)` 与 `training_step(cond, …, policy_1, bev_1)`（采样→forward→loss，梯度回传 graph encoder）。仍是**训练侧 standalone**，未接入 live env。

**简化 / 未实现**：
- BEV-history 条件 token 与 future BEV 目标现由真实 visibility-aware 栅格经 `E_bev` 编码得到（§5.3/§14/§15.4 已实现，见 [docs/wam_bev.md](docs/wam_bev.md)）；BEV-AE 采用 joint+detach 训练（diffusion 目标 detach，重建训 E_bev/D_bev）。
- **per-vehicle 图 deferred**：录制/训练样本里 `vehicle_graphs` 暂为 `[request_graph]`（N=1）；为全部覆盖车构建 local 图（需 per-vehicle route/notable）留待后续。
- `F=2`（objlist/bev）；img/text 未进图；模型**已实现但未训练**。

---

### §13：Diffusion 推理模式

文件：同上（`WAMFlowMatchingUWM` 的方法）。统一**定步长 Euler** ODE 积分（`n_inference_steps`，可传 `n_steps` 覆盖），均以 BS condition tokens 为条件。

- **§13.1 Forward BEV Rollout** `rollout_future_bev(cond…, policy)`：固定 `s_π=1`（给定 policy chunk），BEV 从 `N(0,I)` 沿 `s_z:0→1` 积分 → `Ẑ^bev [B,H,Dz]`。
- **§13.2 Policy Proposal** `propose_policies(cond…, n_candidates)`：固定 `s_z=0`（BEV 边缘化），policy 沿 `s_π:0→1` 一批生成 → `[B,N,H,M,P]`；`decode_policy_vector` 解读 `sel/fmt`（sigmoid）与 `freq/bw`。
- **§13.3 Inverse Policy Search** `inverse_policy_search(cond…, bev_target)`：固定 `s_z=1` 钉住目标未来 BEV，policy 沿 `s_π:0→1` 反推。
- **joint 生成** `joint_generate(cond…)`：两个 diffusion time 同步积分，联合采样 `(π chunk, Ẑ^bev)`。

**简化 / 未实现**：定步长 Euler（可换高阶 solver）；§13.2 候选 reranking 不在本模块（属 §17）。

---

### §16.1：Stage-1 训练 Pipeline（encoder + deterministic heads 预训练）

文件：[car_dreamer/toolkit/wam/stage1.py](car_dreamer/toolkit/wam/stage1.py) +
[car_dreamer/toolkit/wam/stage1_recorder.py](car_dreamer/toolkit/wam/stage1_recorder.py) +
[scripts/record_wam_stage1_data.py](scripts/record_wam_stage1_data.py) +
[scripts/train_wam_stage1.py](scripts/train_wam_stage1.py)。

目标：预训练 graph encoder + temporal encoder + §11.1 Notable Head + §11.2 高斯轨迹头，学到 task-aware 环境理解
与可信的预测不确定度 `U^π`（policy search 的 reward）。**离线录制 + 离线训练**两段式（录制需 CARLA，训练不需）。

> 详细的设计→代码对照、简化清单与后续完善见专文 [docs/wam_stage1_training.md](docs/wam_stage1_training.md)。

- **数据录制**（`WAMStage1DataRecorder`，仿 `TrajectoryTargetBuffer`）：维护最近 `K+1` 张图的滑动窗口；`observe(step, snapshots)` 记全场 actor 世界坐标；`register(step, graph, ego_pose)` 把当前图滑入窗口并登记样本（窗口快照 + 最后一张图的有效 object ids，`valid_object_ids` 与 `WAMPerceptionModel` 的筛选一致以保证对齐）；horizon 到期后用 `build_trajectory_targets` 生成 GT 未来轨迹 `target_xy [Q,H,2]` + `valid [Q,H]`，`torch.save` 成 `.pt`。perception 标签（notable/visible/invisible）已挂在图的 object node 上，**无需单独记录**。
- **样本字段**：`window:[HeteroData×(K+1)]`、`target_xy [Q,H,2]`、`valid [Q,H]`、`object_node_ids [Q]`。
- **训练循环** `WAMStage1Trainer`：每个 window → `WAMPerceptionModel.forward` → `perception_loss`（§15.1 BCE，notable/vis/inv，`λ_occ=0`）+ `gaussian_trajectory_nll`（§15.2，notable 加权，按 `valid` 屏蔽）；`L = λ_perc·perc + λ_traj·nll`，batch 内逐样本求和取平均，backward + grad-clip + Adam。`evaluate` 报告 §20.1/§20.2 指标（notable F1 / invisible recall / ADE / FDE / mean `U^π`）。
- **Stage-2 warm-start**：`init_encoder_from_stage1(stage2_model, ckpt)` 把 Stage-1 的 `graph_net.*` 权重装进 `WAMUnifiedWorldModel.context_encoder.graph_net`（同 `WAMHeteroGraphNet`，键完全匹配）；脚本 `train_wam_stage2.py --init-from-stage1 <ckpt>`（可配 `--freeze-encoder`）。
- **配置**：`env.wam.stage1.*`（lr/batch/steps/λ/history_window/temporal/head/traj），`hidden_dim`/`route_waypoints` 复用 `env.wam.graph.*`，由 `wam_stage1_configs_from_env` 读成 `(WAMPerceptionConfig, WAMStage1Config)`。

**简化 / 未实现**：occluding（§8.3）未实现 → occ 标签恒 0、`λ_occ=0`；window 逐样本编码（非 batch 向量化）；样本每步一张 request 图（per-vehicle 图属 Stage-2 侧 deferred）；早期 step 的 window 长度 < K+1（模型对任意 L≥1 鲁棒）。

---

### §16.2：Stage-2 训练 Pipeline（BS-centric Diffusion UWM）

文件：[car_dreamer/toolkit/wam/stage2.py](car_dreamer/toolkit/wam/stage2.py) +
[car_dreamer/toolkit/wam/flow_recorder.py](car_dreamer/toolkit/wam/flow_recorder.py) +
[scripts/record_wam_flow_data.py](scripts/record_wam_flow_data.py) +
[scripts/train_wam_stage2.py](scripts/train_wam_stage2.py)。

目标：最小化 `L`，学习 `p_θ(π_{q,t:t+H-1}, z^bev_{q,t+1:t+H} | C^BS)`。**离线录制 + 离线训练**两段式（录制需 CARLA，训练不需）。

> 详细的设计→代码对照、简化清单与**后续完善逻辑**见专文 [docs/wam_stage2_training.md](docs/wam_stage2_training.md)。

- **数据录制**（`WAMFlowDataRecorder`）：`observe_policy(step, policy)` 每步记 policy；`register(step, graph, candidate_ids, notable_object_ids)` 登记 request 车样本骨架；horizon（=H-1）到期后用 `encode_policy_chunk` 把 `policy(t..t+H-1)` 按样本的 candidate 顺序编码成 chunk，`torch.save` 成 `.pt`。CARLA 抽取在脚本里，recorder 纯逻辑可单测。
- **样本字段**（一条 = request 车 q 在 t）：`vehicle_graphs:[HeteroData]`（v1 = `[request_graph]`）、`request_index`、`notable_object_ids`、`policy_chunk(π_1) [H,M,P]`、`member_mask [M]`、`policy_step_mask [H]`、`bev_history [K+1,C,H,W]`（真实栅格 uint8）、`bev_future [H,C,H,W]`（真实栅格 uint8，训练时 `E_bev` 编码为 `z_1`）、`bev_step_mask [H]`。
- **Dataset / collate**：`WAMFlowDataset`（读 `.pt` 目录或内存 list）+ `collate_flow_samples`（`samples` 保持 list 逐图编码、张量 pad/stack 到 `[B, H, M/Dz]`）。
- **训练循环** `WAMStage2Trainer`：每 batch → `model.condition_tokens_batch(samples)→ (cond, type_ids, mask)` → `model.training_step(cond, …, policy_1, bev_1)`（采样 `s_π/s_z` + 噪声 → forward → `flow_matching_loss`）→ backward + grad-clip + Adam。日志/定期 checkpoint/可选 val；**graph encoder 与 diffusion 联合端到端**（`freeze_encoder` 冻结 `context_encoder`）。
- **配置**：`env.wam.flow.*`（含 `history_window`、`num_register_tokens`；去掉 `max_objects`/`max_bev_vehicles`）+ `env.wam.stage2.*`（`w_bev` 取代 `w_obs`），`hidden_dim`/`route_waypoints` 复用 `env.wam.graph.*`，由 `wam_configs_from_env` 读。

**简化 / 未实现**：
- `bev_future`/`bev_history` 为真实 visibility-aware 栅格；trainer 用共享 `E_bev` 编码（diffusion 目标 detach）+ `D_bev` 重建 loss（§15.4）。`drivable` 通道暂空。
- `vehicle_graphs = [request_graph]`（N=1）；per-vehicle local 图 deferred。
- 逐图编码（Python 循环，非 PyG `Batch`）；policy 真值 `π_1` 来自占位策略（选全部候选），分布单一；未训练 Stage 1 → encoder 由 `L` 联合训练。

---

## 3. 输出说明

### 3.1 环境 `info`（每步 step / reset 返回）

由 `v2v_comm_mixin._wam_info()` 合并进 `info`（来自 `select`/`policy`/图统计）：

| key | 含义 |
| --- | --- |
| `wam_notable_object_ids` | 当前 notable object 的 actor id |
| `wam_visible_notable_object_ids` | 其中 ego 可见的 |
| `wam_invisible_notable_object_ids` | 其中 ego 不可见但协作者可见的 |
| `wam_uncertainty_max` | notable 最大不确定度 |
| `wam_coop_triggered` | 是否触发协同请求 |
| `wam_policy_selected_vehicle_ids` | policy 选中的协作车 |
| `wam_policy_modality_by_vehicle` | 每辆选中车的模态 |
| `wam_graph_num_{vehicle,object,observation}_nodes` | 各类节点数 |
| `wam_graph_num_{veh_obs,obs_obj,coop}_edges` | 各类边数 |

（注：图本身 `HeteroData` 存在 `env._wam_graph`，**不放进 info**，避免序列化问题。）

### 3.2 模型返回（`WAMPerceptionModel.forward(window)`）

`dict`：`z_object [Q,Ht]`、`object_node_ids [Q]`、`object_mask`、
`notable/visible/invisible/occluding_logits` 与 `_prob`、`perception_logits`（dict）、
`traj_mu [Q,H,2]`、`traj_log_var [Q,H,2]`、`labels`（从当前图取的 notable/visible/invisible 监督标签）。

### 3.3 真值

`build_trajectory_targets` / `TrajectoryTargetBuffer` 返回 numpy：`target_xy [Q,H,2]`、`valid_mask [Q,H]`。

### 3.4 BS-centric Diffusion UWM 返回（`WAMFlowMatchingUWM`）

- `WAMBSContextEncoder(vehicle_graphs, request_index, notable_object_ids, bev_history)` → `(cond_tokens [T_c,d], type_ids [T_c])`；`pad_condition_tokens` → `(cond [B,T,d], type_ids [B,T], mask [B,T])`。
- `forward(cond, type_ids, mask, policy_s, bev_s, s_pi, s_z, …)` → `dict`：`u_pi [B,H,M,P]`、`u_bev [B,H,Dz]`（`enable_bev=False` 时为 `None`）。
- `rollout_future_bev` → `Ẑ^bev [B,H,Dz]`；`propose_policies` → `[B,N,H,M,P]`；`inverse_policy_search` → `[B,H,M,P]`；`joint_generate` → `{policy [B,H,M,P], bev [B,H,Dz]}`。
- `flow_matching_loss(pred, target, ...)` → `{total, policy, bev}`。`WAMUnifiedWorldModel.training_step(...)` 同结构。

### 3.5 Stage-2 训练产出

- **数据样本**：`record_wam_flow_data.py` → `data/wam_flow/sample_*.pt`（每个是 §16.2 的样本 dict）。
- **checkpoint**：`train_wam_stage2.py` → `outputs/wam_stage2/stage2_step{N}.pt`（`{model, optimizer, step, graph_config, flow_config}`），
  可被 `WAMStage2Trainer.load_checkpoint` 还原。
- **训练日志**：`[wam-stage2] step=… L=… policy=… bev=…`（`WAMStage2Trainer.train`，按 `log_interval`）。

---

## 4. 文档 / 日志输出

- **运行日志**：logger `car_dreamer.v2v`（`V2V_LOGGER`）在 debug 间隔打印：spawn 协同车、WAM 每步（notable / max_uncertainty / triggered / selected）、通信轮次、消息投递、reset 图信息。图节点/边计数在 `info` 里（也可用下面脚本直接打印）。
- **调试 JSONL + BEV 帧**：[scripts/record_wam_notable_debug.py](scripts/record_wam_notable_debug.py) 输出到 `outputs/wam_notable_debug/`（notable + 预测 vs 真值未来轨迹）；字段说明见 [docs/wam_notable_debug_jsonl.md](docs/wam_notable_debug_jsonl.md)。
- **图统计 / H_t 形状**：[scripts/check_wam_graph.py](scripts/check_wam_graph.py) 每步打印图的节点/边计数、`coop_triggered`、`selected`，加 `--embed` 还会打印 §5+§9 编码后的 `H_t` 形状。
- **接入指南**：[docs/v2v_comm_guide.md](docs/v2v_comm_guide.md)（如何给一个 task 加上 V2V/通信模块）。

---

## 5. 如何验证实现是否正确

### 5.1 离线（不需要 CARLA，推荐）

```bash
# 全套 WAM 单测：runtime / graph / heads / targets / flow-matching / stage2
conda run -n cardreamer_gnn python -m unittest \
  tests.test_wam_runtime tests.test_wam_graph tests.test_wam_heads tests.test_wam_targets \
  tests.test_wam_flow_matching tests.test_wam_stage1 tests.test_wam_stage2 -v

# 全部测试
conda run -n cardreamer_gnn python -m unittest discover -s tests -p "test_*.py"

# 语法 + 注册 + 配置加载
conda run -n cardreamer_gnn python -m py_compile car_dreamer/toolkit/wam/*.py car_dreamer/v2v_comm_mixin.py
conda run -n cardreamer_gnn python -c "import car_dreamer, gymnasium as gym; \
  assert 'CarlaGroupRightTurnAutoEnv-v0' in gym.envs.registry; \
  c=car_dreamer.load_task_configs('carla_group_right_turn_auto'); print(c.env.wam.graph.num_heads)"
```

单测覆盖要点：
- `test_wam_runtime`：notable 选择、可见性→不确定度、阈值触发、placeholder policy。
- `test_wam_graph`：policy 条件下的节点/边计数、state 维度、invisible 标记、ego-only 图合法、HGT forward 形状/有限性、BEV 占位节点可跑。
- `test_wam_heads`：`align_object_history` 对齐 + presence mask、`WAMPerceptionModel` 输出形状/有限性、单图窗口、三类 loss、NLL 在 `μ→target` 时下降、`policy_uncertainty` 与手算公式一致。
- `test_wam_targets`：ego 帧变换 + 旋转 + 缺失 mask、buffer 在 horizon 后正确发射对齐真值。
- `test_wam_flow_matching`：time embedding / `interpolate` 端点+速度；`WAMGraphContextPool` 形状+凸组合+mask 排除+真实图；
  `WAMFlowMatchingUWM.forward` 形状/有限性、`key_padding_mask` 不产生 NaN、`enable_bev=False` 通路；
  `flow_matching_loss` 在 pred==target 时为 0、mask 成员被排除、过拟合小 batch 时 `L_FM` 下降；
  四种推理模式（rollout/proposal/inverse/joint）形状；policy 编解码；`WAMUnifiedWorldModel` 端到端（梯度回传 graph encoder + rollout）。
- `test_wam_stage2`：`build_object_state_target`（位置=GT/速度=有限差分/static 保持/缺失 mask）；`WAMFlowDataset`+collate 形状与 pad；
  trainer 单 batch 有限、过拟合时 `L_FM` 下降、checkpoint 存取还原 loss、`freeze_encoder` 切换 encoder 梯度；
  `WAMFlowDataRecorder` 纯逻辑（喂假数据 → flush 出带正确 X 真值的样本）；离线端到端训练 + 落 checkpoint。

### 5.2 端到端 smoke（无需 CARLA）

构造 3 图窗口 → `WAMPerceptionModel` → 用 `TrajectoryTargetBuffer` 配真值 →
`perception_loss` / `gaussian_trajectory_nll` / `policy_uncertainty`，确认形状与有限性。
（流程见 §3.2/§3.3 与 `tests/test_wam_heads.py`。）

### 5.3 在线（需要 CARLA 在对应端口）

```bash
# 每步图统计随协同触发变化；--embed 额外打印 H_t 形状
python scripts/check_wam_graph.py --task carla_group_right_turn_auto --carla-port 2000 --steps 200 --print-every 10 --embed
```
期望：`triggered=True` 时出现协作车节点 + `coop_edges≥1`；`triggered=False` 时退回 ego-only（`coop_edges=0`、`vehicle_nodes=1`）；`obs_obj_edges` = ego 可见数 + 各协作车可见数；`--embed` 下 `H_t` 三类节点都为 `(N, hidden)`。

```bash
# notable / 运动预测 的预测 vs 真值（JSONL + BEV 帧）
python scripts/record_wam_notable_debug.py --task carla_group_right_turn_auto --carla-port 2000 --steps 300
```

### 5.4 Stage-2 训练（录制需 CARLA；训练不需）

```bash
# 1) 录制训练样本（需 CARLA）
python scripts/record_wam_flow_data.py --task carla_group_right_turn_auto --carla-port 2000 \
  --steps 400 --out-dir data/wam_flow
# 2) 离线训练 flow-matching UWM（不需 CARLA）
python scripts/train_wam_stage2.py --data-dir data/wam_flow --task carla_group_right_turn_auto --steps 2000
```
期望：`data/wam_flow/sample_*.pt` 出现；训练日志 `L_FM` 随 step 下降；`outputs/wam_stage2/stage2_step*.pt` 写出且可被
`WAMStage2Trainer.load_checkpoint` 还原。

---

## 6. 已知约束与简化清单

**一致性约束**
- 图构造的 `GraphBuildSpec.route_waypoints` 必须等于模型 `WAMPerceptionConfig.route_waypoints` / `WAMGraphModelConfig.route_waypoints`（vehicle state 维度依赖它）。环境里两者都来自 `env.wam.graph.route_waypoints`，天然一致。
- edge-aware 编码器（`TransformerConv`+`HeteroConv`）要求每类 node ≥1：图构造对空 object 类型做 pad + mask；`hidden_dim` 须被 `num_heads` 整除。
- `WAMUnifiedWorldModel` 要求 `WAMGraphModelConfig.hidden_dim == WAMFlowMatchingConfig.hidden_dim`（构造时校验），且 `bev_latent_dim` 默认取 `hidden_dim`（§5 BEV 编码输出维度）。
- Stage-2：训练用的 `route_waypoints` / `hidden_dim`（`wam_configs_from_env` 从 `env.wam.graph.*` 读）必须与录制样本里 graph 的一致；`flow.horizon` 必须等于 recorder 的 `samples`（policy chunk 长度 H）。

**主要简化**
- 用 ground-truth + FOV/遮挡几何判可见性，而非真实 detector；`s_det`（obs_obj 边属性）恒 1、`Δt=0`、`quality=1`。
- rule-based 运动预测是匀速 + 固定不确定度（visible 0.2 / invisible 2.0）。
- placeholder policy 选全部候选；`B_t` 均分、`f_t=comm_period`。
- 坐标 ego 帧、yaw 用 cos/sin（设计写原始值，信息等价）；object node state 11 维（`s_det` 已移到 obs_obj 边）。
- BEV 现为真实 visibility-aware 栅格（ego 帧、heading-up、只含可见物体 + ego/route/drivable 通道）；BEV-AE joint+detach 训练；`drivable` 通道暂空、`enable_bev` 可整体关。
- BS 样本 `vehicle_graphs = [request_graph]`（N=1）；policy chunk 真值来自占位策略（选全部候选）。
- diffusion 推理用定步长 Euler 积分（非高阶 solver）。

**当前未实现（后续步骤）**
- 真实 detector confidence（obs_obj `det_confidence` 暂恒 1）；BEV `drivable` 通道接地图、BEV-AE 预训练+冻结。**真实 per-vehicle BEV `B^sem` 已实现**（§5.3/§14/§15.4，见 [docs/wam_bev.md](docs/wam_bev.md)）。
- live-env per-vehicle BS 图装配（全部覆盖车 local 图；需 per-vehicle route/notable）—— 当前 deferred。
- §8.3 occluding object（occ 标签恒 0）。
- §14 BEV decoder（把 `Ẑ^bev` 解码成 semantic map）、§15.4 BEV 重建 loss。
- §16.3 Stage 3（policy search + planner 验证）。**§16.1 Stage 1 已实现**（encoder + heads 预训练 + `U^π` + Stage-2 warm-start）。
- §17 按 `U^π` 的 policy reranking / 候选枚举（当前直接选全部候选）——`U^π` 已可由 Stage-1 训练好的轨迹头给出，待接入 §16.3。
- §10 对 vehicle/observation 节点的时序编码（机制相同，仅 object 已接）；§6.2 BEV-to-object 边；img/text 模态。
