# WAM 实现说明（对照 `docs/WAM/WAM Design.pdf`）

本文记录 **Graph Flow-Matching Unified World Model (WAM)** 当前已落地的实现，逐节对照设计文档说明：
**对应代码在哪、做了哪些简化、还有哪些没实现、输出是什么、有无文档/日志、以及如何验证**。

> 适用提交范围：rule-based runtime（步骤 1–3）、policy-conditioned 异构图 + 初始 embedding + HGT
> 编码器（§4–§7、§5、§9）、temporal encoder + deterministic task heads（§10、§11）、轨迹真值抽取（§15.2）。

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
| §5 | 三类 node 初始 embedding | ✅ 已实现 |
| §9 | Heterogeneous Graph Transformer (HGT) | ✅ 已实现 |
| §10 | Temporal Encoder | ✅ 已实现（GRU，仅 object 节点） |
| §11.1 | Notable Object Head（+ vis/inv/occ） | ✅ 已实现（occ 标签恒 0） |
| §11.2 | Gaussian Trajectory Head（学习版） | ✅ 模型已实现（未训练） |
| §15.1/§15.2/§11.2 | perception loss / 轨迹 NLL / U^π | ✅ 函数已实现 |
| §15.2 | GT 未来轨迹真值抽取 | ✅ 已实现 |
| §12 | Graph Flow-Matching UWM | ❌ 未实现 |
| §13.1–§13.3 | flow 推理（rollout / proposal / inverse） | ❌ 未实现 |
| §14 | BEV decoder | ❌ 未实现 |
| §16 | 训练 pipeline（Stage 1/2/3） | ❌ 未实现（模块已就绪，缺训练循环 + 数据管线） |
| §17 | policy search（按 U^π reranking） | ❌ 未实现（当前选全部候选） |

---

## 1. 代码结构

```
car_dreamer/toolkit/wam/
├── runtime.py            # 步骤 1-3：notable / 运动预测 / coop request / placeholder policy
├── graph.py              # §4-§7：policy-conditioned 异构图构造（HeteroData）
├── graph_model.py        # §5 初始 embedding + §9 HGT 编码器
├── heads.py              # §10 temporal encoder + §11 task heads + §15/§11.2 loss/reward
├── targets.py            # §15.2 GT 未来轨迹真值抽取
├── debug_recording.py    # rule-based notable/预测 的调试记录器（JSONL）
├── visualization.py      # 调试 BEV 渲染
└── __init__.py           # 统一导出

car_dreamer/v2v_comm_mixin.py   # 把 runtime + 图构造接入仿真环境（每步执行）
car_dreamer/configs/tasks.yaml  # carla_group_right_turn_auto 的 env.wam.* 配置
scripts/check_wam_graph.py          # 在线验证：打印每步图统计 / H_t 形状
scripts/record_wam_notable_debug.py # 在线记录 notable + 预测 vs 真值（JSONL + BEV 帧）
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
- **§5.2 object node state**：`[x,y,z, vx,vy, cos_yaw, sin_yaw, l, w, h, s_det, Δt]`，维度 12；object class 用单独 embedding（不在数值向量内）。
- **§5.3 observation node state**：modality id + 标量 `[payload_kb, latency_s, freshness, quality, sample_age_s]`（维度 5）。模态特征 `z^r` 在 embedding 模块里算（见 §5）。
- **§6 三类 edge**（无 edge attribute）：`(vehicle, veh_obs, observation)`、`(observation, obs_obj, object)`、`(vehicle, coop, vehicle)`。
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
- 真实 per-vehicle BEV 语义图 `B^sem`：见下面 §5 的 `z^bev` 占位。

---

### §5：三类 node 的初始 embedding

文件：[car_dreamer/toolkit/wam/graph_model.py](car_dreamer/toolkit/wam/graph_model.py) → `WAMHeteroGraphEmbedding`。

- 每类 node：`h^0 = MLP_type(state) + e^type + e^agent + e^time (+ e^class for object)`。
  - `MLP_veh / MLP_obj / MLP_obs` 分别编码三类 state；
  - `type` embedding 区分 vehicle / object / observation(objlist) / observation(bev)（共 4 个 type id）；
  - `agent` embedding 按 slot 索引（ego=0，协作车=1…Mmax）；`time` embedding（单步=0）；`class` embedding（object 类别）。
- **objlist 模态特征**（§5.3 `z^objlist = AttentionPooling({MLP_obj(x^obj)})`）：用 `obs_obj` 边，把 observation 节点的 object 邻居的 `MLP_obj` 输出做 **attention pooling**（`softmax` over 边 + scatter）。
- **bev 模态特征** `z^bev = E_bev(B^sem)`：用轻量 CNN，但**输入是全零占位** `B^sem`（真实 per-vehicle BEV 尚未接入）。

**简化 / 未实现**：`z^bev` 是占位（零图过 CNN）；`time` 只有单步（K 步时序在 §10 处理）；img/text 模态未做。

---

### §9：Heterogeneous Graph Transformer

文件：[car_dreamer/toolkit/wam/graph_model.py](car_dreamer/toolkit/wam/graph_model.py) → `WAMHeteroGraphEncoder` / `WAMHeteroGraphNet`。

- 用 PyG `HGTConv` 堆叠（默认 hidden=256、layers=3、heads=8）+ 残差 + LayerNorm。relation-specific 投影与 relation bias 由 HGTConv 提供，对应设计的 `α^r_{ij}` 与 `W_r`。
- 无入边的 node type（如无协同时的 vehicle）自动 carry-forward。
- 输出 `H_t = {h^veh, h^obs, h^obj}`（每类 `[N, hidden]`）。
- `WAMHeteroGraphNet.forward(HeteroData) → H_t`（先 embedding 再 encoder）。

**简化**：与设计基本一致。HGTConv 自带 type/relation 参数化，未额外引入 edge attribute（设计本就说 edge 不带复杂属性）。

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
# 全套 WAM 单测：runtime / graph / heads / targets / debug
conda run -n cardreamer_gnn python -m unittest \
  tests.test_wam_runtime tests.test_wam_graph tests.test_wam_heads tests.test_wam_targets -v

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

---

## 6. 已知约束与简化清单

**一致性约束**
- 图构造的 `GraphBuildSpec.route_waypoints` 必须等于模型 `WAMPerceptionConfig.route_waypoints` / `WAMGraphModelConfig.route_waypoints`（vehicle state 维度依赖它）。环境里两者都来自 `env.wam.graph.route_waypoints`，天然一致。
- `HGTConv` 要求每类 node ≥1：图构造对空 object 类型做 pad + mask。

**主要简化**
- 用 ground-truth + FOV/遮挡几何判可见性，而非真实 detector；`s_det=1`、`Δt=0`、`quality=1`。
- rule-based 运动预测是匀速 + 固定不确定度（visible 0.2 / invisible 2.0）。
- placeholder policy 选全部候选；`B_t` 均分、`f_t=comm_period`。
- 坐标 ego 帧、yaw 用 cos/sin（设计写原始值，信息等价）。
- 轨迹 NLL 默认用加权平均（设计是求和）。
- `z^bev` 为零占位（无真实 per-vehicle BEV）。

**当前未实现（后续步骤）**
- §8.3 occluding object（occ 标签恒 0）。
- §12 Graph Flow-Matching UWM、§13 flow 推理（rollout/proposal/inverse）、§14 BEV decoder。
- §16 训练 pipeline（Stage 1 预训练 graph encoder + heads；Stage 2 UWM；Stage 3 policy search 验证）与 replay 数据管线。
- §17 按 `U^π` 的 policy reranking / 候选枚举（当前直接选全部候选）。
- §10 对 vehicle/observation 节点的时序编码（机制相同，仅 object 已接）。
- §6.2 BEV-to-object 边；img/text 模态。
