# Emulation V1 实现说明与设计理念

## 1. 背景与目标

本次修改的目标，是在当前代码库中实现论文中的 `emulation / prediction` 部分，并且尽量忠实于论文的核心思想，而不是简单复用已有的 `vlm_records*.json` 日志字段拼一个经验性 predictor。

这次实现遵循的核心原则有三点：

1. 节点特征必须对应论文中的 `per-vehicle state`。
2. `query / task` 条件应与车辆基础状态解耦。
3. 整体接口必须支持多场景、多文件、多 episode，而不是只服务于单个 `right_turn` 日志文件。

因此，这一版实现没有继续把 [right_turn_auto_emulation.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/vlm/right_turn_auto_emulation.py) 作为核心，而是新建了一套通用的 `car_dreamer/toolkit/emulation/` 模块。

---

## 2. 本次修改的核心内容

### 2.1 新增通用 emulation 模块

新增目录：

- [car_dreamer/toolkit/emulation/](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation)

其中主要文件包括：

- [schema.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/schema.py)
- [queries.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/queries.py)
- [features.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/features.py)
- [synthetic.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/synthetic.py)
- [adapter_vlm.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/adapter_vlm.py)
- [dataset.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/dataset.py)
- [model.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/model.py)
- [training.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/training.py)
- [__init__.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/__init__.py)

### 2.2 新增训练入口

新增训练脚本：

- [train_emulation.py](/home/peh324/Codes/CarDreamer/train_emulation.py)

该脚本避免直接走 `car_dreamer/__init__.py` 的重依赖导入链，方便在离线环境下直接启动 emulation 训练。

### 2.3 新增测试

新增或补充测试：

- [tests/test_emulation_graph_gru.py](/home/peh324/Codes/CarDreamer/tests/test_emulation_graph_gru.py)
- [tests/test_emulation_training.py](/home/peh324/Codes/CarDreamer/tests/test_emulation_training.py)
- [tests/test_vlm_emulation_targets.py](/home/peh324/Codes/CarDreamer/tests/test_vlm_emulation_targets.py)

---

## 3. 设计理念

## 3.1 为什么不能直接把问题维度塞进 node feature

论文中的 node 表示是 `per-vehicle state`，本质上描述的是每辆协作车在时刻 `t` 的状态，而不是“车辆-问题二元组”的状态。

如果把每个 question 直接展开进 node feature，会导致两个问题：

1. 基础 node state 不再是论文定义的车辆状态，而变成混合后的任务特征。
2. 模型无法自然泛化到新的 query 集合，因为问题维度被硬编码进了输入。

因此，这一版采用：

- `vehicle node state` 只编码车辆本体状态
- `query / task` 作为单独条件输入
- `n_i,t(q)` 作为 query-conditioned 的 task relevance，而不是基础 node feature 的一部分

这也是这次实现里最重要的结构性决策。

## 3.2 为什么引入 canonical schema

当前已有的 `vlm_records*.json` 是“VLM 打分日志”，它的组织方式是：

- `step × question`
- 每条记录中再附带若干 `per_sensor_scores`

这和训练一个图时序预测器所需的数据组织方式并不一致。图模型更需要：

- `step × candidate vehicle`
- 每个 vehicle 拥有统一的状态向量
- 每个 query 拥有统一的 query 表示
- 每个 future horizon 有稳定的监督目标

所以这里先定义了一套 `canonical emulation schema`，让训练主流程依赖统一结构；真实日志只作为 adapter 输入，而不是直接决定训练接口形状。

## 3.3 为什么先支持 synthetic data

用户明确提出，不应被当前日志中字段是否齐全所限制，重点是先实现论文 idea。

因此本次实现中，`synthetic.py` 的意义不是替代真实数据，而是：

1. 先验证 schema 是否合理
2. 先验证 feature 分解是否合理
3. 先验证 dataset 和模型的 shape、mask、监督对齐是否正确

这能避免训练代码一开始就完全被历史日志格式绑死。

---

## 4. 数据表示设计

## 4.1 canonical episode schema

在 [schema.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/schema.py) 中定义了：

- `RegionBox`
- `EgoState`
- `QueryRecord`
- `CandidateVehicleState`
- `CanonicalStepRecord`
- `CanonicalEpisodeRecord`

一个 `CanonicalEpisodeRecord` 表示一个完整 episode，由多个连续 step 组成。

每个 `CanonicalStepRecord` 至少包含：

- `scene_id`
- `episode_id`
- `scene_type`
- `step`
- `dt`
- `ego_state`
- `candidate_vehicles`
- `queries`
- `ego_sc`

## 4.2 per-vehicle state 与论文映射

当前的车辆节点基础状态对应以下分块：

- `x_raw = [delta_pos, delta_vel, delta_yaw]`
- `x_shared = [shared_summary_raw, shared_summary_semantic, shared_confidence, intent_summary]`
- `x_derived = [complementarity, accessibility]`

也就是说，基础 node state 中显式包含：

- `Δp_i,t`
- `Δv_i,t`
- `Δψ_i,t`
- `y_i,t`
- `q_i,t`
- `h_i,t`
- `c_i,t`
- `a_i,t`

而 `n_i,t(q)` 不进入基础 node feature，而是在 dataset 中以独立张量 `task_relevance[K, N, Q]` 输出。

## 4.3 query 表示

在 [queries.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/queries.py) 中，为不同场景定义了默认 query 集合和 `required_region`：

- `right_turn`
- `left_turn`
- `lane_change`
- `car_following`

每个 query 都包含：

- `query_id`
- `query_embedding_input`
- `required_region`

这样模型能在统一 node state 之上，按 query 条件输出不同预测。

---

## 5. 特征构造设计

在 [features.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/features.py) 中实现了几类关键函数：

- `pack_vehicle_node_state`
- `pack_query_features`
- `compute_complementarity`
- `compute_accessibility`
- `compute_task_relevance`
- `build_pairwise_edge_attr`

### 5.1 complementarity

`c_i,t` 通过 sender 可观测区域与 ego 可观测区域的差异近似计算。

当前实现采用离散采样近似：

- 对 sender 的可观测区域采样
- 统计其中多少点不在 ego 可观测区域内
- 以此估计互补观测比例

这是对论文中“额外观测价值”的工程化近似。

### 5.2 accessibility

`a_i,t` 当前按下式构造：

`exp(-lambda_d * distance - lambda_tau * latency)`

这对应论文里“距离 + 通信可达性”共同影响协作价值的思想。

### 5.3 task relevance

`n_i,t(q)` 通过 sender 相比 ego 额外提供的可观测区域与 query 所需区域 `R_need(q)` 的交集比例近似计算。

关键点在于：

- 它依赖具体 query
- 因此不能被当作通用 node attribute
- 必须保留为 query-conditioned 输入

---

## 6. 数据来源与 adapter 设计

## 6.1 synthetic generator

在 [synthetic.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/synthetic.py) 中实现了合成 canonical episode 的逻辑。

它会自动生成：

- 多 step 场景
- 多个候选协作车
- query 集合
- `sender_collab`
- `sender_gain`
- `ego_sc`

作用是先确保训练闭环存在。

## 6.2 真实日志 adapter

在 [adapter_vlm.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/adapter_vlm.py) 中实现了：

- 从 `vlm_records*.json` 读取原始 VLM 打分日志
- 将其重组为 canonical episode

当前 adapter 的主要做法是：

1. 按 `step × question` 重组原始记录
2. 从 `per_sensor_scores` 中恢复每个 sender 的 `pose`
3. 基于跨 step pose 差分估计 sender 与 ego 的速度
4. 由此构造 `delta_pos / delta_vel / delta_yaw`
5. 基于可观测区域计算 `complementarity / task_relevance`
6. 基于距离与 latency 计算 `accessibility`
7. 从 `per_sensor_scores` 汇总 `shared_summary_raw / shared_summary_semantic / shared_confidence`
8. 从日志中的 `confidence_gain` 与 sender 权重近似恢复 `sender_gain`
9. 从 `confidence_with_part2` 或 `ego_plus_shared["confidence"]` 恢复 `ego_sc`

需要注意的是，真实日志并不是天然的 canonical vehicle-state 数据，因此 adapter 中存在工程化近似，这一点是有意为之，因为它本来就是“兼容层”。

---

## 7. Dataset 设计

在 [dataset.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/dataset.py) 中，`CanonicalEmulationDataset` 会把 episode 切成固定窗口样本。

输出的主要张量包括：

- `node_features[K, N, F_node]`
- `component_valid_mask[K, N, F_mask]`
- `node_mask[K, N]`
- `edge_index`
- `edge_attr[K, E, F_edge]`
- `edge_mask[K, E]`
- `query_features[Q, F_query]`
- `query_mask[Q]`
- `task_relevance[K, N, Q]`
- `future_mask[H]`
- `future_node_mask[H, N]`
- `target_sender_collab[H, N, Q]`
- `target_sender_gain[H, N, Q]`
- `target_ego_sc[H, Q]`

这里的关键设计点是：

1. `node_features` 只包含基础车辆状态
2. `task_relevance` 单独输出
3. future supervision 按 horizon 直接对齐
4. `future_node_mask` 用于屏蔽未来不存在的节点

---

## 8. 模型设计

在 [model.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/model.py) 中实现了 `GraphGRUEmulationModel`。

整体结构为：

1. `node_input` 将 node feature 投到 hidden space
2. 多层 graph message passing 编码 step 内图结构
3. `node_temporal GRU` 编码每个节点随时间的演化
4. `global_temporal GRU` 编码图级汇聚状态
5. `query_encoder` 编码 query
6. `node_query_fuser` 输出：
   - `sender_collab`
   - `sender_gain`
7. `global_query_fuser` 输出：
   - `ego_sc`

### 8.1 为什么用 Graph + GRU

原因主要有三点：

1. 节点目标是“每车预测”，天然适合图结构表达。
2. 预测对象是未来 horizon 上的轨迹式演化，天然需要时序建模。
3. 第一版更需要结构清晰、稳定、易验证，而不是一开始就上更复杂的 latent world model。

因此，`Graph encoder + GRU` 是一个很自然的第一版实现。

### 8.2 为什么 direct multi-horizon

当前输出头直接预测未来 `H` 步，而不是自回归 rollout。

这样做的好处是：

1. 训练更稳定
2. 不需要在第一版里处理 exposure bias
3. 更方便对齐监督张量

---

## 9. 损失函数设计

`compute_emulation_loss` 中目前包含三类损失：

- `sender_collab_loss`
- `sender_gain_loss`
- `ego_sc_loss`

默认使用 `Huber loss`，也支持 `MSE`。

mask 组合方式为：

- sender 相关损失使用：
  - `future_mask * future_node_mask * query_mask`
- ego 相关损失使用：
  - `future_mask * query_mask`

这样可以确保：

- padding node 不参与 loss
- 无 future label 的位置不参与 loss
- 无效 query 不参与 loss

---

## 10. 训练代码设计

训练逻辑在 [training.py](/home/peh324/Codes/CarDreamer/car_dreamer/toolkit/emulation/training.py) 中。

主要功能包括：

- `parse_episode_source_spec`
- `load_episode_from_path`
- `load_episodes_from_sources`
- `split_episode_indices`
- `build_dataset_splits`
- `emulation_collate_fn`
- `build_dataloaders`
- `make_model_from_dataset`
- `train_one_epoch`
- `evaluate_emulation_model`
- `save_training_checkpoint`
- `load_training_checkpoint`
- `fit_emulation_model`

### 10.1 训练入口

根目录脚本：

- [train_emulation.py](/home/peh324/Codes/CarDreamer/train_emulation.py)

它的作用是：

- 避免 CARLA 相关顶层依赖影响训练脚本导入
- 直接调用 `training.py` 中的 `main()`

### 10.2 输入数据格式

训练脚本 `--data` 参数支持三种形式：

- `path`
- `path::scene_type`
- `path::scene_type::dt`

例如：

```bash
python train_emulation.py \
  --data data/vlm_records_terminated_step_96.json::right_turn::0.1 \
  --history-len 8 \
  --horizon 5 \
  --batch-size 8 \
  --max-epochs 20 \
  --save-dir logdir/emulation_run
```

多文件训练示例：

```bash
python train_emulation.py \
  --data \
    data/file1.json::right_turn::0.1 \
    data/file2.json::right_turn::0.1 \
    data/file3.json::lane_change::0.1 \
  --history-len 8 \
  --horizon 5 \
  --batch-size 16 \
  --max-epochs 50 \
  --val-ratio 0.2 \
  --save-dir logdir/emulation_multi
```

### 10.3 输出内容

训练目录下会输出：

- `train_config.json`
- `model_config.json`
- `history.jsonl`
- `checkpoint_latest.pt`
- `checkpoint_best.pt`
- `summary.json`

---

## 11. 测试与验证

目前已补充的验证主要包括：

### 11.1 schema / synthetic / adapter / dataset / model

测试文件：

- [tests/test_emulation_graph_gru.py](/home/peh324/Codes/CarDreamer/tests/test_emulation_graph_gru.py)

覆盖内容：

- synthetic canonical episode roundtrip
- 基础 node feature 与 query-conditioned `task_relevance` 分离
- 从 `vlm_records_terminated_step_96.json` 转 canonical episode
- dataset 输出 shape 检查
- model forward 与 loss 的 shape 检查

### 11.2 training utility

测试文件：

- [tests/test_emulation_training.py](/home/peh324/Codes/CarDreamer/tests/test_emulation_training.py)

覆盖内容：

- 数据源 spec 解析
- episode 级 train/val split
- 多 source 加载
- canonical JSON 读取
- 在无 `torch` 环境下的显式错误保护

### 11.3 旧原型兼容

测试文件：

- [tests/test_vlm_emulation_targets.py](/home/peh324/Codes/CarDreamer/tests/test_vlm_emulation_targets.py)

这个测试主要保证之前原型逻辑没有被新的通用实现破坏。

---

## 12. 当前限制

当前这版已经补齐了训练代码，但仍有几个边界需要明确：

### 12.1 当前环境没有 PyTorch

当前开发环境里 `torch` 未安装，所以：

- 训练代码已经补齐
- 训练入口已经补齐
- 相关测试已经补齐
- 但尚未在这台环境里真实跑通一个完整训练 epoch

也就是说，现在是“代码链路完整，依赖未就绪”。

### 12.2 真实日志 adapter 仍然是近似重建

`vlm_records*.json` 不是原生 vehicle-state 数据，因此 adapter 里的部分分量是启发式恢复，例如：

- sender velocity
- intent summary
- sender gain 分摊

这不影响第一版训练管线的完整性，但后续如果你能提供更原生的状态日志，adapter 可以进一步精确化。

### 12.3 当前 graph 结构仍较简单

目前 `dataset.py` 中使用的是 fully connected graph。

这是一种保守且稳定的第一版实现。后续如果需要更贴近交通拓扑，可以进一步加入：

- 距离阈值边
- lane adjacency 边
- relative heading 边属性增强

---

## 13. 为什么这版实现是“论文忠实版 V1”

这版实现之所以可以称为“论文忠实版 V1”，是因为它已经满足了论文思想最关键的结构要求：

1. 节点是 `per-vehicle state`
2. query 与 node state 解耦
3. `n_i,t(q)` 是 query-conditioned 量
4. 图结构与时间结构同时参与预测
5. 预测目标直接覆盖：
   - `sender_collab`
   - `sender_gain`
   - `ego_sc`
6. 接口是多场景、多文件、多 episode 的

它还不是最终版，原因在于：

- adapter 还可以继续变精细
- graph topology 还可以继续增强
- 还没有接进 Dreamer latent world model

但从“论文 idea 已被工程化落地”的角度看，这一版已经完成了主干搭建。

---

## 14. 后续建议

下一步最自然的方向有三个：

1. 在真实多文件数据上跑第一版训练，检查 loss 和过拟合行为。
2. 根据你后续生成的新日志格式，逐步削弱 adapter 中的启发式近似。
3. 如果离线预测效果稳定，再考虑把该 predictor 接入 Dreamer world model 或策略侧决策模块。

如果后续继续开发，建议优先做第 1 步，因为只有先把训练跑起来，后面的结构优化才会更有依据。
