# Sub-action a=(S,B,D,n) 数据收集（V2X 论文 Sec I.B ↔ 代码映射）

同一 episode 内真实执行多种 action：每个 policy 到期时从动作网格均匀采样一个新的
sub-action a=(S,B,D,n) 并装入 `CommunicationProcess`（真实通信队列/延迟随之演化），
同时把执行的 action 与每 slot 通信原语写进 Stage-1 样本 metadata，使 (P2) 式 (41)
的各项可以事后直接从数据计算。

## 设计 ↔ 代码

| 论文概念 | 代码 |
|---|---|
| slot（时隙） | ≡ 1 环境步 = `fixed_dt` = 0.1s（用户设定） |
| S（选择的协作者，\|S^(j)\|≤1） | `random_policy.collaborator_counts: ["0","1"]`；`_random_policy_collaborator_count`（`v2v_comm_mixin.py`），"0" → local-only |
| B（带宽比例） | `random_policy.bandwidth_ratios: [0.2,0.5,0.8,1.0]`，每个 sub-action 采一个 |
| D（模态） | `random_policy.modalities`：objlist / bev / 双模态 |
| n（持续 slot 数） | `random_policy.duration_grid: [10..100]`（环境步，1s~10s）；`_sample_random_policy_duration()` 均匀采样 → `CommPolicy.duration_steps` |
| sub-action 序列（= 执行中的 chunk） | `_update_policy_lifecycle` random_duration 分支：到期（`end_step = start_step + n`）即重采样安装 |
| L_m（每 slot 到达 bits） | `CommunicationProcess.comm_stats()["arrival_bits"]`（`toolkit/communication/process.py`） |
| R_m（链路速率）/ R_m·Ts | `comm_stats()["rate_bps"]` / `["service_bits"]`（rate × dt）；速率在 `generate()` 时也存进 `V2VMessage.rate_bps` |
| Q_m（发送 backlog） | `comm_stats()["backlog_bits"]`（in-flight payload bits）+ `["queue_busy_s"]` |
| 样本中的执行 action | `WAMStage1DataRecorder` metadata：`active_subaction`（预测步生效的 {policy_id, S, B, D, n, start_step, slot_offset}）、`slot_subactions`（窗口每 slot，可跨切换边界）、`slot_comm_stats`（与 `window_steps` 逐 index 对齐） |
| 序列化 | `subaction_from_policy(policy, at_step=...)`（`toolkit/wam/stage1_recorder.py`，已从 `toolkit.wam` 导出） |

数据流：`v2v_comm_mixin._wam_comm_slot_stats(step)`（算成员距离 → `proc.comm_stats`）→
`scripts/record_wam_stage1_data.py::_comm_snapshot` → `recorder.observe_messages(step, msgs,
active_policy_id=, active_policy=, comm_stats=)` → `_emit` 写入 metadata。

## 运行

```bash
# 录制（live CARLA）：chunk 语义的多 action episode
python scripts/record_wam_stage1_data.py --task carla_group_right_turn_auto \
  --policy-sampler random_duration --steps 4000 --out-dir data/wam_stage1_chunk

# 训练管线不变
python scripts/train_wam_stage1.py --data-dir data/wam_stage1_chunk --task carla_group_right_turn_auto

# 测试（离线）
python -m unittest tests.test_wam_stage1_chunk_sampler -v
```

## 简化与约定（与严格论文语义的差距）

1. **切换即清队列**：`set_policy` 默认（`flush_old_policy_queue: False`，命名反直觉——False
   触发 §9.1 丢弃分支）丢弃旧 policy 在途消息并重置发送队列 → 每次 sub-action 边界
   `backlog_bits` 归零。若 (P2) 评估需要 Q_m 跨 sub-action 连续，录制时配
   `--env.communication.flush_old_policy_queue=True`。
2. **一步快照偏移**：录制脚本在循环顶部快照通信状态，t 步内安装的新 policy 在 t+1 才进
   日志（与 `_active_policy_by_step` / slot 消息过滤同一约定）；边界步的 `slot_offset`
   可能等于 n。
3. **到达粒度**：L_m 只在 sensor tick 非零（`sensor_period_s: 0.2` → 每 2 环境步）。若要
   每 slot 都可能产生消息，录制时配 `communication.sensor_period_s: 0.1`。
4. **local_prob 与 "0" 格点**：local-only 主开关是 `collaborator_counts` 里的 "0"（均匀
   格点）；`local_prob` 保留为额外过采样权重，默认 0 避免双重计数。
5. `policy_duration_s`（Td）在 random_duration 模式仅作 duration_grid 为空时的回退；
   request_all / stage2 / lyapunov 模式不受影响。
6. 每个 sub-action 内 B 对所有被选成员同值（当前 |S|≤1 下无区别）。

## 后续计划

- 用新数据按 (P2) 做数据驱动的最优 action 复算：按 `active_subaction` 分组统计每 slot
  U_hat（Stage-1 前向），配合 `slot_comm_stats` 的 L_m / R_m·Ts 与离线演化的 Q_m、Z
  （`toolkit/wam/lyapunov.py`），逐决策点求 argmin 并与 `lyapunov_scheduler` 的模型驱动
  选择对比。
- 网格覆盖检查脚本：按 (|S|, B, D, n) 分桶统计样本量，确认无空洞后再上量录制。
- 若冷启动段（小 `slot_offset`）样本占比不足，可缩短 duration_grid 上限或加权采样。

## 测试

`tests/test_wam_stage1_chunk_sampler.py`（9 例，全离线）：网格采样与多样性、local-only
三条路径的 n 采样、|S|≤1、到期切换（变长 n 无缝衔接）、跨边界 metadata round-trip、
comm_stats 原语（到达/服务/backlog/速率回退/local-only 空）、slot 对齐与向后兼容
（旧调用方式与旧样本照常工作）。`tests/test_wam_runtime.py` 两处 stub 固定
`duration_grid=(5,)` 保持原断言。
