# WAM Stage-1:窗口并集预测(预测协作者可见但 ego 不可见的对象)

记录把 Stage-1 的"预测/监督对象集"从**仅最后一帧**改为**整个图窗口的并集**的改动:
设计动机 → 代码落点 → 简化 → 后续计划。

## 设计动机

Stage-1 在 `t` 预测时构建按 `t_sense` 对齐的图窗口 `{t, t-Δ, …}`,每个 slot 的图只用
"感知时刻 == 该 slot 且 ego 已收到、未失效"的数据(`_messages_for_slot`,因果 + 时间对齐,保留不变)。

由于 V2V 通信延迟,**最后一帧(slot=t)必然拿不到协作者在 `t` 的数据 → 只能是 ego-only 图**。
原实现的预测/监督对象集 = 最后一帧的有效 object 节点([heads.py](car_dreamer/toolkit/wam/heads.py)),
导致协作者看得到、ego 看不到的 invisible 对象(它们作为真实节点存在于较早的协同帧里)被静默丢弃,
协同感知对运动预测失去意义。

修复:**对象集 = 窗口所有帧有效 object 节点的并集**。这些对象在较早帧已有真实 embedding,
`align_object_history` + GRU 凭其历史 + presence mask 即可产出预测;最后一帧 ego-only 不变。

## 设计 → 代码

| 设计点 | 代码落点 | 说明 |
| --- | --- | --- |
| query 集 = 全窗口并集 | [heads.py](car_dreamer/toolkit/wam/heads.py) `WAMPerceptionModel.forward` | 逐帧用 `node_id≥0 & node_mask>0.5` 过滤,`torch.unique(cat(...))`(升序去重)→ `query_ids`;`object_mask` 改为全 True |
| 标签解耦(推理用) | 同上 `forward` 的 `labels` 块 | `out["labels"]` 改为对 `query_ids` 逐 id、从最新含该对象的帧回填的 best-effort 版;仅供 live 推理/调试,不参与训练 |
| 录制对象集 = 并集 | [stage1_recorder.py](car_dreamer/toolkit/wam/stage1_recorder.py) `union_object_ids` + `_emit` | 新增 `union_object_ids(graphs)`(与 heads 同口径:升序去重);`_emit` 的 `object_node_ids` 改用它 |
| 标签 = `t` 时刻真值 | `_perception_labels_at_t` | 取窗口**最后一个 source state** 的 `live_states`(`visible_to_ego`/`visible_to_collaborators`)+ `notable_ids`,按 [runtime.select_notable_objects](car_dreamer/toolkit/wam/runtime.py) 逻辑生成 `notable/visible/invisible`;写入 sample `perception_labels` |
| sample 带标签字段 | [stage1.py](car_dreamer/toolkit/wam/stage1.py) `make_stage1_sample` | 新增可选 `perception_labels`(键 `notable/visible/invisible`,长度=Q,行序对齐 `object_node_ids`) |
| 训练用 `t` 标签 | `_align_labels` + `WAMStage1Trainer._gt_labels` / `_sample_loss` / `evaluate` | 有 `perception_labels` → 按 id 对齐后喂 `perception_loss` 与 NLL 的 `notable_weight`;无则回退 `out["labels"]`(旧样本兼容) |
| live 推理 | [v2v_comm_mixin.py](car_dreamer/v2v_comm_mixin.py) `_predict_wam_with_checkpoint` | 无需改:纯用模型输出构造 `MotionPredictionRecord`,并集 query 自动覆盖协作对象 → 提升 `motion_uncertainty`/coop 触发 |
| policy-augmented 录制 | [stage1_policy.py](car_dreamer/toolkit/wam/stage1_policy.py) `WAMStage1PolicyDataRecorder.register`/`_emit` + `evaluate_stage1_uncertainty_rows` | `register` 改用 `union_object_ids` 取本 policy 窗口的并集,并在传入 `live_states`/`notable_ids` 时附 t 时刻标签(`perception_labels_at_t`);评估行优先用录制标签的 notable 加权。[record_wam_stage1_data.py](scripts/record_wam_stage1_data.py) `_register_policy_augmented_slot` 透传 `live_states=objects`/`notable_ids` |

不改:`build_wam_hetero_graph`、按 `t_sense` 对齐与因果过滤、`targets.py`(`build_trajectory_targets` 本就按任意 id 查 GT、`valid_mask` 处理缺席)、`coverage.py`。

## 简化

1. **best-effort 推理标签**:`forward` 的 `out["labels"]` 取"最新含该对象的帧"的节点标签(可能是 `t-Δ` 的属性),
   仅用于 live 调试;训练一律用 recorder 的 `t` 时刻真值,二者不冲突。
2. **并集范围 = 全窗口**,未加"近窗/任务相关"筛选 → 可能纳入已驶离对象;其 GT 未来被 `valid_mask` 屏蔽、
   `t` 标签全 0,不产生有害监督,但会略增计算量。
3. **policy-augmented recorder**([stage1_policy.py](car_dreamer/toolkit/wam/stage1_policy.py))为反事实 per-policy 评估,
   保留**零延迟理想化**建图(它衡量各 policy 的信息价值上界,与延迟无关);**已同步**改成并集 + t 时刻标签
   (调用方未传 `live_states` 时 `perception_labels=None` → 训练回退节点标签,向后兼容)。
4. **pre-built-graph 录制分支**(`register(graph=...)`)无 source state → `perception_labels=None`(回退);
   主 offline 路径走 `register_slot` → `source_window`,标签齐备。

## 后续计划

- 若并集纳入过多无关对象影响训练,加可配筛选(近 N slot / route 走廊内),heads 与 `union_object_ids` 需同口径。
- 把 `t` 标签语义同样接到 policy-augmented recorder(单图评估的标签现仍来自图节点)。
- (此前已决定不做)对象节点 `Δt` 滞后特征:若发现协作快照陈旧导致外推偏差,可重启该项。

## 验证

离线(conda `cardreamer_gnn`,unittest):
- 既有 + 新增:`conda run -n cardreamer_gnn python -m unittest tests.test_wam_heads tests.test_wam_stage1 tests.test_wam_targets`(34 项通过)
  - `test_wam_heads.test_query_set_is_union_over_window_not_last_frame`:最后一帧 ego-only,断言 query=并集、`traj_mu` 覆盖、presence 正确。
  - `test_wam_stage1.test_union_object_set_and_t_time_perception_labels`:协作对象只在早帧 → 仍入并集;标签取 `t` 真值(invisible)。
  - `test_wam_stage1.test_trainer_uses_recorded_t_time_labels` / `..._falls_back_to_node_labels...`:训练用 sample 标签 + 旧样本回退。
- 全量:`python -m unittest discover -s tests`(除 `test_scenario_actors` 中一项**既有**的 `coop_participation_prob` 配置/测试不一致外全过,与本改动无关)。
