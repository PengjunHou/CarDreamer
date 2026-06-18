# V2V 流式通信过程（对照 `docs/WAM/` 通信模型设计文档）

本文记录把 V2V 通信从「每 `comm_period` 步一次 full-mesh、单次延迟、图由实时真值可见性构建」重写为
**policy 生命周期 + 传感器流 + 发送/接收队列 + 实测延迟驱动 local↔V2V 切换** 的流式通信过程。逐项说明
**设计怎么写 → 代码怎么实现 → 有哪些简化**。

> 与设计文档的一处偏离（用户要求）：**不**为每个 modality 各发一条消息，而是把一辆协作车在当前 policy 下
> 的所有 modality 数据**打包进一条消息**，消息里用不同字段区分类型。

整体流水线：

```
Policy(Td) → MessageGeneration(Ts) → SenderQueue(proc+queue) → Transmission(tx)
          → ReceiveQueue → Window(Tw) → GraphConstruction(local↔V2V)
```

核心代码：CARLA-free 的 [process.py](car_dreamer/toolkit/communication/process.py)（可单测）+ 延迟/速率
模型 [comm.py](car_dreamer/toolkit/communication/comm.py) + CARLA 胶水 [v2v_comm_mixin.py](car_dreamer/v2v_comm_mixin.py)。

---

## 0. 角色与时间参数

| 设计符号 | 含义 | 代码 | 当前配置(`dt=0.1`) |
| --- | --- | --- | --- |
| `q` / `m` | 请求车辆(ego) / 协作者 | `CommunicationProcess.request_vehicle_id` / `CommPolicy.selected_collaborators` | — |
| `Td` | policy 持续时间 (§2.1) | `CommConfig.policy_duration_steps`，config `policy_duration_s` | 2.0s = 20 ticks |
| `Ts` | 传感器采样/流式周期 (§2.2) | `CommConfig.sensor_period_steps`，config `sensor_period_s` | 0.2s = 2 ticks |
| `Tw` | 接收队列预测窗口(基于 sense 时间, §2.3) | `CommConfig.prediction_window_steps`，config `prediction_window_s` | 2.0s = 20 ticks |
| `Ta` | notable motion 预测周期 (§2.4) | `CommConfig.action_period_steps`，config `action_period_s` | 1.0s = 10 ticks |
| `T_proc` | 单条消息处理延迟 | `CommConfig.proc_delay_s`(**保留秒**)，config `proc_delay_s` | 0.05s |

**cadence/window 参数**(Td/Ts/Tw/Ta)以**仿真步(=World tick)**为内部单位,由
`CommConfig.from_seconds(dt=world.fixed_delta_seconds, ...)` 各自 `round` 一次换算。**延迟分量**
`T_proc` 保留**秒**——它要和(连续的)排队、传输延迟相加后**只取一次 round**(见 §3),避免逐项取整误差。

---

## 1. Policy 生命周期（§3, §9）

### 设计
BS 在 `t_start` 生成 `π_{q,k}=(S, B, d)`，有效期 `[t_start, t_start+Td)`；过期后不再产生新消息，除非重新请求。
local-only 也是一种 policy `π_local=(∅,0,∅)`。

### 代码
- [process.py](car_dreamer/toolkit/communication/process.py) `CommPolicy`：带 `policy_id / start_step /
  duration_steps / end_step / selected_collaborators / modalities_by_vehicle(打包多模态) /
  bandwidth_by_vehicle`（每协作者的带宽比例 `[0,1]`，不要求总和为 1）；`active_at(step)` = `start ≤ step < end`；
  `is_local_only` = 无协作者。
  `make_local_policy(...)` 构造 local-only。
- [v2v_comm_mixin.py](car_dreamer/v2v_comm_mixin.py) `_update_policy_lifecycle(step)`（每步、Ta 节流的感知之后）：
  - `env.wam.policy_sampler_mode=request_all`（默认）：协作 policy 在其 `Td` 内**整段保持**（不中途重算）；
    否则有高不确定性请求且有候选 → `_build_coop_policy`(全候选协作、打包 `collaborator_modalities`、
    给每个候选设置 `bandwidth_ratio`)；
    当前是 local-only 且未过期且无请求 → 保持(避免抖动)；否则装新的 local-only。
  - `env.wam.policy_sampler_mode=random_duration`：任意 active policy（含 local-only）在 `Td` 内保持；
    到期后从 `env.wam.random_policy.*` 真实采样一个 local-only 或 cooperative `CommPolicy`，用于多 policy
    Stage-1 数据收集。
  - 转换时 `CommunicationProcess.set_policy(policy, step)` 安装，并 `_sync_policy_views` 把它镜像成
    `WAMPolicy` 视图(供 info/图构建)。

在线 step 顺序是：先 `_deliver_comm_messages(step)` 把已经完成传输的消息放入 receive queue，再
`_update_wam_runtime_state()` 用当前可用消息构建 `G_t` 并滑入 graph window，随后每 `Ta` 选择
`rule` 或 Stage-1 `checkpoint` predictor 重算一次 `build_coop_request`（§2.4）。最后
`_update_policy_lifecycle(step)` 根据 request 安装/保持 policy，并在 sensor tick 通过
`_stream_comm_messages(step)` 发送新消息。

默认 `env.wam.predictor_mode: rule`，用于 bootstrap 数据收集；设置为 `checkpoint` 时，在线 request
触发会加载 `env.wam.predictor_checkpoint` 指向的 Stage-1 `WAMPerceptionModel`，输入最近
`predictor_history_window + 1` 张图组成的 graph window，用 `notable_prob × traj_log_var` 的轨迹不确定性
生成 `CoopRequest`。

---

## 2. 消息生成 + 打包多模态（§2.2, §4, §6 + 用户偏离）

### 设计 / 偏离
协作车在 policy 有效期内每 `Ts` 产生一次数据流式发送。**偏离**:一辆车所有 modality 合并成**一条** message。

### 代码
- [comm.py](car_dreamer/toolkit/communication/comm.py) `V2VMessage`：`modalities: Tuple[str,...]` +
  `data: Dict[str,Any]`（每 modality 一个字段，外加 legacy `feat`/`pose`/`vel`）+ `payload_size` +
  时间线 `t_sense/t_ready/t_send/t_recv` + 延迟分解 `proc/queue/tx/total_latency`。保留
  `created_step/deliver_step/latency_s/payload_bytes/payload` **兼容属性**给旧的车辆节点图与脚本。
- [v2v_comm_mixin.py](car_dreamer/v2v_comm_mixin.py) `_build_sense_snapshot`：按协作车的 `visible_to_collaborators`
  取其可见对象 `ObjectState` 快照；objlist 写 `data["objlist"]`、bev 用 `rasterize_bev` 写 `data["bev"]`；
  `payload_size = Σ 各 modality 字节 + overhead`。`_run_comm_step` 在 sensor tick 调
  `CommunicationProcess.generate(step, snapshots, _link_rate_bps)`。

---

## 3. 发送队列：处理 + 排队 + 传输延迟（§5–§8）

### 设计
`t_ready=t_sense+T_proc`；队列空则 `t_send=t_ready`，否则排队 `t_send=max(t_ready, busy)`；
`tx=D_M/R^π_{m,q}`，`R` 为 policy 分配带宽下的 Shannon 速率；`T^proc+T^tx>Ts` 时产生 backlog。

### 代码
- [process.py](car_dreamer/toolkit/communication/process.py) `SenderQueue`（每条 `m→q` 链路）跟踪
  `busy_until`(**秒**,连续)。`generate` 全程用**连续秒**计算:`t_ready_s=t_sense·dt+T_proc`、
  `t_send_s=max(t_ready_s, busy_until)`、`tx_s=8·payload/R`、`t_recv_s=t_send_s+tx_s`、
  `busy_until=t_recv_s`、`queue_delay=t_send_s-t_ready_s`。
- **「先加和再取一次 round」**:`total_latency = t_recv_s - t_sense·dt = proc+queue+tx`(精确求和),
  仅 `deliver_step = t_sense + round(total_latency/dt)` **取一次整**。`veh_veh` 边权用精确的
  `total_latency`,投递时刻用 `deliver_step`(=消息的 `t_recv` 整数步)。这样不会出现逐项把
  0.04s 处理 + 0.04s 传输各自取整成 0、却把真实的 0.08s 也丢掉的误差。
- 速率 `R`：[comm.py](car_dreamer/toolkit/communication/comm.py) `shannon_rate_bps(d, B, ...)`（FSPL+SNR）。
  mixin 的 `_link_rate_bps` 先把 policy 比例换算成实际带宽 `B^π_{m,q}=policy_bandwidth_hz·ratio`
  再调它。

---

## 4. 接收队列 + 窗口/跨 policy 过滤（§10–§11）

### 设计
`t_recv≤t` 已收；`t-Tw≤t_sense≤t` 新鲜；可选 `policy_id==active`（`allow_cross_policy_messages`）。

### 代码
[process.py](car_dreamer/toolkit/communication/process.py) `ReceiveQueue.available(step, window_steps,
active_policy_id, allow_cross_policy)` 实现这三条；`evict` 丢弃超窗消息。`CommunicationProcess.deliver(step)`
把 `t_recv≤step` 的 in-flight 消息搬进接收队列。`available_messages(step)` 给图构建用。

---

## 5. 旧 policy 队列处理（§9）

`set_policy` 在切换时：`flush_old_policy_queue=False` → 丢弃旧 policy 尚未送达的 in-flight 消息并释放链路
（§9.1）；`=True` → 保留旧消息继续发送，新消息排在其后（§9.2，链路 `busy_until` 延续）。

---

## 6. local↔V2V 图切换 + 实测延迟（§12–§14）

### 设计
当前图由**接收队列实际可用消息**决定:`M_q(t)=∅` → local graph；非空 → V2V graph。`veh_veh` 边用**实测**
`L_M=t_recv-t_sense`（取该协作者窗口内最近一条，§13.2）。

### 代码
[v2v_comm_mixin.py](car_dreamer/v2v_comm_mixin.py) `_build_wam_graph(step)` 重写为消费
`available_messages(step)`：
- ego 节点 + ego **实时本地**可见对象的 objlist 观测；
- 每条可用消息 → 一个协作车节点(位姿取消息 `t_sense` 快照) + **每 modality 一个观测节点**
  （ids/bev 取快照，`latency_s=L_M`，`freshness=exp(-γL_M)`，`sample_age=(step-t_sense)·dt`）；
- 对象节点 = ego 实时可见态 ∪ 各消息携带的快照态（ego 实时态覆盖同 id 的过期快照）；
- 复用已测试的 `build_wam_hetero_graph`，`latency_by_vehicle` 即各协作者最近一条的 `L_M`。
- 无可用消息 → 退化为 ego-only(local) 图。

旧的车辆节点图 [graph_build.py](car_dreamer/toolkit/group/graph_build.py) 也改为消费同一接收队列(env 已按
Tw 过滤)，二次窗口约束改用 **sense 时间** `t_sense`，token 仍走兼容属性 `created_step/latency_s/payload_bytes/
payload["feat"]`。

env [on_step](car_dreamer/carla_group_right_turn_auto_env.py) 顺序:
`_update_group_observations → _update_wam_runtime_state(Ta) → _update_policy_lifecycle → _run_comm_step`。

---

## 7. 有哪些简化

1. **tx 字节口径**:`payload_size` 按 modality 内容估（objlist=`n·OBJECT_STATE_DIM·4`，bev=`C·H·W` 单字节/格）
   + 固定 overhead；不含 LLM `feat`/文本/dict key 字节（沿用原研究假设）。`feat` 仅作旧图的元数据随消息携带。
2. **拓扑请求车辆为中心**:仅协作者→ego，每链路一个 sender queue（去掉原 full-mesh；原本也只消费 ego 的接收）。
3. **感知与预测解耦**:`Ta≠Ts`(当前 Ta=1.0s、Ts=0.2s)。可见性扫描 `_refresh_object_states` **每步**刷新
   (供 ego 自身观测与协作者快照都用当前可见性),notable/预测/请求按 `Ta` 节流。
4. **延迟先加和再取整**:proc/queue/tx 在秒里精确相加,`deliver_step` 只对总和取一次 `round`;
   `busy_until` 保留连续秒以精确建模排队。亚步总延迟可 round 到 0 步(即同 tick 送达),边权仍用精确秒值。
5. **检测置信度/quality=1.0**:`det_confidence`、`quality` 用真值占位,留给真实感知器填。
6. **带宽比例占位**:BS 给每个被选协作者同一个 `bandwidth_ratio`；比例值可全为 1 或全为 0.5，
   不要求求和为 1。实际 Shannon 带宽为 `policy_bandwidth_hz·ratio`。
7. **`per_link` 队列、`local_only_as_policy=True`** 固定为 v1 设置(未做一个协作者服务多请求车辆的共享队列)。

---

## 8. 后续计划

- BS policy 从「全候选 + 固定带宽比例 + 固定 modality」升级为**学习/优化**的协作者与带宽/modality 选择
  （接 UWM 的 policy evaluation）。
- 消息 `feat`/objlist 的真实序列化字节数与压缩，纳入 `payload_size`；区分 uplink/downlink 与多请求车辆共享队列。
- 真实检测器填 `det_confidence`/`quality`，让 `obs_obj` 边权与 `freshness` 反映感知质量。
- 把 §13.2「最近一条」之外的多条窗口消息以时序方式喂入(对接 temporal encoder)。

---

## 9. 测试

[tests/test_communication_process.py](tests/test_communication_process.py)（offline、unittest、CARLA-free）覆盖：
policy 生命周期半开区间/local-only；`from_seconds` 取整;打包多模态(一条消息、`payload_size` 求和)；
发送队列 backlog(`proc+tx>Ts` → `queue_delay>0`、`t_recv` 单调) vs 快链路无排队；**延迟先加和再 round**
(0.04+0.04 各自取整为 0、但总和 0.08 round 成 1 步);`flush_old_policy_queue` True/False;接收队列
Tw(sense 时间)/未送达/跨 policy 过滤;local↔V2V(无消息→空、送达后可用且 `L_M>0`)。

运行:`conda run -n cardreamer_gnn python -m unittest tests.test_communication_process`（14 项）；
全量离线回归 `python -m unittest discover -s tests`（122 项,全过）。
