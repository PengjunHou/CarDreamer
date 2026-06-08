# 在一个 Task 中接入 V2V 协同感知通信模块

> **⚠️ 更新（协同车来源已变更）**：协同车辆不再由 `_spawn_cooperative_vehicles` /
> `num_coop_vehs` / `coop_spawn_radius_m` / `group_spawn_points` 生成。现在统一在任务的
> **`scenario_actors.vehicles`** 配置里声明——**任何带 `start` 点的车辆即为 V2V 候选车**
> （自动挂相机、可被 policy 选中）：无 `destination` → 静止;有 `destination` → TM 路由（`target_speed`
> 默认 25）。`ScenarioActorManager` 负责生成，并通过 `cooperative_hook` 调用
> `V2VCommMixin._register_cooperative_candidate` 完成注册;共享传感器配置仍是单个
> `env.group_observation` 块，候选参与概率仍是 `env.coop_participation_prob`。
> 详见 [`car_dreamer/toolkit/scenario_actors.py`](../car_dreamer/toolkit/scenario_actors.py) 与
> [`docs/wam_implementation.md`](wam_implementation.md)。下文中凡涉及 `_spawn_cooperative_vehicles`/
> near-ego 生成器/`group_spawn_points` 的段落均为**历史写法**，仅作背景参考。

本文档说明如何把可复用的 **V2V 协同感知通信模块**（`V2VCommMixin`）接入到任意一个
CarDreamer task 中，使其具备与 `carla_group_right_turn_auto` 一致的协同感知能力：
带相机的协同车辆、按 episode 随机的候选集、逐步的 policy 选择、带时延的 V2V 消息收发，
以及面向 ego 的 GNN 图输出。

- 模块代码：[`car_dreamer/v2v_comm_mixin.py`](../car_dreamer/v2v_comm_mixin.py)
- 参考实现：[`car_dreamer/carla_group_right_turn_auto_env.py`](../car_dreamer/carla_group_right_turn_auto_env.py)
  + [`car_dreamer/right_turn_auto_runtime.py`](../car_dreamer/right_turn_auto_runtime.py)

---

## 1. 模块提供了什么

`V2VCommMixin` 封装了与车辆来源无关的协同感知核心：

| 能力 | 说明 |
|---|---|
| 协同车辆 + 相机 | 每辆协同车配前置 camera（通过 `Observer` + `group_observation` 配置） |
| 随机候选集 | 每个 episode 按 `coop_participation_prob` 随机决定哪些车“参与协同”（防记忆、促泛化） |
| policy 选择 seam | 每个通信步用 `_select_collaborators` 从候选集中选子集（默认全选） |
| V2V 消息 | 带 Shannon/FSPL 时延模型的消息收发 + 延迟投递队列 |
| GNN 图 | 从 ego 收到的消息构造固定尺寸车辆节点图，合并进 `info` |

底层复用 `toolkit/communication`（时延/消息）、`toolkit/group/graph_build.py`（建图）、
`toolkit/communication/feature_extractor.py`（payload）、`toolkit/observer`（相机观测），无需改动。

---

## 2. 前置条件

### 2.1 宿主 env 必须提供（继承 `CarlaWptEnv` / `CarlaWptFixedEnv` 即自带）

- `self.ego`：ego 车辆 actor
- `self.obs`：ego 的观测字典（含相机帧）
- `self._world`：`WorldManager`
- `self._time_step`：当前步
- `self.get_state()`：返回 env state（喂给协同车 observer）
- `self.get_wpt_dist(loc)`：reset 图信息里会用到（`CarlaWptEnv` 自带）

> 因此**建议宿主 env 继承自 `CarlaWptEnv` 或 `CarlaWptFixedEnv`**，上述方法都现成。

### 2.2 任务配置必须/可选提供（写在 `tasks.yaml` 对应 task 下）

```yaml
  # 必填：协同车的相机/碰撞观测（前置相机已是默认朝向 x=1.5,z=2.0）
  env.group_observation.enabled: [camera, collision]
  env.group_observation.camera.fov: 120        # 可调

  # 协同车辆来源（默认 near-ego 生成器使用）
  env.num_coop_vehs: 3                          # 在 ego 附近生成几辆协同车
  env.coop_spawn_radius_m: 50.0                # 生成半径（米）
  env.coop_participation_prob: 0.5             # 每辆车参与协同的概率

  # 特征 & 图（GNN 输入尺寸）
  env.feature_size: 3072
  env.graph.window_s: 2.0
  env.graph.tmax: 20
  env.graph.max_nodes: 3
  env.graph.feat_dim_max: 3072
  env.graph.star_graph: true
```

> `env.communication.*`（时延模型参数：`comm_period`、`uplink_bps` 等）已在
> [`common.yaml`](../car_dreamer/configs/common.yaml) 给了全局默认值，**通常不用写**，需要时再覆盖。

---

## 3. 五步接入

在你的 env 类里做以下 5 处改动（与 `auto` env 完全一致）：

```python
from .carla_wpt_fixed_env import CarlaWptFixedEnv      # 或 carla_wpt_env
from .v2v_comm_mixin import V2VCommMixin


class CarlaXxxEnv(V2VCommMixin, CarlaWptFixedEnv):       # 1) 继承 mixin（放最左）

    def __init__(self, config):
        super().__init__(config)
        self._init_v2v()                                 # 2) 初始化通信/图/候选状态

    def on_reset(self):
        self._reset_group_runtime_state()                # 清空缓冲/候选/消息
        self._destroy_group_observers()                  # 销毁上一局相机 observer
        super().on_reset()                               # 生成 ego（及该 task 的车流等）
        self._spawn_cooperative_vehicles()               # 3) 生成协同车（默认 near-ego）
        self._refresh_actor_cache()

    def on_step(self):
        self._deliver_messages()                         # 4) 投递到期消息
        self._update_group_observations()                #    刷新参与者相机观测
        if self._time_step % max(self.comm_period, 1) == 0:
            self._run_group_communication()              #    跑一轮 V2V 通信
        super().on_step()

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        info = self._merge_step_info(info, requested_action=action)   # 5) 图信息进 info
        return self.obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info = self._build_reset_info()                  # 5) reset 图信息
        return self.obs, info
```

然后在 `tasks.yaml` 给该 task 加上 §2.2 的配置块，并确保 `env.name` 与类名匹配
（env 由文件名 `*_env.py` 自动注册，类名 = 文件名转驼峰）。

完成。其余（时延、消息队列、建图）模块内部自动处理。

---

## 4. 协同车辆来源：默认 near-ego，可覆盖

### 4.1 默认行为（无需写代码）

`_spawn_cooperative_vehicles` 默认在 **ego 附近**生成 `num_coop_vehs` 辆
**自动驾驶（移动）**的协同车：在半径 `coop_spawn_radius_m` 内的地图 spawn point 里挑点，
调用 `WorldManager.spawn_auto_actors(...)` 生成，给每辆挂相机 observer，并按
`coop_participation_prob` 随机决定是否参与。

### 4.2 自定义来源（覆盖一个方法）

如果你想用**固定 spawn point**（像 `auto` task 那样的静止协同车）或别的来源，
覆盖 `_spawn_cooperative_vehicles` 即可，其它通信/建图逻辑全部复用。参考
[`right_turn_auto_runtime.py`](../car_dreamer/right_turn_auto_runtime.py) 的写法：

```python
from .v2v_comm_mixin import GROUP_ID, V2VCommMixin
import carla, numpy as np

class CarlaXxxEnv(V2VCommMixin, CarlaWptFixedEnv):
    def _spawn_cooperative_vehicles(self):
        self.groups.setdefault(GROUP_ID, set())
        self.groups[GROUP_ID].add(int(self.ego.id))           # ego 永远是中心
        for sp in self._config.group_spawn_points[: self.num_group_vehs]:
            tf = carla.Transform(carla.Location(*sp[:3]), carla.Rotation(yaw=sp[3]))
            veh = self._world.spawn_actor(transform=tf)
            self._create_group_observer(veh)                  # 挂相机
            self.group_vehs.append(veh)
            self._cache_actor(veh)
            if np.random.random() < self.coop_participation_prob:
                self.coop_participant_ids.add(int(veh.id))    # 进候选集
                self.groups[GROUP_ID].add(int(veh.id))
```

> 约定：**所有协同车都挂相机**（调 `self._create_group_observer(veh)` 并 append 到
> `self.group_vehs`）；只有**抽中参与**的车才加入 `self.coop_participant_ids` 和
> `self.groups[GROUP_ID]`。

---

## 5. 候选集、policy 选择与接 policy

两层结构：

| 层 | 变量 | 何时确定 |
|---|---|---|
| 候选集 | `self.coop_participant_ids`（set） | episode 开始，随机 |
| 选中集 | `self.selected_collaborators`（set） | 每个通信步，由 policy 决定 |

每个通信步，`_run_group_communication` 会调用：

```python
selected = self._select_collaborators(candidate_ids) & candidate_ids
```

**默认 `_select_collaborators` 返回全部候选**（保持“候选即通信”行为）。接你的 policy 时，
在 env 子类里覆盖它即可，通信/建图其它部分都不用动：

```python
def _select_collaborators(self, candidate_ids: set) -> set:
    # 例：选离 ego 最近的 K 个；之后换成你的 WAM Flow-Matching policy
    ranked = sorted(candidate_ids,
                    key=lambda vid: _dist_m(self.ego, self._get_group_member_actor(vid)))
    return set(ranked[: self.max_collaborators])
```

- 通信只在 `ego ∪ selected` 之间进行；
- 选择频率 = `comm_period`（对应 WAM 的 sharing frequency `f_t`）；
- 通过 `self` 可访问一切：`self.ego`、`self.group_obs`、`self._build_graph_info()`、距离等。

---

## 6. 输出：GNN 图

`_merge_step_info` / `_build_reset_info` 会把图构建结果合并进 `step()` / `reset()` 返回的
`info` 字典，键包括：`x_seq`、`lengths`、`edge_index`、`edge_attr`、`node_ids`、
`node_mask`、`ego_index`（固定尺寸 `[Nmax, Tmax, F]`，详见
[`graph_build.py`](../car_dreamer/toolkit/group/graph_build.py)）。节点 = ego + 当前时间窗内向
ego 发过消息的协同车（即候选/选中车）。

---

## 7. 完整最小示例

```python
# car_dreamer/carla_xxx_v2v_env.py
from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .v2v_comm_mixin import V2VCommMixin


class CarlaXxxV2VEnv(V2VCommMixin, CarlaWptFixedEnv):
    """某 task + V2V 协同感知（near-ego 协同车）。"""

    def __init__(self, config):
        super().__init__(config)
        self._init_v2v()

    def on_reset(self):
        self._reset_group_runtime_state()
        self._destroy_group_observers()
        super().on_reset()
        self._spawn_cooperative_vehicles()
        self._refresh_actor_cache()

    def on_step(self):
        self._deliver_messages()
        self._update_group_observations()
        if self._time_step % max(self.comm_period, 1) == 0:
            self._run_group_communication()
        super().on_step()

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        info = self._merge_step_info(info, requested_action=action)
        return self.obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.obs, self._build_reset_info()
```

```yaml
# tasks.yaml
carla_xxx_v2v:
  env:
    name: CarlaXxxV2VEnv-v0
    observation.enabled: [camera, collision, birdeye_wpt]
    <<: *carla_wpt
    # ...该 task 自己的 lane_start_point / ego_path / flow 等...
    group_observation.enabled: [camera, collision]
    num_coop_vehs: 3
    coop_spawn_radius_m: 50.0
    coop_participation_prob: 0.5
    feature_size: 3072
    graph:
      window_s: 2.0
      tmax: 20
      max_nodes: 3
      feat_dim_max: 3072
      star_graph: true
```

---

## 8. 验证

```bash
# 直接跑环境观察（不训练）。标准 task 建议加 --autopilot 让 ego 动起来
python scripts/run_env.py --task carla_xxx_v2v --autopilot --episodes 1
```

预期日志：
- `Spawned cooperative vehicles requested=.. spawned=.. participants=[..]`（near-ego 默认生成器）
  或 `Generated group vehicles ...`（固定点覆盖版）；
- `Group communication run step=.. candidates=[..] selected=[..] ...`；
- reset/step 的 `info` 中包含图键（`node_mask`、`edge_index` 等）。

---

## 9. 注意事项

- **协同车相机有 GPU 开销**：每辆协同车一个相机 sensor，`num_coop_vehs` 调大要注意性能。
- **可能 0 个参与者**：`coop_participation_prob` 较小或车少时，某些 episode 没有协作者
  （ego 单干），这是有效的“无外援”场景。需要保证至少 1 个参与者可在覆盖里自行兜底。
- **随机性未绑 env seed**：参与/选择用 `np.random`，每次 run 随机（与现状一致）；要可复现
  需引入 seeded RNG。
- **flow 车不带相机**：`flow_spawn_point` 冒出来的背景车流不属于协同车，不挂相机、不进候选集。
- **图信息默认不进 obs/reward**：只放在 `info` 里，供你的 policy/训练栈读取。
