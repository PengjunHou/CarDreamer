# -----------------------------
# Grouping Strategy Interface
# -----------------------------
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import random
import carla
from ..utils import _dist_m

class GroupingStrategy:
    """
    Implement your own grouping policy by subclassing this.

    Return format:
        groups: Mapping[group_id -> sequence of vehicle actor ids]
    """

    def form_groups(
        self,
        focus_vehicles: Sequence[carla.Actor],
        world,
        time_step: int,
        rng: random.Random,
        **kwargs,
    ) -> Mapping[int, Sequence[int]]:
        raise NotImplementedError


class AllInOneGroup(GroupingStrategy):
    """All focus vehicles in a single group (group_id=0)."""

    def form_groups(
        self,
        focus_vehicles: Sequence[carla.Actor],
        world,
        time_step: int,
        rng: random.Random,
        **kwargs,
    ) -> Mapping[int, Sequence[int]]:
        return {0: [v.id for v in focus_vehicles]}


class NearestNeighborsGrouping(GroupingStrategy):
    """
    Greedy nearest-neighbor grouping with constraints:
    - max_group_size
    - max_pair_distance_m: vehicles farther than this won't be grouped together
    """

    def __init__(self, max_group_size: int = 3, max_pair_distance_m: float = 60.0):
        self.max_group_size = int(max_group_size)
        self.max_pair_distance_m = float(max_pair_distance_m)

    def form_groups(
        self,
        focus_vehicles: Sequence[carla.Actor],
        world,
        time_step: int,
        rng: random.Random,
        **kwargs,
    ) -> Mapping[int, Sequence[int]]:
        vehicles = list(focus_vehicles)
        rng.shuffle(vehicles)

        unassigned = set(v.id for v in vehicles)
        id_to_actor = {v.id: v for v in vehicles}

        groups: Dict[int, List[int]] = {}
        gid = 0

        while unassigned:
            seed_id = next(iter(unassigned))
            unassigned.remove(seed_id)
            group = [seed_id]

            while len(group) < self.max_group_size and unassigned:
                # pick the closest candidate to current group (min distance to any member)
                best_cand = None
                best_dist = float("inf")
                for cand_id in list(unassigned):
                    cand = id_to_actor[cand_id]
                    d = min(_dist_m(cand, id_to_actor[mid]) for mid in group)
                    if d < best_dist:
                        best_dist = d
                        best_cand = cand_id
                if best_cand is None or best_dist > self.max_pair_distance_m:
                    break
                unassigned.remove(best_cand)
                group.append(best_cand)

            groups[gid] = group
            gid += 1

        return groups


class SpawnNearEgoGrouping(GroupingStrategy):
    """
    “组队 + 生成”策略（按你的描述）：

    - 在 ego 周围 max_pair_distance_m 范围内，从地图 spawn points 里选最近的点
    - 生成 (max_group_size - 1) 辆车（如果已有 focus 车不足，则补齐）
    - 把新生成的车 append 进 world.focus_vehicles
    - 最后返回：group_id=0 的一个 group（ego + focus vehicles），你也可以改成只返回 focus 的 group

    适用场景：
    - 你希望 focus 车在 reset 时就生成在 ego 附近（而不是全图随机）
    - 并且把这些车视为“一个组”用于后续通信/协同感知
    """

    def __init__(self, max_group_size: int = 5, max_pair_distance_m: float = 100.0, spawn_max_tries: int = 80):
        self.max_group_size = int(max_group_size)
        self.max_pair_distance_m = float(max_pair_distance_m)
        self.spawn_max_tries = int(spawn_max_tries)

    def form_groups(
        self,
        focus_vehicles: Sequence[carla.Actor],
        world,
        time_step: int,
        rng: random.Random,
        **kwargs,
    ) -> Mapping[int, Sequence[int]]:
        # ---- 0) 需要 world 里有 ego & world manager ----
        ego = getattr(world, "ego", None)
        if ego is None:
            # ego 还没生成，直接按现有 focus 分组
            print("[SpawnNearEgoGrouping] Warning: world has no attribute `ego` yet, fallback to grouping existing focus vehicles.")
            return {0: [v.id for v in focus_vehicles]}

        # ---- 1) 计算目标要有多少 focus ----
        target_focus = max(self.max_group_size - 1, 0)

        # focus_vehicles 传进来可能是 world.focus_vehicles 的一个视图
        # 我们用 world.focus_vehicles 作为真实容器来 append 新车（符合你的要求）
        if not hasattr(world, "focus_vehicles"):
            raise AttributeError("world must have attribute `focus_vehicles` (a list) to append spawned vehicles.")
        
        need = max(target_focus, 0)
        if need <= 0:
            # 已经够了：直接返回组
            return {0: [ego.id] + [v.id for v in world.focus_vehicles][:target_focus]}

        # ---- 2) 获取 spawn points，筛选 ego 附近的 ----
        # 依赖你 env 里 world manager 的字段：world._world._map / world._world._world / try_spawn_actor
        wm = getattr(world, "_world", None)
        if wm is None:
            raise AttributeError("world must have attribute `_world` (WorldManager).")

        carla_map = getattr(wm, "_map", None)
        if carla_map is None:
            raise AttributeError("world._world must have attribute `_map` (carla.Map).")

        spawn_points = list(carla_map.get_spawn_points())
        ego_loc = ego.get_transform().location

        def dist_to_ego(sp: carla.Transform) -> float:
            dx = sp.location.x - ego_loc.x
            dy = sp.location.y - ego_loc.y
            dz = sp.location.z - ego_loc.z
            return float((dx * dx + dy * dy + dz * dz) ** 0.5)

        candidates = [(dist_to_ego(sp), sp) for sp in spawn_points]
        # 只要范围内的
        candidates = [(d, sp) for (d, sp) in candidates if d <= self.max_pair_distance_m]
        # 若附近没有 spawn 点，则退化为全图最近点（否则你永远生成不了）
        if not candidates:
            candidates = [(dist_to_ego(sp), sp) for sp in spawn_points]

        # 按距离升序（最近优先）
        candidates.sort(key=lambda x: x[0])

        # ---- 3) 逐个尝试生成 need 辆车 ----
        # 为了避免“占用/碰撞导致 spawn 失败”，我们允许向后滑动候选点并尝试多次。
        spawned = 0
        cand_idx = 0
        total_cands = len(candidates)

        while spawned < need and cand_idx < total_cands:
            _, sp = candidates[cand_idx]
            cand_idx += 1

            v = None
            # 对同一个点不重复试太多次（意义不大），但给一个轻量 fallback
            for _ in range(min(self.spawn_max_tries, 3)):
                v = wm.try_spawn_actor(transform=sp)
                if v is not None:
                    break

            if v is not None:
                world.focus_vehicles.append(v)
                spawned += 1

        # ---- 4) 返回 group（默认把 ego + 这些 focus 放一个组）----
        # 如果你不希望 ego 在 group 里，把 ego.id 去掉即可。
        members = [ego.id] + [v.id for v in world.focus_vehicles][:target_focus]
        return {0: members}