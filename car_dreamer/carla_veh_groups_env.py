# carla_veh_groups.py
"""
CarlaVehGroupsEnv: a multi-vehicle cooperative perception simulation environment.

Key features
------------
1) Flexible grouping strategy interface
   - Swap different grouping policies easily for comparison.

2) Vehicle-to-vehicle communication with latency
   - Latency depends on distance, payload size, and network resource contention.
   - Messages are delivered in future timesteps via an in-flight queue.

This env is intentionally "minimal but extensible":
- It spawns:
    * 1 ego (RL-controlled)
    * N focus vehicles (the vehicles you care about; default autopilot)
    * M background vehicles (Traffic Manager autopilot)
- It runs:
    * periodic grouping
    * periodic intra-group communication
- It exposes:
    * groups / veh_to_group
    * per-vehicle received message buffers
so you can implement observation fusion in a Handler (recommended).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import torch
import carla
import numpy as np

# from .carla_base_env import CarlaBaseEnv
# from .carla_wpt_env import CarlaWptEnv
from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import _dist_m, GroupingStrategy, AllInOneGroup, NearestNeighborsGrouping, SpawnNearEgoGrouping
from .toolkit import NetResource, V2VMessage, _safe_nbytes, LatencyModel, SimpleWirelessLatency, _tx_bytes_for_latency
from .toolkit import Observer, payload_fn_cnn
from .toolkit import RandomPlanner
from .toolkit import VehicleNodeGraphBuilder, GraphBuildConfig

class CarlaVehGroupsEnv(CarlaWptFixedEnv):
    """
    Multi-vehicle group formation + V2V communication env.

    Design choice:
    - Keep "grouping" and "communication" logic inside env (deterministic simulation logic).
    - Keep "sensor fusion" inside observation handlers (clean separation).
    """

    def __init__(self, config):
        super().__init__(config)

        # RNG for deterministic testing if you want (seed from config if exists)
        seed = getattr(self._config, "seed", None)
        self._rng = random.Random(seed)

        # --- configurable knobs with safe defaults ---
        self.num_focus_vehicles = int(getattr(self._config, "num_focus_vehicles", 3))

        # How often to recompute groups / send comm (in steps)
        self.group_update_period = int(getattr(self._config, "group_update_period", 20))
        self.comm_period = int(getattr(self._config, "comm_period", 5))

        # Grouping policy selection
        grouping_name = getattr(self._config, "grouping_strategy", "spawn_near_ego")
        max_group_size = int(getattr(self._config, "max_group_size", 5))
        max_group_pair_distance_m = float(getattr(self._config, "max_group_pair_distance_m", 100.0))

        if grouping_name == "all":
            self.grouping_strategy: GroupingStrategy = AllInOneGroup()
        elif grouping_name == "nearest":
            self.grouping_strategy = NearestNeighborsGrouping(
                max_group_size=max_group_size,
                max_pair_distance_m=max_group_pair_distance_m,
            )
        elif grouping_name == "spawn_near_ego":
            self.grouping_strategy = SpawnNearEgoGrouping(
                max_group_size=max_group_size,
                max_pair_distance_m=max_group_pair_distance_m
            )
        else:
            # allow user to inject a custom instance externally
            self.grouping_strategy = AllInOneGroup()

        # Network resources (global default per-vehicle caps; you can override per-vehicle later)
        comm_cfg = getattr(self._config, "communication", None)
        bandwidth_hz = float(getattr(comm_cfg, "bandwidth_hz", 10e6))
        self._default_net_res = NetResource(bandwidth_hz=bandwidth_hz)

        # Latency model
        proc_delay_s = float(getattr(comm_cfg, "proc_delay_s", 0.0005))
        proc_delay_per_kb_s = float(getattr(comm_cfg, "proc_delay_per_kb_s", 0.0001))
        distance_decay_m = float(getattr(comm_cfg, "distance_decay_m", 60.0))
        min_rate_factor = float(getattr(comm_cfg, "min_rate_factor", 0.2))
        jitter_s = float(getattr(comm_cfg, "jitter_s", 0.0))

        self.latency_model: LatencyModel = SimpleWirelessLatency(
            proc_delay_s=proc_delay_s,
            proc_delay_per_kb_s=proc_delay_per_kb_s,
            distance_decay_m=distance_decay_m,
            min_rate_factor=min_rate_factor,
            jitter_s=jitter_s,
            rng=self._rng,
            overhead_bytes=64,  # You can adjust overhead_bytes as needed
        )

        # Optional hook to build your cooperative perception payload
        # Signature: payload_fn(sender_actor, env_state) -> Any
        self.payload_fn = payload_fn_cnn

        # runtime containers
        self.ego: Optional[carla.Actor] = None
        self.focus_vehicles: List[carla.Actor] = []
        self.background_vehicles: List[carla.Actor] = []

        # group state
        self.groups: Dict[int, Tuple[int, ...]] = {}
        self.veh_to_group: Dict[int, int] = {}

        # comm buffers
        self._in_flight: List[V2VMessage] = []
        self._received: Dict[int, Deque[V2VMessage]] = defaultdict(lambda: deque(maxlen=256))

        # Per-vehicle network resources (if you want heterogeneous vehicles)
        self._veh_net_res: Dict[int, NetResource] = {}

        # terminal: time limit (reuse world.fixed_delta_seconds already)
        self._time_limit_steps = int(getattr(self._config.terminal, "time_limit", 500))
        
        
        self._other_observers = {} 
        self.group_obs = {}
        self.feature_size = 1024
        
        cfg = GraphBuildConfig(window_s=2.0, Tmax=20, max_nodes=6, feat_dim_max=1024, star_graph=True)
        self._graph_builder = VehicleNodeGraphBuilder(cfg)
        

    # -----------------------------
    # Core lifecycle
    # -----------------------------

    def on_reset(self) -> None:
        # Clear runtime state
        self.focus_vehicles = []
        self.background_vehicles = []
        self.groups = {}
        self.veh_to_group = {}
        self._in_flight = []
        self._received = defaultdict(lambda: deque(maxlen=256))
        self._veh_net_res = {}
        for obs in self._other_observers.values():
            obs.destroy()
        self._other_observers = {}
        self.group_obs = {}

        traffic_lights = self._world.carla_actors(actor_type = 'traffic_light')
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(9999)
            tl.set_red_time(0)
            tl.set_yellow_time(0)

        # Spawn ego
        self.ego = self._world.spawn_actor()
        self._world.spawn_auto_actors(self._config.num_vehicles)
        self.ego_planner = RandomPlanner(vehicle=self.ego)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]
        
        # Spawn focus vehicles near random spawn points
        # (You may want to constrain them to an area / route later.)

        if getattr(self._config, "grouping_strategy", "spawn_near_ego") != "spawn_near_ego":
            blueprints = self._world.get_blueprint_library("vehicle.audi*", {"number_of_wheels": "4"})
            actor_list = self._world.spawn_auto_actors(n = self.num_focus_vehicles, blueprints=blueprints)    # generate vehicle controller by world manager
            for v in actor_list:
                if v is None:
                    continue
                self.focus_vehicles.append(v)

        # Initial grouping immediately at reset
        self._update_groups()
        print(f"[Carla Reset] Spawned ego + {len(self.focus_vehicles)} focus vehicles + {len(self.background_vehicles)} background vehicles.")
        print(f"         Ego ID: {self.ego.id}, Focus IDs: {[v.id for v in self.focus_vehicles]}")
        
        # Initialize per-vehicle net resources
        for v in [self.ego] + self.focus_vehicles:
            if v is not None:
                self._veh_net_res[v.id] = self._default_net_res
                
        # Put focus vehicles on autopilot by default (so they move).
        # (If you later want to control some focus vehicles too, override this logic.)
        tm_port = getattr(self._world, "_tm_port", None)
        for v in self.focus_vehicles:
            actor_id = v.id
            focus_observation = self._config.observation 
            focus_observation = focus_observation.update(
                enabled=["camera", "collision"]
            )
            observer = Observer(self._world, focus_observation)
            self._other_observers.setdefault(actor_id, observer) 
            self._other_observers[actor_id].reset(v)  
            self.group_obs[actor_id], _ = self._other_observers[actor_id].get_observation(self.get_state())

    def on_step(self) -> None:
        # 1) deliver messages whose time has come
        self._deliver_messages()

        # 2) periodic grouping
        # if self._time_step % max(self.group_update_period, 1) == 0:
        #     self._update_groups()
        for v in self.focus_vehicles:
            actor_id = v.id
            observer = self._other_observers.get(actor_id, None)
            if observer is not None:
                self.group_obs[actor_id], _ = observer.get_observation(self.get_state())
                
        # 3) periodic intra-group communication
        if self._time_step % max(self.comm_period, 1) == 0:
            self._run_group_communication()

        super().on_step()
        # 计算ego和focus车辆之间的距离，并打印
        # ego_loc = self.ego.get_transform().location
        # for v in self.focus_vehicles:
        #     v_loc = v.get_transform().location
        #     dist = _dist_m(self.ego, v) # math.sqrt((ego_loc.x - v_loc.x) ** 2 + (ego_loc.y - v_loc.y) ** 2)
        #     print(f"Step {self._time_step}: Distance from Ego (ID {self.ego.id}) to Focus Vehicle (ID {v.id}): {dist:.2f} meters")


    def get_state(self) -> Dict:
        return self._build_state()

    # -----------------------------
    # Grouping Logic
    # -----------------------------

    def _update_groups(self) -> None:
        """
        Compute groups using the current grouping strategy.
        """
        groups = self.grouping_strategy.form_groups(
            focus_vehicles=self.focus_vehicles,
            world=self,
            time_step=self._time_step,
            rng=self._rng,
        )

        # Normalize to tuples, build reverse map
        self.groups = {int(gid): tuple(int(x) for x in members) for gid, members in groups.items()}
        self.veh_to_group = {}
        for gid, members in self.groups.items():
            for vid in members:
                self.veh_to_group[int(vid)] = int(gid)
        
        print(f"[Carla Grouping] Step {self._time_step}: Formed {len(self.groups)} groups with strategy '{type(self.grouping_strategy).__name__}'.")

    # -----------------------------
    # Communication Logic
    # -----------------------------

    def _make_payload(self, sender: carla.Actor) -> Any:
        """
        Default payload is lightweight.
        Replace this by setting self.payload_fn or overriding this method:
            - occupancy grid
            - detected object list
            - feature map (BEV)
            - etc.
        """
        if self.payload_fn is not None:
            obs = self.obs if sender.id == self.ego.id else self.group_obs.get(sender.id, {})
            return self.payload_fn(sender, obs, self.feature_size)

        tf = sender.get_transform()
        vel = sender.get_velocity()
        payload = {
            "pose": np.array([tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw], dtype=np.float32),
            "vel": np.array([vel.x, vel.y, vel.z], dtype=np.float32),
        }
        return payload

    def _run_group_communication(self) -> None:
        """
        For each group:
          each member broadcasts its payload to other members (excluding self).
        """
        if not self.groups:
            return

        # Precompute in/out degrees for contention modeling
        out_deg: Dict[int, int] = {}
        in_deg: Dict[int, int] = {}
        for gid, members in self.groups.items():
            m = list(members)
            for sender in m:
                out_deg[sender] = max(len(m) - 1, 0)
            for receiver in m:
                in_deg[receiver] = max(len(m) - 1, 0)

        # Create messages
        id_to_actor: Dict[int, carla.Actor] = {}
        if self.ego is not None:
            id_to_actor[self.ego.id] = self.ego
        for v in self.focus_vehicles:
            id_to_actor[v.id] = v
        # (background vehicles are not in groups by default)

        fixed_dt = float(self._world._settings.fixed_delta_seconds)

        for gid, members in self.groups.items():
            members = list(members)
            for sender_id in members:
                sender = id_to_actor.get(sender_id, None)
                if sender is None:
                    continue

                payload = self._make_payload(sender)
                # payload_bytes = _safe_nbytes(payload)
                payload_bytes = _tx_bytes_for_latency(payload, overhead_bytes=getattr(self.latency_model, "overhead_bytes", 64))  # You can adjust overhead_bytes as needed

                for receiver_id in members:
                    if receiver_id == sender_id:
                        continue
                    receiver = id_to_actor.get(receiver_id, None)
                    if receiver is None:
                        continue

                    s_res = self._veh_net_res.get(sender_id, self._default_net_res)
                    r_res = self._veh_net_res.get(receiver_id, self._default_net_res)

                    latency_s = self.latency_model.compute_latency_s(
                        sender=sender,
                        receiver=receiver,
                        payload_size_bytes=payload_bytes,
                        sender_res=s_res,
                        receiver_res=r_res,
                        out_degree=max(out_deg.get(sender_id, 1), 1),
                        in_degree=max(in_deg.get(receiver_id, 1), 1),
                    )

                    # Convert seconds to steps (ceil ensures strictly positive delay if latency>0)
                    delay_steps = int(math.ceil(latency_s / max(fixed_dt, 1e-6)))
                    delay_steps = max(delay_steps, 0)
                    deliver_step = int(self._time_step + delay_steps)

                    msg = V2VMessage(
                        sender_id=int(sender_id),
                        receiver_id=int(receiver_id),
                        group_id=int(gid),
                        payload=payload,
                        payload_bytes=int(payload_bytes),
                        created_step=int(self._time_step),
                        deliver_step=int(deliver_step),
                        latency_s=float(latency_s),
                        distance_m=float(_dist_m(sender, receiver)),
                    )
                    self._in_flight.append(msg)

    def _deliver_messages(self) -> None:
        """
        Deliver messages whose deliver_step <= current step.
        """
        if not self._in_flight:
            return

        cur = int(self._time_step)
        remaining: List[V2VMessage] = []
        for msg in self._in_flight:
            if msg.deliver_step <= cur:
                self._received[msg.receiver_id].append(msg)
            else:
                remaining.append(msg)
        self._in_flight = remaining

    # -----------------------------
    # State builder
    # -----------------------------

    def _build_state(self) -> Dict[str, Any]:
        ego_id = self.ego.id if self.ego is not None else -1

        # Summaries for quick debugging / plotting
        recv_buf = self._received.get(ego_id, deque())
        latest = recv_buf[-1] if len(recv_buf) > 0 else None

        state = {
            "time_step": int(self._time_step),
            "ego_waypoints": self.waypoints,
            # "ego_id": int(ego_id),
            # "focus_ids": [int(v.id) for v in self.focus_vehicles],
            # "background_ids": [int(v.id) for v in self.background_vehicles],
            # # group assignments
            # "groups": dict(self.groups),
            # "veh_to_group": dict(self.veh_to_group),
            # # comm buffers (full, for fusion handlers)
            # "comm_in_flight": list(self._in_flight),
            # "comm_received": self._received,  # dict[veh_id] -> deque[V2VMessage]
            # # quick ego summary
            # "comm_received_summary": {
            #     "n_received": int(len(recv_buf)),
            #     "latest_sender": int(latest.sender_id) if latest is not None else -1,
            #     "latest_latency_s": float(latest.latency_s) if latest is not None else 0.0,
            #     "latest_payload_bytes": int(latest.payload_bytes) if latest is not None else 0,
            #     "latest_distance_m": float(latest.distance_m) if latest is not None else 0.0,
            # },
        }
        # print(f"[Carla State] Step {self._time_step}: Ego received {state['comm_received_summary']['n_received']} messages; latest from {state['comm_received_summary']['latest_sender']} with latency {state['comm_received_summary']['latest_latency_s']:.3f}s and size {state['comm_received_summary']['latest_payload_bytes']} bytes at distance {state['comm_received_summary']['latest_distance_m']:.1f}m.")
        # print(f"    Groups: {state['groups']}")
        # print(f"    In-flight messages: {len(state['comm_in_flight'])}")
        # print(f"    Vehicle to group mapping: {state['veh_to_group']}")
        return state
    
    def step(self, action):
        self.apply_control(action)
        self._world.step()
        self._time_step += 1

        env_state = self.get_state()
        is_terminal, terminal_conds = self._is_terminal()
        self.obs, obs_info = self._ego_observer.get_observation(env_state)
        reward, reward_info = self.reward()
        
        print(f"[Carla Step] Step {self._time_step}: obs keys {self.obs.keys()}, reward {reward}, terminal {is_terminal}")
        info = {
            **env_state,
            **terminal_conds,
            **obs_info,
            **reward_info,
            "action": action,
        }
        if self._config.eval:
            info = {f"eval_{k}": v for k, v in info.items()}
            self.obs = {**self.obs, **info}
        if self._config.display.enable:
            self._render(self.obs, info)
            
        ego_feature = self.payload_fn(self.ego, self.obs, self.feature_size)
        msgs = self._received.get(self.ego.id, deque())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        shared_data = self._graph_builder.build(
            ego_actor=self.ego,
            carla_world=self._world._world,  # carla.World
            ego_feat=ego_feature.get("feat", None),
            ego_feat_dim=ego_feature.get("feat_dim", None),
            msgs=msgs,
            t_step=self._time_step,
            dt=float(self._config.world.fixed_delta_seconds),
            device=device,
        )
        # info.update(shared_data)
        info = shared_data

        # Gymnasium API: return (obs, reward, terminated, truncated, info)
        # terminated: episode ended naturally (goal/failure)
        # truncated: episode was cut short (time limit, etc.)
        terminated = is_terminal
        truncated = False  # CarDreamer doesn't use truncated separately
        return self.obs, reward, terminated, truncated, info    
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset environment (Gymnasium API): accepts `seed` and `options`.

        Returns (obs, info).
        """
        print("[CARLA] Reset environment")
        # super().reset(seed=seed)

        # Keep behavior unchanged: seed is accepted but not applied here.
        self._ego_observer.destroy()
        self._world.reset()
        self._ego_observer.reset(self.get_ego_vehicle())

        self._time_step = 0

        print("[CARLA] Environment reset")
        self.obs, info = self._ego_observer.get_observation(self.get_state())
        
        ego_feature = self.payload_fn(self.ego, self.obs, self.feature_size)
        msgs = self._received.get(self.ego.id, deque())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        shared_data = self._graph_builder.build(
            ego_actor=self.ego,
            carla_world=self._world._world,  # carla.World
            ego_feat=ego_feature.get("feat", None),
            ego_feat_dim=ego_feature.get("feat_dim", None),
            msgs=msgs,
            t_step=self._time_step,
            dt=float(self._config.world.fixed_delta_seconds),
            device=device,
        )
        # 把shared_data中的数据也放到obs里，方便后续使用
        # info.update(shared_data)
        info = shared_data
        
        return self.obs, info
