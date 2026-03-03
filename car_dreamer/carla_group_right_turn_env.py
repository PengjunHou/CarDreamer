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
from .toolkit import RandomPlanner, get_vehicle_pos
from .toolkit import VehicleNodeGraphBuilder, GraphBuildConfig


class CarlaGroupRightTurnEnv(CarlaWptFixedEnv):
    """
    Vehicle passes the crossing (turn right) and avoid collision.

    **Provided Tasks**: ``carla_right_turn_simple``, ``carla_right_turn_medium``, ``carla_right_turn_hard``
    """
    def __init__(self, config):
        print("[CARLA Group Right Turn Env] Initializing environment with config:")
        super().__init__(config)
        # Initialize car flow
        # --- configurable knobs with safe defaults ---
        self.groups = {}  # Mapping[group_id -> set of vehicle ids]
        self.group_vehs: List[carla.Actor] = []  # List of group vehicle actors
        self.num_group_vehs = int(getattr(self._config, "num_group_vehs", 2))
        self._other_observers = {} 
        self.group_obs = {}

        # How often to recompute groups / send comm (in steps)
        self.group_update_period = int(getattr(self._config, "group_update_period", 20))
        self.comm_period = int(getattr(self._config, "comm_period", 5))

        # Grouping policy selection
        # grouping_name = getattr(self._config, "grouping_strategy", "fixed")
        # if grouping_name == "all":
        #     self.grouping_strategy: GroupingStrategy = AllInOneGroup
        # elif grouping_name == "fixed":
        #     group_spawn_points = getattr(self._config, "group_spawn_points", None)
        #     assert group_spawn_points is not None, "group_spawn_points must be provided for fixed grouping strategy"
        #     assert len(group_spawn_points) >= self.num_group_vehs, "Not enough spawn points for the number of group vehicles"
        #     self.grouping_strategy = FixedGrouping
        # else:
        #     # allow user to inject a custom instance externally
        #     self.grouping_strategy = AllInOneGroup

        # Network resources (global default per-vehicle caps; you can override per-vehicle later)
        uplink_bps = float(getattr(self._config, "uplink_bps", 6e6))
        downlink_bps = float(getattr(self._config, "downlink_bps", 12e6))
        self._default_net_res = NetResource(uplink_bps=uplink_bps, downlink_bps=downlink_bps)

        # Latency model
        base_rtt_s = float(getattr(self._config, "base_rtt_s", 0.02))
        proc_delay_s = float(getattr(self._config, "proc_delay_s", 0.005))
        distance_decay_m = float(getattr(self._config, "distance_decay_m", 60.0))
        min_rate_factor = float(getattr(self._config, "min_rate_factor", 0.2))
        jitter_s = float(getattr(self._config, "jitter_s", 0.0))

        self.latency_model: LatencyModel = SimpleWirelessLatency(
            base_rtt_s=base_rtt_s,
            proc_delay_s=proc_delay_s,
            distance_decay_m=distance_decay_m,
            min_rate_factor=min_rate_factor,
            jitter_s=jitter_s,
            overhead_bytes=64,  # You can adjust overhead_bytes as needed
        )

        # Optional hook to build your cooperative perception payload
        # Signature: payload_fn(sender_actor, env_state) -> Any
        self.payload_fn = payload_fn_cnn

        # comm buffers
        self._in_flight: List[V2VMessage] = []
        self._received: Dict[int, Deque[V2VMessage]] = defaultdict(lambda: deque(maxlen=256))

        # Per-vehicle network resources (if you want heterogeneous vehicles)
        self._veh_net_res: Dict[int, NetResource] = {}

        # terminal: time limit (reuse world.fixed_delta_seconds already)
        self._time_limit_steps = int(getattr(self._config.terminal, "time_limit", 500))
        
        self.feature_size = 1024
        cfg = GraphBuildConfig(window_s=2.0, Tmax=20, max_nodes=4, feat_dim_max=1024, star_graph=True)
        self._graph_builder = VehicleNodeGraphBuilder(cfg)
        
    def generate_group_vehicles(self):
        # Generate group vehicles based on the grouping strategy
        self.groups.setdefault(0, set())
        spawn_points = self._config.group_spawn_points
        assert spawn_points is not None and len(spawn_points) >= self.num_group_vehs, "Not enough spawn points for the number of group vehicles"
        for spawn_point in spawn_points[:self.num_group_vehs]:
            transform = carla.Transform(
                carla.Location(*spawn_point[:3]),
                carla.Rotation(yaw=spawn_point[3]),
            )
            vehicle = self._world.spawn_actor(transform=transform)
            group_observation = self._config.group_observation
            group_vhe_observer = Observer(self._world, group_observation)
            self._other_observers.setdefault(vehicle.id, group_vhe_observer) 
            self._other_observers[vehicle.id].reset(vehicle)  
            self.group_obs[vehicle.id], _ = self._other_observers[vehicle.id].get_observation(self.get_state())
            
            self.group_vehs.append(vehicle)
            self.groups[0].add(vehicle.id)
        
    def on_reset(self) -> None:
        # vehicle group
        # for veh in self.group_vehs:
        #     veh_id = veh.id
        #     self._world.destroy_actor(veh_id)
        self.group_vehs = []
        self.groups = {}

        # comm buffers
        self._in_flight = []
        self._received = defaultdict(lambda: deque(maxlen=256))
        self._veh_net_res = {}
        
        # observers
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
        
        super().on_reset()
        self.generate_group_vehicles()
        # self.grouping_strategy.form_groups(
        #     focus_vehicles=self.group_vehs,
        #     world=self._world,
        #     time_step=self._time_step,
        #     rng=self._rng,
        # ) 
        
    def on_step(self) -> None:
        self._deliver_messages()
        # 2) periodic grouping
        # if self._time_step % max(self.group_update_period, 1) == 0:
        #     self._update_groups()
        for v in self.group_vehs:
            actor_id = v.id
            observer = self._other_observers.get(actor_id, None)
            if observer is not None:
                self.group_obs[actor_id], _ = observer.get_observation(self.get_state())
                
        # 3) periodic intra-group communication
        if self._time_step % max(self.comm_period, 1) == 0:
            self._run_group_communication()
        
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(self.actor_flow[0])
            if y > -81.2 or x < -38.4 or x > 31.6:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()
        super().on_step()


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
        for v in self.group_vehs:
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

    def get_state(self):
        self._state = {"ego_waypoints": self.waypoints, "timesteps": self._time_step}
        return self._state

    def step(self, action):
        self.get_state()
        _, reward, terminated, truncated, info = super().step(action)  # important! this will trigger on_step() and update the group vehicles
            
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
        print(f"[STEP] Shared data keys: {list(shared_data.keys())}, obs keys: {list(self.obs.keys())}")
        
        return self.obs, reward, terminated, truncated, info    
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset environment (Gymnasium API): accepts `seed` and `options`.

        Returns (obs, info).
        """
        print("[CARLA Group Right Turn Env] Reset environment")
        super().reset(seed=seed)

        # Keep behavior unchanged: seed is accepted but not applied here.
        # self._ego_observer.destroy()
        # self._world.reset()
        # self._ego_observer.reset(self.get_ego_vehicle())

        # self._time_step = 0

        # print("[CARLA] Environment reset")
        # self.obs, info = self._ego_observer.get_observation(self.get_state())
        
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