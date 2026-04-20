from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import carla
import numpy as np
import torch
from runtime_logging import get_runtime_logger, get_runtime_logging_config, should_log_periodic

from .toolkit import (
    NetResource,
    Observer,
    V2VMessage,
    _dist_m,
    _tx_bytes_for_latency,
    get_vehicle_pos,
    payload_fn_llm,
)
from .toolkit.emulation.features import (
    build_observable_region,
    compute_accessibility,
    compute_complementarity,
    compute_task_relevance,
)
from .toolkit.emulation.policy import CollaborationAction, VehicleInfo, get_policy
from .toolkit.emulation.queries import make_query_records


GROUP_ID = 0
RECEIVED_BUFFER_SIZE = 256
RUNTIME_LOGGER = get_runtime_logger("car_dreamer.runtime")


class RightTurnAutoRuntimeMixin:
    def _reset_policy_runtime_state(self) -> None:
        self._policy_current_action = CollaborationAction()
        self._policy_last_action_step = -1
        self._policy_send_credit = {}
        self._policy_send_decision = {}
        self._policy_last_comm_step = -1

    def _get_runtime_debug_interval(self) -> int:
        runtime_cfg = get_runtime_logging_config()
        base_interval = int(runtime_cfg["step_debug_interval"])
        override = int(getattr(self, "_runtime_step_debug_interval_override", 0) or 0)
        if override > 0:
            return max(base_interval, override)
        return base_interval

    def _cache_actor(self, actor: Optional[carla.Actor]) -> None:
        if actor is not None:
            self._actor_cache[int(actor.id)] = actor

    def _refresh_actor_cache(self) -> None:
        self._actor_cache = {}
        self._cache_actor(getattr(self, "ego", None))
        for actor in self.group_vehs:
            self._cache_actor(actor)

    def _get_group_member_actor(self, actor_id: int) -> Optional[carla.Actor]:
        actor = self._actor_cache.get(int(actor_id))
        if actor is not None:
            return actor
        if getattr(self, "ego", None) is not None and int(self.ego.id) == int(actor_id):
            self._cache_actor(self.ego)
            return self.ego
        for actor in self.group_vehs:
            if int(actor.id) == int(actor_id):
                self._cache_actor(actor)
                return actor
        actor = self._world._world.get_actor(int(actor_id))
        if actor is not None:
            self._cache_actor(actor)
        return actor

    def _reset_group_runtime_state(self) -> None:
        self.group_vehs = []
        self.groups = {}
        self._prev_action = None
        self._actor_cache = {}
        self._in_flight = []
        self._received = defaultdict(lambda: deque(maxlen=RECEIVED_BUFFER_SIZE))
        self._veh_net_res = {}
        self._vlm_records = []
        self._vlm_last_eval = {}
        self._vlm_step_cache = {}
        self._vlm_episode_dumped = False
        self._emulation_episode_dumped = False
        self._emulation_episode_steps = []
        self._emulation_step_counter = 0
        self._comm_link_analysis_by_sender = {}
        self._comm_step_summary = {}
        self._comm_step_summary_step = -1
        self._reset_policy_runtime_state()
        RUNTIME_LOGGER.debug("Group runtime state reset.")

    def _ensure_comm_step_summary(self) -> Dict[str, float]:
        current_step = int(getattr(self, "_time_step", 0))
        if int(getattr(self, "_comm_step_summary_step", -1)) != current_step:
            self._comm_step_summary = {
                "attempted_message_count": 0.0,
                "dropped_message_count": 0.0,
                "enqueued_message_count": 0.0,
                "dropped_capacity_exceeded_count": 0.0,
            }
            self._comm_step_summary_step = current_step
        return self._comm_step_summary

    def _record_comm_link_analysis(
        self,
        sender_id: int,
        receiver_id: int,
        payload_bytes: int,
        analysis: Any,
        *,
        dropped: bool,
        drop_reason: str = "",
    ) -> None:
        if dropped and not bool(getattr(self, "_log_dropped_messages", True)):
            return
        if analysis is None:
            return
        self._comm_link_analysis_by_sender[int(sender_id)] = {
            "sender_id": int(sender_id),
            "receiver_id": int(receiver_id),
            "payload_bytes": float(payload_bytes),
            "required_load_bps": float(getattr(analysis, "required_load_bps", 0.0)),
            "link_rate_bps": float(getattr(analysis, "link_rate_bps", 0.0)),
            "shannon_bps": float(getattr(analysis, "shannon_bps", 0.0)),
            "uplink_bps": float(getattr(analysis, "uplink_bps", 0.0)),
            "downlink_bps": float(getattr(analysis, "downlink_bps", 0.0)),
            "comm_bandwidth_hz": float(getattr(analysis, "bandwidth_hz", 0.0)),
            "comm_snr_db": float(getattr(analysis, "snr_db", 0.0)),
            "comm_feasible": 1.0 if bool(getattr(analysis, "feasible", False)) else 0.0,
            "dropped_capacity_exceeded": 1.0 if bool(dropped) else 0.0,
            "drop_reason_capacity_exceeded": 1.0 if str(drop_reason) == "capacity_exceeded" else 0.0,
            "analysis_latency_s": float(getattr(analysis, "latency_s", 0.0)),
            "analysis_distance_m": float(getattr(analysis, "distance_m", 0.0)),
        }

    def _get_latest_comm_link_analysis(self, vehicle_id: int) -> Dict[str, float]:
        stats = self._comm_link_analysis_by_sender.get(int(vehicle_id), {})
        return {str(key): float(value) for key, value in stats.items()}

    def _get_current_comm_step_summary(self) -> Dict[str, float]:
        summary = self._ensure_comm_step_summary()
        return {str(key): float(value) for key, value in summary.items()}

    def _destroy_group_observers(self) -> None:
        for observer in self._other_observers.values():
            observer.destroy()
        self._other_observers = {}
        self.group_obs = {}

    def _configure_traffic_lights(self) -> None:
        for tl in self._world.carla_actors(actor_type="traffic_light"):
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(9999)
            tl.set_red_time(0)
            tl.set_yellow_time(0)

    def _create_group_observer(self, vehicle: carla.Actor) -> None:
        group_observation = self._config.group_observation
        observer = Observer(self._world, group_observation)
        self._other_observers[int(vehicle.id)] = observer
        observer.reset(vehicle)
        self.group_obs[int(vehicle.id)], _ = observer.get_observation(self.get_state())

    def generate_group_vehicles(self):
        self.groups.setdefault(GROUP_ID, set())
        self.groups[GROUP_ID].add(int(self.ego.id))
        spawn_points = self._config.group_spawn_points
        assert spawn_points is not None and len(spawn_points) >= self.num_group_vehs, (
            "Not enough spawn points for the number of group vehicles"
        )
        for spawn_point in spawn_points[: self.num_group_vehs]:
            transform = carla.Transform(
                carla.Location(*spawn_point[:3]),
                carla.Rotation(yaw=spawn_point[3]),
            )
            vehicle = self._world.spawn_actor(transform=transform)
            self._create_group_observer(vehicle)
            self.group_vehs.append(vehicle)
            self.groups[GROUP_ID].add(int(vehicle.id))
            self._cache_actor(vehicle)
        RUNTIME_LOGGER.info(
            "Generated group vehicles count=%d ids=%s group_members=%s",
            len(self.group_vehs),
            [int(vehicle.id) for vehicle in self.group_vehs],
            sorted(self.groups.get(GROUP_ID, set())),
        )

    def _update_group_observations(self) -> None:
        for actor in self.group_vehs:
            observer = self._other_observers.get(int(actor.id))
            if observer is not None:
                self.group_obs[int(actor.id)], _ = observer.get_observation(self.get_state())

    def _setup_basic_agent(self) -> None:
        from .toolkit.planner.agents.navigation.basic_agent import BasicAgent

        self.ego_end = self._config.lane_end_point
        ego_transform = carla.Transform(
            carla.Location(*self.ego_end[:3]),
            carla.Rotation(yaw=self.ego_end[3]),
        )
        self.agent = BasicAgent(self.ego)
        self.agent.set_destination(ego_transform.location)
        self._cache_actor(self.ego)

    def _build_policy_vehicle_infos(self) -> List[VehicleInfo]:
        if getattr(self, "ego", None) is None:
            return []
        ego_tf = self.ego.get_transform()
        ego_vel = self.ego.get_velocity()
        ego_pose = {
            "x": float(ego_tf.location.x),
            "y": float(ego_tf.location.y),
            "yaw_rad": math.radians(float(ego_tf.rotation.yaw)),
        }
        ego_velocity = {
            "vx": float(ego_vel.x),
            "vy": float(ego_vel.y),
        }
        scene_type = str(getattr(self, "_emulation_scene_type", "right_turn"))
        queries = make_query_records(scene_type)
        ego_region = build_observable_region((0.0, 0.0), 0.0, range_m=16.0, width_m=9.0, lookahead_m=8.0)
        infos: List[VehicleInfo] = []
        for actor in self.group_vehs:
            tf = actor.get_transform()
            vel = actor.get_velocity()
            delta_pos = (
                float(tf.location.x) - ego_pose["x"],
                float(tf.location.y) - ego_pose["y"],
            )
            delta_vel = (
                float(vel.x) - ego_velocity["vx"],
                float(vel.y) - ego_velocity["vy"],
            )
            delta_yaw = math.radians(float(tf.rotation.yaw)) - ego_pose["yaw_rad"]
            sender_region = build_observable_region(delta_pos, delta_yaw)
            distance_m = math.sqrt(delta_pos[0] * delta_pos[0] + delta_pos[1] * delta_pos[1])
            complementarity = compute_complementarity(sender_region, ego_region)
            accessibility = compute_accessibility(distance_m, 0.0)
            collab_values = []
            for query in queries:
                relevance = compute_task_relevance(sender_region, ego_region, query.required_region)
                collab_values.append(float(complementarity * relevance * accessibility))
            sender_collab = float(sum(collab_values) / len(collab_values)) if collab_values else 0.0
            infos.append(
                VehicleInfo(
                    vehicle_id=int(actor.id),
                    sender_collab=sender_collab,
                    distance_m=float(distance_m),
                )
            )
        return infos

    def _compute_current_policy_action(self) -> CollaborationAction:
        if getattr(self, "_policy_last_action_step", -1) == int(self._time_step):
            return self._policy_current_action
        policy = getattr(self, "_collaboration_policy", None)
        if policy is None:
            self._policy_current_action = CollaborationAction()
            self._policy_last_action_step = int(self._time_step)
            return self._policy_current_action
        rng = random.Random(int(getattr(self, "_collaboration_policy_seed", 0)) + int(self._time_step))
        infos = self._build_policy_vehicle_infos()
        self._policy_current_action = policy(infos, rng=rng)
        self._policy_last_action_step = int(self._time_step)
        return self._policy_current_action

    def _get_policy_action_value(self, vehicle_id: int) -> Dict[str, float]:
        action = self._compute_current_policy_action()
        vid = int(vehicle_id)
        return {
            "alpha": float(action.alpha.get(vid, 0.0)),
            "nu": float(action.nu.get(vid, 0.0)),
            "bandwidth": float(action.bandwidth.get(vid, 0.0)),
        }

    def _advance_policy_send_schedule(self) -> Dict[int, bool]:
        if getattr(self, "_policy_last_comm_step", -1) == int(self._time_step):
            return dict(self._policy_send_decision)
        action = self._compute_current_policy_action()
        decisions: Dict[int, bool] = {}
        for vid, alpha in action.alpha.items():
            vehicle_id = int(vid)
            if float(alpha) <= 0.5:
                self._policy_send_credit[vehicle_id] = 0.0
                decisions[vehicle_id] = False
                continue
            nu = max(float(action.nu.get(vehicle_id, 0.0)), 0.0)
            credit = float(self._policy_send_credit.get(vehicle_id, 0.0)) + nu
            should_send = credit >= 1.0 - 1e-6
            if should_send:
                credit = max(credit - 1.0, 0.0)
            self._policy_send_credit[vehicle_id] = credit
            decisions[vehicle_id] = should_send
        self._policy_send_decision = decisions
        self._policy_last_comm_step = int(self._time_step)
        return dict(decisions)

    def _scale_net_resource_for_bandwidth(self, base: NetResource, bandwidth: float) -> NetResource:
        scale = max(float(bandwidth), float(getattr(self, "_collaboration_bandwidth_floor", 0.1)))
        return NetResource(
            uplink_bps=float(base.uplink_bps) * scale,
            downlink_bps=float(base.downlink_bps) * scale,
            bandwidth_hz=float(base.bandwidth_hz) * scale,
            tx_power_dbm=float(base.tx_power_dbm),
            noise_figure_db=float(base.noise_figure_db),
            carrier_freq_hz=float(base.carrier_freq_hz),
        )

    # =========================================================
    # Communication helpers
    # =========================================================

    def _make_payload(self, sender: carla.Actor) -> Dict[str, Any]:
        obs = self.obs if int(sender.id) == int(self.ego.id) else self.group_obs.get(int(sender.id), {})
        payload: Dict[str, Any] = {}
        if self.payload_fn is not None:
            sender_id = int(sender.id)
            payload = self.payload_fn(
                sender,
                obs,
                self.feature_size,
                image_proc_fn=lambda img, _unused, sid=sender_id: self._compute_single_image_description_from_array(
                    img,
                    cache_key=("scene_description", sid),
                ),
            )

        tf = sender.get_transform()
        vel = sender.get_velocity()
        payload.update(
            {
                "pose": {
                    "x": float(tf.location.x),
                    "y": float(tf.location.y),
                    "yaw": float(tf.rotation.yaw),
                },
                "vel": {"vx": float(vel.x), "vy": float(vel.y)},
                "sender_id": int(sender.id),
            }
        )
        return payload

    def _build_group_actor_map(self) -> Dict[int, carla.Actor]:
        actor_map: Dict[int, carla.Actor] = {}
        if getattr(self, "ego", None) is not None:
            actor_map[int(self.ego.id)] = self.ego
        for actor in self.group_vehs:
            actor_map[int(actor.id)] = actor
        self._actor_cache.update(actor_map)
        return actor_map

    def _compute_group_degrees(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        out_deg: Dict[int, int] = {}
        in_deg: Dict[int, int] = {}
        for members in self.groups.values():
            member_ids = list(members)
            degree = max(len(member_ids) - 1, 0)
            for sender_id in member_ids:
                out_deg[int(sender_id)] = degree
            for receiver_id in member_ids:
                in_deg[int(receiver_id)] = degree
        return out_deg, in_deg

    def _enqueue_message(
        self,
        group_id: int,
        sender_id: int,
        receiver_id: int,
        payload: Dict[str, Any],
        payload_bytes: int,
        latency_s: float,
        distance_m: float,
        fixed_dt: float,
    ) -> None:
        delay_steps = max(int(math.ceil(latency_s / max(fixed_dt, 1e-6))), 0)
        deliver_step = int(self._time_step + delay_steps)
        self._in_flight.append(
            V2VMessage(
                sender_id=int(sender_id),
                receiver_id=int(receiver_id),
                group_id=int(group_id),
                payload=payload,
                payload_bytes=int(payload_bytes),
                created_step=int(self._time_step),
                deliver_step=deliver_step,
                latency_s=float(latency_s),
                distance_m=float(distance_m),
            )
        )

    def _run_group_communication(self) -> None:
        if not self.groups:
            return
        step_summary = self._ensure_comm_step_summary()
        out_deg, in_deg = self._compute_group_degrees()
        actor_map = self._build_group_actor_map()
        fixed_dt = float(self._world._settings.fixed_delta_seconds)
        ego_id = int(self.ego.id)
        current_action = self._compute_current_policy_action()
        send_decisions = self._advance_policy_send_schedule()
        enqueued_count = 0
        sender_ids = set()
        total_payload_bytes = 0

        for group_id, members in self.groups.items():
            member_ids = list(members)
            for sender_id in member_ids:
                if int(sender_id) == ego_id:
                    continue
                sender = actor_map.get(int(sender_id))
                if sender is None:
                    continue
                alpha = float(current_action.alpha.get(int(sender_id), 0.0))
                if alpha <= 0.5 or not bool(send_decisions.get(int(sender_id), False)):
                    continue
                sender_ids.add(int(sender_id))
                payload = self._make_payload(sender)
                payload["policy_action"] = {
                    "policy_id": str(getattr(self, "_collaboration_policy_id", "")),
                    "alpha": alpha,
                    "nu": float(current_action.nu.get(int(sender_id), 0.0)),
                    "bandwidth": float(current_action.bandwidth.get(int(sender_id), 0.0)),
                }
                payload_bytes = _tx_bytes_for_latency(
                    payload,
                    overhead_bytes=getattr(self.latency_model, "overhead_bytes", 64),
                )
                for receiver_id in member_ids:
                    if int(receiver_id) != ego_id or int(receiver_id) == int(sender_id):
                        continue
                    receiver = actor_map.get(int(receiver_id))
                    if receiver is None:
                        continue
                    bandwidth = float(current_action.bandwidth.get(int(sender_id), 0.0))
                    sender_res = self._scale_net_resource_for_bandwidth(
                        self._veh_net_res.get(int(sender_id), self._default_net_res),
                        bandwidth,
                    )
                    receiver_res = self._scale_net_resource_for_bandwidth(
                        self._veh_net_res.get(int(receiver_id), self._default_net_res),
                        bandwidth,
                    )
                    analysis_fn = getattr(self.latency_model, "analyze_transmission", None)
                    if callable(analysis_fn):
                        analysis = analysis_fn(
                            sender=sender,
                            receiver=receiver,
                            payload_size_bytes=payload_bytes,
                            sender_res=sender_res,
                            receiver_res=receiver_res,
                            out_degree=max(out_deg.get(int(sender_id), 1), 1),
                            in_degree=max(in_deg.get(int(receiver_id), 1), 1),
                            alpha=alpha,
                            nu=float(current_action.nu.get(int(sender_id), 0.0)),
                            fixed_dt=fixed_dt,
                        )
                        latency_s = float(getattr(analysis, "latency_s", 0.0))
                    else:
                        analysis = None
                        latency_s = self.latency_model.compute_latency_s(
                            sender=sender,
                            receiver=receiver,
                            payload_size_bytes=payload_bytes,
                            sender_res=sender_res,
                            receiver_res=receiver_res,
                            out_degree=max(out_deg.get(int(sender_id), 1), 1),
                            in_degree=max(in_deg.get(int(receiver_id), 1), 1),
                            alpha=alpha,
                            nu=float(current_action.nu.get(int(sender_id), 0.0)),
                            fixed_dt=fixed_dt,
                        )
                    step_summary["attempted_message_count"] += 1.0
                    if (
                        bool(getattr(self, "_drop_on_capacity_exceeded", False))
                        and analysis is not None
                        and not bool(getattr(analysis, "feasible", True))
                    ):
                        step_summary["dropped_message_count"] += 1.0
                        step_summary["dropped_capacity_exceeded_count"] += 1.0
                        self._record_comm_link_analysis(
                            int(sender_id),
                            int(receiver_id),
                            payload_bytes,
                            analysis,
                            dropped=True,
                            drop_reason="capacity_exceeded",
                        )
                        if should_log_periodic(
                            int(self._time_step),
                            int(self._get_runtime_debug_interval()),
                            logger=RUNTIME_LOGGER,
                        ):
                            RUNTIME_LOGGER.debug(
                                "Dropped message step=%d sender=%d receiver=%d payload_bytes=%d r_req=%.3f R_link=%.3f",
                                self._time_step,
                                int(sender_id),
                                int(receiver_id),
                                int(payload_bytes),
                                float(getattr(analysis, "required_load_bps", 0.0)),
                                float(getattr(analysis, "link_rate_bps", 0.0)),
                            )
                        continue
                    self._record_comm_link_analysis(
                        int(sender_id),
                        int(receiver_id),
                        payload_bytes,
                        analysis,
                        dropped=False,
                    )
                    self._enqueue_message(
                        group_id=group_id,
                        sender_id=int(sender_id),
                        receiver_id=int(receiver_id),
                        payload=payload,
                        payload_bytes=payload_bytes,
                        latency_s=latency_s,
                        distance_m=_dist_m(sender, receiver),
                        fixed_dt=fixed_dt,
                    )
                    step_summary["enqueued_message_count"] += 1.0
                    enqueued_count += 1
                    total_payload_bytes += int(payload_bytes)
        if should_log_periodic(
            int(self._time_step),
            int(self._get_runtime_debug_interval()),
            logger=RUNTIME_LOGGER,
        ):
            RUNTIME_LOGGER.debug(
                "Communication round step=%d policy=%s senders=%s attempted=%d dropped=%d enqueued=%d in_flight=%d payload_bytes=%d",
                self._time_step,
                getattr(self, "_collaboration_policy_id", ""),
                sorted(sender_ids),
                int(step_summary.get("attempted_message_count", 0.0)),
                int(step_summary.get("dropped_message_count", 0.0)),
                enqueued_count,
                len(self._in_flight),
                total_payload_bytes,
            )

    def _deliver_messages(self) -> None:
        if not self._in_flight:
            return
        current_step = int(self._time_step)
        remaining: List[V2VMessage] = []
        delivered_count = 0
        for msg in self._in_flight:
            if int(msg.deliver_step) <= current_step:
                self._received[int(msg.receiver_id)].append(msg)
                delivered_count += 1
            else:
                remaining.append(msg)
        self._in_flight = remaining
        if should_log_periodic(
            current_step,
            int(self._get_runtime_debug_interval()),
            logger=RUNTIME_LOGGER,
        ):
            RUNTIME_LOGGER.debug(
                "Delivered messages step=%d delivered=%d remaining_in_flight=%d ego_received=%d",
                current_step,
                delivered_count,
                len(self._in_flight),
                len(self._received.get(int(self.ego.id), deque())),
            )

    # =========================================================
    # VLM evaluation
    # =========================================================

    def _build_graph_info(self) -> Dict[str, Any]:
        ego_feature = self.payload_fn(
            self.ego,
            self.obs,
            self.feature_size,
            image_proc_fn=None,
        )
        msgs = self._received.get(int(self.ego.id), deque())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._graph_builder.build(
            ego_actor=self.ego,
            carla_world=self._world._world,
            ego_feat=ego_feature.get("feat"),
            ego_feat_dim=ego_feature.get("feat_dim"),
            msgs=msgs,
            t_step=self._time_step,
            dt=float(self._config.world.fixed_delta_seconds),
            device=device,
        )

    def _merge_step_info(self, info: Dict[str, Any], requested_action: Any) -> Dict[str, Any]:
        del requested_action
        shared_data = self._build_graph_info()
        reward_info = {
            k: v
            for k, v in info.items()
            if k.startswith("r_")
            or k in [
                "wpt_dis",
                "speed_parallel",
                "speed_perpendicular",
                "speed_norm",
                "ttc",
                "time_penalty",
            ]
        }
        reward_info["ego_x"] = self.ego.get_transform().location.x
        reward_info["ego_y"] = self.ego.get_transform().location.y
        shared_data.update(reward_info)
        if should_log_periodic(
            int(self._time_step),
            int(self._get_runtime_debug_interval()),
            logger=RUNTIME_LOGGER,
        ):
            valid_nodes = int(np.asarray(shared_data.get("node_mask", np.zeros(0))).sum())
            edge_index = np.asarray(shared_data.get("edge_index", np.zeros((2, 0))))
            msg_count = len(self._received.get(int(self.ego.id), deque()))
            RUNTIME_LOGGER.debug(
                "Step info merged step=%d valid_nodes=%d num_edges=%d ego_received_msgs=%d reward_keys=%s",
                self._time_step,
                valid_nodes,
                edge_index.shape[1] if edge_index.ndim == 2 else 0,
                msg_count,
                sorted(reward_info.keys()),
            )
        return shared_data

    def _build_reset_info(self) -> Dict[str, Any]:
        shared_data = self._build_graph_info()
        ego_location = np.array([*get_vehicle_pos(self.ego)])
        reward_info = {
            "ego_x": ego_location[0],
            "ego_y": ego_location[1],
            "speed_parallel": 0,
            "speed_perpendicular": 0,
            "speed_norm": 0,
            "wpt_dis": self.get_wpt_dist(ego_location),
            "r_waypoints": 0,
            "r_speed": 0,
            "r_collision": 0,
            "r_out_of_lane": 0,
            "r_destination": 0,
            "time_penalty": 0,
            "ttc": 0,
        }
        shared_data.update(reward_info)
        valid_nodes = int(np.asarray(shared_data.get("node_mask", np.zeros(0))).sum())
        edge_index = np.asarray(shared_data.get("edge_index", np.zeros((2, 0))))
        RUNTIME_LOGGER.info(
            "Built reset graph info valid_nodes=%d num_edges=%d",
            valid_nodes,
            edge_index.shape[1] if edge_index.ndim == 2 else 0,
        )
        return shared_data

    def _maybe_dump_vlm_records_step(self) -> None:
        if not getattr(self, "_dump_vlm_records_each_step", False):
            return
        os.makedirs(self._vlm_dump_dir, exist_ok=True)
        path = os.path.join(self._vlm_dump_dir, "vlm_records_live.json")
        self.dump_vlm_records(path)

    def _maybe_dump_vlm_records(self, suffix: str) -> Optional[str]:
        if not self._dump_vlm_records_on_episode_end or self._vlm_episode_dumped:
            return None
        os.makedirs(self._vlm_dump_dir, exist_ok=True)
        filename = f"vlm_records_{suffix}_step_{int(self._time_step)}.json"
        path = os.path.join(self._vlm_dump_dir, filename)
        self.dump_vlm_records(path)
        self._vlm_episode_dumped = True
        return path

    def _maybe_dump_emulation_episode(self, suffix: str) -> Optional[str]:
        if (
            not getattr(self, "_dump_emulation_records_on_episode_end", False)
            or getattr(self, "_emulation_episode_dumped", False)
            or not getattr(self, "_emulation_episode_steps", [])
        ):
            return None
        os.makedirs(self._emulation_dump_dir, exist_ok=True)
        filename = f"emulation_episode_{suffix}_step_{int(self._time_step)}.json"
        path = os.path.join(self._emulation_dump_dir, filename)
        self.dump_emulation_episode(path)
        self._emulation_episode_dumped = True
        return path

    def _handle_episode_end(self, terminated: bool, truncated: bool, info: Dict[str, Any]) -> Dict[str, Any]:
        if terminated or truncated:
            suffix = "terminated" if terminated else "truncated"
            vlm_dump_path = self._maybe_dump_vlm_records(suffix)
            if vlm_dump_path is not None:
                info["vlm_dump_path"] = vlm_dump_path
                RUNTIME_LOGGER.info(
                    "Episode end dump created step=%d path=%s records=%d",
                    self._time_step,
                    vlm_dump_path,
                    len(getattr(self, "_vlm_records", [])),
                )
            emulation_dump_path = self._maybe_dump_emulation_episode(suffix)
            if emulation_dump_path is not None:
                info["emulation_dump_path"] = emulation_dump_path
                RUNTIME_LOGGER.info(
                    "Predictor-ready episode dump created step=%d path=%s canonical_steps=%d",
                    self._time_step,
                    emulation_dump_path,
                    len(getattr(self, "_emulation_episode_steps", [])),
                )
        return info

    # =========================================================
    # Environment overrides
    # =========================================================

    def _cleanup_actor_flow(self) -> None:
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(vehicle)
            if y > -81.2 or x < -38.4 or x > 31.6:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()
