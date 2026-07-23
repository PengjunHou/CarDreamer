import json
import os
import random
from collections import deque

import carla
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedPathPlanner, get_location_distance, get_vehicle_pos, get_vehicle_velocity
from .toolkit.observer.handlers.utils import get_visibility_from


class CarlaWptFixedEnv(CarlaWptEnv):
    """
    This is the base env for all waypoint following tasks with a fixed route and car flow.
    **DO NOT** instantiate this class directly.

    All envs that inherit from this class also inherits the following config parameters:

    * ``lane_start_point``: The starting point of the ego vehicle in ``[x, y, z, yaw]``
    * ``ego_path``: The fixed path for the ego vehicle in array of ``[x, y, z]``
    * ``use_road_waypoints``: For each segment, whether to adapt the path according to road or use straight line
    * ``flow_spawn_point``: The spawn point of the car flow in ``[x, y, z, yaw]``
    * ``min_flow_dist``: Minimum distance between two cars in the flow, if ``None``, no cars will be spawned
    * ``max_flow_dist``: Maximum distance between two cars in the flow

    """

    def on_reset(self) -> None:
        self.ego_src = self._config.lane_start_point
        ego_transform = carla.Transform(carla.Location(*self.ego_src[:3]), carla.Rotation(yaw=self.ego_src[3]))
        self.ego = self._world.spawn_actor(transform=ego_transform)
        self.ego_path = self._config.ego_path
        self.use_road_waypoints = self._config.use_road_waypoints
        self.ego_planner = FixedPathPlanner(
            vehicle=self.ego,
            vehicle_path=self.ego_path,
            use_road_waypoints=self.use_road_waypoints,
        )
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

        # Initialize car flow
        self.actor_flow = deque()
        flow_spawn_point = self._config.flow_spawn_point
        self.flow_transform = carla.Transform(
            carla.Location(*flow_spawn_point[:3]),
            carla.Rotation(yaw=flow_spawn_point[3]),
        )

        self.shared_intention_ids = []
        self._random_shared_ids = set()
        # Communication buffer: head of a full deque is the packet broadcast
        # latency_steps ago by the collaborators selected at that time
        self._comm_snapshots = deque(maxlen=self._config.intention_sharing.latency_steps + 1)
        self._comm_snapshot_step = None
        # Opt-in per-frame geometry recorder (experiment instrumentation; off
        # unless CARDREAMER_RECORD_GEOMETRY names an output jsonl path).
        self._record_path = os.environ.get("CARDREAMER_RECORD_GEOMETRY", "")
        self._episode_idx = getattr(self, "_episode_idx", -1) + 1

    def on_step(self) -> None:
        super().on_step()

        # Generate and sink car flow
        spawn = False
        if "min_flow_dist" in self._config:
            if len(self.actor_flow) == 0:
                spawn = True
            else:
                spawn_location = np.array(self._config.flow_spawn_point[:2])
                nearest_car_location = np.array(get_vehicle_pos(self.actor_flow[-1]))
                flow_dist = np.random.uniform(self._config.min_flow_dist, self._config.max_flow_dist)
                if get_location_distance(spawn_location, nearest_car_location) >= flow_dist:
                    spawn = True
        if spawn:
            vehicle = self._world.try_spawn_aggresive_actor(self.flow_transform)
            if vehicle is not None:
                self.actor_flow.append(vehicle)

        self.shared_intention_ids = self._select_shared_intentions()

    @staticmethod
    def _empty_comm_packet():
        return {"vehicles": {}, "intentions": {}, "anchors": {}}

    def get_state(self):
        # Snapshot lazily here (once per env step) rather than inside on_step:
        # querying the TM right after a same-step spawn/destroy inside on_step
        # can segfault the CARLA client. get_state runs after on_step settles,
        # which is where the renderer used to query the TM safely.
        if self._comm_snapshot_step != self._time_step:
            self._comm_snapshots.append(self._snapshot_comm_packet())
            self._comm_snapshot_step = self._time_step
        # Head of a full deque is the packet from latency_steps ago. An EMPTY
        # packet (not None!) during warm-up means "no broadcast received yet".
        if len(self._comm_snapshots) == self._comm_snapshots.maxlen:
            packet = self._comm_snapshots[0]
        else:
            packet = self._empty_comm_packet()
        if getattr(self, "_record_path", ""):
            self._record_geometry(packet)
        return {**super().get_state(), "comm_packet": packet}

    def _record_geometry(self, packet):
        """Append one per-frame geometry record for offline ECPG computation."""
        ego = self.get_ego_vehicle()
        rec = {
            "ep": self._episode_idx,
            "t": self._time_step,
            "latency": self._config.intention_sharing.latency_steps,
            "rule": self._config.intention_sharing.rule,
            "ego_pos": [float(v) for v in get_vehicle_pos(ego)],
            "ego_wp": [[float(w[0]), float(w[1])] for w in self.waypoints],
            "veh": {},          # all background vehicles: current pos + velocity
            "shared": {},       # delayed (t-k) intention plans for selected collaborators
        }
        transforms = self._world.actor_transforms
        for aid, tf in transforms.items():
            if aid == ego.id:
                continue
            actor = self._world.actor_dict.get(aid)
            v = get_vehicle_velocity(actor) if actor is not None else (0.0, 0.0)
            rec["veh"][str(aid)] = {"pos": [float(tf.location.x), float(tf.location.y)],
                                    "v": [float(v[0]), float(v[1])]}
        for aid, path in packet["intentions"].items():
            rec["shared"][str(aid)] = [[float(x), float(y)] for (x, y) in path]
        with open(self._record_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _snapshot_comm_packet(self):
        """
        Snapshot what the currently selected collaborators broadcast this step:
        the vehicles their own FOV can see (polygons at this step) and their own
        planned waypoints. The collaborator's own polygon is stored only as the
        anchor for drawing its intention path, never rendered as a vehicle box.
        """
        packet = self._empty_comm_packet()
        selected = [id for id in self.shared_intention_ids]
        if not selected:
            return packet
        sharing = self._config.intention_sharing
        transforms = self._world.actor_transforms
        polygons = self._world.actor_polygons
        actions = self._world.actor_actions
        for c in selected:
            if c not in transforms:
                continue
            visible = get_visibility_from(c, transforms, polygons, sharing.sight_fov, sharing.sight_range)
            for id, seen in visible.items():
                if seen:
                    packet["vehicles"][id] = polygons[id]
            if actions.get(c):
                packet["intentions"][c] = [(action[1].transform.location.x, action[1].transform.location.y) for action in actions[c]]
                packet["anchors"][c] = polygons[c]
        return packet

    def _intention_candidates(self):
        """Vehicles eligible to share their intention with the ego.

        ALL non-ego managed vehicles are candidates -- the car flow *and* the config-driven
        scenario background vehicles (scenario_actors) -- so any surrounding vehicle can be a
        collaborator, not just the flow. (Tasks without scenario actors are unaffected: their
        actor_dict is just ego + flow.)
        """
        ego_id = self.get_ego_vehicle().id
        return [actor for aid, actor in self._world.actor_dict.items() if aid != ego_id]

    def _select_shared_intentions(self):
        """
        Select which vehicles share their intentions with the ego vehicle.
        Only takes effect when the birdeye ``waypoint_obs`` is set to ``designated``.

        Configured by ``intention_sharing``:

        * ``rule``: one of ``all``, ``none``, ``nearest``, ``random``
        * ``num``: number of vehicles for ``nearest``/``random``
        """
        sharing = self._config.intention_sharing
        candidates = self._intention_candidates()
        if sharing.rule == "none" or not candidates:
            return []
        if sharing.rule == "all":
            return [v.id for v in candidates]
        if sharing.rule == "nearest":
            ego_pos = get_vehicle_pos(self.get_ego_vehicle())
            by_dist = sorted((get_location_distance(ego_pos, get_vehicle_pos(v)), v.id) for v in candidates)
            return [id for _, id in by_dist[: sharing.num]]
        if sharing.rule == "random":
            # Persist choices across steps; refill only when chosen vehicles despawn
            alive = {v.id for v in candidates}
            self._random_shared_ids &= alive
            refill = list(alive - self._random_shared_ids)
            random.shuffle(refill)
            while len(self._random_shared_ids) < sharing.num and refill:
                self._random_shared_ids.add(refill.pop())
            return list(self._random_shared_ids)
        raise ValueError(f"Unknown intention_sharing.rule: {sharing.rule}")
