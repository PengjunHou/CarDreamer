"""Offline tests for the "designated" intention-sharing mode.

No CARLA server required: handler/env objects are constructed via __new__
to skip their CARLA-connected __init__. The legacy ``gym`` package (0.12.5)
is not installed in the test environment, so a minimal stub is injected
before importing car_dreamer.
"""

import sys
import types
import unittest

try:
    import gym  # noqa: F401
except ModuleNotFoundError:
    class _StubSpaces(types.ModuleType):
        def __getattr__(self, name):
            cls = type(name, (), {})
            setattr(self, name, cls)
            return cls

    gym = types.ModuleType("gym")
    spaces = _StubSpaces("gym.spaces")
    envs = types.ModuleType("gym.envs")
    registration = types.ModuleType("gym.envs.registration")
    registration.register = lambda *args, **kwargs: None
    envs.registration = registration
    gym.Env = type("Env", (), {})
    gym.spaces = spaces
    gym.envs = envs
    sys.modules["gym"] = gym
    sys.modules["gym.spaces"] = spaces
    sys.modules["gym.envs"] = envs
    sys.modules["gym.envs.registration"] = registration

from car_dreamer.carla_wpt_env import CarlaWptEnv
from car_dreamer.carla_wpt_fixed_env import CarlaWptFixedEnv
from car_dreamer.toolkit import Config
from car_dreamer.toolkit.observer.handlers.birdeye_handler import BirdeyeHandler
from car_dreamer.toolkit.observer.handlers.renderer.constants import Color
from car_dreamer.toolkit.observer.handlers.utils import WaypointObservability


class FakeActor:
    """Stands in for a carla.Actor: id + a transform with x/y location."""

    def __init__(self, id, x, y):
        self.id = id
        self._loc = types.SimpleNamespace(x=x, y=y)

    def get_transform(self):
        return types.SimpleNamespace(location=self._loc)


def make_handler(waypoint_obs):
    handler = BirdeyeHandler.__new__(BirdeyeHandler)
    handler._config = Config({"waypoint_obs": waypoint_obs, "color_by_obs": False})
    return handler


class TestWaypointObservabilityEnum(unittest.TestCase):
    def test_designated_value_exists(self):
        self.assertEqual(WaypointObservability("designated"), WaypointObservability.DESIGNATED)


class TestBackgroundWaypointsColor(unittest.TestCase):
    ACTOR_IDS = [1, 2, 3]
    ALL_VISIBLE = {1: True, 2: True, 3: True}

    def test_designated_whitelists_only_shared_ids(self):
        handler = make_handler("designated")
        colors = handler._get_background_waypoints_color(self.ACTOR_IDS, self.ALL_VISIBLE, [], shared_ids=[2])
        self.assertEqual(colors, {1: None, 2: Color.ORANGE_0, 3: None})

    def test_designated_empty_list_renders_nothing(self):
        handler = make_handler("designated")
        colors = handler._get_background_waypoints_color(self.ACTOR_IDS, self.ALL_VISIBLE, [])
        self.assertEqual(colors, {1: None, 2: None, 3: None})

    def test_designated_ignores_visibility(self):
        handler = make_handler("designated")
        none_visible = {1: False, 2: False, 3: False}
        colors = handler._get_background_waypoints_color(self.ACTOR_IDS, none_visible, [], shared_ids=[1, 3])
        self.assertEqual(colors, {1: Color.ORANGE_0, 2: None, 3: Color.ORANGE_0})

    def test_all_mode_unchanged(self):
        handler = make_handler("all")
        colors = handler._get_background_waypoints_color(self.ACTOR_IDS, self.ALL_VISIBLE, [])
        self.assertEqual(colors, {1: Color.ORANGE_0, 2: Color.ORANGE_0, 3: Color.ORANGE_0})

    def test_visible_mode_unchanged(self):
        handler = make_handler("visible")
        visible = {1: True, 2: False, 3: True}
        colors = handler._get_background_waypoints_color(self.ACTOR_IDS, visible, [])
        self.assertEqual(colors, {1: Color.ORANGE_0, 2: None, 3: Color.ORANGE_0})

    def test_neighbor_mode_unchanged(self):
        handler = make_handler("neighbor")
        colors = handler._get_background_waypoints_color(self.ACTOR_IDS, self.ALL_VISIBLE, [3])
        self.assertEqual(colors, {1: None, 2: None, 3: Color.ORANGE_0})


class TestWptEnvGetState(unittest.TestCase):
    def make_env(self):
        env = CarlaWptEnv.__new__(CarlaWptEnv)
        env.waypoints = []
        env._time_step = 7
        return env

    def test_defaults_to_empty_list(self):
        env = self.make_env()
        state = env.get_state()
        self.assertEqual(state["shared_intention_ids"], [])
        self.assertEqual(state["timesteps"], 7)

    def test_passes_through_shared_ids(self):
        env = self.make_env()
        env.shared_intention_ids = [42, 99]
        self.assertEqual(env.get_state()["shared_intention_ids"], [42, 99])


class TestSharedIntentionSelection(unittest.TestCase):
    def make_env(self, rule, num=1):
        env = CarlaWptFixedEnv.__new__(CarlaWptFixedEnv)
        env._config = Config({"intention_sharing": {"rule": rule, "num": num}})
        # ego at origin; flow vehicles at increasing distance
        env.ego = FakeActor(0, 0.0, 0.0)
        env.actor_flow = [FakeActor(1, 10.0, 0.0), FakeActor(2, 5.0, 0.0), FakeActor(3, 20.0, 0.0)]
        env._random_shared_ids = set()
        return env

    def test_rule_all(self):
        env = self.make_env("all")
        self.assertEqual(env._select_shared_intentions(), [1, 2, 3])

    def test_rule_none(self):
        env = self.make_env("none")
        self.assertEqual(env._select_shared_intentions(), [])

    def test_rule_nearest_picks_closest(self):
        env = self.make_env("nearest", num=1)
        self.assertEqual(env._select_shared_intentions(), [2])

    def test_rule_nearest_num2(self):
        env = self.make_env("nearest", num=2)
        self.assertEqual(env._select_shared_intentions(), [2, 1])

    def test_rule_random_persists_until_despawn(self):
        env = self.make_env("random", num=1)
        first = env._select_shared_intentions()
        self.assertEqual(len(first), 1)
        # same alive set -> same choice
        self.assertEqual(env._select_shared_intentions(), first)
        # chosen vehicle despawns -> refill from remaining
        env.actor_flow = [v for v in env.actor_flow if v.id != first[0]]
        second = env._select_shared_intentions()
        self.assertEqual(len(second), 1)
        self.assertNotEqual(second, first)

    def test_rule_unknown_raises(self):
        env = self.make_env("bogus")
        with self.assertRaises(ValueError):
            env._select_shared_intentions()

    def test_empty_flow_returns_empty(self):
        env = self.make_env("nearest")
        env.actor_flow = []
        self.assertEqual(env._select_shared_intentions(), [])


def make_transform(x, y, yaw=0.0):
    return types.SimpleNamespace(
        location=types.SimpleNamespace(x=x, y=y, z=0.0),
        rotation=types.SimpleNamespace(yaw=yaw),
    )


def square_poly(cx, cy, half=1.0):
    return [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]


class FakeWaypoint:
    def __init__(self, x, y):
        self.transform = types.SimpleNamespace(location=types.SimpleNamespace(x=x, y=y))


class TestGetVisibilityFrom(unittest.TestCase):
    """FOV visibility computed from an arbitrary observer vehicle's pose."""

    def setUp(self):
        from car_dreamer.toolkit.observer.handlers.utils import get_visibility_from

        self.get_visibility_from = get_visibility_from
        # observer 1 at origin facing +x; 2 in front; 3 out of range;
        # 4 behind the wide blocker 2 (occluded)
        self.transforms = {
            1: make_transform(0, 0),
            2: make_transform(10, 0),
            3: make_transform(100, 0),
            4: make_transform(20, 0),
        }
        self.polys = {
            1: square_poly(0, 0),
            2: square_poly(10, 0, half=3.0),
            3: square_poly(100, 0),
            4: square_poly(20, 0, half=0.5),
        }

    def test_front_target_visible(self):
        vis = self.get_visibility_from(1, self.transforms, self.polys, 150, 32)
        self.assertTrue(vis[2])

    def test_out_of_range_not_visible(self):
        vis = self.get_visibility_from(1, self.transforms, self.polys, 150, 32)
        self.assertFalse(vis[3])

    def test_occluded_not_visible(self):
        vis = self.get_visibility_from(1, self.transforms, self.polys, 150, 32)
        self.assertFalse(vis[4])

    def test_observer_excludes_itself(self):
        vis = self.get_visibility_from(1, self.transforms, self.polys, 150, 32)
        self.assertFalse(vis[1])


class TestCommPacket(unittest.TestCase):
    """Env-side communication packet: content, latency, and warm-up."""

    def make_env(self, latency_steps=0):
        from collections import deque

        env = CarlaWptFixedEnv.__new__(CarlaWptFixedEnv)
        env._config = Config(
            {"intention_sharing": {"rule": "all", "num": 1, "latency_steps": latency_steps, "sight_fov": 150, "sight_range": 32}}
        )
        env._comm_snapshots = deque(maxlen=latency_steps + 1)
        env.waypoints = []
        env._time_step = 0
        # equal to _time_step so get_state() does not lazily re-snapshot;
        # tests drive the buffer explicitly via step()
        env._comm_snapshot_step = 0
        env.shared_intention_ids = []
        return env

    def set_world(self, env, t):
        # collaborator 5 at origin facing +x sees vehicle 6 in front;
        # vehicle 7 is far behind ego and unseen; plan waypoint x encodes step t
        env._world = types.SimpleNamespace(
            actor_transforms={5: make_transform(0, 0), 6: make_transform(10, 0), 7: make_transform(200, 0)},
            actor_polygons={5: square_poly(0, 0), 6: square_poly(10, 0), 7: square_poly(200, 0)},
            actor_actions={5: [("cmd", FakeWaypoint(float(t), 0.0))], 6: [], 7: []},
        )

    def test_packet_contains_collaborator_perception_and_intention(self):
        env = self.make_env(0)
        self.set_world(env, 3)
        env.shared_intention_ids = [5]
        packet = env._snapshot_comm_packet()
        self.assertIn(6, packet["vehicles"])  # seen by collaborator
        self.assertNotIn(7, packet["vehicles"])  # out of collaborator's range
        self.assertNotIn(5, packet["vehicles"])  # own box never shared
        self.assertEqual(packet["intentions"], {5: [(3.0, 0.0)]})
        self.assertIn(5, packet["anchors"])

    def test_empty_selection_gives_empty_packet(self):
        env = self.make_env(0)
        self.set_world(env, 0)
        env.shared_intention_ids = []
        self.assertEqual(env._snapshot_comm_packet(), {"vehicles": {}, "intentions": {}, "anchors": {}})

    def step(self, env, t, selected):
        self.set_world(env, t)
        env.shared_intention_ids = selected
        env._comm_snapshots.append(env._snapshot_comm_packet())

    def test_warmup_returns_empty_packet_not_none(self):
        env = self.make_env(2)
        self.step(env, 0, [5])
        packet = env.get_state()["comm_packet"]
        self.assertIsNotNone(packet)
        self.assertEqual(packet["intentions"], {})

    def test_latency_two_returns_old_packet_and_old_selection(self):
        env = self.make_env(2)
        self.step(env, 0, [5])  # t=0: selected 5
        self.step(env, 1, [])  # t=1: selected nobody
        self.step(env, 2, [5])  # t=2: selected 5 again
        # packet arriving now was broadcast at t=0 by the then-selected 5
        packet = env.get_state()["comm_packet"]
        self.assertEqual(packet["intentions"], {5: [(0.0, 0.0)]})
        self.step(env, 3, [5])
        # t=3 receives the t=1 packet: nobody was selected then
        packet = env.get_state()["comm_packet"]
        self.assertEqual(packet["intentions"], {})

    def test_zero_latency_matches_live(self):
        env = self.make_env(0)
        self.step(env, 4, [5])
        packet = env.get_state()["comm_packet"]
        self.assertEqual(packet["intentions"], {5: [(4.0, 0.0)]})


class TestRendererCommContract(unittest.TestCase):
    """Missing packet -> legacy live query; empty packet -> render nothing."""

    def make_renderer(self):
        import numpy as np

        from car_dreamer.toolkit.observer.handlers.renderer.birdeye_renderer import BirdeyeRenderer

        r = BirdeyeRenderer.__new__(BirdeyeRenderer)
        r._world_manager = types.SimpleNamespace(
            actor_actions={2: [("cmd", FakeWaypoint(3.0, 4.0))]},
            actor_polygons={2: square_poly(10, 10)},
        )
        r._ego = FakeActor(1, 0.0, 0.0)
        r._surface = np.zeros((64, 64, 3), dtype=np.uint8)
        r._world_to_pixel = lambda loc: (int(loc.x), int(loc.y))
        return r

    def test_missing_packet_falls_back_to_live(self):
        r = self.make_renderer()
        intentions, anchors = r._get_intentions_and_anchors({})
        self.assertEqual(intentions, {2: [(3.0, 4.0)]})
        self.assertIs(anchors, r._world_manager.actor_polygons)

    def test_empty_packet_renders_nothing(self):
        r = self.make_renderer()
        packet = {"vehicles": {}, "intentions": {}, "anchors": {}}
        intentions, anchors = r._get_intentions_and_anchors({"comm_packet": packet})
        self.assertEqual(intentions, {})

    def test_packet_paths_are_copied(self):
        r = self.make_renderer()
        packet = {"vehicles": {}, "intentions": {9: [(1.0, 2.0)]}, "anchors": {9: square_poly(1, 2)}}
        intentions, _ = r._get_intentions_and_anchors({"comm_packet": packet})
        intentions[9].append((99.0, 99.0))
        self.assertEqual(packet["intentions"], {9: [(1.0, 2.0)]})

    def test_comm_vehicles_drawn_from_stale_polygons(self):
        r = self.make_renderer()
        packet = {"vehicles": {2: square_poly(30, 30, half=3.0)}, "intentions": {}, "anchors": {}}
        r._render_comm_vehicles(comm_packet=packet, background_vehicles_color={2: None})
        self.assertGreater(int(r._surface.sum()), 0)

    def test_comm_vehicles_skip_locally_visible(self):
        from car_dreamer.toolkit.observer.handlers.renderer.constants import Color

        r = self.make_renderer()
        packet = {"vehicles": {2: square_poly(30, 30, half=3.0)}, "intentions": {}, "anchors": {}}
        r._render_comm_vehicles(comm_packet=packet, background_vehicles_color={2: Color.GREEN})
        self.assertEqual(int(r._surface.sum()), 0)

    def test_comm_vehicles_noop_without_packet(self):
        r = self.make_renderer()
        r._render_comm_vehicles(background_vehicles_color={})
        self.assertEqual(int(r._surface.sum()), 0)


if __name__ == "__main__":
    unittest.main()
