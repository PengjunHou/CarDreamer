import unittest

from car_dreamer.toolkit.scenario_actors import ScenarioActorManager, parse_scenario_specs


class ParseScenarioSpecsTest(unittest.TestCase):
    def test_empty_or_none(self):
        for cfg in (None, {}):
            specs = parse_scenario_specs(cfg)
            self.assertEqual(specs["vehicles"], [])
            self.assertEqual(specs["pedestrians"], [])
            self.assertAlmostEqual(specs["cross_factor"], 0.1)

    def test_vehicle_count_only_is_background(self):
        specs = parse_scenario_specs({"vehicles": [{"count": 8}]})
        group = specs["vehicles"][0]
        self.assertEqual(group["count"], 8)
        self.assertIsNone(group["start"])
        self.assertFalse(group["is_candidate"])
        self.assertFalse(group["stationary"])
        self.assertEqual(group["target_speed"], 25.0)  # default

    def test_start_without_destination_is_stationary_candidate(self):
        specs = parse_scenario_specs({"vehicles": [{"start": [1.0, 2.0, 0.1, 0.0]}]})
        group = specs["vehicles"][0]
        self.assertEqual(group["count"], 1)
        self.assertTrue(group["is_candidate"])
        self.assertTrue(group["stationary"])  # no destination -> stationary
        self.assertIsNone(group["destination"])
        self.assertEqual(group["target_speed"], 25.0)  # default

    def test_start_with_destination_is_moving_candidate(self):
        specs = parse_scenario_specs(
            {"vehicles": [{"count": 5, "start": [1.0, 2.0, 0.1, 90.0],
                           "destination": [3.0, 4.0, 0.1], "target_speed": 30, "ignore_lights": True}]}
        )
        group = specs["vehicles"][0]
        self.assertEqual(group["count"], 1)  # explicit start => single actor
        self.assertTrue(group["is_candidate"])
        self.assertFalse(group["stationary"])  # has destination
        self.assertEqual(group["destination"], [3.0, 4.0, 0.1])
        self.assertEqual(group["target_speed"], 30.0)
        self.assertTrue(group["ignore_lights"])

    def test_pedestrian_count_is_expanded(self):
        specs = parse_scenario_specs({"pedestrians": [{"count": 4}]})
        self.assertEqual(len(specs["pedestrians"]), 4)
        for walker in specs["pedestrians"]:
            self.assertIsNone(walker["start"])
            self.assertEqual(walker["on_arrival"], "keep")
            self.assertAlmostEqual(walker["max_speed"], 1.4)

    def test_pedestrian_with_start_and_destination(self):
        specs = parse_scenario_specs(
            {"pedestrians": [{"start": [1.0, 2.0, 0.3], "destination": [5.0, 2.0, 0.3],
                              "max_speed": 2.0, "on_arrival": "stop"}],
             "cross_factor": 0.5}
        )
        self.assertEqual(len(specs["pedestrians"]), 1)
        walker = specs["pedestrians"][0]
        self.assertEqual(walker["start"], [1.0, 2.0, 0.3])
        self.assertEqual(walker["on_arrival"], "stop")
        self.assertAlmostEqual(specs["cross_factor"], 0.5)


class _FakeWorld:
    """Minimal stand-in for WorldManager used to test ScenarioActorManager dispatch."""

    def __init__(self):
        self.vehicle_calls = []
        self.walker_calls = []
        self._next_id = 1000

    def spawn_scenario_vehicle(self, start=None, destination=None, target_speed=None,
                               ignore_lights=False, stationary=False):
        self.vehicle_calls.append(
            {"start": start, "destination": destination, "target_speed": target_speed, "stationary": stationary}
        )
        self._next_id += 1
        return self._next_id  # sentinel "actor"

    def spawn_scenario_walkers(self, specs, cross_factor=0.1):
        self.walker_calls.append((len(specs), cross_factor))
        return []


class ScenarioActorManagerTest(unittest.TestCase):
    def test_cooperative_hook_invoked_only_for_start_vehicles(self):
        cfg = {
            "vehicles": [
                {"count": 3},                                   # background (no hook)
                {"start": [1.0, 2.0, 0.1, 0.0]},                # candidate, stationary
                {"start": [4.0, 5.0, 0.1, 90.0], "destination": [9.0, 9.0, 0.1]},  # candidate, moving
            ]
        }
        world = _FakeWorld()
        registered = []
        manager = ScenarioActorManager(world, cfg, cooperative_hook=registered.append)
        manager.reset_spawn()

        # 3 background + 2 candidates = 5 vehicle spawns
        self.assertEqual(len(world.vehicle_calls), 5)
        # hook fired exactly for the 2 candidate (start) vehicles
        self.assertEqual(len(registered), 2)
        # stationary flags among the start-vehicle spawns: one stationary, one moving
        candidate_calls = [c for c in world.vehicle_calls if c["start"] is not None]
        self.assertEqual(sorted(c["stationary"] for c in candidate_calls), [False, True])
        # background spawns are never stationary and never candidates
        background_calls = [c for c in world.vehicle_calls if c["start"] is None]
        self.assertEqual(len(background_calls), 3)
        self.assertTrue(all(c["stationary"] is False for c in background_calls))

    def test_no_hook_means_no_registration(self):
        cfg = {"vehicles": [{"start": [1.0, 2.0, 0.1, 0.0]}]}
        world = _FakeWorld()
        manager = ScenarioActorManager(world, cfg, cooperative_hook=None)
        manager.reset_spawn()  # must not raise
        self.assertEqual(len(world.vehicle_calls), 1)


class LoadTaskConfigMergeTest(unittest.TestCase):
    def test_per_task_file_is_merged(self):
        import car_dreamer

        config = car_dreamer.load_task_configs("carla_group_right_turn_auto")
        scenario = getattr(config.env, "scenario_actors", None)
        self.assertIsNotNone(scenario, "configs/tasks/carla_group_right_turn_auto.yaml was not merged")
        specs = parse_scenario_specs(scenario)
        self.assertTrue(len(specs["vehicles"]) >= 1)
        self.assertTrue(len(specs["pedestrians"]) >= 1)
        # cooperative candidates are declared as start vehicles
        self.assertTrue(any(v["is_candidate"] for v in specs["vehicles"]))

    def test_cooperative_config_present_and_group_keys_removed(self):
        import car_dreamer

        env = car_dreamer.load_task_configs("carla_group_right_turn_auto").env
        # shared cooperative config stays at env top level
        self.assertEqual(env["coop_participation_prob"], 0.5)
        self.assertEqual(tuple(env["group_observation"]["enabled"]), ("camera", "collision"))
        self.assertEqual(env["group_observation"]["camera"]["fov"], 180)
        self.assertEqual(tuple(env["flow_spawn_point"]), (-3.4, -151.2, 0.1, 90.0))
        # obsolete group-spawn keys are gone
        self.assertNotIn("group_spawn_points", env)
        self.assertNotIn("num_group_vehs", env)
        self.assertNotIn("grouping_strategy", env)
        # perception/model blocks stay in tasks.yaml
        self.assertEqual(env["feature_size"], 3072)
        self.assertIn("graph", env)
        self.assertIn("wam", env)

    def test_other_right_turn_tasks_unaffected(self):
        import car_dreamer

        medium = car_dreamer.load_task_configs("carla_right_turn_medium").env
        self.assertEqual(tuple(medium["flow_spawn_point"]), (-3.4, -151.2, 0.1, 90.0))
        self.assertEqual(medium["min_flow_dist"], 8)


if __name__ == "__main__":
    unittest.main()
