import unittest

import carla

from car_dreamer.toolkit.scenario_actors import ScenarioActorManager, parse_scenario_specs, spawn_problems


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
        self.assertFalse(group["autopilot_roam"])
        self.assertIsNone(group["destination"])
        self.assertEqual(group["target_speed"], 25.0)  # default

    def test_start_without_destination_can_roam_on_autopilot(self):
        specs = parse_scenario_specs({"vehicles": [{"start": [1.0, 2.0, 0.1, 0.0], "autopilot_roam": True}]})
        group = specs["vehicles"][0]
        self.assertTrue(group["is_candidate"])
        self.assertFalse(group["stationary"])
        self.assertTrue(group["autopilot_roam"])
        self.assertIsNone(group["destination"])

    def test_vehicle_can_override_start_mode(self):
        specs = parse_scenario_specs(
            {
                "vehicle_spawn_z_offset": 0.5,
                "vehicles": [{"start": [1.0, 2.0, 0.1, 0.0], "vehicle_start_mode": "exact"}],
            }
        )
        group = specs["vehicles"][0]
        self.assertEqual(group["vehicle_start_mode"], "exact")
        self.assertAlmostEqual(specs["vehicle_spawn_z_offset"], 0.5)

    def test_vehicle_can_override_spawn_z_offset(self):
        specs = parse_scenario_specs(
            {"vehicles": [{"start": [1.0, 2.0, 0.1, 0.0], "vehicle_spawn_z_offset": 0.25}]}
        )
        group = specs["vehicles"][0]
        self.assertAlmostEqual(group["vehicle_spawn_z_offset"], 0.25)

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

    def test_placed_vehicle_can_opt_out_of_cooperation(self):
        # Scripted traffic (a bicycle at a fixed spot) needs a start without becoming a
        # camera-equipped V2V collaborator.
        specs = parse_scenario_specs(
            {"vehicles": [{"start": [1.0, 2.0, 0.1], "cooperative": False, "autopilot_roam": True}]}
        )
        group = specs["vehicles"][0]
        self.assertTrue(group["placed"], "it still spawns at its start point")
        self.assertFalse(group["is_candidate"], "but it is not a cooperative candidate")
        self.assertEqual(group["count"], 1, "a start still means exactly one actor")
        self.assertFalse(group["stationary"])  # autopilot_roam

    def test_placed_vehicle_defaults_to_cooperative(self):
        specs = parse_scenario_specs({"vehicles": [{"start": [1.0, 2.0, 0.1]}]})
        self.assertTrue(specs["vehicles"][0]["is_candidate"])

    def test_cooperative_true_without_start_is_not_placed(self):
        specs = parse_scenario_specs({"vehicles": [{"count": 2, "cooperative": True}]})
        group = specs["vehicles"][0]
        self.assertFalse(group["placed"])
        self.assertEqual(group["count"], 2, "no start -> count still applies")

    def test_blueprint_accepts_a_string_or_a_list(self):
        one = parse_scenario_specs({"vehicles": [{"count": 1, "blueprint": "vehicle.gazelle.omafiets"}]})
        self.assertEqual(one["vehicles"][0]["blueprint"], ("vehicle.gazelle.omafiets",))
        many = parse_scenario_specs(
            {"vehicles": [{"count": 1, "blueprint": ["vehicle.a.b", "vehicle.c.*"]}]}
        )
        self.assertEqual(many["vehicles"][0]["blueprint"], ("vehicle.a.b", "vehicle.c.*"))

    def test_blueprint_defaults_to_empty_and_attributes_to_none(self):
        specs = parse_scenario_specs({"vehicles": [{"count": 1}]})
        self.assertEqual(specs["vehicles"][0]["blueprint"], ())
        self.assertIsNone(specs["vehicles"][0]["blueprint_attributes"])

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


class _FakeJunction:
    def __init__(self, junction_id):
        self.id = junction_id


class _FakeWaypoint:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, lane_width=3.5, junction=None):
        self.transform = carla.Transform(carla.Location(x=x, y=y, z=0.0), carla.Rotation(yaw=yaw))
        self.lane_width = lane_width
        self._junction = junction

    @property
    def is_junction(self):
        return self._junction is not None

    def get_junction(self):
        return self._junction


class _FakeMap:
    """Returns one canned waypoint for any query -- enough to drive spawn_problems offline."""

    def __init__(self, waypoint):
        self._waypoint = waypoint

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        return self._waypoint


def _pose(x=0.0, y=0.0, yaw=0.0):
    return carla.Transform(carla.Location(x=x, y=y, z=0.1), carla.Rotation(yaw=yaw))


class SpawnProblemsTest(unittest.TestCase):
    def test_aligned_point_on_a_plain_lane_is_clean(self):
        carla_map = _FakeMap(_FakeWaypoint(x=0.0, y=0.0, yaw=90.0))
        self.assertEqual(spawn_problems(carla_map, _pose(0.0, 0.0, 90.0)), [])

    def test_junction_is_reported_even_when_snap_distance_is_zero(self):
        # The regression this whole check exists for: the point sits exactly on a junction lane
        # centre, so the snap-distance log stays silent.
        carla_map = _FakeMap(_FakeWaypoint(x=0.0, y=0.0, yaw=90.0, junction=_FakeJunction(1221)))
        problems = spawn_problems(carla_map, _pose(0.0, 0.0, 90.0))
        self.assertEqual(problems, ["inside junction 1221"])

    def test_heading_against_the_lane_is_reported(self):
        carla_map = _FakeMap(_FakeWaypoint(x=0.0, y=0.0, yaw=90.0))
        problems = spawn_problems(carla_map, _pose(0.0, 0.0, -90.0))
        self.assertEqual(len(problems), 1)
        self.assertIn("off the lane direction", problems[0])

    def test_small_heading_difference_is_tolerated(self):
        carla_map = _FakeMap(_FakeWaypoint(x=0.0, y=0.0, yaw=90.0))
        self.assertEqual(spawn_problems(carla_map, _pose(0.0, 0.0, 100.0)), [])

    def test_yaw_wraparound_is_not_a_misalignment(self):
        carla_map = _FakeMap(_FakeWaypoint(x=0.0, y=0.0, yaw=179.0))
        self.assertEqual(spawn_problems(carla_map, _pose(0.0, 0.0, -179.0)), [])

    def test_lateral_offset_beyond_the_lane_is_reported(self):
        carla_map = _FakeMap(_FakeWaypoint(x=0.0, y=0.0, yaw=0.0, lane_width=3.5))
        problems = spawn_problems(carla_map, _pose(0.0, 3.0, 0.0))  # 3.0m > half of 3.5
        self.assertEqual(len(problems), 1)
        self.assertIn("off lane center", problems[0])

    def test_no_drivable_lane(self):
        self.assertEqual(spawn_problems(_FakeMap(None), _pose()), ["no drivable lane nearby"])

    def test_missing_map_or_transform_is_not_an_error(self):
        # WorldManager stand-ins in tests have no _map; validation must stay inert, not crash.
        self.assertEqual(spawn_problems(None, _pose()), [])
        self.assertEqual(spawn_problems(_FakeMap(_FakeWaypoint()), None), [])


class _FakeBlueprint:
    def __init__(self, bp_id, **attributes):
        self.id = bp_id
        self.attributes = attributes

    def __repr__(self):
        return f"<bp {self.id}>"


class _FakeWorld:
    """Minimal stand-in for WorldManager used to test ScenarioActorManager dispatch."""

    def __init__(self, carla_map=None, blueprints=()):
        self.vehicle_calls = []
        self.walker_calls = []
        self._next_id = 1000
        self._map = carla_map
        # blueprints: plain ids, or (id, {attr: value}) pairs
        self._blueprints = []
        for entry in blueprints:
            if isinstance(entry, tuple):
                bp_id, attributes = entry
            else:
                bp_id, attributes = entry, {}
            self._blueprints.append(_FakeBlueprint(bp_id, **attributes))

    def get_blueprint_library(self, pattern, attribute_filter=None):
        # Crude fnmatch-free stand-in: exact id or "prefix.*" wildcard, then attribute filter.
        if pattern.endswith("*"):
            matched = [b for b in self._blueprints if b.id.startswith(pattern[:-1])]
        else:
            matched = [b for b in self._blueprints if b.id == pattern]
        for name, value in (attribute_filter or {}).items():
            matched = [b for b in matched if b.attributes.get(name) == value]
        return matched

    def spawn_scenario_vehicle(self, start=None, destination=None, target_speed=None,
                               ignore_lights=False, stationary=False, blueprint=None):
        self.vehicle_calls.append(
            {"start": start, "destination": destination, "target_speed": target_speed,
             "stationary": stationary, "blueprint": blueprint}
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
                {"start": [6.0, 7.0, 0.1, 90.0], "autopilot_roam": True},  # candidate, TM roam
            ]
        }
        world = _FakeWorld()
        registered = []
        manager = ScenarioActorManager(world, cfg, cooperative_hook=registered.append)
        manager.reset_spawn()

        # 3 background + 3 candidates = 6 vehicle spawns
        self.assertEqual(len(world.vehicle_calls), 6)
        # explicit-start candidates spawn before random background traffic
        self.assertEqual([c["start"] is not None for c in world.vehicle_calls], [True, True, True, False, False, False])
        # hook fired exactly for the 3 candidate (start) vehicles
        self.assertEqual(len(registered), 3)
        # stationary flags among the start-vehicle spawns: one stationary, one moving
        candidate_calls = [c for c in world.vehicle_calls if c["start"] is not None]
        self.assertEqual(sorted(c["stationary"] for c in candidate_calls), [False, False, True])
        # background spawns are never stationary and never candidates
        background_calls = [c for c in world.vehicle_calls if c["start"] is None]
        self.assertEqual(len(background_calls), 3)
        self.assertTrue(all(c["stationary"] is False for c in background_calls))

    def test_placed_non_cooperative_vehicle_spawns_at_start_but_gets_no_hook(self):
        # The regression this guards: a bicycle given a fixed spot must NOT be pulled into the
        # cooperative pool (with coop_participation_prob=1 it would silently join every V2V result).
        cfg = {
            "vehicles": [
                {"start": [1.0, 2.0, 0.1]},                                  # real candidate
                {"start": [9.0, 9.0, 0.1], "cooperative": False},            # placed traffic
            ]
        }
        world = _FakeWorld()
        registered = []
        manager = ScenarioActorManager(world, cfg, cooperative_hook=registered.append)
        manager.reset_spawn()
        self.assertEqual(len(world.vehicle_calls), 2)
        self.assertTrue(all(c["start"] is not None for c in world.vehicle_calls),
                        "both are placed at their start point")
        self.assertEqual(len(registered), 1, "only the cooperative one registers")

    def test_blueprint_filter_is_passed_to_the_spawner(self):
        cfg = {"vehicles": [{"count": 2, "blueprint": "vehicle.gazelle.*"}]}
        world = _FakeWorld(blueprints=["vehicle.gazelle.omafiets", "vehicle.audi.tt"])
        ScenarioActorManager(world, cfg).reset_spawn()
        self.assertEqual(len(world.vehicle_calls), 2)
        for call in world.vehicle_calls:
            self.assertIsNotNone(call["blueprint"])
            self.assertEqual(call["blueprint"].id, "vehicle.gazelle.omafiets")

    def test_blueprint_attributes_filter_selects_by_base_type(self):
        # CARLA tags every vehicle blueprint with base_type (car/truck/van/bicycle/motorcycle),
        # which beats hard-coding model ids: it cannot silently miss one when CARLA adds a model.
        cfg = {
            "vehicles": [
                {"count": 4, "blueprint": "vehicle.*", "blueprint_attributes": {"base_type": "bicycle"}}
            ]
        }
        world = _FakeWorld(blueprints=[
            ("vehicle.gazelle.omafiets", {"base_type": "bicycle"}),
            ("vehicle.diamondback.century", {"base_type": "bicycle"}),
            ("vehicle.yamaha.yzf", {"base_type": "motorcycle"}),
            ("vehicle.audi.tt", {"base_type": "car"}),
        ])
        ScenarioActorManager(world, cfg).reset_spawn()
        self.assertEqual(len(world.vehicle_calls), 4)
        picked = {c["blueprint"].id for c in world.vehicle_calls}
        self.assertTrue(picked <= {"vehicle.gazelle.omafiets", "vehicle.diamondback.century"},
                        f"only bicycles may be picked, got {picked}")

    def test_unmatched_blueprint_falls_back_to_default_with_a_warning(self):
        cfg = {"vehicles": [{"count": 1, "blueprint": "vehicle.nonexistent.bike"}]}
        world = _FakeWorld(blueprints=["vehicle.audi.tt"])
        manager = ScenarioActorManager(world, cfg)
        with self.assertLogs("car_dreamer.scenario", level="WARNING") as logs:
            manager.reset_spawn()
        self.assertTrue(any("No blueprint matches" in line for line in logs.output))
        self.assertIsNone(world.vehicle_calls[0]["blueprint"], "None -> spawner uses its default")

    def test_no_blueprint_filter_leaves_the_spawner_default(self):
        world = _FakeWorld(blueprints=["vehicle.audi.tt"])
        ScenarioActorManager(world, {"vehicles": [{"count": 1}]}).reset_spawn()
        self.assertIsNone(world.vehicle_calls[0]["blueprint"])

    def test_no_hook_means_no_registration(self):
        cfg = {"vehicles": [{"start": [1.0, 2.0, 0.1, 0.0]}]}
        world = _FakeWorld()
        manager = ScenarioActorManager(world, cfg, cooperative_hook=None)
        manager.reset_spawn()  # must not raise
        self.assertEqual(len(world.vehicle_calls), 1)

    def test_exact_mode_start_is_validated(self):
        # `exact` skips projection; it must not also skip the geometry check (the cand1 regression:
        # a hand-typed pose parked inside the ego's junction, spawned with no diagnostic at all).
        junction_map = _FakeMap(_FakeWaypoint(x=1.0, y=2.0, yaw=0.0, junction=_FakeJunction(1221)))
        world = _FakeWorld(carla_map=junction_map)
        cfg = {"vehicles": [{"start": [1.0, 2.0, 0.1, 0.0], "vehicle_start_mode": "exact"}]}
        manager = ScenarioActorManager(world, cfg)
        with self.assertLogs("car_dreamer.scenario", level="WARNING") as logs:
            manager.reset_spawn()
        self.assertTrue(any("inside junction 1221" in line for line in logs.output))
        self.assertEqual(len(world.vehicle_calls), 1)  # still spawned; the check only warns

    def test_clean_start_produces_no_warning(self):
        clean_map = _FakeMap(_FakeWaypoint(x=1.0, y=2.0, yaw=0.0))
        world = _FakeWorld(carla_map=clean_map)
        cfg = {"vehicles": [{"start": [1.0, 2.0, 0.1, 0.0], "vehicle_start_mode": "exact"}]}
        manager = ScenarioActorManager(world, cfg)
        with self.assertNoLogs("car_dreamer.scenario", level="WARNING"):
            manager.reset_spawn()


class LoadTaskConfigMergeTest(unittest.TestCase):
    def test_per_task_file_is_merged(self):
        import car_dreamer

        config = car_dreamer.load_task_configs("carla_group_right_turn_auto")
        scenario = getattr(config.env, "scenario_actors", None)
        self.assertIsNotNone(scenario, "configs/tasks/carla_group_right_turn_auto.yaml was not merged")
        specs = parse_scenario_specs(scenario)
        self.assertTrue(len(specs["vehicles"]) >= 1)
        # cooperative candidates are declared as start vehicles
        self.assertTrue(any(v["is_candidate"] for v in specs["vehicles"]))
        # two-wheeler traffic is declared with a blueprint filter (the default filter is 4-wheel only)
        self.assertTrue(any(v["blueprint"] for v in specs["vehicles"]))
        # placed two-wheelers are traffic, not collaborators
        self.assertTrue(any(v["placed"] and not v["is_candidate"] for v in specs["vehicles"]))
        # pedestrians were removed from this task in favour of two-wheelers
        self.assertEqual(specs["pedestrians"], [])

    def test_cooperative_config_present_and_group_keys_removed(self):
        import car_dreamer

        env = car_dreamer.load_task_configs("carla_group_right_turn_auto").env
        # shared cooperative config stays at env top level
        self.assertEqual(env["coop_participation_prob"], 1)
        self.assertEqual(tuple(env["group_observation"]["enabled"]), ("camera", "collision", "birdeye_wpt"))
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
