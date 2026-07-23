"""Offline tests for the config-driven background scenario actors (cars-only port from V4).

These exercise config parsing and the no-op guard without a live CARLA: ScenarioActorManager's
constructor only parses specs and never touches the simulator.
"""

import unittest

import car_dreamer
from car_dreamer.toolkit import ScenarioActorManager, parse_scenario_specs


class TestScenarioConfigLoads(unittest.TestCase):
    def test_group_right_turn_auto_reuses_policy_env(self):
        cfg = car_dreamer.load_task_configs("carla_group_right_turn_auto")
        # Reuses the policy-driven right-turn env; no BasicAgent/V2V env.
        self.assertEqual(cfg.env.name, "CarlaRightTurnEnv-v0")
        self.assertIn("scenario_actors", cfg.env)

    def test_scenario_specs_cars_only(self):
        cfg = car_dreamer.load_task_configs("carla_group_right_turn_auto")
        specs = parse_scenario_specs(dict(cfg.env.scenario_actors))
        placed = [s for s in specs["vehicles"] if s["placed"]]
        random_bg = [s for s in specs["vehicles"] if not s["placed"]]
        self.assertEqual(sum(s["count"] for s in random_bg), 8)   # 8 random background cars
        self.assertEqual(len(placed), 4)                          # 4 scripted vehicles
        self.assertEqual(sum(1 for s in placed if s["stationary"]), 1)      # 1 parked observer
        self.assertEqual(sum(1 for s in placed if s["autopilot_roam"]), 3)  # 3 roaming
        self.assertEqual(len(specs["pedestrians"]), 0)            # cars only
        self.assertEqual(specs["vehicle_start_mode"], "road")


class TestScenarioNoOp(unittest.TestCase):
    def test_non_scenario_task_has_no_block(self):
        cfg = car_dreamer.load_task_configs("carla_right_turn_hard")
        self.assertNotIn("scenario_actors", cfg.env)

    def test_manager_disabled_without_config(self):
        # world_manager is never touched when there is no scenario config.
        mgr = ScenarioActorManager(world_manager=None, scenario_config=None, cooperative_hook=None)
        self.assertFalse(mgr.enabled)
        mgr.reset_spawn()   # must be a safe no-op
        mgr.step_update()

    def test_manager_enabled_with_scenario(self):
        cfg = car_dreamer.load_task_configs("carla_group_right_turn_auto")
        mgr = ScenarioActorManager(world_manager=None, scenario_config=cfg.env.scenario_actors, cooperative_hook=None)
        self.assertTrue(mgr.enabled)


if __name__ == "__main__":
    unittest.main()
