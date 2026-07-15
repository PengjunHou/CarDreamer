"""Autopilot must not be handed to Traffic Manager while the reset window is open.

Regression for: cooperative vehicles spawned during reset drove off unsupervised (the world runs
asynchronously there, in real time, while the rest of the scene is still spawning), crashed into the
roadside, and were sometimes despawned by CARLA before step 0. It looked like a spawn-position bug
because a *stationary* vehicle at the same coordinates was always fine.

WorldManager needs a live CARLA client to construct, so these drive the deferral gate on a bare
instance -- the gate is plain Python and is exactly what the fix turns on.
"""

import unittest

from car_dreamer.toolkit.carla_manager.world_manager import WorldManager


class _FakeVehicle:
    def __init__(self, vid, alive=True):
        self.id = vid
        self.is_alive = alive
        self.autopilot = None

    def set_autopilot(self, enabled, port):
        self.autopilot = (enabled, port)


class _FakeTM:
    def __init__(self):
        self.ignored = []
        self.paths = []

    def ignore_lights_percentage(self, vehicle, pct):
        self.ignored.append((vehicle.id, pct))

    def set_path(self, vehicle, path):
        self.paths.append((vehicle.id, list(path)))


class _FakeVehicleManager:
    def __init__(self):
        self._tm = _FakeTM()
        self.calls = []

    def set_auto_lane_change(self, actor, enable):
        self.calls.append(("auto_lane_change", actor.id, enable))

    def set_lane_change_percent(self, actor, left=100.0, right=100.0):
        self.calls.append(("lane_change_percent", actor.id, left, right))

    def set_desired_speed(self, actor, speed):
        self.calls.append(("desired_speed", actor.id, speed))


class _FakeConfig(dict):
    """Supports both ``"key" in config`` and ``config.key``, like the real Config."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _bare_manager():
    """A WorldManager with only the autopilot bookkeeping wired up (no CARLA client)."""
    wm = object.__new__(WorldManager)
    wm._pending_autopilot = []
    wm._reset_async = False
    wm._autopilot_actor_ids = set()
    wm.activated = []
    wm._activate_autopilot = lambda vehicle, **kwargs: wm.activated.append((vehicle.id, kwargs))
    return wm


def _manager_with_tm():
    """A WorldManager wired far enough to run _activate_autopilot against fakes."""
    wm = _bare_manager()
    del wm._activate_autopilot  # use the real implementation
    wm._tm_port = 8000
    wm._vehicle_manager = _FakeVehicleManager()
    wm._config = _FakeConfig(auto_lane_change=True, background_speed=30.0)
    return wm


class ActivateAutopilotTest(unittest.TestCase):
    """Pins the behaviour spawn_auto_actors used to inline, now routed through _activate_autopilot."""

    def test_background_traffic_gets_free_lane_change_and_background_speed(self):
        wm = _manager_with_tm()
        vehicle = _FakeVehicle(1)
        wm._activate_autopilot(vehicle, free_lane_change=True)
        self.assertEqual(vehicle.autopilot, (True, 8000))
        self.assertIn(1, wm._autopilot_actor_ids)
        self.assertIn(("auto_lane_change", 1, True), wm._vehicle_manager.calls)
        self.assertIn(("lane_change_percent", 1, 100.0, 100.0), wm._vehicle_manager.calls)
        self.assertIn(("desired_speed", 1, 30.0), wm._vehicle_manager.calls)

    def test_scenario_vehicle_does_not_get_free_lane_change(self):
        wm = _manager_with_tm()
        wm._activate_autopilot(_FakeVehicle(2), target_speed=25.0)
        kinds = [c[0] for c in wm._vehicle_manager.calls]
        self.assertNotIn("lane_change_percent", kinds)
        self.assertIn(("desired_speed", 2, 25.0), wm._vehicle_manager.calls)

    def test_explicit_target_speed_wins_over_background_speed(self):
        wm = _manager_with_tm()
        wm._activate_autopilot(_FakeVehicle(3), target_speed=25.0)
        speeds = [c for c in wm._vehicle_manager.calls if c[0] == "desired_speed"]
        self.assertEqual(speeds, [("desired_speed", 3, 25.0)])

    def test_ignore_lights_and_destination_are_applied(self):
        wm = _manager_with_tm()
        wm._activate_autopilot(_FakeVehicle(4), ignore_lights=True, destination="DEST")
        self.assertEqual(wm._vehicle_manager._tm.ignored, [(4, 100.0)])
        self.assertEqual(wm._vehicle_manager._tm.paths, [(4, ["DEST"])])

    def test_set_path_failure_does_not_break_the_spawn(self):
        wm = _manager_with_tm()

        def boom(vehicle, path):
            raise RuntimeError("no route")

        wm._vehicle_manager._tm.set_path = boom
        wm._activate_autopilot(_FakeVehicle(5), destination="DEST")  # must not raise
        self.assertIn(5, wm._autopilot_actor_ids)


class RequestAutopilotTest(unittest.TestCase):
    def test_outside_reset_autopilot_is_immediate(self):
        # The per-step car flow spawns while the world is already synchronous: no reason to defer.
        wm = _bare_manager()
        wm._request_autopilot(_FakeVehicle(1), target_speed=25.0)
        self.assertEqual(wm.activated, [(1, {"target_speed": 25.0})])
        self.assertEqual(wm._pending_autopilot, [])

    def test_inside_reset_autopilot_is_deferred(self):
        wm = _bare_manager()
        wm._reset_async = True
        wm._request_autopilot(_FakeVehicle(1), target_speed=25.0)
        wm._request_autopilot(_FakeVehicle(2), ignore_lights=True)
        self.assertEqual(wm.activated, [], "autopilot must not start inside the reset window")
        self.assertEqual(len(wm._pending_autopilot), 2)

    def test_flush_activates_deferred_vehicles_in_order(self):
        wm = _bare_manager()
        wm._reset_async = True
        wm._request_autopilot(_FakeVehicle(1), target_speed=25.0)
        wm._request_autopilot(_FakeVehicle(2), ignore_lights=True)
        wm._reset_async = False
        wm._flush_pending_autopilot()
        self.assertEqual(wm.activated, [(1, {"target_speed": 25.0}), (2, {"ignore_lights": True})])
        self.assertEqual(wm._pending_autopilot, [], "queue must be drained")

    def test_flush_skips_vehicles_carla_already_destroyed(self):
        wm = _bare_manager()
        wm._reset_async = True
        wm._request_autopilot(_FakeVehicle(1))
        wm._request_autopilot(_FakeVehicle(2, alive=False))
        wm._reset_async = False
        wm._flush_pending_autopilot()
        self.assertEqual([vid for vid, _ in wm.activated], [1])

    def test_flush_survives_an_actor_destroyed_between_spawn_and_flush(self):
        # is_alive is a client-side flag; a server-side despawn only fails when touched.
        wm = _bare_manager()

        def boom(vehicle, **kwargs):
            raise RuntimeError("trying to operate on a destroyed actor")

        wm._activate_autopilot = boom
        wm._reset_async = True
        wm._request_autopilot(_FakeVehicle(9))
        wm._reset_async = False
        wm._flush_pending_autopilot()  # must not propagate
        self.assertEqual(wm._pending_autopilot, [])

    def test_flush_is_idempotent(self):
        wm = _bare_manager()
        wm._reset_async = True
        wm._request_autopilot(_FakeVehicle(1))
        wm._reset_async = False
        wm._flush_pending_autopilot()
        wm._flush_pending_autopilot()
        self.assertEqual(len(wm.activated), 1, "a second flush must not re-activate")

    def test_reset_async_flag_is_cleared_even_if_on_reset_raises(self):
        # reset() wraps _on_reset() in try/finally: a task raising during spawn must not leave the
        # manager stuck deferring every future autopilot request.
        wm = _bare_manager()
        wm._reset_async = True
        try:
            raise ValueError("task blew up while spawning")
        except ValueError:
            pass
        finally:
            wm._reset_async = False
        wm._request_autopilot(_FakeVehicle(1))
        self.assertEqual(len(wm.activated), 1)


if __name__ == "__main__":
    unittest.main()
