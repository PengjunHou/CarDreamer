"""Offline unit tests for the streaming V2V communication process.

These exercise the Communication Model Specification end-to-end without CARLA: policy lifetime
(Td), sensor streaming (Ts), per-link sender queueing (proc + queue + tx), the receive-queue
window (Tw) + cross-policy filter, bundled multi-modality messages, and old-policy flush.
"""

import unittest

from car_dreamer.toolkit.communication.process import (
    CommConfig,
    CommPolicy,
    CommunicationProcess,
    ReceiveQueue,
    SenseSnapshot,
    make_local_policy,
)
from car_dreamer.toolkit.communication.comm import V2VMessage


REQUEST_ID = 1
COLLAB_ID = 2


def _coop_policy(policy_id=0, start=0, duration=10, modalities=("objlist",), bandwidth_ratio=1.0):
    return CommPolicy(
        policy_id=policy_id,
        request_vehicle_id=REQUEST_ID,
        start_step=start,
        duration_steps=duration,
        selected_collaborators=(COLLAB_ID,),
        modalities_by_vehicle={COLLAB_ID: tuple(modalities)},
        bandwidth_by_vehicle={COLLAB_ID: float(bandwidth_ratio)},
        reason="test",
    )


def _snapshot(payload_size=1000, modalities=("objlist",), data=None):
    return SenseSnapshot(
        sender_id=COLLAB_ID,
        distance_m=10.0,
        payload_size=int(payload_size),
        modalities=tuple(modalities),
        data=data if data is not None else {"objlist": {"observed_object_ids": (7,)}},
    )


def _fixed_rate(rate_bps):
    return lambda sender_id, distance_m, bandwidth_ratio: float(rate_bps)


class CommPolicyLifetimeTest(unittest.TestCase):
    def test_active_window_is_half_open(self):
        policy = _coop_policy(start=5, duration=3)
        self.assertEqual(policy.end_step, 8)
        self.assertFalse(policy.active_at(4))
        self.assertTrue(policy.active_at(5))
        self.assertTrue(policy.active_at(7))
        self.assertFalse(policy.active_at(8))  # expired exactly at end_step

    def test_local_only_policy_has_no_collaborators(self):
        local = make_local_policy(policy_id=3, request_vehicle_id=REQUEST_ID, start_step=0, duration_steps=10)
        self.assertTrue(local.is_local_only)
        self.assertEqual(local.selected_collaborators, ())


class CommConfigConversionTest(unittest.TestCase):
    def test_from_seconds_rounds_to_steps(self):
        cfg = CommConfig.from_seconds(
            dt=0.1,
            policy_duration_s=2.0,
            sensor_period_s=0.5,
            prediction_window_s=2.0,
            action_period_s=0.5,
            proc_delay_s=0.05,
        )
        self.assertEqual(cfg.policy_duration_steps, 20)
        self.assertEqual(cfg.sensor_period_steps, 5)
        self.assertEqual(cfg.prediction_window_steps, 20)
        self.assertEqual(cfg.action_period_steps, 5)
        # proc_delay stays continuous (seconds): it is summed with queue+tx and rounded once.
        self.assertAlmostEqual(cfg.proc_delay_s, 0.05)
        self.assertFalse(cfg.flush_old_policy_queue)


class BundledMessageTest(unittest.TestCase):
    def test_single_message_bundles_all_modalities(self):
        cfg = CommConfig(dt=0.1, sensor_period_steps=1, proc_delay_s=0.0, policy_duration_steps=10)
        proc = CommunicationProcess(cfg, REQUEST_ID)
        proc.set_policy(_coop_policy(modalities=("objlist", "bev")), step=0)
        data = {"objlist": {"observed_object_ids": (7, 8)}, "bev": "raster"}
        snap = _snapshot(payload_size=1234, modalities=("objlist", "bev"), data=data)
        emitted = proc.generate(0, {COLLAB_ID: snap}, _fixed_rate(1e7))
        self.assertEqual(len(emitted), 1)  # ONE message per collaborator, not one per modality
        msg = emitted[0]
        self.assertEqual(msg.modalities, ("objlist", "bev"))
        self.assertEqual(msg.payload_size, 1234)
        self.assertIn("objlist", msg.data)
        self.assertIn("bev", msg.data)
        # backward-compat aliases used by the legacy graph builder / scripts
        self.assertEqual(msg.created_step, msg.t_sense)
        self.assertEqual(msg.deliver_step, msg.t_recv)
        self.assertEqual(msg.payload_bytes, msg.payload_size)

    def test_zero_bandwidth_ratio_emits_no_message(self):
        cfg = CommConfig(dt=0.1, sensor_period_steps=1, proc_delay_s=0.0, policy_duration_steps=10)
        proc = CommunicationProcess(cfg, REQUEST_ID)
        proc.set_policy(_coop_policy(bandwidth_ratio=0.0), step=0)
        emitted = proc.generate(0, {COLLAB_ID: _snapshot(payload_size=10)}, _fixed_rate(0.0))
        self.assertEqual(emitted, [])
        self.assertEqual(proc.in_flight, [])


class SenderQueueingTest(unittest.TestCase):
    def _run_stream(self, *, payload_size, rate_bps, sensor_period_steps, proc_delay_s, n_ticks):
        cfg = CommConfig(
            dt=0.1,
            sensor_period_steps=sensor_period_steps,
            proc_delay_s=proc_delay_s,
            policy_duration_steps=1000,
        )
        proc = CommunicationProcess(cfg, REQUEST_ID)
        proc.set_policy(_coop_policy(duration=1000), step=0)
        rate = _fixed_rate(rate_bps)
        messages = []
        for tick in range(n_ticks):
            step = tick * sensor_period_steps
            messages.extend(proc.generate(step, {COLLAB_ID: _snapshot(payload_size=payload_size)}, rate))
        return messages

    def test_backlog_when_proc_plus_tx_exceeds_sensor_period(self):
        # tx_delay = 8 * 1000 / 26667 ~= 0.3 s = 3 steps; Ts = 1 step (0.1s) -> backlog.
        messages = self._run_stream(
            payload_size=1000, rate_bps=26667.0, sensor_period_steps=1, proc_delay_s=0.0, n_ticks=4
        )
        self.assertEqual(len(messages), 4)
        recvs = [m.t_recv for m in messages]
        self.assertEqual(recvs, sorted(recvs))  # monotonic non-decreasing delivery
        self.assertTrue(any(m.queue_delay > 0 for m in messages))  # queue builds up
        # later messages incur strictly larger queue delay than the first
        self.assertGreater(messages[-1].queue_delay, messages[0].queue_delay)

    def test_no_backlog_when_fast(self):
        # tx_delay = 8 * 10 / 16000 = 0.005 s -> 1 step; Ts = 2 steps -> link idle between ticks.
        messages = self._run_stream(
            payload_size=10, rate_bps=16000.0, sensor_period_steps=2, proc_delay_s=0.0, n_ticks=4
        )
        self.assertTrue(all(m.queue_delay == 0 for m in messages))

    def test_latency_is_summed_then_rounded_once(self):
        # proc=0.04s and tx=0.04s each round to 0 steps individually, but their SUM 0.08s rounds
        # to 1 step. The delivery step must reflect the rounded SUM, not per-component rounding.
        cfg = CommConfig(dt=0.1, sensor_period_steps=1, proc_delay_s=0.04, policy_duration_steps=10)
        proc = CommunicationProcess(cfg, REQUEST_ID)
        proc.set_policy(_coop_policy(), step=0)
        # tx = 8 * 10 / 2000 = 0.04 s
        emitted = proc.generate(0, {COLLAB_ID: _snapshot(payload_size=10)}, _fixed_rate(2000.0))
        msg = emitted[0]
        self.assertAlmostEqual(msg.total_latency, 0.08, places=6)  # exact proc + queue(0) + tx
        self.assertEqual(msg.t_recv - msg.t_sense, 1)  # round(0.08 / 0.1) = 1, not 0 + 0


class FlushOldPolicyTest(unittest.TestCase):
    def _process(self, flush):
        cfg = CommConfig(
            dt=0.1, sensor_period_steps=1, proc_delay_s=0.0, policy_duration_steps=10, flush_old_policy_queue=flush
        )
        proc = CommunicationProcess(cfg, REQUEST_ID)
        proc.set_policy(_coop_policy(policy_id=0), step=0)
        # slow link: tx ~= 1.0 s = 10 steps, so the message is still in flight at step 1.
        proc.generate(0, {COLLAB_ID: _snapshot(payload_size=1000)}, _fixed_rate(8000.0))
        self.assertEqual(len(proc.in_flight), 1)
        return proc

    def test_flush_false_drops_old_policy_messages_on_switch(self):
        proc = self._process(flush=False)
        proc.set_policy(_coop_policy(policy_id=1, start=1), step=1)
        self.assertEqual(len(proc.in_flight), 0)  # §9.1: old not-yet-delivered messages dropped

    def test_flush_true_keeps_old_policy_messages_on_switch(self):
        proc = self._process(flush=True)
        proc.set_policy(_coop_policy(policy_id=1, start=1), step=1)
        self.assertEqual(len(proc.in_flight), 1)  # §9.2: old messages keep transmitting


class ReceiveQueueFilterTest(unittest.TestCase):
    def _msg(self, *, policy_id, t_sense, t_recv):
        return V2VMessage(
            msg_id=0,
            policy_id=policy_id,
            sender_id=COLLAB_ID,
            receiver_id=REQUEST_ID,
            modalities=("objlist",),
            payload_size=100,
            data={},
            t_sense=t_sense,
            t_ready=t_sense,
            t_send=t_sense,
            t_recv=t_recv,
            total_latency=(t_recv - t_sense) * 0.1,
        )

    def test_window_filters_on_sensor_time(self):
        rq = ReceiveQueue()
        rq.add(self._msg(policy_id=0, t_sense=2, t_recv=3))   # fresh: in [step-Tw, step] = [2, 5]
        rq.add(self._msg(policy_id=0, t_sense=0, t_recv=1))   # too old: 0 < step-Tw = 2
        avail = rq.available(5, window_steps=3, active_policy_id=0, allow_cross_policy=False)
        self.assertEqual([m.t_sense for m in avail], [2])

    def test_not_yet_received_excluded(self):
        rq = ReceiveQueue()
        rq.add(self._msg(policy_id=0, t_sense=4, t_recv=12))  # arrives after step 10
        avail = rq.available(10, window_steps=20, active_policy_id=0, allow_cross_policy=False)
        self.assertEqual(avail, [])

    def test_cross_policy_filter(self):
        rq = ReceiveQueue()
        rq.add(self._msg(policy_id=0, t_sense=8, t_recv=9))   # old policy
        rq.add(self._msg(policy_id=1, t_sense=8, t_recv=9))   # active policy
        strict = rq.available(10, window_steps=20, active_policy_id=1, allow_cross_policy=False)
        self.assertEqual([m.policy_id for m in strict], [1])
        loose = rq.available(10, window_steps=20, active_policy_id=1, allow_cross_policy=True)
        self.assertEqual(sorted(m.policy_id for m in loose), [0, 1])


class LocalVsV2VSwitchingTest(unittest.TestCase):
    def test_local_only_policy_emits_nothing(self):
        cfg = CommConfig(dt=0.1, sensor_period_steps=1, proc_delay_s=0.0, policy_duration_steps=10)
        proc = CommunicationProcess(cfg, REQUEST_ID)
        proc.set_policy(make_local_policy(policy_id=0, request_vehicle_id=REQUEST_ID, start_step=0, duration_steps=10), step=0)
        self.assertFalse(proc.is_sensor_tick(0))
        self.assertEqual(proc.generate(0, {}, _fixed_rate(1e7)), [])
        self.assertEqual(proc.available_messages(0), [])  # -> ego rebuilds a local graph

    def test_v2v_messages_become_available_after_delivery(self):
        cfg = CommConfig(dt=0.1, sensor_period_steps=1, proc_delay_s=0.0, prediction_window_steps=50, policy_duration_steps=50)
        proc = CommunicationProcess(cfg, REQUEST_ID)
        proc.set_policy(_coop_policy(duration=50), step=0)
        # fast link: tx ~= 1 step, so it is delivered by step 2.
        proc.generate(0, {COLLAB_ID: _snapshot(payload_size=10)}, _fixed_rate(16000.0))
        self.assertEqual(proc.available_messages(0), [])  # not yet delivered
        proc.deliver(2)
        avail = proc.available_messages(2)
        self.assertEqual(len(avail), 1)
        self.assertEqual(avail[0].sender_id, COLLAB_ID)
        self.assertGreater(avail[0].total_latency, 0.0)  # measured L_M used for the veh_veh edge


if __name__ == "__main__":
    unittest.main()
