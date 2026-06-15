"""CARLA-free V2V communication process (Communication Model Specification).

This module realizes the streaming cooperative-perception communication pipeline:

    Policy -> MessageGeneration(Ts) -> SenderQueue(proc+queue) -> Transmission(tx)
           -> ReceiveQueue -> Window(Tw) -> Graph

It is intentionally simulator-agnostic: it operates purely on integer simulation **steps**,
vehicle ids, payload sizes and an injected ``link_rate_bps(sender_id, distance_m, bandwidth_hz)``
callable, so it is unit-testable without CARLA. :class:`V2VCommMixin` supplies the CARLA glue
(actor poses for distance, per-collaborator observation snapshots, and the Base-Station policy).

Key objects
-----------
* :class:`CommConfig`  -- the four time parameters (Td/Ts/Tw/Ta), T_proc and the two queue
  switches, all expressed in **steps** plus ``dt`` (seconds-per-step).
* :class:`CommPolicy`  -- a Base-Station cooperative-perception policy with an explicit lifetime
  ``[start_step, end_step)``; ``local-only`` is modelled as a policy with no collaborators (§2.1).
* :class:`SenderQueue` -- a per-link (collaborator m -> request vehicle q) FIFO that models
  processing + queueing + transmission delay (§5-§8).
* :class:`ReceiveQueue` -- the request vehicle's buffer; ``available()`` applies the Tw / cross-policy
  filter (§10-§11).
* :class:`CommunicationProcess` -- the orchestrator stepped once per simulation step.

Per the project deviation from the design doc, a collaborator emits **one** bundled message per
sensor tick carrying *all* its policy modalities (see :class:`~..comm.V2VMessage`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .comm import V2VMessage

# link_rate_bps(sender_id, distance_m, bandwidth_hz) -> bits/second
LinkRateFn = Callable[[int, float, float], float]


@dataclass
class CommConfig:
    """Communication timing config.

    The *cadence / window* parameters (Td/Ts/Tw/Ta) index discrete simulation steps, so each is
    rounded to whole steps once at construction. ``proc_delay`` is a *latency component* and is
    kept in **seconds**: it is summed with the (continuous) queueing and transmission delay and
    the total is rounded to a delivery step only **once** (see :meth:`CommunicationProcess.generate`).
    """

    dt: float = 0.1                       # seconds per simulation step
    policy_duration_steps: int = 20       # Td  (steps)
    sensor_period_steps: int = 5          # Ts  (steps)
    prediction_window_steps: int = 20     # Tw  (steps)
    action_period_steps: int = 5          # Ta  (steps)
    proc_delay_s: float = 0.05            # T_proc (seconds; not pre-rounded to steps)
    flush_old_policy_queue: bool = False  # §9: drop vs keep old-policy queued messages on switch
    allow_cross_policy_messages: bool = False  # §11: use only the active policy's messages

    @classmethod
    def from_seconds(
        cls,
        *,
        dt: float,
        policy_duration_s: float,
        sensor_period_s: float,
        prediction_window_s: float,
        action_period_s: float,
        proc_delay_s: float,
        flush_old_policy_queue: bool = False,
        allow_cross_policy_messages: bool = False,
    ) -> "CommConfig":
        dt = max(float(dt), 1e-6)

        def steps(seconds: float, minimum: int) -> int:
            return max(int(round(float(seconds) / dt)), int(minimum))

        return cls(
            dt=dt,
            policy_duration_steps=steps(policy_duration_s, 1),
            sensor_period_steps=steps(sensor_period_s, 1),
            prediction_window_steps=steps(prediction_window_s, 1),
            action_period_steps=steps(action_period_s, 1),
            proc_delay_s=float(proc_delay_s),  # latency component: kept continuous, summed then rounded once
            flush_old_policy_queue=bool(flush_old_policy_queue),
            allow_cross_policy_messages=bool(allow_cross_policy_messages),
        )


@dataclass(frozen=True)
class CommPolicy:
    """A Base-Station cooperative-perception policy with an explicit lifetime (§2.1, §3).

    ``modalities_by_vehicle`` maps each selected collaborator to the **tuple** of modalities it
    streams (bundled into a single message). A ``local-only`` policy has no collaborators (§2.1).
    """

    policy_id: int
    request_vehicle_id: int
    start_step: int
    duration_steps: int
    selected_collaborators: Tuple[int, ...] = ()
    modalities_by_vehicle: Dict[int, Tuple[str, ...]] = field(default_factory=dict)
    bandwidth_by_vehicle: Dict[int, float] = field(default_factory=dict)
    reason: str = ""

    @property
    def end_step(self) -> int:
        return int(self.start_step) + int(self.duration_steps)

    @property
    def is_local_only(self) -> bool:
        return len(self.selected_collaborators) == 0

    def active_at(self, step: int) -> bool:
        return int(self.start_step) <= int(step) < self.end_step


def make_local_policy(*, policy_id: int, request_vehicle_id: int, start_step: int, duration_steps: int) -> CommPolicy:
    """The unified ``local-only`` policy π_local = (∅, 0, ∅) (§2.1)."""
    return CommPolicy(
        policy_id=int(policy_id),
        request_vehicle_id=int(request_vehicle_id),
        start_step=int(start_step),
        duration_steps=int(duration_steps),
        selected_collaborators=(),
        modalities_by_vehicle={},
        bandwidth_by_vehicle={},
        reason="local_only",
    )


@dataclass
class SenseSnapshot:
    """One collaborator's bundled observation produced at a sensor tick.

    The mixin fills ``data`` with per-modality payloads plus ``"feat"``/``"pose"``/``"vel"`` for
    the legacy vehicle-node graph; ``payload_size`` is the total transmitted byte count.
    """

    sender_id: int
    distance_m: float
    payload_size: int
    modalities: Tuple[str, ...]
    data: Dict[str, Any]


@dataclass
class SenderQueue:
    """Per-link (m -> q) sender queue (§5-§7).

    ``busy_until`` is the wall-clock time (in **seconds**) the link stays busy transmitting the
    current message; the next message can only start sending at/after it. Keeping it continuous
    (rather than discretized per message) is what lets the per-message latency be summed exactly.
    """

    sender_id: int
    busy_until: float = -1e18  # seconds the link is busy until (never busy initially)

    def reset(self, t_seconds: float) -> None:
        self.busy_until = float(t_seconds)


class ReceiveQueue:
    """Request vehicle's receive buffer + Tw / cross-policy filtering (§10-§11)."""

    def __init__(self) -> None:
        self._messages: List[V2VMessage] = []

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> List[V2VMessage]:
        return self._messages

    def add(self, message: V2VMessage) -> None:
        self._messages.append(message)

    def evict(self, step: int, window_steps: int) -> None:
        """Drop messages whose sensor time fell out of the prediction window."""
        oldest = int(step) - int(window_steps)
        self._messages = [m for m in self._messages if int(m.t_sense) >= oldest]

    def available(
        self,
        step: int,
        *,
        window_steps: int,
        active_policy_id: Optional[int],
        allow_cross_policy: bool,
    ) -> List[V2VMessage]:
        """Messages usable for graph construction at prediction time ``step`` (§11).

        Conditions: received (``t_recv <= step``), fresh by sensor time
        (``step - Tw <= t_sense <= step``) and -- unless ``allow_cross_policy`` -- belonging to
        the active policy.
        """
        oldest = int(step) - int(window_steps)
        out: List[V2VMessage] = []
        for m in self._messages:
            if int(m.t_recv) > int(step):
                continue
            if not (oldest <= int(m.t_sense) <= int(step)):
                continue
            if not allow_cross_policy and active_policy_id is not None and int(m.policy_id) != int(active_policy_id):
                continue
            out.append(m)
        return out


class CommunicationProcess:
    """Orchestrates policy lifetime, sensor streaming, sender queues and delivery."""

    def __init__(self, config: CommConfig, request_vehicle_id: int) -> None:
        self.config = config
        self.request_vehicle_id = int(request_vehicle_id)
        self.policy: Optional[CommPolicy] = None
        self.receive_queue = ReceiveQueue()
        self._sender_queues: Dict[int, SenderQueue] = {}
        self._in_flight: List[V2VMessage] = []
        self._msg_id = 0

    # ----- introspection (scripts / logging) -----
    @property
    def in_flight(self) -> List[V2VMessage]:
        return self._in_flight

    @property
    def active_policy_id(self) -> Optional[int]:
        return None if self.policy is None else int(self.policy.policy_id)

    def reset(self) -> None:
        self.policy = None
        self.receive_queue = ReceiveQueue()
        self._sender_queues = {}
        self._in_flight = []
        self._msg_id = 0

    # ----- policy lifecycle (§3, §9) -----
    def set_policy(self, policy: CommPolicy, step: int) -> None:
        """Install ``policy`` as the active policy, applying the old-policy flush rule (§9)."""
        start_s = int(step) * self.config.dt
        old = self.policy
        if old is not None and not self.config.flush_old_policy_queue:
            # §9.1: drop the expired policy's not-yet-delivered messages and free its links.
            self._in_flight = [m for m in self._in_flight if int(m.policy_id) != int(old.policy_id)]
            for queue in self._sender_queues.values():
                queue.reset(start_s)
        # else §9.2: keep old in-flight messages; queues stay busy so new messages queue behind them.

        self.policy = policy
        for sender_id in policy.selected_collaborators:
            self._sender_queues.setdefault(int(sender_id), SenderQueue(int(sender_id), busy_until=start_s))

    # ----- message generation at sensor ticks (§2.2, §6) -----
    def is_sensor_tick(self, step: int) -> bool:
        policy = self.policy
        if policy is None or policy.is_local_only or not policy.active_at(step):
            return False
        return (int(step) - int(policy.start_step)) % int(self.config.sensor_period_steps) == 0

    def generate(self, step: int, snapshots: Dict[int, SenseSnapshot], link_rate_bps: LinkRateFn) -> List[V2VMessage]:
        """Produce one bundled message per active collaborator at a sensor tick."""
        if not self.is_sensor_tick(step):
            return []
        policy = self.policy
        assert policy is not None
        dt = self.config.dt
        proc_delay_s = float(self.config.proc_delay_s)
        emitted: List[V2VMessage] = []
        for sender_id in policy.selected_collaborators:
            snap = snapshots.get(int(sender_id))
            if snap is None:
                continue
            bandwidth_hz = float(policy.bandwidth_by_vehicle.get(int(sender_id), 0.0))
            rate_bps = max(float(link_rate_bps(int(sender_id), float(snap.distance_m), bandwidth_hz)), 1.0)
            queue = self._sender_queues.setdefault(
                int(sender_id), SenderQueue(int(sender_id), busy_until=int(step) * dt)
            )

            # Continuous-time timeline (seconds): proc, queue and tx are summed exactly; the total
            # latency is rounded to a delivery step only ONCE (no per-component rounding).
            t_sense_step = int(step)
            t_sense_s = t_sense_step * dt
            t_ready_s = t_sense_s + proc_delay_s
            t_send_s = max(t_ready_s, float(queue.busy_until))
            tx_delay_s = 8.0 * float(snap.payload_size) / rate_bps
            t_recv_s = t_send_s + tx_delay_s
            queue.busy_until = t_recv_s
            total_latency_s = t_recv_s - t_sense_s  # = proc + queue + tx
            deliver_step = t_sense_step + max(int(round(total_latency_s / dt)), 0)

            message = V2VMessage(
                msg_id=self._next_msg_id(),
                policy_id=int(policy.policy_id),
                sender_id=int(sender_id),
                receiver_id=int(policy.request_vehicle_id),
                modalities=tuple(snap.modalities),
                payload_size=int(snap.payload_size),
                data=dict(snap.data),
                t_sense=t_sense_step,
                t_ready=t_ready_s,
                t_send=t_send_s,
                t_recv=deliver_step,
                proc_delay=proc_delay_s,
                queue_delay=t_send_s - t_ready_s,
                tx_delay=float(tx_delay_s),
                total_latency=total_latency_s,
                distance_m=float(snap.distance_m),
            )
            self._in_flight.append(message)
            emitted.append(message)
        return emitted

    # ----- transmission completion (§8, §10) -----
    def deliver(self, step: int) -> List[V2VMessage]:
        """Move messages whose transmission has completed into the receive queue."""
        delivered: List[V2VMessage] = []
        remaining: List[V2VMessage] = []
        for m in self._in_flight:
            if int(m.t_recv) <= int(step):
                self.receive_queue.add(m)
                delivered.append(m)
            else:
                remaining.append(m)
        self._in_flight = remaining
        self.receive_queue.evict(step, self.config.prediction_window_steps)
        return delivered

    # ----- window-based selection for graph construction (§11-§14) -----
    def available_messages(self, step: int) -> List[V2VMessage]:
        return self.receive_queue.available(
            step,
            window_steps=int(self.config.prediction_window_steps),
            active_policy_id=self.active_policy_id,
            allow_cross_policy=bool(self.config.allow_cross_policy_messages),
        )

    def _next_msg_id(self) -> int:
        mid = self._msg_id
        self._msg_id += 1
        return mid
