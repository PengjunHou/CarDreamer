import importlib.util
import json
import logging
import sys
import tempfile
import types
import unittest
from collections import defaultdict, deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_ROOT = REPO_ROOT / "car_dreamer" / "toolkit"
EMULATION_ROOT = TOOLKIT_ROOT / "emulation"
VLM_ROOT = TOOLKIT_ROOT / "vlm"
RUNTIME_PATH = REPO_ROOT / "car_dreamer" / "right_turn_auto_runtime.py"


def _ensure_pkg(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
        return
    module.__path__ = [str(path)]


def _load_module(full_name: str, path: Path):
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_emulation_module(module_name: str):
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", TOOLKIT_ROOT)
    _ensure_pkg("car_dreamer.toolkit.communication", TOOLKIT_ROOT / "communication")
    _ensure_pkg("car_dreamer.toolkit.emulation", EMULATION_ROOT)
    return _load_module(
        f"car_dreamer.toolkit.emulation.{module_name}",
        EMULATION_ROOT / f"{module_name}.py",
    )


def _load_vlm_module(module_name: str):
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", TOOLKIT_ROOT)
    _ensure_pkg("car_dreamer.toolkit.communication", TOOLKIT_ROOT / "communication")
    _ensure_pkg("car_dreamer.toolkit.vlm", VLM_ROOT)
    return _load_module(
        f"car_dreamer.toolkit.vlm.{module_name}",
        VLM_ROOT / f"{module_name}.py",
    )


SCHEMA = _load_emulation_module("schema")
TRAINING = _load_emulation_module("training")
HELPER = _load_vlm_module("right_turn_auto_predictor_logging")


def _load_runtime_module():
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", TOOLKIT_ROOT)

    if "carla" not in sys.modules:
        carla_stub = types.ModuleType("carla")
        carla_stub.Actor = object
        carla_stub.TrafficLightState = types.SimpleNamespace(Green="Green")
        carla_stub.Transform = object
        carla_stub.Location = object
        carla_stub.Rotation = object
        sys.modules["carla"] = carla_stub

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available():
                return False

        torch_stub.cuda = _Cuda()
        sys.modules["torch"] = torch_stub

    if "agents" not in sys.modules:
        sys.modules["agents"] = types.ModuleType("agents")
    if "agents.navigation" not in sys.modules:
        sys.modules["agents.navigation"] = types.ModuleType("agents.navigation")
    if "agents.navigation.basic_agent" not in sys.modules:
        basic_agent_stub = types.ModuleType("agents.navigation.basic_agent")

        class BasicAgent:  # pragma: no cover - import stub
            def __init__(self, *args, **kwargs):
                pass

        basic_agent_stub.BasicAgent = BasicAgent
        sys.modules["agents.navigation.basic_agent"] = basic_agent_stub

    runtime_logging = types.ModuleType("runtime_logging")
    runtime_logging.get_runtime_logger = logging.getLogger
    runtime_logging.get_runtime_logging_config = lambda: {"step_debug_interval": 1000}
    runtime_logging.should_log_periodic = lambda *args, **kwargs: False
    sys.modules["runtime_logging"] = runtime_logging

    toolkit_pkg = sys.modules["car_dreamer.toolkit"]
    class _NetResource:
        def __init__(
            self,
            bandwidth_hz,
            tx_power_dbm=20.0,
            noise_figure_db=9.0,
            carrier_freq_hz=5.9e9,
        ):
            self.bandwidth_hz = bandwidth_hz
            self.tx_power_dbm = tx_power_dbm
            self.noise_figure_db = noise_figure_db
            self.carrier_freq_hz = carrier_freq_hz

    toolkit_pkg.NetResource = _NetResource
    toolkit_pkg.PayloadEncoder = object
    toolkit_pkg.PayloadSelectorDecision = lambda **kwargs: types.SimpleNamespace(**kwargs)
    toolkit_pkg.Observer = object
    toolkit_pkg.V2VMessage = object
    toolkit_pkg._dist_m = lambda *args, **kwargs: 0.0
    toolkit_pkg._tx_bytes_for_latency = lambda payload, overhead_bytes=64: int(overhead_bytes)
    toolkit_pkg.canonicalize_payload_type = lambda payload_type: str(payload_type or "tokens")
    toolkit_pkg.decode_payload_dict = lambda payload: {
        "payload_type": str(payload.get("payload_type", "tokens")),
        "payload_encoder_id": str(payload.get("payload_encoder_id", "tokens_v1")),
        "image": None,
        "scene_description": str(payload.get("scene_description", "")),
        "text": str(payload.get("text", "")),
        "data_nbytes": int(payload.get("data_nbytes", 0)),
    }
    toolkit_pkg.get_vehicle_pos = lambda actor: (0.0, 0.0)
    toolkit_pkg.payload_fn_llm = lambda *args, **kwargs: {}

    return _load_module("car_dreamer.right_turn_auto_runtime", RUNTIME_PATH)


RUNTIME = _load_runtime_module()


def _make_question_results(question_ids):
    results = {}
    for index, question_id in enumerate(question_ids):
        results[question_id] = {
            "question_id": question_id,
            "confidence_with_part2": 0.6 + 0.05 * index,
            "ego_plus_shared": {"confidence": 0.4 + 0.03 * index},
            "confidence_gain": 0.25 + 0.01 * index,
            "sender_importance_positive": {"101": 2.0, "202": 1.0},
            "sender_importance_negative": {},
            "per_sensor_scores": [
                {
                    "sender_id": 101,
                    "is_ego": False,
                    "positive_score": 0.7,
                    "negative_score": 0.1,
                    "unknown_score": 0.1,
                    "confidence": 0.55,
                    "belief": 0.45,
                    "evidence": 0.7,
                    "answerability_score": 1.0,
                    "visibility_score": 0.8,
                }
            ],
            "aggregated_details": {
                "per_sensor": [
                    {"sender_id": 101, "is_ego": False, "confidence": 0.35},
                ]
            },
        }
    return results


class RightTurnAutoPredictorLoggingTest(unittest.TestCase):
    def test_runtime_step_builder_keeps_all_group_vehicles_and_masks_missing_shared_blocks(self):
        question_ids = [
            "clg_left_rear_vehicle",
            "clg_right_rear_vehicle",
            "clg_right_front_vehicle",
            "clg_left_front_vehicle",
        ]
        step = HELPER.build_runtime_emulation_step(
            scene_id="right_turn_scene",
            episode_id="right_turn_episode_000001",
            scene_type="right_turn",
            policy_id="P5",
            predictor_step=0,
            env_step=7,
            dt=0.1,
            ego_pose={"x": 0.0, "y": 0.0, "yaw_rad": 0.0},
            ego_velocity={"vx": 4.0, "vy": 0.0},
            candidate_vehicle_states=[
                {
                    "vehicle_id": 101,
                    "pose": {"x": 8.0, "y": 1.5, "yaw_rad": 0.1},
                    "velocity": {"vx": 5.5, "vy": 0.1},
                    "selected_infos": [
                        {
                            "sender_id": 101,
                            "received_age_s": 0.2,
                            "feat_dim": 64,
                            "scene_description": "rear-right car visible",
                            "text": "vehicle approaching",
                        }
                    ],
                    "window_messages": [
                        {"received_age_s": 0.2, "latency_s": 0.1, "payload_bytes": 512, "distance_m": 8.2},
                        {"received_age_s": 0.3, "latency_s": 0.12, "payload_bytes": 640, "distance_m": 8.1},
                    ],
                    "shared_source": "received_feat",
                    "policy_action": {"alpha": 1.0, "nu": 1.0, "bandwidth": 0.7},
                    "payload_action": {"payload_type": "images", "payload_encoder_id": "images_v1"},
                    "runtime_comm_stats": {
                        "required_load_bps": 6400.0,
                        "link_rate_bps": 3200.0,
                        "comm_feasible": 0.0,
                        "dropped_capacity_exceeded": 1.0,
                    },
                    "policy_id": "P5",
                },
                {
                    "vehicle_id": 202,
                    "pose": {"x": -6.0, "y": -2.0, "yaw_rad": -0.2},
                    "velocity": {"vx": 3.2, "vy": 0.0},
                    "selected_infos": [],
                    "window_messages": [
                        {"received_age_s": 0.5, "latency_s": 0.2, "payload_bytes": 256, "distance_m": 6.5},
                    ],
                    "shared_source": "received_feat",
                    "policy_action": {"alpha": 0.0, "nu": 0.0, "bandwidth": 0.0},
                    "payload_action": {"payload_type": "tokens", "payload_encoder_id": "tokens_v1"},
                    "policy_id": "P5",
                },
            ],
            question_results=_make_question_results(question_ids),
            question_ids=question_ids,
            feature_size=64,
            step_communication_stats={
                "attempted_message_count": 1.0,
                "dropped_message_count": 1.0,
            },
            step_metadata={"reason": "high_collaboration_value", "overridden": False},
        )

        SCHEMA.validate_episode_record(
            HELPER.build_runtime_emulation_episode(
                scene_id="right_turn_scene",
                episode_id="right_turn_episode_000001",
                scene_type="right_turn",
                dt=0.1,
                policy_id="P5",
                steps=[step],
            )
        )

        self.assertEqual(step.step, 0)
        self.assertEqual(step.metadata["env_step"], 7)
        self.assertEqual(step.policy_id, "P5")
        self.assertEqual(step.metadata["policy_selector_reason"], "high_collaboration_value")
        self.assertEqual(step.metadata["payload_types_used"], ["images", "tokens"])
        self.assertEqual([vehicle.vehicle_id for vehicle in step.candidate_vehicles], [101, 202])
        self.assertEqual(set(step.ego_sc.keys()), set(question_ids))

        sender_with_evidence = step.candidate_vehicles[0]
        self.assertTrue(sender_with_evidence.component_valid_mask["shared_summary_raw"])
        self.assertTrue(sender_with_evidence.component_valid_mask["shared_summary_semantic"])
        self.assertGreater(sum(sender_with_evidence.shared_summary_raw), 0.0)
        self.assertGreater(sender_with_evidence.sender_gain["clg_left_rear_vehicle"], 0.0)
        self.assertEqual(sender_with_evidence.alpha, 1.0)
        self.assertEqual(sender_with_evidence.nu, 1.0)
        self.assertAlmostEqual(sender_with_evidence.bandwidth, 0.7)
        self.assertEqual(sender_with_evidence.payload_type, "images")
        self.assertEqual(sender_with_evidence.payload_encoder_id, "images_v1")
        self.assertEqual(sender_with_evidence.communication_stats["required_load_bps"], 6400.0)
        self.assertEqual(sender_with_evidence.communication_stats["link_rate_bps"], 3200.0)
        self.assertEqual(sender_with_evidence.communication_stats["comm_feasible"], 0.0)
        self.assertEqual(sender_with_evidence.communication_stats["dropped_capacity_exceeded"], 1.0)
        self.assertEqual(step.communication_stats["attempted_message_count"], 1.0)
        self.assertEqual(step.communication_stats["dropped_message_count"], 1.0)

        sender_without_evidence = step.candidate_vehicles[1]
        self.assertFalse(sender_without_evidence.component_valid_mask["shared_summary_raw"])
        self.assertFalse(sender_without_evidence.component_valid_mask["shared_summary_semantic"])
        self.assertFalse(sender_without_evidence.component_valid_mask["shared_confidence"])
        self.assertEqual(sender_without_evidence.shared_summary_raw, [0.0] * 8)
        self.assertEqual(sender_without_evidence.shared_summary_semantic, [0.0] * 8)
        self.assertEqual(sender_without_evidence.communication_stats["window_message_count"], 1.0)
        self.assertEqual(set(sender_without_evidence.query_task_relevance.keys()), set(question_ids))
        self.assertEqual(set(sender_without_evidence.sender_collab.keys()), set(question_ids))
        self.assertEqual(sender_without_evidence.alpha, 0.0)
        self.assertEqual(sender_without_evidence.nu, 0.0)
        self.assertEqual(sender_without_evidence.bandwidth, 0.0)
        self.assertEqual(sender_without_evidence.payload_type, "tokens")

    def test_runtime_dual_dump_writes_backward_compatible_vlm_log_and_trainable_episode(self):
        question_ids = [
            "clg_left_rear_vehicle",
            "clg_right_rear_vehicle",
            "clg_right_front_vehicle",
            "clg_left_front_vehicle",
        ]
        step = HELPER.build_runtime_emulation_step(
            scene_id="right_turn_scene",
            episode_id="right_turn_episode_000002",
            scene_type="right_turn",
            policy_id="P3",
            predictor_step=0,
            env_step=12,
            dt=0.1,
            ego_pose={"x": 1.0, "y": 2.0, "yaw_rad": 0.0},
            ego_velocity={"vx": 3.5, "vy": 0.0},
            candidate_vehicle_states=[
                {
                    "vehicle_id": 101,
                    "pose": {"x": 7.0, "y": 0.5, "yaw_rad": 0.0},
                    "velocity": {"vx": 4.0, "vy": 0.0},
                    "selected_infos": [
                        {
                            "sender_id": 101,
                            "received_age_s": 0.1,
                            "feat_dim": 32,
                            "scene_description": "front car visible",
                            "text": "clear lane",
                        }
                    ],
                    "window_messages": [
                        {"received_age_s": 0.1, "latency_s": 0.05, "payload_bytes": 256, "distance_m": 7.0},
                    ],
                    "shared_source": "received_feat",
                    "policy_action": {"alpha": 1.0, "nu": 1.0, "bandwidth": 1.0},
                    "payload_action": {"payload_type": "tokens", "payload_encoder_id": "tokens_v1"},
                    "policy_id": "P3",
                }
            ],
            question_results=_make_question_results(question_ids),
            question_ids=question_ids,
            feature_size=64,
        )

        class DummyRuntime(RUNTIME.RightTurnAutoRuntimeMixin):
            def __init__(self, dump_dir: str):
                self._dump_vlm_records_on_episode_end = True
                self._dump_emulation_records_on_episode_end = True
                self._vlm_dump_dir = dump_dir
                self._emulation_dump_dir = dump_dir
                self._vlm_episode_dumped = False
                self._emulation_episode_dumped = False
                self._time_step = 12
                self._vlm_records = [{"step": 12, "question_id": question_ids[0]}]
                self._emulation_episode_steps = [step]
                self._emulation_scene_id = "right_turn_scene"
                self._emulation_episode_id = "right_turn_episode_000002"
                self._emulation_scene_type = "right_turn"
                self._collaboration_policy_id = "P3"

            def dump_vlm_records(self, path: str) -> None:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(self._vlm_records, handle, ensure_ascii=False, indent=2)

            def dump_emulation_episode(self, path: str) -> None:
                episode = HELPER.build_runtime_emulation_episode(
                    scene_id=self._emulation_scene_id,
                    episode_id=self._emulation_episode_id,
                    scene_type=self._emulation_scene_type,
                    dt=0.1,
                    policy_id=self._collaboration_policy_id,
                    steps=self._emulation_episode_steps,
                    metadata={"source": "test"},
                )
                SCHEMA.validate_episode_record(episode)
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(SCHEMA.episode_to_dict(episode), handle, ensure_ascii=False, indent=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = DummyRuntime(tmpdir)
            info = runtime._handle_episode_end(True, False, {})

            self.assertIn("vlm_dump_path", info)
            self.assertIn("emulation_dump_path", info)
            self.assertTrue(Path(info["vlm_dump_path"]).exists())
            self.assertTrue(Path(info["emulation_dump_path"]).exists())

            with open(info["vlm_dump_path"], "r", encoding="utf-8") as handle:
                vlm_payload = json.load(handle)
            self.assertIsInstance(vlm_payload, list)
            episode = TRAINING.load_episode_from_path(info["emulation_dump_path"])
            SCHEMA.validate_episode_record(episode)
            self.assertEqual(episode.steps[0].metadata["env_step"], 12)
            self.assertEqual(len(episode.steps[0].candidate_vehicles), 1)
            self.assertEqual(episode.policy_id, "P3")
            self.assertEqual(episode.steps[0].candidate_vehicles[0].alpha, 1.0)
            self.assertEqual(episode.steps[0].candidate_vehicles[0].payload_type, "tokens")

    def test_runtime_episode_builder_marks_mixed_policy_and_payload_metadata(self):
        step_a = HELPER.build_runtime_emulation_step(
            scene_id="scene",
            episode_id="episode",
            scene_type="right_turn",
            policy_id="P3",
            predictor_step=0,
            env_step=1,
            dt=0.1,
            ego_pose={"x": 0.0, "y": 0.0, "yaw_rad": 0.0},
            ego_velocity={"vx": 0.0, "vy": 0.0},
            candidate_vehicle_states=[
                {
                    "vehicle_id": 7,
                    "pose": {"x": 4.0, "y": 0.0, "yaw_rad": 0.0},
                    "velocity": {"vx": 0.0, "vy": 0.0},
                    "selected_infos": [],
                    "window_messages": [],
                    "shared_source": "raw",
                    "policy_action": {"alpha": 1.0, "nu": 1.0, "bandwidth": 1.0},
                    "payload_action": {"payload_type": "tokens", "payload_encoder_id": "tokens_v1"},
                }
            ],
            question_results=_make_question_results(["clg_left_rear_vehicle"]),
            question_ids=["clg_left_rear_vehicle"],
            feature_size=64,
        )
        step_b = HELPER.build_runtime_emulation_step(
            scene_id="scene",
            episode_id="episode",
            scene_type="right_turn",
            policy_id="P6",
            predictor_step=1,
            env_step=2,
            dt=0.1,
            ego_pose={"x": 0.0, "y": 0.0, "yaw_rad": 0.0},
            ego_velocity={"vx": 0.0, "vy": 0.0},
            candidate_vehicle_states=[
                {
                    "vehicle_id": 7,
                    "pose": {"x": 4.0, "y": 0.0, "yaw_rad": 0.0},
                    "velocity": {"vx": 0.0, "vy": 0.0},
                    "selected_infos": [],
                    "window_messages": [],
                    "shared_source": "raw",
                    "policy_action": {"alpha": 1.0, "nu": 0.5, "bandwidth": 0.8},
                    "payload_action": {"payload_type": "images", "payload_encoder_id": "images_v1"},
                }
            ],
            question_results=_make_question_results(["clg_left_rear_vehicle"]),
            question_ids=["clg_left_rear_vehicle"],
            feature_size=64,
        )
        episode = HELPER.build_runtime_emulation_episode(
            scene_id="scene",
            episode_id="episode",
            scene_type="right_turn",
            dt=0.1,
            policy_id="P3",
            steps=[step_a, step_b],
        )
        self.assertEqual(episode.policy_id, "mixed")
        self.assertEqual(episode.metadata["policy_ids_used"], ["P3", "P6"])
        self.assertEqual(episode.metadata["payload_types_used"], ["images", "tokens"])
        self.assertEqual(episode.metadata["policy_switch_count"], 1)
        self.assertEqual(episode.metadata["payload_switch_count"], 1)

    def test_runtime_communication_drop_switch_controls_enqueue(self):
        class DummyActor:
            def __init__(self, actor_id: int):
                self.id = actor_id

            def get_transform(self):
                return types.SimpleNamespace(
                    location=types.SimpleNamespace(x=0.0, y=0.0),
                    rotation=types.SimpleNamespace(yaw=0.0),
                )

            def get_velocity(self):
                return types.SimpleNamespace(x=0.0, y=0.0)

        class DummyLatencyModel:
            overhead_bytes = 64

            def __init__(self, feasible: bool):
                self.feasible = feasible

            def analyze_transmission(self, **kwargs):
                return types.SimpleNamespace(
                    required_load_bps=6400.0,
                    link_rate_bps=3200.0 if not self.feasible else 12800.0,
                    shannon_bps=20000.0,
                    bandwidth_hz=1e6,
                    snr_db=12.0,
                    feasible=self.feasible,
                    latency_s=0.05,
                    distance_m=5.0,
                )

            def compute_latency_s(self, **kwargs):
                return 0.05

        class DummyRuntime(RUNTIME.RightTurnAutoRuntimeMixin):
            def __init__(self, *, drop_on_capacity_exceeded: bool, feasible: bool):
                self.ego = DummyActor(1)
                self.group_vehs = [DummyActor(2)]
                self.groups = {0: {1, 2}}
                self._time_step = 4
                self._world = types.SimpleNamespace(
                    _settings=types.SimpleNamespace(fixed_delta_seconds=0.1)
                )
                self._collaboration_policy_id = "P3"
                self._collaboration_bandwidth_floor = 0.1
                self._drop_on_capacity_exceeded = drop_on_capacity_exceeded
                self._log_dropped_messages = True
                self.latency_model = DummyLatencyModel(feasible=feasible)
                self._default_net_res = RUNTIME.NetResource(bandwidth_hz=1e6)
                self._veh_net_res = {}
                self._in_flight = []
                self._received = defaultdict(lambda: deque(maxlen=32))
                self._actor_cache = {}
                self._comm_link_analysis_by_sender = {}
                self._comm_step_summary = {}
                self._comm_step_summary_step = -1
                self._policy_prev_comm_summary = {}
                self.enqueued = []

            def _compute_current_policy_action(self):
                return types.SimpleNamespace(
                    alpha={2: 1.0},
                    nu={2: 0.8},
                    bandwidth={2: 1.0},
                )

            def _advance_policy_send_schedule(self):
                return {2: True}

            def _get_current_policy_decision(self):
                return {"policy_id": "P3", "reason": "fixed_policy_mode", "overridden": False}

            def _compute_current_payload_decisions(self):
                return {
                    2: types.SimpleNamespace(
                        payload_type="tokens",
                        payload_encoder_id="tokens_v1",
                        reason="default_tokens",
                        overridden=False,
                    )
                }

            def _build_group_actor_map(self):
                return {1: self.ego, 2: self.group_vehs[0]}

            def _make_payload(self, sender, payload_decision):
                return {
                    "feat": [1.0, 2.0],
                    "sender_id": int(sender.id),
                    "payload_type": payload_decision.payload_type,
                    "payload_encoder_id": payload_decision.payload_encoder_id,
                    "data": b"payload",
                    "data_nbytes": 7,
                }

            def _enqueue_message(self, **kwargs):
                self.enqueued.append(dict(kwargs))

        dropped_runtime = DummyRuntime(drop_on_capacity_exceeded=True, feasible=False)
        dropped_runtime._run_group_communication()
        self.assertEqual(len(dropped_runtime.enqueued), 0)
        self.assertEqual(dropped_runtime._get_current_comm_step_summary()["attempted_message_count"], 1.0)
        self.assertEqual(dropped_runtime._get_current_comm_step_summary()["dropped_message_count"], 1.0)
        dropped_stats = dropped_runtime._get_latest_comm_link_analysis(2)
        self.assertEqual(dropped_stats["comm_feasible"], 0.0)
        self.assertNotIn("uplink_bps", dropped_stats)
        self.assertNotIn("downlink_bps", dropped_stats)

        pass_runtime = DummyRuntime(drop_on_capacity_exceeded=False, feasible=False)
        pass_runtime._run_group_communication()
        self.assertEqual(len(pass_runtime.enqueued), 1)
        self.assertEqual(pass_runtime._get_current_comm_step_summary()["attempted_message_count"], 1.0)
        self.assertEqual(pass_runtime._get_current_comm_step_summary()["dropped_message_count"], 0.0)

    def test_runtime_normalizes_oversubscribed_bandwidth_before_analysis(self):
        class DummyActor:
            def __init__(self, actor_id: int):
                self.id = actor_id

            def get_transform(self):
                return types.SimpleNamespace(
                    location=types.SimpleNamespace(x=0.0, y=0.0),
                    rotation=types.SimpleNamespace(yaw=0.0),
                )

            def get_velocity(self):
                return types.SimpleNamespace(x=0.0, y=0.0)

        class CapturingLatencyModel:
            overhead_bytes = 64

            def __init__(self):
                self.calls = []

            def analyze_transmission(self, **kwargs):
                self.calls.append(dict(kwargs))
                return types.SimpleNamespace(
                    required_load_bps=3200.0,
                    link_rate_bps=6400.0,
                    shannon_bps=6400.0,
                    bandwidth_hz=float(kwargs["sender_res"].bandwidth_hz),
                    snr_db=12.0,
                    feasible=True,
                    latency_s=0.05,
                    distance_m=5.0,
                )

            def compute_latency_s(self, **kwargs):
                return 0.05

        class DummyRuntime(RUNTIME.RightTurnAutoRuntimeMixin):
            def __init__(self):
                self.ego = DummyActor(1)
                self.group_vehs = [DummyActor(2), DummyActor(3), DummyActor(4)]
                self.groups = {0: {1, 2, 3, 4}}
                self._time_step = 4
                self._world = types.SimpleNamespace(
                    _settings=types.SimpleNamespace(fixed_delta_seconds=0.1)
                )
                self._collaboration_policy_id = "P3"
                self._collaboration_bandwidth_floor = 0.1
                self._drop_on_capacity_exceeded = False
                self._log_dropped_messages = True
                self.latency_model = CapturingLatencyModel()
                self._default_net_res = RUNTIME.NetResource(bandwidth_hz=20e6)
                self._veh_net_res = {}
                self._in_flight = []
                self._received = defaultdict(lambda: deque(maxlen=32))
                self._actor_cache = {}
                self._comm_link_analysis_by_sender = {}
                self._comm_step_summary = {}
                self._comm_step_summary_step = -1
                self._policy_prev_comm_summary = {}
                self.enqueued = []

            def _compute_current_policy_action(self):
                return self._normalize_runtime_policy_action(
                    RUNTIME.CollaborationAction(
                        alpha={2: 1.0, 3: 1.0, 4: 1.0},
                        nu={2: 1.0, 3: 1.0, 4: 1.0},
                        bandwidth={2: 0.5, 3: 0.5, 4: 0.5},
                    )
                )

            def _advance_policy_send_schedule(self):
                return {2: True, 3: True, 4: True}

            def _get_current_policy_decision(self):
                return {"policy_id": "P3", "reason": "fixed_policy_mode", "overridden": False}

            def _compute_current_payload_decisions(self):
                return {
                    sender_id: types.SimpleNamespace(
                        payload_type="tokens",
                        payload_encoder_id="tokens_v1",
                        reason="default_tokens",
                        overridden=False,
                    )
                    for sender_id in (2, 3, 4)
                }

            def _build_group_actor_map(self):
                return {1: self.ego, 2: self.group_vehs[0], 3: self.group_vehs[1], 4: self.group_vehs[2]}

            def _make_payload(self, sender, payload_decision):
                return {
                    "feat": [1.0, 2.0],
                    "sender_id": int(sender.id),
                    "payload_type": payload_decision.payload_type,
                    "payload_encoder_id": payload_decision.payload_encoder_id,
                    "data": b"payload",
                    "data_nbytes": 7,
                }

            def _enqueue_message(self, **kwargs):
                self.enqueued.append(dict(kwargs))

        runtime = DummyRuntime()
        runtime._run_group_communication()

        self.assertEqual(len(runtime.enqueued), 3)
        self.assertEqual(len(runtime.latency_model.calls), 3)
        for call in runtime.latency_model.calls:
            self.assertAlmostEqual(call["sender_res"].bandwidth_hz, 20e6 / 3.0)
            self.assertAlmostEqual(call["receiver_res"].bandwidth_hz, 20e6 / 3.0)
            self.assertEqual(call["out_degree"], 1)
            self.assertEqual(call["in_degree"], 1)
        for item in runtime.enqueued:
            self.assertAlmostEqual(item["payload"]["policy_action"]["bandwidth"], 1.0 / 3.0)

if __name__ == "__main__":
    unittest.main()
