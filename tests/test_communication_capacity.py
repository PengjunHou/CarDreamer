import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_ROOT = REPO_ROOT / "car_dreamer" / "toolkit"
COMM_ROOT = TOOLKIT_ROOT / "communication"


def _ensure_pkg(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
        return
    module.__path__ = [str(path)]


def _load_comm_module():
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", TOOLKIT_ROOT)
    _ensure_pkg("car_dreamer.toolkit.communication", COMM_ROOT)

    if "carla" not in sys.modules:
        carla_stub = types.ModuleType("carla")
        carla_stub.Actor = object
        carla_stub.Location = object
        sys.modules["carla"] = carla_stub
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")

        class Tensor:  # pragma: no cover - import stub
            pass

        torch_stub.Tensor = Tensor
        sys.modules["torch"] = torch_stub

    full_name = "car_dreamer.toolkit.communication.comm"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, COMM_ROOT / "comm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMM = _load_comm_module()


class CommunicationCapacityTest(unittest.TestCase):
    def test_analyze_transmission_reports_required_load_and_shannon_rate(self):
        COMM._dist_m = lambda _sender, _receiver: 1.0
        model = COMM.SimpleWirelessLatency(
            overhead_base_s=0.0,
            overhead_per_kb_s=0.0,
            margin_db=0.0,
            jitter_s=0.0,
        )
        sender_res = COMM.NetResource(bandwidth_hz=10e6, tx_power_dbm=40.0)
        receiver_res = COMM.NetResource(bandwidth_hz=10e6, tx_power_dbm=40.0)

        analysis = model.analyze_transmission(
            sender=object(),
            receiver=object(),
            payload_size_bytes=1000,
            sender_res=sender_res,
            receiver_res=receiver_res,
            out_degree=1,
            in_degree=1,
            alpha=1.0,
            nu=1.0,
            fixed_dt=1.0e-5,
        )

        self.assertAlmostEqual(analysis.payload_size_bits, 8000.0)
        self.assertAlmostEqual(analysis.required_load_bps, 800000000.0, places=4)
        self.assertAlmostEqual(analysis.link_rate_bps, analysis.shannon_bps)
        self.assertFalse(analysis.feasible)
        self.assertGreater(analysis.latency_s, 0.0)

    def test_distance_and_policy_share_reduce_capacity(self):
        model = COMM.SimpleWirelessLatency(
            overhead_base_s=0.0,
            overhead_per_kb_s=0.0,
            margin_db=0.0,
            jitter_s=0.0,
        )
        sender_res = COMM.NetResource(bandwidth_hz=10e6, tx_power_dbm=20.0)
        receiver_res = COMM.NetResource(bandwidth_hz=10e6, tx_power_dbm=20.0)

        COMM._dist_m = lambda _sender, _receiver: 10.0
        nearby = model.analyze_transmission(
            sender=object(),
            receiver=object(),
            payload_size_bytes=1000,
            sender_res=sender_res,
            receiver_res=receiver_res,
            out_degree=1,
            in_degree=1,
        )
        COMM._dist_m = lambda _sender, _receiver: 60.0
        distant = model.analyze_transmission(
            sender=object(),
            receiver=object(),
            payload_size_bytes=1000,
            sender_res=sender_res,
            receiver_res=receiver_res,
            out_degree=1,
            in_degree=1,
        )
        distant_low_share = model.analyze_transmission(
            sender=object(),
            receiver=object(),
            payload_size_bytes=1000,
            sender_res=COMM.NetResource(bandwidth_hz=5e6, tx_power_dbm=20.0),
            receiver_res=COMM.NetResource(bandwidth_hz=5e6, tx_power_dbm=20.0),
            out_degree=2,
            in_degree=2,
        )

        self.assertGreater(nearby.link_rate_bps, distant.link_rate_bps)
        self.assertGreater(distant.link_rate_bps, distant_low_share.link_rate_bps)

    def test_bandwidth_analysis_matches_policy_share(self):
        model = COMM.SimpleWirelessLatency(
            overhead_base_s=0.0,
            overhead_per_kb_s=0.0,
            margin_db=0.0,
            jitter_s=0.0,
        )
        COMM._dist_m = lambda _sender, _receiver: 20.0
        shared_bandwidth_hz = 20e6 / 3.0
        sender_res = COMM.NetResource(bandwidth_hz=shared_bandwidth_hz, tx_power_dbm=20.0)
        receiver_res = COMM.NetResource(bandwidth_hz=shared_bandwidth_hz, tx_power_dbm=20.0)

        analysis = model.analyze_transmission(
            sender=object(),
            receiver=object(),
            payload_size_bytes=1000,
            sender_res=sender_res,
            receiver_res=receiver_res,
            out_degree=3,
            in_degree=1,
        )

        # Bandwidth is now used as-is (the upstream allocator has already split
        # the total band among selected members; no second distance-based decay).
        self.assertAlmostEqual(analysis.bandwidth_hz, shared_bandwidth_hz, places=4)

    def test_pathloss_model_and_overhead_drive_latency_size_dependence(self):
        COMM._dist_m = lambda _sender, _receiver: 1.0
        sender_res = COMM.NetResource(bandwidth_hz=10e6, tx_power_dbm=40.0)
        receiver_res = COMM.NetResource(bandwidth_hz=10e6, tx_power_dbm=40.0)
        model = COMM.SimpleWirelessLatency(
            overhead_base_s=0.030,
            overhead_per_kb_s=0.002,
            pathloss_model="urban_los",
            margin_db=0.0,
            jitter_s=0.0,
        )
        small = model.analyze_transmission(
            sender=object(), receiver=object(), payload_size_bytes=1024,
            sender_res=sender_res, receiver_res=receiver_res,
            out_degree=1, in_degree=1, alpha=1.0, nu=1.0, fixed_dt=0.1,
        )
        large = model.analyze_transmission(
            sender=object(), receiver=object(), payload_size_bytes=4096,
            sender_res=sender_res, receiver_res=receiver_res,
            out_degree=1, in_degree=1, alpha=1.0, nu=1.0, fixed_dt=0.1,
        )

        # Overhead grows linearly with payload KB on top of the fixed base.
        self.assertAlmostEqual(small.processing_delay_s, 0.030 + 0.002 * 1.0)
        self.assertAlmostEqual(large.processing_delay_s, 0.030 + 0.002 * 4.0)
        self.assertAlmostEqual(small.latency_s, small.processing_delay_s + small.tx_time_s)
        self.assertAlmostEqual(large.latency_s, large.processing_delay_s + large.tx_time_s)
        self.assertGreater(large.latency_s, small.latency_s)

    def test_nlos_pathloss_yields_lower_snr_than_los(self):
        COMM._dist_m = lambda _sender, _receiver: 100.0
        sender_res = COMM.NetResource(bandwidth_hz=10e6, tx_power_dbm=20.0)
        receiver_res = COMM.NetResource(bandwidth_hz=10e6, tx_power_dbm=20.0)
        los = COMM.SimpleWirelessLatency(
            overhead_base_s=0.0, overhead_per_kb_s=0.0,
            pathloss_model="urban_los", margin_db=0.0, jitter_s=0.0,
        )
        nlos = COMM.SimpleWirelessLatency(
            overhead_base_s=0.0, overhead_per_kb_s=0.0,
            pathloss_model="urban_nlos", margin_db=0.0, jitter_s=0.0,
        )
        common = dict(
            sender=object(), receiver=object(), payload_size_bytes=1000,
            sender_res=sender_res, receiver_res=receiver_res,
            out_degree=1, in_degree=1,
        )
        a_los = los.analyze_transmission(**common)
        a_nlos = nlos.analyze_transmission(**common)
        self.assertGreater(a_los.snr_db, a_nlos.snr_db)
        self.assertGreater(a_nlos.tx_time_s, a_los.tx_time_s)

    def test_unknown_pathloss_model_raises(self):
        with self.assertRaises(ValueError):
            COMM.SimpleWirelessLatency(pathloss_model="not_a_real_model")


if __name__ == "__main__":
    unittest.main()
