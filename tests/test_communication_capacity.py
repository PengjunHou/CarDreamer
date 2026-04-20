import importlib.util
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
    def test_analyze_transmission_reports_required_load_and_capacity(self):
        COMM._dist_m = lambda _sender, _receiver: 1.0
        model = COMM.SimpleWirelessLatency(base_rtt_s=0.0, proc_delay_s=0.0, jitter_s=0.0)
        sender_res = COMM.NetResource(
            uplink_bps=12345.0,
            downlink_bps=999999.0,
            bandwidth_hz=10e6,
            tx_power_dbm=40.0,
        )
        receiver_res = COMM.NetResource(
            uplink_bps=999999.0,
            downlink_bps=12345.0,
            bandwidth_hz=10e6,
            tx_power_dbm=40.0,
        )

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
            fixed_dt=0.1,
        )

        self.assertAlmostEqual(analysis.payload_size_bits, 8000.0)
        self.assertAlmostEqual(analysis.required_load_bps, 80000.0)
        self.assertAlmostEqual(analysis.link_rate_bps, 12345.0)
        self.assertFalse(analysis.feasible)
        self.assertGreater(analysis.latency_s, 0.0)


if __name__ == "__main__":
    unittest.main()
