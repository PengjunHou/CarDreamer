from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
EMULATION_ROOT = REPO_ROOT / "car_dreamer" / "toolkit" / "emulation"


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_visualization_module():
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", REPO_ROOT / "car_dreamer" / "toolkit")
    _ensure_pkg("car_dreamer.toolkit.emulation", EMULATION_ROOT)
    full_name = "car_dreamer.toolkit.emulation.visualization"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, EMULATION_ROOT / "visualization.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    _load_visualization_module().main()
