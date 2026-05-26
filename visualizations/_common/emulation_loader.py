"""Load individual emulation modules without triggering carla/torch imports
from ``car_dreamer``'s top-level ``__init__``.

The emulation scripts under ``visualizations/emulation/`` cannot do a normal
``from car_dreamer.toolkit.emulation import training`` because importing the
package eagerly drags in carla. Instead we manually create empty package
shells (so submodule imports resolve) and exec each file by absolute path.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from .repo_path import EMULATION_ROOT, REPO_ROOT


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load(full_name: str, file_name: str):
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, EMULATION_ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ensure_emulation_pkgs() -> None:
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", REPO_ROOT / "car_dreamer" / "toolkit")
    _ensure_pkg("car_dreamer.toolkit.emulation", EMULATION_ROOT)


def load_training_module():
    """For GT-only scripts: schema + features + queries + training."""
    _ensure_emulation_pkgs()
    _load("car_dreamer.toolkit.emulation.schema", "schema.py")
    _load("car_dreamer.toolkit.emulation.features", "features.py")
    _load("car_dreamer.toolkit.emulation.queries", "queries.py")
    return _load("car_dreamer.toolkit.emulation.training", "training.py")


def load_emulation_modules():
    """For prediction scripts: dataset + training + model."""
    _ensure_emulation_pkgs()
    _load("car_dreamer.toolkit.emulation.schema", "schema.py")
    _load("car_dreamer.toolkit.emulation.features", "features.py")
    _load("car_dreamer.toolkit.emulation.queries", "queries.py")
    dataset_mod = _load("car_dreamer.toolkit.emulation.dataset", "dataset.py")
    training = _load("car_dreamer.toolkit.emulation.training", "training.py")
    model_mod = _load("car_dreamer.toolkit.emulation.model", "model.py")
    return dataset_mod, training, model_mod


def load_visualization_module():
    """For the topology renderer: the emulation package's visualization module."""
    _ensure_emulation_pkgs()
    return _load("car_dreamer.toolkit.emulation.visualization", "visualization.py")
