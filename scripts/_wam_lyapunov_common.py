"""Shared helpers for the Lyapunov analysis / ablation / argument scripts (imports the offline driver)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def driver():
    """The offline driver module (``build_synthetic_contexts`` / ``run_scheduler`` / ``load_perception_model``)."""
    import run_wam_lyapunov_offline as d

    return d


def scheduler_config(**kw):
    from car_dreamer.toolkit.wam import SchedulerConfig

    return SchedulerConfig(**kw)


def load_model(stage1_ckpt, device="cpu"):
    if stage1_ckpt is None:
        return None
    return driver().load_perception_model(Path(stage1_ckpt), device=device)


def contexts_for(model, *, steps, seed=0, **kw):
    """Build a synthetic episode whose graph matches ``model``'s ``route_waypoints`` (so U_φ input dims line up)."""
    rw = int(getattr(getattr(model, "config", None), "route_waypoints", 2)) if model is not None else 2
    return driver().build_synthetic_contexts(steps=int(steps), seed=int(seed), route_waypoints=rw, **kw)


def make_axes(nrows=1, ncols=1, figsize=(7.0, 4.0)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return plt, fig, axes


def save(plt, fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
