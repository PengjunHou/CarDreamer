"""CLI wrapper for ``car_dreamer.toolkit.emulation.visualization.main``.

Renders ground-truth topology, region overview, world-region overview, and
prediction-comparison GIFs/PNGs from canonical emulation episode JSON. The
actual rendering code lives at ``car_dreamer/toolkit/emulation/visualization.py``
— this file just dodges carla imports by loading that module directly, then
hands argv to its ``main()``.

Usage (from repo root):

    python visualizations/emulation/render_topology.py \\
        --episode data/.../emulation_episode_*.json \\
        --output-dir logdir/topology_viz

    # Batch render across policies P*/ under a root:
    python visualizations/emulation/render_topology.py \\
        --policy-ids P1 P2 P3 --policy-root data/emulation_fixed_20260430 \\
        --output-dir logdir/topology_viz

    # Add --checkpoint to also render prediction-comparison GIFs.

See ``--help`` for the full flag set.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visualizations._common import load_visualization_module  # noqa: E402


if __name__ == "__main__":
    load_visualization_module().main()
