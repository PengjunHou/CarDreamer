"""Repository-root path constants used by every visualization script.

All scripts under ``visualizations/<subdir>/`` import these to avoid
recomputing ``Path(__file__).parents[N]`` independently.
"""
from __future__ import annotations

from pathlib import Path

# This file lives at <REPO_ROOT>/visualizations/_common/repo_path.py
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
EMULATION_ROOT: Path = REPO_ROOT / "car_dreamer" / "toolkit" / "emulation"
