# 2026-08-29, tianqi, shared Vast roots vs git clone; never put SID/CLIP inside a personal folder
"""Resolve shared data / models / experiment artifact dirs.

On Vast the volume layout is:

    /workspace/data          images (sid_set, cifake, val, evalgen)
    /workspace/models        frozen backbones
    /workspace/experiments   run artifacts (ckpts, logs)
    /workspace/<who>/...     git clone (code + recipes only)

Set DATA_ROOT / MODELS_ROOT / EXP_ROOT to override. Recipes live in the
repo under experiments/<name>/recipe.yaml and are the source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[-1]


def data_root() -> Path:
    env = os.environ.get("DATA_ROOT")
    if env:
        return Path(env)
    return _first_existing(Path("/workspace/data"), REPO_ROOT / "data")


def models_root() -> Path:
    env = os.environ.get("MODELS_ROOT")
    if env:
        return Path(env)
    return _first_existing(Path("/workspace/models"), REPO_ROOT / "models")


def exp_root() -> Path:
    env = os.environ.get("EXP_ROOT")
    if env:
        return Path(env)
    shared = Path("/workspace/experiments")
    if Path("/workspace").is_dir():
        shared.mkdir(parents=True, exist_ok=True)
        return shared
    local = REPO_ROOT / "experiments"
    local.mkdir(parents=True, exist_ok=True)
    return local


def recipe_dir(name: str) -> Path:
    return REPO_ROOT / "experiments" / name


def artifact_dir(name: str) -> Path:
    d = exp_root() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def feat_cache_path(backbone: str, split: str, n: int, seed: int, size: int = 224) -> Path:
    # 2026-08-29, tianqi, shared feat cache so lr/epoch sweeps skip backbone
    d = exp_root() / "_featcache" / backbone
    d.mkdir(parents=True, exist_ok=True)
    return d / f"cifake_{split}_n{n}_seed{seed}_s{size}.pt"
    # end


NO_TRAIN_NAMES = frozenset({"val", "evalgen", "demo_wildfake"})
# end
