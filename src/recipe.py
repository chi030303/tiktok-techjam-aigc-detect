# 2026-08-29, tianqi, load and validate experiment recipes
from __future__ import annotations

from pathlib import Path

import yaml

from src.paths import NO_TRAIN_NAMES

# end


def load_recipe(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "name" not in data:
        raise SystemExit(f"invalid recipe: {path}")
    return data


def validate(recipe: dict) -> None:
    # 2026-08-29, tianqi, official val + EvalGEN must never enter the train loader
    train = recipe.get("train") or {}
    datasets = [str(x) for x in train.get("datasets") or []]
    forbid = {str(x) for x in train.get("forbid") or []}
    missing = NO_TRAIN_NAMES - forbid
    if missing:
        raise SystemExit(f"recipe.train.forbid must include {sorted(NO_TRAIN_NAMES)}, missing {sorted(missing)}")
    leak = [d for d in datasets if d in NO_TRAIN_NAMES]
    if leak:
        raise SystemExit(f"refusing to train on {leak}")
    # end
