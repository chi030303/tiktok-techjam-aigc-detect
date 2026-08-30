#!/usr/bin/env python3
# 2026-08-30, yun, entrypoint for SID + online-aug recipes (baseline reproduction, exp4 res336)
"""Run one src.train.sid_online experiment from experiments/<name>/recipe.yaml.

Separate from scripts/run_experiment.py because that one dispatches to
src.train.loop.run_train, which only supports the CIFAKE feature-cache path.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import artifact_dir, data_root, models_root, recipe_dir
from src.recipe import load_recipe, validate

# end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    recipe_path = args.recipe
    if not recipe_path.is_file():
        alt = recipe_dir(recipe_path.name) / "recipe.yaml"
        if alt.is_file():
            recipe_path = alt
        else:
            raise SystemExit(f"recipe not found: {args.recipe}")

    recipe = load_recipe(recipe_path)
    validate(recipe)
    name = recipe["name"]
    art = artifact_dir(name)
    bb = models_root() / recipe["backbone"]

    print(f"experiment: {name}")
    print(f"  recipe:    {recipe_path}")
    print(f"  data:      {data_root()}")
    print(f"  models:    {models_root()}")
    print(f"  backbone:  {bb}  exists={bb.is_dir()}")
    print(f"  artifacts: {art}")
    print(f"  image_size: {recipe.get('image_size')}")
    print(f"  gpu:       {recipe.get('gpu')}")

    if args.train:
        gpu = recipe.get("gpu")
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        from src.train.sid_online import run_train_sid_online

        run_train_sid_online(recipe)
        return

    print("dry-run ok")


if __name__ == "__main__":
    main()
