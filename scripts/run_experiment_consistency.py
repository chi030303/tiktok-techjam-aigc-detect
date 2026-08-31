#!/usr/bin/env python3
# 2026-08-30, yun, exp2 entrypoint: paired-view consistency recipes
"""Run one consistency-ablation experiment from experiments/<name>/recipe.yaml."""

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
    print(f"  consistency: {recipe.get('consistency')}")
    print(f"  gpu:       {recipe.get('gpu')}")

    if args.train:
        gpu = recipe.get("gpu")
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        from src.train.consistency_train import run_train_consistency

        run_train_consistency(recipe)
        return

    print("dry-run ok")


if __name__ == "__main__":
    main()
