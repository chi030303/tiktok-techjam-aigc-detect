# 2026-08-29, tianqi, recipe must forbid official val and EvalGEN
from pathlib import Path

import yaml

from src.paths import NO_TRAIN_NAMES
from src.recipe import load_recipe, validate


def test_clipb16_recipe_forbids_holdouts():
    path = Path(__file__).resolve().parents[1] / "experiments/clipb16_linear_sid/recipe.yaml"
    recipe = load_recipe(path)
    validate(recipe)
    assert recipe["name"] == "clipb16_linear_sid"
    assert NO_TRAIN_NAMES <= set(recipe["train"]["forbid"])


def test_train_on_val_is_rejected():
    recipe = yaml.safe_load(
        """
name: bad
backbone: clip-vit-base-patch16
train:
  datasets: [val]
  forbid: [val, evalgen, demo_wildfake]
"""
    )
    try:
        validate(recipe)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit")
# end
