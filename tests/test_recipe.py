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


def test_smoke_recipes_forbids_holdouts():
    root = Path(__file__).resolve().parents[1] / "experiments"
    for name in ("resnet50_linear_cifake_smoke", "dinov2l_linear_cifake_smoke"):
        recipe = load_recipe(root / name / "recipe.yaml")
        validate(recipe)
        assert "cifake" in recipe["train"]["datasets"]
        assert NO_TRAIN_NAMES <= set(recipe["train"]["forbid"])


def test_full_cifake_recipes_forbids_holdouts():
    # 2026-08-29, tianqi, full CIFAKE recipes must not train on val/evalgen
    root = Path(__file__).resolve().parents[1] / "experiments"
    for name in ("resnet50_linear_cifake", "dinov2l_linear_cifake"):
        recipe = load_recipe(root / name / "recipe.yaml")
        validate(recipe)
        assert "cifake" in recipe["train"]["datasets"]
        assert "smoke" not in recipe
        assert NO_TRAIN_NAMES <= set(recipe["train"]["forbid"])
    # end


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


def test_featcache_path_is_shared():
    # 2026-08-29, tianqi, feat cache is keyed by backbone+split, not experiment name
    from src.paths import feat_cache_path

    a = feat_cache_path("resnet-50", "train", 100000, 0, 224)
    b = feat_cache_path("resnet-50", "train", 100000, 0, 224)
    assert a == b
    assert a.name == "cifake_train_n100000_seed0_s224.pt"
    # end
# end
