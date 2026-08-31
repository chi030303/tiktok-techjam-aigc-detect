# 2026-08-30, yun, model-track (DATA_ABLATION_PLAN.md §13) recipes must forbid holdouts
# and carry the fields their training loop expects.
from pathlib import Path

from src.paths import NO_TRAIN_NAMES
from src.recipe import load_recipe, validate

ROOT = Path(__file__).resolve().parents[1] / "experiments"


def _load(name: str) -> dict:
    recipe = load_recipe(ROOT / name / "recipe.yaml")
    validate(recipe)
    return recipe


def test_sid_aug_baseline_and_res336_share_setup_except_image_size():
    base = _load("clipb16_linear_sid_aug")
    res336 = _load("clipb16_linear_sid_res336")
    assert base["backbone"] == res336["backbone"] == "clip-vit-base-patch16"
    assert base["train"]["datasets"] == res336["train"]["datasets"] == ["sid_set"]
    assert NO_TRAIN_NAMES <= set(base["train"]["forbid"])
    assert NO_TRAIN_NAMES <= set(res336["train"]["forbid"])
    assert res336.get("image_size") == 336
    assert "image_size" not in base or base.get("image_size") in (None, 224)


def test_unfreeze_recipes_declare_partial_unfreeze_block():
    for name, n_layers in (("clipb16_linear_sid_unfreeze2", 2), ("clipb16_linear_sid_unfreeze4", 4)):
        recipe = _load(name)
        assert recipe["freeze_backbone"] is False
        pu = recipe["partial_unfreeze"]
        assert pu["n_layers"] == n_layers
        assert pu["backbone_lr"] < pu["head_lr"]


def test_consistency_recipe_declares_lambda():
    recipe = _load("clipb16_linear_sid_consistency")
    assert recipe["freeze_backbone"] is True
    assert recipe["consistency"]["lambda"] == 0.3


def test_dualbranch_recipe_declares_freq_dim():
    recipe = _load("clipb16_linear_sid_dualbranch")
    assert recipe["freeze_backbone"] is True
    assert recipe["dual_branch"]["freq_dim"] > 0


def test_cifake_full_recipes_cover_the_whole_split():
    # 2026-08-30, yun, "full" here means smoke.max_train/max_eval == the whole CIFAKE split,
    # not an absent smoke block -- see src/data/cifake.subsample_balanced
    for name in ("clipb16_linear_cifake_full", "clipl14_linear_cifake_full"):
        recipe = _load(name)
        assert "cifake" in recipe["train"]["datasets"]
        assert recipe["smoke"]["max_train"] == 100000
        assert recipe["smoke"]["max_eval"] == 20000
# end
