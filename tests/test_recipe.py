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


def test_online_aug_recipes_forbids_holdouts():
    # 2026-08-29, tianqi, online-aug CIFAKE recipes still must not train on val
    root = Path(__file__).resolve().parents[1] / "experiments"
    for name in (
        "resnet50_linear_cifake_aug",
        "dinov2l_linear_cifake_aug",
        "clipb16_linear_cifake_aug",
        "clipl14_linear_cifake_aug",
    ):
        recipe = load_recipe(root / name / "recipe.yaml")
        validate(recipe)
        assert recipe["aug"]["online"] is True
        assert NO_TRAIN_NAMES <= set(recipe["train"]["forbid"])
    # end


def test_six_op_expand_recipes_forbids_holdouts():
    # 2026-08-29, tianqi, six-op expand recipes still must not train on val
    root = Path(__file__).resolve().parents[1] / "experiments"
    for name in (
        "resnet50_linear_cifake_aug6",
        "dinov2l_linear_cifake_aug6",
        "clipb16_linear_cifake_aug6",
        "clipl14_linear_cifake_aug6",
    ):
        recipe = load_recipe(root / name / "recipe.yaml")
        validate(recipe)
        assert recipe["aug"]["online"] is True
        assert recipe["aug"]["expand"] == "six_ops"
        assert NO_TRAIN_NAMES <= set(recipe["train"]["forbid"])
    # end


def test_mlp_fft_sid_recipes_forbids_holdouts():
    # 2026-08-29, tianqi, overnight mlp/fft/SID recipes still must not train on val
    # 2026-08-30, tianqi, CLIP mlp/fft/SID fill-in uses the same forbid + head/mode
    root = Path(__file__).resolve().parents[1] / "experiments"
    checks = {
        "dinov2l_linear_cifake_mlp": ("mlp", "rgb", "cifake"),
        "dinov2l_linear_cifake_fft": ("linear", "fft", "cifake"),
        "resnet50_linear_sid_aug": ("linear", "rgb", "sid_set"),
        "dinov2l_linear_sid_aug": ("linear", "rgb", "sid_set"),
        "clipb16_linear_cifake_mlp": ("mlp", "rgb", "cifake"),
        "clipl14_linear_cifake_mlp": ("mlp", "rgb", "cifake"),
        "clipb16_linear_cifake_fft": ("linear", "fft", "cifake"),
        "clipl14_linear_cifake_fft": ("linear", "fft", "cifake"),
        "clipb16_linear_sid_aug": ("linear", "rgb", "sid_set"),
        "clipl14_linear_sid_aug": ("linear", "rgb", "sid_set"),
        "dinov2l_linear_sid": ("linear", "rgb", "sid_set"),
        "dinov2l_linear_sid_mlp": ("mlp", "rgb", "sid_set"),
        "dinov2l_linear_sid_mlp_aug": ("mlp", "rgb", "sid_set"),
        "dinov2l_linear_sid_fft": ("linear", "fft", "sid_set"),
        "dinov2l_linear_sid_fft_aug": ("linear", "fft", "sid_set"),
    }
    # end
    for name, (head, mode, src) in checks.items():
        recipe = load_recipe(root / name / "recipe.yaml")
        validate(recipe)
        assert recipe["head"] == head
        assert recipe.get("input_mode", "rgb") == mode
        assert src in recipe["train"]["datasets"]
        assert NO_TRAIN_NAMES <= set(recipe["train"]["forbid"])
        if name.endswith("_aug"):
            assert recipe["aug"]["online"] is True
        if name in ("dinov2l_linear_sid", "dinov2l_linear_sid_mlp", "dinov2l_linear_sid_fft"):
            assert recipe["aug"]["online"] is False
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


def test_ablation_manifest_recipes_forbids_holdouts():
    # 2026-08-30, tianqi, D1/C-Flow/C-Pixel train from JSONL, never val/evalgen
    root = Path(__file__).resolve().parents[1] / "experiments"
    for name in (
        "clipb16_linear_D1_sid_only",
        "clipb16_linear_C_flow_sid",
        "clipb16_linear_C_pixel",
    ):
        recipe = load_recipe(root / name / "recipe.yaml")
        validate(recipe)
        assert recipe["train"]["datasets"] == ["manifest"]
        assert recipe["train"]["manifest"].endswith(".jsonl")
        assert recipe["aug"]["online"] is False
        assert NO_TRAIN_NAMES <= set(recipe["train"]["forbid"])
    # end


def test_featcache_path_is_shared():
    # 2026-08-29, tianqi, feat cache is keyed by backbone+split, not experiment name
    from src.paths import feat_cache_path

    a = feat_cache_path("resnet-50", "train", 100000, 0, 224)
    b = feat_cache_path("resnet-50", "train", 100000, 0, 224)
    assert a == b
    assert a.name == "cifake_train_n100000_seed0_s224.pt"
    # end
# end
