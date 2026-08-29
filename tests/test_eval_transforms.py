# 2026-08-29, tianqi, TechJam transform sanity on a tiny synthetic RGB
import numpy as np
from PIL import Image

from src.eval.transforms import (
    CONDITIONS,
    apply_condition,
    canonicalize,
    resolve_conditions,
    seed_for,
)
from src.transforms.ops import apply_setting
from src.transforms.spec import OFFICIAL_SETTINGS, SETTINGS_BY_KEY, derive_seed

# end


def _pattern() -> Image.Image:
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:20, :, 0] = 255
    arr[:, :20, 1] = 180
    arr[10:30, 10:30, 2] = 255
    return Image.fromarray(arr, mode="RGB")


def test_daily_and_full_names():
    daily = resolve_conditions("daily")
    full = resolve_conditions("full")
    assert daily == ["clean", "jpeg_q50", "crop_p80"]
    assert "jpeg_q30" in full and "noise_s010" in full
    assert "jitter_m20" not in full
    assert set(daily) <= set(full)
    assert CONDITIONS == ("clean",) + tuple(s.key for s in OFFICIAL_SETTINGS)
    assert len(full) == 15


def test_jitter_m20_rejected():
    import pytest

    with pytest.raises(SystemExit):
        canonicalize("jitter_m20")


def test_aliases_map_to_spec_keys():
    assert canonicalize("center_crop_80") == "crop_p80"
    assert canonicalize("blur_s0.5") == "blur_s05"
    assert canonicalize("resize_x0.25") == "resize_s025"
    assert resolve_conditions("jpeg_q50,center_crop_80") == ["jpeg_q50", "crop_p80"]


def test_center_crop_shrinks():
    img = _pattern()
    out = apply_condition(img, "crop_p80")
    assert out.size == (32, 32)


def test_resize_restores_hw():
    img = _pattern()
    out = apply_condition(img, "resize_s025")
    assert out.size == img.size
    assert np.mean(np.abs(np.asarray(out).astype(int) - np.asarray(img).astype(int))) > 1.0


def test_jpeg_q30_hurts_more_than_q90():
    img = _pattern()
    orig = np.asarray(img).astype(np.float32)
    d90 = np.mean(np.abs(np.asarray(apply_condition(img, "jpeg_q90")).astype(np.float32) - orig))
    d30 = np.mean(np.abs(np.asarray(apply_condition(img, "jpeg_q30")).astype(np.float32) - orig))
    assert d30 >= d90


def test_noise_is_deterministic_for_seed():
    img = _pattern()
    a = apply_condition(img, "noise_s010", seed=7)
    b = apply_condition(img, "noise_s010", seed=7)
    c = apply_condition(img, "noise_s010", seed=8)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))


def test_jitter_p20_is_seeded_independent_factors():
    img = _pattern()
    a = apply_condition(img, "jitter_p20", seed=11)
    b = apply_condition(img, "jitter_p20", seed=11)
    c = apply_condition(img, "jitter_p20", seed=12)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))
    assert a.size == img.size


def test_seed_for_matches_spec_derive_seed():
    assert seed_for("real/a.jpg", "noise_s005") == derive_seed("real/a.jpg", "noise_s005")
    assert seed_for("real/a.jpg", "noise_s005") != seed_for("fake/a.jpg", "noise_s005")
    assert seed_for("real/a.jpg", "clean") == 0


# 2026-08-29, tianqi, eval adapter must match ops.apply_setting pixels
def test_eval_matches_ops_apply_setting():
    img = _pattern()
    seed = 7
    result, _ = apply_setting(img, SETTINGS_BY_KEY["crop_p80"], np.random.default_rng(seed))
    via_eval = apply_condition(img, "crop_p80", seed=seed)
    assert np.array_equal(np.asarray(result), np.asarray(via_eval))
# end
