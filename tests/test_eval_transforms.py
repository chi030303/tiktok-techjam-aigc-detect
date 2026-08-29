# 2026-08-29, tianqi, TechJam transform sanity on a tiny synthetic RGB
import numpy as np
from PIL import Image

from src.eval.transforms import apply_condition, resolve_conditions, seed_for


def _pattern() -> Image.Image:
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:20, :, 0] = 255
    arr[:, :20, 1] = 180
    arr[10:30, 10:30, 2] = 255
    return Image.fromarray(arr, mode="RGB")


def test_daily_and_full_names():
    daily = resolve_conditions("daily")
    full = resolve_conditions("full")
    assert daily == ["clean", "jpeg_q50", "center_crop_80"]
    assert "jpeg_q30" in full and "noise_s0.10" in full
    assert set(daily) <= set(full)


def test_center_crop_shrinks():
    img = _pattern()
    out = apply_condition(img, "center_crop_80")
    assert out.size == (32, 32)


def test_resize_restores_hw():
    img = _pattern()
    out = apply_condition(img, "resize_x0.25")
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
    a = apply_condition(img, "noise_s0.10", seed=7)
    b = apply_condition(img, "noise_s0.10", seed=7)
    c = apply_condition(img, "noise_s0.10", seed=8)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))


def test_jitter_p20_brightens():
    img = _pattern()
    out = apply_condition(img, "jitter_p20")
    assert np.mean(np.asarray(out)) > np.mean(np.asarray(img))


def test_seed_for_is_stable():
    assert seed_for("real/a.jpg", "noise_s0.05") == seed_for("real/a.jpg", "noise_s0.05")
    assert seed_for("real/a.jpg", "noise_s0.05") != seed_for("fake/a.jpg", "noise_s0.05")
# end
