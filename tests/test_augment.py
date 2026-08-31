# 2026-08-29, zyun, tests for training-time random augmentation
import numpy as np

from src.transforms.augment import apply_one_op, random_augment, sample_random_setting
from tests.test_transforms import make_image

ALL_OPS = {"jpeg", "blur", "resize", "noise", "jitter", "crop"}


def test_random_augment_reproducible():
    img = make_image()
    out1, info1 = random_augment(img, np.random.default_rng(11), p_clean=0.0)
    out2, info2 = random_augment(img, np.random.default_rng(11), p_clean=0.0)
    assert info1 == info2
    assert out1.tobytes() == out2.tobytes()


def test_p_clean_identity():
    img = make_image()
    out, info = random_augment(img, np.random.default_rng(0), p_clean=1.0)
    assert info["clean"] is True and info["op"] is None
    assert out.tobytes() == img.tobytes()


def test_all_ops_reachable_and_official_grids():
    img = make_image(32, 32)
    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(400):
        _, info = random_augment(img, rng, p_clean=0.0)
        assert info["op"] in ALL_OPS
        seen.add(info["op"])
        p = info["params"]
        if info["op"] == "jpeg":
            assert p["quality"] in (90, 70, 50, 30)
        elif info["op"] == "blur":
            assert p["sigma"] in (0.5, 1.0, 2.0)
        elif info["op"] == "resize":
            assert p["scale"] in (0.5, 0.25)
        elif info["op"] == "noise":
            assert p["sigma"] in (0.02, 0.05, 0.10)
        elif info["op"] == "jitter":
            # actual sampled factors are recorded, not the range
            assert set(p) == {"brightness", "contrast", "saturation"}
            assert all(0.8 <= v <= 1.2 for v in p.values())
        else:
            assert p["keep"] == 0.8
    assert seen == ALL_OPS


def test_continuous_ranges():
    rng = np.random.default_rng(3)
    for _ in range(50):
        setting = sample_random_setting(rng, continuous=True)
        p = setting.params
        if setting.op == "jpeg":
            assert isinstance(p["quality"], int) and 30 <= p["quality"] <= 90
        elif setting.op == "blur":
            assert 0.5 <= p["sigma"] <= 2.0
        elif setting.op == "resize":
            assert 0.25 <= p["scale"] <= 0.5
        elif setting.op == "noise":
            assert 0.02 <= p["sigma"] <= 0.10


def test_op_weights_forced():
    img = make_image()
    rng = np.random.default_rng(5)
    for _ in range(20):
        _, info = random_augment(img, rng, p_clean=0.0, op_weights={"crop": 1.0})
        assert info["op"] == "crop"


def test_chain_jpeg_reencode():
    img = make_image()
    rng = np.random.default_rng(9)
    chained = 0
    for _ in range(30):
        out, info = random_augment(img, rng, p_clean=0.0, chain_jpeg_p=1.0)
        if info["op"] != "jpeg":
            assert isinstance(info["chain_jpeg"], int) and 30 <= info["chain_jpeg"] <= 90
            chained += 1
        else:
            assert info["chain_jpeg"] is None
    assert chained > 0


def test_apply_one_op_forces_each_official_op():
    # 2026-08-29, tianqi, expand-six training needs a forced-op API
    img = make_image(32, 32)
    rng = np.random.default_rng(1)
    for op in ALL_OPS:
        out, info = apply_one_op(img, rng, op=op)
        assert info["op"] == op
        assert info["clean"] is False
        assert out.size[0] > 0
    # end
# end
