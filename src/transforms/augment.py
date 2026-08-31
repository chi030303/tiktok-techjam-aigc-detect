# 2026-08-29, zyun, random training-time augmentation over the official transforms
"""Training-time random augmentation ("apply randomly during training").

The official table plays two roles:

- evaluation: the 14 frozen settings in ``spec.py``, applied deterministically;
- training: the same six ops sampled randomly per image, simulating the
  real-world redistribution pipeline.

``sample_random_setting`` picks one op (uniform by default, ``op_weights`` to
re-balance) and one parameter value. Default sampling uses the official
discrete values so train and eval share the same grid; ``continuous=True``
samples uniformly inside each official range instead.

``random_augment`` additionally supports ``p_clean`` (keep some images
untransformed) and ``chain_jpeg_p`` (append a random-quality JPEG re-encode
after any other op — real platforms re-encode almost everything; off by
default so the default policy stays aligned with the single-transform eval).

Recommended policy (docs/transforms.md §7): draw once per image per step,
apply at native resolution BEFORE the model-side resize, keep a small clean
probability, and augment real and fake images alike to avoid the
"real = lossy, fake = lossless" shortcut.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .ops import apply_setting, jpeg_compress_bytes
from .spec import Setting

OFFICIAL_OPS = ("jpeg", "blur", "resize", "noise", "jitter", "crop")

# Official discrete grids (= the eval settings' values).
DISCRETE_PARAMS = {
    "jpeg": {"quality": [90, 70, 50, 30]},
    "blur": {"sigma": [0.5, 1.0, 2.0]},
    "resize": {"scale": [0.5, 0.25]},
    "noise": {"sigma": [0.02, 0.05, 0.10]},
    "jitter": {"range": [0.2]},
    "crop": {"keep": [0.8]},
}

# Continuous ranges spanning the official values (for continuous=True).
CONTINUOUS_RANGES = {
    "jpeg": {"quality": (30, 90)},
    "blur": {"sigma": (0.5, 2.0)},
    "resize": {"scale": (0.25, 0.5)},
    "noise": {"sigma": (0.02, 0.10)},
    "jitter": {"range": (0.2, 0.2)},
    "crop": {"keep": (0.8, 0.8)},
}


def sample_random_setting(
    rng: np.random.Generator,
    op: str | None = None,
    continuous: bool = False,
    op_weights: dict | None = None,
) -> Setting:
    """Sample one Setting; ``op`` forces the op, ``op_weights`` re-balances it."""
    if op is None:
        if op_weights:
            names = list(op_weights)
            weights = np.asarray([float(op_weights[n]) for n in names], dtype=np.float64)
            weights = weights / weights.sum()
            op = names[int(rng.choice(len(names), p=weights))]
        else:
            op = OFFICIAL_OPS[int(rng.integers(len(OFFICIAL_OPS)))]
    if op not in DISCRETE_PARAMS:
        raise ValueError(f"unknown op '{op}'")
    params: dict = {}
    for name, values in DISCRETE_PARAMS[op].items():
        if continuous:
            lo, hi = CONTINUOUS_RANGES[op][name]
            value = float(rng.uniform(lo, hi))
            if name == "quality":
                value = int(round(value))
            params[name] = value
        else:
            params[name] = values[int(rng.integers(len(values)))]
    # The setting key only matters for the frozen eval sets; train draws are
    # recorded through random_augment's info dict instead.
    return Setting(key=f"train_{op}", op=op, params=params)


# 2026-08-29, tianqi, decode jpeg bytes the same way random_augment does
def _result_to_image(result) -> Image.Image:
    if isinstance(result, bytes):
        out = Image.open(io.BytesIO(result))
        out.load()
        return out
    return result
    # end


def apply_one_op(
    img: Image.Image,
    rng: np.random.Generator,
    op: str,
    continuous: bool = False,
    chain_jpeg_p: float = 0.0,
):
    # 2026-08-29, tianqi, force one official op so expand-six training covers all 6 views
    setting = sample_random_setting(rng, op=op, continuous=continuous)
    result, actual = apply_setting(img, setting, rng)
    out = _result_to_image(result)
    chain = None
    if chain_jpeg_p > 0 and setting.op != "jpeg" and float(rng.random()) < chain_jpeg_p:
        if continuous:
            quality = int(round(float(rng.uniform(30, 90))))
        else:
            quality = DISCRETE_PARAMS["jpeg"]["quality"][int(rng.integers(4))]
        out = Image.open(io.BytesIO(jpeg_compress_bytes(out, quality)))
        out.load()
        chain = quality
    return out, {"clean": False, "op": setting.op, "params": actual, "chain_jpeg": chain}
    # end


def random_augment(
    img: Image.Image,
    rng: np.random.Generator,
    p_clean: float = 0.2,
    continuous: bool = False,
    op_weights: dict | None = None,
    chain_jpeg_p: float = 0.0,
):
    """Apply one random official transform (or nothing) to ``img``.

    Returns ``(image, info)`` where info = ``{"clean", "op", "params",
    "chain_jpeg"}`` records exactly what happened, for logging and debugging.
    Deterministic given the same rng state.
    """
    if float(rng.random()) < p_clean:
        return img, {"clean": True, "op": None, "params": {}, "chain_jpeg": None}
    setting = sample_random_setting(rng, continuous=continuous, op_weights=op_weights)
    result, actual = apply_setting(img, setting, rng)
    out = _result_to_image(result)
    chain = None
    if chain_jpeg_p > 0 and setting.op != "jpeg" and float(rng.random()) < chain_jpeg_p:
        if continuous:
            quality = int(round(float(rng.uniform(30, 90))))
        else:
            quality = DISCRETE_PARAMS["jpeg"]["quality"][int(rng.integers(4))]
        out = Image.open(io.BytesIO(jpeg_compress_bytes(out, quality)))
        out.load()
        chain = quality
    return out, {"clean": False, "op": setting.op, "params": actual, "chain_jpeg": chain}
# end
