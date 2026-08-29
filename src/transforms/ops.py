# 2026-08-29, zyun, pixel-level implementations of the 6 official transforms
"""Pixel ops for the Track 5 robustness transforms.

Every op takes an RGB ``PIL.Image`` plus its params and returns the transformed
result without touching file paths, so two consumers share one implementation:

- ``build.py``: frozen eval sets, driven by per-image seeds (deterministic);
- training augmentation: same ops, fresh random params per draw.

Size contract: every op keeps the input size except ``crop`` (keeps the center
``keep`` fraction of each side) — the statement only asks to upscale back for
``resize`` ("then upscale"). See docs/transforms.md for each decision and the
``--crop-resize-back`` escape hatch.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

Resample = Image.Resampling.BILINEAR


def to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def jpeg_compress_bytes(img: Image.Image, quality: int) -> bytes:
    """One JPEG re-encode; the builder writes these bytes out as-is."""
    buf = io.BytesIO()
    to_rgb(img).save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """JPEG re-encode decoded back to an Image (unit tests / in-memory use)."""
    out = Image.open(io.BytesIO(jpeg_compress_bytes(img, quality)))
    out.load()
    return out


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    # PIL's GaussianBlur radius *is* the standard deviation.
    return to_rgb(img).filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize_roundtrip(img: Image.Image, scale: float) -> Image.Image:
    """Downscale by ``scale`` then upscale back to the original size."""
    im = to_rgb(img)
    w, h = im.size
    small = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Resample)
    return small.resize((w, h), Resample)


def gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    """Additive Gaussian noise on [0,1] pixels, clipped back to uint8."""
    arr = np.asarray(to_rgb(img), dtype=np.float32) / 255.0
    noise = rng.normal(0.0, float(sigma), size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGB")


def sample_jitter_factors(rng: np.random.Generator, jitter_range: float) -> dict:
    """brightness/contrast/saturation each uniform in [1-r, 1+r]."""
    low, high = 1.0 - float(jitter_range), 1.0 + float(jitter_range)
    return {
        "brightness": float(rng.uniform(low, high)),
        "contrast": float(rng.uniform(low, high)),
        "saturation": float(rng.uniform(low, high)),
    }


def color_jitter(
    img: Image.Image, brightness: float = 1.0, contrast: float = 1.0, saturation: float = 1.0
) -> Image.Image:
    out = to_rgb(img)
    out = ImageEnhance.Brightness(out).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    return ImageEnhance.Color(out).enhance(saturation)


def center_crop(img: Image.Image, keep: float = 0.8) -> Image.Image:
    """Center crop keeping ``keep`` of each side; output is smaller by design."""
    im = to_rgb(img)
    w, h = im.size
    cw, ch = max(1, round(w * keep)), max(1, round(h * keep))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return im.crop((x0, y0, x0 + cw, y0 + ch))


def apply_setting(
    img: Image.Image, setting, rng: np.random.Generator, crop_resize_back: bool = False
):
    """Apply one Setting; returns ``(result, actual_params)``.

    ``result`` is JPEG bytes for the jpeg op (so the builder writes the exact
    encoded bytes instead of re-encoding) and a PIL.Image otherwise. For
    jitter, the sampled factors are returned in ``actual_params``.
    """
    p = setting.params
    if setting.op == "jpeg":
        return jpeg_compress_bytes(img, p["quality"]), dict(p)
    if setting.op == "blur":
        return gaussian_blur(img, p["sigma"]), dict(p)
    if setting.op == "resize":
        return resize_roundtrip(img, p["scale"]), dict(p)
    if setting.op == "noise":
        return gaussian_noise(img, p["sigma"], rng), dict(p)
    if setting.op == "jitter":
        factors = sample_jitter_factors(rng, p["range"])
        return color_jitter(img, **factors), factors
    if setting.op == "crop":
        out = center_crop(img, p["keep"])
        if crop_resize_back:
            out = out.resize(to_rgb(img).size, Resample)
        return out, dict(p)
    raise ValueError(f"unknown op: {setting.op}")
# end
