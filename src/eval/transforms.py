# 2026-08-29, tianqi, TechJam robustness ops (JPEG / blur / resize / noise / jitter / crop)
"""Pixel transforms from the challenge brief. Eval applies these at fixed levels."""

from __future__ import annotations

import hashlib
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# end

try:
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _RESAMPLE = Image.BILINEAR


def seed_for(rel_posix: str, condition: str) -> int:
    digest = hashlib.sha256(f"{condition}:{rel_posix}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def down_up(img: Image.Image, scale: float) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    small = rgb.resize((nw, nh), resample=_RESAMPLE)
    return small.resize((w, h), resample=_RESAMPLE)


def gaussian_noise(img: Image.Image, sigma: float, seed: int) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)
    noisy = np.clip(rgb + rng.normal(0.0, sigma, rgb.shape), 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).round().astype(np.uint8), mode="RGB")


def color_jitter(img: Image.Image, factor: float) -> Image.Image:
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(factor)
    rgb = ImageEnhance.Contrast(rgb).enhance(factor)
    rgb = ImageEnhance.Color(rgb).enhance(factor)
    return rgb


def center_crop(img: Image.Image, keep: float = 0.8) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    nw = max(1, int(round(w * keep)))
    nh = max(1, int(round(h * keep)))
    left = (w - nw) // 2
    top = (h - nh) // 2
    return rgb.crop((left, top, left + nw, top + nh))


_OPS = {
    "clean": lambda img, seed: img.convert("RGB"),
    "jpeg_q90": lambda img, seed: jpeg(img, 90),
    "jpeg_q70": lambda img, seed: jpeg(img, 70),
    "jpeg_q50": lambda img, seed: jpeg(img, 50),
    "jpeg_q30": lambda img, seed: jpeg(img, 30),
    "blur_s0.5": lambda img, seed: gaussian_blur(img, 0.5),
    "blur_s1.0": lambda img, seed: gaussian_blur(img, 1.0),
    "blur_s2.0": lambda img, seed: gaussian_blur(img, 2.0),
    "resize_x0.5": lambda img, seed: down_up(img, 0.5),
    "resize_x0.25": lambda img, seed: down_up(img, 0.25),
    "noise_s0.02": lambda img, seed: gaussian_noise(img, 0.02, seed),
    "noise_s0.05": lambda img, seed: gaussian_noise(img, 0.05, seed),
    "noise_s0.10": lambda img, seed: gaussian_noise(img, 0.10, seed),
    "jitter_p20": lambda img, seed: color_jitter(img, 1.2),
    "jitter_m20": lambda img, seed: color_jitter(img, 0.8),
    "center_crop_80": lambda img, seed: center_crop(img, 0.8),
}

CONDITIONS = tuple(_OPS.keys())
DAILY_CONDITIONS = ("clean", "jpeg_q50", "center_crop_80")
FULL_CONDITIONS = CONDITIONS


def apply_condition(img: Image.Image, name: str, seed: int = 0) -> Image.Image:
    if name not in _OPS:
        raise SystemExit(f"unknown transform {name!r}; choose from {list(_OPS)}")
    return _OPS[name](img, seed)


def resolve_conditions(spec: str) -> list[str]:
    spec = spec.strip().lower()
    if spec in {"daily", "ops"}:
        return list(DAILY_CONDITIONS)
    if spec in {"full", "all"}:
        return list(FULL_CONDITIONS)
    names = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [n for n in names if n not in _OPS]
    if unknown:
        raise SystemExit(f"unknown transform(s) {unknown}; known={list(_OPS)}")
    return names
# end
