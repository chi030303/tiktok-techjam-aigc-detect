# 2026-08-29, tianqi, frequency-domain views for AIGC residual / FFT spectrum
"""High-pass residual and log-FFT magnitude as extra model inputs."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# end


def highpass_residual(img: Image.Image, sigma: float = 1.5) -> Image.Image:
    rgb = img.convert("RGB")
    blur = rgb.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    a = np.asarray(rgb, dtype=np.int16)
    b = np.asarray(blur, dtype=np.int16)
    out = np.clip(a - b + 128, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def fft_mag_rgb(img: Image.Image) -> Image.Image:
    # 2026-08-29, tianqi, grayscale log-FFT, tiled to 3ch so the frozen RGB backbone can consume it
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    mag = mag / (float(mag.max()) + 1e-6)
    mag8 = (mag * 255.0).round().astype(np.uint8)
    plane = Image.fromarray(mag8, mode="L")
    return Image.merge("RGB", (plane, plane, plane))
    # end


def apply_input_mode(img: Image.Image, mode: str) -> Image.Image:
    mode = (mode or "rgb").lower()
    if mode in ("rgb", "clean"):
        return img.convert("RGB")
    if mode in ("highpass", "hp"):
        return highpass_residual(img)
    if mode in ("fft", "spectrum"):
        return fft_mag_rgb(img)
    raise ValueError(f"unknown input_mode {mode!r}")
# end
