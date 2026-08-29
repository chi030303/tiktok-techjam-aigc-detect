# 2026-08-29, tianqi, thin eval adapter over src.transforms spec/ops (one 14-setting grid)
"""Eval conditions = official spec keys + clean. Pixel ops live in src.transforms."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from src.transforms.ops import apply_setting, to_rgb
from src.transforms.spec import OFFICIAL_SETTINGS, SETTINGS_BY_KEY, derive_seed

# end

CLEAN = "clean"

# 2026-08-29, tianqi, map the old eval-only names so existing CLI still resolves
_ALIASES = {
    "blur_s0.5": "blur_s05",
    "blur_s1.0": "blur_s10",
    "blur_s2.0": "blur_s20",
    "resize_x0.5": "resize_s05",
    "resize_x0.25": "resize_s025",
    "noise_s0.02": "noise_s002",
    "noise_s0.05": "noise_s005",
    "noise_s0.10": "noise_s010",
    "center_crop_80": "crop_p80",
}
# end

CONDITIONS = (CLEAN,) + tuple(s.key for s in OFFICIAL_SETTINGS)
DAILY_CONDITIONS = (CLEAN, "jpeg_q50", "crop_p80")
FULL_CONDITIONS = CONDITIONS
KNOWN = frozenset(CONDITIONS)


def canonicalize(name: str) -> str:
    raw = name.strip()
    if raw == "jitter_m20":
        raise SystemExit(
            "jitter_m20 was dropped: jitter is one official setting (jitter_p20, "
            "per-image brightness/contrast/sat sampled ±20%). See docs/transforms.md"
        )
    key = _ALIASES.get(raw, raw)
    if key not in KNOWN:
        raise SystemExit(f"unknown transform {name!r}; choose from {list(CONDITIONS)}")
    return key


def seed_for(rel_posix: str, condition: str) -> int:
    """Same sha1(image_id|key|v2) rule as src.transforms.spec.derive_seed."""
    key = canonicalize(condition)
    if key == CLEAN:
        return 0
    return derive_seed(rel_posix, key)


def apply_condition(img: Image.Image, name: str, seed: int = 0) -> Image.Image:
    key = canonicalize(name)
    if key == CLEAN:
        return to_rgb(img)
    setting = SETTINGS_BY_KEY[key]
    result, _actual = apply_setting(img, setting, np.random.default_rng(int(seed)))
    if isinstance(result, bytes):
        out = Image.open(io.BytesIO(result))
        out.load()
        return to_rgb(out)
    return result


def resolve_conditions(spec: str) -> list[str]:
    spec = spec.strip()
    lowered = spec.lower()
    if lowered in {"daily", "ops"}:
        return list(DAILY_CONDITIONS)
    if lowered in {"full", "all"}:
        return list(FULL_CONDITIONS)
    names = [canonicalize(x) for x in spec.split(",") if x.strip()]
    if not names:
        raise SystemExit("empty --conditions")
    return names
# end
