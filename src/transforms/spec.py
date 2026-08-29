# 2026-08-29, zyun, official transform settings registry (frozen by the problem statement)
"""The 14 official robustness settings of TechJam 2026 Track 5.

clean + these 14 = the 15 eval conditions. Parameters come verbatim from the
problem statement table; do not change them here without bumping the seed salt
(see ``derive_seed``) and updating docs/transforms.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Setting:
    key: str  # filesystem-safe setting name, e.g. "jpeg_q70"
    op: str  # jpeg | blur | resize | noise | jitter | crop
    params: dict = field(default_factory=dict)


OFFICIAL_SETTINGS: tuple[Setting, ...] = (
    Setting("jpeg_q90", "jpeg", {"quality": 90}),
    Setting("jpeg_q70", "jpeg", {"quality": 70}),
    Setting("jpeg_q50", "jpeg", {"quality": 50}),
    Setting("jpeg_q30", "jpeg", {"quality": 30}),
    Setting("blur_s05", "blur", {"sigma": 0.5}),
    Setting("blur_s10", "blur", {"sigma": 1.0}),
    Setting("blur_s20", "blur", {"sigma": 2.0}),
    Setting("resize_s05", "resize", {"scale": 0.5}),
    Setting("resize_s025", "resize", {"scale": 0.25}),
    Setting("noise_s002", "noise", {"sigma": 0.02}),
    Setting("noise_s005", "noise", {"sigma": 0.05}),
    Setting("noise_s010", "noise", {"sigma": 0.10}),
    Setting("jitter_p20", "jitter", {"range": 0.2}),
    Setting("crop_p80", "crop", {"keep": 0.8}),
)

SETTINGS_BY_KEY: dict[str, Setting] = {s.key: s for s in OFFICIAL_SETTINGS}


def resolve_settings(keys: str) -> list[Setting]:
    """Parse a comma-separated key list; empty or 'all' means the 14 official."""
    keys = (keys or "").strip()
    if keys.lower() in ("", "all"):
        return list(OFFICIAL_SETTINGS)
    out: list[Setting] = []
    for key in keys.split(","):
        key = key.strip()
        if key not in SETTINGS_BY_KEY:
            raise SystemExit(
                f"unknown setting '{key}'; official keys: {', '.join(SETTINGS_BY_KEY)}"
            )
        out.append(SETTINGS_BY_KEY[key])
    return out


def derive_seed(image_id: str, transform_key: str) -> int:
    """Stable per-(image, setting) seed: same inputs -> byte-identical outputs.

    Bump the salt suffix (v1 -> v2) whenever spec parameters change so old
    eval sets are never silently mixed with new ones.
    """
    digest = hashlib.sha1(f"{image_id}|{transform_key}|v1".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
# end
