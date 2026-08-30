# 2026-08-29, zyun, transform package public API
"""Official Track 5 robustness transforms.

- ``spec``: the 14 frozen official settings + deterministic seed derivation
- ``ops``: pixel-level transform implementations (eval + train share them)
- ``manifest``: strict JSONL schemas for source and transformed records
- ``build_source`` / ``build``: CLIs to index raw images and generate the
  frozen transformed eval set (docs/transforms.md has the full design)
"""

from .manifest import (
    SourceRecord,
    TransformRecord,
    # 2026-08-30, tianqi, phash leak helpers for train filtering
    average_phash,
    filter_train_rows,
    is_trainable,
    phash_collisions,
    # end
    read_jsonl,
    write_jsonl,
)
from .spec import (
    OFFICIAL_SETTINGS,
    SETTINGS_BY_KEY,
    Setting,
    derive_seed,
    resolve_settings,
)

__all__ = [
    "OFFICIAL_SETTINGS",
    "SETTINGS_BY_KEY",
    "Setting",
    "SourceRecord",
    "TransformRecord",
    "average_phash",
    "derive_seed",
    "filter_train_rows",
    "is_trainable",
    "phash_collisions",
    "read_jsonl",
    "resolve_settings",
    "write_jsonl",
]
