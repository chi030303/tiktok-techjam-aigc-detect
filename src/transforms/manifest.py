# 2026-08-29, zyun, manifest schemas: source images and transformed eval records
"""Two JSONL manifests, one row per image.

``source``    — one row per raw image on disk (built by build_source.py).
``transform`` — one row per derived image (built by build.py); source fields
are denormalized into it so the eval pipeline never needs a join.

Field tables: docs/transforms.md. Validation is strict on purpose: missing or
unknown fields are errors, so schema drift breaks at build time instead of
silently corrupting eval tables later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .spec import SETTINGS_BY_KEY

SPLITS = ("train", "val", "test", "unseen")


class _Record:
    @classmethod
    def from_dict(cls, row: dict):
        known = {f.name for f in fields(cls)}
        missing = sorted(known - row.keys())
        unknown = sorted(row.keys() - known)
        if missing:
            raise ValueError(f"{cls.__name__}: missing fields {missing}")
        if unknown:
            raise ValueError(f"{cls.__name__}: unknown fields {unknown}")
        rec = cls(**row)
        rec.validate()
        return rec

    def validate(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _check_common(rec, cls_name: str) -> None:
    if rec.label not in (0, 1):
        raise ValueError(f"{cls_name}: label must be 0 or 1, got {rec.label!r}")
    if rec.split not in SPLITS:
        raise ValueError(f"{cls_name}: split must be one of {SPLITS}, got {rec.split!r}")
    if not rec.path:
        raise ValueError(f"{cls_name}: path must be non-empty")


@dataclass
class SourceRecord(_Record):
    image_id: str
    path: str  # relative to repo root, posix style (absolute if outside CWD)
    label: int  # 1 = AIGC, 0 = real
    source_dataset: str  # cifake | sid_set | wildfake | flux_gen | ...
    generator: str | None  # generator family for fakes, None for reals
    split: str  # train | val | test | unseen
    width: int
    height: int

    def validate(self) -> None:
        _check_common(self, "SourceRecord")
        if not self.image_id:
            raise ValueError("SourceRecord: image_id must be non-empty")


@dataclass
class TransformRecord(_Record):
    row_id: str
    source_image_id: str
    source_path: str
    transform: str  # op name
    transform_key: str  # setting name, e.g. "jpeg_q70"
    params: dict  # actual params (jitter: the sampled factors)
    seed: int
    path: str
    label: int
    source_dataset: str
    generator: str | None
    split: str
    width: int
    height: int

    def validate(self) -> None:
        _check_common(self, "TransformRecord")
        if not self.source_image_id:
            raise ValueError("TransformRecord: source_image_id must be non-empty")
        if self.transform_key not in SETTINGS_BY_KEY:
            raise ValueError(
                f"TransformRecord: unknown transform_key {self.transform_key!r}"
            )
        if not isinstance(self.params, dict) or not self.params:
            raise ValueError("TransformRecord: params must be a non-empty dict")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"TransformRecord: bad seed {self.seed!r}")


KINDS = {"source": SourceRecord, "transform": TransformRecord}


def read_jsonl(path: str | Path, kind: str) -> list:
    if kind not in KINDS:
        raise SystemExit(f"unknown manifest kind '{kind}' (want source|transform)")
    cls = KINDS[kind]
    rows: list = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(cls.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise SystemExit(f"{path}:{lineno}: {exc}")
    return rows


def write_jsonl(path: str | Path, records: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")
# end
