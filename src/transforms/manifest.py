# 2026-08-29, zyun, manifest schemas: source images and transformed eval records
"""Two JSONL manifests, one row per image.

``source``    — one row per raw image on disk (built by build_source.py).
``transform`` — one row per derived image (built by build.py); source fields
are denormalized into it so the eval pipeline never needs a join.

Field tables: docs/transforms.md. Validation is strict on purpose: missing or
unknown fields are errors, so schema drift breaks at build time instead of
silently corrupting eval tables later.

Core columns (image_id/path/label/...) must be present. Ablation/audit columns
(family/arch/content_type/original_format/phash) default to null so older JSONL
still loads. New data should fill them via build_source or a custom writer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

from PIL import Image

from .spec import SETTINGS_BY_KEY

# end

SPLITS = ("train", "val", "test", "unseen")

# 2026-08-30, tianqi, ablation + leak-audit enums (reals use null, not the string "real")
FAMILIES = ("t2i", "i2i")
ARCHES = ("unet", "dit", "flow", "pixel", "gan")
CONTENT_TYPES = ("real", "full_synthetic", "partial_manipulation")
FORMAT_ALIASES = {"jpeg": "jpg", "tiff": "tif"}
# end


class _Record:
    @classmethod
    def from_dict(cls, row: dict):
        if not isinstance(row, dict):
            raise ValueError(
                f"{cls.__name__}: expected a JSON object per line, got {type(row).__name__}"
            )
        known = {f.name for f in fields(cls)}
        unknown = sorted(row.keys() - known)
        if unknown:
            raise ValueError(f"{cls.__name__}: unknown fields {unknown}")
        payload = {}
        missing = []
        for f in fields(cls):
            if f.name in row:
                payload[f.name] = row[f.name]
            elif f.default is not MISSING:
                payload[f.name] = f.default
            elif f.default_factory is not MISSING:  # pragma: no cover
                payload[f.name] = f.default_factory()
            else:
                missing.append(f.name)
        if missing:
            raise ValueError(f"{cls.__name__}: missing fields {sorted(missing)}")
        rec = cls(**payload)
        rec.validate()
        return rec

    def validate(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _check_common(rec, cls_name: str) -> None:
    # bool is an int subclass; reject it explicitly so label=True/False fails.
    if isinstance(rec.label, bool) or not isinstance(rec.label, int) or rec.label not in (0, 1):
        raise ValueError(f"{cls_name}: label must be int 0 or 1, got {rec.label!r}")
    if rec.split not in SPLITS:
        raise ValueError(f"{cls_name}: split must be one of {SPLITS}, got {rec.split!r}")
    if not rec.path:
        raise ValueError(f"{cls_name}: path must be non-empty")
    _check_size(rec, cls_name)


def _check_size(rec, cls_name: str) -> None:
    for name in ("width", "height"):
        value = getattr(rec, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{cls_name}: {name} must be a positive int, got {value!r}")


def _normalize_generator(label: int, generator: str | None) -> str | None:
    if generator in ("", "real"):
        generator = None
    if label == 0 and generator is not None:
        raise ValueError("reals must have generator=null (not the string 'real')")
    return generator


def _normalize_format(value: str | None) -> str | None:
    if value is None:
        return None
    fmt = str(value).strip().lower().lstrip(".")
    if not fmt:
        return None
    return FORMAT_ALIASES.get(fmt, fmt)


def _check_phash(value: str | None, cls_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value) != 16:
        raise ValueError(f"{cls_name}: phash must be 16 hex chars, got {value!r}")
    try:
        int(value, 16)
    except ValueError:
        raise ValueError(f"{cls_name}: phash must be 16 hex chars, got {value!r}") from None


def _check_ablation(rec, cls_name: str) -> None:
    if rec.family is not None and rec.family not in FAMILIES:
        raise ValueError(f"{cls_name}: family must be t2i|i2i|null, got {rec.family!r}")
    if rec.arch is not None and rec.arch not in ARCHES:
        raise ValueError(f"{cls_name}: arch must be unet|dit|flow|pixel|gan|null, got {rec.arch!r}")
    if rec.content_type is not None and rec.content_type not in CONTENT_TYPES:
        raise ValueError(
            f"{cls_name}: content_type must be real|full_synthetic|"
            f"partial_manipulation|null, got {rec.content_type!r}"
        )
    if rec.label == 0:
        if rec.family is not None or rec.arch is not None:
            raise ValueError(f"{cls_name}: reals must have family=arch=null")
        if rec.content_type not in (None, "real"):
            raise ValueError(f"{cls_name}: real label requires content_type=real")
    if rec.label == 1 and rec.content_type == "real":
        raise ValueError(f"{cls_name}: fake label cannot have content_type=real")
    _check_phash(rec.phash, cls_name)


def infer_content_type(label: int, content_type: str | None) -> str:
    if content_type is not None:
        return content_type
    return "real" if int(label) == 0 else "full_synthetic"


# 2026-08-30, tianqi, 8x8 average hash; no extra dep. Catches copies, not local inpaint.
def average_phash(img: Image.Image, size: int = 8) -> str:
    """Perceptual hash for dedupe / val-leak audit. 64-bit aHash as 16 hex chars.

    Tampered (local edit) images usually will NOT match the COCO original —
    exclude those with content_type=partial_manipulation, do not rely on phash.
    """
    gray = img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = list(gray.tobytes())
    avg = sum(pixels) / max(1, len(pixels))
    bits = 0
    for i, pix in enumerate(pixels):
        if pix >= avg:
            bits |= 1 << i
    width = size * size // 4
    return f"{bits:0{width}x}"
    # end


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def phash_collisions(
    rows_a: list,
    rows_b: list,
    max_distance: int = 0,
) -> list[dict]:
    """Pairs that share a perceptual hash (SID train vs official val, etc.)."""
    by_hash: dict[str, list] = defaultdict(list)
    for rec in rows_b:
        if getattr(rec, "phash", None):
            by_hash[rec.phash].append(rec)
    hits = []
    seen = set()
    for rec in rows_a:
        ph = getattr(rec, "phash", None)
        if not ph:
            continue
        candidates = []
        if max_distance <= 0:
            candidates = by_hash.get(ph, [])
        else:
            for other_hash, group in by_hash.items():
                dist = hamming_hex(ph, other_hash)
                if dist <= max_distance:
                    candidates.extend((other, dist) for other in group)
        if max_distance <= 0:
            for other in candidates:
                key = (rec.image_id, other.image_id)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    {
                        "phash": ph,
                        "distance": 0,
                        "image_id": rec.image_id,
                        "path": rec.path,
                        "other_image_id": other.image_id,
                        "other_path": other.path,
                    }
                )
        else:
            for other, dist in candidates:
                key = (rec.image_id, other.image_id)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    {
                        "phash": rec.phash,
                        "distance": dist,
                        "image_id": rec.image_id,
                        "path": rec.path,
                        "other_image_id": other.image_id,
                        "other_path": other.path,
                    }
                )
    return hits


def is_trainable(rec) -> bool:
    """Train loader should keep split=train and drop SID tampered."""
    if rec.split != "train":
        return False
    if rec.content_type == "partial_manipulation":
        return False
    return True


def filter_train_rows(
    rows: list,
    holdout: list | None = None,
    max_distance: int = 0,
) -> tuple[list, list[dict]]:
    """Drop tampered / non-train, then rows whose phash hits a holdout image."""
    leaks = phash_collisions(rows, holdout or [], max_distance=max_distance) if holdout else []
    leak_ids = {h["image_id"] for h in leaks}
    kept = [r for r in rows if is_trainable(r) and r.image_id not in leak_ids]
    return kept, leaks


@dataclass
class SourceRecord(_Record):
    image_id: str
    path: str  # relative to repo root, posix style (absolute if outside CWD)
    label: int  # 1 = AIGC, 0 = real
    source_dataset: str  # cifake | sid_set | wildfake | flux_gen | …
    generator: str | None  # concrete generator for fakes, None for reals
    split: str  # train | val | test | unseen
    width: int
    height: int
    # 2026-08-30, tianqi, ablation + leak-audit columns (optional in old JSONL)
    family: str | None = None  # t2i | i2i | None
    arch: str | None = None  # unet | dit | flow | pixel | gan | None
    content_type: str | None = None  # real | full_synthetic | partial_manipulation
    original_format: str | None = None  # jpg | png | webp | … (suffix, jpeg→jpg)
    phash: str | None = None  # 16-hex aHash; None if not computed
    # end

    def validate(self) -> None:
        _check_common(self, "SourceRecord")
        if not self.image_id:
            raise ValueError("SourceRecord: image_id must be non-empty")
        self.generator = _normalize_generator(self.label, self.generator)
        self.original_format = _normalize_format(self.original_format)
        self.content_type = infer_content_type(self.label, self.content_type)
        _check_ablation(self, "SourceRecord")


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
    # 2026-08-30, tianqi, copy source ablation columns so eval/badcase skip a join
    family: str | None = None
    arch: str | None = None
    content_type: str | None = None
    original_format: str | None = None
    phash: str | None = None
    # end

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
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"TransformRecord: bad seed {self.seed!r}")
        self.generator = _normalize_generator(self.label, self.generator)
        self.original_format = _normalize_format(self.original_format)
        self.content_type = infer_content_type(self.label, self.content_type)
        _check_ablation(self, "TransformRecord")


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
