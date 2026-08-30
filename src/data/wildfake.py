# 2026-08-30, samily, WildFake scanner + generator rule matching
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

from src.transforms.manifest import SourceRecord, average_phash

# end

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_generators_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


def match_rule(rel_posix: str, rules: list[dict]) -> dict | None:
    for rule in rules:
        if re.search(rule["pattern"], rel_posix, flags=re.IGNORECASE):
            return rule
    return None


def rule_for_generator(generator: str, rules: list[dict]) -> dict:
    matches = [rule for rule in rules if rule.get("generator") == generator]
    if len(matches) != 1:
        raise ValueError(
            f"expected one WildFake rule for generator={generator!r}, got {len(matches)}"
        )
    return matches[0]


def is_excluded(rel_posix: str, exclude: list[re.Pattern]) -> bool:
    return any(p.search(rel_posix) for p in exclude)


def scan_wildfake(
    root: Path,
    cfg: dict,
    *,
    split: str = "train",
    generators: set[str] | None = None,
    min_side: int = 0,
    max_per_generator: int | None = None,
    compute_phash: bool = True,
    report_unmatched: bool = False,
    force_generator: str | None = None,
) -> tuple[list[SourceRecord], dict]:
    from PIL import Image
    import hashlib

    root = root.resolve()
    rules = cfg.get("rules") or []
    forced_rule = rule_for_generator(force_generator, rules) if force_generator else None
    exclude = _compile(cfg.get("exclude_patterns") or [])
    records: list[SourceRecord] = []
    unmatched = 0
    per_gen: dict[str, int] = {}

    for path in sorted(root.rglob("*")):
        if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            continue
        rel = path.relative_to(root).as_posix()
        if is_excluded(rel, exclude):
            continue
        rule = forced_rule or match_rule(rel, rules)
        if rule is None:
            unmatched += 1
            continue
        gen = rule.get("generator")
        gen_key = gen or "real"
        if generators and gen_key not in generators and gen_key != "real":
            continue
        if max_per_generator is not None and per_gen.get(gen_key, 0) >= max_per_generator:
            continue
        with Image.open(path) as im:
            width, height = im.size
            if min(width, height) < min_side:
                continue
            phash = average_phash(im) if compute_phash else None
        fmt = path.suffix.lower().lstrip(".")
        is_real = int(rule["label"]) == 0
        try:
            stored = path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            stored = path.as_posix()
        image_id = hashlib.sha1(f"{stored}:{path.stat().st_size}".encode()).hexdigest()
        rec = SourceRecord(
            image_id=image_id,
            path=stored,
            label=int(rule["label"]),
            source_dataset="wildfake",
            generator=None if is_real else gen,
            split=split,
            width=width,
            height=height,
            family=None if is_real else rule.get("family"),
            arch=None if is_real else rule.get("arch"),
            generation_type=None if is_real else rule.get("generation_type"),
            content_type=rule.get("content_type") or ("real" if is_real else "full_synthetic"),
            original_format=fmt or None,
            phash=phash,
        )
        records.append(rec)
        per_gen[gen_key] = per_gen.get(gen_key, 0) + 1

    stats = {"total": len(records), "unmatched": unmatched, "per_generator": per_gen}
    if report_unmatched and unmatched:
        print(f"warning: {unmatched} images did not match any rule", file=sys.stderr)
    return records, stats
