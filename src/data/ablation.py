# 2026-08-30, samily, controlled ablation manifest sampling (DATA_ABLATION_PLAN §5–§10)
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.paths import REPO_ROOT, data_root
from src.transforms.manifest import SourceRecord, read_jsonl, write_jsonl

# end


def _resolve_manifest(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    if path.parts and path.parts[0] == "data":
        return data_root().joinpath(*path.parts[1:])
    candidate = data_root() / path
    return candidate if candidate.exists() else REPO_ROOT / path


@dataclass
class PoolSpec:
    manifest: str
    n: int
    label: int | None = None  # 0 real, 1 fake; None = any trainable
    family: list[str] | None = None
    arch: list[str] | None = None
    generation_type: list[str] | None = None
    generator: list[str] | None = None
    exclude_content_types: list[str] = field(default_factory=lambda: ["partial_manipulation"])


@dataclass
class AblationSpec:
    name: str
    seed: int = 20260830
    split: str = "train"
    real: PoolSpec | None = None
    fake: PoolSpec | None = None
    pools: list[PoolSpec] = field(default_factory=list)


def _match(rec: SourceRecord, spec: PoolSpec) -> bool:
    if rec.split != "train":
        return False
    if spec.label is not None and rec.label != spec.label:
        return False
    if rec.content_type in spec.exclude_content_types:
        return False
    if spec.family and rec.family not in spec.family:
        return False
    if spec.arch and rec.arch not in spec.arch:
        return False
    if spec.generation_type and rec.generation_type not in spec.generation_type:
        return False
    if spec.generator and rec.generator not in spec.generator:
        return False
    return True


def _sample_pool(spec: PoolSpec, rng: random.Random) -> list[SourceRecord]:
    manifest = _resolve_manifest(spec.manifest)
    rows = read_jsonl(manifest, kind="source")
    matched = [r for r in rows if _match(r, spec)]
    if len(matched) < spec.n:
        raise SystemExit(
            f"pool {manifest}: need {spec.n}, matched {len(matched)} "
            f"(label={spec.label} family={spec.family} arch={spec.arch})"
        )
    return rng.sample(matched, spec.n)


def build_ablation(spec: AblationSpec) -> list[SourceRecord]:
    rng = random.Random(spec.seed)
    out: list[SourceRecord] = []
    if spec.real is not None:
        out.extend(_sample_pool(spec.real, rng))
    if spec.fake is not None:
        out.extend(_sample_pool(spec.fake, rng))
    for pool in spec.pools:
        out.extend(_sample_pool(pool, rng))
    rng.shuffle(out)
    return out


def load_ablation_yaml(path: Path) -> AblationSpec:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "name" not in raw:
        raise SystemExit(f"invalid ablation config: {path}")

    def _pool(d: dict) -> PoolSpec:
        return PoolSpec(
            manifest=d["manifest"],
            n=int(d["n"]),
            label=d.get("label"),
            family=d.get("family"),
            arch=d.get("arch"),
            generation_type=d.get("generation_type"),
            generator=d.get("generator"),
            exclude_content_types=d.get(
                "exclude_content_types", ["partial_manipulation"]
            ),
        )

    return AblationSpec(
        name=str(raw["name"]),
        seed=int(raw.get("seed", 20260830)),
        split=str(raw.get("split", "train")),
        real=_pool(raw["real"]) if raw.get("real") else None,
        fake=_pool(raw["fake"]) if raw.get("fake") else None,
        pools=[_pool(p) for p in raw.get("pools") or []],
    )


def run_ablation_config(config: Path, out: Path) -> list[SourceRecord]:
    spec = load_ablation_yaml(config)
    records = build_ablation(spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, records)
    n_real = sum(1 for r in records if r.label == 0)
    n_fake = sum(1 for r in records if r.label == 1)
    print(
        f"ablation {spec.name}: {len(records)} rows ({n_real} real / {n_fake} fake) -> {out}",
        flush=True,
    )
    return records
