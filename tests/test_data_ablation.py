# 2026-08-30, samily, ablation manifest sampling tests
from pathlib import Path

import pytest

from src.data.ablation import AblationSpec, PoolSpec, build_ablation, load_ablation_yaml
from src.transforms.manifest import SourceRecord, write_jsonl


def _sid_row(**kw) -> SourceRecord:
    base = dict(
        image_id="x" * 40,
        path="data/x.png",
        label=1,
        source_dataset="sid_set",
        generator="flux",
        split="train",
        width=512,
        height=512,
        family="diffusion",
        arch="flow",
        generation_type="t2i",
        content_type="full_synthetic",
    )
    base.update(kw)
    return SourceRecord(**base)


def _write_pool(tmp_path: Path, name: str, rows: list[SourceRecord]) -> Path:
    path = tmp_path / f"{name}.jsonl"
    write_jsonl(path, rows)
    return path


def test_ablation_d1_balanced_sample(tmp_path):
    reals = [
        _sid_row(image_id=f"r{i}" * 20, label=0, generator=None, family=None, arch=None,
                 generation_type=None, content_type="real", path=f"r{i}.png")
        for i in range(100)
    ]
    fakes = [
        _sid_row(image_id=f"f{i}" * 20, path=f"f{i}.png")
        for i in range(100)
    ]
    tampered = [
        _sid_row(
            image_id=f"t{i}" * 20,
            path=f"t{i}.png",
            content_type="partial_manipulation",
            generation_type="i2i",
        )
        for i in range(20)
    ]
    manifest = _write_pool(tmp_path, "sid", reals + fakes + tampered)
    spec = AblationSpec(
        name="smoke",
        seed=0,
        real=PoolSpec(manifest=str(manifest), n=40, label=0),
        fake=PoolSpec(
            manifest=str(manifest),
            n=40,
            label=1,
            family=["diffusion"],
            arch=["flow"],
        ),
    )
    out = build_ablation(spec)
    assert len(out) == 80
    assert sum(r.label == 0 for r in out) == 40
    assert sum(r.label == 1 for r in out) == 40
    assert all(r.content_type != "partial_manipulation" for r in out)


def test_ablation_yaml_load(tmp_path):
    rows = [
        _sid_row(
            image_id="r" * 40,
            label=0,
            generator=None,
            family=None,
            arch=None,
            generation_type=None,
            content_type="real",
        ),
        _sid_row(image_id="f" * 40),
    ]
    manifest = _write_pool(tmp_path, "sid", rows)
    cfg = tmp_path / "D1.yaml"
    cfg.write_text(
        f"""
name: D1
seed: 1
real:
  manifest: {manifest}
  n: 1
  label: 0
fake:
  manifest: {manifest}
  n: 1
  label: 1
  family: [diffusion]
""",
        encoding="utf-8",
    )
    spec = load_ablation_yaml(cfg)
    assert spec.name == "D1"
    assert spec.real.n == 1
    rows = build_ablation(spec)
    assert len(rows) == 2


def test_ablation_resolves_data_root_paths(tmp_path, monkeypatch):
    data = tmp_path / "shared-data"
    manifest = data / "manifests" / "source.jsonl"
    manifest.parent.mkdir(parents=True)
    write_jsonl(
        manifest,
        [
            _sid_row(
                image_id="r" * 40,
                label=0,
                generator=None,
                family=None,
                arch=None,
                generation_type=None,
                content_type="real",
            )
        ],
    )
    monkeypatch.setenv("DATA_ROOT", str(data))
    spec = AblationSpec(
        name="shared-path",
        real=PoolSpec(
            manifest="data/manifests/source.jsonl",
            n=1,
            label=0,
        ),
    )
    assert len(build_ablation(spec)) == 1
