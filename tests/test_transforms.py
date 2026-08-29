# 2026-08-29, zyun, tests for transform ops, manifest schema, and both builders
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

from src.transforms import ops, spec
from src.transforms.build import run_build
from src.transforms.build_source import check_train_forbidden, collect_records
from src.transforms.manifest import (
    SourceRecord,
    TransformRecord,
    read_jsonl,
    write_jsonl,
)

EXPECTED_KEYS = {
    "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
    "blur_s05", "blur_s10", "blur_s20",
    "resize_s05", "resize_s025",
    "noise_s002", "noise_s005", "noise_s010",
    "jitter_p20", "crop_p80",
}


def make_image(w: int = 64, h: int = 48, seed: int = 0) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    arr[:, :, 1] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    arr[:, :, 2] = (seed * 37) % 256
    return Image.fromarray(arr, mode="RGB")


def decode(result):
    if isinstance(result, bytes):
        out = Image.open(io.BytesIO(result))
        out.load()
        return out
    return result


def source_row(**overrides) -> dict:
    row = {
        "image_id": "a" * 40,
        "path": "data/x.png",
        "label": 1,
        "source_dataset": "cifake",
        "generator": "sd14",
        "split": "test",
        "width": 32,
        "height": 32,
    }
    row.update(overrides)
    return row


def test_official_settings_match_problem_statement():
    assert {s.key for s in spec.OFFICIAL_SETTINGS} == EXPECTED_KEYS
    assert len(spec.OFFICIAL_SETTINGS) == 14
    assert [s.key for s in spec.OFFICIAL_SETTINGS if s.op == "jpeg"] == [
        "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
    ]


def test_seed_derivation_stable_and_distinct():
    assert spec.derive_seed("img1", "jpeg_q50") == spec.derive_seed("img1", "jpeg_q50")
    assert spec.derive_seed("img1", "jpeg_q50") != spec.derive_seed("img2", "jpeg_q50")
    assert spec.derive_seed("img1", "jpeg_q50") != spec.derive_seed("img1", "jpeg_q30")
    assert 0 <= spec.derive_seed("img1", "jpeg_q50") < 2**64


def test_size_contract():
    img = make_image()
    rng = np.random.default_rng(0)
    for key in ("jpeg_q50", "blur_s10", "resize_s025", "jitter_p20", "noise_s005"):
        out, _ = ops.apply_setting(img, spec.SETTINGS_BY_KEY[key], rng)
        assert decode(out).size == img.size, key
    crop, _ = ops.apply_setting(img, spec.SETTINGS_BY_KEY["crop_p80"], rng)
    assert crop.size == (round(64 * 0.8), round(48 * 0.8))
    back, _ = ops.apply_setting(
        img, spec.SETTINGS_BY_KEY["crop_p80"], rng, crop_resize_back=True
    )
    assert decode(back).size == img.size


def test_jpeg_quality_changes_bytes():
    img = make_image()
    b90 = ops.jpeg_compress_bytes(img, 90)
    b30 = ops.jpeg_compress_bytes(img, 30)
    assert b90 != b30
    assert len(b90) >= len(b30)


def test_noise_seeded_determinism_and_strength():
    img = make_image()
    same = ops.gaussian_noise(img, 0.05, np.random.default_rng(7))
    again = ops.gaussian_noise(img, 0.05, np.random.default_rng(7))
    other = ops.gaussian_noise(img, 0.05, np.random.default_rng(8))
    assert same.tobytes() == again.tobytes()
    assert same.tobytes() != other.tobytes()

    base = np.asarray(img, dtype=np.float32)
    weak = np.abs(np.asarray(ops.gaussian_noise(img, 0.02, np.random.default_rng(1)), dtype=np.float32) - base).mean()
    strong = np.abs(np.asarray(ops.gaussian_noise(img, 0.10, np.random.default_rng(1)), dtype=np.float32) - base).mean()
    assert strong > weak


def test_jitter_factors_in_range_and_stable():
    f1 = ops.sample_jitter_factors(np.random.default_rng(3), 0.2)
    f2 = ops.sample_jitter_factors(np.random.default_rng(3), 0.2)
    assert f1 == f2
    for v in f1.values():
        assert 0.8 <= v <= 1.2


def test_manifest_schema_strict():
    rec = SourceRecord.from_dict(source_row())
    assert rec.label == 0 or rec.label == 1
    with pytest.raises(ValueError):
        SourceRecord.from_dict(source_row(label=2))
    with pytest.raises(ValueError):
        SourceRecord.from_dict({**source_row(), "oops": 1})
    with pytest.raises(ValueError):
        SourceRecord.from_dict({k: v for k, v in source_row().items() if k != "split"})

    trow = {
        "row_id": "a_jpeg_q50",
        "source_image_id": "a" * 40,
        "source_path": "data/x.png",
        "transform": "jpeg",
        "transform_key": "jpeg_q50",
        "params": {"quality": 50},
        "seed": 1,
        "path": "data/transforms/jpeg_q50/aa/a.jpg",
        "label": 1,
        "source_dataset": "cifake",
        "generator": "sd14",
        "split": "test",
        "width": 32,
        "height": 32,
    }
    TransformRecord.from_dict(trow)
    with pytest.raises(ValueError):
        TransformRecord.from_dict({**trow, "transform_key": "jpeg_q66"})
    with pytest.raises(ValueError):
        TransformRecord.from_dict({**trow, "params": {}})


def test_do_not_train_guard(tmp_path):
    root = tmp_path / "demo"
    (root / "sub").mkdir(parents=True)
    (root / "DO_NOT_TRAIN").touch()
    (root / "sub" / "x.png").touch()
    with pytest.raises(SystemExit):
        check_train_forbidden(root)


def test_build_source_and_transform_end_to_end(tmp_path):
    root = tmp_path / "cifake" / "train"
    (root / "FAKE").mkdir(parents=True)
    (root / "REAL").mkdir(parents=True)
    make_image(seed=1).save(root / "FAKE" / "a.png")
    make_image(seed=2).save(root / "FAKE" / "b.png")
    make_image(seed=3).save(root / "REAL" / "c.png")

    records = collect_records(root=root, dataset="cifake", split="train")
    assert [r.label for r in records] == [1, 1, 0]  # sorted: FAKE/a, FAKE/b, REAL/c
    assert all(r.width == 64 and r.height == 48 for r in records)
    source_manifest = tmp_path / "source.jsonl"
    write_jsonl(source_manifest, records)
    assert all(Path(r.path).is_file() for r in read_jsonl(source_manifest, kind="source"))

    settings = spec.resolve_settings("blur_s10,crop_p80")
    out_root = tmp_path / "transforms"
    out_manifest = tmp_path / "t.jsonl"
    stats = run_build(read_jsonl(source_manifest, kind="source"), settings, out_root, out_manifest)
    assert stats == {"rows": 6, "made": 6, "skipped": 0}

    rows = read_jsonl(out_manifest, kind="transform")
    assert len(rows) == 6
    for row in rows:
        assert Path(row.path).is_file()
        assert row.seed == spec.derive_seed(row.source_image_id, row.transform_key)
        assert row.split == "train"
        if row.transform == "crop":
            assert (row.width, row.height) == (round(64 * 0.8), round(48 * 0.8))
            assert Image.open(row.path).size == (row.width, row.height)
        else:
            assert (row.width, row.height) == (64, 48)

    # rerun: skip existing files, manifest content byte-identical in structure
    first = [r.to_json() for r in rows]
    stats2 = run_build(read_jsonl(source_manifest, kind="source"), settings, out_root, out_manifest)
    assert stats2 == {"rows": 6, "made": 0, "skipped": 6}
    assert [r.to_json() for r in read_jsonl(out_manifest, kind="transform")] == first


def test_run_build_split_filter(tmp_path):
    src = SourceRecord.from_dict(
        source_row(image_id="b" * 40, split="train", path="data/nowhere.png")
    )
    # split mismatch and empty input are both hard errors (CLI guard)
    with pytest.raises(SystemExit):
        run_build(
            [src],
            spec.resolve_settings("blur_s10"),
            tmp_path / "t",
            tmp_path / "m.jsonl",
            splits={"unseen"},
        )
    with pytest.raises(SystemExit):
        run_build([], spec.resolve_settings("blur_s10"), tmp_path / "t", tmp_path / "m.jsonl")
def test_collect_records_allows_holdout_indexing(tmp_path, capsys):
    """DO_NOT_TRAIN forbids training, not evaluation: non-train indexing passes."""
    root = tmp_path / "val"
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    (root / "DO_NOT_TRAIN").touch()
    make_image(seed=1).save(root / "real" / "a.png")
    make_image(seed=2).save(root / "fake" / "b.png")
    records = collect_records(root=root, dataset="demo_wildfake", split="val")
    assert sorted(r.label for r in records) == [0, 1]  # one real, one fake
    assert "DO_NOT_TRAIN" in capsys.readouterr().err  # notice printed


def test_collect_records_refuses_train_on_marked_tree(tmp_path):
    root = tmp_path / "val"
    (root / "real").mkdir(parents=True)
    (root / "DO_NOT_TRAIN").touch()
    make_image().save(root / "real" / "a.png")
    with pytest.raises(SystemExit):
        collect_records(root=root, dataset="demo_wildfake", split="train")


def _run_cli(module_args: list[str]):
    return subprocess.run(
        [sys.executable, "-m", *module_args], capture_output=True, text=True, cwd=ROOT
    )


def test_cli_defaults_match_documented_flow(tmp_path):
    """The README flow must run with CLI default args (review #1)."""
    root = tmp_path / "cifake" / "test"
    (root / "FAKE").mkdir(parents=True)
    (root / "REAL").mkdir(parents=True)
    make_image(seed=1).save(root / "FAKE" / "a.png")
    make_image(seed=2).save(root / "REAL" / "b.png")

    r1 = _run_cli([
        "src.transforms.build_source", "--root", str(root),
        "--dataset", "cifake", "--split", "test",
        "--out", str(tmp_path / "source.jsonl"),
    ])
    assert r1.returncode == 0, r1.stderr

    r2 = _run_cli([
        "src.transforms.build",
        "--source-manifest", str(tmp_path / "source.jsonl"),
        "--out-manifest", str(tmp_path / "t.jsonl"),
        "--out-root", str(tmp_path / "tr"),
    ])
    assert r2.returncode == 0, r2.stderr
    rows = (tmp_path / "t.jsonl").read_text().splitlines()
    assert len(rows) == 2 * len(spec.OFFICIAL_SETTINGS)  # all 14 settings applied


def test_cli_default_excludes_train_source(tmp_path):
    """A train-only source must fail under CLI default splits (review #1/#2)."""
    root = tmp_path / "cifake" / "train"
    (root / "FAKE").mkdir(parents=True)
    make_image().save(root / "FAKE" / "a.png")

    r1 = _run_cli([
        "src.transforms.build_source", "--root", str(root),
        "--dataset", "cifake", "--split", "train",
        "--out", str(tmp_path / "source.jsonl"),
    ])
    assert r1.returncode == 0, r1.stderr

    r2 = _run_cli([
        "src.transforms.build",
        "--source-manifest", str(tmp_path / "source.jsonl"),
        "--out-manifest", str(tmp_path / "t.jsonl"),
        "--out-root", str(tmp_path / "tr"),
    ])
    assert r2.returncode != 0
    assert "no source rows" in (r2.stderr + r2.stdout)
# end
