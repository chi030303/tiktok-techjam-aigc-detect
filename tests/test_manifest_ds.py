# 2026-08-30, tianqi, load (path, y) from a tiny source JSONL
from pathlib import Path

from PIL import Image

from src.data.manifest_ds import load_source_manifest_rows, resolve_manifest_path
from src.transforms.manifest import SourceRecord, write_jsonl

# end


def _row(tmp: Path, name: str, label: int, image_id: str) -> SourceRecord:
    img = tmp / name
    Image.new("RGB", (8, 8), (10, 20, 30)).save(img)
    return SourceRecord(
        image_id=image_id,
        path=str(img),
        label=label,
        source_dataset="sid_set",
        generator=None if label == 0 else "flux",
        split="train",
        width=8,
        height=8,
        family=None if label == 0 else "diffusion",
        arch=None if label == 0 else "flow",
        generation_type=None if label == 0 else "t2i",
        content_type="real" if label == 0 else "full_synthetic",
    )


def test_load_source_manifest_rows(tmp_path: Path):
    recs = [
        _row(tmp_path, "r.png", 0, "a" * 40),
        _row(tmp_path, "f.png", 1, "b" * 40),
    ]
    man = tmp_path / "src.jsonl"
    write_jsonl(man, recs)
    rows = load_source_manifest_rows(man)
    assert {(p.name, y) for p, y in rows} == {("r.png", 0), ("f.png", 1)}


def test_resolve_manifest_under_data_prefix(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    dest = tmp_path / "manifests" / "ablation"
    dest.mkdir(parents=True)
    write_jsonl(dest / "D1_sid_only.jsonl", [_row(tmp_path, "r.png", 0, "c" * 40)])
    path = resolve_manifest_path("data/manifests/ablation/D1_sid_only.jsonl")
    assert path.is_file()
# end
