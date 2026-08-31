# 2026-08-31, tianqi, A2 i2i triplet pairing
import importlib.util
from pathlib import Path

from PIL import Image

from src.transforms.manifest import SourceRecord

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_a2_i2i", ROOT / "scripts" / "build_a2_i2i.py"
)
_build = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_build)
collect_groups = _build.collect_groups
complete_triplets = _build.complete_triplets
source_id_from_name = _build.source_id_from_name


def test_source_id_optional_on_old_jsonl():
    rec = SourceRecord.from_dict(
        {
            "image_id": "a" * 40,
            "path": "data/x.png",
            "label": 1,
            "source_dataset": "cifake",
            "generator": "sd14",
            "split": "test",
            "width": 32,
            "height": 32,
        }
    )
    assert rec.source_id is None


def test_i2i_name_and_complete_triplets(tmp_path: Path):
    root = tmp_path / "i2i"
    (root / "real").mkdir(parents=True)
    (root / "i2i_codex").mkdir()
    (root / "i2i_nano_banana").mkdir()
    sid = "abc123"
    Image.new("RGB", (8, 8), "red").save(root / "real" / f"{sid}.jpg")
    Image.new("RGB", (8, 8), "blue").save(root / "i2i_codex" / f"{sid}_codex.png")
    Image.new("RGB", (8, 8), "green").save(root / "i2i_nano_banana" / f"{sid}_nano.png")
    Image.new("RGB", (8, 8), "red").save(root / "real" / "orphan.jpg")
    assert source_id_from_name(root / "i2i_codex" / f"{sid}_codex.png") == sid
    groups = collect_groups(root)
    assert complete_triplets(groups) == [sid]
    assert "real" in groups["orphan"]
# end
