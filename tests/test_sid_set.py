from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from src.data.sid_set import collect_parquet


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    out = BytesIO()
    Image.new("RGB", (32, 24), color).save(out, format="PNG")
    return out.getvalue()


def _write_shard(path, rows):
    table = pa.Table.from_pylist(
        [{"image": {"bytes": raw, "path": None}, "label": label} for raw, label in rows]
    )
    pq.write_table(table, path)


def test_sid_parquet_is_resumable_and_drops_tampered(tmp_path):
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    raw_real = _png_bytes((1, 2, 3))
    raw_fake = _png_bytes((4, 5, 6))
    _write_shard(
        parquet_dir / "train-00000-of-00001.parquet",
        [(raw_real, 0), (_png_bytes((7, 8, 9)), 2), (raw_fake, 1)],
    )
    extracted = tmp_path / "extracted"

    first = collect_parquet(parquet_dir, extracted, max_rows=2, compute_phash=False)
    second = collect_parquet(parquet_dir, extracted, max_rows=2, compute_phash=False)

    assert len(first) == len(second) == 2
    assert {row.label for row in second} == {0, 1}
    assert all(row.content_type != "partial_manipulation" for row in second)
    assert sorted(path.read_bytes() for path in extracted.rglob("*.png")) == sorted(
        [raw_real, raw_fake]
    )


def test_sid_val_reads_validation_shards_only(tmp_path):
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    _write_shard(
        parquet_dir / "train-00000-of-00001.parquet",
        [(_png_bytes((1, 1, 1)), 0)],
    )
    _write_shard(
        parquet_dir / "validation-00000-of-00001.parquet",
        [(_png_bytes((2, 2, 2)), 1)],
    )
    rows = collect_parquet(
        parquet_dir,
        tmp_path / "extracted",
        split="val",
        compute_phash=False,
    )
    assert len(rows) == 1
    assert rows[0].split == "val"
    assert rows[0].label == 1
