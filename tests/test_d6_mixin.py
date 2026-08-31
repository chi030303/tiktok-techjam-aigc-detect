# 2026-08-31, tianqi, D6 = D5 jsonl + A2 i2i fakes, skip paired reals
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_d6_adds_i2i_fakes_not_reals(tmp_path: Path):
    d5 = tmp_path / "D5.jsonl"
    a2 = tmp_path / "A2.jsonl"
    out = tmp_path / "D6.jsonl"
    d5_rows = [
        {
            "path": str(tmp_path / f"d5_{i}.png"),
            "label": 1,
            "generator": "flux2",
            "arch": "flow",
            "source": "data_new",
        }
        for i in range(8000)
    ]
    _write_jsonl(d5, d5_rows)
    a2_rows = []
    for i in range(60):
        a2_rows.append(
            {"path": str(tmp_path / f"real_{i}.jpg"), "label": 0, "generator": None}
        )
        a2_rows.append(
            {
                "path": str(tmp_path / f"sid{i}_nano.png"),
                "label": 1,
                "generator": "nano_banana",
            }
        )
        a2_rows.append(
            {
                "path": str(tmp_path / f"sid{i}_codex.png"),
                "label": 1,
                "generator": "codex",
            }
        )
    _write_jsonl(a2, a2_rows)
    rc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_d6_mixin.py"),
            "--d5",
            str(d5),
            "--a2",
            str(a2),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    mixed = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(mixed) == 8000 + 120
    assert sum(r["label"] == 0 for r in mixed) == 0
    gens = {r["generator"] for r in mixed}
    assert "codex" in gens
    assert "nano_banana" in gens
    assert all(
        r.get("generation_type") == "i2i"
        for r in mixed
        if r["generator"] in ("codex", "nano_banana")
    )
    assert "wrote" in rc.stdout
    assert "i2i_added=120" in rc.stdout
# end
