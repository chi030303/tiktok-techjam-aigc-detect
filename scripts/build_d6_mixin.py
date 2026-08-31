#!/usr/bin/env python3
# 2026-08-31, tianqi, D6 = D5 mixin + A2 i2i fakes (label=1 only; no paired reals)
"""Build D6 mixin: D5 SID mix plus whole-image i2i reconstructions.

Same protocol as D3–D5: mixin fakes replace equal SID FLUX. Adds the A2
triplet fakes (Codex + nano i2i, ~118) on top of D5. Does **not** add the
59 paired reals (SID already has ~70k reals; A3 also mixed fakes only).
Dedup by lowercase filename. No Hunyuan, no official val, no EvalGEN.

  DATA_ROOT=/workspace/data python scripts/build_d6_mixin.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import data_root

# end


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d5", type=Path, default=None)
    p.add_argument("--a2", type=Path, default=None)
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    root = data_root()
    d5_path = args.d5 or (root / "manifests" / "ablation" / "D5_sid_mixin.jsonl")
    a2_path = args.a2 or (root / "manifests" / "ablation" / "A2_i2i_hard60.jsonl")
    out = args.out or (root / "manifests" / "ablation" / "D6_sid_mixin.jsonl")
    if not d5_path.is_file():
        raise SystemExit(f"missing D5 {d5_path}; run scripts/build_d5_mixin.py")
    if not a2_path.is_file():
        raise SystemExit(f"missing A2 {a2_path}; run scripts/build_a2_i2i.py")

    d5 = _load_jsonl(d5_path)
    i2i_fakes = [r for r in _load_jsonl(a2_path) if int(r["label"]) == 1]
    if len(d5) < 8000:
        raise SystemExit(f"D5 mixin too small: {len(d5)}")
    if len(i2i_fakes) < 50:
        raise SystemExit(f"too few i2i fakes: {len(i2i_fakes)}")

    seen: set[str] = set()
    rows: list[dict] = []
    by_gen: Counter[str] = Counter()
    i2i_added = 0

    def add(rec: dict, *, i2i: bool) -> None:
        nonlocal i2i_added
        path = Path(rec["path"])
        key = path.name.lower()
        if key in seen:
            return
        seen.add(key)
        row = {
            "path": str(path),
            "label": int(rec["label"]),
            "generator": rec.get("generator"),
            "arch": rec.get("arch"),
            "source": rec.get("source") or ("self_built_i2i" if i2i else "data_new"),
        }
        if i2i:
            row["generation_type"] = "i2i"
            row["source"] = "self_built_i2i"
            i2i_added += 1
        rows.append(row)
        by_gen[str(row["generator"])] += 1

    for rec in d5:
        add(rec, i2i=False)
    for rec in i2i_fakes:
        add(rec, i2i=True)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"wrote {out} n={len(rows)} n_d5={len(d5)} i2i_added={i2i_added} "
        f"by_gen={dict(by_gen)}",
        flush=True,
    )
    if "hunyuan" in by_gen:
        raise SystemExit(f"D6 must not include hunyuan, got {by_gen['hunyuan']}")
    if i2i_added < 50:
        raise SystemExit(f"D6 i2i add too small after dedup: {i2i_added}")
    if len(rows) < len(d5):
        raise SystemExit(f"D6 smaller than D5: {len(rows)} < {len(d5)}")


if __name__ == "__main__":
    main()
# end
