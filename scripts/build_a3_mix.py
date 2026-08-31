#!/usr/bin/env python3
# 2026-08-31, tianqi, A3 = D2 (8k real + t2i) plus i2i fakes from 60 triplets
"""A3 mix: D2_selfbuilt.jsonl + A2 i2i fakes (label=1). Reals stay D1 8k."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import data_root
from src.transforms.manifest import SourceRecord, read_jsonl, write_jsonl

# end


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d2", type=Path, default=None)
    p.add_argument("--a2", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    root = data_root()
    d2_path = args.d2 or (root / "manifests/ablation/D2_selfbuilt.jsonl")
    a2_path = args.a2 or (root / "manifests/ablation/A2_i2i_hard60.jsonl")
    out = args.out or (root / "manifests/ablation/A3_t2i_i2i_mix.jsonl")
    if not d2_path.is_file():
        raise SystemExit(f"missing D2 {d2_path}; run build_d2_selfbuilt.py")
    if not a2_path.is_file():
        raise SystemExit(f"missing A2 {a2_path}; run build_a2_i2i.py")

    # 2026-08-31, tianqi, read_jsonl requires kind=source
    d2 = read_jsonl(d2_path, kind="source")
    i2i_fakes = [r for r in read_jsonl(a2_path, kind="source") if r.label == 1]
    # end
    if len(i2i_fakes) < 50:
        raise SystemExit(f"too few i2i fakes: {len(i2i_fakes)}")

    seen = {r.path for r in d2}
    extra = [r for r in i2i_fakes if r.path not in seen]
    rows: list[SourceRecord] = list(d2) + extra
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    print(
        f"wrote {out} n={len(rows)} "
        f"real={sum(r.label==0 for r in rows)} fake={sum(r.label==1 for r in rows)} "
        f"i2i_added={len(extra)} by_gt={dict(Counter(r.generation_type for r in rows))}"
    )


if __name__ == "__main__":
    main()
