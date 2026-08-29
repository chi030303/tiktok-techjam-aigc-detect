# 2026-08-29, zyun, source manifest builder (directory-tree labels)
"""Create a source manifest (JSONL) from a folder of images.

Labels come from the immediate parent directory name, or ``--label`` to force
one for the whole tree. image_id defaults to sha1("{rel_path}:{size}") so it is
stable across machines without hashing file bodies; ``--hash-content`` switches
to content hashing (needed for cross-dataset dedupe).

The official demo/val set carries a DO_NOT_TRAIN marker: indexing it as
``--split train`` is refused; any other split (evaluation) is allowed with a
stderr notice — the marker forbids training, not evaluation (docs/data.md).
Run from the repo root so stored paths stay relative.

Examples:
    python -m src.transforms.build_source --root data/cifake/train \\
        --dataset cifake --split train \\
        --out data/manifests/source_cifake_train.jsonl

    python -m src.transforms.build_source --root data/flux_out \\
        --dataset flux_gen --split unseen --generator flux1-dev --label 1 \\
        --out data/manifests/source_flux_unseen.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image

from .manifest import SPLITS, SourceRecord, write_jsonl

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def find_train_forbidden(root: Path) -> list[Path]:
    """DO_NOT_TRAIN markers under ``root`` (the official demo/val set carries one)."""
    return list(root.rglob("DO_NOT_TRAIN"))


def check_train_forbidden(root: Path) -> None:
    """Refuse to index a DO_NOT_TRAIN tree as training data (``split="train"``)."""
    for marker in find_train_forbidden(root):
        raise SystemExit(f"refusing to index train-forbidden tree: {marker}")


def image_id_for(rel_path: str, size: int, path: Path, hash_content: bool) -> str:
    if hash_content:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    return hashlib.sha1(f"{rel_path}:{size}".encode("utf-8")).hexdigest()


def collect_records(
    root: str | Path,
    dataset: str,
    split: str,
    generator: str | None = None,
    label: int | None = None,
    real_names: str = "REAL,real,authentic,nature,0",
    fake_names: str = "FAKE,fake,AI,aigc,synthetic,1",
    exts: set[str] | None = None,
    hash_content: bool = False,
) -> list[SourceRecord]:
    root = Path(root)
    if split == "train":
        # DO_NOT_TRAIN forbids training on the tree, not evaluating on it:
        # only a train-split index is refused (PR #1 review).
        check_train_forbidden(root)
    else:
        markers = find_train_forbidden(root)
        if markers:
            print(
                f"note: holdout tree carries DO_NOT_TRAIN ({markers[0]}); "
                "eval-only, never train",
                file=sys.stderr,
            )
    real = {s.strip().lower() for s in real_names.split(",") if s.strip()}
    fake = {s.strip().lower() for s in fake_names.split(",") if s.strip()}
    overlap = real & fake
    if overlap:
        raise SystemExit(f"names listed as both real and fake: {sorted(overlap)}")
    exts = exts or IMAGE_EXTS

    records: list[SourceRecord] = []
    n_real = n_fake = 0
    for path in sorted(root.rglob("*")):
        if not (path.is_file() and path.suffix.lower() in exts):
            continue
        try:
            rel = path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            rel = path.resolve().as_posix()  # outside CWD: keep it absolute
        if label is None:
            dir_name = path.parent.name.lower()
            if dir_name in real:
                lab = 0
            elif dir_name in fake:
                lab = 1
            else:
                raise SystemExit(
                    f"cannot infer label from parent dir '{path.parent.name}' for {path}; "
                    "extend --real-names/--fake-names or pass --label"
                )
        else:
            lab = label
        with Image.open(path) as im:
            width, height = im.size
        records.append(
            SourceRecord(
                image_id=image_id_for(rel, path.stat().st_size, path, hash_content),
                path=rel,
                label=lab,
                source_dataset=dataset,
                generator=generator,
                split=split,
                width=width,
                height=height,
            )
        )
        n_real += lab == 0
        n_fake += lab == 1
    print(
        f"collected {len(records)} images ({n_real} real / {n_fake} fake) from {root}",
        file=sys.stderr,
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True, help="cifake | sid_set | wildfake | flux_gen | ...")
    parser.add_argument("--split", required=True, choices=list(SPLITS))
    parser.add_argument("--generator", default=None, help="generator family for fakes, e.g. sd14, flux1-dev")
    parser.add_argument("--label", type=int, choices=(0, 1), default=None, help="force label; default: parent dir name")
    parser.add_argument("--real-names", default="REAL,real,authentic,nature,0")
    parser.add_argument("--fake-names", default="FAKE,fake,AI,aigc,synthetic,1")
    parser.add_argument("--ext", default=",".join(sorted(IMAGE_EXTS)))
    parser.add_argument(
        "--hash-content",
        action="store_true",
        help="image_id = sha1(file bytes) instead of sha1(rel:size)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    exts = {e.strip().lower() for e in args.ext.split(",") if e.strip()}
    records = collect_records(
        args.root,
        args.dataset,
        args.split,
        generator=args.generator,
        label=args.label,
        real_names=args.real_names,
        fake_names=args.fake_names,
        exts=exts,
        hash_content=args.hash_content,
    )
    write_jsonl(args.out, records)
    print(f"wrote {len(records)} rows -> {args.out}")


if __name__ == "__main__":
    main()
# end
