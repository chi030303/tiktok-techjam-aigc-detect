# 2026-08-29, tianqi, apply frozen TechJam transforms then score via predict.py
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from src.eval.score import score_predictions
from src.eval.transforms import apply_condition, seed_for
from src.paths import REPO_ROOT

# end

PredictFn = Callable[[Path, Path], None]


def save_rgb(img: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ext = dest.suffix.lower()
    rgb = img.convert("RGB")
    if ext in {".jpg", ".jpeg"}:
        rgb.save(dest, format="JPEG", quality=100)
    else:
        rgb.save(dest, format="PNG")


def write_condition(
    rows: list[tuple[Path, int]],
    src_root: Path,
    dest_root: Path,
    condition: str,
) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    for path, _y in rows:
        rel = path.relative_to(src_root)
        img = Image.open(path).convert("RGB")
        seed = seed_for(rel.as_posix(), condition)
        out = apply_condition(img, condition, seed=seed)
        save_rgb(out, dest_root / rel)


def default_predict(image_dir: Path, out_json: Path, ckpt: Path | None = None) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "predict.py"), str(image_dir), str(out_json)]
    if ckpt is not None:
        cmd.extend(["--ckpt", str(ckpt)])
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def run_condition(
    rows: list[tuple[Path, int]],
    src_root: Path,
    condition: str,
    work_root: Path,
    predict_fn: PredictFn,
    threshold: float,
    split_name: str,
) -> tuple[dict, dict]:
    pred_json = work_root / f"pred_{condition}.json"
    # 2026-08-29, tianqi, always copy the selected subset so --max-images cannot leak the rest of val
    image_dir = work_root / "images" / condition
    write_condition(rows, src_root, image_dir, condition)
    # end
    predict_fn(image_dir, pred_json)
    preds = json.loads(pred_json.read_text(encoding="utf-8"))
    metrics, errors = score_predictions(
        preds,
        rows,
        src_root=src_root,
        predict_root=image_dir,
        threshold=threshold,
    )
    metrics["split"] = split_name
    metrics["condition"] = condition
    return metrics, errors


def robustness_table(
    rows: list[tuple[Path, int]],
    src_root: Path,
    conditions: list[str],
    work_root: Path,
    predict_fn: PredictFn,
    threshold: float = 0.5,
    split_name: str = "custom",
) -> tuple[list[dict], dict[str, dict]]:
    table: list[dict] = []
    errors_by: dict[str, dict] = {}
    work_root.mkdir(parents=True, exist_ok=True)
    for cond in conditions:
        print(f"eval condition {cond}  n={len(rows)}", flush=True)
        metrics, errors = run_condition(
            rows,
            src_root,
            cond,
            work_root,
            predict_fn,
            threshold,
            split_name,
        )
        table.append(metrics)
        errors_by[cond] = errors
        acc = metrics["acc"]
        roc = metrics["auroc"]
        roc_s = f"{roc:.3f}" if isinstance(roc, float) else "na"
        print(f"  acc={acc:.3f}  auroc={roc_s}  n={metrics['n']}", flush=True)
    return table, errors_by
# end
