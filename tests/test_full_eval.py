# 2026-08-30, tianqi, full-eval formula / EvalGEN pairing / on-the-fly transforms (no GPU)
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.evalgen_pool import (
    filter_fakes_by_generators,
    generator_from_path,
    pair_evalgen,
    subsample_fakes_per_generator,
)
from src.eval.formula import N_ROBUST, official_formula
from src.eval.score import score_paired
from src.eval.transforms import apply_condition

# end


def _write_pattern(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:20, :, 0] = 255
    arr[:, :20, 1] = 180
    arr[10:30, 10:30, 2] = 255
    Image.fromarray(arr, mode="RGB").save(path)


def test_official_formula_complete_and_partial():
    clean = {"condition": "clean", "auroc": 0.90}
    robust = [{"condition": k, "auroc": 0.80} for k in (
        "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
        "blur_s05", "blur_s10", "blur_s20",
        "resize_s05", "resize_s025",
        "noise_s002", "noise_s005", "noise_s010",
        "jitter_p20", "crop_p80",
    )]
    full = official_formula([clean, *robust])
    assert N_ROBUST == 14
    assert full["complete"] is True
    assert full["formula"] == pytest.approx(0.85)
    partial = official_formula([clean, {"condition": "jpeg_q50", "auroc": 0.70}])
    assert partial["complete"] is False
    assert partial["n_robust"] == 1
    assert partial["formula"] == 0.5 * 0.90 + 0.5 * 0.70
    clean_only = official_formula([clean])
    assert clean_only["formula"] == 0.90
    assert clean_only["complete"] is False


def test_score_paired_uses_y_on_row():
    preds = [
        {"image_path": "real/a.png", "pred": 0.1, "y": 0},
        {"image_path": "fake/b.png", "pred": 0.9, "y": 1},
    ]
    metrics, errors = score_paired(preds)
    assert metrics["acc"] == 1.0
    assert metrics["auroc"] == 1.0
    assert errors["false_positives"] == []


def test_generator_from_evalgen_and_val_paths(tmp_path: Path):
    flux = tmp_path / "evalgen" / "flux" / "a.png"
    coco = tmp_path / "val" / "real" / "b.png"
    dalle = tmp_path / "val" / "fake" / "dalle3" / "c.png"
    assert generator_from_path(flux, tmp_path / "evalgen") == "flux"
    assert generator_from_path(coco, tmp_path / "val") == "real"
    assert generator_from_path(dalle, tmp_path / "val") == "dalle3"


def test_pair_evalgen_and_per_gen_subsample(tmp_path: Path):
    root = tmp_path / "evalgen"
    reals = tmp_path / "reals"
    for g in ("flux", "got"):
        for i in range(4):
            p = root / g / f"{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(p)
    for i in range(3):
        p = reals / f"r{i}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 20, dtype=np.uint8)).save(p)
    fakes = [(p, 1) for p in (root / "flux").glob("*.png")] + [
        (p, 1) for p in (root / "got").glob("*.png")
    ]
    real_rows = [(p, 0) for p in reals.glob("*.png")]
    picked = subsample_fakes_per_generator(fakes, root, n_per_gen=2, seed=0)
    assert len(picked) == 4
    by = {}
    for p, _y in picked:
        by.setdefault(generator_from_path(p, root), 0)
        by[generator_from_path(p, root)] += 1
    assert by == {"flux": 2, "got": 2}
    paired = pair_evalgen(picked, real_rows)
    assert sum(y == 0 for _p, y in paired) == 3
    assert sum(y == 1 for _p, y in paired) == 4
    # 2026-08-31, tianqi, Nova-only robust must drop other EvalGEN folders
    nova_only = filter_fakes_by_generators(fakes, root, {"nova"})
    assert nova_only == []
    flux_only = filter_fakes_by_generators(fakes, root, {"flux"})
    assert len(flux_only) == 4
    assert all(generator_from_path(p, root) == "flux" for p, _y in flux_only)
    # end


def test_jpeg_condition_differs_from_clean(tmp_path: Path):
    path = tmp_path / "real" / "a.png"
    _write_pattern(path)
    img = Image.open(path).convert("RGB")
    clean = apply_condition(img, "clean")
    jpeg = apply_condition(img, "jpeg_q30", seed=0)
    assert not np.array_equal(np.asarray(clean), np.asarray(jpeg))


def test_run_full_eval_dry_run_evalgen(tmp_path: Path):
    fake_root = tmp_path / "evalgen"
    reals = tmp_path / "reals"
    _write_pattern(fake_root / "flux" / "a.png")
    _write_pattern(reals / "r.png")
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_full_eval.py"),
            "--split",
            "evalgen",
            "--image-dir",
            str(fake_root),
            "--reals",
            "dir",
            "--reals-dir",
            str(reals),
            "--conditions",
            "clean",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "evalgen_dir" in proc.stdout
    assert "fake=1" in proc.stdout
    assert "real=1" in proc.stdout
# end
