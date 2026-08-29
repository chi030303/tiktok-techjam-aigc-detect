# 2026-08-29, tianqi, score JSON + robustness loop against a 2-image labeled folder
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.labels import load_labeled_dir, subsample_balanced
from src.eval.robustness import robustness_table
from src.eval.score import score_predictions
from src.eval.table import write_csv, write_markdown

# end


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((16, 16, 3), color, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _tiny_split(tmp: Path) -> Path:
    root = tmp / "val"
    _write_rgb(root / "real" / "a.png", (10, 20, 30))
    _write_rgb(root / "fake" / "b.png", (200, 10, 10))
    return root


def test_load_real_fake_dirs(tmp_path: Path) -> None:
    root = _tiny_split(tmp_path)
    rows = load_labeled_dir(root)
    by = {p.name: y for p, y in rows}
    assert by == {"a.png": 0, "b.png": 1}


def test_score_predictions_matches_rel_paths(tmp_path: Path) -> None:
    root = _tiny_split(tmp_path)
    rows = load_labeled_dir(root)
    preds = [
        {"image_path": str(root / "real" / "a.png"), "pred": 0.1},
        {"image_path": str(root / "fake" / "b.png"), "pred": 0.9},
    ]
    metrics, errors = score_predictions(preds, rows, src_root=root, predict_root=root)
    assert metrics["acc"] == 1.0
    assert metrics["auroc"] == 1.0
    assert errors["false_positives"] == []
    assert errors["false_negatives"] == []


def test_score_false_positive_is_ranked(tmp_path: Path) -> None:
    root = _tiny_split(tmp_path)
    rows = load_labeled_dir(root)
    preds = [
        {"image_path": str(root / "real" / "a.png"), "pred": 0.99},
        {"image_path": str(root / "fake" / "b.png"), "pred": 0.01},
    ]
    metrics, errors = score_predictions(preds, rows, src_root=root, predict_root=root)
    assert metrics["fp"] == 1 and metrics["fn"] == 1
    assert errors["false_positives"][0]["pred"] == 0.99
    assert errors["false_negatives"][0]["pred"] == 0.01


def test_robustness_table_via_injected_predict(tmp_path: Path) -> None:
    root = _tiny_split(tmp_path)
    rows = load_labeled_dir(root)

    def predict_fn(image_dir: Path, out_json: Path) -> None:
        files = sorted(p for p in image_dir.rglob("*.png"))
        payload = []
        for p in files:
            pred = 0.1 if "real" in p.parts else 0.9
            payload.append({"image_path": str(p), "pred": pred})
        out_json.write_text(json.dumps(payload), encoding="utf-8")

    table, errors_by = robustness_table(
        rows,
        src_root=root,
        conditions=["clean", "jpeg_q50"],
        work_root=tmp_path / "work",
        predict_fn=predict_fn,
        split_name="val",
    )
    assert [r["condition"] for r in table] == ["clean", "jpeg_q50"]
    assert all(r["acc"] == 1.0 for r in table)
    assert "jpeg_q50" in errors_by
    csv_path = tmp_path / "t.csv"
    md_path = tmp_path / "t.md"
    write_csv(table, csv_path)
    write_markdown(table, md_path)
    text = csv_path.read_text(encoding="utf-8")
    assert "jpeg_q50" in text and "acc" in text
    assert "center_crop_80" not in text
    # 2026-08-29, tianqi, spec key crop_p80 is the daily crop name now
    assert "crop_p80" not in text
    # end
    assert md_path.read_text(encoding="utf-8").startswith("| model")


def test_subset_copy_does_not_include_other_files(tmp_path: Path) -> None:
    root = tmp_path / "val"
    _write_rgb(root / "real" / "keep.png", (1, 2, 3))
    _write_rgb(root / "real" / "drop.png", (4, 5, 6))
    _write_rgb(root / "fake" / "keep.png", (7, 8, 9))
    rows = [(root / "real" / "keep.png", 0), (root / "fake" / "keep.png", 1)]

    def predict_fn(image_dir: Path, out_json: Path) -> None:
        files = list(image_dir.rglob("*.png"))
        assert {p.name for p in files} == {"keep.png"}
        payload = [{"image_path": str(p), "pred": 0.2 if "real" in p.parts else 0.8} for p in files]
        out_json.write_text(json.dumps(payload), encoding="utf-8")

    table, _ = robustness_table(
        rows,
        src_root=root,
        conditions=["clean"],
        work_root=tmp_path / "work",
        predict_fn=predict_fn,
        split_name="val",
    )
    assert table[0]["n"] == 2


def test_run_eval_score_cli(tmp_path: Path) -> None:
    import subprocess
    import sys

    root = _tiny_split(tmp_path)
    pred = tmp_path / "pred.json"
    pred.write_text(
        json.dumps(
            [
                {"image_path": str(root / "real" / "a.png"), "pred": 0.2},
                {"image_path": str(root / "fake" / "b.png"), "pred": 0.8},
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "score.json"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_eval.py"),
            "score",
            "--image-dir",
            str(root),
            "--pred",
            str(pred),
            "--out",
            str(out),
        ],
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metrics"]["acc"] == 1.0


def test_subsample_keeps_both_classes(tmp_path: Path) -> None:
    root = tmp_path / "val"
    for i in range(6):
        _write_rgb(root / "real" / f"r{i}.png", (i, 0, 0))
        _write_rgb(root / "fake" / f"f{i}.png", (0, i, 0))
    rows = load_labeled_dir(root)
    picked = subsample_balanced(rows, 4, seed=0)
    assert len(picked) == 4
    assert {y for _p, y in picked} == {0, 1}
# end
