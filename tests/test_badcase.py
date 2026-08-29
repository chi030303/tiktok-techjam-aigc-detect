# 2026-08-29, zyun, tests for the bad-case collection & statistics pipeline
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.badcase import (
    join_predictions,
    load_manifest_rows,
    summarize,
    write_stats_csv,
)
from tests.test_transforms import ROOT, make_image


def _label_rows(paths_by_label):
    return [(p, y) for y, ps in paths_by_label.items() for p in ps]


def _pred_rows(paths_scores):
    return [{"image_path": str(p), "pred": s} for p, s in paths_scores]


def _manifest_row(path, generator, dataset):
    return {"path": str(path), "generator": generator, "source_dataset": dataset}


def test_join_classifies_and_attaches_metadata(tmp_path):
    fake = tmp_path / "fake" / "a.png"
    real = tmp_path / "real" / "b.png"
    fake.parent.mkdir(parents=True)
    real.parent.mkdir(parents=True)
    make_image(seed=1).save(fake)
    make_image(seed=2).save(real)

    preds = _pred_rows([(fake, 0.9), (real, 0.8), (real.parent / "c.png", 0.5)])
    preds.append({"image_path": "data/nowhere.png", "pred": 0.1})  # unmatched label
    manifest = [_manifest_row(fake, "sd14", "cifake")]  # real/b.png has no metadata

    res = join_predictions(
        preds,
        rows=_label_rows({1: [fake], 0: [real]}),
        src_root=tmp_path,
        predict_root=tmp_path,
        threshold=0.5,
        manifest_rows=manifest,
        default_condition="clean",
    )
    joined = {r["image_path"]: r for r in res["joined"]}
    assert joined[str(fake)]["error_type"] == "TP"
    assert joined[str(real)]["error_type"] == "FP"  # real image scored 0.8 >= 0.5
    assert joined[str(fake)]["generator"] == "sd14"
    assert joined[str(real)]["generator"] == "unknown"  # absent from manifest
    assert joined[str(real)]["condition"] == "clean"
    assert res["unmatched_labels"] == 2  # c.png has no label; nowhere.png is foreign
    assert res["unmatched_metadata"] == 1  # only real/b.png lacks manifest metadata


def test_summarize_groups_and_worst_k(tmp_path):
    rows = [
        {"image_path": "f1", "pred": 0.9, "label": 0, "error_type": "FP",
         "condition": "jpeg_q50", "generator": "sd14", "source_dataset": "cifake"},
        {"image_path": "f2", "pred": 0.6, "label": 0, "error_type": "FP",
         "condition": "clean", "generator": "unknown", "source_dataset": "cifake"},
        {"image_path": "m1", "pred": 0.4, "label": 1, "error_type": "FN",
         "condition": "jpeg_q50", "generator": "sd14", "source_dataset": "cifake"},
        {"image_path": "ok", "pred": 0.95, "label": 1, "error_type": "TP",
         "condition": "clean", "generator": "unknown", "source_dataset": "cifake"},
    ]
    summary = summarize(rows, threshold=0.5, worst_k=1)
    assert summary["n_images"] == 4 and summary["n_fp"] == 2 and summary["n_fn"] == 1
    assert summary["by_generator"]["sd14"] == {
        "n_images": 2, "n_fp": 1, "n_fn": 1, "fp_rate": 0.5, "fn_rate": 0.5,
    }
    assert summary["by_condition"]["jpeg_q50"]["n_images"] == 2
    assert [r["image_path"] for r in summary["worst_fp"]] == ["f1"]  # highest pred first
    assert [r["image_path"] for r in summary["worst_fn"]] == ["m1"]  # lowest pred first

    out = tmp_path / "stats.csv"
    write_stats_csv(out, summary)
    with out.open(newline="", encoding="utf-8") as fh:
        table = list(csv.reader(fh))
    assert table[0] == ["group_type", "group_value", "n_images", "n_fp", "n_fn", "fp_rate", "fn_rate"]
    assert ["overall", "all", "4", "2", "1", "0.5", "0.25"] in table
    assert ["generator", "sd14", "2", "1", "1", "0.5", "0.5"] in table


def test_cli_end_to_end(tmp_path):
    real = tmp_path / "real"
    fake = tmp_path / "fake"
    real.mkdir()
    fake.mkdir()
    images = {
        "fake/a.png": (fake / "a.png", 0.9),   # TP
        "fake/b.png": (fake / "b.png", 0.3),   # FN
        "real/c.png": (real / "c.png", 0.8),   # FP
        "real/d.png": (real / "d.png", 0.1),   # TN
    }
    preds = []
    for rel, (path, score) in images.items():
        make_image(seed=len(rel)).save(path)
        preds.append({"image_path": str(path), "pred": score})

    pred_json = tmp_path / "pred.json"
    pred_json.write_text(json.dumps(preds), encoding="utf-8")

    manifest = tmp_path / "source.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"path": str(fake / "a.png"), "generator": "sd14",
                             "source_dataset": "cifake"}) + "\n")
        fh.write(json.dumps({"path": str(real / "c.png"), "generator": None,
                             "source_dataset": "cifake"}) + "\n")

    out_dir = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "scripts/run_badcase.py", "--pred", str(pred_json),
         "--image-dir", str(tmp_path), "--manifest", str(manifest),
         "--out-dir", str(out_dir)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr

    badcases = [json.loads(l) for l in (out_dir / "badcases.jsonl").read_text().splitlines()]
    assert {(b["image_path"], b["error_type"]) for b in badcases} == {
        (str(fake / "b.png"), "FN"),
        (str(real / "c.png"), "FP"),
    }
    by_path = {b["image_path"]: b for b in badcases}
    assert by_path[str(fake / "b.png")]["generator"] == "unknown"

    summary = json.loads((out_dir / "badcase_summary.json").read_text(encoding="utf-8"))
    assert summary["n_images"] == 4 and summary["n_fp"] == 1 and summary["n_fn"] == 1
    assert summary["metrics"]["auroc"] == 0.75  # 3/4 positive-negative pairs ranked right
    assert summary["by_generator"]["sd14"]["n_fp"] == 0
    assert (out_dir / "badcase_stats.csv").is_file()


def test_manifest_reader_validates(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"path": "x.png"}\nnot json\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        load_manifest_rows([bad])
    empty = tmp_path / "empty.jsonl"
    empty.write_text('{"label": 1}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        load_manifest_rows([empty])
# end
