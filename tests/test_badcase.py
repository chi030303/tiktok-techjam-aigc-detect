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
    assert joined[str(real)]["generator"] == "real"  # reals have no generator by definition
    assert joined[str(real)]["condition"] == "clean"
    assert res["unmatched_labels"] == 2  # c.png has no label; nowhere.png is foreign
    assert res["unmatched_metadata"] == 1  # only real/b.png lacks manifest metadata


def test_join_metadata_survives_path_form_mismatch(tmp_path, monkeypatch):
    """Absolute pred paths must still attach to a CWD-relative manifest (PR#5 review).

    predict.py emits whatever form the caller passed (absolute on Vast) while
    build_source stores CWD-relative paths; string-equality matching silently
    dropped every generator. Both directions must survive.
    """
    monkeypatch.chdir(tmp_path)
    fake = Path("images/fake/a.png")
    fake.parent.mkdir(parents=True)
    make_image(seed=1).save(fake)
    manifest = [{"path": "images/fake/a.png", "generator": "dalle3",
                 "source_dataset": "demo_wildfake"}]

    # direction 1: absolute pred paths (Vast style) + relative manifest
    res = join_predictions(
        [{"image_path": str(fake.resolve()), "pred": 0.9}],
        rows=[(fake, 1)],
        src_root=Path("images"),
        predict_root=Path("images"),
        threshold=0.5,
        manifest_rows=manifest,
    )
    assert res["joined"][0]["generator"] == "dalle3"
    assert res["unmatched_metadata"] == 0

    # direction 2: relative pred paths + absolute manifest
    res2 = join_predictions(
        [{"image_path": "images/fake/a.png", "pred": 0.9}],
        rows=[(fake.resolve(), 1)],
        src_root=fake.resolve().parent.parent,
        predict_root=fake.resolve().parent.parent,
        threshold=0.5,
        manifest_rows=[{**manifest[0], "path": str(fake.resolve())}],
    )
    assert res2["joined"][0]["generator"] == "dalle3"
    assert res2["unmatched_metadata"] == 0


def test_join_rejects_malformed_pred_json(tmp_path):
    real = tmp_path / "real" / "a.png"
    real.parent.mkdir(parents=True)
    make_image(seed=1).save(real)
    rows = [(real, 0)]
    with pytest.raises(SystemExit):  # top level not a list
        join_predictions({"not": "a list"}, rows=rows, src_root=tmp_path, predict_root=tmp_path)
    with pytest.raises(SystemExit):  # row not an object
        join_predictions(["data/val/a.png"], rows=rows, src_root=tmp_path, predict_root=tmp_path)
    with pytest.raises(SystemExit):  # NaN score would silently become an FN
        join_predictions(
            [{"image_path": str(real), "pred": float("nan")}],
            rows=rows, src_root=tmp_path, predict_root=tmp_path,
        )


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
    assert summary["n_real"] == 2 and summary["n_fake"] == 2
    assert summary["fpr"] == 1.0 and summary["fnr"] == 0.5
    assert summary["by_generator"]["sd14"] == {
        "n_images": 2, "n_real": 1, "n_fake": 1, "n_fp": 1, "n_fn": 1,
        "fpr": 1.0, "fnr": 1.0,
    }
    assert summary["by_condition"]["jpeg_q50"]["n_images"] == 2
    assert [r["image_path"] for r in summary["worst_fp"]] == ["f1"]  # highest pred first
    assert [r["image_path"] for r in summary["worst_fn"]] == ["m1"]  # lowest pred first

    out = tmp_path / "stats.csv"
    write_stats_csv(out, summary)
    with out.open(newline="", encoding="utf-8") as fh:
        table = list(csv.reader(fh))
    assert table[0] == [
        "group_type", "group_value", "n_images", "n_real", "n_fake",
        "n_fp", "n_fn", "fpr", "fnr",
    ]
    assert ["overall", "all", "4", "2", "2", "2", "1", "1.0", "0.5"] in table
    assert ["generator", "sd14", "2", "1", "1", "1", "1", "1.0", "1.0"] in table


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
    assert by_path[str(real / "c.png")]["generator"] == "real"  # label-0, manifest generator null

    summary = json.loads((out_dir / "badcase_summary.json").read_text(encoding="utf-8"))
    assert summary["n_images"] == 4 and summary["n_fp"] == 1 and summary["n_fn"] == 1
    assert summary["metrics"]["auroc"] == 0.75  # 3/4 positive-negative pairs ranked right
    assert summary["by_generator"]["sd14"]["n_fp"] == 0
    assert summary["by_generator"]["real"]["n_images"] == 2  # c.png + d.png
    assert (out_dir / "badcase_stats.csv").is_file()
    # b.png and d.png have no manifest rows; the count is surfaced in the CLI line
    assert "unmatched_metadata=2" in r.stdout


def test_manifest_reader_validates(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"path": "x.png"}\nnot json\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        load_manifest_rows([bad])
    nonobj = tmp_path / "nonobj.jsonl"
    nonobj.write_text('["path"]\n', encoding="utf-8")
    with pytest.raises(SystemExit):  # non-object row would TypeError on m["path"]
        load_manifest_rows([nonobj])
    empty = tmp_path / "empty.jsonl"
    empty.write_text('{"label": 1}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        load_manifest_rows([empty])


def test_gallery_script_smoke(tmp_path):
    """badcase_gallery.py renders a self-contained HTML with embedded thumbs."""
    root = tmp_path / "val"
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    preds = []
    for rel, val in [
        ("real/a.png", 0.9),   # FP
        ("real/b.png", 0.95),  # FP (worst)
        ("fake/c.png", 0.2),   # FN
        ("fake/d.png", 0.8),   # TP
    ]:
        p = root / rel
        make_image(seed=len(rel)).save(p)
        preds.append({"image_path": str(p), "pred": val})
    pred_json = tmp_path / "pred.json"
    pred_json.write_text(json.dumps(preds), encoding="utf-8")

    out = tmp_path / "gallery.html"
    r = subprocess.run(
        [sys.executable, "scripts/badcase_gallery.py", "--pred", str(pred_json),
         "--image-dir", str(root), "--out", str(out), "--max-per-type", "10",
         "--thumb", "64", "--title", "smoke"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    html_text = out.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64" in html_text
    assert html_text.count('class="badge">FP') == 2
    assert html_text.count('class="badge">FN') == 1
    # worst first: FP sorted by pred desc
    assert html_text.find("0.950") < html_text.find("0.900")
# end
