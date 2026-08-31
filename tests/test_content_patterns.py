import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.content_patterns import (
    aggregate_groups,
    aggregate_slices,
    extract_low_level,
    load_feature_cache,
    render_candidate_gallery,
    render_report,
    write_feature_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def _save(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)
    return path


def test_extract_low_level_distinguishes_basic_patterns(tmp_path):
    black = _save(tmp_path / "black.png", np.zeros((40, 40, 3), dtype=np.uint8))
    red = np.zeros((30, 60, 3), dtype=np.uint8)
    red[..., 0] = 255
    red_path = _save(tmp_path / "red.png", red)

    black_features = extract_low_level(black)
    red_features = extract_low_level(red_path)

    assert black_features["aspect"] == "square"
    assert black_features["brightness"] == "dark"
    assert black_features["black_border"] == "black_border"
    assert black_features["entropy"] == "low_entropy"
    assert red_features["aspect"] == "landscape"
    assert red_features["saturation"] == "high_saturation"
    assert red_features["format"] == "png"


def test_feature_cache_round_trip_and_version_filter(tmp_path):
    cache_path = tmp_path / "features.jsonl"
    rows = [
        {"feature_version": 1, "path": "/a.jpg", "aspect": "square"},
        {"feature_version": 0, "path": "/old.jpg", "aspect": "portrait"},
    ]
    write_feature_cache(cache_path, rows)
    loaded = load_feature_cache(cache_path)

    assert set(loaded) == {"/a.jpg"}
    assert loaded["/a.jpg"]["aspect"] == "square"


def _analysis_rows():
    return [
        {
            "image_path": "r1.jpg", "_resolved_path": "r1.jpg", "label": 0,
            "pred": 0.9, "generator": "real", "condition": "clean",
            "aspect": "square", "resolution": "small", "format": "jpg",
            "brightness": "dark", "saturation": "low_saturation",
            "sharpness": "soft", "entropy": "low_entropy",
            "black_border": "no_black_border",
        },
        {
            "image_path": "r2.jpg", "_resolved_path": "r2.jpg", "label": 0,
            "pred": 0.1, "generator": "real", "condition": "clean",
            "aspect": "landscape", "resolution": "small", "format": "jpg",
            "brightness": "bright", "saturation": "mid_saturation",
            "sharpness": "sharp", "entropy": "high_entropy",
            "black_border": "no_black_border",
        },
        {
            "image_path": "f1.jpg", "_resolved_path": "f1.jpg", "label": 1,
            "pred": 0.1, "generator": "nova", "condition": "clean",
            "aspect": "square", "resolution": "small", "format": "jpg",
            "brightness": "dark", "saturation": "low_saturation",
            "sharpness": "soft", "entropy": "low_entropy",
            "black_border": "no_black_border",
        },
        {
            "image_path": "f2.jpg", "_resolved_path": "f2.jpg", "label": 1,
            "pred": 0.9, "generator": "flux", "condition": "jpeg_q50",
            "aspect": "landscape", "resolution": "small", "format": "jpg",
            "brightness": "bright", "saturation": "mid_saturation",
            "sharpness": "sharp", "entropy": "high_entropy",
            "black_border": "no_black_border",
        },
    ]


def test_aggregate_slices_candidates_and_single_class_auc():
    rows, overall = aggregate_slices(
        _analysis_rows(), min_support=1, min_generator_support=1
    )
    by_slice = {(row["feature"], row["value"]): row for row in rows}

    square = by_slice[("aspect", "square")]
    assert square["n"] == 2
    assert square["fpr"] == 1.0
    assert square["fnr"] == 1.0
    assert square["blind_spot_candidate"]
    assert overall["fpr"] == 0.5
    assert overall["fnr"] == 0.5

    generator_rows = aggregate_groups(_analysis_rows(), "generator")
    by_generator = {row["generator"]: row for row in generator_rows}
    assert by_generator["nova"]["n_real"] == 2
    assert by_generator["nova"]["n_fake"] == 1
    assert by_generator["nova"]["auroc"] is not None

    condition_rows = aggregate_groups(_analysis_rows(), "condition")
    jpeg = next(row for row in condition_rows if row["condition"] == "jpeg_q50")
    assert jpeg["auroc"] is None


def test_report_and_gallery_mark_candidates(tmp_path):
    rows, overall = aggregate_slices(
        _analysis_rows(), min_support=1, min_generator_support=1
    )
    report = render_report(rows, overall, "official_val", 0.5)
    gallery = render_candidate_gallery(_analysis_rows(), rows, max_patterns=1)

    assert "Candidate patterns are correlations" in report
    assert "must never feed training" in report
    assert "Content pattern candidates" in gallery
    assert "Human confirmation required" in gallery


def test_cli_smoke_without_clip(tmp_path):
    data = tmp_path / "data"
    real = _save(data / "val" / "real" / "r.png", np.zeros((16, 16, 3)))
    fake_pixels = np.full((16, 16, 3), 255, dtype=np.uint8)
    fake = _save(data / "val" / "fake" / "f.png", fake_pixels)
    predictions = tmp_path / "pred.json"
    predictions.write_text(
        json.dumps(
            [
                {"image_path": str(real), "pred": 0.8, "y": 0},
                {"image_path": str(fake), "pred": 0.2, "y": 1},
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    env = {**os.environ, "DATA_ROOT": str(data)}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_content_patterns.py",
            "--split", "official_val",
            "--preds", str(predictions),
            "--out-dir", str(out),
            "--min-support", "1",
            "--min-generator-support", "1",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out / "image_features.jsonl").is_file()
    assert (out / "slice_metrics.csv").is_file()
    assert (out / "slice_metrics.md").is_file()
    assert (out / "pattern_report.md").is_file()
    assert (out / "pattern_gallery.html").is_file()
    assert "evaluation only" in result.stdout


def test_invalid_max_side_is_rejected(tmp_path):
    image = _save(tmp_path / "image.png", np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="max_side"):
        extract_low_level(image, max_side=0)
