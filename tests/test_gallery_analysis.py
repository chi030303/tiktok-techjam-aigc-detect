import json
from pathlib import Path

import pytest

from src.eval.gallery_analysis import (
    build_report,
    image_key,
    load_gallery,
    render_markdown,
)


def _write_gallery(
    path: Path,
    fp: list[tuple[str, float]],
    fn: list[tuple[str, float]],
    joined: int = 6,
    threshold: float = 0.5,
) -> None:
    cards = {
        "FP": [
            {"etype": "FP", "path": image_path, "pred": pred, "label": 0}
            for image_path, pred in fp
        ],
        "FN": [
            {"etype": "FN", "path": image_path, "pred": pred, "label": 1}
            for image_path, pred in fn
        ],
    }
    path.write_text(
        "<p class=\"meta\">"
        f"threshold={threshold} · FP={len(fp)} · FN={len(fn)} · "
        f"joined={joined} · unmatched_labels=0"
        "</p>\n"
        f"<script>const CARDS = {json.dumps(cards)};\n"
        "let kind = \"FP\";</script>\n",
        encoding="utf-8",
    )


def test_load_summarize_overlap_and_render(tmp_path):
    left_path = tmp_path / "left.html"
    right_path = tmp_path / "right.html"
    _write_gallery(
        left_path,
        fp=[("/a/real/shared.jpg", 0.9), ("/a/real/left.jpg", 0.6)],
        fn=[("/a/fake/shared.jpg", 0.1)],
    )
    _write_gallery(
        right_path,
        fp=[("/different/root/real/shared.jpg", 0.8)],
        fn=[
            ("/different/root/fake/shared.jpg", 0.2),
            ("/different/root/fake/right.jpg", 0.4),
        ],
    )

    report = build_report(
        [load_gallery("left", left_path), load_gallery("right", right_path)],
        n_real=3,
        n_fake=3,
        aurocs={"left": 0.8, "right": 0.9},
    )

    left = report["models"][0]
    assert left["fp"] == 2 and left["fn"] == 1
    assert left["fpr"] == pytest.approx(2 / 3)
    assert left["fnr"] == pytest.approx(1 / 3)
    assert left["accuracy"] == pytest.approx(0.5)
    assert left["score_distribution"]["FP"]["median"] == pytest.approx(0.75)

    overlaps = {
        row["error_type"]: row for row in report["overlaps"]
    }
    assert overlaps["FP"]["intersection"] == 1
    assert overlaps["FP"]["union"] == 2
    assert overlaps["FN"]["intersection"] == 1
    assert overlaps["FN"]["union"] == 2

    markdown = render_markdown(report, "Test Note", "test models")
    assert "# Test Note" in markdown
    assert "| left | 0.8000 | 2 | 66.67% | 1 | 33.33% | 50.00% |" in markdown
    assert "cannot support claims about JPEG" in markdown


def test_load_gallery_rejects_inconsistent_metadata(tmp_path):
    path = tmp_path / "bad.html"
    _write_gallery(path, fp=[("/real/a.jpg", 0.8)], fn=[])
    text = path.read_text(encoding="utf-8").replace("FP=1", "FP=2")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="metadata says 2 FP"):
        load_gallery("bad", path)


def test_report_rejects_wrong_denominator_and_threshold(tmp_path):
    left_path = tmp_path / "left.html"
    right_path = tmp_path / "right.html"
    _write_gallery(left_path, fp=[], fn=[], threshold=0.5)
    _write_gallery(right_path, fp=[], fn=[], threshold=0.6)
    left = load_gallery("left", left_path)
    right = load_gallery("right", right_path)

    with pytest.raises(ValueError, match="same threshold"):
        build_report([left, right], n_real=3, n_fake=3)
    with pytest.raises(ValueError, match=r"n_real\+n_fake"):
        build_report([left, load_gallery("right2", left_path)], n_real=2, n_fake=3)


def test_image_key_is_root_independent():
    assert image_key("/workspace/data/val/fake/a/b.jpg") == "fake/a/b.jpg"
    assert image_key("/local/copy/fake/a/b.jpg") == "fake/a/b.jpg"
