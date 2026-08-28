# 2026-08-29, tianqi, schema smoke test for predict.py
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_predict_writes_image_path_and_pred(tmp_path: Path) -> None:
    out = tmp_path / "pred.json"
    fixture = ROOT / "fixtures" / "sample_images"
    subprocess.run(
        [sys.executable, str(ROOT / "predict.py"), str(fixture), str(out)],
        check=True,
    )
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows, "expected at least one prediction"
    for row in rows:
        assert "image_path" in row and "pred" in row
        assert isinstance(row["pred"], (int, float))
        assert 0.0 <= float(row["pred"]) <= 1.0
# end
