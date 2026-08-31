# 2026-09-01, tianqi, robustness figure JSON + SVG for deliverable 4
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "robustness" / "official_val400.json"
FIG = ROOT / "docs" / "robustness"


def test_official_val400_snapshot_matches_formula():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    assert len(payload["conditions"]) == 15
    assert payload["conditions"][0] == "clean"
    for name, row in payload["models"].items():
        series = row["auroc"]
        assert len(series) == 15, name
        assert abs(series[0] - row["auc_clean"]) < 1e-6, name
        robust = sum(series[1:]) / 14
        assert abs(robust - row["auc_robust"]) < 0.004, (name, robust, row["auc_robust"])
        formula = 0.5 * row["auc_clean"] + 0.5 * row["auc_robust"]
        assert abs(formula - row["formula"]) < 0.004, (name, formula, row["formula"])


def test_plot_robustness_writes_svg():
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "plot_robustness.py")])
    bars = (FIG / "clean_vs_robust.svg").read_text(encoding="utf-8")
    lines = (FIG / "auroc_15cond.svg").read_text(encoding="utf-8")
    assert "AUC clean" in bars
    assert "0.993" in bars
    assert "J30" in lines
    assert (FIG / "clean_vs_robust.svg").is_file()
    assert (FIG / "auroc_15cond.svg").is_file()
# end
