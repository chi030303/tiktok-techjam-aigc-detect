# 2026-08-30, tianqi, official 0.50*AUC_clean + 0.50*AUC_robust (14 spec keys)
"""TechJam formula from a robustness table. Robust = macro mean of the 14 spec keys."""

from __future__ import annotations

from src.transforms.spec import OFFICIAL_SETTINGS

# end

ROBUST_KEYS = tuple(s.key for s in OFFICIAL_SETTINGS)
N_ROBUST = len(ROBUST_KEYS)


def official_formula(table_rows: list[dict]) -> dict:
    by = {str(r.get("condition") or ""): r.get("auroc") for r in table_rows}
    clean = by.get("clean")
    robust_vals = [float(by[k]) for k in ROBUST_KEYS if isinstance(by.get(k), (int, float))]
    robust = sum(robust_vals) / len(robust_vals) if robust_vals else None
    complete = isinstance(clean, (int, float)) and len(robust_vals) == N_ROBUST
    if isinstance(clean, (int, float)) and robust is not None:
        formula = 0.5 * float(clean) + 0.5 * robust
    elif isinstance(clean, (int, float)) and not robust_vals:
        formula = float(clean)
    else:
        formula = None
    return {
        "auc_clean": clean,
        "auc_robust": robust,
        "n_robust": len(robust_vals),
        "n_robust_expected": N_ROBUST,
        "complete": complete,
        "formula": formula,
    }
# end
