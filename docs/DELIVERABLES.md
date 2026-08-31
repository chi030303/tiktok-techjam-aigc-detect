# 2026-08-31, tianqi, contest deliverable index (Devpost + GitHub + video + tables)
# Contest deliverables (TikTok TechJam 2026 Challenge 5)

Deadline: **1 Sep 2026 12:00 GMT+8**. Score = **0.50×AUC_clean + 0.50×AUC_robust**. Acc@0.5 is not the score.

| # | Required | Where to copy from |
|---|---|---|
| 1 | Written project description (Devpost) | [devpost.md](devpost.md) — paste into Devpost |
| 2 | Public GitHub + `predict.py` + README | repo README + `predict.py` |
| 3 | Demo video (YouTube public) | [demo_script.md](demo_script.md) — record, then paste the URL into Devpost |
| 4 | Robustness summary | [robustness.md](robustness.md) — also linked from Devpost |
| 5 | Error analysis note | [error_analysis.md](error_analysis.md) |

Self-built data release (not a contest field; **do not publish tonight**): [dataset_release.md](dataset_release.md).

Submit model: **CLIP-B/16 last-4** (single ckpt, formula **0.990**) or **mean-logit fuse last4 + D3** (**0.993**) if two checkpoints are allowed.
# end
