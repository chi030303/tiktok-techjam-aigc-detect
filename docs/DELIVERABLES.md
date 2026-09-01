# Contest deliverables (TikTok TechJam 2026 Challenge 5)

Deadline: **1 Sep 2026 12:00 GMT+8**. Score = **0.50×AUC_clean + 0.50×AUC_robust**. Acc@0.5 is not the score.

| # | Required | Where to copy from |
|---|---|---|
| 1 | Written project description (Devpost) | [devpost.md](devpost.md) — paste into Devpost |
| 2 | Public GitHub + `predict.py` + README | repo README + `predict.py` |
| 3 | Demo video (YouTube public) | [demo_script.md](demo_script.md) — record with [slides/TechJam_Challenge5_demo.pptx](slides/TechJam_Challenge5_demo.pptx) (6 pages). Full talk: [slides/TechJam_Challenge5.pptx](slides/TechJam_Challenge5.pptx) |
| 4 | Robustness summary | [robustness.md](robustness.md) + figures in [robustness/](robustness/) |
| 5 | Error analysis note | [error_analysis.md](error_analysis.md) (last4/fuse). Frozen SID galleries: [error_analysis_backbone_ablation.md](error_analysis_backbone_ablation.md). Content slices: [content-pattern-analysis.md](content-pattern-analysis.md). |

Self-built mix-in (not a required contest field): [dataset_release.md](dataset_release.md) · [Kaggle aigctrace-mix](https://www.kaggle.com/datasets/wwjjames/aigctrace-mix).

Submit weights: [GitHub release v1.0-submit](https://github.com/chi030303/tiktok-techjam-aigc-detect/releases/tag/v1.0-submit). Do not use `v0.1-model`.

Submit model: **CLIP-B/16 last-4** (single ckpt, formula **0.990**) or **mean-logit fuse last4 + D3** (**0.993**) if two checkpoints are allowed. D4/D5/D6 mix tables: [robustness.md](robustness.md).
