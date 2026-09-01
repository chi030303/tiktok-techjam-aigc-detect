# 2026-08-31, tianqi, contest deliverable index (Devpost + GitHub + video + tables)
# Contest deliverables (TikTok TechJam 2026 Challenge 5)

Deadline: **1 Sep 2026 12:00 GMT+8**. Score = **0.50×AUC_clean + 0.50×AUC_robust**. Acc@0.5 is not the score.

| # | Required | Where to copy from |
|---|---|---|
| 1 | Written project description (Devpost) | [devpost.md](devpost.md) — paste into Devpost |
| 2 | Public GitHub + `predict.py` + README | repo README + `predict.py` |
<!-- 2026-08-31, tianqi, demo video uses the generated 16:9 PPT -->
| 3 | Demo video (YouTube public) | [demo_script.md](demo_script.md) — record with [slides/TechJam_Challenge5_demo.pptx](slides/TechJam_Challenge5_demo.pptx) (6 pages). Full talk: [slides/TechJam_Challenge5.pptx](slides/TechJam_Challenge5.pptx) |
<!-- end -->
<!-- 2026-09-01, tianqi, robustness deliverable includes GitHub figures -->
| 4 | Robustness summary | [robustness.md](robustness.md) + figures in [robustness/](robustness/) |
<!-- end -->
<!-- 2026-09-01, tianqi, error-analysis deliverable links samily gallery + content-pattern notes -->
| 5 | Error analysis note | [error_analysis.md](error_analysis.md) (last4/fuse). Frozen SID galleries: [error_analysis_backbone_ablation.md](error_analysis_backbone_ablation.md). Content slices: [content-pattern-analysis.md](content-pattern-analysis.md). |
<!-- end -->

Self-built data release (not a contest field; **do not publish tonight**): [dataset_release.md](dataset_release.md).

Submit model: **CLIP-B/16 last-4** (single ckpt, formula **0.990**) or **mean-logit fuse last4 + D3** (**0.993**) if two checkpoints are allowed. D4/D5/D6 mix tables: [robustness.md](robustness.md).
# end
