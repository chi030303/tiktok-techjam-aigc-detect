# Backbone Ablation Error Analysis Note

<!-- 2026-09-01, tianqi, contest last4/fuse note is docs/error_analysis.md; this file stays frozen-SID gallery stats -->
> **Status:** Frozen CLIP-B/L SID gallery stats (full official val, clean). Contest submit analysis is [error_analysis.md](error_analysis.md) (last-4 / fuse).
> **Model scope:** Frozen CLIP backbone SID ablation; not the last4/fuse submit.
> **Data scope:** official demonstration validation, clean only; 5000 real + 8843 fake; threshold=0.50.
<!-- end -->

## 1. Quantitative summary

| Model | AUROC | FP | FPR | FN | FNR | Acc@threshold |
|---|---:|---:|---:|---:|---:|---:|
| clipb16_sid | 0.9655 | 150 | 3.00% | 1865 | 21.09% | 85.44% |
| clipl14_sid | 0.9766 | 80 | 1.60% | 2173 | 24.57% | 83.72% |

AUROC measures ranking quality; FP/FN and accuracy depend on the chosen threshold. A model can therefore have higher AUROC but lower Acc@0.5 when its score calibration shifts.

## 2. Error-score distribution

| Model | Type | min | p25 | median | p75 | max |
|---|---|---:|---:|---:|---:|---:|
| clipb16_sid | FP | 0.5000 | 0.6450 | 0.7800 | 0.9038 | 0.9980 |
| clipb16_sid | FN | 0.0000 | 0.0180 | 0.0930 | 0.2430 | 0.4980 |
| clipl14_sid | FP | 0.5000 | 0.6230 | 0.7985 | 0.9023 | 0.9970 |
| clipl14_sid | FN | 0.0000 | 0.0140 | 0.0840 | 0.2450 | 0.5000 |

## 3. Cross-model error overlap

| Pair | Type | Shared errors | Union | Jaccard | Share of left | Share of right |
|---|---|---:|---:|---:|---:|---:|
| clipb16_sid ↔ clipl14_sid | FP | 33 | 197 | 0.168 | 22.00% | 41.25% |
| clipb16_sid ↔ clipl14_sid | FN | 1174 | 2864 | 0.410 | 62.95% | 54.03% |

## 4. Evidence-backed findings

- **Lowest false-positive rate:** clipl14_sid (1.60%).
- **Lowest false-negative rate:** clipb16_sid (21.09%).
- The overlap table separates model-specific mistakes from shared hard cases. Shared FN are the strongest candidates for generator-coverage analysis; model-specific errors are candidates for ensembling.
- This clean-only gallery cannot support claims about JPEG, blur, resize, noise, jitter, or crop robustness.

## 5. Manual visual review protocol

Review at least the top 50 highest-confidence FP and top 50 lowest-confidence FN per model. Assign one or more tags:

- `blur_or_low_detail`
- `stylized_or_artwork`
- `regular_geometry`
- `cinematic_lighting`
- `text_or_sign`
- `unusual_composition`
- `photorealistic_fake`
- `unclear`

Report tag counts separately for FP and FN. Until those counts exist, visual patterns must be described as observations, not conclusions.

## 6. Recommended actions

1. Calibrate the decision threshold on an allowed validation split; do not retrain solely to fix Acc@0.5 when AUROC is already strong.
2. Inspect shared high-confidence FN first. They are more likely to represent a stable blind spot than a backbone-specific error.
3. Test whether model-specific errors are reduced by score/logit fusion.
4. Regenerate this report for the final last4 and fuse models, then add 15-condition robustness and EvalGEN generator slices.

## 7. Limitations

- The official demonstration set is evaluation-only and must not be used for hard-negative mining or training.
- The galleries contain only errors, not all prediction scores; calibration curves and optimal thresholds require the original prediction JSON.
- No visual category frequencies are claimed before manual tagging.
