# Content Pattern Analysis

Evaluation-only pipeline for finding image properties associated with model
false positives and false negatives. It supports the competition's feature
engineering, evaluation design, error analysis, and explainability scope.

This tool reports **correlations**, not causal AIGC fingerprints. Never copy
official-val or EvalGEN images into training, perform hard-negative mining on
them, or tune repeatedly against individual hold-out examples.

## What it measures

Low-level features are always available:

| Feature | Measurement / slice |
|---|---|
| aspect | portrait / square / landscape |
| resolution | tiny / small / medium / large, based on longest original side |
| format | decoded source format |
| brightness | mean luminance |
| saturation | mean HSV-style channel spread |
| sharpness | mean absolute luminance gradient |
| edge density | fraction of luminance gradients above 0.12 |
| entropy | 8-bit luminance histogram entropy |
| black border | fraction of near-black pixels in the outer 4% frame |

`--semantic` additionally runs local CLIP zero-shot slicing:

- style: photoreal / illustration / anime / painting
- scene: indoor / outdoor / studio product / screenshot or UI
- subject: people / animal / food / architecture / landscape / other object
- composition: close-up / wide scene / text-heavy / abstract

CLIP labels are forced-choice hypotheses. The per-image output keeps the top
similarity and top-1/top-2 margin so low-confidence labels can be audited.

## Official demonstration validation

First produce predictions using the model being analyzed, then run:

```bash
DATA_ROOT=/workspace/data python scripts/analyze_content_patterns.py \
  --split official_val \
  --preds /workspace/experiments/<experiment>/official_val/pred_clean.json \
  --out-dir /workspace/experiments/<experiment>/content_patterns/official_val
```

Add semantic slices when the local CLIP model is available:

```bash
DATA_ROOT=/workspace/data MODELS_ROOT=/workspace/models \
python scripts/analyze_content_patterns.py \
  --split official_val \
  --preds /workspace/experiments/<experiment>/official_val/pred_clean.json \
  --semantic --batch-size 32 \
  --out-dir /workspace/experiments/<experiment>/content_patterns/official_val
```

The prediction JSON must contain `image_path` and `pred`. If it also contains
`y` or `label`, that label is used; otherwise official-val labels are joined
from `DATA_ROOT/val/{real,fake}`. For transformed predictions, pass the matching
name such as `--condition jpeg_q50`; clean is the default.

## EvalGEN

EvalGEN is an **additional hold-out**, not the official competition test set.
It contains unseen-generator fakes (Flux, GoT, Infinity, OmniGen, Nova).
Predictions from `run_full_eval.py` include paired hold-out reals and `y`, which
allows AUC and per-generator slices:

```bash
DATA_ROOT=/workspace/data MODELS_ROOT=/workspace/models \
python scripts/analyze_content_patterns.py \
  --split evalgen \
  --preds /workspace/experiments/<experiment>/evalgen/pred_clean.json \
  --semantic \
  --out-dir /workspace/experiments/<experiment>/content_patterns/evalgen
```

EvalGEN never enters training. Use it to check whether a candidate pattern
persists across unseen generators; do not tailor training to its individual
images.

## Outputs

| File | Purpose | Commit? |
|---|---|---|
| `feature_cache.jsonl` | reusable low-level and CLIP features | no |
| `image_features.jsonl` | predictions joined with per-image features | no |
| `slice_metrics.csv/json/md` | pattern-level AUC, FPR, FNR and prevalence | small summary only if needed |
| `group_metrics.json` | overall, condition and generator metrics | small summary only if needed |
| `pattern_report.md` | ranked shortcut/blind-spot candidates | yes, after human review |
| `pattern_gallery.html` | representative high-confidence FP/FN | no |

The cache records the extraction resolution and semantic model path. Changing
either causes the relevant features to be recomputed.

## Candidate rules

A slice becomes a **shortcut candidate** when the difference between its fake
and real prevalence is at least 0.15. It becomes a **blind-spot candidate**
when FPR or FNR is at least 0.10 above the overall rate, or when elevated FNR
repeats across at least two generators with sufficient support.

These thresholds rank review work; they do not prove causality. Before adding
a finding to the final Error Analysis Note:

1. inspect its representative FP/FN gallery;
2. confirm adequate sample support;
3. check whether the trend repeats across generators;
4. check whether it survives relevant JPEG/blur/resize/noise/crop conditions;
5. distinguish a useful forensic cue from a dataset shortcut.

## Fast smoke run

For pipeline validation without CLIP:

```bash
python scripts/analyze_content_patterns.py \
  --split official_val --preds <pred.json> \
  --max-images 100 --min-support 5 \
  --out-dir /tmp/content-pattern-smoke
```

`--max-images` samples deterministically using `--seed`. Omit it for the final
report.
