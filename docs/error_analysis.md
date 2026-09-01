# Error Analysis Note

<!-- 2026-08-31, tianqi, contest note for last4 / fuse (replaces SID DINOv2 0.90 write-up) -->
**Models:** CLIP-B/16 last-4 (`unfreeze4`) and optional **mean-logit fuse** with frozen D3 mix.  
**Score:** 0.50×AUC_clean + 0.50×AUC_robust. Threshold 0.5 is **not** the contest metric.  
**Set:** official demo val, 400 balanced (200 COCO real / 200 DALL·E Advanced) unless noted.  
**Galleries:** download `outputs/tables/badcase_galleries/` and open `index.html` locally ([badcase-galleries.md](badcase-galleries.md)).
<!-- 2026-09-01, tianqi, point contest error note at samily gallery + content-pattern pipelines -->
**Related (evaluation-only):** frozen CLIP-B/L full-val gallery stats — [error_analysis_backbone_ablation.md](error_analysis_backbone_ablation.md); image-property slices — [content-pattern-analysis.md](content-pattern-analysis.md). Reproduce galleries: `python scripts/analyze_badcase_galleries.py`.
<!-- end -->

## 1. What the errors look like

At threshold **0.5** on the same 400 images (clean):

| Model | FP (real → fake) | FN (fake → real) | Acc@0.5 | AUC_clean |
|---|---:|---:|---:|---:|
| **Fuse last4+D3** | **1** | **44** | 0.888 | **0.995** |
| CLIP-B last-4 | 1 | 60 | 0.848 | 0.991 |
| D3 mix | 4 | 20 | 0.940 | 0.985 |
| CLIP-B SID-aug | 3 | 37 | 0.900 | 0.969 |

Both last-4 and fuse are **low false-accusation** systems: almost no COCO photo is called AIGC at 0.5. The cost is **missed DALL·E** (FN). D3 is less conservative (4 FP, 20 FN) and **loses official AUC** relative to last-4.

**Trade-off:** ranking (AUROC) is already very high; 0.5 is a harsh cut on last-4 because scores for fakes sit lower. Downstream should pick the operating point from an FPR budget. `predict.py` emits a continuous `pred`, not a hard label.

## 2. Representative false positives

On the 400-subset, fuse and last-4 each have **one** FP at 0.5. That is the expensive error for a platform (accusing a real user).

- Open `outputs/tables/badcase_galleries/fuse_u4_d3_400_clean.html` and `unfreeze4_400_clean.html`, **FP** section (highest `pred` first).
- Pairwise: `outputs/tables/badcase_compare/unfreeze4_to_fuse_u4_d3.html` — fuse does not create a new FP pile; it mainly **reduces FN** vs last-4 (60 → 44).

Full official val for frozen SID-aug (not the submit) is still FN-heavy: CLIP-B 150 FP / 1865 FN. Last-4/fuse 400-subset is the relevant picture for DALL·E.

## 3. Representative false negatives

<!-- 2026-09-01, tianqi, FN cluster is non-photoreal; residual photoreal DALL·E still exists -->
Last-4 misses 60/200 fakes at 0.5; fuse still misses 44/200. **FPR is the small number** (1/200 reals). Looking at the gallery, a large FN cluster is **non-photoreal DALL·E** (comics / anime / illustration / painterly), often with `pred ≈ 0.001`. Train is SID social photos + FLUX, so those styles are off-target for a social-feed detector. On photoreal social AIGC we expect fewer misses at the same threshold.

This does **not** mean FNR goes to zero in the wild: some FN are still photoreal DALL·E with very low `pred` (the model is sure they are real). Those remain a ranking problem, not a style mismatch.

- Gallery FN section, sorted by **lowest** `pred` first.
- JPEG-30 and resize ×0.25 are the weakest **AUC** keys for fuse (~0.984), still far above CIFAKE (~0.56). Hard JPEG can erase generator traces last-4 uses.
<!-- end -->

## 4. Unseen generators (EvalGEN)

Never trained on EvalGEN. **Nova** is the shared blind spot; Flux/GoT/OmniGen AUROC ≈ 1.0 for strong models and should not be used as the unseen proxy.

| Model | Nova AUC | Nova recall@0.5 | Infinity recall@0.5 |
|---|---:|---:|---:|
| D3 mix | **0.988** | **0.86** | **0.93** |
| fuse last4+D3 | 0.988 | 0.56 | 0.67 |
| CLIP-B last-4 | 0.963 | 0.49 | 0.57 |

**Trade-off:** last-4 wins **DALL·E / contest formula**; D3 wins **Nova recall**; fuse keeps Nova **AUC** and last-4’s official ranking, but at 0.5 it still under-recalls Nova. Mixing last-4 **training** into D3 (one network) **hurt** official DALL·E (0.976). Complementary heads + fuse beats “unfreeze on the mix”.

## 5. What we would not ship

- CIFAKE-only heads (official ~0.50–0.79)
- Pixel/FFT-only probes (C-Pixel ~0.65, never fires on DALL·E)
- First-4 unfreeze (0.974) and CLIP-L last-4 (0.980)
# 2026-09-01, tianqi, D6 also below D3; fuse last4+D6 does not beat D3 fuse
- D4/D5/D6 frozen mix-ins as submit (official **0.973 / 0.975 / 0.977** — all below D3 **0.978**; fuse last4+D6 **0.9929** vs last4+D3 **0.9930**)
# end

## 6. If we had more time

1. Calibrate last-4 so 0.5 matches a stated FPR without changing AUROC.  
2. More whole-image **i2i** (only 59 triplets; D6’s 118 fakes moved pair_acc 0.79 → 0.805, not the contest score) — paired ranking, not Acc@0.5.  
3. Nova-family t2i that is **not** EvalGEN (license-clean stand-in).  
4. Keep fuse if two files are allowed; do not train last-4 on D3 again.

Reproduce the 400 table:

```bash
python scripts/run_full_eval.py --split official_val --conditions full --max-images 400 --seed 0 \
  --ckpt last4=experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt
```
