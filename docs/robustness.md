# 2026-08-31, tianqi, deliverable 4: clean vs 14 official transforms (contest formula)
# Robustness Evaluation Summary

Contest score = **0.50×AUC_clean + 0.50×AUC_robust**, where AUC_robust is the mean AUROC over the **14** official keys (not Acc@0.5).

Screen = official demonstration val, **400** images (200 COCO real / 200 DALL·E Advanced), seed 0. Full val (13,843) for last-4 is **0.989**, same ranking.

## Headline

| Model | Formula | AUC_clean | AUC_robust (14) | Acc@0.5 clean |
|---|---:|---:|---:|---:|
| **Fuse last4 + D3** (mean logit) | **0.993** | 0.995 | 0.991 | 0.888 |
| **CLIP-B last-4** (submit, 1 ckpt) | **0.990** | 0.991 | 0.988 | 0.848 |
| D3 mix (frozen CLIP-B) | 0.978 | 0.985 | 0.972 | 0.940 |
| CLIP-B SID-aug (frozen) | 0.970 | 0.969 | 0.970 | 0.900 |
| CIFAKE CLIP-B | 0.569 | 0.561 | — | 0.498 |

## 15-condition AUROC (same 400 images)

| model | clean | JPEG90 | 70 | 50 | 30 | blur0.5 | 1.0 | 2.0 | ×0.5 | ×0.25 | n0.02 | 0.05 | 0.10 | jitter | crop80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fuse last4+D3 | 0.995 | 0.995 | 0.992 | 0.988 | 0.984 | 0.996 | 0.996 | 0.989 | 0.994 | 0.984 | 0.992 | 0.989 | 0.989 | 0.994 | 0.993 |
| CLIP-B last-4 | 0.991 | 0.991 | 0.988 | 0.987 | 0.983 | 0.992 | 0.993 | 0.985 | 0.991 | 0.980 | 0.991 | 0.987 | 0.988 | 0.990 | 0.991 |
| D3 mix | 0.985 | 0.984 | 0.978 | 0.954 | 0.944 | 0.985 | 0.986 | 0.973 | 0.984 | 0.958 | 0.974 | 0.969 | 0.967 | 0.980 | 0.971 |
| CLIP-B SID-aug | 0.969 | 0.971 | 0.968 | 0.968 | 0.964 | 0.972 | 0.978 | 0.978 | 0.979 | 0.973 | 0.961 | 0.971 | 0.976 | 0.963 | 0.955 |

Last-4 stays **≥ 0.980** on every key. Weakest fuse keys are JPEG-30 and resize ×0.25 (~0.984). Frozen D3 is the one that dips on JPEG-30 / tiny resize.

## Held-out generators (not the contest score)

EvalGEN never enters training. Nova is the hard family.

| Model | EvalGEN AUROC | Nova AUROC | Nova recall@0.5 |
|---|---:|---:|---:|
| fuse last4+D3 | **0.997** | 0.988 | 0.56 |
| D3 mix | 0.995 | **0.988** | **0.86** |
| CLIP-B last-4 | 0.989 | 0.963 | 0.49 |

Last-4 is conservative at 0.5 (misses Nova); D3 recovers recall; fuse keeps last-4’s DALL·E ranking and D3’s Nova AUC.

CSV sources: `outputs/tables/official_val400_fuse_u4_d3.csv`, `outputs/tables/compare_spec/README.md`.
# end
