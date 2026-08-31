# 2026-08-31, tianqi, spec-compare after D3 dualbranch + fuse 400
# Spec-compare eval (as of 2026-08-31)

Headline: **0.50×AUC_clean + 0.50×AUC_robust**. Acc@0.5 is not the score.
400 = balanced official-val screen (200/200). Train size is still ~140k unless noted.

Open the canvas beside chat: spec-compare-eval.

D3 mix is **frozen CLIP-B linear + online aug** (same protocol as `clipb16_linear_sid_aug`). D3 dualbranch is the same mix plus an RGB + highpass CNN tower. Fuse is **mean logit of CLIP-B last4 + D3 mix at infer** (no retrain).

CLIP-L SID-aug **0.995 is EvalGEN clean AUROC**, not official formula. Official is 0.976 (400) / 0.977 (full). Train is full SID ~140k.

## Iteration

1. CIFAKE CLIP-B ~0.50 on official val → do not pick the backbone from CIFAKE.
2. SID 14万 + online aug: CLIP-B 0.970 / CLIP-L 0.976 (400). Domain, not architecture.
3. 8k probes (D1 0.734, C-Flow 0.654, C-Pixel 0.649, full val): quantity then generator family. Pixel-only never fires.
4. CLIP-B unfreeze4: **0.990** (400) / **0.989** (full 13843). SID dualbranch 0.966, consistency 0.965, res336 0.969 — do not stack those.
5. CLIP-L unfreeze4 0.980 and unfreeze4+336 0.977 both lose to CLIP-B unfreeze4. SID-val Acc is saturated and misleading.
6. D3 (SID 14万, replace 9622 FLUX with self-built + SDXL UNet + ADM/DDPM): **0.978** official 400, best EvalGEN Infinity/NOVA.
7. D3 dualbranch (same mix, frozen CLIP-B RGB + highpass CNN): **0.983** official 400, EvalGEN 0.995. Beats frozen D3, still loses to last4 and fuse.
8. Fuse last4 + D3: **0.993** official 400, EvalGEN 0.997. Best dual-metric if two ckpts are allowed.
9. D3 + last4 **trained together**: official 400 **0.976** (drop vs last4). EvalGEN 400/gen×15 formula **0.9955** is the best unseen-robust screen so far — still do not submit it; contest score is DALL·E.


## Ranking (400 formula unless n=13843)

`unfreeze4` = **CLIP-B/16**, last 4 of 12 encoder blocks. `CLIP-L unfreeze4` = **CLIP-L/14**, last 4 of 24 blocks. Same SID 14万 + online aug.

| Model | Train | Formula | DALL·E AUC | Acc@0.5 |
|---|---|---:|---:|---:|
| fuse last4+D3 | mean logit, no retrain | **0.993** | 0.995 | 0.888 |
| CLIP-B unfreeze4 | SID 14万, last 4 of 12 | **0.990** | 0.991 | 0.848 |
| D3 dualbranch | D3 mix + RGB + highpass CNN | **0.983** | 0.988 | **0.948** |
| CLIP-L unfreeze4 | SID 14万, last 4 of 24 | 0.980 | 0.981 | 0.883 |
| D3 mix | frozen CLIP-B, SID + 9622 mix-in | **0.978** | 0.985 | **0.940** |
| CLIP-B unfreeze4@336 | CLIP-B last-4 @ 336 | 0.977 | 0.981 | 0.823 |
| D3 + last4 train | D3 mix + unfreeze last 4 | 0.976 | 0.984 | — |
| CLIP-L SID-aug | frozen CLIP-L | 0.976 | 0.976 | 0.885 |
| CLIP-B first-4 unfreeze | encoder blocks 0–3 | 0.974 | 0.974 | — |
| CLIP-B SID-aug | frozen CLIP-B | 0.970 | 0.969 | 0.900 |
| DINOv2 CIFAKE | CIFAKE | 0.786 | 0.779 | 0.623 |
| D1 8k | 8k FLUX, no aug | 0.734 full | 0.737 | 0.676 |
| C-Pixel | 1k ADM+DDPM | 0.649 full | 0.649 | 0.362 |
| CIFAKE CLIP-B | CIFAKE | 0.569 | 0.561 | 0.498 |

Full-val check: CLIP-B SID 0.966, CLIP-L SID 0.977, unfreeze4 0.989, res336 0.969.

## 15-condition AUROC (official val)

Same 15 official transforms as the spec grid. n=400 except D1 / C-pixel (full 13843).

| model | clean | q90 | q70 | q50 | q30 | b0.5 | b1 | b2 | r×0.5 | r×0.25 | n0.02 | n0.05 | n0.10 | jitter | crop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fuse last4+D3 | 0.995 | 0.995 | 0.992 | 0.988 | 0.984 | 0.996 | 0.996 | 0.989 | 0.994 | 0.984 | 0.992 | 0.989 | 0.989 | 0.994 | 0.993 |
| CLIP-B unfreeze4 | 0.991 | 0.991 | 0.988 | 0.987 | 0.983 | 0.992 | 0.993 | 0.985 | 0.991 | 0.980 | 0.991 | 0.987 | 0.988 | 0.990 | 0.991 |
| D3 dualbranch | 0.988 | 0.988 | 0.984 | 0.969 | 0.963 | 0.989 | 0.989 | 0.974 | 0.987 | 0.958 | 0.978 | 0.975 | 0.976 | 0.985 | 0.977 |
| CLIP-L unfreeze4 | 0.981 | 0.982 | 0.981 | 0.973 | 0.959 | 0.984 | 0.988 | 0.985 | 0.989 | 0.982 | 0.974 | 0.975 | 0.977 | 0.981 | 0.973 |
| D3_mix | 0.985 | 0.984 | 0.978 | 0.954 | 0.944 | 0.985 | 0.986 | 0.973 | 0.984 | 0.958 | 0.974 | 0.969 | 0.967 | 0.980 | 0.971 |
| clipl14_sid | 0.976 | 0.973 | 0.976 | 0.972 | 0.964 | 0.981 | 0.987 | 0.983 | 0.986 | 0.980 | 0.971 | 0.977 | 0.982 | 0.974 | 0.963 |
| unfreeze4_res336 | 0.981 | 0.982 | 0.981 | 0.980 | 0.973 | 0.983 | 0.979 | 0.964 | 0.975 | 0.962 | 0.970 | 0.968 | 0.960 | 0.977 | 0.959 |
| clipb16_sid | 0.969 | 0.971 | 0.968 | 0.968 | 0.964 | 0.972 | 0.978 | 0.978 | 0.979 | 0.973 | 0.961 | 0.971 | 0.976 | 0.963 | 0.955 |
| dinov2_clean | 0.779 | 0.784 | 0.778 | 0.780 | 0.794 | 0.784 | 0.792 | 0.813 | 0.794 | 0.804 | 0.797 | 0.808 | 0.817 | 0.778 | 0.770 |
| clipb16_clean | 0.561 | 0.572 | 0.582 | 0.619 | 0.654 | 0.571 | 0.567 | 0.520 | 0.552 | 0.528 | 0.583 | 0.588 | 0.613 | 0.557 | 0.587 |
| D1 (full) | 0.737 | 0.752 | 0.758 | 0.747 | 0.749 | 0.737 | 0.719 | 0.736 | 0.710 | 0.722 | 0.764 | 0.706 | 0.689 | 0.726 | 0.722 |
| C_pixel (full) | 0.649 | 0.607 | 0.588 | 0.627 | 0.572 | 0.658 | 0.700 | 0.714 | 0.706 | 0.701 | 0.533 | 0.643 | 0.679 | 0.653 | 0.692 |

## Unseen AUROC (DALL·E + EvalGEN full ~11k/gen)

Train never sees DALL·E or EvalGEN. **NOVA** is the stress-test column (hardest, biggest spread). GoT / OmniGen are ~0.999 for every strong model — do not use them as the unseen proxy. Flux is SID-family, weakest unseen.

| Model | DALL·E | EvalGEN | NOVA | Infinity | GoT | OmniGen | Flux |
|---|---:|---:|---:|---:|---:|---:|---:|
| fuse last4+D3 | **0.995** | **0.997** | 0.988 | 0.995 | 1.000 | 1.000 | 1.000 |
| CLIP-B unfreeze4 | **0.991** | 0.989 | 0.963 | 0.983 | 1.000 | 1.000 | 1.000 |
| D3 dualbranch | 0.988 | 0.995 | 0.987 | 0.994 | 0.999 | 0.997 | 0.999 |
| CLIP-L unfreeze4 | 0.981 | 0.991 | 0.962 | 0.993 | 1.000 | 1.000 | 1.000 |
| D3 mix (frozen CLIP-B) | 0.985 | **0.995** | **0.988** | **0.994** | 0.999 | 0.997 | 0.999 |
| D3 + last4 train | 0.984 | 0.996 | 0.992 | 0.991 | 1.000 | 0.999 | 0.999 |
| CLIP-L SID-aug | 0.976 | **0.995** | 0.981 | 0.998 | 1.000 | 0.999 | 1.000 |
| CLIP-B SID-aug | 0.969 | 0.992 | 0.970 | 0.992 | 1.000 | 0.999 | 0.999 |
| CLIP-B unfreeze4@336 | 0.981 | 0.954 | 0.868 | 0.904 | 0.999 | 0.998 | 1.000 |

## EvalGEN full (clean, ~11k fakes/gen)

| Model | AUROC | Infinity rec | NOVA rec |
|---|---:|---:|---:|
| fuse last4+D3 | **0.997** | 0.672 | 0.557 |
| D3 + last4 train | 0.996 | 0.606 | 0.578 |
| D3 mix | **0.995** | **0.931** | **0.857** |
| D3 dualbranch | 0.995 | 0.916 | 0.832 |
| CLIP-L SID | 0.995 | 0.905 | 0.649 |
| CLIP-B SID | 0.992 | 0.868 | 0.711 |
| CLIP-B unfreeze4 | 0.989 | 0.570 | 0.487 |
| unfreeze4+336 | 0.954 | 0.289 | 0.267 |

## EvalGEN 400/gen × 15-cond (unseen robust screen)

Same 14 official keys + clean. n=4000 (400 fakes/gen × 5 gens + 2000 SID-val reals). **Not the contest score.** CLIP-L SID and fuse still running on GPU1 (then phase2 full-n). D2 waits behind this job.

| Model | formula | clean AUROC | robust |
|---|---:|---:|---:|
| D3 + last4 train | **0.9955** | 0.9957 | 0.9953 |
| D3 mix | 0.9944 | 0.9953 | 0.9936 |
| CLIP-B SID-aug | 0.9913 | 0.9915 | 0.9910 |
| CLIP-B unfreeze4 | 0.9887 | 0.9889 | 0.9885 |

D3+last4 wins this unseen-robust screen and loses official DALL·E (0.976 vs last4 0.990). Keep last4 / fuse as submit.

## Bad cases

Local HTML: `outputs/tables/badcase_galleries/index.html`. **Download and open locally** — Vast Jupyter / file browser will not render the galleries.

400-subset @0.5 (same 200/200): fuse 1 FP / 44 FN; last4 1/60; D3 mix 4/20; D3 dualbranch 2/19 (metrics, no extra gallery). Pairwise HTML: `outputs/tables/badcase_compare/`.

How to share: [docs/badcase-galleries.md](../../../docs/badcase-galleries.md).

Full official_val: CLIP-B SID 150 FP / 1865 FN; CLIP-L 80 / 2173; D1 1730 / 2759; C-Pixel 1 / 8838.

# end
