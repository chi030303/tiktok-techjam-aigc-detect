# Devpost submission — copy-paste

Repo (make **public** before submit): https://github.com/chi030303/tiktok-techjam-aigc-detect

Submit weights: [v1.0-submit](https://github.com/chi030303/tiktok-techjam-aigc-detect/releases/tag/v1.0-submit). Self-built mix-in: [Kaggle aigctrace-mix](https://www.kaggle.com/datasets/wwjjames/aigctrace-mix). Do **not** upload COCO / DALL·E / EvalGEN / SID 14万 / `i2i_nano_banana.zip`.

---

## 1. General info

### Project name (≤60)

```
Robust AIGC Detection under Social Transforms
```

(45 characters)

### Elevator pitch (≤200)

```
CLIP-B last-4 (~86M) scores 0.990 on 0.50×clean AUC + 0.50×14 social transforms. A mean-logit fuse with a mixed-generator head reaches 0.993. Official val and EvalGEN never enter training.
```

(188 characters)

---

## 2. Project story → About the project

Paste the whole Markdown block below into the big box. Change `What's next for Untitled` is already using the project name.

```markdown
## Inspiration

Social photos are JPEG-compressed, blurred, resized, and cropped long before a detector sees them. TikTok TechJam Challenge 5 scores **$0.50 \times \mathrm{AUC}_{clean} + 0.50 \times \mathrm{AUC}_{robust}$** over 14 official transforms — not accuracy at 0.5. CIFAKE-trained CLIP-B scored ~0.50 on the official demonstration set. We wanted a detector that still ranks well after those transforms, stays well under the 2B parameter cap, and does not train on the hold-out val.

## What it does

`predict.py` takes a folder of images and writes JSON `{image_path, pred}` with `pred = P(AIGC)`.

- **Single checkpoint:** CLIP ViT-B/16 with the last 4 vision blocks unfrozen. Official val 400 formula **0.990** (full 13,843 **0.989**).
- **Optional two checkpoints:** mean-logit fuse of last-4 and a frozen mixed-generator head → **0.993**.
- Robustness table: all 15 official conditions ([docs/robustness.md](https://github.com/chi030303/tiktok-techjam-aigc-detect/blob/main/docs/robustness.md)). Fuse stays ≥ 0.984 on every key.
- Error analysis and HTML FP/FN galleries (open locally, not Jupyter).

Official demonstration val (COCO reals + DALL·E Advanced) and EvalGEN are **never used for training**.

## How we built it

1. **Domain first.** Train on [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) (~140k social photos, ~70k real + ~70k FLUX) with the official transforms as **online** augmentation. CIFAKE 32×32 does not transfer.
2. **Architecture mix, not more FLUX.** last-4 trains on SID only. Mix-ins replace an equal number of SID FLUX (not stacked). Hunyuan is not used. See the D3–D6 table below.
3. **Stop stacking towers.** CLIP-L last-4 (0.980), ResNet (0.779), DINOv2 (~0.79), resize 336, RGB+frequency, and unfreezing the *first* 4 all lose on the contest formula. Last-4 wins DALL·E ranking; the **D3** mixed head helps unseen **Nova**.
4. **Fuse at inference, do not retrain last-4 on the mix.** Training last-4 *on* the mix dropped official score to 0.976. Fuse last4+D5 / last4+D6 (0.9927 / 0.9929) do not beat last4+D3 (**0.9930**).
5. Stack: PyTorch + Hugging Face Transformers, dual RTX 4090 on Vast.ai, GitHub PRs gated by `scripts/check.sh`. No Streamlit.

SID mix-ins (same protocol: frozen CLIP-B + online aug, ~140k). Official val / EvalGEN never enter train. 400 = 200 COCO + 200 DALL·E.

| Mix | Extra fakes (replace SID FLUX) | Official 400 | EvalGEN clean | Nova AUC | i2i pair_acc |
|---|---|---:|---:|---:|---:|
| **D3** (submit mix head) | WildFake UNet 4k + flux2/sd35 + ADM/DDPM 1k each (~9.6k; 603 nano then) | **0.978** | **0.995** | **0.988** | 0.79 |
| D4 | full nano 1.5k + PixArt 1.5k + SDXL 1.5k + GPT-image 1.5k | 0.973 | 0.989 | 0.963 | — |
| D5 = D3 ∪ D4 | ~15k | 0.975 | 0.995 | 0.986 | 0.79 |
| D6 = D5 + 118 i2i | D5 + Codex/nano reconstructions (no paired reals) | 0.977 | 0.994 | 0.984 (15-cond) | **0.805** |
| fuse last4+D3 | mean logit, no retrain | **0.993** | **0.997** | 0.988 | — |
| fuse last4+D4 / D5 / D6 | same | 0.990 / 0.9927 / 0.9929 | — | — / 0.987 / 0.986 | — |

D4’s extra T2I did not lift Nova (0.963, same as last-4). D5 ≈ D3, not a jump. D6 lifts pair ranking slightly; contest score is unchanged. Submit stays **last-4** or **fuse last4+D3**.

```bash
python predict.py /path/to/images out.json \
  --ckpt experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt
# optional fuse:
python predict.py /path/to/images out.json \
  --ckpt experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt \
  --ckpt-b experiments/clipb16_linear_sid_d3_mix/ckpts/best.pt
```

## Challenges we ran into

- Acc@0.5 lied. Last-4 raised AUROC but looked worse at 0.5 because fake scores sat lower — the contest metric is AUROC.
- A 59-triplet i2i-only probe scored **0.443** on official val (overfit). D6’s 118 i2i fakes on SID mix only moved pair_acc 0.79 → 0.805; contest score stayed below D3.
- False negatives at 0.5 look high (fuse **44/200**), but the gallery cluster is **non-photoreal** DALL·E (comics / anime / illustration), outside a social-photo target. False positives on COCO are rare (**1 FP**). Residual photoreal DALL·E misses still exist; we do not claim FNR vanishes in the wild.
- Official val and EvalGEN had to stay strictly out of train; mixing them in would invalidate the score.

## Accomplishments that we're proud of

- Last-4 **0.990** / fuse **0.993** on the official formula, model **~86M ≪ 2B**.
- Last-4 **≥ 0.980** on every official transform key; fuse **≥ 0.984**.
- Held-out EvalGEN: mixed data lifts Nova recall@0.5 from **0.49 → 0.86** while fuse keeps DALL·E ranking.
- Reproducible `predict.py`, 15-condition figures in-repo, and a local HTML bad-case gallery.

## What we learned

- Domain (SID vs CIFAKE) moved the score more than a larger backbone.
- Complementary heads + mean-logit fuse beat “unfreeze on the mix”.
- Generator coverage (UNet / pixel diffusion in **D3**) mattered for Nova; extra T2I (D4 PixArt / GPT / SDXL / nano) and 118 i2i fakes (D6) did not beat D3 on the contest formula.
- High AUC is not a license to ship threshold 0.5 — pick an FPR budget.

## What's next for Robust AIGC Detection under Social Transforms

Calibrate last-4 so 0.5 matches a stated FPR without changing AUROC. Grow whole-image i2i beyond 59 triplets (D6’s 118 fakes only moved pair_acc 0.79 → 0.805). Add a license-clean Nova-family t2i stand-in that is not EvalGEN. Keep fuse last4+D3 if two files are allowed; do not train last-4 on the mix again.
```

---

## 3. Built with (type these tags, up to 25)

Add one by one:

`python` · `pytorch` · `torchvision` · `huggingface-transformers` · `clip` · `pillow` · `numpy` · `pandas` · `pyyaml` · `pyarrow` · `pytest` · `vast.ai` · `github`

---

## 4. Try it out links

1. **GitHub (required):** `https://github.com/chi030303/tiktok-techjam-aigc-detect`
2. **Robustness write-up:** `https://github.com/chi030303/tiktok-techjam-aigc-detect/blob/main/docs/robustness.md`
3. **SID_Set (train data, already public):** `https://huggingface.co/datasets/saberzl/SID_Set`
4. Tomorrow, if you publish self-built images: paste the Kaggle URL as a fourth link. **Do not** link `data/val` or EvalGEN.

---

## 5. Project media

### Image gallery (3:2 if possible, ≤5 MB each)

Upload from the repo (drag these):

1. `docs/robustness/clean_vs_transforms.png` — clean vs 14 transforms
2. `docs/slides/figures/fuse_arch.png` — last-4 + fuse
3. `docs/slides/figures/eval_full.png` — official val + EvalGEN
4. `docs/slides/figures/arch_table.png` — SID vs mix-in counts
5. `docs/slides/figures/fn_badcases.png` — FN gallery
6. `docs/slides/figures/fp_badcases.png` — FP gallery

Cover/title: first image is the one Devpost shows as the card. Use **clean_vs_transforms.png** or `fuse_arch.png`.

### Video demo link

Leave empty tonight. Tomorrow: record with `docs/slides/TechJam_Challenge5_demo.pptx` + [demo_script.md](demo_script.md) → YouTube **Public** → paste the URL here.

---

## 6. Additional info (judges only, ≤35 MB)

**Do not** upload SID / COCO / EvalGEN / the 88 MB i2i zip.

Zip only small judge files, e.g.:

```bash
cd /Users/kiki/Desktop/LLM/tiktok_techjam
zip -r /tmp/techjam_judge_pack.zip \
  docs/slides/TechJam_Challenge5_demo.pptx \
  docs/robustness.md docs/error_analysis.md docs/demo_script.md \
  docs/robustness/clean_vs_transforms.png
ls -lh /tmp/techjam_judge_pack.zip
```

Upload `/tmp/techjam_judge_pack.zip`. Weights stay in GitHub Releases or the recipe paths in README after the repo is public — they will not fit this 35 MB box.

---

## 7. Tonight vs tomorrow

| Tonight | Tomorrow (before 12:00 GMT+8) |
|---|---|
| Repo **public** | YouTube Public + paste URL |
| Paste name, pitch, story, tags, GitHub links, images | Optional Kaggle for **self-built** mix-in only |
| Judge zip (slides + robustness) | Do not re-host official val / EvalGEN |
| Invite teammates; they **Accept** | Hit Submit if not already; you can edit until noon |

Submit model in the story: **last-4 0.990** (one ckpt) or **fuse 0.993** (two ckpts).
