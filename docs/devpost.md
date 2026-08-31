# 2026-08-31, tianqi, Devpost paste (English). Numbers = official val 400 unless noted
# Devpost — Written Project Description

**Project name (suggestion):** Robust AIGC Image Detection with CLIP-B Last-4 and Generator Mix

**Repo:** https://github.com/chi030303/tiktok-techjam-aigc-detect  
**Demo video:** *(paste YouTube URL after upload)*  
**Robustness table:** `docs/robustness.md` (clean vs 14-transform figure in `docs/robustness/`)  
**Error analysis:** `docs/error_analysis.md`

---

## How the solution addresses the problem

Social images are compressed, resized, blurred, and re-encoded before they reach a detector. The contest score is **0.50×AUC_clean + 0.50×AUC_robust** over 14 official transforms (JPEG, blur, resize, noise, color jitter, center crop) — not accuracy at 0.5.

We treat this as a **domain + generator** problem, not a bigger-backbone problem:

1. **Train on social-domain photos**, not CIFAKE. CIFAKE CLIP-B scored ~0.50 on official val; **SID_Set (~140k)** with the same official transforms as **online augmentation** jumped CLIP-B to **0.970**.
2. **Unfreeze the last 4 CLIP-B/16 vision blocks** (not the first 4, not CLIP-L last-4). Last-4 reaches **0.990** (400-subset) / **0.989** (full official val, 13,843 images). First-4 is **0.974**. CLIP-L last-4 is **0.980**.
3. **Mix in non-FLUX generators** (D3: self-built Flux.2 / SD3.5 / Nano Banana + WildFake SDXL-UNet / ADM / DDPM) so the head is not a FLUX detector. Frozen D3 is **0.978** official but much stronger on held-out **EvalGEN Nova**.
4. If two checkpoints are allowed, **average logits** of last-4 and D3 at inference (no retrain). Fuse is **0.993** official and **0.997** EvalGEN. Training last-4 *on* D3 **drops** official to **0.976** — do not stack that way.

The model is **≪ 2B** (CLIP-B/16 ≈ 86M). Official demo val (`data/val`, COCO + DALL·E Advanced) and EvalGEN are **never used for training**. `predict.py` takes an image directory and writes JSON `{image_path, pred}` with `pred` = P(AIGC).

---

## Development tools

- **VS Code / Cursor** for code and docs
- **GitHub** for review (`scripts/check.sh` before merge)
- **Vast.ai** dual RTX 4090 for training and 15-condition eval (not Colab)
- Local HTML bad-case galleries (no Streamlit)

---

## Models or APIs

| Role | Model | Notes |
|---|---|---|
| Submit (single) | OpenAI **CLIP ViT-B/16** | Last 4 of 12 vision blocks unfrozen + linear head |
| Optional fuse | Same CLIP-B frozen linear **D3 mix** | Mean logit with last-4 |
| Ablations (not submit) | CLIP-L/14, ResNet-50, DINOv2 ViT-L/14 | CIFAKE / SID probes |

No closed generation API at inference. Training used public checkpoints from Hugging Face (`openai/clip-vit-base-patch16`). Self-built fakes were generated offline (Flux.2, SD3.5, PixArt, SDXL, Nano Banana, GPT-image) and stored as files; they are **not** called at test time.

---

## Libraries and frameworks

- **PyTorch**, **torchvision**
- **Hugging Face Transformers** (CLIP vision encoder)
- **pandas** / CSV+Markdown tables (`outputs/tables/`)
- **Pillow**, **NumPy**, **PyYAML**, **pyarrow** (SID parquet)
- **pytest** for CI (`scripts/check.sh`)

---

## Datasets and assets

**Train (allowed):**

- [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) — main 140k social real/fake (FLUX-heavy)
- Self-built t2i (Flux.2, SD3.5, Nano Banana, GPT-image, PixArt, SDXL) and a thin WildFake UNet/ADM/DDPM mix-in (D3)
- Optional WildFake slices **after** dropping overlap with `data/val`

**Hold-out (never train):**

- Official demonstration set: COCO val2017 reals + DALL·E Advanced fakes → `data/val/`
- EvalGEN: Flux / GoT / Infinity / OmniGen / **Nova** (unseen family)

**Weights:** CLIP-B/16 under `models/clip-vit-base-patch16`. Checkpoints: `experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt` and `experiments/clipb16_linear_sid_d3_mix/ckpts/best.pt`.

---

## One-command inference

```bash
python predict.py /path/to/images out.json \
  --ckpt experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt
# optional fuse:
python predict.py /path/to/images out.json \
  --ckpt experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt \
  --ckpt-b experiments/clipb16_linear_sid_d3_mix/ckpts/best.pt
```
# end
