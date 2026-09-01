# TikTok TechJam 2026 — AIGC Image Detection

Image-level detector for real vs AI-generated images (`pred` = P(AIGC)), robust to JPEG, blur, resize, noise, color jitter, and center crop.

**Contest score:** `0.50×AUC_clean + 0.50×AUC_robust`. Acc@0.5 is not the score.  
**Submit:** CLIP-B/16 **last-4** (formula **0.990**) or **fuse last4+D3** (**0.993**) if two checkpoints are allowed.

Contest write-ups: [docs/DELIVERABLES.md](docs/DELIVERABLES.md) · [Devpost text](docs/devpost.md) · [Robustness](docs/robustness.md) (clean vs 14 transforms) · [Error analysis](docs/error_analysis.md) · [Content patterns](docs/content-pattern-analysis.md)

**Do not commit datasets, checkpoints, or API keys.** Weights and images live on disk / Hugging Face / cloud storage. See [docs/data.md](docs/data.md).

## What this repo contains vs what it does not

| In git | Not in git (download locally) |
|---|---|
| Training / eval / `predict.py` scripts | Raw images (`data/`) |
| Small CSV robustness tables + `docs/robustness/` figures | `*.pth` / `*.safetensors` / CLIP weights |
| Docs, SOP, issue templates | Official demo set (COCO val2017 + DALL·E Advanced) |
| `requirements.txt` | `.env`, Vast credentials |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# inference / train also need a CUDA PyTorch (not the CPU wheel):
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
cp .env.example .env   # add HF_TOKEN locally if needed

# smoke: dummy scores, no weights required
python predict.py ./fixtures/sample_images ./outputs/pred.json
bash scripts/check.sh
```

## Reproduce

Two paths: **(A)** download the submitted weights and score, or **(B)** train from datasets then evaluate. Official demonstration val (`data/val`) and EvalGEN **never enter training**.

SID is **not** a folder of JPEGs you dump into ImageFolder. The train loader reads SID **Hugging Face parquet**, keeps `label ∈ {0,1}` (real vs fully synthetic), **drops tampered `label=2`**, and applies the **14 official social transforms as online augmentation** (`p_clean=0.2`).

### Hardware and software

Trained on a Vast.ai box with **2× NVIDIA RTX 4090 (24 GB)**. One job per GPU — we do not data-parallel one CLIP across both cards.

| GPU | Role |
|---|---|
| GPU 1 | Training (last-4 and D3 mix). Recipes set `gpu: "1"`. |
| GPU 0 | Inference and 15-condition eval |

On a **single** GPU, change `gpu: "0"` in the recipe (or the `CUDA_VISIBLE_DEVICES` export below) so the job can see a device.

| Stack | Version we trained / eval’d with |
|---|---|
| OS | Ubuntu (Vast PyTorch image) |
| Python | **3.12.14** |
| PyTorch | **2.11.0+cu128** (`torch.cuda.is_available()` must be true) |
| CUDA | **12.8** (driver on the 4090 box) |
| torchvision | **0.26.0+cu128** |
| transformers | **5.16.1** |
| datasets | **5.0.1** |
| pyarrow | **25.0.1** |
| CLIP-B/16 | `openai/clip-vit-base-patch16` via `scripts/download_backbones.py` |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# CUDA wheel matching the train box (not the CPU default from pip):
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A CUDA 12.4 wheel also runs inference; numbers above are the train/eval box. Copy `.env.example` → `.env`. On a local disk:

```bash
export DATA_ROOT="${DATA_ROOT:-$PWD/data}"
export MODELS_ROOT="${MODELS_ROOT:-$PWD/models}"
export EXP_ROOT="${EXP_ROOT:-$PWD/experiments}"
```

On the train box they default to `/workspace/data`, `/workspace/models`, `/workspace/experiments`.

### A. Run the submitted weights

Do **not** use GitHub release `v0.1-model` — that is a frozen SID DINOv2 probe, not the submit model.

Public checkpoints: [github.com/.../releases/tag/v1.0-submit](https://github.com/chi030303/tiktok-techjam-aigc-detect/releases/tag/v1.0-submit)

| File | What it is | Contest |
|---|---|---|
| `clipb16_last4_unfreeze.pt` (~327 MB) | CLIP-B/16, last 4 vision blocks unfrozen (~86M) | **0.990** (1 ckpt) |
| `clipb16_d3_mix.pt` (~5 KB) | frozen CLIP-B linear head on the D3 mix | fuse 2nd ckpt → **0.993** |

The CLIP-B backbone is **not** inside those files.

```bash
python scripts/download_backbones.py --only clip-vit-base-patch16

mkdir -p checkpoints/submit
curl -L -o checkpoints/submit/clipb16_last4_unfreeze.pt \
  https://github.com/chi030303/tiktok-techjam-aigc-detect/releases/download/v1.0-submit/clipb16_last4_unfreeze.pt
curl -L -o checkpoints/submit/clipb16_d3_mix.pt \
  https://github.com/chi030303/tiktok-techjam-aigc-detect/releases/download/v1.0-submit/clipb16_d3_mix.pt

# 1 ckpt
python predict.py <image_dir> out.json \
  --ckpt checkpoints/submit/clipb16_last4_unfreeze.pt

# 2 ckpts: mean-logit fuse
python predict.py <image_dir> out.json \
  --ckpt checkpoints/submit/clipb16_last4_unfreeze.pt \
  --ckpt-b checkpoints/submit/clipb16_d3_mix.pt
```

### B. Dataset paths

| Asset | URL | Local path | Used by |
|---|---|---|---|
| SID_Set (~140k parquet) | [saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) | `$DATA_ROOT/sid_set` | last-4 and D3 |
| Self-built t2i / i2i | [Kaggle wwjjames/aigctrace-mix](https://www.kaggle.com/datasets/wwjjames/aigctrace-mix) | `$DATA_ROOT/self_built/` | D3 (`flux2`, `sd35`, `nano_banana_vertex_batch`) |
| WildFake UNet / ADM / DDPM | [ModelScope WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary) | `$DATA_ROOT/wildfake/cross_arch/{sd_original,adm,ddpm}` | D3 |
| Official demonstration val | `python scripts/download_official_val.py` | `$DATA_ROOT/val/{real,fake}` | **eval only** |
| EvalGEN | `python scripts/download_evalgen.py` | `$DATA_ROOT/evalgen` | **eval only** |
| CLIP-B/16 | `openai/clip-vit-base-patch16` | `$MODELS_ROOT/clip-vit-base-patch16` | backbone |

```bash
# SID parquet (not a JPEG dump)
huggingface-cli download saberzl/SID_Set --repo-type dataset --local-dir "${DATA_ROOT:-data}/sid_set"

# official hold-out (never train)
python scripts/download_official_val.py
python scripts/download_evalgen.py

# D3 mix-in only (skip if you only retrain last-4)
# Kaggle zip: unzip so you see flux2/, sd35/, nano_banana_vertex_batch/ under self_built/
# GPT/, pixart_sigma_quality_v2/, sdxl_full_refiner_v1/, i2i/ are D4/D6, not D3
python scripts/download_wildfake_subset.py C_unet_sd_original C_pixel_adm C_pixel_ddpm
```

### C. Train last-4 (submit, 1 ckpt) — SID parquet + online aug

Recipe: `experiments/clipb16_linear_sid_unfreeze4/recipe.yaml`. Unfreeze CLIP-B blocks 9–12, `backbone_lr=1e-5`, `head_lr=1e-3`, 3 epochs, batch 64, official online augs. **No Kaggle / WildFake mix-in.**

```bash
export CUDA_VISIBLE_DEVICES=1   # or 0 on a single GPU (also set recipe gpu: "0")
python scripts/run_experiment.py experiments/clipb16_linear_sid_unfreeze4/recipe.yaml --train
# writes $EXP_ROOT/clipb16_linear_sid_unfreeze4/ckpts/best.pt
```

### D. Train D3 mix head (fuse 2nd ckpt) — replace SID FLUX, do not stack

D3 is **not** “SID plus extra files on top”. `replace_sid_fakes: true` drops one SID FLUX fake per mix-in fake (~9.6k: WildFake original-SD UNet ~4k, Kaggle flux2/sd35/nano, ADM 1k, DDPM 1k). Frozen CLIP-B linear + the same online augs. Hunyuan is not used. Official val / EvalGEN stay out.

Rebuild the mixin jsonl on **your** disk (the jsonl stores local paths):

```bash
python scripts/build_d3_mixin.py \
  --self-root "${DATA_ROOT:-data}/self_built" \
  --out "${DATA_ROOT:-data}/manifests/ablation/D3_sid_mixin.jsonl"

export CUDA_VISIBLE_DEVICES=1
python scripts/run_experiment.py experiments/clipb16_linear_sid_d3_mix/recipe.yaml --train
```

Do **not** unfreeze last-4 on this mix (that run scored **0.976** on official val). Fuse at inference instead.

### E. Contest-formula eval

Score = `0.50×AUC_clean + 0.50×AUC_robust` (mean of 14 keys). Acc@0.5 is not the score.

```bash
export CUDA_VISIBLE_DEVICES=0
# last-4
python scripts/run_full_eval.py --split official_val --conditions full --max-images 400 --seed 0 \
  --ckpt last4=$EXP_ROOT/clipb16_linear_sid_unfreeze4/ckpts/best.pt

# mean-logit fuse (exactly two --ckpt plus --fuse)
python scripts/run_full_eval.py --split official_val --conditions full --max-images 400 --seed 0 \
  --ckpt last4=$EXP_ROOT/clipb16_linear_sid_unfreeze4/ckpts/best.pt \
  --ckpt D3=$EXP_ROOT/clipb16_linear_sid_d3_mix/ckpts/best.pt \
  --fuse --fuse-weight 0.5
```

If you skipped training and only downloaded release weights, point `--ckpt` at `checkpoints/submit/clipb16_last4_unfreeze.pt` (and `--ckpt D3=.../clipb16_d3_mix.pt` for fuse).

Expected last-4 on the 400 screen: **0.990** (full 13,843: **0.989**). Fuse last4+D3: **0.993**.

## Transforms (src/transforms/)

Official 6 robustness transforms: **14 frozen eval settings + clean = 15 conditions**. Field tables, ambiguity decisions and seed rules: [docs/transforms.md](docs/transforms.md).

```bash
# folder tree -> source manifest (labels from parent dir names; DO_NOT_TRAIN trees refused as --split train)
python -m src.transforms.build_source --root data/val --dataset demo_wildfake --split val --out data/manifests/source_demo_val.jsonl

# source manifest -> frozen transformed eval set (default splits: val,test,unseen; rerunnable/idempotent)
python -m src.transforms.build --source-manifest data/manifests/source_demo_val.jsonl --out-manifest data/manifests/transforms_eval.jsonl
```

Training-time random augmentation (official grids by default): `from src.transforms.augment import random_augment` → `img, info = random_augment(img, rng, p_clean=0.2)`.

## Download data and models (not in git)

Full rules: [docs/data.md](docs/data.md). Demo set **must not** be used for training.

```bash
# creates data/ and models/ (gitignored except .gitkeep)
bash scripts/download_assets.sh
```

Manual equivalent:

1. **CIFAKE** (smoke / small experiments)  
   [Kaggle: CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) → `data/cifake/`
2. **SID_Set** (preferred train)  
   [Hugging Face: saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) → `data/sid_set/`
3. **WildFake D3 slices** (mix-in only; do not snapshot the full 3.5M set)  
   `python scripts/download_wildfake_subset.py C_unet_sd_original C_pixel_adm C_pixel_ddpm` → `$DATA_ROOT/wildfake/cross_arch/`
4. **Self-built mix-in** (D3 / D4 / D6; not required for last-4)  
   [Kaggle: wwjjames/aigctrace-mix](https://www.kaggle.com/datasets/wwjjames/aigctrace-mix) → `$DATA_ROOT/self_built/` so that `flux2/`, `sd35/`, `nano_banana_vertex_batch/` sit directly under that folder.
5. **Demo / reference only (no train)**  
   `python scripts/download_official_val.py` → `data/val/real` (COCO val2017) + `data/val/fake` (DALL·E Advanced 8843).  
   This **is** the official TechJam demonstration set. It does not count toward the score. Do not train on it.
6. **EvalGEN** (extra hold-out, also no train)  
   `python scripts/download_evalgen.py` → `data/evalgen/`
7. **Backbones** (all ≪ 2B)  
   `python scripts/download_backbones.py` — CLIP-B/16, CLIP-L/14, ResNet-50, DINOv2 ViT-L/14 → `models/`
8. **Submit checkpoints** (optional if you train yourself)  
   [GitHub release v1.0-submit](https://github.com/chi030303/tiktok-techjam-aigc-detect/releases/tag/v1.0-submit)

On Vast, images and weights live in **`/workspace/data` and `/workspace/models`**, not inside a personal clone. Experiment recipes live in git under `experiments/<name>/recipe.yaml`; run artifacts go to `/workspace/experiments/<name>/`. See [experiments/README.md](experiments/README.md).

Official inference contract: a directory of images in, JSON out:

```json
[{"image_path": "data/val/fake/0001.jpg", "pred": 0.87}]
```

`pred` = confidence the image is AIGC.

## Evaluate (robustness table)

Uses the same `predict.py` contract. Demo val / EvalGEN stay out of training.

```bash
# daily compact table: clean vs JPEG-50 vs center-crop 80%
python scripts/run_eval.py robustness --split official_val --conditions daily --max-images 400

# score an existing JSON
python scripts/run_eval.py score --pred ./outputs/pred.json --split official_val
```

Tables land in `outputs/tables/` (csv / md / json). Full transform list and hold-out rules: [docs/data.md](docs/data.md).

## Results

**Single ckpt: CLIP-B/16 last-4** — SID ~140k + official online aug, unfreeze last 4 vision blocks. Official val 400 formula **0.990** (full 13,843 **0.989**).  
**Two ckpts:** mean-logit fuse of last-4 + **D3** mix = **0.9930**. last4+D5 = 0.9927, last4+D6 = 0.9929 — complementary heads, not “more mix-in always wins.”

| model | formula 400 | DALL·E AUC | Acc@0.5 | EvalGEN AUROC |
|---|---:|---:|---:|---:|
| **fuse last4+D3** | **0.9930** | 0.995 | 0.888 | **0.997** |
| fuse last4+D6 | 0.9929 | 0.995 | — | — |
| fuse last4+D5 | 0.9927 | 0.995 | — | — |
| fuse last4+D4 | 0.990 | — | — | — |
| CLIP-B last-4 | **0.990** | 0.991 | 0.848 | 0.989 |
| D3 dualbranch | 0.983 | 0.988 | 0.948 | 0.995 |
| D3 mix (frozen CLIP-B) | 0.978 | 0.985 | 0.940 | 0.995 |
| D6 mix | 0.977 | 0.984 | — | 0.994 |
| D5 mix | 0.975 | 0.982 | — | 0.995 |
| D4 mix | 0.973 | — | — | 0.989 |
| CLIP-L SID-aug | 0.976 | 0.976 | 0.885 | 0.995 |
| CLIP-B SID-aug | 0.970 | 0.969 | 0.900 | 0.992 |
| SID DINOv2 frozen | 0.900 | 0.904 | — | 0.964 |

**Mix-in contents** (replace equal SID FLUX; Hunyuan never used):

| Mix | What was added | Why it exists | Contest takeaway |
|---|---|---|---|
| D3 | WildFake UNet ~4k + flux2/sd35 + ADM/DDPM 1k each (~9.6k) | architecture holes (UNet / pixel), not more FLUX | **submit mix head** — lifts Nova 0.963 → 0.988 |
| D4 | nano + PixArt + SDXL + GPT (~6k) | extra T2I families | official **0.973**, Nova still **0.963** — do not submit |
| D5 | D3 ∪ D4 (~15k) | union probe | 0.975 ≈ D3, not a jump |
| D6 | D5 + 118 whole-image i2i fakes | pair ranking | pair_acc 0.79 → **0.805**; official 0.977, fuse 0.9929 |

15-condition table, Nova split, and the full D3–D6 table: [docs/robustness.md](docs/robustness.md). Full grid: [outputs/tables/compare_spec/README.md](outputs/tables/compare_spec/README.md). Errors: [docs/error_analysis.md](docs/error_analysis.md).

## Limitations

- **0.5 is not calibrated.** Last-4 / fuse are miss-heavy at 0.5 (1 FP / 44–60 FN on 400). Use `pred` as a score; pick the FPR you can afford ([docs/error_analysis.md](docs/error_analysis.md)).
- **Nova / Infinity** stay hard on recall at 0.5 even when AUROC is high. Official val is DALL·E-only.
- **D4/D5/D6 mix-ins** did not beat D3 on official DALL·E (0.973 / 0.975 / 0.977 vs D3 0.978). Fuse last4+D6 is 0.9929 vs last4+D3 **0.9930**. Do not submit those heads.
- **Image-only**, English docs. Very small thumbnails are under-tested (eval is ~1024²-heavy).
- Training last-4 **on** D3 dropped official score vs last-4 alone — complementary fuse beats stacking.

## Team & contributions

| Member | Focus |
|---|---|
| kiki (`chi030303`) | Training/eval pipeline, CLIP-B last-4 submit, D3–D6 mix-ins, last4+mix fuse, EvalGEN (incl. Nova), contest write-ups |
| yun | Model ablations (backbone, last-4 vs first-4, CLIP-L, 336, dual-branch, consistency) |
| samily | Error / bad-case **analysis** (`analyze_badcase_galleries.py`, content-pattern slices); data-ablation design (A/D axes) |
| zhengcongyun | Bad-case **collection** pipeline; official robustness transforms |
| James | ComfyUI data generation (self-built t2i / i2i) |

Repo is **public**. Self-built mix-in (not SID / official val): [docs/dataset_release.md](docs/dataset_release.md) · [Kaggle aigctrace-mix](https://www.kaggle.com/datasets/wwjjames/aigctrace-mix).

## Team workflow

- **Daily handbook** (SSH, tmux, venv, GPUs): [docs/dev.md](docs/dev.md)
- Branch SOP (merge to `main` only after checks pass): [docs/SOP-git.md](docs/SOP-git.md)
- Roles: [docs/roles.md](docs/roles.md)
- GPU / Vast: [docs/gpu.md](docs/gpu.md)
- Communication and freeze dates: [docs/ops.md](docs/ops.md)

