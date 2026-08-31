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
# inference / train also need:
pip install torch torchvision transformers
cp .env.example .env   # add HF_TOKEN locally if needed

# smoke: dummy scores, no weights required
python predict.py ./fixtures/sample_images ./outputs/pred.json
bash scripts/check.sh
```

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
3. **WildFake full** — optional, huge. Do **not** train on the official demo subset below.
4. **Demo / reference only (no train)**  
   `python scripts/download_official_val.py` → `data/val/real` (COCO val2017) + `data/val/fake` (DALL·E Advanced 8843).  
   This **is** the official TechJam demonstration set. It does not count toward the score. Do not train on it.
5. **EvalGEN** (extra hold-out, also no train)  
   `python scripts/download_evalgen.py` → `data/evalgen/`
6. **Backbones** (all ≪ 2B)  
   `python scripts/download_backbones.py` — CLIP-B/16, CLIP-L/14, ResNet-50, DINOv2 ViT-L/14 → `models/`

On Vast, images and weights live in **`/workspace/data` and `/workspace/models`**, not inside a personal clone. Experiment recipes live in git under `experiments/<name>/recipe.yaml`; run artifacts go to `/workspace/experiments/<name>/`. See [experiments/README.md](experiments/README.md).

Official inference contract: a directory of images in, JSON out:

```json
[{"image_path": "data/val/fake/0001.jpg", "pred": 0.87}]
```

`pred` = confidence the image is AIGC.

<!-- 2026-08-29, tianqi, eval pipeline uses predict.py so judges and we share one interface -->
## Evaluate (robustness table)

Uses the same `predict.py` contract. Demo val / EvalGEN stay out of training.

```bash
# daily compact table: clean vs JPEG-50 vs center-crop 80%
python scripts/run_eval.py robustness --split official_val --conditions daily --max-images 400

# score an existing JSON
python scripts/run_eval.py score --pred ./outputs/pred.json --split official_val
```

Tables land in `outputs/tables/` (csv / md / json). Full transform list and hold-out rules: [docs/data.md](docs/data.md).
<!-- end -->

## Results

<!-- 2026-08-31, tianqi, contest ranking last4 0.990 / fuse 0.993 -->
**Single ckpt: CLIP-B/16 last-4** — SID ~140k + official online aug, unfreeze last 4 vision blocks. Official val 400 formula **0.990** (full 13,843 **0.989**).  
<!-- 2026-08-31, tianqi, fuse D5 matches D3 fuse on official; D5 alone does not -->
**Two ckpts:** mean-logit fuse of last-4 + D3 mix = **0.993**. Fuse last-4 + **D5** (D3∪D4 mix) is also **0.993** — D5 alone is only 0.975, so the story is complementary heads, not “more mix-in always wins.”
# end

| model | formula 400 | DALL·E AUC | Acc@0.5 | EvalGEN AUROC |
|---|---:|---:|---:|---:|
| fuse last4+D3 | **0.993** | 0.995 | 0.888 | **0.997** |
| fuse last4+D5 | **0.993** | 0.995 | — | — |
| CLIP-B last-4 | **0.990** | 0.991 | 0.848 | 0.989 |
| D3 dualbranch | 0.983 | 0.988 | 0.948 | 0.995 |
| D3 mix (frozen CLIP-B) | 0.978 | 0.985 | 0.940 | 0.995 |
| CLIP-L SID-aug | 0.976 | 0.976 | 0.885 | 0.995 |
| CLIP-B SID-aug | 0.970 | 0.969 | 0.900 | 0.992 |
| SID DINOv2 frozen | 0.900 | 0.904 | — | 0.964 |

15-condition table and Nova split: [docs/robustness.md](docs/robustness.md). Full grid: [outputs/tables/compare_spec/README.md](outputs/tables/compare_spec/README.md). Errors: [docs/error_analysis.md](docs/error_analysis.md).
# end

## Reproduce

Need GPU + CLIP-B weights. Official val / EvalGEN stay **out of training**.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision transformers
python scripts/download_backbones.py --only clip-vit-base-patch16

# single submit ckpt
python predict.py <image_dir> out.json \
  --ckpt experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt

# optional fuse (last4 + D3)
python predict.py <image_dir> out.json \
  --ckpt experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt \
  --ckpt-b experiments/clipb16_linear_sid_d3_mix/ckpts/best.pt
```

Retrain last-4: `python scripts/run_experiment.py experiments/clipb16_linear_sid_unfreeze4/recipe.yaml --train`  
D3 mix: `experiments/clipb16_linear_sid_d3_mix/recipe.yaml`. Demo val robustness:

```bash
python scripts/run_full_eval.py --split official_val --conditions full --max-images 400 --seed 0 \
  --ckpt last4=experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt
```

Frozen SID DINOv2 (ablation ~0.90) remains in release `v0.1-model` if you need that probe.

## Limitations

- **0.5 is not calibrated.** Last-4 / fuse are miss-heavy at 0.5 (1 FP / 44–60 FN on 400). Use `pred` as a score; pick the FPR you can afford ([docs/error_analysis.md](docs/error_analysis.md)).
- **Nova / Infinity** stay hard on recall at 0.5 even when AUROC is high. Official val is DALL·E-only.
- **D4/D5 mix-ins** did not beat D3 on official DALL·E; do not submit those heads.
- **Image-only**, English docs. Very small thumbnails are under-tested (eval is ~1024²-heavy).
- Training last-4 **on** D3 dropped official score vs last-4 alone — complementary fuse beats stacking.

## Team & contributions

<!-- 2026-08-31, tianqi, names/roles from kiki for Devpost -->
| Member | Focus |
|---|---|
| kiki (`chi030303`) | Training/eval pipeline, CLIP-B last-4 submit, D3/D4/D5 mix-ins, last4+mix fuse, EvalGEN (incl. Nova), contest write-ups |
| yun | Model ablations (backbone, last-4 vs first-4, CLIP-L, 336, dual-branch, consistency) |
| samily | Error / bad-case **analysis** (`analyze_badcase_galleries.py`, content-pattern slices); data-ablation design (A/D axes) |
| zhengcongyun | Bad-case **collection** pipeline; official robustness transforms |
| James | ComfyUI data generation (self-built t2i / i2i) |
# end

Repo must be **public** by 1 Sep 12:00 GMT+8. Self-built images: [docs/dataset_release.md](docs/dataset_release.md) (**publish tomorrow morning, not tonight**).

## Team workflow

- **Daily handbook** (SSH, tmux, venv, GPUs): [docs/dev.md](docs/dev.md)
- Branch SOP (merge to `main` only after checks pass): [docs/SOP-git.md](docs/SOP-git.md)
- Roles: [docs/roles.md](docs/roles.md)
- GPU / Vast: [docs/gpu.md](docs/gpu.md)
- Communication and freeze dates: [docs/ops.md](docs/ops.md)

