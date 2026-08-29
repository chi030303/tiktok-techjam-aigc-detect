# TikTok TechJam 2026 — AIGC Image Detection

Image-level detector for real vs AI-generated images, with robustness under JPEG, blur, resize, noise, color jitter, and center crop.

**Do not commit datasets, checkpoints, or API keys.** This repo is code + docs only. Weights and images live on disk / Hugging Face / cloud storage. See [docs/data.md](docs/data.md).

## What this repo contains vs what it does not

| In git | Not in git (download locally) |
|---|---|
| Training / eval / `predict.py` scripts | Raw images (`data/`) |
| Small CSV robustness tables | `*.pth` / `*.safetensors` / CLIP weights |
| Docs, SOP, issue templates | Official demo set (COCO val2017 + DALL·E Advanced) |
| `requirements.txt` | `.env`, Vast credentials |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

## Team workflow

- **Daily handbook** (SSH, tmux, venv, GPUs): [docs/dev.md](docs/dev.md)
- Branch SOP (merge to `main` only after checks pass): [docs/SOP-git.md](docs/SOP-git.md)
- Roles: [docs/roles.md](docs/roles.md)
- GPU / Vast: [docs/gpu.md](docs/gpu.md)
- Communication and freeze dates: [docs/ops.md](docs/ops.md)

## Reproduce (after a real model exists)

```bash
python predict.py /path/to/image_dir ./outputs/pred.json --ckpt checkpoints/best.pt
```

Limitations and Day-3 write-up go here before Devpost submit. Repo must be **public** by 1 Sep 12:00 GMT+8.
