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

## Results

**Submission model: `sid_dinov2`** — frozen DINOv2 ViT-L/14 backbone (~0.3B params, well under the 2B limit) + a 1,025-param linear head trained on SID_Set. Checkpoint: [release `v0.1-model`](https://github.com/chi030303/tiktok-techjam-aigc-detect/releases/tag/v0.1-model) (5.95 KB).

Robustness on the official demo val (COCO val2017 + DALL·E Advanced, n=400 balanced, AUROC):

| clean | jpeg q90 | q70 | q50 | q30 | blur σ0.5 | σ1.0 | σ2.0 | resize 0.5× | 0.25× | noise σ0.02 | σ0.05 | σ0.10 | jitter | crop 80% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.904 | 0.911 | 0.881 | 0.872 | 0.889 | 0.905 | 0.905 | 0.905 | 0.902 | 0.872 | 0.911 | 0.899 | 0.887 | 0.905 | 0.902 |

Robust macro-average over the 14 transformed conditions: **0.897** (min 0.872). On the
held-out **EvalGEN** benchmark (55,298 fakes from 5 unseen generator families + 10K real
pool): **AUROC 0.964**, recall 0.888 — best-in-team on the hardest family (Nova, recall
0.777). Full error analysis and per-generator breakdown:
[docs/error_analysis.md](docs/error_analysis.md).

## Reproduce

End-to-end, verified on a fresh `MODELS_ROOT` (no pre-existing weights):

```bash
# 1. environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. backbone (~1.2 GB DINOv2-L; or --only the one you need)
python scripts/download_backbones.py --only dinov2-vit-large-patch14

# 3. checkpoint (5.95 KB linear head)
gh release download v0.1-model -R chi030303/tiktok-techjam-aigc-detect -p sid_dinov2_best.pt -O checkpoints/sid_dinov2_best.pt

# 4. inference: directory in, JSON out
python predict.py <image_dir> out.json --ckpt checkpoints/sid_dinov2_best.pt
```

Sanity check from our run (2 COCO reals, 2 DALL·E fakes): reals scored 0.12 / 0.12,
fakes 0.79 / 0.92. The checkpoint stores only `{head, backbone-name}` — the backbone is
resolved from `models/`, so keep step 2 and step 3 together.

Retraining from scratch: recipes live under `experiments/<name>/recipe.yaml`
(e.g. `dinov2l_linear_sid`), run with
`python scripts/run_experiment.py experiments/<name>/recipe.yaml`; ablation
results and the model-selection matrix are in [docs/error_analysis.md](docs/error_analysis.md).

## Limitations

- **Linear-probe ceiling**: the backbone is frozen; accuracy is bounded by DINOv2
  features. Partial unfreezing and MLP heads were ablated and did not beat the probe
  on the demo val.
- **Scores are not calibrated**: at threshold 0.5 `sid_dinov2` is miss-heavy
  (FP 10 / FN 88). `pred` is a score — downstream should pick the operating point by
  their false-accusation budget (see [docs/error_analysis.md](docs/error_analysis.md)).
- **Generator coverage**: Nova and Infinity remain hard (recall 0.45–0.74 on EvalGEN
  across all our models). The demo val contains only DALL·E fakes.
- **Image modality only**, English-only project scope; evaluation subsets are
  1024²-heavy, so very small thumbnails are under-tested.

## Team & contributions

| Member | Focus |
|---|---|
| chi030303 (tianqi) | Tech lead: training/eval infra, recipes, model & data ablations |
| Jasminetothemoon (Zyun) | Official transforms & manifests, bad-case pipeline + gallery, error analysis |
| kiki | Training runs, EvalGEN streaming eval, feature cache |
| James | Dataset research & selection |

*(names/handles — please double-check before Devpost submit)*

Repo must be **public** by 1 Sep 12:00 GMT+8 (Settings → General → Danger zone).

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
