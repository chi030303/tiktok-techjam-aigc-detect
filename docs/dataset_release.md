# Self-built mix-in (not a required contest field)

SID, COCO, DALL·E, and EvalGEN are **not** republished here. Official val / EvalGEN stay hold-out.

Self-built images: [Kaggle wwjjames/aigctrace-mix](https://www.kaggle.com/datasets/wwjjames/aigctrace-mix) (CC BY 4.0).

## What to put where

| Asset | Where | Publish? |
|---|---|---|
| SID_Set | already on [Hugging Face](https://huggingface.co/datasets/saberzl/SID_Set) | Link only — do not re-upload |
| CIFAKE | already on Kaggle | Link only |
| COCO val2017 + DALL·E Advanced (`data/val`) | official demo | **No.** Hold-out, not ours to republish as a mix |
| EvalGEN | hold-out | **No.** |
| **Self-built t2i / i2i** (Flux.2, SD3.5, nano, PixArt, SDXL, GPT, i2i) | [Kaggle: wwjjames/aigctrace-mix](https://www.kaggle.com/datasets/wwjjames/aigctrace-mix) | Yes — mix-in only |
| Hunyuan | never used | Do not add |

Unzip so these folders sit under `$DATA_ROOT/self_built/` (that directory is `--self-root` for `scripts/build_d3_mixin.py`):

| Folder | Used by |
|---|---|
| `flux2/` | D3 |
| `sd35/` | D3 |
| `nano_banana_vertex_batch/` | D3 |
| `GPT/`, `pixart_sigma_quality_v2/`, `sdxl_full_refiner_v1/` | D4 / D5 (not the submit mix) |
| `i2i/` | D6 (not the submit mix) |

D3 also needs WildFake slices via `python scripts/download_wildfake_subset.py C_unet_sd_original C_pixel_adm C_pixel_ddpm`. last-4 training does not need the Kaggle zip.

Do not claim official val numbers were trained on the Kaggle zip. Submit last-4 is SID-only; the fused D3 head replaces equal SID FLUX with the mix.
