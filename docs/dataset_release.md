# 2026-08-31, tianqi, self-built data release: wait until 1 Sep morning; not a contest field
# Dataset release (after the deadline)

**Do not publish tonight.** Contest freeze is 1 Sep 12:00 GMT+8. Open the dataset **tomorrow morning** after the repo is public and the Devpost is in, so we do not scramble licenses during judging.

This is **not** a required TechJam field. The official demo val and EvalGEN must **not** be re-hosted as “our dataset”.

## What to put where

| Asset | Where | Publish? |
|---|---|---|
| SID_Set | already on [Hugging Face](https://huggingface.co/datasets/saberzl/SID_Set) | Link only — do not re-upload |
| CIFAKE | already on Kaggle | Link only |
| COCO val2017 + DALL·E Advanced (`data/val`) | official demo | **No.** Hold-out, not ours to republish as a mix |
| EvalGEN | hold-out | **No.** |
| **Self-built t2i** (Flux.2, SD3.5, Nano Banana, PixArt, SDXL, …) | **Kaggle Dataset** (good for zips + license + card) | Tomorrow morning, if licenses allow |
| **i2i 59 triplets** (real + nano + Codex, `source_id`) | same Kaggle dataset, separate folder | Yes, with pairing file |
| GPT-image fakes | same zip or drop | **Check OpenAI image ToS** before including; drop if unclear |
| Hunyuan | never used | Do not add |

**Kaggle vs Hugging Face:** Kaggle is fine for a **self-built image zip** (versioned, license box, DOI-ish URL). Hugging Face is better if you later add a loading script. Pick **one** primary URL and put it in README. Do not upload 14万 SID.

## Suggested Kaggle card (draft)

- **Title:** TechJam AIGC mix-in (self-built t2i + 59 i2i triplets)
- **Task:** binary labels only (`REAL=0`, `FAKE=1`); not for object detection
- **Must include:** `LICENSE`, generator name per file, `source_id` for i2i, “not trained on COCO val / DALL·E / EvalGEN”
- **Do not include:** prompts that copy trademarked characters; `data/val`; EvalGEN

## Tomorrow morning checklist

1. Repo already **public**.
2. Strip GPT-image if legal is unsure.
3. Upload zip of **self-built only** + jsonl manifests (`D3_sid_mixin` / `A2_i2i_hard60` paths rewritten to relative).
4. Add one paragraph + URL under README “Datasets”.
5. Do not claim official val numbers were trained on this zip.

Vast paths stay private (`/workspace/...`).
# end
