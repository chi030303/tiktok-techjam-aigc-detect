# Experiments

One folder per experiment. The **recipe** is what you commit; weights and logs are not.

```text
experiments/<name>/recipe.yaml     # git (this clone)
/workspace/experiments/<name>/    # Vast artifacts: ckpts, logs, pred.json
```

`data/` and `models/` are **shared** (`/workspace/data`, `/workspace/models`). Do not copy SID_Set or CLIP into `/workspace/kiki/...`.

```bash
python scripts/run_experiment.py experiments/clipb16_linear_sid/recipe.yaml
```

| Field | Meaning |
|---|---|
| `backbone` | Folder name under `MODELS_ROOT` |
| `train.datasets` | Keys under `DATA_ROOT` (e.g. `sid_set`, `cifake`) |
| `train.forbid` | Must include `val` and `evalgen` |
| `eval` | `official_val` = TechJam demo set; `evalgen` = extra hold-out. Score with `scripts/run_eval.py`, not the train loader. |

<!-- 2026-08-29, tianqi, eval is run_eval.py after train, not the train loop -->
```bash
python scripts/run_experiment.py experiments/clipb16_linear_sid/recipe.yaml --train
python scripts/run_eval.py robustness --split official_val --conditions daily --max-images 400 \
  --experiment clipb16_linear_sid --ckpt /workspace/experiments/clipb16_linear_sid/ckpts/best.pt
```
<!-- end -->

Name folders like `<backbone>_<head>_<data>_<trick>` so the directory listing is the experiment index.
