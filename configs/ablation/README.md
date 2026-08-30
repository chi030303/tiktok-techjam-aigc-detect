# Ablation manifest configs (DATA_ABLATION_PLAN.md)

Build manifests after source indexes exist:

```bash
# 1. SID source index (once SID_Set is on disk)
python scripts/build_sid_manifest.py

# 2. Official val holdout index (for phash leak filter during train)
python -m src.transforms.build_source --root data/val --dataset demo_wildfake \
  --split val --out data/manifests/source_demo_val.jsonl

# 3. Sample ablation grids
python scripts/build_ablation_manifest.py configs/ablation/D1_sid_only.yaml
python scripts/build_ablation_manifest.py configs/ablation/ --all
```

## WildFake (C-UNet / C-Pixel)

```bash
# Download cross-arch packs (ModelScope; see configs/wildfake/subsets.yaml)
python scripts/download_wildfake_subset.py --list
python scripts/download_wildfake_subset.py C_pixel_ddpm C_pixel_adm   # ~8GB + ~18GB
python scripts/download_wildfake_subset.py C_unet_sd_original         # originalSD part_7 ~14GB

# Index -> per-generator source manifests (min-side 512 for UNet)
python scripts/build_wildfake_manifest.py --root data/wildfake/cross_arch/ddpm \
  --force-generator ddpm --out data/manifests/source_wildfake_ddpm.jsonl
python scripts/build_wildfake_manifest.py --root data/wildfake/cross_arch/adm \
  --force-generator adm --out data/manifests/source_wildfake_adm.jsonl
python scripts/build_wildfake_manifest.py --root data/wildfake/cross_arch/sd_original \
  --force-generator sd_original --min-side 512 \
  --out data/manifests/source_wildfake_sd_original.jsonl

# Ablation manifests
python scripts/build_ablation_manifest.py configs/ablation/C_unet_sd_original.yaml
python scripts/build_ablation_manifest.py configs/ablation/C_pixel_adm_ddpm.yaml
```

Each config samples **fixed seed**, **8k real + 8k fake** (unless noted), excluding `partial_manipulation`.
The data owner delivers these JSONL files; wiring them into the training recipe
belongs to the model/training owner.
