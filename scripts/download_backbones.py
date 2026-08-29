#!/usr/bin/env python3
# 2026-08-29, tianqi, cache CLIP / ResNet / DINOv2 weights into models/ (shared, not git)
"""Download the four detector backbones into a local folder (default /workspace/models)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

# end

# 2026-08-29, tianqi, OpenAI CLIP B/16 + L/14, torchvision-compatible ResNet-50, DINOv2 ViT-L/14
BACKBONES = {
    "clip-vit-base-patch16": "openai/clip-vit-base-patch16",
    "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
    "resnet-50": "microsoft/resnet-50",
    "dinov2-vit-large-patch14": "facebook/dinov2-large",
}
# end


def default_models_root() -> Path:
    shared = Path("/workspace/models")
    if shared.is_dir():
        return shared
    return Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    # 2026-08-29, tianqi, allow overriding dest so teammates share /workspace/models
    parser = argparse.ArgumentParser(description="Download CLIP-B/16, CLIP-L/14, ResNet-50, DINOv2-L/14")
    parser.add_argument("--out", type=Path, default=default_models_root())
    parser.add_argument("--only", nargs="*", default=None, help="Subset of local folder names")
    args = parser.parse_args()
    # end

    args.out.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    wanted = set(args.only) if args.only else set(BACKBONES)
    unknown = wanted - set(BACKBONES)
    if unknown:
        raise SystemExit(f"unknown backbone keys: {sorted(unknown)}")

    for local_name, repo_id in BACKBONES.items():
        if local_name not in wanted:
            continue
        dest = args.out / local_name
        print(f"==> {local_name}  ({repo_id}) -> {dest}", flush=True)
        # 2026-08-29, tianqi, snapshot into models/<name> so training can load offline
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            token=token,
            # 2026-08-29, tianqi, skip TF/Flax; CLIP-B/16 has no safetensors so keep .bin
            ignore_patterns=["*.msgpack", "tf_model.h5", "tf_model.h5.*"],
            # end
        )
        # end
        n_files = sum(1 for p in dest.rglob("*") if p.is_file())
        print(f"    done  files={n_files}", flush=True)

    print("all requested backbones are in", args.out)


if __name__ == "__main__":
    main()
