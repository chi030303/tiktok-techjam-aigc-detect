# 2026-08-30, yun, model-track: SID + online official-transform aug, frozen backbone + linear head
"""Reproduces the SID_Set + online-aug baseline, and its image_size ablation (exp4).

Self-contained: does not modify src/train/loop.py (CIFAKE feature-cache path)
or src/paths.py -- this module owns its own transform/cache-path helpers so
the model-ablation branch adds a second training engine instead of changing
the one other experiments already depend on.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.sid import SidHfDataset
from src.models.sid_linear_probe import FrozenLinearProbe
from src.paths import artifact_dir, exp_root, models_root
from src.recipe import validate

# end

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def _tfm(size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _image_size(recipe: dict) -> int:
    # 2026-08-30, yun, exp4: input resolution 224 (native) vs 336 (interpolated pos-emb)
    return int(recipe.get("image_size") or IMAGE_SIZE)


def _interpolate_pos(recipe: dict) -> bool:
    return _image_size(recipe) != IMAGE_SIZE


def _head_kind(recipe: dict) -> str:
    return str(recipe.get("head") or "linear")


def _input_mode(recipe: dict) -> str:
    return str(recipe.get("input_mode") or "rgb")


def _aug_expand(recipe: dict) -> bool:
    return str((recipe.get("aug") or {}).get("expand") or "") in {"six_ops", "all_ops"}


def _enable_fast_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def _sid_feat_cache_path(backbone: str, split: str, n: int, seed: int, size: int) -> Path:
    # 2026-08-30, yun, own cache namespace (sid_online/) so this never collides with
    # src/train/featcache.py's cifake-keyed cache under _featcache/<backbone>/
    d = exp_root() / "_featcache_sid_online" / backbone
    d.mkdir(parents=True, exist_ok=True)
    return d / f"sid_{split}_n{n}_seed{seed}_s{size}.pt"


@torch.no_grad()
def evaluate_head(head: nn.Module, feat: torch.Tensor, y: torch.Tensor, device: torch.device, batch: int) -> dict:
    head.eval()
    n = 0
    correct = 0
    tp = fp = fn = 0
    for i in range(0, feat.size(0), batch):
        xb = feat[i : i + batch].float().to(device, non_blocking=True)
        yb = y[i : i + batch].long().to(device)
        logit = head(xb).squeeze(-1)
        pred = (logit.sigmoid() >= 0.5).long()
        n += yb.numel()
        correct += int((pred == yb).sum().item())
        tp += int(((pred == 1) & (yb == 1)).sum().item())
        fp += int(((pred == 1) & (yb == 0)).sum().item())
        fn += int(((pred == 0) & (yb == 1)).sum().item())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"n": n, "acc": correct / max(1, n), "precision_fake": prec, "recall_fake": rec}


@torch.no_grad()
def _extract_features(model, dataset, batch: int, workers: int, device: torch.device, tag: str):
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    amp = device.type == "cuda"
    feats, labels = [], []
    n_batches = max(1, len(loader))
    model.eval()
    for step, (x, y, _) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            f = model.encode(x)
        feats.append(f.detach().to(dtype=torch.float16, device="cpu"))
        labels.append(y.cpu())
        if step == 1 or step % 100 == 0 or step == n_batches:
            print(f"  extract {tag} {step}/{n_batches}", flush=True)
    return torch.cat(feats, 0), torch.cat(labels, 0)


def _load_or_extract_eval_feats(model, eval_ds, backbone: str, image_size: int, seed: int, batch: int, workers: int, device: torch.device):
    path = _sid_feat_cache_path(backbone, "validation", len(eval_ds), seed, image_size)
    if path.is_file():
        blob = torch.load(path, map_location="cpu", weights_only=True)
        print(f"feat cache hit {path}  n={blob['feat'].size(0)}", flush=True)
        return blob["feat"], blob["y"]
    feat, y = _extract_features(model, eval_ds, batch, workers, device, tag="validation")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feat": feat.contiguous(), "y": y.contiguous()}, path)
    print(f"wrote feat cache {path}", flush=True)
    return feat, y


def run_train_sid_online(recipe: dict) -> dict:
    """Baseline (clipb16_linear_sid_aug) and its image_size ablation (exp4, res336)."""
    validate(recipe)
    name = recipe["name"]
    art = artifact_dir(name)
    ckpt_dir = art / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    backbone = recipe["backbone"]
    head_kind = _head_kind(recipe)
    input_mode = _input_mode(recipe)
    image_size = _image_size(recipe)
    interp = _interpolate_pos(recipe)
    expand = _aug_expand(recipe)

    optim = recipe.get("optim") or {}
    epochs = int(optim.get("epochs", 3))
    batch = int(optim.get("batch_size", 64))
    lr = float(optim.get("lr", 1e-3))
    workers = int(optim.get("num_workers", 4))
    seed = int(optim.get("seed", 0))

    smoke = recipe.get("smoke") or {}
    max_train = smoke.get("max_train")
    max_eval = smoke.get("max_eval")

    print(
        f"sid-online  backbone={backbone}  head={head_kind}  image_size={image_size}  "
        f"batch={batch}  epochs={epochs}  expand={expand}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp = device.type == "cuda"

    model = FrozenLinearProbe(
        models_root() / backbone, head_kind=head_kind, interpolate_pos_encoding=interp
    ).to(device)

    train_ds = SidHfDataset(
        "train",
        _tfm(image_size),
        augment=True,
        expand_ops=expand,
        seed=seed,
        max_images=max_train,
        input_mode=input_mode,
    )
    eval_ds = SidHfDataset(
        "validation",
        _tfm(image_size),
        augment=False,
        seed=seed + 1,
        max_images=max_eval,
        input_mode=input_mode,
    )
    print(f"  sid n_train={len(train_ds)} n_eval={len(eval_ds)}", flush=True)

    eval_feat, eval_y = _load_or_extract_eval_feats(
        model, eval_ds, backbone, image_size, seed + 1, batch, min(workers, 4), device
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n_batches = max(1, len(train_loader))

    t0 = time.time()
    best_acc = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        running = 0.0
        seen = 0
        for step, (x, y, _) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.float().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logit = model(x)
                loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * y.size(0)
            seen += y.size(0)
            if step == 1 or step % 200 == 0 or step == n_batches:
                print(f"  epoch {epoch}/{epochs}  step {step}/{n_batches}  loss={loss.item():.4f}", flush=True)
        ev = evaluate_head(model.head, eval_feat, eval_y, device, 8192)
        row = {"epoch": epoch, "train_loss": running / max(1, seen), **ev}
        history.append(row)
        print(
            f"epoch {epoch}/{epochs}  loss={row['train_loss']:.4f}  clean_eval_acc={ev['acc']:.3f}  n={ev['n']}",
            flush=True,
        )
        if ev["acc"] > best_acc:
            best_acc = ev["acc"]
            torch.save(
                {
                    "head": model.head.state_dict(),
                    "backbone": backbone,
                    "recipe": name,
                    "head_kind": head_kind,
                    "input_mode": input_mode,
                    "image_size": image_size,
                },
                ckpt_dir / "best.pt",
            )
    train_s = time.time() - t0
    final = evaluate_head(model.head, eval_feat, eval_y, device, 8192)
    metrics = {
        "experiment": name,
        "backbone": backbone,
        "online_aug": True,
        "expand_ops": expand,
        "source": "sid_set",
        "input_mode": input_mode,
        "head_kind": head_kind,
        "image_size": image_size,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "train_seconds": round(train_s, 2),
        "best_eval_acc": best_acc,
        "final": final,
        "history": history,
    }
    out = art / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}", flush=True)
    print(f"done  train={train_s:.1f}s  clean_acc={final['acc']:.3f}", flush=True)
    return metrics
# end
