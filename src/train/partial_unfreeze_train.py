# 2026-08-31, tianqi, port yun partial-unfreeze loop; CLIP-L / res336 combo
"""Train PartialUnfreezeClipProbe on SID with online aug. No feat cache: backbone moves."""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.sid import SidHfDataset
from src.models.partial_unfreeze_probe import PartialUnfreezeClipProbe
from src.paths import artifact_dir, models_root
from src.recipe import validate
from src.train.loop import _aug_expand, _input_mode

# end

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _image_size(recipe: dict) -> int:
    return int(recipe.get("image_size") or 224)


def _tfm(size: int):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@torch.no_grad()
def _evaluate_full(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    n = 0
    correct = 0
    tp = fp = fn = 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        logit = model(x)
        pred = (logit.sigmoid() >= 0.5).long()
        n += y.numel()
        correct += int((pred == y).sum().item())
        tp += int(((pred == 1) & (y == 1)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"n": n, "acc": correct / max(1, n), "precision_fake": prec, "recall_fake": rec}


def run_train_partial_unfreeze(recipe: dict) -> dict:
    validate(recipe)
    name = recipe["name"]
    art = artifact_dir(name)
    ckpt_dir = art / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pu = recipe.get("partial_unfreeze") or {}
    n_unfreeze = int(pu.get("n_layers", 2))
    unfreeze_from = str(pu.get("unfreeze_from") or "last")
    backbone_lr = float(pu.get("backbone_lr", 1e-5))
    head_lr = float(pu.get("head_lr", 1e-3))

    backbone_name = recipe["backbone"]
    head_kind = str(recipe.get("head") or "linear")
    input_mode = _input_mode(recipe)
    image_size = _image_size(recipe)
    interp = image_size != 224
    expand = _aug_expand(recipe)

    optim = recipe.get("optim") or {}
    epochs = int(optim.get("epochs", 3))
    batch = int(optim.get("batch_size", 64))
    workers = int(optim.get("num_workers", 4))
    seed = int(optim.get("seed", 0))

    smoke = recipe.get("smoke") or {}
    max_train = smoke.get("max_train")
    max_eval = smoke.get("max_eval")

    print(
        f"partial-unfreeze  n_layers={n_unfreeze}  from={unfreeze_from}  backbone_lr={backbone_lr}  head_lr={head_lr}  "
        f"batch={batch}  epochs={epochs}  image_size={image_size}  backbone={backbone_name}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    bb = models_root() / backbone_name
    model = PartialUnfreezeClipProbe(
        bb,
        n_unfreeze=n_unfreeze,
        head_kind=head_kind,
        interpolate_pos_encoding=interp,
        unfreeze_from=unfreeze_from,
    ).to(device)

    # 2026-08-31, tianqi, D3 mixin: same replace-FLUX path as frozen SID mix
    extra_rows = None
    train_cfg = recipe.get("train") or {}
    mixin = train_cfg.get("mixin_manifest")
    if mixin:
        from src.data.manifest_ds import load_mixin_rows

        extra_rows = load_mixin_rows(mixin)
        print(f"  mixin_manifest={mixin} n_extra={len(extra_rows)}", flush=True)
    replace_sid = bool(train_cfg.get("replace_sid_fakes", True))
    # end

    train_ds = SidHfDataset(
        "train",
        _tfm(image_size),
        augment=True,
        expand_ops=expand,
        seed=seed,
        max_images=max_train,
        input_mode=input_mode,
        extra_rows=extra_rows,
        replace_sid_fakes=replace_sid if extra_rows else False,
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

    loader_kw = dict(
        num_workers=min(workers, 4),
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    if workers > 0:
        loader_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, **loader_kw)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=min(workers, 4),
        pin_memory=True,
    )

    opt = torch.optim.Adam(
        [
            {"params": model.backbone_trainable_parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]
    )
    loss_fn = nn.BCEWithLogitsLoss()
    amp = device.type == "cuda"
    n_batches = max(1, len(train_loader))

    def ckpt_payload() -> dict:
        return {
            "kind": "partial_unfreeze_clip",
            "head": model.head.state_dict(),
            "backbone_state_dict": model.backbone.state_dict(),
            "n_unfreeze": n_unfreeze,
            "unfreeze_from": unfreeze_from,
            "backbone": backbone_name,
            "recipe": name,
            "head_kind": head_kind,
            "input_mode": input_mode,
            "image_size": image_size,
        }

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
                print(
                    f"  epoch {epoch}/{epochs}  step {step}/{n_batches}  loss={loss.item():.4f}",
                    flush=True,
                )
        ev = _evaluate_full(model, eval_loader, device)
        row = {"epoch": epoch, "train_loss": running / max(1, seen), **ev}
        history.append(row)
        print(
            f"epoch {epoch}/{epochs}  loss={row['train_loss']:.4f}  "
            f"eval_acc={ev['acc']:.3f}  n={ev['n']}",
            flush=True,
        )
        if ev["acc"] > best_acc:
            best_acc = ev["acc"]
            torch.save(ckpt_payload(), ckpt_dir / "best.pt")
    train_s = time.time() - t0
    final = _evaluate_full(model, eval_loader, device)
    metrics = {
        "experiment": name,
        "backbone": backbone_name,
        "n_unfreeze": n_unfreeze,
        "unfreeze_from": unfreeze_from,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
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
