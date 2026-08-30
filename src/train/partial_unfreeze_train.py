# 2026-08-30, yun, exp1 training loop: partial-unfreeze needs a fresh forward every eval too
"""Train a PartialUnfreezeClipProbe on SID_Set with online official-transform aug.

Cannot reuse src.train.loop's cached-eval-features trick: the backbone changes
every step here, so eval must re-run the full (current) model each epoch, not
score a linear head against features frozen at epoch 0.
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sid import SidHfDataset
from src.models.partial_unfreeze_probe import PartialUnfreezeClipProbe
from src.paths import artifact_dir, models_root
from src.recipe import validate
from src.train.sid_online import _aug_expand, _image_size, _input_mode, _interpolate_pos, _tfm

# end


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
    backbone_lr = float(pu.get("backbone_lr", 1e-5))
    head_lr = float(pu.get("head_lr", 1e-3))

    backbone_name = recipe["backbone"]
    head_kind = str(recipe.get("head") or "linear")
    input_mode = _input_mode(recipe)
    image_size = _image_size(recipe)
    interp = _interpolate_pos(recipe)
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
        f"partial-unfreeze  n_layers={n_unfreeze}  backbone_lr={backbone_lr}  head_lr={head_lr}  "
        f"batch={batch}  epochs={epochs}  image_size={image_size}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    bb = models_root() / backbone_name
    model = PartialUnfreezeClipProbe(
        bb, n_unfreeze=n_unfreeze, head_kind=head_kind, interpolate_pos_encoding=interp
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

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    opt = torch.optim.Adam(
        [
            {"params": model.backbone_trainable_parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]
    )
    loss_fn = nn.BCEWithLogitsLoss()
    n_batches = max(1, len(train_loader))

    def ckpt_payload() -> dict:
        return {
            "kind": "partial_unfreeze_clip",
            "head": model.head.state_dict(),
            "backbone_state_dict": model.backbone.state_dict(),
            "n_unfreeze": n_unfreeze,
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
