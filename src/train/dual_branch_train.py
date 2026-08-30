# 2026-08-30, yun, exp3 training loop: frozen CLIP branch + trainable freq CNN, full eval each epoch
"""freq_cnn + head are trainable, so (unlike the plain frozen-linear-probe path)
eval features aren't static across epochs -- re-run the whole model on eval
each epoch instead of scoring against a one-time feature cache.
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.sid_dual import SidDualBranchDataset
from src.models.dual_branch_probe import DualBranchProbe
from src.paths import artifact_dir, models_root
from src.recipe import validate
from src.train.sid_online import _head_kind, _image_size, _tfm

# end


def _freq_tfm(size: int):
    # 2026-08-30, yun, residual pixels center on 128, not natural-image stats -> plain [0,1] scale
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])


@torch.no_grad()
def _evaluate_full(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    n = 0
    correct = 0
    tp = fp = fn = 0
    for xa, xb, y in loader:
        xa = xa.to(device, non_blocking=True)
        xb = xb.to(device, non_blocking=True)
        y = y.to(device)
        logit = model(xa, xb)
        pred = (logit.sigmoid() >= 0.5).long()
        n += y.numel()
        correct += int((pred == y).sum().item())
        tp += int(((pred == 1) & (y == 1)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"n": n, "acc": correct / max(1, n), "precision_fake": prec, "recall_fake": rec}


def run_train_dual_branch(recipe: dict) -> dict:
    validate(recipe)
    name = recipe["name"]
    art = artifact_dir(name)
    ckpt_dir = art / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    backbone_name = recipe["backbone"]
    head_kind = _head_kind(recipe)
    image_size = _image_size(recipe)
    freq_dim = int((recipe.get("dual_branch") or {}).get("freq_dim", 128))

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
        f"dual-branch  freq_dim={freq_dim}  batch={batch}  epochs={epochs}  image_size={image_size}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = DualBranchProbe(
        models_root() / backbone_name, freq_dim=freq_dim, head_kind=head_kind
    ).to(device)

    rgb_tfm = _tfm(image_size)
    freq_tfm = _freq_tfm(image_size)
    train_ds = SidDualBranchDataset(
        "train", rgb_tfm, freq_tfm, augment=True, seed=seed, max_images=max_train
    )
    eval_ds = SidDualBranchDataset(
        "validation", rgb_tfm, freq_tfm, augment=False, seed=seed + 1, max_images=max_eval
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
    eval_loader = DataLoader(eval_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)

    trainable = list(model.freq_cnn.parameters()) + list(model.head.parameters())
    opt = torch.optim.Adam(trainable, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n_batches = max(1, len(train_loader))

    def ckpt_payload() -> dict:
        return {
            "kind": "dual_branch_clip",
            "head": model.head.state_dict(),
            "freq_cnn_state_dict": model.freq_cnn.state_dict(),
            "freq_dim": freq_dim,
            "backbone": backbone_name,
            "recipe": name,
            "head_kind": head_kind,
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
        for step, (xa, xb, y) in enumerate(train_loader, 1):
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            y = y.float().to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(xa, xb)
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
        "freq_dim": freq_dim,
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
