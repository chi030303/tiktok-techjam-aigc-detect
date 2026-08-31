# 2026-08-30, yun, exp2: BCE(A) + BCE(B) + lambda * |logit_A - logit_B|
"""Frozen CLIP-B/16 + linear head, trained on paired (clean, one-random-op) views.

Backbone stays frozen (identical to the aug baseline), so eval reuses the same
cached-features path as src.train.sid_online -- only the training inner loop
and the dataset (paired views) differ.
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sid import SidHfDataset
from src.data.sid_paired import SidPairedViewDataset
from src.models.sid_linear_probe import FrozenLinearProbe
from src.paths import artifact_dir, models_root
from src.recipe import validate
from src.train.sid_online import (
    _enable_fast_cuda,
    _head_kind,
    _image_size,
    _input_mode,
    _load_or_extract_eval_feats,
    _tfm,
    evaluate_head,
)

# end


def run_train_consistency(recipe: dict) -> dict:
    validate(recipe)
    name = recipe["name"]
    art = artifact_dir(name)
    ckpt_dir = art / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    backbone_name = recipe["backbone"]
    head_kind = _head_kind(recipe)
    input_mode = _input_mode(recipe)
    image_size = _image_size(recipe)

    lam = float((recipe.get("consistency") or {}).get("lambda", 0.3))
    optim = recipe.get("optim") or {}
    epochs = int(optim.get("epochs", 3))
    batch = int(optim.get("batch_size", 64))
    lr = float(optim.get("lr", 1e-3))
    workers = int(optim.get("num_workers", 4))
    seed = int(optim.get("seed", 0))

    smoke = recipe.get("smoke") or {}
    max_train = smoke.get("max_train")
    max_eval = smoke.get("max_eval")

    print(f"consistency  lambda={lam}  batch={batch}  epochs={epochs}", flush=True)

    _enable_fast_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    model = FrozenLinearProbe(models_root() / backbone_name, head_kind=head_kind).to(device)

    train_ds = SidPairedViewDataset(
        "train", _tfm(image_size), seed=seed, max_images=max_train
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

    t_extract = time.time()
    eval_feat, eval_y = _load_or_extract_eval_feats(
        model, eval_ds, backbone_name, image_size, seed + 1, batch, min(workers, 4), device
    )
    eval_extract_s = time.time() - t_extract

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
    bce = nn.BCEWithLogitsLoss()
    n_batches = max(1, len(train_loader))

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
            with torch.amp.autocast("cuda", enabled=amp):
                logit_a = model(xa)
                logit_b = model(xb)
                loss = bce(logit_a, y) + bce(logit_b, y) + lam * (logit_a - logit_b).abs().mean()
            loss.backward()
            opt.step()
            running += float(loss.item()) * y.size(0)
            seen += y.size(0)
            if step == 1 or step % 200 == 0 or step == n_batches:
                print(
                    f"  epoch {epoch}/{epochs}  step {step}/{n_batches}  loss={loss.item():.4f}",
                    flush=True,
                )
        ev = evaluate_head(model.head, eval_feat, eval_y, device, 8192)
        row = {"epoch": epoch, "train_loss": running / max(1, seen), **ev}
        history.append(row)
        print(
            f"epoch {epoch}/{epochs}  loss={row['train_loss']:.4f}  "
            f"clean_eval_acc={ev['acc']:.3f}  n={ev['n']}",
            flush=True,
        )
        if ev["acc"] > best_acc:
            best_acc = ev["acc"]
            torch.save(
                {
                    "head": model.head.state_dict(),
                    "backbone": backbone_name,
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
        "backbone": backbone_name,
        "consistency_lambda": lam,
        "source": "sid_set",
        "input_mode": input_mode,
        "head_kind": head_kind,
        "image_size": image_size,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "eval_extract_seconds": round(eval_extract_s, 2),
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
