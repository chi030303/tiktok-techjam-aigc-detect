# 2026-08-29, tianqi, extract frozen feats once, then train only the linear head
from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from src.data.cifake import load_cifake_rows
from src.models.linear_probe import FrozenLinearProbe
from src.paths import artifact_dir, models_root
from src.recipe import validate
from src.train.featcache import cache_path, extract_features, load_cached, save_cached

# end

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def _tfm():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _enable_fast_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


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


def _load_or_extract(
    *,
    backbone: str,
    split: str,
    rows: list,
    seed: int,
    model,
    batch: int,
    workers: int,
    device: torch.device,
    use_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    path = cache_path(backbone, split, len(rows), seed, IMAGE_SIZE)
    t0 = time.time()
    if use_cache:
        cached = load_cached(path)
        if cached is not None:
            feat, y = cached
            print(f"feat cache hit {path}  n={feat.size(0)} d={feat.size(1)}", flush=True)
            return feat, y, time.time() - t0
    feat, y = extract_features(model, rows, _tfm(), batch, workers, device, tag=split)
    if use_cache:
        save_cached(path, feat, y)
        print(f"wrote feat cache {path}", flush=True)
    return feat, y, time.time() - t0


def run_train(recipe: dict) -> dict:
    validate(recipe)
    if not recipe.get("freeze_backbone", True):
        raise SystemExit("this loop extracts features once; freeze_backbone must be true")
    name = recipe["name"]
    art = artifact_dir(name)
    ckpt_dir = art / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    smoke = recipe.get("smoke")
    seed = 0
    if smoke:
        max_train = int(smoke.get("max_train", 400))
        max_eval = int(smoke.get("max_eval", 200))
        seed = int(smoke.get("seed", 0))
    else:
        max_train = None
        max_eval = None
        seed = int((recipe.get("optim") or {}).get("seed", 0))
    optim = recipe.get("optim") or {}
    epochs = int(optim.get("epochs", 3))
    extract_batch = int(optim.get("batch_size", 32))
    head_batch = int(optim.get("head_batch_size", 8192))
    lr = float(optim.get("lr", 1e-3))
    workers = int(optim.get("num_workers", 8))
    use_cache = bool(optim.get("cache_features", True))

    train_rows = load_cifake_rows("train", max_train, seed)
    eval_rows = load_cifake_rows("test", max_eval, seed + 1)
    print(
        f"n_train={len(train_rows)} n_eval={len(eval_rows)} "
        f"extract_batch={extract_batch} head_batch={head_batch} epochs={epochs}",
        flush=True,
    )

    _enable_fast_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = recipe["backbone"]
    train_cache = cache_path(backbone, "train", len(train_rows), seed, IMAGE_SIZE)
    eval_cache = cache_path(backbone, "test", len(eval_rows), seed + 1, IMAGE_SIZE)
    need_backbone = not (
        use_cache and train_cache.is_file() and eval_cache.is_file()
    )

    model = None
    if need_backbone:
        bb = models_root() / backbone
        model = FrozenLinearProbe(bb).to(device)
        print(f"loaded backbone {bb}", flush=True)
    else:
        print("both feat caches present; skip loading backbone", flush=True)

    t_extract = time.time()
    train_feat, train_y, train_extract_s = _load_or_extract(
        backbone=backbone,
        split="train",
        rows=train_rows,
        seed=seed,
        model=model,
        batch=extract_batch,
        workers=workers,
        device=device,
        use_cache=use_cache,
    )
    eval_feat, eval_y, eval_extract_s = _load_or_extract(
        backbone=backbone,
        split="test",
        rows=eval_rows,
        seed=seed + 1,
        model=model,
        batch=extract_batch,
        workers=workers,
        device=device,
        use_cache=use_cache,
    )
    extract_s = time.time() - t_extract
    if model is not None:
        head = model.head
    else:
        head = nn.Linear(train_feat.size(1), 1).to(device)
    if model is not None:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    head_loader = DataLoader(
        TensorDataset(train_feat, train_y.float()),
        batch_size=head_batch,
        shuffle=True,
        drop_last=False,
    )
    n_batches = max(1, len(head_loader))
    print(
        f"train head on {tuple(train_feat.shape)}  extract={extract_s:.1f}s "
        f"(train={train_extract_s:.1f}s eval={eval_extract_s:.1f}s)",
        flush=True,
    )

    t0 = time.time()
    best_acc = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        running = 0.0
        seen = 0
        for step, (xb, yb) in enumerate(head_loader, 1):
            xb = xb.float().to(device, non_blocking=True)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logit = head(xb).squeeze(-1)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * yb.size(0)
            seen += yb.size(0)
            if step == 1 or step == n_batches:
                print(
                    f"  epoch {epoch}/{epochs}  step {step}/{n_batches}  "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
        ev = evaluate_head(head, eval_feat, eval_y, device, head_batch)
        row = {"epoch": epoch, "train_loss": running / max(1, seen), **ev}
        history.append(row)
        print(
            f"epoch {epoch}/{epochs}  loss={row['train_loss']:.4f}  "
            f"eval_acc={ev['acc']:.3f}  n={ev['n']}",
            flush=True,
        )
        if ev["acc"] > best_acc:
            best_acc = ev["acc"]
            torch.save(
                {"head": head.state_dict(), "backbone": backbone, "recipe": name},
                ckpt_dir / "best.pt",
            )
    train_s = time.time() - t0
    t1 = time.time()
    final = evaluate_head(head, eval_feat, eval_y, device, head_batch)
    eval_s = time.time() - t1
    metrics = {
        "experiment": name,
        "backbone": backbone,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "extract_seconds": round(extract_s, 2),
        "head_seconds": round(train_s, 2),
        "train_seconds": round(extract_s + train_s, 2),
        "eval_seconds": round(eval_s, 2),
        "best_eval_acc": best_acc,
        "final": final,
        "history": history,
    }
    out = art / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}", flush=True)
    print(
        f"done  extract={extract_s:.1f}s  head={train_s:.1f}s  "
        f"eval={eval_s:.1f}s  acc={final['acc']:.3f}",
        flush=True,
    )
    return metrics
