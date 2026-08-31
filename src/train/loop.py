# 2026-08-29, tianqi, extract frozen feats once, then train only the linear head
from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from src.data.cifake import ImagePathDataset, load_cifake_rows
from src.models.linear_probe import FrozenLinearProbe, build_head
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


def _aug_online(recipe: dict) -> bool:
    return bool((recipe.get("aug") or {}).get("online"))


def _aug_expand(recipe: dict) -> bool:
    # 2026-08-29, tianqi, expand=six_ops means each image is trained on all 6 official ops
    return str((recipe.get("aug") or {}).get("expand") or "") in {"six_ops", "all_ops"}
    # end


def _train_source(recipe: dict) -> str:
    train = recipe.get("train") or {}
    # 2026-08-30, tianqi, ablation grids train from a sampled source JSONL
    if train.get("manifest"):
        return "manifest"
    # end
    names = list(train.get("datasets") or [])
    if not names:
        raise SystemExit("recipe.train.datasets is empty")
    return str(names[0])


def _input_mode(recipe: dict) -> str:
    return str(recipe.get("input_mode") or "rgb")


def _head_kind(recipe: dict) -> str:
    return str(recipe.get("head") or "linear")


def _ckpt_payload(head, backbone: str, recipe: dict) -> dict:
    return {
        "head": head.state_dict(),
        "backbone": backbone,
        "recipe": recipe["name"],
        "head_kind": _head_kind(recipe),
        "input_mode": _input_mode(recipe),
    }


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
    rows: list | None,
    seed: int,
    model,
    batch: int,
    workers: int,
    device: torch.device,
    use_cache: bool,
    source: str = "cifake",
    input_mode: str = "rgb",
    dataset=None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    n = len(dataset) if dataset is not None else len(rows or [])
    path = cache_path(
        backbone, split, n, seed, IMAGE_SIZE, source=source, input_mode=input_mode
    )
    t0 = time.time()
    if use_cache:
        cached = load_cached(path)
        if cached is not None:
            feat, y = cached
            print(f"feat cache hit {path}  n={feat.size(0)} d={feat.size(1)}", flush=True)
            return feat, y, time.time() - t0
    feat, y = extract_features(
        model,
        rows or [],
        _tfm(),
        batch,
        workers,
        device,
        tag=split,
        input_mode=input_mode,
        dataset=dataset,
    )
    if use_cache:
        save_cached(path, feat, y)
        print(f"wrote feat cache {path}", flush=True)
    return feat, y, time.time() - t0


def _run_train_online(
    recipe: dict,
    train_rows: list | None,
    eval_rows: list | None,
    seed: int,
    art,
    ckpt_dir,
    source: str = "cifake",
    max_train: int | None = None,
    max_eval: int | None = None,
) -> dict:
    # 2026-08-29, tianqi, online aug cannot cache train feats; eval stays clean
    name = recipe["name"]
    backbone = recipe["backbone"]
    optim = recipe.get("optim") or {}
    epochs = int(optim.get("epochs", 3))
    batch = int(optim.get("batch_size", 32))
    lr = float(optim.get("lr", 1e-3))
    workers = int(optim.get("num_workers", 8))
    expand = _aug_expand(recipe)
    input_mode = _input_mode(recipe)
    head_kind = _head_kind(recipe)
    print(
        f"online aug  source={source} input={input_mode} head={head_kind} "
        f"batch={batch} epochs={epochs} expand={expand}",
        flush=True,
    )
    _enable_fast_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    model = FrozenLinearProbe(models_root() / backbone, head_kind=head_kind).to(device)
    if source == "sid_set":
        from src.data.sid import SidHfDataset

        # 2026-08-31, tianqi, D3 full-SID mix-in: extra disk fakes replace SID FLUX
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
            _tfm(),
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
            _tfm(),
            augment=False,
            seed=seed + 1,
            max_images=max_eval,
            input_mode=input_mode,
        )
        print(
            f"  sid n_train={len(train_ds)} n_eval={len(eval_ds)}",
            flush=True,
        )
        eval_feat, eval_y, eval_extract_s = _load_or_extract(
            backbone=backbone,
            split="validation",
            rows=None,
            seed=seed + 1,
            model=model,
            batch=batch,
            workers=min(workers, 4),
            device=device,
            use_cache=True,
            source="sid_set",
            input_mode=input_mode,
            dataset=eval_ds,
        )
        n_train_src = len(train_ds.keep) + len(train_ds.extra_rows)
        n_eval_src = len(eval_ds.keep)
    else:
        train_ds = ImagePathDataset(
            train_rows,
            _tfm(),
            augment=True,
            expand_ops=expand,
            seed=seed,
            input_mode=input_mode,
        )
        if expand:
            print(f"  six-op expand  n_items={len(train_ds)} views={train_ds.views}", flush=True)
        eval_feat, eval_y, eval_extract_s = _load_or_extract(
            backbone=backbone,
            split="test",
            rows=eval_rows,
            seed=seed + 1,
            model=model,
            batch=batch,
            workers=workers,
            device=device,
            use_cache=True,
            source="cifake",
            input_mode=input_mode,
        )
        n_train_src = len(train_rows or [])
        n_eval_src = len(eval_rows or [])
    print(
        f"  n_train={n_train_src} n_eval={n_eval_src} n_items={len(train_ds)}",
        flush=True,
    )
    loader_kw: dict = dict(
        num_workers=workers if source != "sid_set" else min(workers, 4),
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    if workers > 0:
        loader_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, **loader_kw)
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
                print(
                    f"  epoch {epoch}/{epochs}  step {step}/{n_batches}  "
                    f"loss={loss.item():.4f}",
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
            torch.save(_ckpt_payload(model.head, backbone, recipe), ckpt_dir / "best.pt")
    train_s = time.time() - t0
    final = evaluate_head(model.head, eval_feat, eval_y, device, 8192)
    metrics = {
        "experiment": name,
        "backbone": backbone,
        "online_aug": True,
        "expand_ops": expand,
        "source": source,
        "input_mode": input_mode,
        "head_kind": head_kind,
        "n_train": n_train_src,
        "n_train_items": len(train_ds),
        "n_eval": n_eval_src,
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
    source = _train_source(recipe)
    input_mode = _input_mode(recipe)
    head_kind = _head_kind(recipe)
    if source not in ("cifake", "sid_set", "manifest"):
        raise SystemExit(f"unknown train dataset {source}")
    train_rows = None
    eval_rows = None
    if source == "cifake":
        train_rows = load_cifake_rows("train", max_train, seed)
        eval_rows = load_cifake_rows("test", max_eval, seed + 1)
    # 2026-08-30, tianqi, DATA_ABLATION_PLAN grids: JSONL in, SID val for in-loop pick
    if source == "manifest":
        from src.data.manifest_ds import load_source_manifest_rows

        train_cfg = recipe.get("train") or {}
        train_rows = load_source_manifest_rows(train_cfg["manifest"])
        eval_man = train_cfg.get("eval_manifest")
        if not eval_man:
            raise SystemExit("recipe.train.eval_manifest is required for manifest training")
        eval_rows = load_source_manifest_rows(eval_man)
        if max_train is not None:
            train_rows = train_rows[:max_train]
        if max_eval is not None:
            eval_rows = eval_rows[:max_eval]
    # end
    if _aug_online(recipe):
        return _run_train_online(
            recipe,
            train_rows,
            eval_rows,
            seed,
            art,
            ckpt_dir,
            source=source,
            max_train=max_train,
            max_eval=max_eval,
        )
    # 2026-08-30, tianqi, SID clean linear/mlp/fft uses the same feat-cache head loop as CIFAKE
    train_ds = None
    eval_ds = None
    extract_workers = workers
    if source == "sid_set":
        from src.data.sid import SidHfDataset

        extract_workers = min(workers, 4)
        train_ds = SidHfDataset(
            "train",
            _tfm(),
            augment=False,
            seed=seed,
            max_images=max_train,
            input_mode=input_mode,
        )
        eval_ds = SidHfDataset(
            "validation",
            _tfm(),
            augment=False,
            seed=seed + 1,
            max_images=max_eval,
            input_mode=input_mode,
        )
        n_train = len(train_ds)
        n_eval = len(eval_ds)
        train_split, eval_split = "train", "validation"
        cache_source = source
    else:
        n_train = len(train_rows or [])
        n_eval = len(eval_rows or [])
        train_split, eval_split = "train", "test"
        # 2026-08-30, tianqi, C_pixel and C_flow are both ~9k; cache by recipe name
        cache_source = name if source == "manifest" else source
        if source == "manifest":
            eval_split = "sid_val"
        # end
    print(
        f"n_train={n_train} n_eval={n_eval} "
        f"extract_batch={extract_batch} head_batch={head_batch} epochs={epochs} "
        f"head={head_kind} input={input_mode} source={source}",
        flush=True,
    )

    _enable_fast_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = recipe["backbone"]
    train_cache = cache_path(
        backbone, train_split, n_train, seed, IMAGE_SIZE, source=cache_source, input_mode=input_mode
    )
    eval_cache = cache_path(
        backbone, eval_split, n_eval, seed + 1, IMAGE_SIZE, source=cache_source, input_mode=input_mode
    )
    need_backbone = not (
        use_cache and train_cache.is_file() and eval_cache.is_file()
    )
    # end

    model = None
    if need_backbone:
        bb = models_root() / backbone
        model = FrozenLinearProbe(bb, head_kind=head_kind).to(device)
        print(f"loaded backbone {bb}", flush=True)
    else:
        print("both feat caches present; skip loading backbone", flush=True)

    t_extract = time.time()
    train_feat, train_y, train_extract_s = _load_or_extract(
        backbone=backbone,
        split=train_split,
        rows=train_rows,
        seed=seed,
        model=model,
        batch=extract_batch,
        workers=extract_workers,
        device=device,
        use_cache=use_cache,
        source=cache_source,
        input_mode=input_mode,
        dataset=train_ds,
    )
    eval_feat, eval_y, eval_extract_s = _load_or_extract(
        backbone=backbone,
        split=eval_split,
        rows=eval_rows,
        seed=seed + 1,
        model=model,
        batch=extract_batch,
        workers=extract_workers,
        device=device,
        use_cache=use_cache,
        source=cache_source,
        input_mode=input_mode,
        dataset=eval_ds,
    )
    extract_s = time.time() - t_extract
    if model is not None:
        head = model.head
    else:
        head = build_head(train_feat.size(1), head_kind).to(device)
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
            torch.save(_ckpt_payload(head, backbone, recipe), ckpt_dir / "best.pt")
    train_s = time.time() - t0
    t1 = time.time()
    final = evaluate_head(head, eval_feat, eval_y, device, head_batch)
    eval_s = time.time() - t1
    metrics = {
        "experiment": name,
        "backbone": backbone,
        "source": source,
        "input_mode": input_mode,
        "head_kind": head_kind,
        "n_train": n_train,
        "n_eval": n_eval,
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
