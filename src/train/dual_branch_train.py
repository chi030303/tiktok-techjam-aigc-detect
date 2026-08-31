# 2026-08-31, tianqi, D3 dual-branch: frozen CLIP-B + highpass CNN on SID mix
"""Train DualBranchProbe. CLIP frozen; freq CNN + head train. Same D3 mixin as frozen mix."""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.freq import highpass_residual
from src.data.sid import SidHfDataset
from src.models.dual_branch_probe import DualBranchProbe
from src.paths import artifact_dir, models_root
from src.recipe import validate
from src.train.loop import _aug_expand, _input_mode

# end

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pil_identity(img):
    return img


def _image_size(recipe: dict) -> int:
    return int(recipe.get("image_size") or 224)


def _rgb_tfm(size: int):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _freq_tfm(size: int):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])


class DualSidDataset(Dataset):
    """Highpass of the same (possibly augmented) PIL the RGB branch sees."""

    def __init__(self, sid: SidHfDataset, image_size: int):
        self.sid = sid
        self.rgb_tfm = _rgb_tfm(image_size)
        self.freq_tfm = _freq_tfm(image_size)

    def __len__(self):
        return len(self.sid)

    def set_epoch(self, epoch: int) -> None:
        self.sid.set_epoch(epoch)

    def __getitem__(self, idx: int):
        img, y, tag = self.sid[idx]
        return self.rgb_tfm(img), self.freq_tfm(highpass_residual(img)), y, tag


@torch.no_grad()
def _evaluate_full(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    n = 0
    correct = 0
    tp = fp = fn = 0
    for rgb, freq, y, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        freq = freq.to(device, non_blocking=True)
        y = y.to(device)
        logit = model(rgb, freq)
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

    db = recipe.get("dual_branch") or {}
    freq_dim = int(db.get("freq_dim", 128))
    backbone_name = recipe["backbone"]
    head_kind = str(recipe.get("head") or "linear")
    input_mode = _input_mode(recipe)
    image_size = _image_size(recipe)
    expand = _aug_expand(recipe)

    optim = recipe.get("optim") or {}
    epochs = int(optim.get("epochs", 3))
    batch = int(optim.get("batch_size", 64))
    workers = int(optim.get("num_workers", 4))
    seed = int(optim.get("seed", 0))
    lr = float(optim.get("lr", 1e-3))

    smoke = recipe.get("smoke") or {}
    max_train = smoke.get("max_train")
    max_eval = smoke.get("max_eval")

    print(
        f"dual-branch  freq_dim={freq_dim}  lr={lr}  batch={batch}  epochs={epochs}  "
        f"image_size={image_size}  backbone={backbone_name}",
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

    extra_rows = None
    train_cfg = recipe.get("train") or {}
    mixin = train_cfg.get("mixin_manifest")
    if mixin:
        from src.data.manifest_ds import load_mixin_rows

        extra_rows = load_mixin_rows(mixin)
        print(f"  mixin_manifest={mixin} n_extra={len(extra_rows)}", flush=True)
    replace_sid = bool(train_cfg.get("replace_sid_fakes", True))

    train_sid = SidHfDataset(
        "train",
        _pil_identity,
        augment=True,
        expand_ops=expand,
        seed=seed,
        max_images=max_train,
        input_mode=input_mode,
        extra_rows=extra_rows,
        replace_sid_fakes=replace_sid if extra_rows else False,
    )
    eval_sid = SidHfDataset(
        "validation",
        _pil_identity,
        augment=False,
        seed=seed + 1,
        max_images=max_eval,
        input_mode=input_mode,
    )
    train_ds = DualSidDataset(train_sid, image_size)
    eval_ds = DualSidDataset(eval_sid, image_size)
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
        list(model.freq_cnn.parameters()) + list(model.head.parameters()),
        lr=lr,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    amp = device.type == "cuda"
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
            "input_mode": "rgb",
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
        for step, (rgb, freq, y, _) in enumerate(train_loader, 1):
            rgb = rgb.to(device, non_blocking=True)
            freq = freq.to(device, non_blocking=True)
            y = y.float().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logit = model(rgb, freq)
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
