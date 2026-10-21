"""Single-view baseline: the conventional noise-augmentation setup --- ONE view
per step, corrupted with probability 0.5 (one Bernoulli coin), ONE segmentation
loss (one forward/backward pass). This is the 1-view counterpart of the two-view
conditions; Independent (rho=0) is its budget-doubled 2-view generalization.

Example:
    python train_singleview.py --corruption speckle --train-severity 0.15 \\
        --seed 17 --data-root /path/to/data --ckpt-dir experiments/singleview/speckle/seed17
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.transforms import build_transforms
from src.engine import evaluate_full
from src.datasets import TN3KDataset, load_tn3k_paths, split_train_val
from src.losses import DiceBCELoss
from src.model import ResUNet
from src.utils import get_device, set_seed
from src.corruptions import REGISTRY


def _sanity_check_masks(dataset: TN3KDataset):
    mask = dataset[0][1]
    mmin, mmax = float(mask.min()), float(mask.max())
    if mmin < -1e-6 or mmax > 1.0 + 1e-6:
        raise ValueError(f"Mask values out of [0,1]. min={mmin:.4f}, max={mmax:.4f}")


def _build_single_view(images, corrupt_fn, severity, mean, std):
    """One view: each sample corrupted with probability 0.5 (one Bernoulli coin)."""
    B = images.shape[0]
    m = (torch.rand(B, 1, 1, 1, device=images.device) < 0.5).float()
    return images * (1 - m) + corrupt_fn(images, severity, mean, std) * m


def run_one_seed(cfg: Config, seed: int, corruption: str, train_severity: float):
    corrupt_fn = REGISTRY[corruption]
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = get_device()
    print(f"Run: single-view | {corruption} train_severity={train_severity} | seed={seed}", flush=True)

    paths = load_tn3k_paths(cfg.data_dir)
    train_tf, test_tf = build_transforms(cfg)
    (tr_i, tr_m), (va_i, va_m) = split_train_val(paths["train"]["images"], paths["train"]["masks"], val_ratio=0.2, seed=seed)
    train_ds = TN3KDataset(tr_i, tr_m, transform=train_tf, img_size=cfg.img_size)
    val_ds = TN3KDataset(va_i, va_m, transform=test_tf, img_size=cfg.img_size)
    test_ds = TN3KDataset(paths["test"]["images"], paths["test"]["masks"], transform=test_tf, img_size=cfg.img_size)
    _sanity_check_masks(train_ds)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    print(f"DATA SPLIT: Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}", flush=True)

    model = ResUNet(cfg.in_channels, cfg.num_classes, pretrained=True, dropout_p=0.2).to(device)
    criterion = DiceBCELoss(dice_weight=0.7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr, epochs=cfg.epochs,
                                                      steps_per_epoch=len(train_loader), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_path = cfg.ckpt_dir / "best_model.pth"
    best_dice, best_epoch, no_improve = -1.0, 0, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                v = _build_single_view(images, corrupt_fn, train_severity, cfg.input_norm_mean, cfg.input_norm_std)
                loss = criterion(model(v), masks)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.item())

        train_loss = total / len(train_loader)
        val = evaluate_full(model, val_loader, device, cfg.threshold, criterion=criterion)
        print(f"Epoch {epoch:2d}/{cfg.epochs} | train_loss={train_loss:.4f} | val_dice={val['dice_mean']:.4f}", flush=True)

        if val["dice_mean"] > best_dice:
            best_dice, best_epoch, no_improve = val["dice_mean"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "condition": "singleview",
                        "corruption": corruption, "train_severity": train_severity, "metrics": val}, str(best_path))
        else:
            no_improve += 1
            if no_improve >= cfg.early_stopping_patience:
                print(f"EARLY STOP at epoch {epoch} (best_epoch={best_epoch}, best_dice={best_dice:.4f})", flush=True)
                break

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    tm = evaluate_full(model, test_loader, device, cfg.threshold, criterion=criterion)
    print(f"FINAL TEST (single-view, {corruption}): dice_mean={tm['dice_mean']:.4f}", flush=True)
    return {"best_epoch": best_epoch, "test_metrics": tm}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corruption", choices=["speckle", "gaussian", "blur", "gamma"], default="speckle")
    p.add_argument("--train-severity", type=float, default=0.15)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = Config()
    cfg.data_dir = args.data_root
    cfg.ckpt_dir = args.ckpt_dir
    cfg.results_dir = args.results_dir or args.ckpt_dir
    cfg.compute_train_full_metrics = False

    run_one_seed(cfg, args.seed, args.corruption, args.train_severity)


if __name__ == "__main__":
    main()
