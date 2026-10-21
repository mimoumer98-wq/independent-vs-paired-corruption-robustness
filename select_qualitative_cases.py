"""Pre-registered case selection for the qualitative figure (computed before any
image is drawn, so the choice cannot be tuned to flatter either condition).

Criterion: rank the DDTI test images by difficulty = mean over the two
conditions of the CLEAN (severity 0) per-image Dice, using the seed-17
speckle-trained checkpoints. Pick the images at the 25th/50th/75th
rank-percentiles = hard/median/easy. Reports filename + per-condition clean
Dice for each, so the selection is auditable before generating overlays.

Example:
    python select_qualitative_cases.py \\
        --ddti-root /path/to/DDTI \\
        --paired-ckpt experiments/paired/speckle/seed17/best_model.pth \\
        --independent-ckpt experiments/independent/speckle/seed17/best_model.pth \\
        --out results/qualitative_case_selection.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.datasets import load_ddti_paths, TN3KDataset
from src.transforms import build_transforms
from src.utils import get_device
from evaluate_zeroshot import load_model


def per_image_dice(model, loader, device):
    out = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = (torch.sigmoid(model(x)) > 0.5).float()
            yb = (y > 0.5).float()
            inter = (p * yb).sum(dim=(1, 2, 3))
            denom = p.sum(dim=(1, 2, 3)) + yb.sum(dim=(1, 2, 3))
            d = torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
            out.extend(d.cpu().numpy().tolist())
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ddti-root", type=Path, required=True)
    p.add_argument("--paired-ckpt", type=Path, required=True)
    p.add_argument("--independent-ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = Config()
    device = get_device()
    _, tf = build_transforms(cfg)
    imgs, masks = load_ddti_paths(args.ddti_root)
    ds = TN3KDataset(imgs, masks, transform=tf, img_size=cfg.img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"DDTI images: {len(ds)}", flush=True)

    ckpts = {"independent": args.independent_ckpt, "paired": args.paired_ckpt}
    dice = {}
    for cond, pth in ckpts.items():
        m = load_model(pth, device)
        dice[cond] = per_image_dice(m, loader, device)
        print(f"{cond}: mean clean Dice = {dice[cond].mean():.4f}", flush=True)

    difficulty = (dice["independent"] + dice["paired"]) / 2.0
    order = np.argsort(difficulty)  # ascending: hardest first
    N = len(order)
    picks = {"hard (p25)": order[int(round(0.25 * (N - 1)))],
             "median (p50)": order[int(round(0.50 * (N - 1)))],
             "easy (p75)": order[int(round(0.75 * (N - 1)))]}

    print("\n=== PRE-REGISTERED QUALITATIVE-FIGURE CASES (DDTI, seed17, clean Dice) ===")
    rows = []
    for name, idx in picks.items():
        fn = Path(imgs[idx]).name
        rec = {"label": name, "index": int(idx), "file": fn,
               "dice_independent": round(float(dice["independent"][idx]), 4),
               "dice_paired": round(float(dice["paired"][idx]), 4),
               "difficulty_mean": round(float(difficulty[idx]), 4)}
        rows.append(rec)
        print(f"  {name:13s} idx={idx:3d} file={fn:28s} "
              f"independent={rec['dice_independent']:.4f} paired={rec['dice_paired']:.4f} "
              f"mean={rec['difficulty_mean']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "dataset": "ddti", "seed": 17,
        "criterion": "mean clean Dice of independent+paired, rank-percentile 25/50/75",
        "cases": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
