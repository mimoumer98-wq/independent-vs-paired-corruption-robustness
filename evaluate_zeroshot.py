"""Zero-shot corruption-transfer evaluation: loads the SPECKLE-trained Paired
(rho=-1) and Independent (rho=0) checkpoints and evaluates them under FOUR
corruptions (speckle, gaussian, blur, gamma) across a severity sweep, on TN3K
(in-distribution test split) and DDTI (external, zero-shot dataset).

Dice only. Fixed noise realization per pass (constants.NOISE_SEED). Paired
Independent-vs-Paired stats (mean_diff = independent - paired; Cohen's d_z;
Holm correction within each corruption's own severity family, per dataset).

Example:
    python evaluate_zeroshot.py \\
        --data-root /path/to/data --ddti-root /path/to/DDTI \\
        --paired-ckpt-dir experiments/paired/speckle \\
        --independent-ckpt-dir experiments/independent/speckle \\
        --out results/zeroshot_analysis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats

from src.config import Config
from src.datasets import load_tn3k_paths, load_ddti_paths, TN3KDataset
from src.transforms import build_transforms
from src.metrics import empty_counts, update_confusion_counts, evaluate_mean_per_image
from src.model import ResUNet
from src.utils import get_device
from src.constants import NOISE_SEED, SEEDS_ORDER
from src.corruptions import REGISTRY, SEVERITIES


def load_model(ckpt_path, device):
    m = ResUNet(in_channels=1, out_channels=1, pretrained=False).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    m.eval()
    return m


@torch.no_grad()
def dice_at(model, loader, device, corrupt_fn, severity, cfg):
    model.eval()
    torch.manual_seed(NOISE_SEED)
    torch.cuda.manual_seed_all(NOISE_SEED)
    counts = empty_counts()
    probs_all, masks_all = [], []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if severity > 0:
            images = corrupt_fn(images, severity, cfg.input_norm_mean, cfg.input_norm_std)
        probs = torch.sigmoid(model(images))
        update_confusion_counts(probs, masks, threshold=cfg.threshold, counts=counts)
        probs_all.append(probs.cpu())
        masks_all.append(masks.cpu())
    mean_metrics = evaluate_mean_per_image(torch.cat(probs_all), torch.cat(masks_all), threshold=cfg.threshold)
    return float(mean_metrics["dice_mean"])


def holm(pv):
    items = sorted(pv.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, rm = {}, 0.0
    for i, (n, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        rm = max(rm, a)
        adj[n] = rm
    return adj


def dz(v):
    v = np.asarray(v, float)
    sd = v.std(ddof=1)
    return float(v.mean() / sd) if sd > 1e-12 else float("nan")


def evaluate_dataset(name, loader, device, cfg, ckpt_dirs):
    raw = {"paired": {}, "independent": {}}
    for cond, cdir in ckpt_dirs.items():
        models = [load_model(cdir / f"seed{s}" / "best_model.pth", device) for s in SEEDS_ORDER]
        for corr in REGISTRY:
            raw[cond].setdefault(corr, {})
            for sev in SEVERITIES[corr]:
                vals = [dice_at(m, loader, device, REGISTRY[corr], sev, cfg) for m in models]
                raw[cond][corr][sev] = vals
                print(f"[{name}][{cond}][{corr}] sev={sev}: mean_dice={np.mean(vals):.4f}", flush=True)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_seed = {corr: {str(sev): {"paired": raw["paired"][corr][sev], "independent": raw["independent"][corr][sev]}
                        for sev in SEVERITIES[corr]} for corr in REGISTRY}

    out = {"means": {}, "independent_vs_paired": {}}
    for corr in REGISTRY:
        out["means"][corr] = {str(sev): {"paired": float(np.mean(raw["paired"][corr][sev])),
                                          "independent": float(np.mean(raw["independent"][corr][sev]))}
                               for sev in SEVERITIES[corr]}
        sevs = [s for s in SEVERITIES[corr] if s > 0]
        raw_p, per = {}, {}
        for sev in sevs:
            a = np.array(raw["paired"][corr][sev])
            f = np.array(raw["independent"][corr][sev])
            d = f - a
            t, p = stats.ttest_1samp(d, 0.0)
            per[str(sev)] = {"mean_diff": float(d.mean()), "cohens_d": dz(d),
                              "p_value": float(p), "n_independent_wins": int((d > 0).sum())}
            raw_p[str(sev)] = float(p)
        hp = holm(raw_p)
        for sev in sevs:
            per[str(sev)]["p_holm"] = hp[str(sev)]
        out["independent_vs_paired"][corr] = per
    out["per_seed"] = per_seed
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True, help="Directory containing TN3K/{train,test}/{images,masks}")
    p.add_argument("--ddti-root", type=Path, required=True, help="Directory containing DDTI/{images,masks}")
    p.add_argument("--paired-ckpt-dir", type=Path, required=True, help="Directory with seed*/best_model.pth")
    p.add_argument("--independent-ckpt-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = Config()
    device = get_device()
    print(f"Device: {device}", flush=True)
    _, test_tf = build_transforms(cfg)

    paths = load_tn3k_paths(args.data_root)
    tn3k = TN3KDataset(paths["test"]["images"], paths["test"]["masks"], transform=test_tf, img_size=cfg.img_size)
    tn3k_loader = DataLoader(tn3k, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"TN3K: {len(tn3k)} images", flush=True)

    imgs, masks = load_ddti_paths(args.ddti_root)
    ddti = TN3KDataset(imgs, masks, transform=test_tf, img_size=cfg.img_size)
    ddti_loader = DataLoader(ddti, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"DDTI: {len(ddti)} images", flush=True)

    ckpt_dirs = {"paired": args.paired_ckpt_dir, "independent": args.independent_ckpt_dir}
    results = {}
    for name, loader in [("tn3k", tn3k_loader), ("ddti", ddti_loader)]:
        results[name] = evaluate_dataset(name, loader, device, cfg, ckpt_dirs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out}")
    print("GUARD: this JSON's 'means' are the arithmetic mean of its own 'per_seed' "
          "arrays by construction (single evaluation pass; no separate regeneration step).")


if __name__ == "__main__":
    main()
