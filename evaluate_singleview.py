"""Zero-shot evaluation of the single-view baseline, across the four corruption
families x two datasets, compared to Independent (rho=0) and Paired (rho=-1)
using the per-seed arrays saved by evaluate_zeroshot.py.

Paired t-tests (scipy.stats.ttest_rel), Holm-corrected within each corruption's
own severity family per dataset: independent-vs-singleview (d = independent -
singleview) and singleview-vs-paired (d = singleview - paired).

Example:
    python evaluate_singleview.py \\
        --data-root /path/to/data --ddti-root /path/to/DDTI \\
        --singleview-ckpt-dir experiments/singleview/speckle \\
        --zero-shot-json results/zeroshot_analysis.json \\
        --out results/singleview_analysis.json
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
from src.utils import get_device
from src.constants import SEEDS_ORDER
from src.corruptions import REGISTRY, SEVERITIES
from evaluate_zeroshot import load_model, dice_at, holm


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--ddti-root", type=Path, required=True)
    p.add_argument("--singleview-ckpt-dir", type=Path, required=True, help="Directory with seed*/best_model.pth")
    p.add_argument("--zero-shot-json", type=Path, required=True, help="evaluate_zeroshot.py output (needs its 'per_seed' field)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = Config()
    device = get_device()
    print(f"Device: {device}", flush=True)
    _, test_tf = build_transforms(cfg)

    paths = load_tn3k_paths(args.data_root)
    tn3k = TN3KDataset(paths["test"]["images"], paths["test"]["masks"], transform=test_tf, img_size=cfg.img_size)
    imgs, masks = load_ddti_paths(args.ddti_root)
    ddti = TN3KDataset(imgs, masks, transform=test_tf, img_size=cfg.img_size)
    loaders = {"tn3k": DataLoader(tn3k, batch_size=cfg.batch_size, shuffle=False, num_workers=0),
               "ddti": DataLoader(ddti, batch_size=cfg.batch_size, shuffle=False, num_workers=0)}
    print(f"TN3K {len(tn3k)} | DDTI {len(ddti)}", flush=True)

    models = [load_model(args.singleview_ckpt_dir / f"seed{s}" / "best_model.pth", device) for s in SEEDS_ORDER]
    per = {ds: {c: {} for c in REGISTRY} for ds in loaders}
    for ds, loader in loaders.items():
        for corr in REGISTRY:
            for sev in SEVERITIES[corr]:
                vals = [float(dice_at(m, loader, device, REGISTRY[corr], sev, cfg)) for m in models]
                per[ds][corr][str(sev)] = vals
                print(f"[SV][{ds}][{corr}] sev={sev}: mean={np.mean(vals):.4f}", flush=True)

    zs = json.loads(args.zero_shot_json.read_text())
    out = {}
    for ds in loaders:
        out[ds] = {}
        for corr in REGISTRY:
            sevs = [s for s in SEVERITIES[corr] if s > 0]
            raw_i, raw_p, rows = {}, {}, {}
            for sev in sevs:
                sv = np.array(per[ds][corr][str(sev)])
                ind = np.array(zs[ds]["per_seed"][corr][str(sev)]["independent"])
                pr = np.array(zs[ds]["per_seed"][corr][str(sev)]["paired"])
                _, p_i = stats.ttest_rel(ind, sv)  # independent - singleview
                _, p_p = stats.ttest_rel(sv, pr)   # singleview - paired
                raw_i[str(sev)] = float(p_i)
                raw_p[str(sev)] = float(p_p)
                rows[str(sev)] = {"singleview": float(sv.mean()), "independent": float(ind.mean()),
                                   "paired": float(pr.mean()),
                                   "diff_independent_minus_singleview": float((ind - sv).mean()),
                                   "diff_singleview_minus_paired": float((sv - pr).mean())}
            h_i, h_p = holm(raw_i), holm(raw_p)
            for sev in sevs:
                rows[str(sev)]["p_holm_independent_vs_singleview"] = h_i[str(sev)]
                rows[str(sev)]["p_holm_singleview_vs_paired"] = h_p[str(sev)]
            out[ds][corr] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"per_seed_dice": per, "comparison": out}, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
