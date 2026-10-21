"""Train-matched evaluation: Paired/Independent checkpoints trained UNDER a
given corruption, evaluated under a severity sweep of that SAME corruption,
on TN3K + DDTI. Mirrors evaluate_zeroshot.py's protocol/metrics exactly, so a
--zero-shot-json can optionally be passed to print the zero-shot numbers for
the same corruption side by side (does training under the corruption recover
or exceed what zero-shot transfer only hinted at?).

Example:
    python evaluate_matched.py --corruption blur \\
        --data-root /path/to/data --ddti-root /path/to/DDTI \\
        --paired-ckpt-dir experiments/paired/blur \\
        --independent-ckpt-dir experiments/independent/blur \\
        --out results/blur_matched_analysis.json \\
        --zero-shot-json results/zeroshot_analysis.json
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
from evaluate_zeroshot import load_model, dice_at, holm, dz


def evaluate_dataset(name, loader, device, cfg, corruption, ckpt_dirs):
    sev_list = SEVERITIES[corruption]
    raw = {"paired": {}, "independent": {}}
    for cond, cdir in ckpt_dirs.items():
        models = [load_model(cdir / f"seed{s}" / "best_model.pth", device) for s in SEEDS_ORDER]
        for sev in sev_list:
            raw[cond][sev] = [dice_at(m, loader, device, REGISTRY[corruption], sev, cfg) for m in models]
            print(f"[{name}][{cond}] {corruption} sev={sev}: mean_dice={np.mean(raw[cond][sev]):.4f}", flush=True)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = {"means": {}, "independent_vs_paired": {},
           "per_seed": {str(sev): {"paired": raw["paired"][sev], "independent": raw["independent"][sev]}
                        for sev in sev_list}}
    for sev in sev_list:
        out["means"][str(sev)] = {"paired": float(np.mean(raw["paired"][sev])),
                                   "independent": float(np.mean(raw["independent"][sev]))}
    raw_p, per = {}, {}
    for sev in [s for s in sev_list if s > 0]:
        a = np.array(raw["paired"][sev])
        f = np.array(raw["independent"][sev])
        d = f - a
        t, p = stats.ttest_1samp(d, 0.0)
        per[str(sev)] = {"mean_diff": float(d.mean()), "cohens_d": dz(d),
                          "p_value": float(p), "n_independent_wins": int((d > 0).sum())}
        raw_p[str(sev)] = float(p)
    hp = holm(raw_p)
    for sev in [s for s in sev_list if s > 0]:
        per[str(sev)]["p_holm"] = hp[str(sev)]
    out["independent_vs_paired"] = per
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corruption", choices=["speckle", "gaussian", "blur", "gamma"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--ddti-root", type=Path, required=True)
    p.add_argument("--paired-ckpt-dir", type=Path, required=True)
    p.add_argument("--independent-ckpt-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--zero-shot-json", type=Path, default=None,
                    help="Optional evaluate_zeroshot.py output, printed side by side")
    args = p.parse_args()

    cfg = Config()
    device = get_device()
    print(f"Device: {device}", flush=True)
    _, test_tf = build_transforms(cfg)

    paths = load_tn3k_paths(args.data_root)
    tn3k = TN3KDataset(paths["test"]["images"], paths["test"]["masks"], transform=test_tf, img_size=cfg.img_size)
    tn3k_loader = DataLoader(tn3k, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    imgs, masks = load_ddti_paths(args.ddti_root)
    ddti = TN3KDataset(imgs, masks, transform=test_tf, img_size=cfg.img_size)
    ddti_loader = DataLoader(ddti, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    ckpt_dirs = {"paired": args.paired_ckpt_dir, "independent": args.independent_ckpt_dir}
    results = {}
    for name, loader in [("tn3k", tn3k_loader), ("ddti", ddti_loader)]:
        results[name] = evaluate_dataset(name, loader, device, cfg, args.corruption, ckpt_dirs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    zs = json.loads(args.zero_shot_json.read_text()) if args.zero_shot_json else None
    for name in ["tn3k", "ddti"]:
        print(f"\n{name.upper()}  {args.corruption}: train-matched vs zero-shot")
        print("  mean_diff = independent - paired (+ => independent better); * = Holm-significant")
        for sev in SEVERITIES[args.corruption]:
            mm = results[name]["means"][str(sev)]
            line = f"  sev={sev:<5} matched: paired={mm['paired']:.4f} independent={mm['independent']:.4f}"
            if str(sev) in results[name]["independent_vs_paired"]:
                r = results[name]["independent_vs_paired"][str(sev)]
                sig = "*" if r["p_holm"] < 0.05 else " "
                line += f"  diff={r['mean_diff']:+.4f} d_z={r['cohens_d']:+.2f} p_holm={r['p_holm']:.3f}{sig}"
            if zs and str(sev) in zs[name]["independent_vs_paired"].get(args.corruption, {}):
                z = zs[name]["independent_vs_paired"][args.corruption][str(sev)]
                line += f"   | zero-shot diff={z['mean_diff']:+.4f} p_holm={z['p_holm']:.3f}"
            print(line)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
