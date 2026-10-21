"""Qualitative figure: segmentation contours under increasing speckle, for the
pre-registered cases from select_qualitative_cases.py. Rows = easy / median /
hard; columns = increasing speckle severity. Contours only (ground truth
dashed yellow, Independent solid blue, Paired solid vermillion) overlaid on a
zoomed, ground-truth-bbox-centred crop, common scale across all panels.

Run: python fig_qualitative.py --ddti-root ... \\
        --paired-ckpt ... --independent-ckpt ... --selection-json ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style
from src.config import Config
from src.datasets import TN3KDataset, load_ddti_paths
from src.transforms import build_transforms
from src.utils import get_device, denorm_to_01
from src.corruptions import REGISTRY
from src.constants import NOISE_SEED
from evaluate_zeroshot import load_model

HERE = Path(__file__).resolve().parent
style.apply()

SEV = [0.0, 0.2, 0.3, 0.4]
GT_COLOR = "#F0E442"  # Okabe-Ito yellow, dashed
ROW_ORDER = ["easy (p75)", "median (p50)", "hard (p25)"]
ROW_LABEL = {"easy (p75)": "Easy", "median (p50)": "Median", "hard (p25)": "Hard"}


def dice(pred, gt):
    p = (pred > 0.5).astype(np.float32)
    g = (gt > 0.5).astype(np.float32)
    d = p.sum() + g.sum()
    return 1.0 if d == 0 else float(2 * (p * g).sum() / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddti-root", type=Path, required=True)
    ap.add_argument("--paired-ckpt", type=Path, required=True)
    ap.add_argument("--independent-ckpt", type=Path, required=True)
    ap.add_argument("--selection-json", type=Path, required=True)
    args = ap.parse_args()

    cfg = Config()
    device = get_device()
    _, tf = build_transforms(cfg)
    imgs, masks = load_ddti_paths(args.ddti_root)
    name2idx = {Path(p).name: i for i, p in enumerate(imgs)}
    ds = TN3KDataset(imgs, masks, transform=tf, img_size=cfg.img_size)
    models = {"independent": load_model(args.independent_ckpt, device),
              "paired": load_model(args.paired_ckpt, device)}

    sel = json.loads(args.selection_json.read_text())["cases"]

    def bbox_dim(mask):
        ys, xs = np.where(mask > 0.5)
        return ys.min(), ys.max(), xs.min(), xs.max()

    rows = []
    for key in ROW_ORDER:
        rec = next(r for r in sel if r["label"] == key)
        idx = name2idx[rec["file"]]
        x, y = ds[idx]
        gt = y.squeeze().cpu().numpy()
        r0, r1, c0, c1 = bbox_dim(gt)
        rows.append({"key": key, "x": x.unsqueeze(0).to(device), "gt": gt,
                     "cx": (c0 + c1) / 2.0, "cy": (r0 + r1) / 2.0, "dim": max(r1 - r0, c1 - c0)})
    H, W = rows[0]["gt"].shape
    side = min(H, W, int(max(r["dim"] for r in rows) * 1.30) + 1)

    torch.manual_seed(NOISE_SEED)
    fig, axes = plt.subplots(3, len(SEV), figsize=(7.2, 5.7))
    for i, row in enumerate(rows):
        x, gt = row["x"], row["gt"]
        cx = min(max(row["cx"], side / 2), W - side / 2)
        cy = min(max(row["cy"], side / 2), H - side / 2)
        for j, sev in enumerate(SEV):
            ax = axes[i, j]
            xc = REGISTRY["speckle"](x, sev, cfg.input_norm_mean, cfg.input_norm_std) if sev > 0 else x
            with torch.no_grad():
                preds = {k: torch.sigmoid(m(xc)).squeeze().cpu().numpy() for k, m in models.items()}
            disp = denorm_to_01(xc, cfg.input_norm_mean, cfg.input_norm_std).squeeze().cpu().numpy()
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            ax.contour(gt, levels=[0.5], colors=[GT_COLOR], linewidths=1.4, linestyles="--")
            ax.contour(preds["independent"], levels=[0.5], colors=[style.COLOR["independent"]], linewidths=1.6)
            ax.contour(preds["paired"], levels=[0.5], colors=[style.COLOR["paired"]], linewidths=1.6)
            ax.set_xlim(cx - side / 2, cx + side / 2)
            ax.set_ylim(cy + side / 2, cy - side / 2)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
            dI, dP = dice(preds["independent"], gt), dice(preds["paired"], gt)
            ax.text(0.035, 0.035, f"I {dI:.2f}  P {dP:.2f}", transform=ax.transAxes,
                    fontsize=6.5, va="bottom", ha="left", color="white",
                    bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.6))
            if i == 0:
                ax.set_title("clean" if sev == 0 else f"$\\sigma={sev}$", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(ROW_LABEL[row["key"]], fontsize=9)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=GT_COLOR, ls="--", lw=1.6, label="Ground truth"),
               Line2D([0], [0], color=style.COLOR["independent"], lw=1.9, label=style.LEGEND_LABEL["independent"]),
               Line2D([0], [0], color=style.COLOR["paired"], lw=1.9, label=style.LEGEND_LABEL["paired"])]
    fig.subplots_adjust(left=0.055, right=0.995, top=0.905, bottom=0.015, wspace=0.05, hspace=0.07)
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    fig.savefig(HERE / "fig_qualitative.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(HERE / "fig_qualitative.png", dpi=400, bbox_inches="tight", pad_inches=0.06)
    print(f"wrote fig_qualitative.pdf and .png | common crop side = {side}px (max GT dim)")


if __name__ == "__main__":
    main()
