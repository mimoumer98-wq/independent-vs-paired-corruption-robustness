"""Zero-shot vs train-matched: the Independent-minus-Paired gap. For the
corruption families that were re-trained (additive Gaussian, Gaussian blur) x
two datasets, plots the gap under zero-shot evaluation (speckle-trained
models; grey, open, dashed) and after matched re-training (blue, filled,
solid). An arrow connects the two at each severity; a star marks severities
where the train-matched gap is Holm-significant. Y-axis starts at 0.

Data: outputs of evaluate_zeroshot.py and evaluate_matched.py (gaussian, blur).
Run: python fig_matched_vs_zeroshot_gap.py --zero-shot-json ... \\
        --gaussian-matched-json ... --blur-matched-json ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style

HERE = Path(__file__).resolve().parent
style.apply()

ROWS = [("gaussian", "Additive Gaussian"), ("blur", "Gaussian blur")]
COLS = [("tn3k", "TN3K (in-distribution)"), ("ddti", "DDTI (out-of-distribution)")]
C_ZS, C_TM = "#7f7f7f", "#0072B2"
PANEL = "abcd"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-shot-json", type=Path, required=True)
    p.add_argument("--gaussian-matched-json", type=Path, required=True)
    p.add_argument("--blur-matched-json", type=Path, required=True)
    args = p.parse_args()

    p0 = json.loads(args.zero_shot_json.read_text())
    TM = {"gaussian": json.loads(args.gaussian_matched_json.read_text()),
          "blur": json.loads(args.blur_matched_json.read_text())}

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    k = 0
    for i, (corr, corr_name) in enumerate(ROWS):
        for j, (ds, ds_name) in enumerate(COLS):
            ax = axes[i, j]
            zs = p0[ds]["independent_vs_paired"][corr]
            tm = TM[corr][ds]["independent_vs_paired"]
            sevs = sorted(tm.keys(), key=float)
            xs = np.array([float(s) for s in sevs])
            zg = np.array([zs[s]["mean_diff"] for s in sevs])
            tg = np.array([tm[s]["mean_diff"] for s in sevs])
            sig = [tm[s]["p_holm"] < 0.05 for s in sevs]

            ax.axhline(0, color="0.8", lw=0.8, zorder=1)
            for x, a, b in zip(xs, zg, tg):
                ax.annotate("", xy=(x, b), xytext=(x, a),
                            arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                            color=("#1b7837" if b >= a else "#c2410c"), alpha=0.8),
                            zorder=2)
            ax.plot(xs, zg, color=C_ZS, lw=1.3, ls="--", marker="o",
                    markerfacecolor="white", markeredgecolor=C_ZS, zorder=3,
                    label="Zero-shot" if (i == 0 and j == 0) else None)
            ax.plot(xs, tg, color=C_TM, lw=2.3, marker="o", zorder=4,
                    label="Train-matched" if (i == 0 and j == 0) else None)
            span = max(tg.max(), zg.max()) - min(0, tg.min())
            for x, b, s in zip(xs, tg, sig):
                if s:
                    ax.annotate("*", (x, b + 0.03 * span), ha="center", va="bottom", fontsize=11, zorder=5)

            lo = min(0.0, zg.min(), tg.min())
            hi = max(zg.max(), tg.max())
            pad = 0.12 * (hi - lo)
            ax.set_ylim(lo - 0.02 * (hi - lo), hi + pad)
            if i == 0:
                ax.set_title(ds_name)
            if j == 0:
                ax.set_ylabel(f"{corr_name}\nIndep. $-$ Paired Dice")
            if i == len(ROWS) - 1:
                ax.set_xlabel("Severity")
            ax.text(0.03, 0.95, f"({PANEL[k]})", transform=ax.transAxes, ha="left", va="top", fontweight="bold")
            k += 1

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_matched_vs_zeroshot_gap.{ext}")
    print("wrote fig_matched_vs_zeroshot_gap.pdf and .png")


if __name__ == "__main__":
    main()
