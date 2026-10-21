"""Independent (rho=0) vs Paired (rho=-1), zero-shot. 4x2 grid: rows =
speckle / additive Gaussian / Gaussian blur / gamma-speckle; columns = TN3K
(in-distribution) / DDTI (out-of-distribution). Mean per-image Dice vs
severity, +/- std band over seeds; a star marks severities where the
Independent-minus-Paired paired test is Holm-significant (p<0.05).

Data: output of evaluate_zeroshot.py.
Run: python fig_zeroshot_sweep.py --zero-shot-json results/zeroshot_analysis.json
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

ROWS = [("speckle", "Speckle"), ("gaussian", "Additive Gaussian"),
        ("blur", "Gaussian blur"), ("gamma", "Gamma-speckle")]
COLS = [("tn3k", "TN3K (in-distribution)"), ("ddti", "DDTI (out-of-distribution)")]
PANEL = "abcdefgh"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-shot-json", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.zero_shot_json.read_text())

    fig, axes = plt.subplots(4, 2, figsize=(7.0, 9.0), sharex="row")
    k = 0
    for i, (corr, corr_name) in enumerate(ROWS):
        for j, (ds, ds_name) in enumerate(COLS):
            ax = axes[i, j]
            per_seed = data[ds]["per_seed"][corr]
            sevs = sorted(per_seed.keys(), key=float)
            xs = [float(s) for s in sevs]
            for cond in ("paired", "independent"):
                mean = np.array([np.mean(per_seed[s][cond]) for s in sevs])
                std = np.array([np.std(per_seed[s][cond], ddof=1) for s in sevs])
                ax.fill_between(xs, mean - std, mean + std, color=style.COLOR[cond],
                                 alpha=style.BAND_ALPHA, linewidth=0, zorder=style.ZORDER[cond])
                ax.plot(xs, mean, color=style.COLOR[cond], lw=style.LW[cond],
                        marker=style.MARKER[cond], zorder=style.ZORDER[cond] + 1,
                        label=style.LEGEND_LABEL[cond] if (i == 0 and j == 0) else None)
            avf = data[ds]["independent_vs_paired"][corr]
            ymax = ax.get_ylim()[1]
            for s in sevs:
                if s in avf and avf[s]["p_holm"] < 0.05:
                    ax.annotate("*", (float(s), ymax), ha="center", va="top",
                                fontsize=11, color="black", annotation_clip=False)
            if i == 0:
                ax.set_title(ds_name)
            if j == 0:
                ax.set_ylabel(f"{corr_name}\nMean Dice")
            if i == len(ROWS) - 1:
                ax.set_xlabel("Severity")
            ax.text(0.02, 0.04, f"({PANEL[k]})", transform=ax.transAxes,
                    ha="left", va="bottom", fontweight="bold")
            k += 1

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_zeroshot_sweep.{ext}")
    print("wrote fig_zeroshot_sweep.pdf and .png")


if __name__ == "__main__":
    main()
