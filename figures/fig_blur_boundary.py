"""The blur x DDTI boundary: on the out-of-distribution DDTI set under
train-matched Gaussian blur, the Paired (rho=-1) model's accuracy is
non-monotonic in severity, while Independent (rho=0) stays high and nearly
flat. Mean per-image Dice vs blur severity, +/- std band over seeds, with
per-seed points; a dotted line marks sigma=1.0.

Data: evaluate_matched.py --corruption blur output, DDTI branch.
Run: python fig_blur_boundary.py --blur-matched-json results/blur_matched_analysis.json
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blur-matched-json", type=Path, required=True)
    args = p.parse_args()

    per_seed = json.loads(args.blur_matched_json.read_text())["ddti"]["per_seed"]
    sevs = sorted(per_seed.keys(), key=float)
    xs = np.array([float(s) for s in sevs])

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.axvline(1.0, color="0.6", ls=":", lw=0.9, zorder=1)

    for cond in ("paired", "independent"):
        col = style.COLOR[cond]
        mean = np.array([np.mean(per_seed[s][cond]) for s in sevs])
        std = np.array([np.std(per_seed[s][cond], ddof=1) for s in sevs])
        ax.fill_between(xs, mean - std, mean + std, color=col, alpha=style.BAND_ALPHA,
                         linewidth=0, zorder=style.ZORDER[cond])
        for xi, s in zip(xs, sevs):
            ys = per_seed[s][cond]
            ax.scatter([xi] * len(ys), ys, s=7, color=col, alpha=0.35,
                       edgecolors="none", zorder=style.ZORDER[cond] + 1)
        ax.plot(xs, mean, color=col, lw=style.LW[cond], marker=style.MARKER[cond],
                zorder=style.ZORDER[cond] + 2, label=style.LEGEND_LABEL[cond])

    peak = np.mean(per_seed["1.0"]["paired"])
    ax.annotate("Paired peak\n($\\sigma=1.0$)", xy=(1.0, peak), xytext=(1.7, peak - 0.028),
                fontsize=7, ha="left", va="top", arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"))

    ax.set_xlabel("Blur severity ($\\sigma$, px)")
    ax.set_ylabel("Mean Dice (DDTI)")
    ax.legend(loc="lower left", ncol=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_blur_boundary.{ext}")
    print("wrote fig_blur_boundary.pdf and .png")


if __name__ == "__main__":
    main()
