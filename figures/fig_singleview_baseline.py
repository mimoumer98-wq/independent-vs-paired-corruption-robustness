"""Single-view baseline under speckle: 2 panels (TN3K | DDTI), 3 curves
(Paired / single-view / Independent), mean per-image Dice +/- std band over
seeds. A star marks severities where Independent is Holm-significantly above
Paired (from the zero-shot comparison).

Data: evaluate_zeroshot.py output + evaluate_singleview.py output.
Run: python fig_singleview_baseline.py --zero-shot-json ... --singleview-json ...
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

COLS = [("tn3k", "TN3K (in-distribution)"), ("ddti", "DDTI (out-of-distribution)")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-shot-json", type=Path, required=True)
    p.add_argument("--singleview-json", type=Path, required=True)
    args = p.parse_args()

    zs = json.loads(args.zero_shot_json.read_text())
    sv = json.loads(args.singleview_json.read_text())["per_seed_dice"]

    SERIES = [
        ("paired", lambda ds, s: zs[ds]["per_seed"]["speckle"][s]["paired"]),
        ("singleview", lambda ds, s: sv[ds]["speckle"][s]),
        ("independent", lambda ds, s: zs[ds]["per_seed"]["speckle"][s]["independent"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=False)
    for j, (ds, ds_name) in enumerate(COLS):
        ax = axes[j]
        sevs = sorted(sv[ds]["speckle"].keys(), key=float)
        xs = np.array([float(s) for s in sevs])
        for name, getter in SERIES:
            mean = np.array([np.mean(getter(ds, s)) for s in sevs])
            std = np.array([np.std(getter(ds, s), ddof=1) for s in sevs])
            col = style.COLOR[name]
            ax.fill_between(xs, mean - std, mean + std, color=col, alpha=style.BAND_ALPHA,
                             linewidth=0, zorder=style.ZORDER[name])
            ax.plot(xs, mean, color=col, lw=style.LW[name], marker=style.MARKER[name],
                    zorder=style.ZORDER[name] + 2, label=style.LEGEND_LABEL[name] if j == 0 else None)
        avf = zs[ds]["independent_vs_paired"]["speckle"]
        ymax = ax.get_ylim()[1]
        for s in sevs:
            if s in avf and avf[s]["p_holm"] < 0.05:
                ax.annotate("*", (float(s), ymax), ha="center", va="top", fontsize=11,
                            color="black", annotation_clip=False)
        ax.set_title(ds_name)
        ax.set_xlabel("Speckle severity ($\\sigma$)")
        if j == 0:
            ax.set_ylabel("Mean Dice")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_singleview_baseline.{ext}")
    print("wrote fig_singleview_baseline.pdf and .png")


if __name__ == "__main__":
    main()
