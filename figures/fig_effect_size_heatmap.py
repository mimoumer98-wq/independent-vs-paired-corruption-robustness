"""Effect-size landscape: Cohen's d_z of the Independent-minus-Paired
difference, for every corruption family (rows) across severity (columns), on
TN3K and DDTI. Colour = d_z (viridis); a white/black star marks cells
Holm-significant at p<0.05.

Data: output of evaluate_zeroshot.py.
Run: python fig_effect_size_heatmap.py --zero-shot-json results/zeroshot_analysis.json
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

ROWS = [("speckle", "Speckle"), ("gaussian", "Add. Gaussian"),
        ("blur", "Gaussian blur"), ("gamma", "Gamma-speckle")]
COLS = [("tn3k", "TN3K  (in-distribution)"), ("ddti", "DDTI  (out-of-distribution)")]


def matrices(st, ds):
    dz, sig = [], []
    for corr, _ in ROWS:
        avf = st[ds]["independent_vs_paired"][corr]
        sevs = sorted(avf.keys(), key=float)
        dz.append([avf[s]["cohens_d"] for s in sevs])
        sig.append([avf[s]["p_holm"] < 0.05 for s in sevs])
    return np.array(dz), np.array(sig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-shot-json", type=Path, required=True)
    args = p.parse_args()
    st = json.loads(args.zero_shot_json.read_text())

    vmax = max(np.nanmax(matrices(st, ds)[0]) for ds, _ in COLS)
    vmin = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    im = None
    for j, (ds, ds_name) in enumerate(COLS):
        ax = axes[j]
        dz, sig = matrices(st, ds)
        im = ax.imshow(dz, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(dz.shape[1]))
        ax.set_xticklabels([f"s{t + 1}" for t in range(dz.shape[1])])
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([n for _, n in ROWS] if j == 0 else [""] * len(ROWS))
        ax.set_title(ds_name, fontsize=8.5)
        ax.set_xlabel("increasing severity $\\rightarrow$")
        for r in range(dz.shape[0]):
            for c in range(dz.shape[1]):
                val = dz[r, c]
                txt = f"{val:.1f}" + ("$^*$" if sig[r, c] else "")
                ax.text(c, r, txt, ha="center", va="center", fontsize=6.5,
                        color="white" if val < 0.55 * vmax else "black")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("Cohen's $d_z$  (Independent $-$ Paired)")
    cbar.outline.set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_effect_size_heatmap.{ext}", bbox_inches="tight")
    print("wrote fig_effect_size_heatmap.pdf and .png")


if __name__ == "__main__":
    main()
