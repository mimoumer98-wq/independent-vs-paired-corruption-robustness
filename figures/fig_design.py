"""Design schematic: Paired (rho=-1) vs Independent (rho=0) view construction.
Two panels, no external data dependency (purely illustrative).

Run: python fig_design.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import style

HERE = Path(__file__).resolve().parent
style.apply()

COL = {"paired": style.COLOR["paired"], "independent": style.COLOR["independent"]}

CLEAN_FC = "#EEF2F6"
CORR_FC = "#E7E7E7"
NET_FC = "#F5F1E6"
LOSS_FC = "#FFFFFF"
EDGE = "#2B2B2B"
EDGE_SOFT = "#6B6B6B"

plt.rcParams.update({"font.size": 9.0, "hatch.linewidth": 0.6, "hatch.color": "#A8A8A8"})


def box(ax, x, y, w, h, text, fc, fs=9.0, hatch=None, bold=False, edgecolor=EDGE, lw=1.15):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.022,rounding_size=0.11",
        linewidth=lw, edgecolor=edgecolor, facecolor=fc, hatch=hatch, joinstyle="round",
    ))
    text_bbox = (dict(facecolor="#FAFAFA", edgecolor="none", pad=3.2,
                       boxstyle="round,pad=0.25,rounding_size=0.15") if hatch else None)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=("semibold" if bold else "normal"), color="#1A1A1A",
            bbox=text_bbox, zorder=5)


def arrow(ax, x0, y0, x1, y1, lw=1.05, color=EDGE_SOFT):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9.5,
                                  linewidth=lw, color=color, shrinkA=1.5, shrinkB=1.5))


def panel(ax, label, title, tcol, v1_text, v1_fc, v1_hatch, v2_text, v2_fc, v2_hatch, note):
    ax.set_xlim(0, 10); ax.set_ylim(0.15, 10.1); ax.axis("off")
    ax.text(0.05, 9.95, f"({label})", fontsize=11.5, fontweight="bold", va="top", color="#1A1A1A")
    ax.text(5.0, 9.95, title, ha="center", va="top", fontsize=12.5, fontweight="bold", color=tcol)
    ax.plot([0.3, 9.7], [9.45, 9.45], color="#D8D8D8", linewidth=0.8, zorder=0)

    box(ax, 3.1, 8.05, 3.8, 0.95, "training image  $x$", LOSS_FC)
    box(ax, 0.5, 5.55, 4.1, 1.30, v1_text, v1_fc, hatch=v1_hatch)
    box(ax, 5.4, 5.55, 4.1, 1.30, v2_text, v2_fc, hatch=v2_hatch)
    arrow(ax, 4.30, 8.05, 2.90, 6.92)
    arrow(ax, 5.70, 8.05, 7.10, 6.92)
    ax.text(0.5, 7.00, "view 1", fontsize=8.3, style="italic", color="#444444")
    ax.text(5.4, 7.00, "view 2", fontsize=8.3, style="italic", color="#444444")

    box(ax, 2.6, 3.60, 4.8, 1.05, r"shared network  $f_\theta$", NET_FC, bold=True, lw=1.4)
    arrow(ax, 2.55, 5.55, 4.05, 4.68)
    arrow(ax, 7.45, 5.55, 5.95, 4.68)

    box(ax, 0.875, 1.80, 3.35, 0.95, r"$\mathcal{L}_{\mathrm{seg}}(\mathrm{view\,1},\,y)$", LOSS_FC, fs=8.6)
    box(ax, 5.775, 1.80, 3.35, 0.95, r"$\mathcal{L}_{\mathrm{seg}}(\mathrm{view\,2},\,y)$", LOSS_FC, fs=8.6)
    arrow(ax, 4.00, 3.60, 2.70, 2.76)
    arrow(ax, 6.00, 3.60, 7.30, 2.76)

    ax.text(5.0, 1.10, "loss = sum of the two segmentation losses", ha="center", fontsize=8.8, color="#1A1A1A")
    ax.text(5.0, 0.55, note, ha="center", fontsize=8.5, color="#5A5A5A", style="italic")


fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.95))
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.09)

panel(axes[0], "a", r"Paired  ($\rho=-1$)", COL["paired"],
      "clean", CLEAN_FC, None, r"corrupted  $c(x;\sigma)$", CORR_FC, "..",
      "every step is clean + corrupted")
panel(axes[1], "b", r"Independent  ($\rho=0$)", COL["independent"],
      "clean or corrupted", CLEAN_FC, None, "clean or corrupted", CLEAN_FC, None,
      "each view corrupted w.p. 0.5, drawn independently")

fig.add_artist(plt.Line2D([0.503, 0.503], [0.04, 0.96], transform=fig.transFigure,
                           color="#E2E2E2", linewidth=0.9))

for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_design.{ext}", bbox_inches="tight", pad_inches=0.06, dpi=300)
print("wrote fig_design.pdf and fig_design.png")
