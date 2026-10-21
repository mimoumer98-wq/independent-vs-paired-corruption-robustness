"""Shared figure style. Imported by every figure script so colours, fonts and
axis treatment are identical across figures.

Palette: Okabe-Ito (colour-blind safe). Independent (rho=0) is the highlighted
condition: distinctive blue, thicker line. Paired (rho=-1) is the vermillion
comparison line. Never use jet/rainbow; heatmaps use viridis/cividis.
"""
import matplotlib as mpl

COLOR = {"independent": "#0072B2", "paired": "#D55E00", "singleview": "#009E73"}
LW = {"independent": 2.3, "paired": 1.4, "singleview": 1.7}
MARKER = {"independent": "o", "paired": "s", "singleview": "^"}
ZORDER = {"independent": 3, "paired": 2, "singleview": 2}
LEGEND_LABEL = {"independent": r"Independent ($\rho=0$)",
                 "paired": r"Paired ($\rho=-1$)",
                 "singleview": r"Single-view baseline"}
BAND_ALPHA = 0.18


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.8,
        "lines.markersize": 4,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
