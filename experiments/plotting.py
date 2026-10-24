"""Plotting functions for the figures of the numerical example."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager
from matplotlib.lines import Line2D

try:
    from .simulation import W_TYPE_LABELS, W_TYPE_TITLES, W_TYPES
except ImportError:
    from simulation import W_TYPE_LABELS, W_TYPE_TITLES, W_TYPES


def use_paper_style():
    """Helvetica (if installed) and TrueType fonts embedded in PDF, as in the paper."""
    if any(f.name == "Helvetica" for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["pdf.fonttype"] = 42
    logging.getLogger("fontTools").setLevel(logging.ERROR)  # harmless warnings on embedding Helvetica


# --------------------------------------------------------------------------
# Spatial weight matrices (Fig. 4 and 5 of the paper)
# --------------------------------------------------------------------------
def plot_W_arrows(W, ax):
    """Draw W on the grid as arrows from location i to location j (width ~ |W[j, i]|^3)."""
    n = int(round(W.shape[0] ** 0.5))
    max_w = np.max(np.abs(W))
    offset = 0.15
    for i in range(n * n):
        r_i, c_i = divmod(i, n)
        for j in range(n * n):
            w = W[j, i]
            if w == 0:
                continue
            r_j, c_j = divmod(j, n)
            dx, dy = c_j - c_i, r_j - r_i
            length = np.sqrt(dx ** 2 + dy ** 2)
            dx_norm, dy_norm = dx / length, dy / length
            ax.arrow(c_i + offset * dx_norm, r_i + offset * dy_norm,
                     dx - 2 * offset * dx_norm, dy - 2 * offset * dy_norm,
                     length_includes_head=True, head_width=0.15, head_length=0.2,
                     linewidth=4.0 * (abs(w) / max_w) ** 3, alpha=0.9)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def plot_W_heatmap(W, ax, vmax=0.5):
    """Heatmap of W (coolwarm, centered at 0)."""
    sns.heatmap(W, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax, ax=ax,
                cbar=True, square=True, xticklabels=False, yticklabels=False)
    ax.collections[0].colorbar.set_ticks(np.arange(-vmax, vmax + 0.1, 0.1))
    ax.set_aspect("equal", adjustable="box")


def plot_W_types(Ws, kind="heatmap", titles=False):
    """Three panels (Type 1, 2, 3) of W.

    Parameters
    ----------
    Ws : dict
        W_type -> W matrix.
    kind : {"heatmap", "arrows"}
    titles : bool
        Whether to show the titles of the spatial structures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, W_type in zip(axes, W_TYPES):
        if kind == "arrows":
            plot_W_arrows(Ws[W_type], ax)
        else:
            plot_W_heatmap(Ws[W_type], ax)
        if titles:
            ax.set_title(W_TYPE_TITLES[W_type], fontsize=14)
    return fig


# --------------------------------------------------------------------------
# Monte Carlo MAE (Fig. 6 of the paper)
# --------------------------------------------------------------------------
def plot_mae(df, methods=None, P=(9, 25, 49), center="median", band=("q25", "q75"),
             sharey="col", ymax=None, height=3.0, aspect=1.3):
    """MAE against T. Rows: number of locations P, columns: spatial structure.

    Parameters
    ----------
    df : DataFrame
        Output of ``aggregate.py`` (columns P, type, T, method, mean, std, median, q25, q75).
    methods : list of str, optional
        Methods to draw (default: all methods in ``df``).
    P : tuple of int
        Numbers of locations to draw (the paper uses 9, 25 and 49).
    center : {"median", "mean"}
        Statistic drawn as the line.
    band : ("q25", "q75"), "std" or None
        Band: interquartile range, mean +/- standard deviation, or no band.
    sharey : "col", True or False
        Share the y-axis within each column, over all panels, or not at all.
    ymax : dict, optional
        Type label -> upper limit of the y-axis (e.g. ``{"Type 1": 0.2}``).
    """
    methods = list(methods or df["method"].unique())
    rows = [p for p in P if p in set(df["P"])]
    cols = [W_TYPE_LABELS[w] for w in W_TYPES]
    colors = dict(zip(methods, sns.color_palette("tab10", n_colors=len(methods))))

    fig, axes = plt.subplots(
        len(rows), len(cols), squeeze=False,
        sharey=("col" if sharey == "col" else ("all" if sharey else "none")),
        figsize=(height * aspect * len(cols), height * len(rows)),
    )
    for r, p in enumerate(rows):
        for c, tp in enumerate(cols):
            ax = axes[r][c]
            sub = df[(df["P"] == p) & (df["type"] == tp)]
            for m in methods:
                d = sub[sub["method"] == m].sort_values("T")
                ax.plot(d["T"], d[center], marker="o", color=colors[m], label=m)
                if band is not None:
                    if band == "std":
                        lo, hi = d["mean"] - d["std"], d["mean"] + d["std"]
                    else:
                        lo, hi = d[band[0]], d[band[1]]
                    ax.fill_between(d["T"], lo.clip(lower=0), hi, color=colors[m], alpha=0.2, lw=0)
            ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
            ax.set_title(f"P={p}, W_type={tp.replace('Type ', '')}")
            if r == len(rows) - 1:
                ax.set_xlabel("T")
            if c == 0:
                ax.set_ylabel("MAE")
            if ymax and tp in ymax:
                ax.set_ylim(top=ymax[tp])

    handles = [Line2D([0], [0], color=colors[m], marker="o", label=m) for m in methods]
    fig.legend(handles=handles, title="Method", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    return fig
