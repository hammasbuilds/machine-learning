"""One visual language for all twenty projects.

A portfolio of twenty charts drawn twenty different ways reads as twenty weekend
projects. The same palette, the same grid, the same caption convention reads as one body
of work, and costs nothing but deciding once.

Rules encoded here, each because the default is worse:

  - No top or right spine. They box in the data and carry no information.
  - Gridlines behind the data, light, horizontal only. Vertical gridlines on a time
    series compete with the line itself.
  - A caption under every figure stating the source and the sample size. A chart without
    n is a chart that cannot be argued with.
  - Colourblind-safe palette (Okabe-Ito). Roughly one man in twelve cannot distinguish
    the default matplotlib red from its green.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito: the standard colourblind-safe qualitative palette.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}
CYCLE = [PALETTE[k] for k in ("blue", "orange", "green", "red", "purple", "sky")]

INK = "#1a1a1a"
MUTED = "#8a8a8a"
GRID = "#e4e4e4"


def use_house_style() -> None:
    """Apply the shared style. Call once at the top of any script that plots."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.9,
            "axes.labelcolor": INK,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=CYCLE),
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,  # data on top of the grid, never under it
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "figure.autolayout": False,
        }
    )


def caption(fig: plt.Figure, text: str, *, y: float = -0.02) -> None:
    """Source-and-sample-size line under a figure.

    Deliberately not optional in the project scripts. Every number on every chart here
    came from somewhere, and saying where is the difference between evidence and
    decoration.
    """
    fig.text(0.0, y, text, ha="left", va="top", fontsize=8, color=MUTED, style="italic")


def annotate(ax: plt.Axes, x, y, text: str, *, dx: float = 6, dy: float = 6, color=None) -> None:
    """Label a specific point, because a reader should not have to hunt for the finding."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9,
        color=color or INK,
        fontweight="bold",
    )


def save(fig: plt.Figure, path: str | Path) -> Path:
    """Write a figure and return where it went. Creates parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def percent_axis(ax: plt.Axes, axis: str = "y", decimals: int = 0) -> None:
    """Format an axis as percentages. Proportions on a 0-1 axis get misread as counts."""
    fmt = mpl.ticker.PercentFormatter(xmax=1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def money_axis(ax: plt.Axes, axis: str = "y", symbol: str = "$") -> None:
    """Thousands separators and a currency symbol, so 1e6 never appears on a chart."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{symbol}{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
