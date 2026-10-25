"""One chart style for every figure in the project (thin marks, hairline grid, quiet axes)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LAYOUT_COLORS = {"thyroid": "#2a78d6", "liver": "#eb6834", "vitamin": "#1baf7a"}   # fixed per entity
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]                               # fixed slot order
FIELD_BOX_BGR = {"test_name": (214, 120, 42), "value": (52, 104, 235),
                 "unit": (122, 175, 27), "reference_range": (0, 161, 237)}           # same hues, BGR


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9, "lines.linewidth": 2,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def bar_labels(ax, bars, fmt="{:.0f}"):
    """Direct-label bar ends in ink colour (never the series colour)."""
    for b in bars:
        v = b.get_height()
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)


def hbar_labels(ax, bars, fmt="{:.0f}"):
    for b in bars:
        v = b.get_width()
        ax.annotate(fmt.format(v), (v, b.get_y() + b.get_height() / 2), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8, color=INK)
