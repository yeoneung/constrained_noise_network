import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seqblue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("divbr", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                                                   "#f3b3b2", "#e66767", "#b8302f"])
TEXTW = 6.3

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsmath,amssymb}",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def panel_label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)
