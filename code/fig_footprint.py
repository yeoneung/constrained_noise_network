from pathlib import Path
import numpy as np, json, networkx as nx
from figstyle import *
from core import psi, mu
from matplotlib.collections import LineCollection
import matplotlib as mpl

foot = json.load(open("results/footprint.json"))
fig = plt.figure(figsize=(TEXTW, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[0.78, 0.78, 1.6], wspace=0.45)
norm_e = mpl.colors.Normalize(vmin=0, vmax=0.5)
cmap = mpl.colors.LinearSegmentedColormap.from_list("t", ["#e1e0d9", SEQ(0.35), SEQ(1.0)])


def draw(ax, G, pos, rows, u, v, title, labels):
    share = {tuple(r["edge"]): r["share"] for r in rows}
    segs, cols, lws = [], [], []
    for i, j in G.edges():
        e = (min(i, j), max(i, j))
        segs.append([pos[i], pos[j]]); s = share[e]
        cols.append(cmap(norm_e(s))); lws.append(1.0 + 7.0 * s)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=1, capstyle="round"))
    P = np.array([pos[i] for i in range(G.number_of_nodes())])
    ax.scatter(P[:, 0], P[:, 1], s=60, c="white", edgecolors=INK2, linewidths=0.8, zorder=3)
    ax.scatter([pos[u][0]], [pos[u][1]], s=80, marker="^", c=RED, edgecolors="white", linewidths=0.6, zorder=4)
    ax.scatter([pos[v][0]], [pos[v][1]], s=80, marker="v", c=BLUE, edgecolors="white", linewidths=0.6, zorder=4)
    for k, (x, y) in enumerate(P):
        ax.text(x * 1.27, y * 1.27, labels[k], ha="center", va="center", fontsize=7, color=INK2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
    ax.set_title(title, loc="left", fontsize=8.5)


Gc = nx.cycle_graph(8)
posc = {i: (np.sin(2 * np.pi * i / 8), np.cos(2 * np.pi * i / 8)) for i in range(8)}
draw(fig.add_subplot(gs[0, 0]), Gc, posc, foot["Cycle"], 0, 4, r"(a) cycle, $R_{uv}=2$", [str(i + 1) for i in range(8)])
Gs = nx.star_graph(7)
poss = {0: (0.0, 0.0)}
for k in range(1, 8):
    ang = 2 * np.pi * (k - 1) / 7
    poss[k] = (np.sin(ang), np.cos(ang))
ax = fig.add_subplot(gs[0, 1])
draw(ax, Gs, poss, foot["Star"], 1, 2, r"(b) star, $R_{uv}=2$", [str(i + 1) for i in range(8)])
ax.texts[0].set_position((0.17, -0.2))

ax = fig.add_subplot(gs[0, 2])
xs, lab, xt = [], [], []
pos_x = 0
groups = [("Cycle", foot["Cycle"]), ("Star", foot["Star"])]
for gname, rows in groups:
    for r in rows:
        i, j = r["edge"]
        x = pos_x
        ax.bar(x - 0.19, r["raw"], width=0.36, color=ORANGE, alpha=0.85, lw=0, zorder=2)
        ax.errorbar(x - 0.19, r["raw"], yerr=2 * r["raw_se"], fmt="none", ecolor=INK, elinewidth=0.9, capsize=1.2, zorder=3)
        ax.bar(x + 0.19, r["cen"], width=0.36, color=BLUE, alpha=0.85, lw=0, zorder=2)
        ax.errorbar(x + 0.19, r["cen"], yerr=2 * r["cen_se"], fmt="none", ecolor=INK, elinewidth=0.9, capsize=1.2, zorder=3)
        ax.plot([x - 0.37, x - 0.01], [psi(0) * r["share"]] * 2, color=INK, lw=1.1, zorder=4)
        ax.plot([x + 0.01, x + 0.37], [-(mu(0) ** 2 - psi(0)) * r["share"]] * 2, color=INK, lw=1.1, zorder=4)
        xt.append(x); lab.append(rf"${i + 1}{j + 1}$")
        pos_x += 1
    pos_x += 0.8
ax.axhline(0, color=AXIS, lw=0.7, zorder=1)
ax.axvline(7.9, color=GRID, lw=0.8)
ax.text(3.5, 0.62, "cycle", ha="center", fontsize=8, color=INK2)
ax.text(11.8, 0.62, "star", ha="center", fontsize=8, color=INK2)
ax.set_xticks(xt); ax.set_xticklabels(lab, fontsize=6.5)
ax.set_xlim(-0.7, pos_x - 0.6); ax.set_ylim(-0.38, 0.7)
ax.grid(axis="x", visible=False)
ax.set_ylabel(r"edge excess over $R_{ij}$")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
hd = [Patch(color=ORANGE, label="mean square"), Patch(color=BLUE, label="variance"), Line2D([], [], color=INK, lw=1.1, label="theory")]
ax.legend(handles=hd, loc="upper left", fontsize=7, handlelength=1.2, bbox_to_anchor=(0.01, 0.86), labelspacing=0.25)
ax.set_title(r"(c) footprint at exact contact, $\varepsilon=0.0500$", loc="left", fontsize=8.5)
fig.savefig(Path(__file__).resolve().parent.parent / "figs" / "fig5_footprint.pdf")
print("ok")
