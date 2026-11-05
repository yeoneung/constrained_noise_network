from pathlib import Path
import numpy as np, networkx as nx
from figstyle import *
from core import psi, mu
from matplotlib.collections import LineCollection
import matplotlib as mpl

d = np.load("results/spatial.npz")
R, h, pos, E = d["R"], d["h"], d["pos"], d["edges"]
u, v = int(d["u"]), int(d["v"])
N = len(h)
fig = plt.figure(figsize=(TEXTW, 2.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.22)
ax = fig.add_subplot(gs[0, 0])
ew = np.array([(h[i] - h[j]) ** 2 for i, j in E])
segs = [[pos[i], pos[j]] for i, j in E]
norm_e = mpl.colors.Normalize(vmin=0, vmax=ew.max())
lc = LineCollection(segs, colors=SEQ(0.15 + 0.85 * norm_e(ew)), linewidths=0.6 + 2.6 * ew / ew.max(), zorder=1)
ax.add_collection(lc)
hm = np.max(np.abs(h))
sc = ax.scatter(pos[:, 0], pos[:, 1], c=h, cmap=DIV, vmin=-hm, vmax=hm, s=46, edgecolors="white", linewidths=0.7, zorder=3)
ax.scatter([pos[u, 0]], [pos[u, 1]], marker="^", s=120, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=4)
ax.scatter([pos[v, 0]], [pos[v, 1]], marker="v", s=120, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=4)
ax.annotate(r"$u$", pos[u], xytext=(7, 5), textcoords="offset points", fontsize=9)
ax.annotate(r"$v$", pos[v], xytext=(7, -10), textcoords="offset points", fontsize=9)
ax.set_aspect("equal"); ax.axis("off")
cax = ax.inset_axes([0.02, -0.06, 0.42, 0.035])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.set_label(r"contact field $h_i$", fontsize=7.5, labelpad=1); cb.ax.tick_params(labelsize=7, length=2); cb.outline.set_linewidth(0.4)
cax2 = ax.inset_axes([0.56, -0.06, 0.42, 0.035])
cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_e, cmap=mpl.colors.LinearSegmentedColormap.from_list("t", [SEQ(0.15), SEQ(1.0)])), cax=cax2, orientation="horizontal")
cb2.set_label(r"edge weight $(h_i-h_j)^2$", fontsize=7.5, labelpad=1); cb2.ax.tick_params(labelsize=7, length=2); cb2.outline.set_linewidth(0.4)
ax.set_title(r"(a) small world network, $N=40$, $R_{uv}=%.4f$" % R[u, v], loc="left", fontsize=8.5)

ax = fig.add_subplot(gs[0, 1])
iu = np.triu_indices(N, 1)
hh = ((h[:, None] - h[None, :]) ** 2)[iu]
xg = np.linspace(0, hh.max() * 1.03, 10)
for z, c, mk in [(0.0, BLUE, "o"), (1.0, ORANGE, "s")]:
    raw = (d[f"raw_mean_{z}"] - R)[iu]; cen = (d[f"cen_mean_{z}"] - R)[iu]
    ax.plot(xg, psi(z) * xg, color=c, lw=1.0, zorder=2)
    ax.plot(xg, (psi(z) - mu(z) ** 2) * xg, color=c, lw=1.0, ls=(0, (3, 2)), zorder=2)
    ax.plot(hh, raw, mk, color=c, ms=2.6, mec="none", alpha=0.75, zorder=3, label=rf"mean square, $z={z:g}$")
    ax.plot(hh, cen, mk, mfc="white", mec=c, mew=0.6, ms=2.6, alpha=0.9, zorder=3, label=rf"variance, $z={z:g}$")
ax.axhline(0, color=AXIS, lw=0.7)
ax.set_xlabel(r"$(h_i-h_j)^2=R_{ij,uv}^2/R_{uv}$")
ax.set_ylabel(r"excess over $R_{ij}$")
ax.legend(loc="upper left", fontsize=7, handlelength=1.0, labelspacing=0.25, ncol=1)
ax.set_xlim(0, hh.max() * 1.03)
ax.set_title(r"(b) all $780$ pairs, $\varepsilon=0.0500$", loc="left", fontsize=8.5)
fig.savefig(Path(__file__).resolve().parent.parent / "figs" / "fig6_spatial.pdf")
print("ok", R[u, v], hh.max())
