from pathlib import Path
import numpy as np
from figstyle import *
from core import psi

d = np.load("results/schematic.npz")
N = 6; u, v = 1, 4
regs = [("overlap", r"(a) overlapping intervals, $\alpha<\beta$"),
        ("contact", r"(b) contact, $\alpha=\beta$"),
        ("separated", r"(c) incompatible intervals, $\alpha>\beta$")]

fig = plt.figure(figsize=(TEXTW, 4.35))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.62, wspace=0.28)
xs = np.arange(1, N + 1)
for k, (key, title) in enumerate(regs):
    ax = fig.add_subplot(gs[0, k])
    lo, hi = d[f"{key}_lo"], d[f"{key}_hi"]
    X = d[f"{key}_X"]
    alpha, beta = lo.max(), hi.min()
    for i in range(N):
        ax.add_patch(plt.Rectangle((xs[i] - 0.17, lo[i]), 0.34, hi[i] - lo[i], facecolor="#f0efec",
                                   edgecolor=AXIS, linewidth=0.6, zorder=1))
    if alpha < beta:
        ax.axhspan(alpha, beta, color=AQUA, alpha=0.16, lw=0, zorder=0)
    elif alpha == beta:
        ax.axhline(alpha, color=AQUA, lw=1.0, zorder=0)
    for s in X[:30]:
        ax.plot(xs, s, color=INK2, lw=0.5, alpha=0.22, zorder=2)
    ax.plot(xs, X[0], color=BLUE, lw=1.3, marker="o", ms=3.6, mec="white", mew=0.6, zorder=4)
    if key == "separated":
        ax.plot(xs, d[f"{key}_xstar"], color=INK, lw=1.0, ls=(0, (3, 2)), zorder=3)
        ax.plot([xs[u] + 0.25, 6.62], [lo[u], lo[u]], color=ORANGE, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        ax.plot([xs[v] + 0.25, 6.62], [hi[v], hi[v]], color=ORANGE, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        ax.annotate("", xy=(6.55, lo[u]), xytext=(6.55, hi[v]),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
        ax.text(6.38, 0.5 * (lo[u] + hi[v]), r"$\delta$", va="center", ha="right", fontsize=8, color=INK2)
    ax.plot([xs[u] - 0.25, xs[u] + 0.25], [lo[u], lo[u]], color=ORANGE, lw=2.2, solid_capstyle="round", zorder=5)
    ax.plot([xs[v] - 0.25, xs[v] + 0.25], [hi[v], hi[v]], color=ORANGE, lw=2.2, solid_capstyle="round", zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r"$1$", r"$u$", r"$3$", r"$4$", r"$v$", r"$6$"])
    ax.set_xlim(0.45, N + 0.75)
    ax.set_ylim(-0.85, 1.0)
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=8.5, color=INK)
    if k == 0:
        ax.set_ylabel(r"state $x_i$")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(r"node $i$")

ax = fig.add_subplot(gs[1, :])
z = np.linspace(-4, 10, 1401)
Nn = 6
ex = Nn - 1 + psi(z) - np.clip(z, 0, None) ** 2
ax.plot(z, ex, color=BLUE, lw=1.6, zorder=3)
for val, lab in [(Nn - 1, r"$N-1$"), (Nn, r"$N$"), (Nn + 2, r"$N+2$")]:
    ax.axhline(val, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(10.15, val, lab, va="center", ha="left", fontsize=8, color=INK2)
ax.plot([0], [Nn], "o", color=BLUE, mec="white", mew=0.8, ms=5, zorder=4)
ax.annotate(r"overlap: $\chi^2_{N-1}$", xy=(-2.8, Nn - 1 + 0.02), xytext=(-2.4, Nn - 1 + 0.85),
            fontsize=8, color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.annotate(r"exact contact: $\chi^2_{N}$", xy=(0, Nn), xytext=(-1.6, Nn + 1.25),
            fontsize=8, color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.annotate(r"separation: excess $\chi^2_{N+|\mathcal A|}$, $|\mathcal A|=2$", xy=(6.5, Nn - 1 + psi(6.5) - 6.5**2), xytext=(5.6, Nn - 0.55),
            fontsize=8, color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_xlim(-4, 10); ax.set_ylim(Nn - 1.4, Nn + 2.6)
ax.set_yticks([Nn - 1, Nn, Nn + 1, Nn + 2])
ax.set_xlabel(r"dimensionless gap $z=(\ell_u-r_v)/(\varepsilon\sqrt{R_{uv}})$")
ax.set_ylabel(r"$\lim\,\mathbb{E}\,Q_\varepsilon-z_+^2$")
ax.set_title(r"(d) mean energy above the deterministic floor, $N=6$", loc="left", fontsize=8.5, color=INK)
fig.savefig(Path(__file__).resolve().parent.parent / "figs" / "fig1_regimes.pdf")
print("ok")
