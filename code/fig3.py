from pathlib import Path
import numpy as np
from figstyle import *
from core import psi

# Figure 3 was rendered with Matplotlib's PDF/mathtext backend.
plt.switch_backend("pdf")
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"

d = np.load("results/two_node.npz")
fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.45), gridspec_kw=dict(wspace=0.28))
ax = axs[0]
z = d["z"]
ax.plot(z, 1 + psi(z), color=INK, lw=1.0, zorder=5, label=r"limit $1+\psi(z)$")
mk = ["o", "s", "^", "D"]
for k, (e, c, m) in enumerate(zip([0.2, 0.1, 0.05, 0.025], CAT, mk)):
    sl = slice(3 * k, len(z), 12)
    ax.plot(z[sl], d[f"eps{e}"][sl], m, color=c, ms=3.8, mec="white", mew=0.5, lw=0, label=rf"$\varepsilon={e:.4f}$")
ax.set_xlabel(r"$z=\delta/\varepsilon$")
ax.set_ylabel(r"$\varepsilon^{-2}\,\mathbb{E}[(X_2-X_1)^2]$")
ax.set_xlim(-3, 3); ax.set_ylim(0, 13)
ax.legend(loc="upper left", fontsize=7, handlelength=1.2, labelspacing=0.25)
ax.set_title(r"(a) collapse in the contact scale", loc="left", fontsize=8.5)

ax = axs[1]
dd = d["delta"]
ax.plot(dd, np.clip(dd, 0, None) ** 2, color=INK, lw=1.0, ls=(0, (3, 2)), label=r"$\varepsilon=0$: $\delta_+^2$")
for (e, c) in zip([0.1, 0.05, 0.025], [BLUE, ORANGE, AQUA]):
    ax.plot(dd, d[f"phys{e}"], color=c, lw=1.4, label=rf"$\varepsilon={e:.4f}$")
    ax.plot(dd[4::24], e ** 2 * (1 + psi(dd[4::24] / e)), "o", color=c, ms=3.4, mec="white", mew=0.5)
ax.plot([], [], "o", color=MUTED, ms=3.4, mec="white", mew=0.5, label=r"$\varepsilon^2\{1+\psi(\delta/\varepsilon)\}$")
ax.set_xlabel(r"physical gap $\delta=\ell_2-r_1$")
ax.set_ylabel(r"$\mathbb{E}[(X_2-X_1)^2]$")
ax.set_xlim(-0.3, 0.3); ax.set_ylim(0, 0.11)
ax.legend(loc="upper left", fontsize=7, handlelength=1.6, labelspacing=0.25)
ax.set_title(r"(b) noise rounds the change of curvature", loc="left", fontsize=8.5)
fig.savefig(Path(__file__).resolve().parent.parent / "figs" / "fig3_twonode.pdf")
print("ok")
