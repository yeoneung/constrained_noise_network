from pathlib import Path
import numpy as np
from figstyle import *
from core import mu, psi, var_S, f_z
from scipy.stats import norm

fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.25), gridspec_kw=dict(wspace=0.36))
ax = axs[0]
s = np.linspace(-3.5, 6, 800)
ax.plot(s, norm.pdf(s), color=MUTED, lw=1.0, ls=(0, (3, 2)), label=r"$N(0,1)$")
cols = [BLUE, ORANGE, AQUA, VIOLET]
for zz, c in zip([-1.0, 0.0, 1.0, 2.0], cols):
    ax.plot(s, f_z(s, zz), color=c, lw=1.3, label=rf"$z={zz:g}$")
ax.set_xlabel(r"$s$"); ax.set_ylabel(r"density $f_z(s)$")
ax.set_xlim(-3.5, 6); ax.set_ylim(0, 1.15)
ax.legend(loc="upper left", fontsize=7, handlelength=1.4, labelspacing=0.25)
ax.set_title(r"(a) density of $S$", loc="left", fontsize=8.5)

ax = axs[1]
z = np.linspace(-3, 3, 601)
ax.plot(z, 1 + psi(z), color=BLUE, lw=1.5, label=r"$\mathbb{E}S^2=1+\psi(z)$")
ax.plot(z, mu(z) ** 2, color=ORANGE, lw=1.5, label=r"$(\mathbb{E}S)^2=\mu(z)^2$")
ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.plot([0], [2], "o", color=BLUE, mec="white", mew=0.7, ms=4.5)
ax.plot([0], [np.pi / 2], "o", color=ORANGE, mec="white", mew=0.7, ms=4.5)
ax.set_xlabel(r"$z$"); ax.set_ylim(0, 13.5); ax.set_xlim(-3, 3)
ax.legend(loc="upper left", fontsize=7, handlelength=1.4)
ax.set_title("(b) second moments", loc="left", fontsize=8.5)

ax = axs[2]
ax.plot(z, var_S(z), color=AQUA, lw=1.5, label=r"$\mathrm{Var}\,S=1-\mu'(z)$")
ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 2)))
zl = np.linspace(-3, -1.4, 40); ax.plot(zl, 1 - zl ** -2.0, color=INK2, lw=0.7, ls=(0, (1, 1.2)))
ax.text(-2.9, 0.52, r"$1-z^{-2}$", fontsize=7, color=INK2)
zr = np.linspace(1.9, 3, 40); ax.plot(zr, 2 / zr ** 2, color=INK2, lw=0.7, ls=(0, (1, 1.2)))
ax.text(2.2, 0.47, r"$2z^{-2}$", fontsize=7, color=INK2)
ax.plot([0], [2 - np.pi / 2], "o", color=AQUA, mec="white", mew=0.7, ms=4.5)
ax.annotate(r"$2-\pi/2\approx 0.4292$", xy=(0, 2 - np.pi / 2), xytext=(0.15, 0.80), fontsize=7, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_xlabel(r"$z$"); ax.set_ylim(0, 1.08); ax.set_xlim(-3, 3)
ax.set_title(r"(c) variance of $S$", loc="left", fontsize=8.5)
fig.savefig(Path(__file__).resolve().parent.parent / "figs" / "fig2_response.pdf")
print("ok")
