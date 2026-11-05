from pathlib import Path
import numpy as np
from figstyle import *
from core import psi, var_S

d = np.load("results/crossover.npz")
z = d["zgrid"]
names = ["Path", "Cycle", "Star", "Complete"]
mks = ["o", "s", "^", "D"]
fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.35), gridspec_kw=dict(wspace=0.34))
zz = np.linspace(-3.2, 3.2, 400)
for k, (nm, c, mk) in enumerate(zip(names, CAT, mks)):
    R = float(d[f"{nm}_R"]); M = d[f"{nm}_mean"]; S = d[f"{nm}_se"]
    lab = rf"{nm.lower()}, $R_{{uv}}={R:.4f}$"
    x = np.sqrt(R) * z
    xx = np.linspace(-5.2, 5.2, 500)
    axs[0].plot(xx, 1 + psi(xx / np.sqrt(R)), color=c, lw=1.0, zorder=2)
    sl = slice(k, len(z), 4)
    axs[0].plot(x[sl], M[sl, 0], mk, color=c, ms=3.8, mec="white", mew=0.5, zorder=3)
    axs[1].errorbar(z[sl], M[sl, 0], yerr=2 * S[sl, 0], fmt=mk, color=c, ms=3.8, mec="white", mew=0.5, elinewidth=0.7, zorder=3, label=lab)
    axs[2].errorbar(z[sl], M[sl, 1], yerr=2 * S[sl, 1], fmt=mk, color=c, ms=3.8, mec="white", mew=0.5, elinewidth=0.7, zorder=3)
axs[1].plot(zz, 1 + psi(zz), color=INK, lw=1.0, zorder=2)
axs[2].plot(zz, var_S(zz), color=INK, lw=1.0, zorder=2)
axs[2].axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 2)))
axs[0].set_xlim(-5, 5); axs[0].set_ylim(0, 13)
axs[0].set_xlabel(r"$\delta/\varepsilon$"); axs[0].set_ylabel(r"$\mathbb{E}[(X_u-X_v)^2]/(\varepsilon^2R_{uv})$")
axs[1].legend(loc="upper left", fontsize=6.5, handlelength=1.0, labelspacing=0.2, borderaxespad=0.2)
axs[0].set_title(r"(a) physical gap", loc="left", fontsize=8.5)
axs[1].set_xlim(-3.2, 3.2); axs[1].set_ylim(0, 13)
axs[1].set_xlabel(r"$z=\delta/(\varepsilon\sqrt{R_{uv}})$"); axs[1].set_ylabel(r"$\mathbb{E}S^2$")
axs[1].set_title(r"(b) resistance scale", loc="left", fontsize=8.5)
axs[2].set_xlim(-3.2, 3.2); axs[2].set_ylim(0, 1.08)
axs[2].set_xlabel(r"$z$"); axs[2].set_ylabel(r"$\mathrm{Var}\,S$")
axs[2].set_title(r"(c) centered variance", loc="left", fontsize=8.5)
fig.savefig(Path(__file__).resolve().parent.parent / "figs" / "fig4_universality.pdf")
print("ok")
for nm in names:
    M = d[f"{nm}_mean"]; S = d[f"{nm}_se"]
    dev = (M[:, 0] - (1 + psi(z))) / S[:, 0]
    dv = (M[:, 1] - var_S(z)) / S[:, 1]
    dq = (M[:, 2] - (7 + psi(z))) / S[:, 2]
    print(nm, "max |z-score| ES2 %.2f VarS %.2f EQ %.2f" % (np.abs(dev).max(), np.abs(dv).max(), np.abs(dq).max()), "ess min %.4f" % M[:, 3].min())
