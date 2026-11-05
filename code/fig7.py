from pathlib import Path
import numpy as np
from figstyle import *
from scipy.stats import chi2

d = np.load("results/dynamics.npz")
traj, dt, eps, R = d["traj"], float(d["dt"]), float(d["eps"]), float(d["R"])
t = np.arange(len(traj)) * dt
fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.3), gridspec_kw=dict(wspace=0.34, width_ratios=[1.45, 1, 1]))
ax = axs[0]
sel = (t >= 100) & (t <= 300)
ax.axhline(0, color=INK, lw=0.8, zorder=1)
ax.plot(t[sel], traj[sel, 0] / eps, color=ORANGE, lw=0.8, label=r"$X_u/\varepsilon$")
ax.plot(t[sel], traj[sel, 7] / eps, color=BLUE, lw=0.8, label=r"$X_v/\varepsilon$")
ax.plot(t[sel], traj[sel].mean(1) / eps, color=INK2, lw=0.8, alpha=0.9, label=r"$M/\varepsilon$")
ax.set_xlim(100, 300); ax.set_ylim(-4.5, 6.8)
ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"scaled state")
ax.legend(loc="upper left", fontsize=6.8, ncol=3, handlelength=1.0, columnspacing=0.8, borderaxespad=0.1)
ax.set_title(r"(a) reflected trajectories", loc="left", fontsize=8.5)

ax = axs[1]
e, h = d["edges"], d["hist"]
w = np.diff(e); dens = h / (h.sum() * w)
ax.bar(e[:-1], dens, width=w, align="edge", color=ORANGE, alpha=0.25, lw=0)
ax.step(e[:-1], dens, where="post", color=ORANGE, lw=0.7)
s = np.linspace(0, 5, 400)
ax.plot(s, s * np.exp(-s ** 2 / 2), color=INK, lw=1.2, label="Rayleigh")
ax.set_xlim(0, 4.5); ax.set_ylim(0, 0.72)
ax.set_xlabel(r"$S=(X_u-X_v)/(\varepsilon\sqrt{R_{uv}})$"); ax.set_ylabel("density")
ax.legend(loc="upper right", fontsize=7)
ax.set_title(r"(b) contact contrast", loc="left", fontsize=8.5)

ax = axs[2]
e, h = d["Qedges"], d["Qhist"]
w = np.diff(e); dens = h / (h.sum() * w)
ax.bar(e[:-1], dens, width=w, align="edge", color=BLUE, alpha=0.22, lw=0)
ax.step(e[:-1], dens, where="post", color=BLUE, lw=0.7)
x = np.linspace(0.01, 30, 400)
ax.plot(x, chi2.pdf(x, 8), color=INK, lw=1.2, label=r"$\chi^2_8$")
ax.plot(x, chi2.pdf(x, 7), color=MUTED, lw=0.9, ls=(0, (3, 2)), label=r"$\chi^2_7$")
ax.set_xlim(0, 26); ax.set_ylim(0, 0.13)
ax.set_xlabel(r"$Q_\varepsilon$"); ax.set_ylabel("density")
ax.legend(loc="upper right", fontsize=7)
ax.set_title(r"(c) interaction energy", loc="left", fontsize=8.5)
fig.savefig(Path(__file__).resolve().parent.parent / "figs" / "fig8_dynamics.pdf")
print("ok")
