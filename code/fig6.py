from pathlib import Path
import numpy as np
from figstyle import *
from scipy.stats import chi2

d = np.load("results/ladder.npz")
m = np.load("results/ladder_means.npz")
inc = np.load("results/incompat.npz")
fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.55), gridspec_kw=dict(wspace=0.28, width_ratios=[1.15, 1]))
ax = axs[0]
bins = np.linspace(0, 32, 65)
x = np.linspace(0.01, 32, 600)
for key, wkey, k, c, lab in [("overlap_Q", "overlap_w", 7, BLUE, r"overlap"),
                              ("contact_Q", "contact_w", 8, ORANGE, r"contact"),
                              ("sep_exc", "sep_w", 10, AQUA, r"incompatible, excess")]:
    ax.hist(d[key], bins=bins, weights=d[wkey], density=True, histtype="stepfilled", color=c, alpha=0.18, lw=0)
    ax.hist(d[key], bins=bins, weights=d[wkey], density=True, histtype="step", color=c, lw=0.8)
    ax.plot(x, chi2.pdf(x, k), color=c, lw=1.5, label=lab + rf" vs $\chi^2_{{{k}}}$")
ax.set_xlim(0, 32); ax.set_ylim(0, 0.135)
ax.set_xlabel(r"normalized energy"); ax.set_ylabel(r"density")
ax.legend(loc="upper right", fontsize=7, handlelength=1.4, labelspacing=0.3)
ax.set_title(r"(a) path graph, $N=8$", loc="left", fontsize=8.5)

ax = axs[1]
e = m["epss"]
rows = [(m["overlap_mean"], m["overlap_se"], 7, BLUE, "o", r"overlap"),
        (m["contact_mean"], m["contact_se"], 8, ORANGE, "s", r"contact"),
        (inc["A2_mean"][:, 0], inc["A2_se"][:, 0], 10, AQUA, "^", r"$|\mathcal A|=2$"),
        (inc["A3_mean"][:, 0], inc["A3_se"][:, 0], 11, VIOLET, "D", r"$|\mathcal A|=3$"),
        (inc["A4_mean"][:, 0], inc["A4_se"][:, 0], 12, RED, "v", r"$|\mathcal A|=4$")]
for mean, se, k, c, mk, lab in rows:
    ax.axhline(k, color=c, lw=0.8, ls=(0, (3, 2)), alpha=0.9, zorder=1)
    ax.errorbar(e, mean, yerr=2 * se, fmt=mk, color=c, ms=4, mec="white", mew=0.5, elinewidth=0.8, capsize=0, zorder=3)
    ax.text(0.052, k + 0.07, lab, va="bottom", ha="left", fontsize=7, color=INK2)
ax.set_xscale("log"); ax.set_xlim(0.002, 0.11)
ax.set_xticks([0.0025, 0.005, 0.01, 0.02, 0.04]); ax.set_xticklabels([r"$0.0025$", r"$0.0050$", r"$0.0100$", r"$0.0200$", r"$0.0400$"], fontsize=6.5, rotation=30)
ax.minorticks_off()
ax.set_yticks([7, 8, 9, 10, 11, 12])
ax.set_ylim(6.5, 12.5)
ax.set_xlabel(r"noise length $\varepsilon$")
ax.set_ylabel(r"mean energy above the floor")
ax.set_title(r"(b) the ladder $N-1,\ N,\ N+|\mathcal A|$", loc="left", fontsize=8.5)
fig.savefig(Path(__file__).resolve().parent.parent / "figs" / "fig7_ladder.pdf")
print("ok")
