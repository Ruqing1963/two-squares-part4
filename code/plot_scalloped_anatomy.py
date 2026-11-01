r"""
Figure 1 of Part IV: (a) the scalloped bottom y = sqrt(1 - {x}^2) of Waibel's fundamental domain against the lines y = 1
and y = sqrt3/2 of the sandwich; (b) the parts of ||G_{1,p}||^2 - (log p)^2/pi against log p (results/subleading_log_anatomy.txt)
with their fits b log p + c over p >= 10^4 and the two proved slopes 0.38301 (free part) and 2 kappa/pi = 0.02879 (arcs on
the free part).   Usage: python plot_scalloped_anatomy.py
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import pi

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, PLUM, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#898781", "#8e44ad", "#c9a227"

rows = []
for line in open(os.path.join(RES, 'subleading_log_anatomy.txt'), encoding='utf-8'):
    m = re.match(r"\s+(\d+)\s+((?:[-+]?[\d.]+|nan)\s+){14}(?:[-+]?[\d.]+|nan)", line)
    if m and not line.strip().startswith('p'):
        vals = line.split()
        if len(vals) >= 16 and vals[0].isdigit():
            rows.append([float(v) for v in vals[:16]])
rows = np.array(rows)
p = rows[:, 0]
L = np.log(p)
C1 = rows[:, 1] - L ** 2 / pi
C2 = rows[:, 3]
C3 = rows[:, 4] + rows[:, 5] + rows[:, 6]
arc_main, arc_free, arc_other = rows[:, 7], rows[:, 8], rows[:, 9]
off = rows[:, 10] + rows[:, 11]
exact = rows[:, 14] - L ** 2 / pi

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
# (a) the scallop
x = np.linspace(-1.5, 2.5, 1601)
scallop = np.sqrt(1 - (x - np.round(x)) ** 2)
ax.fill_between(x, scallop, 1.0, color=AQUA, alpha=0.25, lw=0, label=r"between the arcs and $y=1$: $\Psi_0(c)-e^{-c}$")
ax.fill_between(x, np.sqrt(3) / 2, scallop, color=ORANGE, alpha=0.18, lw=0, label=r"below the arcs: outside $\bigcup_t t\mathcal{F}$")
ax.plot(x, scallop, color=INK, lw=1.8, label=r"$y=\sqrt{1-\{x\}^2}$ (exact bottom)")
ax.axhline(1.0, color=BLUE, lw=1.3, ls="--", label=r"$y=1$ ($\Lambda_1$, lower bound)")
ax.axhline(np.sqrt(3) / 2, color=ORANGE, lw=1.3, ls=":", label=r"$y=\sqrt{3}/2$ ($\Lambda_{\sqrt{3}/2}$, upper bound)")
for t in range(-1, 3):
    ax.axvline(t - 0.5, color=MUTED, lw=0.5, ls=":")
ax.set_xlim(-1.5, 2.5)
ax.set_ylim(0.8, 1.35)
ax.set_xlabel(r"$x$ (width coordinate of the cusp; period $1$ of the scallop, period $Q$ of $G|_2\varrho$)")
ax.set_ylabel(r"$y$")
ax.set_title(r"(a) the bottom of $\bigcup_{0\leq t<L}t\mathcal{F}$", fontsize=10)
ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# (b) the anatomy
big = p >= 1e4


def fit(y):
    A = np.vstack([L[big], np.ones(big.sum())]).T
    return np.linalg.lstsq(A, y[big], rcond=None)[0]


series = [(exact, INK, r"$\Vert G_{1,p}\Vert^2-\frac{1}{\pi}(\log p)^2$", "o"),
          (C1, BLUE, r"$C_1'$: free part of the main cusps", "s"),
          (C2, ORANGE, r"$C_2$: Siegel term and $k\geq p$", "^"),
          (arc_main, AQUA, r"$C_4$: arcs, main cusps", "D"),
          (arc_free, PLUM, r"arcs on the free part", "v"),
          (C3 + arc_other + off, GOLD, r"$C_3$ + rest (other cusps, correlations)", "x")]
Lf = np.linspace(L.min(), L.max(), 50)
for y, col, lab, mk in series:
    b, c = fit(y)
    bx.plot(L, y, color=col, lw=0, marker=mk, ms=4.5, label=f"{lab}: $b={b:+.4f}$")
    bx.plot(Lf, b * Lf + c, color=col, lw=1.1, ls="--")
bx.text(L.max() - 0.1, 1.05, r"proved: $b=0.38301$ (Lemma 6.10 of Part II)" + "\n" + r"$b=2\kappa/\pi=0.02879$ (Mellin, Section 2)",
        fontsize=7.5, color=MUTED, ha="right", va="bottom")
bx.set_xlabel(r"$\log p$")
bx.set_ylabel(r"contribution to $\Vert G_{1,p}\Vert^2-\frac{1}{\pi}(\log p)^2$")
bx.set_title(r"(b) the anatomy of $B\approx0.48$, $p$ from $101$ to $400009$ (fits over $p\geqq10^4$)", fontsize=10)
bx.legend(fontsize=7, loc="upper left", framealpha=0.95)
bx.set_xlim(L.min() - 0.3, L.max() + 0.3)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "scalloped_anatomy.pdf"))
fig.savefig(os.path.join(FIG, "scalloped_anatomy.png"), dpi=160)
print("written figures/scalloped_anatomy.pdf/.png")
