r"""
Figure 2 of Part IV: (a) Lambda_1(G^{(D)}_{1,p})/(log p)^2 against p for the eight computed discriminants (D = 4 from Part II,
results/cusp_lower_bound_1p.txt there: Lambda(1,p) at p = 101, ..., 1000003; the odd D from results/heegner_theta_unified_*.txt),
with the limits 2/vol(Gamma_0(D)\H) = 6/(pi psi(D)); (b) the fitted coefficient of (log p)^2 against D with the curve 6/(pi psi(D)).
Usage: python plot_heegner_volume_scaling.py
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import pi, log

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
COLORS = {3: "#2a78d6", 4: "#0b0b0b", 7: "#eb6834", 11: "#1baf7a", 19: "#8e44ad", 43: "#c9a227", 67: "#d6336c", 163: "#5c6b73"}


def psi(D):
    out, d, n = D, 2, D
    while d * d <= n:
        if n % d == 0:
            out = out // d * (d + 1)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        out = out // n * (n + 1)
    return out


data = {}      # D -> (p array, Lambda_1 array, fitted a)
for path in sorted(glob.glob(os.path.join(RES, 'heegner_theta_unified_*.txt'))):
    D = None
    in_table = False
    for line in open(path, encoding='utf-8'):
        m = re.match(r"=+ D = (\d+):", line)
        if m:
            D = int(m.group(1))
            data[D] = ([], [], None)
            in_table = False
            continue
        if D is None:
            continue
        if re.match(r"\s+p\s+RS", line):
            in_table = True
            continue
        if in_table:
            vals = line.split()
            if len(vals) >= 7 and vals[0].isdigit():
                data[D][0].append(float(vals[0]))
                data[D][1].append(float(vals[2]))
            else:
                in_table = False
        m = re.search(r"fit Lambda_1 = .*: a = ([\d.]+)", line)
        if m:
            data[D] = (data[D][0], data[D][1], float(m.group(1)))
# D = 4 from Part II (Section 6.6(f) there): Lambda(1,p) over the Atkin-Lehner cusps and the cusps of denominator 2
data[4] = ([101, 1009, 10007, 100003, 300007, 1000003], [7.247, 16.678, 29.466, 45.665, 54.586, 65.245], 0.319)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4))
for D in sorted(data):
    p, lam, a = data[D]
    p, lam = np.array(p, dtype=float), np.array(lam)
    sel = p >= 100
    lim = 6 / (pi * psi(D))
    ax.plot(p[sel], lam[sel] / np.log(p[sel]) ** 2, color=COLORS[D], marker="o", ms=4, lw=1.4, label=f"$D={D}$")
    ax.axhline(lim, color=COLORS[D], lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$\Lambda_1(G^{(D)}_{1,p})/(\log p)^2$")
ax.set_title(r"(a) towards $2/\mathrm{vol}(\Gamma_0(D)\backslash\mathbb{H})=6/(\pi\psi(D))$ (dotted)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.95)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

Ds = np.arange(3, 170)
bx.plot(Ds, [6 / (pi * psi(D)) for D in Ds], color="#898781", lw=1, ls="-", label=r"$6/(\pi\psi(D))$, $D$ prime")
for D in sorted(data):
    a = data[D][2]
    bx.plot([D], [a], color=COLORS[D], marker="o", ms=6, lw=0)
    bx.annotate(f"{D}", (D, a), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=COLORS[D])
bx.plot([], [], color="#0b0b0b", marker="o", lw=0, label=r"fitted coefficient of $(\log p)^2$ ($D=4$: Part II)")
bx.set_xscale("log")
bx.set_yscale("log")
bx.set_xlabel(r"$D$")
bx.set_ylabel(r"coefficient of $(\log p)^2$")
bx.set_title(r"(b) the growth constant against $D$ ($D=8$ not computed)", fontsize=10)
bx.legend(fontsize=7.5, loc="lower left", framealpha=0.95)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "heegner_volume_scaling.pdf"))
fig.savefig(os.path.join(FIG, "heegner_volume_scaling.png"), dpi=160)
print("written figures/heegner_volume_scaling.pdf/.png", {D: data[D][2] for D in sorted(data)})
