r"""
Figure 2 of Part IV: (a) ||G^{(D)}_{1,p}||^2/(log p)^2 against p for all nine discriminants, from the exact scalloped
Parseval formula (D = 4: results/subleading_log_anatomy.txt, column 'exact'; the odd D: results/heegner_closed_form_B.txt;
D = 8: results/heegner_D8.txt),
with the limits 2/vol(Gamma_0(D)\H) = 6/(pi psi(D)); (b) the coefficient of (log p)^2 fitted with the proved B_D
(||G||^2 - B_D log p = a (log p)^2 + c over p >= 1009) against D, with the curve 6/(pi psi(D)).
Usage: python plot_heegner_volume_scaling.py
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
COLORS = {3: "#2a78d6", 4: "#0b0b0b", 7: "#eb6834", 8: "#7a5c3a", 11: "#1baf7a", 19: "#8e44ad", 43: "#c9a227", 67: "#d6336c", 163: "#5c6b73"}


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


data = {}      # D -> (p list, norm list, B_D)
for fname in ('heegner_closed_form_B.txt', 'heegner_D8.txt'):      # the odd D; D = 8 (block at the end of the file)
    D = None
    for line in open(os.path.join(RES, fname), encoding='utf-8'):
        m = re.match(r"=+ D = (\d+):", line)
        if m:
            D = int(m.group(1))
            data[D] = ([], [], None)
            continue
        if D is None:
            continue
        m = re.search(r"B_D = sum\s*=\s*([-+\d.]+)", line)
        if m:
            data[D] = (data[D][0], data[D][1], float(m.group(1)))
            continue
        vals = line.split()
        if len(vals) >= 12 and vals[0].isdigit() and vals[-1].endswith("s]"):
            data[D][0].append(float(vals[0]))
            data[D][1].append(float(vals[8]))
# D = 4: the exact norm of subleading_log_anatomy.txt (column 14) and B = 0.482043 (Theorem 2.4)
p4, n4 = [], []
for line in open(os.path.join(RES, 'subleading_log_anatomy.txt'), encoding='utf-8'):
    vals = line.split()
    if len(vals) >= 16 and vals[0].isdigit():
        p4.append(float(vals[0]))
        n4.append(float(vals[14]))
data[4] = (p4, n4, 0.482043)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4))
fitted = {}
for D in sorted(data):
    p, nrm, B = data[D]
    p, nrm = np.array(p, dtype=float), np.array(nrm)
    L = np.log(p)
    lim = 6 / (pi * psi(D))
    ax.plot(p, nrm / L ** 2, color=COLORS[D], marker="o", ms=4, lw=1.4, label=f"$D={D}$")
    ax.axhline(lim, color=COLORS[D], lw=0.8, ls=":")
    big = p >= 1000
    A = np.vstack([L[big] ** 2, np.ones(big.sum())]).T
    a, c = np.linalg.lstsq(A, nrm[big] - B * L[big], rcond=None)[0]
    fitted[D] = (a, lim)
ax.set_xscale("log")
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$\Vert G^{(D)}_{1,p}\Vert^2/(\log p)^2$")
ax.set_title(r"(a) towards $2/\mathrm{vol}(\Gamma_0(D)\backslash\mathbb{H})=6/(\pi\psi(D))$ (dotted)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.95)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

Ds = np.arange(3, 170)
bx.plot(Ds, [6 / (pi * psi(D)) for D in Ds], color="#898781", lw=1, ls="-", label=r"$6/(\pi\psi(D))$, $D$ prime")
for D in sorted(fitted):
    a, lim = fitted[D]
    bx.plot([D], [a], color=COLORS[D], marker="o", ms=6, lw=0)
    bx.annotate(f"{D}", (D, a), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=COLORS[D])
bx.plot([], [], color="#0b0b0b", marker="o", lw=0, label=r"fitted $a$ in $\Vert G\Vert^2-B_D\log p=a(\log p)^2+c$")
bx.set_xscale("log")
bx.set_yscale("log")
bx.set_xlabel(r"$D$")
bx.set_ylabel(r"coefficient of $(\log p)^2$")
bx.set_title(r"(b) the growth constant against $D$, all nine discriminants", fontsize=10)
bx.legend(fontsize=7.5, loc="lower left", framealpha=0.95)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "heegner_volume_scaling.pdf"))
fig.savefig(os.path.join(FIG, "heegner_volume_scaling.png"), dpi=160)
print("written figures/heegner_volume_scaling.pdf/.png; fitted a (B_D fixed) against 6/(pi psi(D)):")
for D in sorted(fitted):
    print(f"   D = {D:>3}: a = {fitted[D][0]:.4f}, 6/(pi psi(D)) = {fitted[D][1]:.4f}, ratio {fitted[D][0] / fitted[D][1]:.3f}")
