r"""
Figure 3 of Part IV: the three newforms of level 77 (a(7) = +1, a(11) = -1) carry both G^{(7)}_{1,11} and G^{(11)}_{1,7};
their shares of the two norms, and the central values L(1,phi), L(1,phi x chi_{-7}), L(1,phi x chi_{-11}) that decide them
through the Rankin-Selberg formula <Theta, phi> = C_D chi(p) eta_p L(1,phi) L(1,phi x chi_{-D}).  Data parsed from
results/heegner_theta_unified_3_7_11.txt.   Usage: python plot_level77_duality.py
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#898781"

blocks = {}      # (D, p) -> list of dicts
D = None
cur = None
for line in open(os.path.join(RES, 'heegner_theta_unified_3_7_11.txt'), encoding='utf-8'):
    m = re.match(r"=+ D = (\d+):", line)
    if m:
        D = int(m.group(1))
        continue
    m = re.match(r"\s+p =\s*(\d+) .*level 77,", line)
    if m:
        cur = (D, int(m.group(1)))
        blocks[cur] = []
        continue
    if cur is None:
        continue
    m = re.match(r"\s+a\(2\) = ([-+\d.]+), a\(\d+\) = ([-+\d.]+), a\(5\) = ([-+\d.]+), a\(\d+\) = ([-+\d.]+): NEW level 77, share ([\d.]+)", line)
    if m:
        blocks[cur].append({"a2": float(m.group(1)), "share": float(m.group(5))})
        continue
    m = re.search(r"L\(1,phi\) = ([\d.]+) .*L\(1,phi x chi\) = ([\d.]+)", line)
    if m and blocks[cur]:
        blocks[cur][-1]["L"] = float(m.group(1))
        blocks[cur][-1]["Lchi"] = float(m.group(2))
        continue
    if "predictions hold" in line:
        cur = None

A = sorted(blocks[(7, 11)], key=lambda d: d["a2"])
B = sorted(blocks[(11, 7)], key=lambda d: d["a2"])
labels = [f"$a(2)={d['a2']:+.3f}$" for d in A]
x = np.arange(3)
w = 0.36

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
ax.bar(x - w / 2, [d["share"] for d in A], w, color=ORANGE, label=r"share of $\|G^{(7)}_{1,11}\|^2$")
ax.bar(x + w / 2, [d["share"] for d in B], w, color=AQUA, label=r"share of $\|G^{(11)}_{1,7}\|^2$")
for i, (a, b) in enumerate(zip(A, B)):
    ax.text(i - w / 2, a["share"] + 0.02, f"{a['share']:.3f}", ha="center", fontsize=8)
    ax.text(i + w / 2, b["share"] + 0.02, f"{b['share']:.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylim(0, 1.15)
ax.set_ylabel("share of the norm")
ax.set_title(r"(a) the three newforms of level $77$ ($a(7)=+1$, $a(11)=-1$) in the two theta series", fontsize=10)
ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

bx.bar(x - w, [d["L"] for d in A], w * 0.95, color=INK, alpha=0.75, label=r"$L(1,\varphi)$")
bx.bar(x, [d["Lchi"] for d in A], w * 0.95, color=ORANGE, label=r"$L(1,\varphi\otimes\chi_{-7})$ (twist level $539$)")
bx.bar(x + w, [d["Lchi"] for d in B], w * 0.95, color=AQUA, label=r"$L(1,\varphi\otimes\chi_{-11})$ (twist level $847$)")
for i, (a, b) in enumerate(zip(A, B)):
    bx.text(i - w, a["L"] + 0.08, f"{a['L']:.2f}", ha="center", fontsize=7.5)
    bx.text(i, a["Lchi"] + 0.08, f"{a['Lchi']:.2f}", ha="center", fontsize=7.5)
    bx.text(i + w, b["Lchi"] + 0.08, f"{b['Lchi']:.2f}", ha="center", fontsize=7.5)
bx.set_xticks(x)
bx.set_xticklabels(labels, fontsize=8.5)
bx.set_ylim(0, 6.3)
bx.set_ylabel("central value")
bx.set_title(r"(b) $\langle\Theta^{(D)},\varphi\rangle=C_D\,\chi(p)\eta_p(\varphi)\,L(1,\varphi)L(1,\varphi\otimes\chi_{-D})$: the twist decides", fontsize=10)
bx.legend(fontsize=8, loc="upper right", framealpha=0.95)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "level77_duality.pdf"))
fig.savefig(os.path.join(FIG, "level77_duality.png"), dpi=160)
print("written figures/level77_duality.pdf/.png", A, B)
