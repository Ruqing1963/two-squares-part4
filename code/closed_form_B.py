r"""
The closed form of B in ||G_{1,p}||^2 = (1/pi)(log p)^2 + B log p + O(1)   (Part IV, Section 2.2).

Every piece of the anatomy (results/subleading_log_anatomy.txt) that grows like log p is a sum over the
"diagonal" alpha = alpha' of the tail T'(k) = sum_{alpha >= 1} r(alpha) r(k - p alpha), k >= p, where
sum_{a <= x} r(a)^2 = 4 x log x + O(x) supplies the log p; everything else (Siegel main term, cross terms,
shifted convolutions) is O(1).  With E_1(x) = int_x^infty e^{-t}/t dt, s = sqrt(1 - x^2) and
I_j(v_0; lam) = int_{v_0}^infty Psi_j(lam v)/(lam v) dv the pieces are

   b_1 = (c_1 - 2 log(4 pi^2))/(2 pi)                                  free part, height 1   (Part II, Lemma 6.3)
   b_2 = 2 kappa/pi                                                     arcs on the free part (Proposition 2.2)
   b_3 = (1/pi) sum_{b>=1} r(b)^2 (E_1(pi b) + E_1(4 pi b))             tail k >= p, main cusps, height 1
   b_4 = (1/pi) sum_{b>=1} r(b)^2 int_{-1/2}^{1/2} (E_1(pi b s) + E_1(4 pi b s) - E_1(pi b) - E_1(4 pi b)) dx
                                                                        arcs on the tail
   b_5 = (1/pi) sum_{b = 2 (4)} r(b)^2 E_1(pi b)                        tail of the cusp of denominator 2, width 4p
   b_5'= (1/pi) sum_{b = 2 (4)} r(b)^2 int (E_1(pi b s) - E_1(pi b)) dx  its arcs
   b_6 = sum_{j>=1} [ 4 sum_{b>=0} r(b) r(b+j) I_j(2b+j; 2pi) + 2 sum_{b>=0} r(b) r(b+4j) I_j(b+2j; pi) ]
                                                                        correlations at the main cusps (r(0) = 1)
   b_6'= sum_{j>=1} 2 sum_{b = 2 (4)} r(b) r(b+4j) I_j(b+2j; pi)       correlations at the cusp of denominator 2
and B = b_1 + ... + b_6'.  The narrow cusps (widths 4, 1) have the weights e^{-pi k}, e^{-4 pi k} and no log p.

The script (i) evaluates the closed forms, (ii) checks them against the local slopes of the exact anatomy,
and (iii) decomposes C_2 (main cusps, height 1 and arcs) at several p into the diagonal part and the rest,
which must be flat in log p.   Usage: python closed_form_B.py [p ...]
"""
import os
import re
import sys
import numpy as np
from math import pi, log, sqrt
from scipy.special import exp1
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
from sup_error_scan import r_table, sigma_over_n_odd      # noqa: E402
from local_densities import siegel_all                    # noqa: E402
from subleading_log_anatomy import Psi, KAPPA, C1_LEMMA   # noqa: E402

RES = os.path.join(HERE, '..', 'results')
PRIMES = [int(a) for a in sys.argv[1:]] or [10007, 40009, 100003, 200003, 400009]
CUT = 14
BMAX = 40
xg, wg = np.polynomial.legendre.leggauss(200)
xg, wg = 0.5 * xg, 0.5 * wg
sg = np.sqrt(1 - xg ** 2)

r = r_table(BMAX * 4 + 200)
r0 = r.astype(float)
r0[0] = 1.0                                               # r(0) = 1: the free part is the term alpha = 0 of T


def arc_E1(lam, b):
    """int_{-1/2}^{1/2} (E_1(lam b s) - E_1(lam b)) dx"""
    return float(wg @ (exp1(lam * b * sg) - exp1(lam * b)))


def I_j(j, v0, lam):
    f = lambda v: float(Psi(j, np.array([lam * v]))[0]) / (lam * v)
    val, _ = quad(f, v0, v0 + 60.0 / lam, limit=200)
    return val


lines = []
b1 = (C1_LEMMA - 2 * log(4 * pi ** 2)) / (2 * pi)
b2 = 2 * KAPPA / pi
b3 = sum(r[b] ** 2 * (exp1(pi * b) + exp1(4 * pi * b)) for b in range(1, BMAX)) / pi
b4 = sum(r[b] ** 2 * (arc_E1(pi, b) + arc_E1(4 * pi, b)) for b in range(1, BMAX)) / pi
odd = [b for b in range(2, BMAX, 4)]
b5 = sum(r[b] ** 2 * exp1(pi * b) for b in odd) / pi
b5a = sum(r[b] ** 2 * arc_E1(pi, b) for b in odd) / pi
b6 = 0.0
b6o = 0.0
for j in range(1, 9):
    t = 0.0
    for b in range(0, BMAX):
        if r0[b] * r0[b + j]:
            t += 4 * r0[b] * r0[b + j] * I_j(j, 2 * b + j, 2 * pi)
        if r0[b] * r0[b + 4 * j]:
            t += 2 * r0[b] * r0[b + 4 * j] * I_j(j, b + 2 * j, pi)
    b6 += t
    b6o += sum(2 * r[b] * r[b + 4 * j] * I_j(j, b + 2 * j, pi) for b in odd if r[b] * r[b + 4 * j])
B = b1 + b2 + b3 + b4 + b5 + b5a + b6 + b6o
lines += [f"E_1(pi) = {exp1(pi):.7f}, E_1(2 pi) = {exp1(2 * pi):.7f}, E_1(4 pi) = {exp1(4 * pi):.3e}",
          f"b_1 (free part, Part II Lemma 6.3)          = {b1:+.6f}",
          f"b_2 = 2 kappa/pi (arcs on the free part)    = {b2:+.6f}",
          f"b_3 (tail, main cusps, height 1)            = {b3:+.6f}",
          f"b_4 (arcs on the tail)                      = {b4:+.6f}   [b_2 + b_4 = {b2 + b4:+.6f}]",
          f"b_5 (tail, cusp of denominator 2, width 4p) = {b5:+.6f}",
          f"b_5' (its arcs)                             = {b5a:+.6f}",
          f"b_6 (correlations, main cusps)              = {b6:+.6f}",
          f"b_6' (correlations, cusp of denominator 2)  = {b6o:+.6f}",
          f"B = sum                                     = {B:+.6f}", ""]

# ---------------------------------------------------------------- against the exact anatomy: local slopes
rows = []
for line in open(os.path.join(RES, 'subleading_log_anatomy.txt'), encoding='utf-8'):
    v = line.split()
    if len(v) >= 16 and v[0].isdigit():
        rows.append([float(x) for x in v[:15]])
rows = np.array(rows)
P = rows[:, 0]
cols = {"C2": 3, "C3 odd 4p": 5, "C4 arc main": 7, "C4 arc free part": 8, "C4 arc other": 9, "C4 off main": 10,
        "C4 off other": 11}
pred = {"C2": b3, "C3 odd 4p": b5, "C4 arc main": b2 + b4, "C4 arc free part": b2, "C4 arc other": b5a,
        "C4 off main": b6, "C4 off other": b6o}
lines.append("local slopes (y(p_2) - y(p_1))/log(p_2/p_1) of the exact anatomy against the closed forms:")
for k, c in cols.items():
    y = rows[:, c]
    sl = np.diff(y) / np.diff(np.log(P))
    lines.append(f"   {k:>18}: predicted {pred[k]:+.5f}; local slopes " + " ".join(f"{s:+.5f}" for s in sl[-5:])
                 + f"  (p = {int(P[-6])} ... {int(P[-1])})")
ex = rows[:, 14] - np.log(P) ** 2 / pi
sl = np.diff(ex) / np.diff(np.log(P))
lines.append(f"   {'exact - log^2/pi':>18}: predicted {B:+.5f}; local slopes " + " ".join(f"{s:+.5f}" for s in sl[-5:]))
lines.append("")

# ---------------------------------------------------------------- the decomposition of C_2 at the main cusps
lines.append("C_2 at the main cusps = diagonal tail + rest; 'rest - rest(prev)' must be O(1/p), not O(log):")
lines.append(f"{'p':>8} {'C2 (height 1)':>14} {'diagonal':>10} {'rest':>10} | {'C2 (arcs)':>10} {'diag arcs':>10} {'rest':>10}")
Kmax = CUT * max(PRIMES) + 1
rt = r_table(Kmax)
f_sigma, v2 = sigma_over_n_odd(Kmax)
for p in PRIMES:
    K = CUT * p
    T, M = siegel_all(1, p, K, rt[:K + 1], f_sigma[:K + 1], v2[:K + 1])
    a = T - M
    k = np.arange(1, K + 1, dtype=float)
    rr = rt[:K + 1].astype(float)
    c2 = c2arc = diag = diagarc = 0.0
    for Q in (p, 4 * p):
        c0 = 4 * pi * k / Q
        c2 += float(np.sum((a[1:] ** 2 - rr[1:] ** 2) * np.exp(-c0) / c0)) / Q
        c2arc += float(np.sum((a[1:] ** 2 - rr[1:] ** 2) * (Psi(0, c0) - np.exp(-c0)) / c0)) / Q
        for b in range(1, CUT):
            aa = np.arange(0, K - p * b + 1, dtype=float)
            kk = aa + p * b
            cc = 4 * pi * kk / Q
            w = rr[b] ** 2 * rr[:len(aa)] ** 2 / cc
            diag += float(np.sum(w * np.exp(-cc))) / Q
            diagarc += float(np.sum(w * (Psi(0, cc) - np.exp(-cc)))) / Q
    lines.append(f"{p:>8} {c2:>14.5f} {diag:>10.5f} {c2 - diag:>10.5f} | {c2arc:>10.5f} {diagarc:>10.5f} {c2arc - diagarc:>10.5f}")
    print(lines[-1], flush=True)
text = "\n".join(lines)
print(text)
with open(os.path.join(RES, "closed_form_B.txt"), "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
