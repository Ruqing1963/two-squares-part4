r"""
The term B log p in ||G_{1,p}||^2 = (1/pi)(log p)^2 + B log p + O(1)   (Part II, Theorem 6.14 and Remark 6.16).

EXACT NORM BY THE SCALLOPED PARSEVAL FORMULA.  Waibel's formula (Part II, (22)) writes ||G||^2 as a sum over the
cusp classes d | L = 4p of integrals of |G|_2 rho|^2 over the union of the L translates t.F, 0 <= t < L, of the
standard fundamental domain, i.e. over the scalloped strip {-1/2 <= x <= L - 1/2, y >= sqrt(1 - {x}^2)}, {x} the
distance to the nearest integer.  With G|_2 rho = sum_k b(k) e(k z/Q), Q the width of the cusp, Parseval over the
L unit periods of the scallop gives EXACTLY
    class(d) = w_d sum_{k >= 1} sum_{j} b(k) b(k + jQ) Psi_j(c_{k,j}) / c_{k,j},
    c_{k,j} = 2 pi (2k + jQ)/Q,     Psi_j(c) = int_{-1/2}^{1/2} cos(2 pi j x) exp(-c sqrt(1 - x^2)) dx,
with w_d = 1/Q for the Atkin-Lehner cusps (b = a_G, the coefficients of G_{1,p}; d = delta, 4 delta) and
w_d = (Q/4)(delta/p)^2 for the cusps of denominator 2 delta (b = c = T^odd - M^odd, Q = 4p/delta); here
delta in {1, p}, so the six classes have the widths 4p, p, 4p (odd) and 4, 1, 4 (odd).  The j = 0 terms with Psi_0
replaced by e^{-c} are the height-1 sums Lambda_1 of Part II, Section 6.7 (Lemmas 6.2, 6.11 and (34)); Psi_0(c) - e^{-c}
integrates over the region between the arcs and the line y = 1; the j != 0 terms are the correlations
a(k) a(k + jQ) that the arcs let through (they vanish for a flat bottom).  Only j <= 7 matters (c >= 2 pi j).

ANATOMY.  ||G_{1,p}||^2 - (1/pi)(log p)^2 is split into
   C1  = (1/4pi)(S(p/4pi) + S(p/pi)),  S(Y) = sum r(k)^2/k e^{-k/Y}: the free part of the two main cusps
         (widths p, 4p); Part II, Lemma 6.3: 2(log Y)^2 + c_1 log Y + c_2, so C1 = (1/pi)(log p)^2 + 0.38301 log p + O(1);
   C2  = Lambda_1(main cusps) - C1: the correction r(k)^2 -> |a(k)|^2 = (r(k) - M(k) + T'(k))^2 (Siegel main term
         for k < p, the terms T' with beta >= 1 for k >= p);
   C3  = Lambda_1 of the other four cusps: the narrow Atkin-Lehner cusps (widths 4, 1) and the two cusps of
         denominator 2 (widths 4p, 4);
   C4  = the part below height 1 = exact norm - Lambda_1, split into the diagonal arc terms (j = 0) of the main
         cusps, of the other cusps, and the correlation terms (j != 0).
Each component is fitted by b log p + c over the primes (C1 and the total also by a (log p)^2 + b log p + c).
Analytic predictions: C1 has b = (c_1 - 2 log(4 pi^2))/(2 pi) = 0.38312; the diagonal arc term of the free part of
the main cusps has b = 2 kappa/pi, kappa = int_{-1/2}^{1/2} log(1/sqrt(1-x^2)) dx = 0.045231 (Mellin: the arc
replaces Gamma(w) by Gamma(w) int (1-x^2)^{-w/2} dx = Gamma(w)(1 + kappa w + ...)), i.e. 0.028795 log p.
Checks: exact norm against the Rankin-Selberg estimate (p <= 2 10^4) and Lambda_1 <= exact <= Lambda_{sqrt3/2}.
Floating point.   Usage: python subleading_log_anatomy.py [p ...]
"""
import os
import sys
import time
import numpy as np
from math import pi, log, sqrt
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
from sup_error_scan import r_table, sigma_over_n_odd, factor      # noqa: E402
from local_densities import siegel_all                            # noqa: E402

PRIMES = [int(a) for a in sys.argv[1:]] or [101, 211, 401, 1009, 2003, 4001, 10007, 20011, 40009, 100003, 200003,
                                             400009, 1000003]
CUT = 14            # coefficients k <= CUT p (e^{-pi k/p} < e^{-44} beyond)
JMAX = 8            # correlation terms j <= JMAX (Psi_j(c) <= e^{-sqrt3 pi j} for c >= 2 pi j)
X_RS = 2 ** 21      # Rankin-Selberg window for the check (p <= 2 10^4)
H = sqrt(3) / 2
RES = os.path.join(HERE, '..', 'results')

# ---------------------------------------------------------------- Psi_j(c) on a log grid, cubic splines
xg, wg = np.polynomial.legendre.leggauss(240)
xg, wg = 0.5 * xg, 0.5 * wg                                      # nodes and weights on [-1/2, 1/2]
sg = np.sqrt(1 - xg ** 2)
CGRID = np.geomspace(1e-9, 70.0, 8000)
EXPT = np.exp(-np.outer(CGRID, sg))
SPL = {j: CubicSpline(np.log(CGRID), EXPT @ (wg * np.cos(2 * pi * j * xg))) for j in range(JMAX + 1)}
KAPPA = float(-(wg @ np.log(sg)))                                 # int log(1/sqrt(1-x^2)) dx = 0.045231
KAPPA1 = float(wg @ (1 - sg))                                     # int (1 - sqrt(1-x^2)) dx = 0.043422


def Psi(j, c):
    c = np.asarray(c, dtype=float)
    out = np.zeros_like(c)
    m = c < 70.0
    out[m] = SPL[j](np.log(np.maximum(c[m], 1e-9)))
    return out


def class_terms(b, Q, weight, h=1.0):
    """For G|rho = sum b(k) e(kz/Q): (flat_h, arc, off) with flat_h = weight sum b^2 e^{-c h}/c (height h),
    arc = weight sum b^2 (Psi_0(c) - e^{-c})/c and off = 2 weight sum_{j>=1} sum_k b(k) b(k+jQ) Psi_j(c)/c."""
    K = len(b) - 1
    k = np.arange(1, K + 1, dtype=float)
    c0 = 4 * pi * k / Q
    bb = b[1:] ** 2
    flat_h = weight * float(np.sum(bb * np.exp(-h * c0) / c0))
    flat = flat_h if h == 1.0 else weight * float(np.sum(bb * np.exp(-c0) / c0))
    arc = weight * float(np.sum(bb * Psi(0, c0) / c0)) - flat
    off = 0.0
    for j in range(1, JMAX + 1):
        if j * Q >= K:
            break
        kk = np.arange(1, K - j * Q + 1)
        c = 2 * pi * (2 * kk + j * Q) / Q
        m = c < 70.0
        if not m.any():
            break
        kk, c = kk[m], c[m]
        off += 2 * weight * float(np.sum(b[kk] * b[kk + j * Q] * Psi(j, c) / c))
    return flat_h, arc, off


def S_free(r, Y, K):
    k = np.arange(1, K + 1, dtype=float)
    return float(np.sum(r[1:K + 1] ** 2 / k * np.exp(-k / Y)))


C1_LEMMA = 9.758   # c_1 of Part II, Lemma 6.3
C2_LEMMA = 6.12


def anatomy(p, r, f_sigma, v2):
    n, L = p, 4 * p
    K = CUT * p
    T, M = siegel_all(1, p, K, r[:K + 1], f_sigma[:K + 1], v2[:K + 1])
    a = T - M                                                     # a_{G_{1,p}}(k) = a_{G_{p,1}}(k)
    To, Mo = siegel_all(1, p, K, r[:K + 1], f_sigma[:K + 1], v2[:K + 1], odd=True)
    c_odd = To - Mo                                               # T^odd(1,p;k) - M^odd = the same for (p,1)
    out = {}
    # the six classes: (label, b, Q, weight)
    classes = [("AL width 4p", a, 4 * p, 1.0 / (4 * p)), ("AL width p", a, p, 1.0 / p),
               ("odd width 4p", c_odd, 4 * p, p * (1.0 / p) ** 2),
               ("AL width 4", a[:16 * 4 + 1], 4, 1.0 / 4), ("AL width 1", a[:16 + 1], 1, 1.0),
               ("odd width 4", c_odd[:16 * 4 + 1], 4, 1.0)]
    tot_flat1 = tot_flath = tot_arc = tot_off = 0.0
    for lab, b, Q, w in classes:
        flat1, arc, off = class_terms(b, Q, w, 1.0)
        flath, _, _ = class_terms(b, Q, w, H)
        out[lab] = (flat1, arc, off)
        tot_flat1 += flat1
        tot_flath += flath
        tot_arc += arc
        tot_off += off
    out["Lambda_1"] = tot_flat1
    out["Lambda_h"] = tot_flath
    out["exact"] = tot_flat1 + tot_arc + tot_off
    out["C1"] = (S_free(r, p / (4 * pi), K) + S_free(r, p / pi, K)) / (4 * pi)
    out["C1 asymptotic"] = (2 * log(p / (4 * pi)) ** 2 + 2 * log(p / pi) ** 2 + C1_LEMMA * (log(p / (4 * pi)) + log(p / pi))
                            + 2 * C2_LEMMA) / (4 * pi)
    out["C2"] = out["AL width 4p"][0] + out["AL width p"][0] - out["C1"]
    out["C3 narrow AL"] = out["AL width 4"][0] + out["AL width 1"][0]
    out["C3 odd 4p"] = out["odd width 4p"][0]
    out["C3 odd 4"] = out["odd width 4"][0]
    out["C4 arc main"] = out["AL width 4p"][1] + out["AL width p"][1]
    out["C4 arc other"] = tot_arc - out["C4 arc main"]
    out["C4 off main"] = out["AL width 4p"][2] + out["AL width p"][2]
    out["C4 off other"] = tot_off - out["C4 off main"]
    # the arc term of the free part alone (r(k)^2 in place of |a(k)|^2), whose log p coefficient is 2 kappa/pi
    arc_free = 0.0
    for Q in (4 * p, p):
        k = np.arange(1, K + 1, dtype=float)
        c0 = 4 * pi * k / Q
        arc_free += float(np.sum(r[1:K + 1] ** 2 * (Psi(0, c0) - np.exp(-c0)) / c0)) / Q
    out["C4 arc free part"] = arc_free
    out["RS"] = float('nan')
    if p <= 20011:
        Tn, Mn = siegel_all(1, p, X_RS, r[:X_RS + 1], f_sigma[:X_RS + 1], v2[:X_RS + 1])
        nn = np.arange(X_RS + 1, dtype=float)
        vol = (pi / 3) * L * 1.5 * (1 + 1 / p)
        out["RS"] = vol * float(np.mean((Tn[1:] - Mn[1:]) ** 2 / nn[1:])) / (16 * pi ** 2)
    return out


def fit(ps, ys, quad=False):
    x = np.log(np.array(ps, dtype=float))
    A = np.vstack([x ** 2, x, np.ones_like(x)]).T if quad else np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    return coef


if __name__ == "__main__":
    Kmax = CUT * max(PRIMES) + 1
    t0 = time.time()
    r = r_table(max(Kmax, X_RS + 1))
    f_sigma, v2 = sigma_over_n_odd(max(Kmax, X_RS + 1))
    lines = [f"kappa = int_(-1/2)^(1/2) log(1/sqrt(1-x^2)) dx = {KAPPA:.6f}, 2 kappa/pi = {2 * KAPPA / pi:.6f}; "
             f"kappa_1 = int (1 - sqrt(1-x^2)) dx = {KAPPA1:.6f}; tables to {max(Kmax, X_RS + 1)} in {time.time() - t0:.0f}s",
             f"Psi_0(c) - e^(-c) at c = 0.01, 0.1, 1: " + ", ".join(f"{float(Psi(0, c) - np.exp(-c)):.3e}" for c in (0.01, 0.1, 1.0)),
             ""]
    keys = ["C1", "C1 asymptotic", "C2", "C3 narrow AL", "C3 odd 4p", "C3 odd 4", "C4 arc main", "C4 arc free part",
            "C4 arc other", "C4 off main", "C4 off other", "Lambda_1", "Lambda_h", "exact", "RS"]
    hdr = f"{'p':>8} " + " ".join(f"{k:>14}" for k in keys)
    lines.append(hdr)
    print(hdr, flush=True)
    rows = {}
    for p in PRIMES:
        t1 = time.time()
        o = anatomy(p, r, f_sigma, v2)
        rows[p] = o
        line = f"{p:>8} " + " ".join(f"{o[k]:>14.5f}" for k in keys) + f"   [{time.time() - t1:.0f}s]"
        lines.append(line)
        print(line, flush=True)
    lines.append("")
    lines.append("minus (1/pi)(log p)^2: exact and C1; check Lambda_1 <= exact <= Lambda_h and exact vs RS")
    for p in PRIMES:
        o = rows[p]
        lines.append(f"   p = {p:>8}: exact - (log p)^2/pi = {o['exact'] - log(p) ** 2 / pi:>9.5f}, "
                     f"C1 - (log p)^2/pi = {o['C1'] - log(p) ** 2 / pi:>9.5f}, (exact - (log p)^2/pi)/log p = "
                     f"{(o['exact'] - log(p) ** 2 / pi) / log(p):.5f}; Lambda_1 <= exact <= Lambda_h: "
                     f"{o['Lambda_1'] <= o['exact'] <= o['Lambda_h']}; exact - RS = {o['exact'] - o['RS']:+.4f}")
    lines.append("")
    lines.append("fits b log p + c over all primes (and over p >= 10^4 in brackets); C1 and exact also a log^2 + b log + c:")
    big = [p for p in PRIMES if p >= 10000]
    for k in ["C1", "C2", "C3 narrow AL", "C3 odd 4p", "C3 odd 4", "C4 arc main", "C4 arc free part", "C4 arc other",
              "C4 off main", "C4 off other", "Lambda_1", "exact"]:
        ys = [rows[p][k] for p in PRIMES]
        if k in ("C1", "Lambda_1", "exact"):
            q = fit(PRIMES, ys, quad=True)
            qb = fit(big, [rows[p][k] for p in big], quad=True) if len(big) >= 4 else [float('nan')] * 3
            lines.append(f"   {k:>18}: a = {q[0]:.5f}, b = {q[1]:+.5f}, c = {q[2]:+.3f}   [a = {qb[0]:.5f}, b = {qb[1]:+.5f}]"
                         f"   (1/pi = {1 / pi:.5f})")
            ys2 = [rows[p][k] - log(p) ** 2 / pi for p in PRIMES]
            l2 = fit(PRIMES, ys2)
            lb = fit(big, [rows[p][k] - log(p) ** 2 / pi for p in big]) if len(big) >= 3 else [float('nan')] * 2
            lines.append(f"   {k:>18} - (log p)^2/pi: b = {l2[0]:+.5f}, c = {l2[1]:+.3f}   [b = {lb[0]:+.5f}, c = {lb[1]:+.3f}]")
        else:
            l = fit(PRIMES, ys)
            lb = fit(big, [rows[p][k] for p in big]) if len(big) >= 3 else [float('nan')] * 2
            lines.append(f"   {k:>18}: b = {l[0]:+.5f}, c = {l[1]:+.3f}   [b = {lb[0]:+.5f}, c = {lb[1]:+.3f}]")
    lines.append(f"   predictions: C1: b = (c_1 - 2 log(4 pi^2))/(2 pi) = {(C1_LEMMA - 2 * log(4 * pi ** 2)) / (2 * pi):.5f}; "
                 f"C4 arc free part: b = 2 kappa/pi = {2 * KAPPA / pi:.5f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RES, "subleading_log_anatomy.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
