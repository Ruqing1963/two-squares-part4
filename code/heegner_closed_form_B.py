r"""
The closed form of B_D for the odd Heegner discriminants (Part IV, Theorem 3.7) against the exact scalloped Parseval
formula for G^{(D)}_{1,p}, D in {3, 7, 11, 19, 43, 67, 163}.

    ||G^{(D)}_{1,p}||^2 = (6/(pi (D+1))) (log p)^2 + B_D log p + O(1),      B_D = b_1 + b_2 + b_3 + b_4 + b_6,

with A = 24/(D+1) the residue coefficient of sum r_D(k)^2 k^{-s} (sum_{a<=X} r_D(a)^2 = A X log X + O(X)), the two wide
Atkin-Lehner cusps of widths Dp and p (so 4 pi p beta/Q in {4 pi beta/D, 4 pi beta}), E_1 the exponential integral,
s = sqrt(1 - x^2), I_j(v_0; lam) = int_{v_0}^infty Psi_j(lam v)/(lam v) dv, r_D(0) = 1:
   b_1 = (6/(pi(D+1))) log(D/(16 pi^2)) + c_1^{(D)}/(2 pi)                     free part of the two wide cusps (Theorem 3.5)
   b_2 = A kappa/(2 pi) = 12 kappa/(pi (D+1))                                   arcs on the free part
   b_3 = (A/(4 pi)) sum_{b>=1} r_D(b)^2 (E_1(4 pi b/D) + E_1(4 pi b))           tail k >= p of the wide cusps, height 1
   b_4 = (A/(4 pi)) sum_{b>=1} r_D(b)^2 int (E_1(4 pi b s/D) + E_1(4 pi b s) - E_1(4 pi b/D) - E_1(4 pi b)) dx   arcs on the tail
   b_6 = sum_{j>=1} [ A sum_{b>=0} r_D(b) r_D(b+j) I_j(2b+j; 2 pi) + (2A/D) sum_{b>=0} r_D(b) r_D(b+Dj) I_j(b+Dj/2; 4 pi/D) ]
                                                                                  correlations at the wide cusps
(no cusp of denominator 2: the level Dp is square-free; the narrow cusps, widths D and 1, carry e^{-4 pi k/D}, e^{-4 pi k}
and contribute O(1)).  For D = 4 the same expressions give b_3, b_4 and the main-cusp part of b_6 of Theorem 2.4.

The exact norm is the scalloped Parseval formula (Proposition 2.1) over the four Atkin-Lehner cusps, weight 1/Q and
b = a(k) at each, with the coefficients a(k) = T(k) - M(k) computed up to K = 3Dp, so that e^{-4 pi K/(Dp)} = e^{-12 pi}
at the widest cusp: the earlier runs (heegner_theta_unified.py) truncated at k <= 12p, which is not enough at the cusp
of width Dp when D >= 19 (e^{-48 pi/D} = 3.6e-4, 0.030, 0.105, 0.40 for D = 19, 43, 67, 163), so their Lambda_1,
Lambda_h and the fitted leading coefficients for these D are superseded by this script.  The primes are limited by
K <= 8.1e6 (memory).   Usage: python heegner_closed_form_B.py [D ...]   (default: all seven)
"""
import os
import sys
import time
import numpy as np
from math import pi, log, sqrt, ceil
from scipy.special import exp1
from scipy.integrate import quad
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
from heegner_theta_unified import Heegner, sigma_over_n, EG, ZPZ2, norm_RS      # noqa: E402
from subleading_log_anatomy import Psi, KAPPA, class_terms, S_free, JMAX       # noqa: E402
from sup_error_scan import r_table as r_table_gauss                            # noqa: E402

DS = [int(a) for a in sys.argv[1:]] or [3, 7, 11, 19, 43, 67, 163]
KMAX = 8_100_000
ALL_PRIMES = [101, 1009, 4001, 10007, 20011, 40009, 100003, 200003, 400009]
X_RS = 2 ** 21
H = sqrt(3) / 2
xg, wg = np.polynomial.legendre.leggauss(200)
xg, wg = 0.5 * xg, 0.5 * wg
sg = np.sqrt(1 - xg ** 2)


def arc_E1(lam, b):
    """int_{-1/2}^{1/2} (E_1(lam b s) - E_1(lam b)) dx"""
    return float(wg @ (exp1(lam * b * sg) - exp1(lam * b)))


def I_j(j, v0, lam):
    f = lambda v: float(Psi(j, np.array([lam * v]))[0]) / (lam * v)
    val, _ = quad(f, v0, v0 + 60.0 / lam, limit=200)
    return val


def closed_form(D, r, A, c1):
    """b_1 ... b_6 for the two wide cusps of widths Dp and p; r = r_D table with r[0] = 1."""
    lead = A / (4 * pi)
    BMAX = int(4 * D) + 40
    b1 = (2 * (A / 2) * (log(D / (4 * pi)) + log(1 / (4 * pi))) + 2 * c1) / (4 * pi)
    b2 = A * KAPPA / (2 * pi)
    b3 = lead * sum(r[b] ** 2 * (exp1(4 * pi * b / D) + exp1(4 * pi * b)) for b in range(1, BMAX) if r[b])
    b4 = lead * sum(r[b] ** 2 * (arc_E1(4 * pi / D, b) + arc_E1(4 * pi, b)) for b in range(1, BMAX) if r[b])
    b6 = 0.0
    for j in range(1, JMAX + 1):
        t = 0.0
        for b in range(0, BMAX):
            if r[b] * r[b + j]:
                t += A * r[b] * r[b + j] * I_j(j, 2 * b + j, 2 * pi)
            if r[b] * r[b + D * j]:
                t += (2 * A / D) * r[b] * r[b + D * j] * I_j(j, b + D * j / 2, 4 * pi / D)
        b6 += t
    return b1, b2, b3, b4, b6


def T_shift(r, p, K):
    """T(k) = sum_{alpha + p beta = k} r(alpha) r(beta), k <= K, r[0] = 1, by K/p shifted additions."""
    T = r[:K + 1].copy()
    for b in range(1, K // p + 1):
        if r[b]:
            T[p * b:] += r[b] * r[:K + 1 - p * b]
    return T


def main():
    out = []

    def say(s=""):
        out.append(s)
        print(s, flush=True)

    t0 = time.time()
    son = sigma_over_n(max(KMAX, X_RS))
    say(f"sigma_{{-1}} table to {max(KMAX, X_RS)} in {time.time() - t0:.0f}s; kappa = {KAPPA:.6f}")
    # ------------------------------------------------------------ D = 4: the generic expressions against Theorem 2.4
    r4 = r_table_gauss(14 * 4 * 8 + 400).astype(float)
    r4[0] = 1.0
    C1_LEMMA = 9.758
    b1, b2, b3, b4, b6 = closed_form(4, r4, 4.0, C1_LEMMA)
    say(f"D = 4 check (A = 4, widths 4p and p): b_1 = {b1:.6f} (0.383005), b_2 = {b2:.6f} (0.028794), b_3 = {b3:.6f} (0.056874), "
        f"b_4 = {b4:.6f} (0.011936), b_6 (main cusps) = {b6:.6f} (-0.000460)   [Theorem 2.4 / closed_form_B.py in brackets]")
    say()
    summary = []
    for D in DS:
        Hg = Heegner(D)
        A = 24 / (D + 1)
        lead = 6 / (pi * (D + 1))
        LpL = Hg.LpL1()
        a1 = 2 * EG + 2 * LpL - 2 * ZPZ2 + log(D) / (D + 1)
        c1 = A * (a1 - EG)
        primes = [p for p in ALL_PRIMES if p != D and 3 * D * p <= KMAX]
        K_D = 3 * D * max(primes)
        t1 = time.time()
        r = Hg.r_table(max(K_D, X_RS))          # r[0] = 1
        say(f"==================== D = {D}: A = 24/(D+1) = {A:.5f}, 6/(pi(D+1)) = {lead:.5f}, c_1 = {c1:.4f}; "
            f"r_D table to {max(K_D, X_RS)} in {time.time() - t1:.0f}s ====================")
        t1 = time.time()
        b1, b2, b3, b4, b6 = closed_form(D, r, A, c1)
        BD = b1 + b2 + b3 + b4 + b6
        say(f"closed form ({time.time() - t1:.0f}s):")
        say(f"   b_1 (free part, Theorem 3.5)        = {b1:+.6f}")
        say(f"   b_2 = 12 kappa/(pi(D+1)) (arcs)      = {b2:+.6f}")
        say(f"   b_3 (tail, height 1)                = {b3:+.6f}   [E_1(4 pi/D) = {exp1(4 * pi / D):.5f}]")
        say(f"   b_4 (arcs on the tail)              = {b4:+.6f}")
        say(f"   b_6 (correlations)                  = {b6:+.6f}")
        say(f"   B_D = sum                           = {BD:+.6f}")
        say(f"exact norm by the scalloped Parseval formula, four Atkin-Lehner cusps (widths Dp, p, D, 1), k <= 3Dp:")
        hdr = (f"   {'p':>7} {'C1 free':>10} {'C2 tail':>10} {'arcs wide':>10} {'corr wide':>10} {'narrow':>9} "
               f"{'Lambda_1':>10} {'Lambda_h':>10} {'exact':>10} {'RS 2^21':>9} {'ex-lead':>9} {'/log p':>8}")
        say(hdr)
        rows = []
        for p in primes:
            t1 = time.time()
            K = 3 * D * p
            T = T_shift(r, p, K)
            M = Hg.M(p, K, son[:K + 1])
            a = T - M
            del T, M
            tot = {}
            for Q in (D * p, p, D, 1):
                b = a[:min(K, 6 * Q) + 1]                     # e^{-4 pi k/Q} < e^{-75} beyond 6Q
                flat1, arc, off = class_terms(b, Q, 1.0 / Q, 1.0)
                flath, _, _ = class_terms(b, Q, 1.0 / Q, H)
                tot[Q] = (flat1, flath, arc, off)
            C1 = (S_free(r, D * p / (4 * pi), K) + S_free(r, p / (4 * pi), K)) / (4 * pi)
            C2 = tot[D * p][0] + tot[p][0] - C1
            arcs = tot[D * p][2] + tot[p][2]
            corr = tot[D * p][3] + tot[p][3]
            narrow = sum(tot[Q][0] + tot[Q][2] + tot[Q][3] for Q in (D, 1))
            lam1 = sum(tot[Q][0] for Q in tot)
            lamh = sum(tot[Q][1] for Q in tot)
            exact = sum(tot[Q][0] + tot[Q][2] + tot[Q][3] for Q in tot)
            rs = float('nan')
            if p <= 1009 or (p <= 10007 and D <= 7):
                bb = np.zeros(X_RS + 1)
                bb[::p] = r[: X_RS // p + 1]
                Tn = np.round(fftconvolve(r[:X_RS + 1], bb)[: X_RS + 1])
                an = Tn - Hg.M(p, X_RS, son[:X_RS + 1])
                rs = norm_RS(an, X_RS, (pi / 3) * D * p * (1 + 1 / D) * (1 + 1 / p))
                del bb, Tn, an
            del a
            rows.append((p, C1, C2, arcs, corr, narrow, lam1, lamh, exact, rs))
            ex = exact - lead * log(p) ** 2
            say(f"   {p:>7} {C1:>10.4f} {C2:>10.5f} {arcs:>10.5f} {corr:>10.5f} {narrow:>9.5f} {lam1:>10.4f} {lamh:>10.4f} "
                f"{exact:>10.4f} {rs:>9.4f} {ex:>9.4f} {ex / log(p):>8.4f}   [{time.time() - t1:.0f}s]")
        P = np.array([row[0] for row in rows], dtype=float)
        L = np.log(P)
        cols = {"C2 tail": (2, b3), "arcs wide": (3, b2 + b4), "corr wide": (4, b6), "narrow": (5, 0.0)}
        say("local slopes (y(p_2) - y(p_1))/log(p_2/p_1) against the closed forms (last pairs of primes):")
        for name, (c, pred) in cols.items():
            y = np.array([row[c] for row in rows])
            sl = np.diff(y) / np.diff(L)
            say(f"   {name:>10}: predicted {pred:+.5f}; local slopes " + " ".join(f"{s:+.5f}" for s in sl[-4:]))
        ex = np.array([row[8] for row in rows]) - lead * L ** 2
        sl = np.diff(ex) / np.diff(L)
        say(f"   {'exact-lead':>10}: predicted {BD:+.5f}; local slopes " + " ".join(f"{s:+.5f}" for s in sl[-4:])
            + f"   (p = {int(P[-5]) if len(P) > 4 else int(P[0])} ... {int(P[-1])})")
        big = P >= 1000
        Amat = np.vstack([L[big] ** 2, L[big], np.ones(big.sum())]).T
        q = np.linalg.lstsq(Amat, np.array([row[8] for row in rows])[big], rcond=None)[0]
        lin = np.linalg.lstsq(np.vstack([L[big], np.ones(big.sum())]).T, ex[big], rcond=None)[0]
        say(f"fits over p >= 1009: exact = a (log p)^2 + b log p + c with a = {q[0]:.4f} (6/(pi(D+1)) = {lead:.4f}), "
            f"b = {q[1]:+.4f}, c = {q[2]:+.3f}; exact - lead (log p)^2 = b log p + c with b = {lin[0]:+.5f} (B_D = {BD:+.5f}), "
            f"c = {lin[1]:+.3f}")
        ok = all(row[6] <= row[8] <= row[7] for row in rows)
        rs101 = rows[0][9]
        say(f"sandwich Lambda_1 <= exact <= Lambda_h at every p: {ok}; at p = {int(P[0])}: exact = {rows[0][8]:.4f}, "
            f"RS (window 2^21) = {rs101:.4f}, difference {rows[0][8] - rs101:+.4f}")
        say()
        summary.append((D, lead, b1, b2, b3, b4, b6, BD, q[0], lin[0], sl[-1], int(P[-1])))
        del r
        Hg.r_cache.clear()
    say("summary: D, 6/(pi(D+1)), b_1, b_2, b_3, b_4, b_6, B_D, fitted a (p >= 1009), fitted b, last local slope, p_max")
    for s in summary:
        say(f"   {s[0]:>4} {s[1]:.5f} {s[2]:+.5f} {s[3]:+.5f} {s[4]:+.5f} {s[5]:+.5f} {s[6]:+.5f} {s[7]:+.5f} | "
            f"{s[8]:.4f} {s[9]:+.4f} {s[10]:+.4f} {s[11]:>7}")
    with open(os.path.join(RES, "heegner_closed_form_B.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
