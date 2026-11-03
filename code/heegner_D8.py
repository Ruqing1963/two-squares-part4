r"""
D = 8: the exact scalloped Parseval norm of G^{(8)}_{1,p}, Theta = theta_8(z) theta_8(pz), theta_8 = sum e((x^2+2y^2) z),
over ALL eight cusps of Gamma_0(8p) (denominators c = 1, 2, 4, 8, p, 2p, 4p, 8p; widths 8p, 2p, p, p, 8, 2, 1, 1), and the
closed form of B_8 in ||G||^2 = (1/(2 pi)) (log p)^2 + B_8 log p + O(1)   (Part IV, Section 3.7).

Expansions at the cusps.  For the quaternary form Q(x) = x1^2 + 2x2^2 + p x3^2 + 2p x4^2 (Gram matrix A = diag(2,4,2p,4p),
det A = 64 p^2) and rho_c = [[1,0],[c,1]], Poisson summation gives
     Theta|_2 rho_c (z) = -(8p)^{-1} c^{-2} sum_{xi in Z^4} G(xi; c) e(Q*(xi)/c) e(Q*(xi) z),
     G(xi; c) = prod_i g(q_i, xi_i; c),  g(a, xi; c) = sum_{h mod c} e((a h^2 + xi h)/c),  q = (1, 2, p, 2p),
     Q*(xi) = xi1^2/4 + xi2^2/8 + xi3^2/(4p) + xi4^2/(8p)                     (the dual form).
The Eisenstein part E of Theta is the combination sum_{d | 8p, d > 1} c_d (E_2(z) - d E_2(dz)) whose constant terms at the
eight cusps are those of Theta (constant term of E_2(dz)|rho_c is gcd(c,d)^2/d^2), and
     (E_2(z) - d E_2(dz))|rho_c = E_2(z) - (g^2/d) E_2((g^2 z + g m)/d),   g = gcd(c, d),  m = -u,  u (c/g) = -1 mod (d/g).
Both expansions are checked against direct evaluation of Theta(rho_c z)(cz+1)^{-2} and E(rho_c z)(cz+1)^{-2} at a point,
the constant-term system is checked for consistency (8 equations, 7 unknowns), the width of each cusp is read off the
support of the expansion, and the total is checked against the Rankin-Selberg estimate at p = 101 and the sandwich
Lambda_1 <= ||G||^2 <= Lambda_{sqrt3/2} at every p.

Closed form (Theorem 3.10 of the paper): the four Atkin-Lehner cusps give the five pieces of Theorem 3.8 with D = 8,
A_8 = 24/psi(8) = 2; the cusps of denominator 2 (width 2p) and 4 (width p) carry sum_{p k1 + k2 = 2K, k1, k2 odd}
(phases) r_8(k1) r_8(k2), copies of the odd part of r_8 (sum_{k <= X odd} r_8(k)^2 ~ X log X) at the positions p k1/2:
     b_5^{(2)} = (1/(2 pi)) sum_{k odd} r_8(k)^2 int E_1(pi k s) dx,        b_6^{(2)} = sum_j (-1)^j sum_{k odd} r_8(k) r_8(k+4j) I_j(k+2j; pi),
     b_5^{(4)} = (1/(2 pi)) sum_{k odd} r_8(k)^2 int E_1(2 pi k s) dx,      b_6^{(4)} = 2 sum_j sum_{k odd} r_8(k) r_8(k+2j) I_j(k+j; 2 pi),
and the four narrow cusps O(1).   Usage: python heegner_D8.py [check | p1 p2 ...]
"""
import os
import sys
import time
from math import pi, log, sqrt, gcd, lgamma
import numpy as np
from scipy.special import exp1
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]                                   # the imported scripts parse sys.argv at import time
from heegner_theta_unified import kronecker, sigma_over_n, EG, ZPZ2, norm_RS      # noqa: E402
from subleading_log_anatomy import Psi, KAPPA, JMAX                                # noqa: E402
from heegner_closed_form_B import closed_form, T_shift, arc_E1, I_j, X_RS, H      # noqa: E402
sys.argv = sys.argv[:1] + ARGV

D = 8
A8 = 2.0                     # 24/psi(8): residue coefficient of sum r_8(k)^2 k^{-s}
LEAD = A8 / (4 * pi)         # = 1/(2 pi)
PRIMES = [101, 1009, 4001, 10007, 20011, 40009, 100003, 200003]
TMAX = 3                     # expansions to t = k/Q <= 3 at every cusp (e^{-12 pi} at the last term)


def e(x):
    return np.exp(2j * pi * np.asarray(x, dtype=float))


def chi8(n):
    return kronecker(-8, n)


def r8_table(X):
    r = np.zeros(X + 1)
    for d in range(1, X + 1):
        c = chi8(d)
        if c:
            r[d::d] += c
    r *= 2
    r[0] = 1.0
    return r


def sigma_table(X):
    s = np.zeros(X + 1)
    for d in range(1, X + 1):
        s[d::d] += d
    return s


def gauss_direct(a, xi, c):
    h = np.arange(c)
    return complex(np.sum(e(((a * h * h + xi * h) % c) / c)))


def gauss_vec(a, xi, c, p):
    """g(a, xi; c) for an array xi, c = 2^e p or 2^e (e <= 3), by CRT and the closed form modulo p."""
    xi = np.asarray(xi, dtype=np.int64)
    if c % p:                                               # c | 8: direct table
        tab = np.array([gauss_direct(a % c, x, c) for x in range(c)])
        return tab[xi % c]
    c2 = c // p
    out = np.ones(len(xi), dtype=complex)
    if c2 > 1:
        pb = pow(p, -1, c2)
        tab = np.array([gauss_direct((a * pb) % c2, x, c2) for x in range(c2)])
        out *= tab[(xi * pb) % c2]
    ib = pow(c2, -1, p)
    beta = (a * ib) % p
    eta = (xi * ib) % p
    if beta == 0:
        out *= np.where(eta == 0, p, 0.0)
    else:
        inv4b = pow(4 * beta, -1, p)
        eps = 1.0 if p % 4 == 1 else 1j
        out *= e(-(eta * eta % p) * inv4b % p / p) * kronecker(beta, p) * eps * sqrt(p)
    return out


class D8:
    def __init__(self, p, tmax=TMAX):
        self.p = p
        self.L = 8 * p
        self.cusps = [1, 2, 4, 8, p, 2 * p, 4 * p, 8 * p]
        self.widths = {c: self.L // gcd(c * c, self.L) for c in self.cusps}
        self.weights = {c: self.L * phi(self.L // c) * phi(c) / (c * phi(self.L)) for c in self.cusps}
        self.tmax = tmax
        self.N = self.L * tmax                                    # n = 8p t <= N
        self.q = (1, 2, p, 2 * p)
        self.twoa = (4, 8, 4 * p, 8 * p)                          # Q*(xi) = sum xi_i^2 / twoa_i
        self.c_d = None

    # ---------------------------------------------------------------- Theta at the cusp c: coefficients on n = 8p t
    def theta_coeffs(self, c):
        p, L, N = self.p, self.L, self.N
        lists = []
        for i in range(4):
            bound = int(sqrt(N * self.twoa[i] / L)) + 1
            xi = np.arange(-bound, bound + 1, dtype=np.int64)
            g = gauss_vec(self.q[i], xi, c, p)
            keep = np.abs(g) > 1e-9
            lists.append((xi[keep], g[keep]))
        x3, g3 = lists[2]
        x4, g4 = lists[3]
        n34 = (2 * x3[:, None] ** 2 + x4[None, :] ** 2).ravel()
        w34 = (g3[:, None] * g4[None, :]).ravel()
        m = n34 <= N
        h34 = np.bincount(n34[m], weights=w34[m].real, minlength=N + 1) + 1j * np.bincount(n34[m], weights=w34[m].imag, minlength=N + 1)
        full = np.zeros(N + 1, dtype=complex)
        x1, g1 = lists[0]
        x2, g2 = lists[1]
        for a1, w1 in zip(x1, g1):
            for a2, w2 in zip(x2, g2):
                sh = 2 * p * a1 * a1 + p * a2 * a2
                if sh <= N:
                    full[sh:] += (w1 * w2) * h34[:N + 1 - sh]
        n = np.arange(N + 1, dtype=np.int64)
        return -(1.0 / L) / c ** 2 * e(n / (L * c)) * full          # coefficient of e(t z), t = n/(8p)

    # ---------------------------------------------------------------- the Eisenstein part
    def solve_eisenstein(self):
        divs = [d for d in self.cusps if d > 1]
        rows, rhs = [], []
        for c in self.cusps:
            kappa = -(1.0 / self.L) / c ** 2 * np.prod([gauss_vec(self.q[i], np.array([0]), c, self.p)[0] for i in range(4)])
            rows.append([1 - gcd(c, d) ** 2 / d for d in divs])
            rhs.append(kappa)
        Amat, b = np.array(rows, dtype=complex), np.array(rhs)
        sol, *_ = np.linalg.lstsq(Amat, b, rcond=None)
        self.c_d = dict(zip(divs, sol))
        self.kappa = dict(zip(self.cusps, rhs))
        return float(np.max(np.abs(Amat @ sol - b)))

    def eis_coeffs(self, c, sig):
        """coefficients of E|rho_c on n = 8p t (t > 0), from sigma table sig (sig[n] = sigma(n))."""
        p, L, N = self.p, self.L, self.N
        out = np.zeros(N + 1, dtype=complex)
        n = np.arange(N + 1, dtype=np.int64)                      # int64: k * inv below exceeds 2^31 for large p
        for d, cd in self.c_d.items():
            # E_2(z): t integer -> n multiple of 8p
            tint = (n % L == 0) & (n > 0)
            out[tint] += -24 * cd * sig[n[tint] // L]
            g = gcd(c, d)
            dg = d // g
            # (g^2/d) E_2((g^2 z + g m)/d): term e(k g^2 z/d) e(k g m/d), k >= 1: n = 8p k g^2/d
            step = L * g * g
            if step % d:
                continue
            step //= d
            k = n // step
            ok = (n % step == 0) & (n > 0)
            inv = pow(c // g, -1, dg) if dg > 1 else 0
            ph = e((k[ok] * inv) % dg / dg) if dg > 1 else 1.0
            out[ok] += 24 * cd * (g * g / d) * sig[k[ok]] * ph
        return out

    def eis_at_infty(self, X, sig):
        M = np.zeros(X + 1)
        k = np.arange(X + 1, dtype=np.int64)
        for d, cd in self.c_d.items():
            M += -24 * cd.real * (sig[:X + 1] - d * np.where(k % d == 0, sig[k // d], 0.0))
        return M

    # ---------------------------------------------------------------- direct values for the checks
    def theta8_value(self, w, r):
        terms = min(len(r) - 1, int(45 / (2 * pi * w.imag)) + 10)
        n = np.arange(terms + 1)
        return np.sum(r[:terms + 1] * np.exp(2j * pi * n * w))

    def Theta_value(self, w, r):
        return self.theta8_value(w, r) * self.theta8_value(self.p * w, r)

    def E_value(self, w, sig):
        def E2(z):
            terms = min(len(sig) - 1, int(45 / (2 * pi * z.imag)) + 10)
            n = np.arange(1, terms + 1)
            return 1 - 24 * np.sum(sig[1:terms + 1] * np.exp(2j * pi * n * z))
        return sum(cd * (E2(w) - d * E2(d * w)) for d, cd in self.c_d.items())


def phi(n):
    out, m, q = n, n, 2
    while q * q <= m:
        if m % q == 0:
            while m % q == 0:
                m //= q
            out -= out // q
        q += 1
    if m > 1:
        out -= out // m
    return out


def cusp_terms(b, Q, W, h=H):
    """flat_1, flat_h, arc, off for complex b (index k = 1..K), width Q, class weight W."""
    K = len(b) - 1
    k = np.arange(1, K + 1, dtype=float)
    c0 = 4 * pi * k / Q
    bb = np.abs(b[1:]) ** 2
    flat1 = W * float(np.sum(bb * np.exp(-c0) / c0))
    flath = W * float(np.sum(bb * np.exp(-h * c0) / c0))
    arc = W * float(np.sum(bb * Psi(0, c0) / c0)) - flat1
    off = 0.0
    for j in range(1, JMAX + 1):
        if j * Q >= K:
            break
        kk = np.arange(1, K - j * Q + 1)
        c = 2 * pi * (2 * kk + j * Q) / Q
        off += 2 * W * float(np.real(np.sum(b[kk] * np.conj(b[kk + j * Q]) * Psi(j, c) / c)))
    return flat1, flath, arc, off


def check(p):
    """consistency checks at a small prime."""
    print(f"--- checks at p = {p} ---")
    for (a, xi, c) in ((3, 5, 2 * p), (2, 7, 4 * p), (p, 11, 8 * p), (1, 4, p)):
        d, v = gauss_direct(a, xi, c), gauss_vec(a, np.array([xi]), c, p)[0]
        print(f"   Gauss sum g({a},{xi};{c}): direct {d:.6f}, CRT/closed form {v:.6f}, diff {abs(d - v):.1e}")
    obj = D8(p)
    res = obj.solve_eisenstein()
    print(f"   constant-term system: residual {res:.2e}; c_d = " + ", ".join(f"{d}: {v.real:+.6f}{v.imag:+.1e}i" for d, v in obj.c_d.items()))
    N = obj.N
    r = r8_table(max(N, 200000))
    sig = sigma_table(max(N, 200000))
    T = T_shift(r, p, N)
    M = obj.eis_at_infty(N, sig)
    print(f"   widths {obj.widths}, weights {obj.weights}")
    for c in obj.cusps:
        th = obj.theta_coeffs(c)
        Ec = obj.eis_coeffs(c, sig)
        Q = obj.widths[c]
        step = obj.L // Q
        n = np.arange(N + 1)
        offgrid = np.max(np.abs(th[n % step != 0])) if np.any(n % step != 0) else 0.0
        # direct evaluation at z = i y0
        z = 0.37 + 1.0j
        w = z / (c * z + 1)
        direct = obj.Theta_value(w, r) * (c * z + 1) ** -2
        series = np.sum(th * np.exp(2j * pi * n / obj.L * z))
        Edirect = obj.E_value(w, sig) * (c * z + 1) ** -2
        Eser = obj.kappa[c] + np.sum(Ec * np.exp(2j * pi * n / obj.L * z))
        print(f"   c = {c:>4} (Q = {Q:>4}): off-grid max {offgrid:.1e}; Theta|rho: direct {direct:.6f} series {series:.6f} "
              f"[{abs(direct - series):.1e}]; E|rho: direct {Edirect:.6f} series {Eser:.6f} [{abs(Edirect - Eser):.1e}]; "
              f"const Theta {obj.kappa[c]:.5f}")
        if c in (1, 8, p, 8 * p):
            b = (th - Ec)[::step]
            a = (T - M)[:len(b)]
            ratio = np.abs(b[1:40]) * Q / np.maximum(np.abs(a[1:40]), 1e-12)
            print(f"        AL cusp: |b(k)| Q / |a(k)| for k < 40: min {ratio[np.abs(a[1:40]) > 1e-6].min():.6f} max {ratio[np.abs(a[1:40]) > 1e-6].max():.6f}")
    # Atkin-Lehner signs
    for Q, pred in ((p, chi8(p)), (8, -chi8(p)), (8 * p, -1)):
        Np = 8 * p // Q
        dd = next(x for x in range(1, 10 * Np + 2) if (Q * x - 1) % Np == 0)
        bb = (Q * dd - 1) // Np
        Wm = np.array([[Q, bb], [8 * p, Q * dd]], dtype=float)
        z = 0.1 + 0.3j
        wz = (Wm[0, 0] * z + Wm[0, 1]) / (Wm[1, 0] * z + Wm[1, 1])
        val = Q * (Wm[1, 0] * z + Wm[1, 1]) ** -2 * obj.Theta_value(wz, r) / obj.Theta_value(z, r)
        print(f"   Theta|W_{Q} / Theta = {val:.6f} (predicted {pred})")
    # the odd part of r_8: sum_{k <= X odd} r_8(k)^2 / (X log X)
    X = 200000
    kk = np.arange(1, X + 1, 2)
    print(f"   sum_(k<=X odd) r_8(k)^2 / (X log X) at X = {X}: {np.sum(r[kk] ** 2) / (X * log(X)):.4f} (A_odd = 1, plus O(1/log X))")


def LpL1_8():
    """L'/L(1, chi_{-8}) by the formula of Theorem 3.6 with conductor 8."""
    return EG + log(2) + log(pi) - (2 / 2) * sum(chi8(a) * lgamma(a / 8) for a in range(1, 8))


def main():
    out = []

    def say(s=""):
        out.append(s)
        print(s, flush=True)

    primes = [int(a) for a in sys.argv[1:]] or PRIMES
    t0 = time.time()
    Kmax = 8 * max(primes) * TMAX
    r = r8_table(max(Kmax, X_RS))
    sig = sigma_table(max(Kmax, X_RS))
    say(f"r_8 and sigma tables to {max(Kmax, X_RS)} in {time.time() - t0:.0f}s")
    LpL = LpL1_8()
    a1 = 2 * EG + 2 * LpL - 2 * ZPZ2 + log(2) / 3
    c1 = A8 * (a1 - EG)
    say(f"L'/L(1, chi_-8) = {LpL:.6f}, c_1^(8) = {c1:.4f} (ramified factor (1 + 2^-s)^-1: log 2/3 in a_1)")
    b1, b2, b3, b4, b6 = closed_form(D, r, A8, c1)
    kodd = np.arange(1, 4 * D + 60, 2)
    b5_2 = sum(r[k] ** 2 * (exp1(pi * k) + arc_E1(pi, k)) for k in kodd if r[k]) / (2 * pi)
    b5_4 = sum(r[k] ** 2 * (exp1(2 * pi * k) + arc_E1(2 * pi, k)) for k in kodd if r[k]) / (2 * pi)
    b5_2h = sum(r[k] ** 2 * exp1(pi * k) for k in kodd if r[k]) / (2 * pi)
    b5_4h = sum(r[k] ** 2 * exp1(2 * pi * k) for k in kodd if r[k]) / (2 * pi)
    b6_2 = sum((-1) ** j * sum(r[k] * r[k + 4 * j] * I_j(j, k + 2 * j, pi) for k in kodd if r[k] * r[k + 4 * j]) for j in range(1, JMAX + 1))
    b6_4 = 2 * sum(sum(r[k] * r[k + 2 * j] * I_j(j, k + j, 2 * pi) for k in kodd if r[k] * r[k + 2 * j]) for j in range(1, JMAX + 1))
    B8 = b1 + b2 + b3 + b4 + b6 + b5_2 + b5_4 + b6_2 + b6_4
    say("closed form, D = 8 (A = 2, lead = 1/(2 pi) = %.5f):" % LEAD)
    say(f"   b_1 (free part, widths 8p, p)           = {b1:+.6f}")
    say(f"   b_2 = A kappa/(2 pi) (arcs)              = {b2:+.6f}")
    say(f"   b_3 (tail, height 1)                    = {b3:+.6f}   [E_1(pi/2) = {exp1(pi / 2):.5f}]")
    say(f"   b_4 (arcs on the tail)                  = {b4:+.6f}")
    say(f"   b_6 (correlations, AL cusps)            = {b6:+.6f}")
    say(f"   b_5^(2) (cusp 1/2, height 1 + arcs)     = {b5_2:+.6f}   [height 1 alone {b5_2h:+.6f}]")
    say(f"   b_6^(2) (cusp 1/2, correlations)        = {b6_2:+.6f}")
    say(f"   b_5^(4) (cusp 1/4, height 1 + arcs)     = {b5_4:+.6f}   [height 1 alone {b5_4h:+.6f}]")
    say(f"   b_6^(4) (cusp 1/4, correlations)        = {b6_4:+.6f}")
    say(f"   B_8 = sum                               = {B8:+.6f}")
    say("exact norm over the eight cusps of Gamma_0(8p), t = k/Q <= 3 at every cusp:")
    hdr = (f"   {'p':>7} {'C1 free':>10} {'C2 tail':>10} {'arcs AL':>10} {'corr AL':>10} {'cusp2 h1':>10} {'cusp2 arc':>10} {'cusp2 cor':>10} "
           f"{'cusp4 h1':>10} {'cusp4 arc':>10} {'cusp4 cor':>10} {'narrow':>9} {'Lambda_1':>10} {'Lambda_h':>10} {'exact':>10} {'RS 2^21':>9} {'ex-lead':>9} {'/log p':>8}")
    say(hdr)
    rows = []
    for p in primes:
        t1 = time.time()
        obj = D8(p)
        res = obj.solve_eisenstein()
        N = obj.N
        tot = {}
        T = T_shift(r, p, N)
        M = obj.eis_at_infty(N, sig)
        a = T - M
        aldev = 0.0
        for c in obj.cusps:
            Q = obj.widths[c]
            step = obj.L // Q
            b = (obj.theta_coeffs(c) - obj.eis_coeffs(c, sig))[::step]
            b[0] = 0.0
            if c in (1, 8, p, 8 * p):                              # Atkin-Lehner cusps: |b(k)| Q = |a(k)| exactly
                n_ = min(len(b), N + 1)
                aldev = max(aldev, float(np.max(np.abs(np.abs(b[1:n_]) * Q - np.abs(a[1:n_])))) / max(1.0, float(np.max(np.abs(a[1:n_])))))
            tot[c] = cusp_terms(b, Q, obj.weights[c])
        k = np.arange(1, N + 1, dtype=float)
        C1 = sum(float(np.sum(r[1:N + 1] ** 2 / k * np.exp(-4 * pi * k / Q))) for Q in (8 * p, p)) / (4 * pi)
        C2 = tot[1][0] + tot[8][0] - C1
        arcsAL = tot[1][2] + tot[8][2]
        corrAL = tot[1][3] + tot[8][3]
        narrow = sum(tot[c][0] + tot[c][2] + tot[c][3] for c in (p, 2 * p, 4 * p, 8 * p))
        lam1 = sum(v[0] for v in tot.values())
        lamh = sum(v[1] for v in tot.values())
        exact = sum(v[0] + v[2] + v[3] for v in tot.values())
        rs = float('nan')
        if p <= 1009:
            bb = np.zeros(X_RS + 1)
            bb[::p] = r[: X_RS // p + 1]
            Tn = np.round(fftconvolve(r[:X_RS + 1], bb)[: X_RS + 1])
            an = Tn - obj.eis_at_infty(X_RS, sig)
            rs = norm_RS(an, X_RS, 4 * pi * (p + 1))
        ex = exact - LEAD * log(p) ** 2
        rows.append((p, C1, C2, arcsAL, corrAL, tot[2][0], tot[2][2], tot[2][3], tot[4][0], tot[4][2], tot[4][3], narrow, lam1, lamh, exact, rs))
        say(f"   {p:>7} {C1:>10.4f} {C2:>10.5f} {arcsAL:>10.5f} {corrAL:>10.5f} {tot[2][0]:>10.5f} {tot[2][2]:>10.5f} {tot[2][3]:>10.5f} "
            f"{tot[4][0]:>10.5f} {tot[4][2]:>10.5f} {tot[4][3]:>10.5f} {narrow:>9.5f} {lam1:>10.4f} {lamh:>10.4f} {exact:>10.4f} {rs:>9.4f} "
            f"{ex:>9.4f} {ex / log(p):>8.4f}   [{time.time() - t1:.0f}s; Eisenstein residual {res:.0e}; AL |b|Q vs |a|: {aldev:.0e}]")
    P = np.array([row[0] for row in rows], dtype=float)
    L = np.log(P)
    cols = {"C2 tail": (2, b3), "arcs AL": (3, b2 + b4), "corr AL": (4, b6), "cusp2 h1": (5, b5_2h), "cusp2 arc": (6, b5_2 - b5_2h),
            "cusp2 cor": (7, b6_2), "cusp4 h1": (8, b5_4h), "cusp4 arc": (9, b5_4 - b5_4h), "cusp4 cor": (10, b6_4), "narrow": (11, 0.0)}
    say("local slopes (y(p_2) - y(p_1))/log(p_2/p_1) against the closed forms (last pairs of primes):")
    for name, (ci, pred) in cols.items():
        y = np.array([row[ci] for row in rows])
        sl = np.diff(y) / np.diff(L)
        say(f"   {name:>10}: predicted {pred:+.5f}; local slopes " + " ".join(f"{s:+.5f}" for s in sl[-4:]))
    ex = np.array([row[14] for row in rows]) - LEAD * L ** 2
    sl = np.diff(ex) / np.diff(L)
    say(f"   {'exact-lead':>10}: predicted {B8:+.5f}; local slopes " + " ".join(f"{s:+.5f}" for s in sl[-4:]))
    big = P >= 1000
    if big.sum() >= 3:
        Amat = np.vstack([L[big] ** 2, L[big], np.ones(big.sum())]).T
        q = np.linalg.lstsq(Amat, np.array([row[14] for row in rows])[big], rcond=None)[0]
        lin = np.linalg.lstsq(np.vstack([L[big], np.ones(big.sum())]).T, ex[big], rcond=None)[0]
        say(f"fits over p >= 1009: exact = a (log p)^2 + b log p + c with a = {q[0]:.4f} (1/(2 pi) = {LEAD:.4f}), b = {q[1]:+.4f}, "
            f"c = {q[2]:+.3f}; exact - lead (log p)^2 = b log p + c with b = {lin[0]:+.5f} (B_8 = {B8:+.5f}), c = {lin[1]:+.3f}")
    ok = all(row[12] <= row[14] <= row[13] for row in rows)
    say(f"sandwich Lambda_1 <= exact <= Lambda_h at every p: {ok}; at p = {int(P[0])}: exact = {rows[0][14]:.4f}, RS (window 2^21) = {rows[0][15]:.4f}, "
        f"difference {rows[0][14] - rows[0][15]:+.4f}")
    say("the block below repeats the exact norms in the format of heegner_closed_form_B.txt (for plot_heegner_volume_scaling.py and chi_grouped_slopes.py)")
    say(f"==== D = 8: A = 2, 1/(2 pi) = {LEAD:.5f}, c_1 = {c1:.4f} ====")
    say(f"   B_D = sum                           = {B8:+.6f}")
    say(f"         p    C1 free    C2 tail  arcs wide  corr wide    narrow   Lambda_1   Lambda_h      exact   RS 2^21   ex-lead   /log p")
    for row in rows:
        ex = row[14] - LEAD * log(row[0]) ** 2
        say(f"   {row[0]:>7} {row[1]:>10.4f} {row[2]:>10.5f} {row[3]:>10.5f} {row[4]:>10.5f} {row[11]:>9.5f} {row[12]:>10.4f} {row[13]:>10.4f} "
            f"{row[14]:>10.4f} {row[15]:>9.4f} {ex:>9.4f} {ex / log(row[0]):>8.4f}   [0s]")
    with open(os.path.join(RES, "heegner_D8.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check(int(sys.argv[2]) if len(sys.argv) > 2 else 13)
    else:
        main()
