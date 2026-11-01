r"""
From the Gaussian integers (D = 4, Parts I-III) to the Eisenstein integers (D = 3): the theta series
    theta_3(z) = sum_{x,y} e((x^2+xy+y^2) z) = 1 + 6 sum_{n>=1} sigma_chi(n) e(nz)  in M_1(Gamma_0(3), chi),  chi = chi_{-3},
    Theta_{1,p} = theta_3(z) theta_3(pz) = E + G  in M_2(Gamma_0(3p)),   p != 3 prime (p = 2 allowed: level 6).
1. Main term (Siegel): M(k) = (4 pi^2 k/(3p)) prod_l beta_l(k), beta_l = (1 - l^-2) sigma_{-1}(l^{v_l(k)}) for l nmid 3p,
       beta_p(k) = (1 + chi(p)) - chi(p)(1 + 1/p) p^{-v_p(k)},   beta_3(k) = (1 + eps) - eps (4/3) 3^{-v_3(k)},  eps = -chi(p),
   by the singular/non-singular recursion (x^2+xy+y^2 = u^2 + 3v^2 over Z_3 with 2 invertible).  Checked against counts
   modulo 3^a and p^a, against T = M at level 6 (no cusp forms) and T - M proportional to f_15 at level 15.
2. Atkin-Lehner: Theta|_2 W_p = chi(p) Theta, Theta|_2 W_3 = -chi(p) Theta, Theta|_2 W_{3p} = -Theta (checked numerically).
   Hence <Theta, phi> != 0 for a newform phi of level 3p forces eta_p(phi) = chi(p) and eta_3(phi) = -chi(p), i.e.
   a_phi(3) = chi_{-3}(p), a_phi(p) = -chi_{-3}(p): the support is the half of the newforms with a_phi(3) = chi(p); there is
   no total vanishing as for D = 4 (Part II, Corollary 6.18), because at the ramified prime 3 both signs of eta_3 occur.
   Verified by the T_2-decomposition (Prony) of G_{1,p}.
3. Rankin-Selberg: the proof of Part II, Theorem 6.7 carries over with r_3 = 6 sigma_chi and L(1, chi_{-3}) = pi/(3 sqrt3):
       <Theta_{1,p}, phi> = (9 sqrt3/(2 pi^2)) chi(p) eta_p(phi) L(1, phi) L(1, phi x chi).      Verified numerically.
4. Norm: all four cusps of Gamma_0(3p) are Atkin-Lehner (widths 3p, p, 3, 1) and carry G_{1,p};
   S_3(Y) = sum r_3(k)^2/k e^{-k/Y} = 3 (log Y)^2 + c_1' log Y + c_2' (D_3(s) = 36 zeta^2 L(s,chi)^2/((1+3^-s) zeta(2s))),
   so ||G_{1,p}||^2 ~ (3/(2 pi)) (log p)^2; the sandwich Lambda_1 <= ||G||^2 <= Lambda_{sqrt3/2} is evaluated as in Part II.
Floating point.   Usage: python prime_discriminant_norm.py [--quick]
"""
import os
import sys
import time
import numpy as np
from math import pi, log, sqrt, lgamma, gcd
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
QUICK = '--quick' in sys.argv
EG = 0.57721566490153286
C3_RS = 9 * sqrt(3) / (2 * pi ** 2)


def chi3(k):
    k = np.asarray(k)
    return np.where(k % 3 == 0, 0, np.where(k % 3 == 1, 1, -1))


def chi3_int(k):
    return 0 if k % 3 == 0 else (1 if k % 3 == 1 else -1)


def r3_table(X):
    """r_3(n) = #{(x,y): x^2+xy+y^2 = n} = 6 sum_{d|n} chi_{-3}(d), r_3(0) = 1."""
    sig = np.zeros(X + 1)
    for d in range(1, X + 1):
        c = chi3_int(d)
        if c:
            sig[d::d] += c
    r = 6 * sig
    r[0] = 1.0
    return r


def sigma_over_n(X):
    s = np.zeros(X + 1)
    for d in range(1, X + 1):
        s[d::d] += 1.0 / d
    return s


def vp_table(X, p):
    v = np.zeros(X + 1, dtype=np.int64)
    q = p
    while q <= X:
        v[q::q] += 1
        q *= p
    return v


def theta_product(p, X, r3=None):
    """T(k) = sum_{a + p b = k} r_3(a) r_3(b), 0 <= k <= X."""
    if r3 is None:
        r3 = r3_table(X)
    b = np.zeros(X + 1)
    b[::p] = r3[: X // p + 1]
    return np.round(fftconvolve(r3[:X + 1], b)[: X + 1])


def siegel3(p, X, son=None):
    """M(k) = (4 pi^2 k/(3p)) prod_l beta_l(k)."""
    if son is None:
        son = sigma_over_n(X)
    k = np.arange(X + 1, dtype=float)
    chi = chi3_int(p)
    eps = -chi
    v3, vp = vp_table(X, 3), vp_table(X, p)
    good = (6 / pi ** 2) / ((1 - 1 / 9) * (1 - 1 / p ** 2))
    s3 = (1 - 3.0 ** -(v3 + 1)) / (1 - 1 / 3)
    sp = (1 - float(p) ** -(vp + 1)) / (1 - 1 / p)
    beta3 = (1 + eps) - eps * (4 / 3) * 3.0 ** (-v3)
    betap = (1 + chi) - chi * (1 + 1 / p) * float(p) ** (-vp)
    M = 4 * pi ** 2 * k / (3 * p) * good * son / s3 / sp * beta3 * betap
    M[0] = 0.0
    return M


def count_density(p, ell, a):
    """beta_ell(k) for k mod ell^a by counting (x,y,z,w) mod ell^a with N(x,y) + p N(z,w) = k (q = ell^a <= 1000)."""
    q = ell ** a
    x = np.arange(q)
    N = (np.add.outer(x * x, x * x) + np.outer(x, x)) % q            # x^2 + y^2 + xy
    dist = np.bincount(N.ravel(), minlength=q).astype(float)
    distp = np.zeros(q)
    for t in range(q):
        distp[(p * t) % q] += dist[t]
    cnt = np.round(np.fft.irfft(np.fft.rfft(dist) * np.fft.rfft(distp), q))
    return cnt / float(q) ** 3


def euler_series(X, step):
    s = np.zeros(X + 1)
    k = 0
    while True:
        e1, e2 = k * (3 * k - 1) // 2 * step, k * (3 * k + 1) // 2 * step
        if e1 > X and e2 > X:
            break
        if e1 <= X:
            s[e1] += (-1) ** k
        if e2 <= X and k > 0:
            s[e2] += (-1) ** k
        k += 1
    return s


def f15_table(X):
    """f_15 = eta(z) eta(3z) eta(5z) eta(15z) = q prod (1-q^n)(1-q^{3n})(1-q^{5n})(1-q^{15n})."""
    s = euler_series(X, 1)
    for st in (3, 5, 15):
        s = np.round(fftconvolve(s, euler_series(X, st))[: X + 1])
    f = np.zeros(X + 1)
    f[1:] = s[:X]
    return f


R3_CACHE = {}


def theta3_value(z):
    """theta_3(z) by its q-series, truncated where e^{-2 pi n Im z} < e^{-45}."""
    terms = int(45 / (2 * pi * z.imag)) + 10
    if terms not in R3_CACHE:
        R3_CACHE[terms] = r3_table(terms)
    n = np.arange(terms + 1)
    return np.sum(R3_CACHE[terms] * np.exp(2j * pi * n * z))


def Theta_value(p, z):
    return theta3_value(z) * theta3_value(p * z)


def check_AL(p):
    """(Theta|_2 W_Q)(z)/Theta(z) for Q = p, 3, 3p, with W_Q = [[Q, b], [N, Q d]] (c = 1, a = 1, Q d - (N/Q) b = 1) at a
    base point z = -Qd/N + i sqrt(Q)/N, where Im z = Im(W_Q z) = sqrt(Q)/N; predictions chi(p), -chi(p), -1."""
    N = 3 * p
    out = []
    for Q, pred in ((p, chi3_int(p)), (3, -chi3_int(p)), (3 * p, -1)):
        Np = N // Q
        d = next(dd for dd in range(1, 10 * Np + 2) if (Q * dd - 1) % Np == 0)
        b = (Q * d - 1) // Np
        W = np.array([[Q, b], [N, Q * d]], dtype=float)
        assert abs(np.linalg.det(W) - Q) < 1e-9
        z = -Q * d / N + 1j * sqrt(Q) / N
        wz = (W[0, 0] * z + W[0, 1]) / (W[1, 0] * z + W[1, 1])
        val = Q * (W[1, 0] * z + W[1, 1]) ** -2 * Theta_value(p, wz)       # (Theta|_2 W)(z) = det W (cz+d)^{-2} Theta(Wz)
        out.append((Q, pred, val / Theta_value(p, z)))
    return out


# ---------------------------------------------------------------- Hecke decomposition (Prony)
def hecke(v, ell, n_max):
    n = np.arange(n_max + 1)
    H = np.zeros(n_max + 1)
    H[1:] = v[ell * n[1:]]
    H[ell::ell] += ell * v[n[ell::ell] // ell]
    return H


def prony_eigs(v, ell, kmax, tol=1e-10):
    """Distinct T_ell-eigenvalues on the span of v, from s_j = (T_ell^j v)(1), j <= kmax."""
    seqs = [v[: ell ** kmax + 1].copy()]
    for _ in range(kmax):
        seqs.append(hecke(seqs[-1], ell, (len(seqs[-1]) - 1) // ell))
    s = np.array([w[1] for w in seqs])
    for d in range(1, kmax // 2 + 1):
        A = np.array([s[i:i + d] for i in range(len(s) - d)])
        coef = np.linalg.lstsq(A, s[d:], rcond=None)[0]
        if np.abs(A @ coef - s[d:]).max() < tol * np.abs(s).max():
            return np.sort(np.roots(np.r_[1, -coef[::-1]]).real)
    return None


def project(v, ell, lam, i, X):
    """The T_ell-eigencomponent of v for lam[i], on n <= X (v needed up to ell^(len(lam)-1) X)."""
    w = v[: ell ** (len(lam) - 1) * X + 1].copy()
    for j, lj in enumerate(lam):
        if j != i:
            w = hecke(w, ell, (len(w) - 1) // ell) - lj * w[: (len(w) - 1) // ell + 1]
    return w[: X + 1] / np.prod([lam[i] - lj for j, lj in enumerate(lam) if j != i])


def multiplicative(a, primes):
    for i, l1 in enumerate(primes):
        for l2 in primes[i + 1:]:
            if abs(a[l1 * l2] - a[l1] * a[l2]) > 1e-6 * max(1.0, abs(a[l1] * a[l2])):
                return False
    return True


def decompose(G, X, level_primes, test_primes=(2, 5, 7, 11, 13), kmax=24):
    """Split G (coefficients to >= 2^kmax) into Hecke eigencomponents on n <= X by T_2, then T_5, T_7, ... where needed."""
    comps = [G]
    for ell in test_primes:
        if ell in level_primes:
            continue
        new = []
        for v in comps:
            a = v / v[1] if abs(v[1]) > 1e-12 * np.abs(v[:X + 1]).max() else None
            if a is not None and multiplicative(a, [q for q in test_primes if q not in level_primes]):
                new.append(v)
                continue
            lam = prony_eigs(v, ell, min(kmax, int(np.floor(np.log(len(v) - 1) / np.log(ell)))))
            if lam is None or len(lam) == 1:
                new.append(v)
                continue
            Xi = min(X, (len(v) - 1) // ell ** (len(lam) - 1))
            for i in range(len(lam)):
                w = project(v, ell, lam, i, Xi)
                if abs(w[1]) > 1e-9 * max(1.0, np.abs(w).max()):
                    new.append(w)
        comps = new
    # drop duplicates (the same Hecke system reached through two mixtures), keeping the longest copy
    uniq = {}
    for v in comps:
        a = v / v[1]
        key = tuple(np.round(a[[2, 3, 5, 7, 11, 13]], 4))
        if key not in uniq or len(v) > len(uniq[key]):
            uniq[key] = v
    return list(uniq.values())


def hecke_ok(a, p):
    """Hecke relations at 2 and 5 (both prime to 3p) and multiplicativity: a clean single eigensystem."""
    if p == 5:
        return abs(a[4] - (a[2] ** 2 - 2)) < 1e-6 and abs(a[10] - a[2] * a[5]) < 1e-6
    return (abs(a[4] - (a[2] ** 2 - 2)) < 1e-6 and abs(a[25] - (a[5] ** 2 - 5)) < 1e-6
            and abs(a[10] - a[2] * a[5]) < 1e-6 and abs(a[14] - a[2] * a[7]) < 1e-6)


def L_at_1(coef, Nl, X):
    """L(1, phi) by the smoothed functional equation (weight 2, level Nl): S(t) + w S(1/t); root number w by t-independence."""
    nn = np.arange(X + 1, dtype=float)

    def S(t):
        return float(np.sum(coef[1:] / nn[1:] * np.exp(-2 * pi * nn[1:] * t / sqrt(Nl))))
    best = None
    for w in (1, -1):
        vals = [S(t) + w * S(1 / t) for t in (0.6, 1.0, 1.5)]
        spread = max(vals) - min(vals)
        if best is None or spread < best[0]:
            best = (spread, w, float(np.mean(vals)))
    return best


def norm_RS(a, X, vol):
    nn = np.arange(X + 1, dtype=float)
    return float(np.sum(a[1:X + 1] ** 2 / nn[1:X + 1])) * vol / (16 * pi ** 2 * X)


def main():
    out = []

    def say(s=""):
        out.append(s)
        print(s, flush=True)

    say("D = 3: theta_3(z) theta_3(pz), level 3p")
    # ---------------------------------------------------------------- 1. local densities and the main term
    say("\n1. Local densities against counts modulo ell^a (formula / count):")
    for p in (2, 5, 7, 11):
        for ell, a in ((3, 5), (p, 3 if p <= 7 else 2)):
            dens = count_density(p, ell, a)
            q = ell ** a
            errs = []
            for k in (1, 2, 3, 4, 6, 9, 12, 18, 27, p, 2 * p, 3 * p, p * p):
                kk = k % q
                v = 0
                t = k
                while t % ell == 0 and t:
                    t //= ell
                    v += 1
                if ell == 3:
                    eps = -chi3_int(p)
                    f = (1 + eps) - eps * (4 / 3) * 3.0 ** (-v)
                else:
                    chi = chi3_int(p)
                    f = (1 + chi) - chi * (1 + 1 / p) * float(p) ** (-v)
                if v <= a - 2:
                    errs.append(abs(f / dens[kk] - 1) if dens[kk] else abs(f))
            say(f"   p = {p:>2}, ell = {ell:>2}, a = {a}: max relative error {max(errs):.1e}")
    X = 2 ** 16
    r3 = r3_table(X)
    son = sigma_over_n(X)
    T = theta_product(2, X, r3)
    M = siegel3(2, X, son)
    say(f"   level 6 (p = 2, no cusp forms): max |T - M| over k <= {X}: {np.abs(T[1:] - M[1:]).max():.2e}")
    T = theta_product(5, X, r3)
    M = siegel3(5, X, son)
    a = T - M
    f15 = f15_table(X)
    c = a[1]
    say(f"   level 15 (p = 5): T - M = c f_15 with c = {c:.6f}; max |a - c f_15| = {np.abs(a[1:] - c * f15[1:]).max():.2e}; "
        f"f_15: a(2) = {f15[2]:+.0f}, a(3) = {f15[3]:+.0f}, a(5) = {f15[5]:+.0f} (prediction a(3) = chi(5) = -1, a(5) = -chi(5) = +1)")
    for p in (7, 11, 13):
        T = theta_product(p, X, r3)
        M = siegel3(p, X, son)
        a = T - M
        nn = np.arange(X + 1, dtype=float)
        ratios = [np.sum(a[1:y + 1] ** 2) / y ** 2 for y in (X // 4, X // 2, X)]
        say(f"   level {3 * p}: sum_(k<=y) a(k)^2 / y^2 at y = X/4, X/2, X: {ratios[0]:.4f} {ratios[1]:.4f} {ratios[2]:.4f} "
            f"(constant: T - M is a cusp form)")
    # ---------------------------------------------------------------- 2. Atkin-Lehner
    say("\n2. Atkin-Lehner: ratio (Theta|_2 W_Q)(z)/Theta(z) against the prediction (W_p: chi(p), W_3: -chi(p), W_3p: -1):")
    for p in (5, 7, 11, 13):
        res = check_AL(p)
        say(f"   p = {p:>2} (chi(p) = {chi3_int(p):+d}): " + ", ".join(f"W_{Q}: {r.real:+.6f}{r.imag:+.1e}i (pred {pred:+d})"
                                                                       for Q, pred, r in res))
    # ---------------------------------------------------------------- 3. spectral support and the Rankin-Selberg constant
    say("\n3. Hecke decomposition of G_{1,p} (T_2, then T_5, T_7, ... where components mix); newform of level 3p iff "
        "a(3) = +-1, a(9) = a(3)^2, a(p) = +-1; prediction for every newform component: a(3) = chi(p), a(p) = -chi(p)")
    KMAX = 22
    Xn = 2 ** 14
    Gmax = 2 ** KMAX
    r3g = r3_table(Gmax)
    song = sigma_over_n(Gmax)
    chi_arr = chi3(np.arange(Xn + 1)).astype(float)
    plist = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37] if not QUICK else [5, 7, 11, 13]
    rs_rows = []
    for p in plist:
        t0 = time.time()
        N = 3 * p
        G = theta_product(p, Gmax, r3g) - siegel3(p, Gmax, song)
        vol = (pi / 3) * N * (4 / 3) * (1 + 1 / p)
        normG = norm_RS(G, Xn, vol)
        comps = decompose(G, Xn, (3, p), kmax=KMAX)
        say(f"   p = {p:>2} (chi(p) = {chi3_int(p):+d}), level {N}: ||G||^2 (RS, X = 2^14) = {normG:.4f}, "
            f"{len(comps)} eigencomponents; dim S_2(Gamma_0({N})) = {(p - chi3_int(p) - 3) // 3}")
        ok_all = True
        total_share = 0.0
        for v in comps:
            a = v / v[1]
            Xv = len(v) - 1
            clean = hecke_ok(a, p)
            is_new = (abs(abs(a[3]) - 1) < 1e-6 and abs(a[9] - a[3] ** 2) < 1e-6 and abs(abs(a[p]) - 1) < 1e-6)
            Xs = min(Xn, Xv)
            share = norm_RS(v, Xs, vol) / normG if Xv >= 2000 else float('nan')
            if share == share:
                total_share += share
            kind = f"NEW level {N}" if is_new else f"old (level {p})"
            flag = "" if clean else "  [mixture/inaccurate: Hecke relations at 2, 5 fail]"
            if is_new:
                good = (abs(a[3] - chi3_int(p)) < 1e-6) and (abs(a[p] + chi3_int(p)) < 1e-6)
                ok_all &= good
                flag += "  <- prediction " + ("OK" if good else "VIOLATED")
            say(f"      a(2) = {a[2]:+.4f}, a(3) = {a[3]:+.4f}, a(5) = {a[5]:+.4f}, a(7) = {a[7]:+.4f}, a({p}) = {a[p]:+.4f}: "
                f"{kind}, share of ||G||^2 = {share:.3f} (X = 2^{int(np.log2(Xs))}){flag}")
            if is_new and clean and Xv >= Xn:
                sp, w, L1 = L_at_1(a[:Xn + 1], N, Xn)
                cand = [(Nl,) + L_at_1(chi_arr * a[:Xn + 1], Nl, Xn) for Nl in (N, 3 * N, 9 * N)]
                Nl, sp2, w2, L1c = min(cand, key=lambda t: t[1])
                normphi = norm_RS(a, Xn, vol)
                inner = v[1] * normphi                                  # <Theta, phi> = c ||phi||^2
                eta_p = -a[p]
                pred = C3_RS * chi3_int(p) * eta_p * L1 * L1c
                say(f"         L(1,phi) = {L1:.6f} (w = {w:+d}, spread {sp:.0e}), L(1,phi x chi) = {L1c:.6f} (twist level {Nl}, "
                    f"w = {w2:+d}, spread {sp2:.0e}), ||phi||^2 = {normphi:.5f}, <Theta,phi> = {inner:.6f}, "
                    f"(9 sqrt3/2pi^2) chi eta L L = {pred:.6f}, ratio = {inner / pred:.5f}")
                rs_rows.append((p, inner / pred))
        say(f"      all newform components obey a(3) = chi(p), a(p) = -chi(p): {ok_all}; shares sum to {total_share:.3f}"
            f"   [{time.time() - t0:.0f}s]")
    if rs_rows:
        say(f"   Rankin-Selberg ratios <Theta,phi> / ((9 sqrt3/(2 pi^2)) chi(p) eta_p L L): "
            + ", ".join(f"{r:.4f}" for _, r in rs_rows) + f"  (9 sqrt3/(2 pi^2) = {C3_RS:.6f}; D = 4 had 4/pi^2 = {4 / pi ** 2:.6f})")
    # ---------------------------------------------------------------- 4. the norm
    say("\n4. The norm. S_3(Y) = 3 (log Y)^2 + c_1' log Y + c_2' with")
    LpL = EG + log(2) + log(pi) - 3 * (lgamma(1 / 3) - lgamma(2 / 3))      # L'/L(1, chi_{-3})
    zpz2 = -0.5699609990552      # zeta'(2)/zeta(2)
    a1 = 2 * EG + 2 * LpL - 2 * zpz2 + log(3) / 4
    c1p = 6 * (a1 - EG)
    say(f"   L'/L(1,chi_-3) = {LpL:.6f} (gamma + log 2 + log pi - 3 log(Gamma(1/3)/Gamma(2/3))), c_1' = 6(a_1' - gamma) = {c1p:.4f}")
    try:
        import mpmath as mp
        mp.mp.dps = 20
        Lp = mp.diff(lambda s: mp.dirichlet(s, [0, 1, -1]), 1)
        L1 = mp.dirichlet(1, [0, 1, -1])
        say(f"   check with mpmath: L'/L(1,chi_-3) = {float(Lp / L1):.6f}, L(1,chi_-3) = {float(L1):.6f} (pi/(3 sqrt3) = {pi / (3 * sqrt(3)):.6f})")
    except Exception as exc:                                               # noqa: BLE001
        say(f"   (mpmath check skipped: {exc})")
    Xs = 2 ** 22
    r3s = r3_table(Xs)
    k = np.arange(Xs + 1, dtype=float)
    fitrows = []
    for Y in (1e2, 1e3, 1e4, 1e5):
        S = float(np.sum(r3s[1:] ** 2 / k[1:] * np.exp(-k[1:] / Y)))
        fitrows.append((log(Y), S))
        say(f"   S_3({Y:.0e}) = {S:.4f}; minus 3 (log Y)^2 - c_1' log Y = {S - 3 * log(Y) ** 2 - c1p * log(Y):.4f} (-> c_2')")
    main_coef = (3 * 2 * (log(3 / (4 * pi)) + log(1 / (4 * pi))) + 2 * c1p) / (4 * pi)
    say(f"   two main cusps (widths 3p, p): (1/4pi)(S_3(3p/4pi) + S_3(p/4pi)) = (3/(2 pi)) (log p)^2 + {main_coef:.4f} log p + O(1)"
        f"   [D = 4: (1/pi)(log p)^2 + 0.3831 log p]")
    say("   sandwich over the four cusps (widths 3p, p, 3, 1), Lambda_1 <= ||G||^2 <= Lambda_h, and the RS estimate:")
    say(f"   {'p':>8} {'RS (X=2^21)':>12} {'Lambda_1':>10} {'Lambda_h':>10} {'(3/2pi)log^2':>12} {'Lambda_1/log^2':>14} {'RS/log^2':>9}")
    CUT = 12
    plist2 = [5, 7, 11, 13, 101, 1009, 10007, 100003] if not QUICK else [5, 7, 101, 1009]
    Xrs = 2 ** 21
    r3b = r3_table(max(Xrs, CUT * max(plist2)) + 1)
    sonb = sigma_over_n(max(Xrs, CUT * max(plist2)) + 1)
    lam_rows = []
    for p in plist2:
        t0 = time.time()
        N = 3 * p
        K = CUT * p
        a = theta_product(p, K, r3b[:K + 1]) - siegel3(p, K, sonb[:K + 1])
        kk = np.arange(1, K + 1, dtype=float)
        a2 = a[1:] ** 2 / kk
        lam = {}
        for h in (1.0, sqrt(3) / 2):
            lam[h] = sum(float(np.sum(a2 * np.exp(-4 * pi * h * kk / Q))) for Q in (3 * p, p, 3, 1)) / (4 * pi)
        rs = float('nan')
        if p <= 20000:
            aa = theta_product(p, Xrs, r3b[:Xrs + 1]) - siegel3(p, Xrs, sonb[:Xrs + 1])
            rs = norm_RS(aa, Xrs, (pi / 3) * N * (4 / 3) * (1 + 1 / p))
        lam_rows.append((p, lam[1.0]))
        say(f"   {p:>8} {rs:>12.4f} {lam[1.0]:>10.4f} {lam[sqrt(3) / 2]:>10.4f} {3 / (2 * pi) * log(p) ** 2:>12.4f} "
            f"{lam[1.0] / log(p) ** 2:>14.4f} {rs / log(p) ** 2:>9.4f}   [{time.time() - t0:.0f}s]")
    if len(lam_rows) >= 4:
        x = np.log(np.array([p for p, _ in lam_rows if p >= 100], dtype=float))
        y = np.array([l for p, l in lam_rows if p >= 100])
        A = np.vstack([x ** 2, x, np.ones_like(x)]).T
        q = np.linalg.lstsq(A, y, rcond=None)[0]
        say(f"   fit Lambda_1(1,p) = a (log p)^2 + b log p + c over p >= 100: a = {q[0]:.4f} (3/(2pi) = {3 / (2 * pi):.4f}), "
            f"b = {q[1]:+.4f}, c = {q[2]:+.3f}")
    text = "\n".join(out)
    with open(os.path.join(RES, "prime_discriminant_norm.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
