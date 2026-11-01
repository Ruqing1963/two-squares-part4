r"""
The odd Heegner discriminants D in {3, 7, 11, 19, 43, 67, 163} (class number one, D prime, D = 3 mod 4):
    N_D(x,y) = x^2 + xy + ((D+1)/4) y^2,  r_D(n) = w_D sum_{d|n} chi_{-D}(d)  (w_3 = 6, w_D = 2 otherwise),
    theta_D(z) = sum r_D(n) e(nz) in M_1(Gamma_0(D), chi_{-D}),
    Theta_{1,p} = theta_D(z) theta_D(pz) = E + G in M_2(Gamma_0(Dp)),  p != D prime; the level Dp is square-free.
1. Main term: M(k) = (4 pi^2 k/(Dp)) prod_l beta_l(k), beta_l = (1 - l^-2) sigma_{-1}(l^{v_l(k)}) for l nmid Dp (also l = 2),
       beta_p(k) = (1 + chi(p)) - chi(p)(1 + 1/p) p^{-v_p(k)},  beta_D(k) = (1 + eps) - eps (1 + 1/D) D^{-v_D(k)},  eps = -chi_{-D}(p).
   Checked against counts modulo D^a, p^a and against the known cusp forms at small levels (f_15, f_21, f_14, f_11, f_33, ...).
2. Atkin-Lehner: Theta|W_p = chi(p) Theta, Theta|W_D = -chi(p) Theta, Theta|W_{Dp} = -Theta  (numerically to 1e-15), hence the
   support: newforms phi of level Dp with a_phi(D) = chi_{-D}(p), a_phi(p) = -chi_{-D}(p) (root number +1).  Verified by the
   Hecke decomposition of G_{1,p}.
3. Rankin-Selberg: <Theta, phi> = C_D chi(p) eta_p(phi) L(1,phi) L(1,phi x chi_{-D}),  C_D = w_D/(4 pi L(1,chi_{-D})) = w_D^2 sqrt D/(8 pi^2).
4. Norm: D_D(s) = sum r_D(k)^2 k^-s = w_D^2 zeta(s)^2 L(s,chi)^2 /((1 + D^-s) zeta(2s)), residue coefficient 24/(D+1), so
   S_D(Y) = (12/(D+1)) (log Y)^2 + c_1 log Y + c_2 and ||G_{1,p}||^2 ~ (6/(pi (D+1))) (log p)^2 (two wide cusps, widths Dp and p).
Floating point.   Usage: python heegner_theta_unified.py [D ...]   (default 3 7 11)
"""
import os
import sys
import time
import numpy as np
from math import pi, log, sqrt, lgamma, gcd
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
EG = 0.57721566490153286
ZPZ2 = -0.5699609990552        # zeta'(2)/zeta(2)
DS = [int(a) for a in sys.argv[1:]] or [3, 7, 11]


def kronecker(a, n):
    """Kronecker symbol (a/n), n >= 1."""
    if n == 1:
        return 1
    if n == 0:
        return 1 if abs(a) == 1 else 0
    res = 1
    while n % 2 == 0:
        n //= 2
        if a % 2 == 0:
            return 0
        if a % 8 in (3, 5):
            res = -res
    a %= n
    # Jacobi symbol (a/n) for odd n
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                res = -res
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            res = -res
        a %= n
    return res if n == 1 else 0


class Heegner:
    def __init__(self, D):
        assert D % 4 == 3 and D in (3, 7, 11, 19, 43, 67, 163)
        self.D = D
        self.w = 6 if D == 3 else 2
        self.c = (D + 1) // 4                      # N_D = x^2 + xy + c y^2
        self.C_RS = self.w ** 2 * sqrt(D) / (8 * pi ** 2)
        self.L1 = 2 * pi / (self.w * sqrt(D))      # L(1, chi_{-D}), h = 1
        self.r_cache = {}

    def chi(self, n):
        return kronecker(-self.D, n)

    def chi_table(self, X):
        return np.array([self.chi(n) for n in range(X + 1)], dtype=float)

    def r_table(self, X):
        if X in self.r_cache:
            return self.r_cache[X]
        sig = np.zeros(X + 1)
        for d in range(1, X + 1):
            cd = self.chi(d)
            if cd:
                sig[d::d] += cd
        r = self.w * sig
        r[0] = 1.0
        self.r_cache[X] = r
        return r

    def theta_value(self, z):
        terms = int(45 / (2 * pi * z.imag)) + 10
        r = self.r_table(terms)
        n = np.arange(terms + 1)
        return np.sum(r * np.exp(2j * pi * n * z))

    def Theta_value(self, p, z):
        return self.theta_value(z) * self.theta_value(p * z)

    def T(self, p, X, r=None):
        r = self.r_table(X) if r is None else r
        b = np.zeros(X + 1)
        b[::p] = r[: X // p + 1]
        return np.round(fftconvolve(r[:X + 1], b)[: X + 1])

    def M(self, p, X, son=None):
        D = self.D
        if son is None:
            son = sigma_over_n(X)
        k = np.arange(X + 1, dtype=float)
        chi = self.chi(p)
        eps = -chi
        vD, vp = vp_table(X, D), vp_table(X, p)
        good = (6 / pi ** 2) / ((1 - 1 / D ** 2) * (1 - 1 / p ** 2))
        sD = (1 - float(D) ** -(vD + 1)) / (1 - 1 / D)
        sp = (1 - float(p) ** -(vp + 1)) / (1 - 1 / p)
        betaD = (1 + eps) - eps * (1 + 1 / D) * float(D) ** (-vD)
        betap = (1 + chi) - chi * (1 + 1 / p) * float(p) ** (-vp)
        M = 4 * pi ** 2 * k / (D * p) * good * son / sD / sp * betaD * betap
        M[0] = 0.0
        return M

    def count_density(self, p, ell, a):
        q = ell ** a
        x = np.arange(q)
        x2 = (x * x) % q
        dist = np.zeros(q)
        for xi in range(q):                                   # row by row: q <= 3 10^4 is fine
            vals = (x2[xi] + xi * x + self.c * x2) % q
            dist += np.bincount(vals, minlength=q)
        distp = np.zeros(q)
        for t in range(q):
            distp[(p * t) % q] += dist[t]
        return np.round(np.fft.irfft(np.fft.rfft(dist) * np.fft.rfft(distp), q)) / float(q) ** 3

    def beta_formula(self, p, ell, k):
        v = 0
        t = k
        while t and t % ell == 0:
            t //= ell
            v += 1
        if ell == self.D:
            eps = -self.chi(p)
            return (1 + eps) - eps * (1 + 1 / ell) * float(ell) ** (-v)
        chi = self.chi(p)
        return (1 + chi) - chi * (1 + 1 / ell) * float(ell) ** (-v)

    def check_AL(self, p):
        D, N = self.D, self.D * p
        out = []
        for Q, pred in ((p, self.chi(p)), (D, -self.chi(p)), (D * p, -1)):
            Np = N // Q
            d = next(dd for dd in range(1, 10 * Np + 2) if (Q * dd - 1) % Np == 0)
            b = (Q * d - 1) // Np
            W = np.array([[Q, b], [N, Q * d]], dtype=float)
            assert abs(np.linalg.det(W) - Q) < 1e-9
            z = 0.1 + 0.3j                                   # a generic base point (Im Wz >= 0.002 for D <= 11)
            wz = (W[0, 0] * z + W[0, 1]) / (W[1, 0] * z + W[1, 1])
            val = Q * (W[1, 0] * z + W[1, 1]) ** -2 * self.Theta_value(p, wz)
            out.append((Q, pred, val / self.Theta_value(p, z)))
        return out

    def LpL1(self):
        """L'/L(1, chi_{-D}) = gamma + log 2 + log pi - (w/2) sum_a chi(a) log Gamma(a/D)."""
        D = self.D
        return EG + log(2) + log(pi) - (self.w / 2) * sum(self.chi(a) * lgamma(a / D) for a in range(1, D))

    def genus(self, p):
        D = self.D
        mu = (D + 1) * (p + 1)
        nu3 = (1 + kronecker(-3, D)) * (1 + kronecker(-3, p))
        return 1 + mu // 12 - nu3 // 3 - 2 if mu % 12 == 0 else round(1 + mu / 12 - nu3 / 3 - 2)


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


def eta_product(X, steps, powers):
    """q^{sum steps*powers/24} prod_n prod_i (1 - q^{steps_i n})^{powers_i}; returns the coefficients a(1..) shifted so that a(1) is first."""
    s = np.zeros(X + 1)
    s[0] = 1.0
    for st, pw in zip(steps, powers):
        e = euler_series(X, st)
        for _ in range(pw):
            s = np.round(fftconvolve(s, e)[: X + 1])
    shift = sum(st * pw for st, pw in zip(steps, powers)) // 24
    f = np.zeros(X + 1)
    f[shift:] = s[: X + 1 - shift]
    return f


# ---------------------------------------------------------------- Hecke decomposition (as in prime_discriminant_norm.py)
def hecke(v, ell, n_max):
    n = np.arange(n_max + 1)
    H = np.zeros(n_max + 1)
    H[1:] = v[ell * n[1:]]
    H[ell::ell] += ell * v[n[ell::ell] // ell]
    return H


def prony_eigs(v, ell, kmax, tol=1e-10):
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


def decompose(G, X, level_primes, test_primes=(2, 3, 5, 7, 13), kmax=22):
    comps = [G]
    tp = [q for q in test_primes if q not in level_primes]
    for ell in tp:
        new = []
        for v in comps:
            a = v / v[1] if abs(v[1]) > 1e-12 * np.abs(v[:X + 1]).max() else None
            if a is not None and multiplicative(a, tp):
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
    uniq = {}
    for v in comps:
        a = v / v[1]
        key = tuple(np.round(a[[2, 3, 5, 7, 11, 13]], 4))
        if key not in uniq or len(v) > len(uniq[key]):
            uniq[key] = v
    return list(uniq.values())


def hecke_ok(a, lp):
    good = [q for q in (2, 3, 5, 7) if q not in lp][:2]
    q1, q2 = good
    return (abs(a[q1 ** 2] - (a[q1] ** 2 - q1)) < 1e-6 and abs(a[q2 ** 2] - (a[q2] ** 2 - q2)) < 1e-6
            and abs(a[q1 * q2] - a[q1] * a[q2]) < 1e-6)


def L_at_1(coef, Nl, X):
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

    X = 2 ** 15
    son = sigma_over_n(X)
    known = {  # level Dp with S_2 spanned by known eta products (newforms): (D, p): list of (steps, powers)
        (3, 5): [((1, 3, 5, 15), (1, 1, 1, 1))],
        (7, 2): [((1, 2, 7, 14), (1, 1, 1, 1))],
        (11, 2): [((1, 11), (2, 2))],          # f_11 and f_11(2z) span S_2(Gamma_0(22))
    }
    for D in DS:
        H = Heegner(D)
        say(f"==================== D = {D}: N_D = x^2 + xy + {H.c} y^2, w_D = {H.w}, L(1,chi) = {H.L1:.6f}, "
            f"C_D = w^2 sqrt D/(8 pi^2) = {H.C_RS:.6f} ====================")
        # 1. densities
        say("1. local densities (closed forms against counts modulo ell^a):")
        for p in (2, 5, 7, 11, 13):
            if p == D:
                continue
            for ell in (D, p):
                a = max(2, int(np.floor(log(30000) / log(ell))))
                dens = H.count_density(p, ell, a)
                q = ell ** a
                errs = [abs(H.beta_formula(p, ell, k) / dens[k % q] - 1)
                        for k in (1, 2, 3, 4, 6, 9, 12, 18, ell, 2 * ell, 3 * ell, ell * ell, p, 2 * p, p * p)
                        if vp_of(k, ell) <= a - 2 and dens[k % q]]
                say(f"   p = {p:>2}, ell = {ell:>2}, a = {a}: max relative error {max(errs):.1e}")
        for p in (2, 3, 5, 7, 11, 13):
            if p == D:
                continue
            T = H.T(p, X)
            M = H.M(p, X, son)
            a = T - M
            nn = np.arange(X + 1, dtype=float)
            if (D, p) in known:
                basis = []
                for steps, powers in known[(D, p)]:
                    f = eta_product(X, steps, powers)
                    basis.append(f)
                    if (D, p) == (11, 2):
                        g = np.zeros(X + 1)
                        g[::2] = f[: X // 2 + 1]
                        basis.append(g)
                B = np.array(basis).T
                coef, *_ = np.linalg.lstsq(B[1:], a[1:], rcond=None)
                err = np.abs(B[1:] @ coef - a[1:]).max()
                say(f"   level {D * p}: T - M = {' + '.join(f'{c:.6f} f_{i}' for i, c in enumerate(coef))} with max error {err:.1e} "
                    f"(the known newform(s) of level {D * p}; a(D) of f: {basis[0][D]:+.0f}, chi(p) = {H.chi(p):+d})")
            else:
                ratios = [np.sum(a[1:y + 1] ** 2) / y ** 2 for y in (X // 4, X // 2, X)]
                say(f"   level {D * p}: sum_(k<=y) a(k)^2/y^2 at y = X/4, X/2, X: {ratios[0]:.4f} {ratios[1]:.4f} {ratios[2]:.4f} (cusp form)")
        # 2. AL
        say("2. Atkin-Lehner: (Theta|W_Q)(z)/Theta(z), predictions W_p: chi(p), W_D: -chi(p), W_Dp: -1")
        for p in (2, 5, 7, 11, 13):
            if p == D:
                continue
            res = H.check_AL(p)
            say(f"   p = {p:>2} (chi(p) = {H.chi(p):+d}): " + ", ".join(f"W_{Q}: {r.real:+.6f} (pred {pred:+d})" for Q, pred, r in res))
        # 3. decomposition and RS
        say("3. Hecke decomposition of G_{1,p}; newform of level Dp iff a(D) = +-1, a(D^2) = a(D)^2, a(p) = +-1; "
            "prediction: a(D) = chi(p), a(p) = -chi(p); Rankin-Selberg ratio against C_D chi(p) eta_p L L")
        KMAX = 22
        Xn = 2 ** 14
        Gmax = 2 ** KMAX
        rg = H.r_table(Gmax)
        song = sigma_over_n(Gmax)
        chi_arr = H.chi_table(Xn)
        ratios = []
        for p in (5, 7, 11, 13, 17, 19):
            if p == D:
                continue
            t0 = time.time()
            N = D * p
            G = H.T(p, Gmax, rg) - H.M(p, Gmax, song)
            vol = (pi / 3) * N * (1 + 1 / D) * (1 + 1 / p)
            normG = norm_RS(G, Xn, vol)
            comps = decompose(G, Xn, (D, p), kmax=KMAX)
            say(f"   p = {p:>2} (chi(p) = {H.chi(p):+d}), level {N}, dim S_2 = {H.genus(p)}: ||G||^2 (RS) = {normG:.4f}, "
                f"{len(comps)} eigencomponents")
            ok = True
            tot = 0.0
            for v in comps:
                a = v / v[1]
                Xv = len(v) - 1
                clean = hecke_ok(a, (D, p))
                stD = abs(abs(a[D]) - 1) < 1e-6 and (D * D > Xv or abs(a[D * D] - a[D] ** 2) < 1e-6)   # Steinberg at D
                stp = abs(abs(a[p]) - 1) < 1e-6 and (p * p > Xv or abs(a[p * p] - a[p] ** 2) < 1e-6)   # Steinberg at p
                is_new = stD and stp
                Xs = min(Xn, Xv)
                share = norm_RS(v, Xs, vol) / normG if Xv >= 2000 else float('nan')
                if share == share:
                    tot += share
                kind = f"NEW level {N}" if is_new else (f"old from level {p} (f(z), f({D}z))" if stp else
                                                       f"old from level {D} (f(z), f({p}z))")
                flag = "" if clean else "  [mixture]"
                if is_new:
                    good = abs(a[D] - H.chi(p)) < 1e-6 and abs(a[p] + H.chi(p)) < 1e-6
                    ok &= good
                    flag += "  <- prediction " + ("OK" if good else "VIOLATED")
                elif stp:
                    good = abs(a[p] + H.chi(p)) < 1e-6
                    ok &= good
                    flag += "  <- a_f(p) = -chi(p): " + ("OK" if good else "VIOLATED")
                else:
                    good = H.chi(p) == 1
                    ok &= good
                    flag += "  <- needs chi(p) = +1: " + ("OK" if good else "VIOLATED")
                say(f"      a(2) = {a[2]:+.4f}, a({D}) = {a[D]:+.4f}, a(5) = {a[5]:+.4f}, a({p}) = {a[p]:+.4f}: {kind}, share {share:.3f}{flag}")
                if is_new and clean and Xv >= Xn:
                    sp, w, L1 = L_at_1(a[:Xn + 1], N, Xn)
                    cand = [(Nl,) + L_at_1(chi_arr * a[:Xn + 1], Nl, Xn) for Nl in (N, D * N, D * D * N)]
                    Nl, sp2, w2, L1c = min(cand, key=lambda t: t[1])
                    inner = v[1] * norm_RS(a, Xn, vol)
                    pred = H.C_RS * H.chi(p) * (-a[p]) * L1 * L1c
                    ratios.append(inner / pred)
                    say(f"         L(1,phi) = {L1:.6f} (w = {w:+d}), L(1,phi x chi) = {L1c:.6f} (twist level {Nl}, w = {w2:+d}), "
                        f"<Theta,phi> = {inner:.6f}, C_D chi eta L L = {pred:.6f}, ratio = {inner / pred:.5f}")
            say(f"      predictions hold for all newform components: {ok}; shares sum to {tot:.3f}   [{time.time() - t0:.0f}s]")
        if ratios:
            say(f"   Rankin-Selberg ratios: min {min(ratios):.4f}, max {max(ratios):.4f} over {len(ratios)} newforms "
                f"(C_D = {H.C_RS:.6f})")
        # 4. norm
        LpL = H.LpL1()
        a1 = 2 * EG + 2 * LpL - 2 * ZPZ2 + log(D) / (D + 1)
        lead = 12 / (D + 1)
        c1 = 2 * lead * (a1 - EG)
        say(f"4. norm: L'/L(1,chi_-D) = {LpL:.6f}; S_D(Y) = {lead:.4f} (log Y)^2 + {c1:.4f} log Y + c_2")
        if D <= 19:          # the mpmath check of the closed form (the runs for D = 43 were stopped by the system at this point)
            try:
                import mpmath as mp
                mp.mp.dps = 20
                chis = [H.chi(n) for n in range(D)]
                Lp = mp.diff(lambda s: mp.dirichlet(s, chis), 1)
                L1v = mp.dirichlet(1, chis)
                say(f"   mpmath: L'/L(1,chi) = {float(Lp / L1v):.6f}, L(1,chi) = {float(L1v):.6f} (2 pi/(w sqrt D) = {H.L1:.6f})")
            except Exception as exc:                                     # noqa: BLE001
                say(f"   (mpmath check skipped: {exc})")
        else:
            say("   (mpmath check skipped for D > 19; the closed form is verified at D = 3, 7, 11, 19)")
        Xs = 2 ** 21
        rs = H.r_table(Xs)
        k = np.arange(Xs + 1, dtype=float)
        for Y in (1e3, 1e4, 1e5):
            S = float(np.sum(rs[1:] ** 2 / k[1:] * np.exp(-k[1:] / Y)))
            say(f"   S_D({Y:.0e}) = {S:.4f}, minus {lead:.4f} (log Y)^2 - c_1 log Y = {S - lead * log(Y) ** 2 - c1 * log(Y):.4f}")
        main_b = (2 * lead * (log(D / (4 * pi)) + log(1 / (4 * pi))) + 2 * c1) / (4 * pi)
        say(f"   two wide cusps (widths Dp, p): (1/4pi)(S_D(Dp/4pi) + S_D(p/4pi)) = {6 / (pi * (D + 1)):.4f} (log p)^2 + {main_b:.4f} log p + O(1)"
            f"   [6/(pi(D+1)); sqrt D/(2 pi) would be {sqrt(D) / (2 * pi):.4f}]")
        say(f"   {'p':>8} {'RS (2^21)':>10} {'Lambda_1':>10} {'Lambda_h':>10} {'lead log^2':>11} {'Lam_1/log^2':>11} {'RS/log^2':>9}")
        CUT = 12
        plist = [q for q in (5, 7, 11, 13, 101, 1009, 10007, 100003) if q != D]
        rb = H.r_table(max(Xs, CUT * max(plist)) + 1)
        sonb = sigma_over_n(max(Xs, CUT * max(plist)) + 1)
        rows = []
        for p in plist:
            N = D * p
            K = CUT * p
            a = H.T(p, K, rb[:K + 1]) - H.M(p, K, sonb[:K + 1])
            kk = np.arange(1, K + 1, dtype=float)
            a2 = a[1:] ** 2 / kk
            lam = {h: sum(float(np.sum(a2 * np.exp(-4 * pi * h * kk / Q))) for Q in (D * p, p, D, 1)) / (4 * pi)
                   for h in (1.0, sqrt(3) / 2)}
            rsv = float('nan')
            if p <= 20000:
                aa = H.T(p, Xs, rb[:Xs + 1]) - H.M(p, Xs, sonb[:Xs + 1])
                rsv = norm_RS(aa, Xs, (pi / 3) * N * (1 + 1 / D) * (1 + 1 / p))
            rows.append((p, lam[1.0]))
            say(f"   {p:>8} {rsv:>10.4f} {lam[1.0]:>10.4f} {lam[sqrt(3) / 2]:>10.4f} {6 / (pi * (D + 1)) * log(p) ** 2:>11.4f} "
                f"{lam[1.0] / log(p) ** 2:>11.4f} {rsv / log(p) ** 2:>9.4f}")
        x = np.log(np.array([p for p, _ in rows if p >= 100], dtype=float))
        y = np.array([l for p, l in rows if p >= 100])
        A = np.vstack([x ** 2, x, np.ones_like(x)]).T
        q = np.linalg.lstsq(A, y, rcond=None)[0]
        say(f"   fit Lambda_1 = a (log p)^2 + b log p + c over p >= 100: a = {q[0]:.4f} (6/(pi(D+1)) = {6 / (pi * (D + 1)):.4f}), "
            f"b = {q[1]:+.4f} (wide cusps: {main_b:+.4f}), c = {q[2]:+.3f}")
        say()
    with open(os.path.join(RES, "heegner_theta_unified_" + "_".join(str(D) for D in DS) + ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def vp_of(k, ell):
    v = 0
    while k and k % ell == 0:
        k //= ell
        v += 1
    return v


if __name__ == "__main__":
    main()
