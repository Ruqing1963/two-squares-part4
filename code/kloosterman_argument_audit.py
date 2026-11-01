r"""
Audit of the Kloosterman arguments of Parts II-IV (Part IV, Sections 1.3 and 5.3).  S(a,b;c) = sum_{t mod c, (t,c)=1}
e((at + b tbar)/c) depends on (a,b) only through ab mod c: S(a,b;c) = S(1,ab;c).  Checked numerically, for random
N, m, h and moduli c (odd and even, square-free and not):
   (G1)  S(N mbar, inv(16)|h|^2; c) = S(inv(16m) N, |h|^2; c)          [the corrected form of Parts II-III]
   (G2)  S(N mbar, inv(16)|h|^2; c) != S(16N, m|h|^2; c) in general   [the form of the archived versions]
   (E1)  S(N mbar, inv(3)|inv(3) h|_3^2; c) = S(inv(27m) N, |h|_3^2; c)  [Section 5.3: |inv(3) h|_3^2 = inv(9)|h|_3^2]
   (E2)  the leading term g = 1 of F_c(N mbar; inv(3) h) is chi_{-3}(c) S(inv(27m) N, |h|_3^2; c) for c with (c, h) = 1
         (Lemma 5.1 applied to the twisted vector inv(3) h),
with the products of the arguments printed modulo c.   Usage: python kloosterman_argument_audit.py
"""
import os
import random
from math import gcd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
random.seed(4)


def S(a, b, c):
    t = np.array([x for x in range(1, c + 1) if gcd(x, c) == 1])
    tb = np.array([pow(int(x), -1, c) for x in t])
    return float(np.sum(np.cos(2 * np.pi * ((a * t + b * tb) % c) / c)))


def kron3(c):      # chi_{-3}(c), c prime to 3
    v = 1
    for q, e in factor(c):
        v *= (1 if q % 3 == 1 else -1) ** e
    return v


def factor(n):
    out, q = [], 2
    while q * q <= n:
        e = 0
        while n % q == 0:
            n //= q
            e += 1
        if e:
            out.append((q, e))
        q += 1
    if n > 1:
        out.append((n, 1))
    return out


def F3(a, h, c):
    """F_c(a; h) = sum_{x,y mod c, x^2+xy+y^2 = a (c)} e((h1 x + h2 y)/c)"""
    x = np.arange(c)
    X, Y = np.meshgrid(x, x, indexing='ij')
    m = (X * X + X * Y + Y * Y - a) % c == 0
    return float(np.sum(np.cos(2 * np.pi * ((h[0] * X[m] + h[1] * Y[m]) % c) / c)))


lines = ["Kloosterman arguments: S(a,b;c) depends only on ab mod c.  Random N, m, h; moduli c odd and even.", ""]
ok = {"G1": 0, "G2": 0, "E1": 0, "E2": 0}
tot = {"G1": 0, "G2": 0, "E1": 0, "E2": 0}
for trial in range(60):
    c = random.choice([21, 33, 35, 45, 49, 55, 63, 65, 77, 85, 91, 99, 22, 26, 34, 38, 44, 46, 50, 52, 58, 62])
    N = random.randrange(1, 10 ** 6)
    m = random.choice([5, 13, 17, 29, 37, 41, 53, 61])
    h = (random.randrange(-9, 10), random.randrange(-9, 10))
    if gcd(N, c) != 1 or gcd(m, c) != 1:
        continue
    h2 = h[0] ** 2 + h[1] ** 2
    h3 = h[0] ** 2 - h[0] * h[1] + h[1] ** 2
    if c % 2 == 1:                                     # Gaussian case: Part III, Lemma 3.1 needs c odd
        mb, i16, i16m = pow(m, -1, c), pow(16, -1, c), pow(16 * m, -1, c)
        lhs = S(N * mb % c, i16 * h2 % c, c)
        good = S(i16m * N % c, h2 % c, c)
        bad = S(16 * N % c, m * h2 % c, c)
        tot["G1"] += 1
        ok["G1"] += abs(lhs - good) < 1e-8
        tot["G2"] += 1
        ok["G2"] += abs(lhs - bad) > 1e-6
        if trial < 6:
            lines.append(f"   c = {c:>3}, N = {N:>7}, m = {m:>2}, |h|^2 = {h2:>3}: products mod c: N mbar inv16 |h|^2 = "
                         f"{N * mb * i16 * h2 % c:>3}, inv(16m) N |h|^2 = {i16m * N * h2 % c:>3}, 16 N m |h|^2 = {16 * N * m * h2 % c:>3}; "
                         f"S = {lhs:+.4f}, {good:+.4f}, {bad:+.4f}")
    if c % 3 != 0:                                     # Eisenstein case: any c prime to 3
        mb, i3, i27m = pow(m, -1, c), pow(3, -1, c), pow(27 * m, -1, c)
        th = ((i3 * h[0]) % c, (i3 * h[1]) % c)
        th3 = (th[0] ** 2 - th[0] * th[1] + th[1] ** 2) % c
        lhs = S(N * mb % c, i3 * th3 % c, c)
        good = S(i27m * N % c, h3 % c, c)
        tot["E1"] += 1
        ok["E1"] += abs(lhs - good) < 1e-8
        if gcd(gcd(h[0], h[1]), c) == 1 and c <= 65:
            F = F3(N * mb % c, th, c)
            tot["E2"] += 1
            ok["E2"] += abs(F - kron3(c) * good) < 1e-6
            if trial < 12:
                lines.append(f"   c = {c:>3}, N = {N:>7}, m = {m:>2}, |h|_3^2 = {h3:>3}: F_c(N mbar; inv3 h) = {F:+.4f}, "
                             f"chi(c) S(inv(27m) N, |h|_3^2; c) = {kron3(c) * good:+.4f}; product N mbar inv3 |inv3 h|_3^2 = "
                             f"{N * mb * i3 * th3 % c:>3} = inv(27m) N |h|_3^2 = {i27m * N * h3 % c:>3}")
lines.append("")
for k, name in (("G1", "S(N mbar, inv16 |h|^2; c) = S(inv(16m) N, |h|^2; c)"), ("G2", "... != S(16N, m|h|^2; c)"),
                ("E1", "S(N mbar, inv3 |inv3 h|_3^2; c) = S(inv(27m) N, |h|_3^2; c)"),
                ("E2", "F_c(N mbar; inv3 h) = chi(c) S(inv(27m) N, |h|_3^2; c) for (c,h) = 1")):
    lines.append(f"   {k}: {name}: {ok[k]}/{tot[k]}")
text = "\n".join(lines)
print(text)
with open(os.path.join(RES, "kloosterman_argument_audit.txt"), "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
