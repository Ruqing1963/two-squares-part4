r"""
The circle N_3(x,y) = x^2 + xy + y^2 = a modulo c, Fourier transformed: for (c, 3) = 1 (c odd OR even), (a, c) = 1,
    F_c(a; h) := sum_{x,y mod c, N_3(x,y) = a (c)} e((h_1 x + h_2 y)/c)
              = sum_{g | (c, h_1, h_2)} g chi_{-3}((c/g)^*) S(a, 3bar |h/g|_3^2; c/g),      |h|_3^2 = h_1^2 - h_1 h_2 + h_2^2,
with c^* the product of the primes dividing c to an odd power (chi_{-3}(2) = -1 for the even part) and 3bar the inverse of
3 modulo c/g.  The analogue of Part III, Lemma 3.1, with the dual norm of the hexagonal lattice and no restriction to odd c
(N_3 is unimodular at 2).  The script checks the identity for every a, h and every c <= CMAX coprime to 3, and also the
second identity used in the paper, S(N mbar, 16bar |h|^2; c) = S(inv(16m) N, |h|^2; c) != S(16N, m|h|^2; c).
Usage: python eisenstein_circle_check.py [CMAX]
"""
import os
import sys
from math import gcd
from cmath import exp, pi
from sympy.ntheory import jacobi_symbol

CMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 40
HERE = os.path.dirname(os.path.abspath(__file__))


def e(x):
    return exp(2j * pi * x)


def inv(a, c):
    return pow(a, -1, c)


def S(a, b, c):
    if c == 1:
        return 1
    return sum(e((a * t + b * inv(t, c)) / c) for t in range(c) if gcd(t, c) == 1)


def kron(a, n):
    res = 1
    while n % 2 == 0:
        n //= 2
        if a % 2 == 0:
            return 0
        if a % 8 in (3, 5):
            res = -res
    return res * (jacobi_symbol(a % n, n) if n > 1 else 1)


def star(c):
    s, d = 1, 2
    while d * d <= c:
        v = 0
        while c % d == 0:
            c //= d
            v += 1
        if v % 2:
            s *= d
        d += 1
    return s * c if c > 1 else s


def main():
    bad, tested = [], 0
    for c in range(2, CMAX + 1):
        if c % 3 == 0:
            continue
        for a in range(1, c):
            if gcd(a, c) != 1:
                continue
            for h1 in range(c):
                for h2 in range(c):
                    F = sum(e((h1 * x + h2 * y) / c) for x in range(c) for y in range(c) if (x * x + x * y + y * y - a) % c == 0)
                    tot = 0
                    for g in range(1, c + 1):
                        if c % g == 0 and h1 % g == 0 and h2 % g == 0:
                            cg = c // g
                            H = (h1 // g) ** 2 - (h1 // g) * (h2 // g) + (h2 // g) ** 2
                            tot += g if cg == 1 else g * kron(-3, star(cg)) * S(a % cg, (inv(3, cg) * H) % cg, cg)
                    tested += 1
                    if abs(F - tot) > 1e-6:
                        bad.append((c, a, h1, h2, round(F.real, 3), round(tot.real, 3)))
    lines = [f"D = 3 circle lemma for 2 <= c <= {CMAX} coprime to 3 (odd and even c): {tested} cases, "
             f"failures: {bad[:6] if bad else 'none'}",
             "Kloosterman sums depend only on the product of their arguments: S(N mbar, 16bar h2; c) against S(16N, m h2; c) "
             "and S(inv(16m) N, h2; c):"]
    for (N, m, c, H) in [(10, 5, 21, 5), (14, 9, 55, 13), (22, 13, 35, 2), (30, 7, 33, 10)]:
        A = S((N * inv(m, c)) % c, (inv(16, c) * H) % c, c)
        B = S((16 * N) % c, (m * H) % c, c)
        C = S(N % c, (inv(16 * m, c) * H) % c, c)
        lines.append(f"   N={N} m={m} c={c} |h|^2={H}: {A.real:+.4f}  {B.real:+.4f}  {C.real:+.4f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, '..', 'results', 'eisenstein_circle_check.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
