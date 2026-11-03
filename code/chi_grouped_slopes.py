r"""
Local slopes of ||G^{(D)}_{1,p}||^2 - (6/(pi(D+1))) (log p)^2 in log p between primes with EQUAL chi_{-D}(p)
(Part IV, Section 4, "Four different quantities").  The O(1) term of the asymptotic formula depends on p through
chi_{-D}(p), so the local slope between two primes with different chi carries the difference of two constants divided
by log(p_2/p_1); grouping the primes by the sign of chi_{-D}(p) removes this.  Reads results/heegner_closed_form_B.txt
(column "ex-lead" = exact norm minus the leading term) and writes results/chi_grouped_slopes.txt.
Usage: python chi_grouped_slopes.py
"""
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')


def kron(a, p):            # Legendre symbol (a/p) for an odd prime p
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def slopes(rows):
    return [(rows[i][0], rows[i + 1][0], (rows[i + 1][1] - rows[i][1]) / math.log(rows[i + 1][0] / rows[i][0]))
            for i in range(len(rows) - 1)]


text = open(os.path.join(RES, 'heegner_closed_form_B.txt'), encoding='utf-8').read()
text8 = open(os.path.join(RES, 'heegner_D8.txt'), encoding='utf-8').read()          # D = 8: the block at the end
text8 = text8[text8.index('==== D = 8'):] if '==== D = 8' in text8 else ''
out = ["Local slopes of ||G||^2 - (6/(pi psi(D)))(log p)^2 between primes with equal chi_{-D}(p); B_D for comparison.", ""]
summary = ["", "   D   last slope, chi = +1          last slope, chi = -1          B_D"]
for block in re.split(r'=+ D = (?=\d+:)', text + "\n" + text8)[1:]:
    D = int(block.split(':')[0])
    BD = float(re.search(r'B_D = sum\s+=\s+([+-]\d+\.\d+)', block).group(1))
    rows = []
    for line in block.splitlines():
        m = re.match(r'\s+(\d+)\s+' + r'([-\d.]+|nan)\s+' * 10 + r'([-\d.]+)\s+\[', line)
        if m:
            p = int(m.group(1))
            rows.append((p, float(m.group(11)), kron(-D, p)))
    out.append(f"D = {D}, B_D = {BD:+.4f}")
    out.append("   primes, chi_{-D}(p): " + ", ".join(f"{p}:{'+' if c > 0 else '-'}" for p, _, c in rows))
    out.append("   consecutive primes:  " + ", ".join(f"({a},{b}) {s:+.4f}" for a, b, s in slopes(rows)))
    last = {}
    for sign in (+1, -1):
        sl = slopes([r for r in rows if r[2] == sign])
        tag = '+' if sign > 0 else '-'
        out.append(f"   chi = {tag} only:        " + (", ".join(f"({a},{b}) {s:+.4f}" for a, b, s in sl) if sl else "(fewer than two primes)"))
        last[sign] = f"({sl[-1][0]},{sl[-1][1]}) {sl[-1][2]:+.4f}" if sl else "--"
    out.append("")
    summary.append(f" {D:>3}   {last[+1]:<28}  {last[-1]:<28}  {BD:+.4f}")
out += summary
result = "\n".join(out)
print(result)
with open(os.path.join(RES, 'chi_grouped_slopes.txt'), 'w', encoding='utf-8') as fh:
    fh.write(result + "\n")
