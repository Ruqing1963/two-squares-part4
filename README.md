# Sums of two squares and Heegner norms with few prime factors, IV: the scalloped Parseval formula, the Heegner classification, and Eisenstein dispersion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23866356.svg)](https://doi.org/10.5281/zenodo.23866356)

Part IV of the series after Part I ([two-squares-theta-level](https://github.com/Ruqing1963/two-squares-theta-level),
DOI 10.5281/zenodo.23826176), Part II ([two-squares-mean-square](https://github.com/Ruqing1963/two-squares-mean-square),
version 3, DOI 10.5281/zenodo.23866064) and Part III ([two-squares-short-intervals](https://github.com/Ruqing1963/two-squares-short-intervals),
DOI 10.5281/zenodo.23826860). The repository holds

* `paper/two_squares_part4.tex` (+ PDF, 3 figures): the paper;
* `notes/part4_explorations.tex`, `notes/heegner_unified.tex` (+ PDFs): the working notes from which the paper grew;
* `code/`: the scripts; `results/`: their outputs; `figures/`: the figures (PDF and PNG).

## The paper

1. **The scalloped Parseval formula (Section 2).** Waibel's fundamental domain has the lower boundary y = √(1 − {x}²), and
   Parseval over it gives the Petersson norm *exactly* as a double series over the cusps with correlation terms
   a(k) a(k + jQ) (Proposition 2.1). Evaluated to p = 400009 for G_{1,p}, it reproduces the Rankin–Selberg values where
   they have converged (7.42858 vs 7.42871 at p = 101) and gives ‖G_{1,p}‖² = (1/π)(log p)² + B log p + O(1) with
   B = 0.48204 in closed form (Theorem 2.4): 0.383 from the free part of the two main cusps (Part II, Lemma 6.3),
   0.0288 = 2κ/π from the arcs on the free part (Proposition 2.2, Mellin inversion, κ = ∫ log(1 − x²)^{−1/2} dx over
   [−1/2, 1/2] = 0.045231), 0.0569 + 0.0119 from the coefficients k ≥ p of the main cusps at height 1 and through the
   arcs, (1/π) Σ r(β)² (E₁(πβ) + E₁(4πβ)) and its arc version, 0.0019 from the cusp of denominator 2 and −0.0005 from
   the correlation terms (sums of exponential integrals; `closed_form_B.py`). The Siegel main term contributes nothing
   to B (Lemma 2.3: Σ r(k)β(k) and Σ β(k)² grow linearly, without a logarithm). The fit over p ≥ 10⁴ gives 0.4817, the
   local slopes at the largest primes 0.4819, 0.4820.
2. **The odd Heegner discriminants (Section 3).** For D ∈ {3, 7, 11, 19, 43, 67, 163}, Θ^{(D)}_{1,p} = θ_D(z) θ_D(pz) of
   square-free level Dp: Siegel main term with closed-form densities at D and p (Theorem 3.1); Atkin–Lehner signs
   Θ|W_p = χ(p)Θ, Θ|W_D = −χ(p)Θ (Theorem 3.2); spectral support on the newforms with a(D) = χ_{−D}(p), a(p) = −χ_{−D}(p),
   the necessary conditions, with the non-vanishing of L(1,φ)L(1,φ⊗χ) as the remaining criterion (Theorem 3.3,
   Proposition 3.4); Rankin–Selberg formula with C_D = w_D²√D/(8π²) (Theorem 3.5); the asymptotic
   ‖G‖² = (6/(π(D+1)))(log p)² + B_D log p + O(1) with B_D in closed form (Lemma 3.7, Theorem 3.8: the same five pieces
   as for D = 4, with the tail (6/(π(D+1))) Σ r_D(β)²(E₁(4πβ/D) + E₁(4πβ)) growing with D; B_3 = 0.8211, B_7 = 0.3289,
   B_11 = 0.2941, B_19 = 0.3225, B_43 = 0.3637, B_67 = 0.3632, B_163 = 0.3216), and for all nine discriminants the leading
   term (6/(πψ(D)))(log p)² = (2/vol(Γ₀(D)\H))(log p)² (Theorem 3.6 and Section 3.7). For D = 8 (level 8p, not square-free)
   Lemma 3.9 gives the expansions of Θ at all eight cusps of Γ₀(8p) by Poisson summation with Gauss sums on the dual
   lattice, and the Eisenstein part in closed form, E = (−F₂ + χF₄ − 2χF₈ + 2χF_p − χF_{2p} + F_{4p} − 2F_{8p})/(12(p + χ)),
   F_d = E₂(z) − dE₂(dz), χ = χ_{−8}(p); Theorem 3.10 then gives ‖G‖² = (1/(2π))(log p)² + B_8 log p + O(1) with
   B_8 = 0.2329 in closed form, the cusps of denominators 2 and 4 contributing 0.0086 (`heegner_D8.py`: every piece
   matched by the local slope of its component of the exact norm to 10⁻⁵).
3. **Numerical verification (Section 4).** D = 3, 7, 11, 19, 43: 91 newform components, all obeying the support law,
   Rankin–Selberg ratios in [0.9936, 1.0048]; the exact scalloped norm for all seven odd D with coefficients k ≤ 3Dp
   (`heegner_closed_form_B.py`) and for D = 8 over the eight cusps of Γ₀(8p) (`heegner_D8.py`): the local slopes of
   ‖G‖² − (6/(πψ(D)))(log p)² agree with B_D to 10⁻³ (D ≤ 11) or 10⁻² (D ≥ 19, shorter range of p), and to 3·10⁻⁵ for
   D = 8, and with B_D fixed the fitted leading coefficients are 0.4774, 0.2387, 0.1591, 0.1591, 0.0956, 0.0433, 0.0277,
   0.0115 (D = 3, 7, 8, 11, 19, 43, 67, 163) against 0.4775, 0.2387, 0.1592, 0.1592, 0.0955, 0.0434, 0.0281, 0.0116. The O(1) term depends on
   χ_{−D}(p); grouping the primes by that sign (`chi_grouped_slopes.py`, Table 5) brings the local slopes within 0.0012 of
   B_D for D ≤ 43, while for D = 67 both groups give 0.358–0.359 against 0.3632 over a range too short to decide. (The earlier norm scans
   of `heegner_theta_unified.py` truncate at k ≤ 12p, which is not enough at the cusp of width Dp for D ≥ 43; their Λ values
   for these D are superseded.) The three newforms of level 77 carry both G^{(7)}_{1,11} and G^{(11)}_{1,7} with opposite
   shares (Figure 3).
4. **Sieve limits and the Eisenstein dispersion form (Section 5).** A large discriminant does not help the sieve
   (remainder/main grows like √D; the level 1/2 and the dimension 2 are independent of D). For the Eisenstein integers the
   circle x² + xy + y² ≡ a mod c has a Kloosterman Fourier transform for every c prime to 3, even c included (Lemma 5.1,
   checked in 285 902 cases); the dispersion form has the sums S(inv(27m) N, |h|₃²; m'k) with the dual condition 3 | |h|₃²
   in place of 2 | |h|², and the obstruction of Part III is unchanged (Proposition 5.2).
5. **Correction to Parts II–III (Section 1.3).** The Kloosterman sums of the dispersion form are S(inv(16m) N, |h|²; m'k),
   not S(16N, m|h|²; m'k); the corrected versions are archived as Part II, version 3 (DOI 10.5281/zenodo.23866064) and Part III (DOI
   10.5281/zenodo.23826860), and Table 1 of the paper
   lists which statements of Parts II–III depend on the identity (the Kuznetsov heuristic of Part III, Section 3.5(b) is
   withdrawn; everything that only opens the sums is unchanged).

## Reproducing the results

All scripts are run from `code/` with Python 3 (numpy, scipy, matplotlib; sympy and mpmath for two checks). The anatomy
script imports the tables and the Siegel main term of Part II from `../../two-squares-mean-square/code`, so that
repository must sit beside this one.

```
python subleading_log_anatomy.py                 # Section 2; ~1 min; writes results/subleading_log_anatomy.txt
python closed_form_B.py                          # Theorem 2.4: the closed form of B against the anatomy; ~1 min
python heegner_theta_unified.py 3 7 11           # Sections 3-4 for D = 3, 7, 11; ~3 min
python heegner_theta_unified.py 19               # one D at a time for the large discriminants (memory); also 43, 67, 163
python heegner_closed_form_B.py                  # Theorem 3.8: B_D in closed form against the exact norm, all seven D; ~10 min, ~1 GB
python heegner_D8.py                             # Lemma 3.9, Theorem 3.10: D = 8 over the eight cusps of Gamma_0(8p); ~2 min
python heegner_D8.py check 13                    # the consistency checks of Lemma 3.9 at p = 13; seconds
python prime_discriminant_norm.py                # the first D = 3 exploration (notes/part4_explorations.tex)
python eisenstein_circle_check.py 40             # Lemma 5.1; ~15 min (pure Python)
python kloosterman_argument_audit.py             # Sections 1.3 and 5.3: the Kloosterman arguments, products modulo c; seconds
python chi_grouped_slopes.py                     # Section 4, Table 5: local slopes grouped by the sign of chi_{-D}(p); seconds
python plot_scalloped_anatomy.py                 # Figure 1
python plot_heegner_volume_scaling.py            # Figure 2
python plot_level77_duality.py                   # Figure 3
```

Outputs: `results/subleading_log_anatomy.txt`, `results/closed_form_B.txt`, `results/heegner_theta_unified_{3_7_11,19,43,67,163}.txt`,
`results/heegner_closed_form_B.txt`, `results/heegner_D8.txt`,
`results/prime_discriminant_norm.txt`, `results/eisenstein_circle_check.txt`, `results/kloosterman_argument_audit.txt`,
`results/chi_grouped_slopes.txt`; figures in `figures/`. The paper compiles with
`pdflatex` (run twice) from `paper/`.

Floating point throughout; evidence, not proof. The theorems do not depend on the computations.
