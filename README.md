# Sums of two squares and Heegner norms with few prime factors, IV: the scalloped Parseval formula, the Heegner classification, and Eisenstein dispersion

Part IV of the series after Part I ([two-squares-theta-level](https://github.com/Ruqing1963/two-squares-theta-level),
DOI 10.5281/zenodo.23826176), Part II ([two-squares-mean-square](https://github.com/Ruqing1963/two-squares-mean-square),
DOI 10.5281/zenodo.23826297) and Part III ([two-squares-short-intervals](https://github.com/Ruqing1963/two-squares-short-intervals),
DOI 10.5281/zenodo.23826860). The repository holds

* `paper/two_squares_part4.tex` (+ PDF, 3 figures): the paper;
* `notes/part4_explorations.tex`, `notes/heegner_unified.tex` (+ PDFs): the working notes from which the paper grew;
* `code/`: the scripts; `results/`: their outputs; `figures/`: the figures (PDF and PNG).

## The paper

1. **The scalloped Parseval formula (Section 2).** Waibel's fundamental domain has the lower boundary y = √(1 − {x}²), and
   Parseval over it gives the Petersson norm *exactly* as a double series over the cusps with correlation terms
   a(k) a(k + jQ) (Proposition 2.1). Evaluated to p = 400009 for G_{1,p}, it reproduces the Rankin–Selberg values where
   they have converged (7.42858 vs 7.42871 at p = 101) and gives ‖G_{1,p}‖² = (1/π)(log p)² + B log p + O(1) with B = 0.48:
   0.383 from the free part of the two main cusps (Part II, Lemma 6.10), 0.0288 = 2κ/π from the arcs on the free part
   (Proposition 2.2, Mellin inversion, κ = ∫ log(1 − x²)^{−1/2} dx over [−1/2, 1/2] = 0.045231), 0.057 + 0.012 from the
   Siegel and tail corrections, 0.002 from the other cusps; the correlation terms give −0.0005.
2. **The odd Heegner discriminants (Section 3).** For D ∈ {3, 7, 11, 19, 43, 67, 163}, Θ^{(D)}_{1,p} = θ_D(z) θ_D(pz) of
   square-free level Dp: Siegel main term with closed-form densities at D and p (Theorem 3.1); Atkin–Lehner signs
   Θ|W_p = χ(p)Θ, Θ|W_D = −χ(p)Θ (Theorem 3.2); spectral support on the newforms with a(D) = χ_{−D}(p), a(p) = −χ_{−D}(p),
   no inert vanishing (Theorem 3.3); Rankin–Selberg formula with C_D = w_D²√D/(8π²) (Theorem 3.4); norm
   ‖G‖² ≥ (6/(π(D+1)))(log p)² + b_D log p + O(1) and, for all nine discriminants, (6/(πψ(D)))(log p)² =
   (2/vol(Γ₀(D)\H))(log p)² (Theorem 3.5 and Section 3.6).
3. **Numerical verification (Section 4).** D = 3, 7, 11, 19, 43: 91 newform components, all obeying the support law,
   Rankin–Selberg ratios in [0.9936, 1.0048]; norm scans for all seven odd D with fitted coefficients 0.4815, 0.2408, 0.1601,
   0.0961, 0.0443, 0.0266, 0.0128 against 0.4775, 0.2387, 0.1592, 0.0955, 0.0434, 0.0281, 0.0116. The three newforms of level
   77 carry both G^{(7)}_{1,11} and G^{(11)}_{1,7} with opposite shares (Figure 3).
4. **Sieve limits and the Eisenstein dispersion form (Section 5).** A large discriminant does not help the sieve
   (remainder/main grows like √D; the level 1/2 and the dimension 2 are independent of D). For the Eisenstein integers the
   circle x² + xy + y² ≡ a mod c has a Kloosterman Fourier transform for every c prime to 3, even c included (Lemma 5.1,
   checked in 285 902 cases); the dispersion form has the sums S(inv(27m) N, |h|₃²; m'k) with the dual condition 3 | |h|₃²
   in place of 2 | |h|², and the obstruction of Part III is unchanged (Proposition 5.2).
5. **Correction to Parts II–III (Section 1.3).** The Kloosterman sums of the dispersion form are S(inv(16m) N, |h|²; m'k),
   not S(16N, m|h|²; m'k); the corrected versions of Parts II and III are in their repositories.

## Reproducing the results

All scripts are run from `code/` with Python 3 (numpy, scipy, matplotlib; sympy and mpmath for two checks). The anatomy
script imports the tables and the Siegel main term of Part II from `../../two-squares-mean-square/code`, so that
repository must sit beside this one.

```
python subleading_log_anatomy.py                 # Section 2; ~1 min; writes results/subleading_log_anatomy.txt
python heegner_theta_unified.py 3 7 11           # Sections 3-4 for D = 3, 7, 11; ~3 min
python heegner_theta_unified.py 19               # one D at a time for the large discriminants (memory); also 43, 67, 163
python prime_discriminant_norm.py                # the first D = 3 exploration (notes/part4_explorations.tex)
python eisenstein_circle_check.py 40             # Lemma 5.1; ~15 min (pure Python)
python plot_scalloped_anatomy.py                 # Figure 1
python plot_heegner_volume_scaling.py            # Figure 2
python plot_level77_duality.py                   # Figure 3
```

Outputs: `results/subleading_log_anatomy.txt`, `results/heegner_theta_unified_{3_7_11,19,43,67,163}.txt`,
`results/prime_discriminant_norm.txt`, `results/eisenstein_circle_check.txt`; figures in `figures/`. The paper compiles with
`pdflatex` (run twice) from `paper/`.

Floating point throughout; evidence, not proof. The theorems do not depend on the computations.
