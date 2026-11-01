# Two explorations after Part III (working directory)

Working notes and scripts for two questions raised after Part III of the series
([Part I](https://github.com/Ruqing1963/two-squares-theta-level), [Part II](https://github.com/Ruqing1963/two-squares-mean-square),
[Part III](https://github.com/Ruqing1963/two-squares-short-intervals)). Not a paper yet.

* `notes/part4_explorations.tex` (+ PDF): the derivations and the results.
* `code/subleading_log_anatomy.py`: the exact norm ‖G_{1,p}‖² by Parseval over the scalloped bottom y = √(1 − {x}²) of
  Waibel's fundamental domain (an exact double series over the cusps, with correlation terms a(k) a(k + jQ)), and the
  anatomy of the coefficient B of log p in ‖G_{1,p}‖² = (1/π)(log p)² + B log p + O(1). Imports the tables and the Siegel
  main term of Part II from `../two-squares-mean-square/code`. Result: B ≈ 0.48 (not the 0.46 of Part II, Remark 6.16,
  which fitted non-converged Rankin–Selberg estimates): 0.383 from the free part of the two main cusps (Part II, Lemma 6.10),
  0.0288 = 2κ/π from the arcs acting on the free part (proved by Mellin inversion, κ = ∫_{−1/2}^{1/2} log(1 − x²)^{−1/2} dx =
  0.045231), 0.012 from the arcs on the corrections, 0.057 from the Siegel and tail corrections at height 1, 0.002 from the
  other four cusps; the correlation terms contribute −0.0005 log p.
* `code/prime_discriminant_norm.py`: the Eisenstein integers, θ_3(z) θ_3(pz) ∈ M_2(Γ_0(3p)): Siegel main term with the
  closed-form densities at 3 and p (checked against counts; T = M at level 6; T − M = (9/2) f_15 at level 15), the
  Atkin–Lehner signs Θ|W_p = χ(p)Θ, Θ|W_3 = −χ(p)Θ, Θ|W_{3p} = −Θ (checked numerically), the spectral support (every newform
  component of G_{1,p} has a(3) = χ_{−3}(p), a(p) = −χ_{−3}(p): 21 newforms for p ≤ 37, no inert vanishing), the Rankin–Selberg
  formula with the constant 9√3/(2π²) (ratios 0.994–1.005), and the norm ‖G^{(3)}_{1,p}‖² ~ (3/(2π))(log p)² with the sandwich
  Λ_1 ≤ ‖G‖² ≤ Λ_{√3/2}.
* `results/`: the outputs.

Floating point throughout; evidence, not proof, except where the notes give a proof.
