# Finite-step theory checks

Two standalone CPU scripts accompany the paper's finite-step analysis and GroupEval properties. They require no model weights, datasets, API keys, or network access. The proofs are in the paper; the scripts independently check identities and examples on finite spaces.

## Run

Use Python 3.10 or newer with NumPy and SciPy installed. From the release root:

```bash
python theory/verify_refined_theory.py --output build/theory/finite_step_checks.json
python theory/verify_generic_suboptimality.py --output build/theory/random_target_checks.json
```

The first command is a short check. The second enumerates 280 targets and searches their best inverse temperatures, so allow a few minutes on a laptop. Both print a JSON summary, create the requested output directory, and stop with an assertion error if a check fails. To try a smaller random-target run first:

```bash
python theory/verify_generic_suboptimality.py --trials 2 --output build/theory/small_random_target_checks.json
```

`--trials` is the number of targets per configuration. The default of 40 reproduces the paper's 280-target experiment; a reduced run is only a smoke check and should not replace the paper's figure data.

## Coverage

| Script | Connection to the paper |
| --- | --- |
| `verify_refined_theory.py` | Theorem 1 and Appendix C: entropy-area and conditional-mutual-information identities, augmented and marginal KL accounting, conditional total correlation, grid bounds, temperature derivatives, and clipped-schedule reveal masses. Proposition 2: exact repair, its boundary regime, perturbation stability, and the small-collision expansion. GroupEval appendix: GED range, ceiling, and monotonicity properties. |
| `verify_generic_suboptimality.py` | Theorem 2 and Figure 3c: exact enumeration of random full-support targets, the derivative covariance formula, pair entropy inflation, and sequential Bayes sampling. Proposition 2: the finite-repair example and inverted rankings. Proposition 1: the two-point optimum in random linear programs. |

The entropy identities use integer and rational coefficient arithmetic for formal subset entropies through sequence length eight. Distribution checks enumerate small state spaces and use floating-point arithmetic with explicit tolerances. Numerical enumeration supports the examples and checks the formulas; it does not establish the theorem's almost-everywhere statement in place of the proof.

## Reference results

`refined_verification_results.json` records 15,035 successful assertions. `generic_suboptimality_results.json` records 2,937 assertions and the 280 optimal inverse temperatures used by the theory figure. Keep these reference files unchanged when checking a new environment; write new output under `build/` as above.

The random-target result has 279 targets that improve by sharpening and one that improves by smoothing. The stored optimum arrays are rounded to five decimal places, so a displayed value of `1.0` can be a rounded optimum slightly different from one. Stationarity is checked separately using the unrounded derivative. Floating-point residuals and final optimizer digits may vary slightly across numerical-library versions.
