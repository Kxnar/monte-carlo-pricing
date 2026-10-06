# Experiment provenance and review

The original source was unavailable. Its surviving description mentioned European put pricing under a two-component generalised-normal mixture, three estimators and a random-walk Metropolis-Hastings baseline. Original parameters, seeds, estimator details and result files were not recoverable.

The reconstruction therefore makes new, explicit modelling choices. Historical claims of 5x variance reduction or 60x runtime improvement are not asserted as reproduced. All current results have new raw experiment records.

## Recorded experiment rounds

| Folder | Core implementation commit | Purpose |
|---|---|---|
| `results/reconstruction-v2` | `aa71d52` | First five-estimator experiment; MH tuned by estimated pilot payoff variance |
| `results/reconstruction-v3` | `9039497` | Final experiment; MH tuned by expected squared jump distance |

Each round contains 13,500 estimates. Random streams are fixed by scenario, strike, group, budget, repeat and method. The independent-method estimates are identical across rounds; only MH's algorithm changes, while measured timings naturally fluctuate. This provides a controlled comparison of the MH tuning change rather than a search for a flattering seed.

The original smaller reconstruction's 6.1x empirical variance ratio came from only 40 estimates at the at-the-money strike. The expanded comparison finds 9.46x empirically and 9.40x by integrating population moments. The empirical bootstrap interval is broad (5.16-19.47x), illustrating the uncertainty in a variance ratio estimated from finite samples.

## Review findings and actions

1. **Risk-neutral law needed to be explicit.** The model is a chosen one-maturity pricing law with an exponential-moment correction. It is not a conversion from historical returns or a volatility-surface calibration.
2. **Variance and speed were conflated by the old description.** New outputs separate variance, RMSE, standard error, timing, pilot cost and cost-efficiency. A 9.40x variance gain implies a 3.07x standard-error gain at the same sample count.
3. **A chain's draws are not IID.** MH error estimates use batch means. Independent replicate coverage checks expose remaining limitations.
4. **Optimising rare-payoff pilot variance can select a poorly exploring chain.** The final MH pilot maximises expected squared jump distance. In the default mixture's strike-80, 20,000-sample case, coverage rose from 89% to 96% and RMSE fell from approximately 0.11647 to 0.05062. This is one measured case, not a guarantee of improved coverage everywhere.
5. **An asymmetric mixture cannot be reflected around zero.** Antithetic shocks are reflected around the sampled component's centre. Standard errors use pair averages and comparisons count both payoff evaluations.
6. **Tuning can contaminate evaluation.** IS candidates and the CV coefficient are fitted on separate pilot streams. Five independent pilot groups expose some tuning variation.
7. **Benchmark timing can be misleading.** Timings include estimator diagnostics and MH burn-in, while pilot costs are recorded separately. Method order is randomised. Python-loop MH versus vectorised IID remains an implementation-level confound, so MH speed ratios are not headline claims.
8. **Numerical parameter limits need to be honest.** The implementation supports shape in [1.25, 4], scale in [0.005, 1] and component location in [-2, 2]. Exponential-moment integration is checked for refinement convergence. These are implementation limits, not the mathematical support of the distribution family.

## Validation record

All 16 tests passed locally on Windows, Python 3.12.14, NumPy 2.3.5. A package wheel was built and installed separately from the source checkout, and its CLI executed successfully. Tests cover the Black-Scholes limit, density/sample moments, discounted-stock expectation, put-call identity, proposal support, estimators, antithetic pairing, population moments, delta, repeatability, invalid inputs and complete report generation.

GitHub Actions and Docker definitions are included for wider validation but were not executed in the reconstruction environment. They are configuration, not evidence of passing hosted checks.

## Reconstruct a run

Use the core commit above and the arguments in its `metadata.json`. Metadata also includes SHA-256 hashes of each package source file, dependency/runtime versions and scenario parameters. Both rounds used the default grid with seed 20261006. Future runs should use new folders.

Results are tied to these synthetic laws and implementation choices. The next research step is a prespecified extension with its own mathematical target and benchmark, such as component stratification or a carefully specified path-dependent model.
