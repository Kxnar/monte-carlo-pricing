# Benchmark report

13,500 estimates across 3 synthetic distributions, 3 strikes and 3 payoff budgets. 5 independent pilot groups and 20 held-out repeats per group. Tables below use 20,000 payoff evaluations per estimate.

These are synthetic pricing experiments, not trading returns or market-calibrated results. Each proposal and control coefficient is fitted on separate pilot samples. The original 6.1x figure came from a smaller experiment; this report evaluates the whole tuning procedure.

## Gaussian

![RMSE convergence](gaussian-convergence.svg)

| Strike | Method | RMSE | Empirical VRF (95% interval) | Population VRF | Coverage | Runtime ms | Pilot ms | Efficiency incl. pilot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | IID | 0.02364 | 1.00 (1.00-1.00) | 1.00 | 95% | 2.54 | 0.00 | 1.00 |
| 80 | IS | 0.00559 | 17.95 (11.08-29.53) | 15.28 | 95% | 5.54 | 24.52 | 1.52 |
| 80 | AV | 0.02103 | 1.26 (0.83-1.94) | 1.09 | 93% | 1.52 | 0.00 | 2.11 |
| 80 | CV | 0.01806 | 1.72 (1.09-2.67) | 1.29 | 96% | 2.67 | 0.98 | 1.20 |
| 80 | MH | 0.04847 | 0.24 (0.15-0.40) | N/A | 93% | 35.23 | 90.36 | 0.00 |
| 100 | IID | 0.06707 | 1.00 (1.00-1.00) | 1.00 | 95% | 2.44 | 0.00 | 1.00 |
| 100 | IS | 0.02731 | 5.85 (3.41-9.35) | 5.55 | 96% | 5.27 | 23.23 | 0.50 |
| 100 | AV | 0.05052 | 1.70 (0.81-3.44) | 1.91 | 95% | 1.55 | 0.00 | 2.68 |
| 100 | CV | 0.04023 | 2.72 (1.53-4.87) | 2.60 | 94% | 2.61 | 0.96 | 1.86 |
| 100 | MH | 0.14050 | 0.22 (0.13-0.39) | N/A | 95% | 35.54 | 91.08 | 0.00 |
| 120 | IID | 0.10220 | 1.00 (1.00-1.00) | 1.00 | 97% | 2.42 | 0.00 | 1.00 |
| 120 | IS | 0.05802 | 3.09 (2.05-4.63) | 3.47 | 96% | 5.24 | 24.04 | 0.26 |
| 120 | AV | 0.02824 | 13.06 (8.49-22.63) | 18.00 | 93% | 1.56 | 0.00 | 20.23 |
| 120 | CV | 0.04033 | 6.42 (4.16-10.42) | 7.79 | 91% | 2.60 | 0.98 | 4.34 |
| 120 | MH | 0.27355 | 0.14 (0.09-0.25) | N/A | 94% | 35.41 | 89.40 | 0.00 |

## Mixture

![RMSE convergence](mixture-convergence.svg)

| Strike | Method | RMSE | Empirical VRF (95% interval) | Population VRF | Coverage | Runtime ms | Pilot ms | Efficiency incl. pilot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | IID | 0.02034 | 1.00 (1.00-1.00) | 1.00 | 97% | 2.39 | 0.00 | 1.00 |
| 80 | IS | 0.00358 | 30.80 (19.07-48.66) | 33.00 | 98% | 6.24 | 27.89 | 2.15 |
| 80 | AV | 0.02270 | 0.81 (0.53-1.27) | 1.03 | 93% | 1.44 | 0.00 | 1.34 |
| 80 | CV | 0.01919 | 1.07 (0.60-1.80) | 1.28 | 96% | 2.54 | 0.99 | 0.72 |
| 80 | MH | 0.11647 | 0.03 (0.01-0.19) | N/A | 89% | 35.43 | 89.20 | 0.00 |
| 100 | IID | 0.05346 | 1.00 (1.00-1.00) | 1.00 | 95% | 2.47 | 0.00 | 1.00 |
| 100 | IS | 0.01754 | 9.46 (5.16-19.47) | 9.40 | 96% | 6.44 | 28.31 | 0.67 |
| 100 | AV | 0.05334 | 1.01 (0.57-2.03) | 1.38 | 93% | 1.55 | 0.00 | 1.61 |
| 100 | CV | 0.04017 | 1.80 (1.09-2.94) | 2.41 | 94% | 2.66 | 1.03 | 1.20 |
| 100 | MH | 0.12871 | 0.17 (0.11-0.26) | N/A | 98% | 36.04 | 91.46 | 0.00 |
| 120 | IID | 0.09409 | 1.00 (1.00-1.00) | 1.00 | 94% | 2.50 | 0.00 | 1.00 |
| 120 | IS | 0.03086 | 9.02 (4.77-17.24) | 5.20 | 99% | 6.28 | 28.50 | 0.65 |
| 120 | AV | 0.04162 | 4.96 (2.53-8.51) | 4.80 | 96% | 1.52 | 0.00 | 8.16 |
| 120 | CV | 0.03249 | 8.14 (4.37-13.98) | 7.99 | 92% | 2.60 | 1.02 | 5.62 |
| 120 | MH | 0.24029 | 0.15 (0.08-0.31) | N/A | 94% | 35.88 | 94.10 | 0.00 |

## Stress

![RMSE convergence](stress-convergence.svg)

| Strike | Method | RMSE | Empirical VRF (95% interval) | Population VRF | Coverage | Runtime ms | Pilot ms | Efficiency incl. pilot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | IID | 0.04763 | 1.00 (1.00-1.00) | 1.00 | 94% | 2.43 | 0.00 | 1.00 |
| 80 | IS | 0.01811 | 6.98 (4.37-11.45) | 6.97 | 91% | 6.33 | 29.33 | 0.48 |
| 80 | AV | 0.04243 | 1.25 (0.78-2.02) | 1.08 | 96% | 1.45 | 0.00 | 2.10 |
| 80 | CV | 0.03508 | 1.83 (0.99-3.68) | 1.50 | 96% | 2.56 | 1.00 | 1.25 |
| 80 | MH | 0.22705 | 0.04 (0.01-0.21) | N/A | 93% | 35.06 | 88.85 | 0.00 |
| 100 | IID | 0.07941 | 1.00 (1.00-1.00) | 1.00 | 95% | 2.45 | 0.00 | 1.00 |
| 100 | IS | 0.02667 | 8.71 (4.36-18.45) | 10.18 | 95% | 6.27 | 27.86 | 0.63 |
| 100 | AV | 0.07021 | 1.36 (0.66-2.39) | 1.23 | 96% | 1.50 | 0.00 | 2.23 |
| 100 | CV | 0.05179 | 2.36 (1.30-4.17) | 2.10 | 97% | 2.56 | 0.99 | 1.63 |
| 100 | MH | 0.28588 | 0.08 (0.04-0.12) | N/A | 86% | 34.99 | 88.68 | 0.00 |
| 120 | IID | 0.12500 | 1.00 (1.00-1.00) | 1.00 | 95% | 2.47 | 0.00 | 1.00 |
| 120 | IS | 0.06726 | 3.46 (2.16-5.41) | 2.97 | 96% | 6.28 | 29.37 | 0.24 |
| 120 | AV | 0.08460 | 2.23 (1.08-4.23) | 2.16 | 92% | 1.51 | 0.00 | 3.64 |
| 120 | CV | 0.05891 | 4.52 (2.77-7.17) | 3.62 | 96% | 2.58 | 1.00 | 3.11 |
| 120 | MH | 0.30994 | 0.16 (0.08-0.30) | N/A | 93% | 34.96 | 89.00 | 0.00 |

## Reading the results

- VRF = IID estimator variance / method estimator variance at the same payoff budget. Above 1 means lower variance. AV uses half as many independent pair averages, with two payoffs per pair.
- Population VRF is obtained by integrating payoff moments under the model, averaged over the frozen pilot choices. It checks the empirical comparison without relying on another finite set of Monte Carlo estimates. MH has no population VRF here because its serial autocovariances are not integrated.
- Bootstrap intervals resample pilot groups and held-out repeats. Five pilot groups still give limited evidence about tuning variability; the intervals are exploratory, not simultaneous confidence bands over the grid.
- Coverage is the observed fraction of nominal 95% intervals containing the reference. It has sampling uncertainty. MH uses batch means; IID formulas must not be used on correlated chain draws.
- Efficiency incl. pilot = (IID variance x IID runtime) / (method variance x (runtime + one full method-specific pilot)). Above 1 is better. This is a variance-times-cost comparison, not an observed speedup at equal RMSE. Reusing a pilot amortises its cost; per-call figures excluding pilots are in summary.csv.
- Pilot time includes all 12 IS candidate trials, all six MH step trials, or the CV coefficient fit, respectively. Shared quadrature, imports and report writing are excluded from estimator timing. IS weight ESS and MH payoff ESS are different diagnostics and should not be directly compared.
- The benchmark samples a one-maturity terminal law. No calibrated volatility surface, dynamic hedging, transaction costs or realised trading profit is claimed. Quadrature is the practical preferred solver for this simple one-dimensional contract.
- Antithetic sampling is performed within each mixture component. Between-component randomness can outweigh the within-component negative covariance, so it is not guaranteed to beat IID for these mixtures.
- Hardware timing is machine-specific and sub-millisecond measurements are noisy. The code stores source hashes, random seeds, parameters, raw trials and environment information. Reordering or selecting a subset of scenarios leaves their random streams unchanged.

## Reproduce

```sh
python -m mcpricing benchmark --output local-results
```

Use the exact arguments in metadata.json for non-default runs. Existing result directories are never overwritten.
