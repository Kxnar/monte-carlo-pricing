# Frozen pricing workbench: verified findings

The complete campaign produced **61,308 final trial rows and passed all 146 prespecified correctness, reference-resolution, memory and count checks**. Geometric controls substantially reduced Asian-price variance; the portfolio retained covariance and matched common-path singletons; proposal reuse showed a qualified timing crossing in one of six held-out workloads. Warm quadrature remained the strongest baseline for the existing one-dimensional terminal model.

These are synthetic-model and machine-specific results. They establish numerical behavior under the frozen protocol, not market calibration or production trading performance.

## A. Arithmetic Asian puts and geometric controls

The six cases were monthly strikes 80/100/120, a two-year 52-fixing stress case, an irregular calendar schedule, and a one-fixing identity check. Each used IID, antithetic and independently fitted geometric-control estimators at 4,096/16,384/65,536 physical paths, with three pilot groups and eight repeats. Antithetic observations were pairs, with half as many independent observations as physical paths.

All 54 case/method/budget pooled comparisons passed the frozen five-SE agreement gate. Each case had four independently seeded QuantLib arithmetic references, each using 262,144 antithetic pairs, or 524,288 physical paths. The aggregate reference SE was `sqrt(sum(run_SE²))/4`; it was retained as shared uncertainty rather than divided by the 24 native repeats. All six references passed the frozen resolution criterion relative to the largest-budget CV uncertainty. Analytic geometric prices matched QuantLib, including irregular dates and negative rates.

Observed performance against IID at 65,536 paths:

| Case | Empirical variance reduction | Efficiency including one full pilot | Efficiency amortizing the pilot across 24 calls |
| --- | ---: | ---: | ---: |
| Monthly K80 | 136.81× | 94.30× | 134.48× |
| Monthly K100 | 1,695.58× | 1,466.76× | 1,666.16× |
| Monthly K120 | 2,439.56× | 2,136.22× | 2,415.58× |
| Stress, 52 fixings | 138.57× | 124.59× | 141.43× |
| Irregular schedule | 425.62× | 348.66× | 410.89× |

Variance here is empirical across the independent groups and repeats; efficiency compares variance times measured cost. Each fit used 8,192 independent pilot paths. The saved summary includes group-then-repeat bootstrap intervals. With only three pilot groups these performance intervals are exploratory. The one-fixing control makes arithmetic and geometric payoffs identical; its near-zero numerical variance is a correctness identity, so its enormous floating-point ratio is excluded from the performance table.

Both estimator-only interval coverage relative to the stochastic reference and combined native/reference-uncertainty coverage are retained with Wilson intervals. Combined coverage is the uncertainty-aware comparison. Observed deviation RMSE against QuantLib is labelled as such; reference-corrected MSE is secondary and can be negative. QuantLib clamps negative control estimates; the native estimator does not.

## B. Joint portfolio price and first-order risk

The primary book contained nine signed legs at 91/182/365 days, strikes 90/100/110, and three distinct fixing schedules. An irregular-schedule book was evaluated separately. Each method used 131,072 physical paths with a fixed block size of 4,096. Separate independent controls were fitted for price, delta and vega.

The primary book's geometric-control totals were:

| Quantity | Estimate | SE |
| --- | ---: | ---: |
| Signed price | 2.898107 | 0.000777 |
| Delta | −0.102930 | 0.000746 |
| Vega per unit absolute volatility | 2.091984 | 0.019327 |

The corresponding underlying delta hedge is **+0.102930 units**. A spot change of −1 combined with an absolute volatility change of +0.01 gives first-order P&L of approximately **+0.123850**. This is a delta/vega approximation, with no gamma, VaR or market-calibration claim.

For all three methods, every leg's price/delta/vega and all three SEs matched same-union-grid singletons with maximum absolute discrepancy zero. CV singleton comparisons sliced the original fitted betas while preserving the same union grid and pilot provenance. Both finest common-random-number finite-difference bumps passed the paired discrepancy gate for every leg and each signed portfolio. Analytic geometric Greeks passed separate deterministic checks.

Across three retained, randomized-order timing comparisons, median joint-CV time was **229.08 ms**, versus **399.68 ms** for common-union CV singletons: **1.745×** faster for the same three-metric workload. Own-grid IID singleton time was 300.05 ms; it is a separate workload and method, not the primary batching comparison.

The nine-leg book peaked at **5,246,544 traced bytes** at both 131,072 and 524,288 paths with the same block size. The irregular book peaked at 1,221,232 bytes at both sizes. Both bounded-memory checks passed. These are separate untimed `tracemalloc` allocation measurements, not process RSS.

## C. Proposal reuse on held-out valuation histories

Threshold selection used distinct TRAIN histories anchored at strikes 90 and 110, with candidates 0.02/0.05/0.10/infinity. **0.10** was selected using measured MSE times total cost. TRAIN and final strike sets, case IDs and seed roots were disjoint. Final cases covered repeated, nearby and jumping strikes under both mixture and stress laws, with 32 calls each, budgets 2,000/8,000/32,000, three pilot groups, eight complete histories, and IID/per-call fitting/reuse/threshold/quadrature policies. Every required retune was physically executed and charged to its actual history.

Qualification required the upper bootstrap bound on pooled horizon RMSE to be at most 0.05. Pilot groups and then entire repeat histories were resampled; the fastest qualifying size was reselected inside each bootstrap replicate. A crossing required a qualified IID comparator and a negative upper bound on the cumulative timing difference. This is a finite-grid exploratory search, without familywise confidence across cases or horizons. Pooled horizon qualification does not establish an error bound on every call; per-valuation results are retained.

| Held-out case | Earliest reuse crossing | Earliest threshold crossing | IID qualified at 32 calls? |
| --- | ---: | ---: | --- |
| Mixture, repeated | 8 calls | 16 calls | Yes |
| Mixture, nearby | None observed | None observed | Yes |
| Mixture, jumps | None observed | None observed | No |
| Stress, repeated | None observed | None observed | No |
| Stress, nearby | None observed | None observed | No |
| Stress, jumps | None observed | None observed | No |

At the repeated-mixture eight-call crossing, reuse selected 2,000 samples per call and IID selected 32,000. Mean cumulative costs were **14.323 ms versus 30.325 ms**, approximately **2.12×** faster, with upper timing-difference bound −1.234 ms. Their upper RMSE bounds were 0.047709 and 0.040278 respectively. The 0.10 threshold policy did not cross at eight calls; its first qualified crossing was sixteen. On repeated strikes, threshold and reuse policies retain the same proposal; their distinct production streams and measured timings explain different qualification/crossing observations, without establishing that either rule is universally better. Eight is the first tested crossing horizon, not a precisely identified or universal break-even point. “None observed” retains the finite budget/qualification limitation; it does not prove that crossover is impossible.

Warm 256-order quadrature took **117–162 µs per call** across the six cases, with maximum absolute deviation **2.025×10⁻¹³** from the separately refined reference. Its uncertainty is a deterministic refinement diagnostic, not a stochastic SE. On this one-dimensional terminal problem, quadrature remains a faster and more accurate practical baseline than the tested Monte Carlo policies; the reuse result is a bounded comparison against IID.

## Protocol, cost and verification

| Saved quantity | Value |
| --- | ---: |
| A final price rows / controls / independent reference runs | 1,296 / 18 / 24 |
| B final leg/portfolio/metric rows | 108 |
| C final valuation rows | 59,904 |
| Prespecified checks passed | 146 / 146 |
| Recorded campaign wall time | 443.455 s |
| Production sampling | 231.056 s |
| Actual tuning events | 130.695 s |
| Independent references | 52.831 s |
| Separate FD/singleton/timing validation | 9.755 s |
| Separate memory passes | 1.980 s |
| Statistical summaries | 3.739 s |
| First report-render pass | 2.270 s |

The campaign stayed within the frozen 900-second production limit and 512-MiB tracked-memory limit. Recorded wall time includes the first report pass; subsequent saved-only report regeneration and post-campaign packaging/tests are separate verification work. A controls are charged once per case/group; B's shared method call is charged once across its metric rows; C charges every real fitting event. Shared setup, decisions, references and memory passes are recorded separately.

Frozen configuration SHA256: `031a0d3087abd4a9f042e14b01449ae4f755bae03c64c1068a4820e56a88fd8d`.

The recorded environment was Windows, Python 3.14.6, NumPy 2.5.4, QuantLib 1.43. NumPy uses 128-bit semantic seeds; QuantLib seeds use a distinct checked nonzero signed-31-bit mapping for Windows SWIG compatibility. No source files changed after this freeze. The source/configuration and retained pilot/TRAIN hashes still verified after the campaign.

Post-campaign verification: all 60 tests passed with QuantLib installed. A fresh NumPy-only virtual environment installed the built wheel: 54 tests passed and six optional-reference tests were skipped, out of 60 run. Installed `asian-price` and `portfolio-demo` commands passed outside the repository, without importing the examples directory. The wheel is retained under [verification/wheels](verification/wheels/). Distribution metadata reports version 0.3.0; the unchanged legacy `mcpricing.__version__` marker still says 0.2.0. Use `importlib.metadata.version('monte-carlo-pricing-lab')` for the installed distribution version; this metadata discrepancy does not affect pricing.

Earlier attempts remain evidence: [pilot-v1 failure](../workbench-pilot-v1/failure.json) exposed a Windows QuantLib seed-bound issue; [final-v2 status](../workbench-final-v2/status.json) records intentional interruption after 648 rows to fix the optional-dependency test guard. Its [frozen configuration](../workbench-config-v2/configuration.json) and partial rows remain untouched. The v3 protocol preserved the same budgets, cases, gates and selected threshold; only the required test guard changed between the v2/v3 freezes.

## Saved evidence and reproduction

- [Final trials](trials.csv), [summary](summary.json), [all checks](validation.json), [memory](memory.json), [costs](costs.json).
- [Frozen configuration](configuration.json), [source/configuration manifest](manifest.json), [environment](environment.json).
- [A controls](A-controls.json), [independent A reference runs](A-reference-runs.json), [A pooled validation](A-validation.json).
- [B controls](B-controls.json), [paired FD ladders](B-fd.json), [randomized timing observations](B-timings.json), [B memory observations](B-memory.json).
- [C summaries](C-summary.json); separate `C-diagnostics-heldout-*.json` files retain every reference, tuning event/candidate and history ordering.
- [Detailed generated Markdown](report.md) and [generated HTML with full saved JSON details](report.html). The HTML is about 53 MB because it embeds the full diagnostics; this compact findings file is the primary reading path.
- [Retained timing pilot](../workbench-pilot-v3/pilot.json) and [TRAIN selection evidence](../workbench-config-v3/training-evidence.json).
- [Reproduction instructions](../../docs/WORKBENCH.md) and [pinned environment](../../requirements-workbench-reproduce.txt).

From a source checkout, use new output names for each execution; existing evidence is never overwritten:

```powershell
.venv\Scripts\python.exe -m mcpricing workbench-pilot --output results/my-pilot
.venv\Scripts\python.exe -m mcpricing workbench-freeze --pilot results/my-pilot --output results/my-config
.venv\Scripts\python.exe -m mcpricing workbench-run --configuration results/my-config/configuration.json --output results/my-final
.venv\Scripts\python.exe -m mcpricing workbench-report results/my-final
mcpricing asian-price --paths 16384
mcpricing portfolio-demo --paths 16384 --pilot-paths 4096
```

Report regeneration reads saved artifacts and performs no new sampling or reference evaluation. Frozen loader rejection is intentional when code/docs/dependencies, configuration or retained pilot/TRAIN evidence changes.
