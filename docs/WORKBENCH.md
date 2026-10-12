# Asian portfolio pricing workbench

This workbench asks three bounded questions: how much an exact geometric control helps arithmetic Asian pricing; what sharing Brownian paths buys for a signed portfolio with price, delta and vega; and when reusing an importance proposal beats alternatives at the same observed accuracy. It records the failures and costs needed to interpret each answer.

The earlier European reconstruction and its published results stay in `results/reconstruction-v3`. Its Python 3.12 / NumPy 2.3.5 pins remain in `requirements-reproduce.txt`.

## Install and try

The base package supports Python 3.10+ and requires only NumPy. QuantLib is optional for normal pricing, but required for the independent reference gates and the complete workbench.

```sh
python -m pip install '.[reference]'
python -m mcpricing asian-price --paths 2000
python -m mcpricing portfolio-demo --paths 2000 --pilot-paths 1000
python examples/portfolio_risk_demo.py --paths 2000 --pilot-paths 1000 --output portfolio-demo.json
```

The recorded workbench environment is Windows, Python 3.14.6, NumPy 2.5.4, QuantLib 1.43 and SciPy 1.18.1. SciPy is recorded in the environment pins; the workbench implementation does not require it. This newer environment is separate from the original reconstruction pins.

```sh
python -m venv .venv
# Activate the virtual environment using your shell's usual command.
python -m pip install -r requirements-workbench-reproduce.txt
python -m pip install --no-deps .
python -m unittest discover -s tests -v
```

On Windows, use `.venv\Scripts\python.exe` in place of `python` if the environment is not activated. On Unix, use `.venv/bin/python`.

## Pilot, freeze, evaluate, report

Every run needs a fresh output directory. Nonempty output directories are rejected. Pilot data determines the permitted choices; it does not count as held-out evaluation evidence.

```sh
python -m mcpricing workbench-pilot --output workbench-pilot
python -m mcpricing workbench-freeze --pilot workbench-pilot --output workbench-frozen
python -m mcpricing workbench-run --configuration workbench-frozen/configuration.json --output workbench-evaluation
python -m mcpricing workbench-report workbench-evaluation
```

Use each command's `--help` for bounded pilot options. Any reduced case list must be chosen before freezing. A frozen run checks its recorded source hashes before evaluation. Finish implementation, documentation, dependency pins, and tests before the freeze; changing the source afterward requires a new pilot and freeze. Regenerating `report.md` and `report.html` from saved data does not resample, rerun references, or alter the evaluated source.

The finite default evaluation uses six Asian cases, three budgets, three methods, three independent pilot groups and eight repeats: 1,296 A rows. Each Asian case has four independent QuantLib arithmetic reference runs with 262,144 antithetic pairs per run. These are 24 separate reference runs, with two physical paths per pair. The pilot may use smaller reference budgets; the saved configuration is the authority for an actual run.

Proposal reuse covers mixture and stress cases over repeated, nearby and jump workloads, with 32 calls, budgets 2,000 / 8,000 / 32,000, three groups and eight repeats. Threshold candidates are 0.02, 0.05, 0.10 and infinity. Training anchors are strikes 90 and 110; evaluation strikes are distinct. The selected threshold is saved at freeze and remains fixed during evaluation. Always inspect expected counts, actual counts and validation status in the saved report.

## A: independent Asian accuracy

The path model is GBM with explicit fixing dates, maturity, rate and dividend yield. Arithmetic and geometric path averages share the same paths. The geometric Asian put has a closed-form expectation, and its control coefficient is fitted on an independent pilot and fixed for production. IID and control methods count whole paths as observations; antithetic methods count whole pairs. Physical paths, payoff evaluations, fixing evaluations and independent observations are separate fields.

QuantLib geometric checks provide an independent analytic implementation. Arithmetic references use independent QuantLib simulation runs and retain both their reported error estimates and between-run dispersion. Their uncertainty is stochastic. The same reference is shared by native repeats, so its SE is not divided by the number of native repeats. QuantLib clamps negative control-variate price estimates to zero; that limitation is recorded.

Observed deviation RMSE measures the distance to the saved stochastic reference. It is not error against exact truth. Estimator-only interval coverage compares native intervals against that stochastic reference while omitting its error, so it is not ordinary exact-target CI coverage. The primary comparison combines native and shared-reference uncertainty. Coverage summaries retain Wilson intervals. Variance and timing bootstrap resampling follows the pilot-group / repeat hierarchy; with three pilot groups these intervals are exploratory. A one-fixing Asian equals a European payoff, so the control may leave nearly zero noise; numerical residuals do not justify a claim of infinite efficiency.

## B: shared paths and first-order risk

The demo contains nine signed positions, alongside an irregular-schedule case. The simulator builds one sorted union of absolute fixing times, streams blocks of paths, and prices all legs on that grid. Every observation contributes price, pathwise delta and pathwise vega. Signed portfolio totals and their SEs use the full cross-leg covariance.

Batch and singleton gates compare all three estimates and all three SEs for all three methods on the same union grid, with absolute tolerance `1e-11`. Own-grid singleton timing is a different workload. Common random number finite differences use raw paired observations, a bump ladder, and a five-SE-plus-numerical-floor gate at the finest two bumps; a coarse bump alone cannot establish convergence.

Vega is per unit absolute volatility, so a one-percentage-point volatility change uses `0.01 * vega`. The hedge and scenario table are first-order delta/vega P&L under GBM. They do not include gamma, market calibration or VaR.

Timing repeats run separately from allocation measurements. `tracemalloc` measures traced allocations at N and 4N with the same block size; it is not resident process memory. The saved result should be read as evidence about block streaming and the recorded allocator, rather than a universal memory claim.

## C: a finite-workload crossing

The policies are IID, retune at every valuation, reuse one proposal, and threshold-triggered retuning. Every actual retune and decision is timed. Setup, pilot tuning, production, decisions and cumulative workload costs remain distinct. Charging the whole initial pilot to one call differs from amortising it over many calls.

For each horizon from one through 32 calls, the experiment seeks a budget meeting a frozen RMSE target of 0.05. Its upper 95% hierarchical bootstrap RMSE must qualify. Qualification is for pooled RMSE over the horizon, not a per-call accuracy guarantee. The fastest measured qualifying budget is selected again within each bootstrap draw. A crossing is reported only if the timing difference interval favours reuse and both policies qualify often enough. The first crossing is exploratory; the intervals do not establish familywise 95% confidence across the searched horizons and cases. The default uses 300 outer and 100 inner bootstrap draws. `No observed crossing` is a complete result; projecting a future crossing from a variance ratio is not part of this claim.

Warm numerical integration is timed as a practical baseline for this one-dimensional terminal-law workload. Its refinement diagnostic is deterministic and has no stochastic SE. Synthetic terminal mixture laws are separate from the GBM Asian path model.

## Saved evidence and limits

`configuration.json` identifies the selected protocol and its hash. `environment.json` records Python, dependencies, OS, BLAS configuration, thread environment, Git state and source hashes. Semantic seeds use SHA-256 structured keys with 128-bit entropy for NumPy; QuantLib reference seeds use an explicitly checked nonzero signed 31-bit mapping (`1..2**31-1`) for portable SWIG bindings. Package, example and test Python files are hashed recursively, along with README, documentation, dependency files and the CI workflow.

The suite CSVs retain individual trial fields, counts, SEs, references, seeds, timings, memory labels, tuning identifiers and status. Companion JSON files retain controls, reference runs, finite-difference ladders, timing repeats, memory passes and reuse diagnostics. `validation.json` and `summary.json` make gates and aggregation inspectable. Reports list raw row counts, all saved failure statuses, validation gates, and input artifact hashes; expandable HTML details expose the saved JSON.

These are bounded synthetic experiments with a small number of independent tuning groups. Results apply to the saved cases, implementation, machine and environment. They do not establish market pricing accuracy, production latency, general portfolio capacity or a guaranteed reuse benefit. Hosted CI is configured separately from the recorded local run; inspect its actual workflow result before claiming it passed.
