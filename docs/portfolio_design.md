# Asian-put portfolio engine

`mcpricing.portfolio` jointly prices an ordered sequence of arithmetic-average
Asian puts under a single risk-neutral GBM model. Quantities are signed, and
duplicate contracts remain separate legs. The three outputs per leg are price,
spot delta and volatility vega, with standard errors. Vega is per unit absolute
volatility; multiply by 0.01 for one volatility percentage point.

The default simulation grid is the sorted union of every **absolute** fixing
time. Brownian increments have variance equal to consecutive union-date gaps.
Each distinct schedule accumulates arithmetic and log averages and their
volatility derivatives once per block; strikes reuse these summaries. A caller
may supply `simulation_times` as a strictly increasing required-date superset.
This is necessary when comparing a singleton to a batch with the same seed:
same seeds on different grids are different random variables.

The implementation retains one shock block plus per-schedule vectors, not a
matrix of all paths and dates. Shock draws are path-major, so changing blocks
preserves paths including remainder blocks. `block_size` counts independent
observations: physical paths for IID/control, whole pairs for antithetic.
Antithetic `n_paths` is an even physical-path budget, with `n_paths/2`
observations; the entire shock vector and its negative are averaged before any
moment updates. `max_block_paths` makes this distinction explicit.

Portfolio moments come from the signed weighted sum on **each observation**.
They retain covariance, so identical +1/-1 legs cancel both estimates and
errors, while +1/+1 doubles both. Adding leg errors in quadrature would violate
this requirement. Streamed block moments are merged with parallel Welford
updates; standard errors use the sample variance of independent observations.

For arithmetic average A and discount D, pathwise delta is
`-D * 1(A < K) * A / spot`. Pathwise vega is
`-D * 1(A < K) * mean(S_t * (W_t - sigma*t))`. The analogous geometric
control uses G and `G * mean(W_t - sigma*t)`. Positive volatility gives a
continuous law; the zero-probability strike boundary needs no extra term.

`fit_portfolio_control(model, contracts, pilot_paths, seed, ...)` fits one
coefficient per leg **and per metric**. Coefficients are copied and frozen.
`price_portfolio(..., method="geometric_cv", control=fit)` requires matching
model, ordered contracts and simulation grid, plus an independent evaluation
seed. The analytic discrete-geometric price, delta and vega supply each control
expectation. Adjusted prices are not clamped: negative finite-sample estimates
can occur and must be retained. No pilot is refitted during evaluation.

`fd_convergence(...)` checks raw pathwise Greeks against raw central price
differences using the same seed and union grid for the base and both bumps.
Spot bump sizes are relative, volatility bump sizes absolute. Its discrepancy
standard error is computed from the per-observation FD-minus-pathwise
difference. Decreasing bumps need not yield a strictly monotonic finite-sample
error near payoff kinks. A Greek control with a separately fitted coefficient
is not the derivative of a price control, so those are never compared as if
they were the same estimator.

`examples.portfolio_risk_demo.run_evaluation(output=None, ...)` returns a JSON
compatible report without writing. Passing an output path creates a new file
and refuses overwrite. The nine default contracts use strikes 90/100/110,
91/182/365-day maturities and 3/6/12 rounded integer-day fixing offsets. The
report retains seeds and frozen coefficients, all leg and portfolio errors,
the paired raw FD ladder, and first-order delta/vega P&L. Its delta hedge is
`-portfolio.delta` underlying units. These linear scenarios describe local
sensitivity; they are not VaR or full repricing and omit gamma and model risk.
The CLI prints all nine controlled leg estimates, three signed portfolio
totals, the delta hedge, and a compact table of first-order scenarios. Shared
`workbench_common.trial_record` rows retain the common schema; batch timing
fields are repeated per metric/leg for context and must not be summed across
rows as if they were separate runs. Environment metadata records source hashes.

Timing compares a batch calculating all three metrics with a loop of
singletons using the **same union grid and frozen coefficient slices**. This
comparison has a same-seed numerical equality check. Separate own-grid IID
singleton timings are labelled as a different workload. Timings are single
descriptive measurements, not throughput guarantees. Memory is measured in an
additional untimed `tracemalloc` run and represents traced allocations, not
process RSS. Final evidence generation and budget selection belong to the
integration gate; module and demo defaults do not establish a benchmark result.
