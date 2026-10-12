# Proposal reuse experiment (C)

This experiment retains the original one-dimensional synthetic mixture law and
European put estimator API. Reuse applies only to a proposal, never to production
draws. It does not imply a calibrated market model or a performance improvement.

`tune_proposal(model, contract, seed, pilot_n)` evaluates the original twelve IS
shift/inflation combinations and returns a frozen `ProposalFit`. It does not run
MH or CV tuning. Its candidate diagnostics and `12 * pilot_n` payoff cost are
retained. Candidate pilot streams are independent of every valuation stream.

`ReusePolicy` has `iid`, `per_valuation`, `reuse`, and `threshold` modes. IS always
tunes on its first valuation. Threshold retuning uses the strict condition
`abs(log(K / K_anchor)) > threshold`; equality does not trigger. Retuning updates
the anchor. A fitted proposal cannot be used with different model parameters.

`ReuseCase` defines a model, ordered tuple of contracts, split, and workload.
`ReuseConfig` defines seeds, budgets, pilot groups, repeats, policies and
quadrature orders. `run_reuse_case(case, config)` returns trials and diagnostics,
without writing files. Each repeat is a fresh complete history. Pilot groups
share pilot seeds across repeats and budgets, so pilot uncertainty is separate
from production uncertainty. Every fitting event executes the full pilot again
and measures its own current elapsed cost inside its actual policy, budget and
repeat history. Cumulative costs sum measured decision, tuning and production
times. Every fitting event retains its method, budget, repeat, valuation, seed,
cost and candidate diagnostics; proposal identity stays reproducible for the
same group and valuation.
Methods and budgets are shuffled with a semantic ordering seed. Production
seeds include case, group, repeat, policy, budget and valuation. Case IDs must
identify the experimental phase; training and held-out cases must have disjoint
IDs and roots or semantic phase identities.

References are computed once outside production timing using a separately
refined high-order quadrature. Its refinement difference is a numerical
diagnostic, never a stochastic standard error. The QUAD baseline times an actual
warmed single-order quadrature call on every valuation and reports no stochastic
SE or coverage interval. Setup warms the model moment and quadrature node caches.
One-time model-moment and QUAD-node-cache setup costs are recorded separately in
diagnostics; deployment histories are measured after this warmup. European
path/fixing counts are null because these are terminal payoff evaluations.
Tracked memory should be measured separately from timed production; ordinary
trials mark memory as unmeasured until a separate probe supplies evidence.

`run_timing_pilot` limits the case to eight TRAIN valuations, one pilot group and
two repeats at budgets 2,000/8,000/32,000 with pilot size 1,000. The integration
runner retains its timing evidence and fixes the final grid before evaluation.
`select_reuse_threshold` accepts TRAIN cases only. Prespecified log-strike
thresholds .02/.05/.10/infinity are scored by empirical MSE times measured total
cost; exact ties favour fewer retunes, then the larger threshold. All trials,
costs, diagnostics and failures are returned. Infinity is represented as the
string `infinity` in evidence records. The returned policy is a Python dataclass
and must be serialized by the caller with the same infinity convention.

`summarize_reuse` examines complete histories at finite horizons 1/2/4/8/16/32.
A sample size qualifies only when the upper endpoint of its 95% bootstrap RMSE
interval is at most the absolute target (.05 by default). The fastest measured
qualified size is selected independently for each method. The bootstrap first
resamples pilot groups and then full histories within each group. An inner
bootstrap reapplies qualification and size selection inside every outer
replicate. This preserves shared pilot uncertainty, serial calls, and selection
uncertainty. A crossing requires qualified IID and reuse sizes and a negative
upper 95% cumulative-time difference. Replicates without joint qualification
are counted as failures; they are never silently discarded. Otherwise the
conclusion is `no observed crossing`. Candidate and missing-qualified-size
states are distinct. QUAD is retained as a deterministic comparator in raw
trials and is not assigned stochastic qualification or SE by this summary.

Tests cover the original candidate grid, deterministic fits, stream separation,
fresh production draws, charged retunes, strict boundaries and anchor updates,
cross-model prohibition, weighted estimates against quadrature, timing
instrumentation invariance, complete records and conservative summary states.
They never require an empirical speed improvement.
