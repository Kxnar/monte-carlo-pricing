"""Independently tuned proposal reuse for the existing terminal mixture model.

No estimator API or historical results are modified. Experiment helpers return
plain records; the workbench runner owns freezing configurations and file I/O.
"""
from dataclasses import dataclass, replace
from time import perf_counter
from types import MappingProxyType
import math
import numpy as np

from .estimators import estimate, importance_values, validate_budget
from .model import Contract, Mixture, Proposal
from .quadrature import positive_integral, radius, rule


@dataclass(frozen=True)
class ProposalFit:
    proposal: Proposal
    seconds: float
    candidate_rows: tuple
    payoff_evaluations: int
    seed_key: str


def tune_proposal(model, contract, seed=1729, pilot_n=1000):
    """Copy only the original twelve IS candidates, on independent pilots."""
    if not isinstance(pilot_n, (int, np.integer)) or pilot_n < 200:
        raise ValueError("Pilot size must be an integer of at least 200.")
    start = perf_counter()
    children = np.random.SeedSequence(seed).spawn(12)
    candidates, rows = [], []
    for shift in (0.0, -0.1, -0.2, -0.35):
        for inflation in (1.0, 1.25, 1.5):
            proposal = Proposal(model, shift, inflation)
            values = importance_values(model, contract, proposal, pilot_n,
                                       np.random.default_rng(children[len(rows)]))
            variance = float(np.var(values, ddof=1))
            candidates.append((variance, proposal))
            rows.append(MappingProxyType(dict(method="IS", shift=shift,
                        inflation=inflation, pilot_variance=variance)))
    best = min(candidates, key=lambda item: item[0])[1]
    return ProposalFit(best, perf_counter()-start, tuple(rows), 12*pilot_n, str(seed))


@dataclass(frozen=True)
class ReusePolicy:
    mode: str
    threshold: float = math.inf

    def __post_init__(self):
        if self.mode not in ("iid", "per_valuation", "reuse", "threshold"):
            raise ValueError("Unknown reuse policy.")
        if math.isnan(self.threshold) or self.threshold < 0:
            raise ValueError("A log-strike threshold must be nonnegative.")

    @property
    def method(self):
        return {"iid": "IID", "per_valuation": "IS_PER_VALUATION",
                "reuse": "IS_REUSE", "threshold": "IS_THRESHOLD"}[self.mode]


def needs_retune(policy, model, contract, fit=None, anchor_strike=None):
    """The anchor is the last tuning strike, not the preceding valuation."""
    if fit is not None and fit.proposal.target != model:
        raise ValueError("Proposal reuse across different models is prohibited.")
    if policy.mode == "iid":
        return False
    if fit is None:
        return True
    if anchor_strike is None or not math.isfinite(anchor_strike) or anchor_strike <= 0:
        raise ValueError("A fitted proposal requires a positive anchor strike.")
    if policy.mode == "per_valuation":
        return True
    return policy.mode == "threshold" and abs(math.log(contract.strike/anchor_strike)) > policy.threshold


@dataclass(frozen=True)
class ReuseCase:
    case_id: str
    model: Mixture
    contracts: tuple
    split: str = "heldout"
    workload_id: str = "repeated"

    def __post_init__(self):
        object.__setattr__(self, "contracts", tuple(self.contracts))
        if not self.contracts or any(not isinstance(c, Contract) for c in self.contracts):
            raise ValueError("A reuse case needs at least one existing Contract.")


@dataclass(frozen=True)
class ReuseConfig:
    root_seed: int = 261012
    sample_sizes: tuple = (2000, 8000, 32000)
    pilot_n: int = 1000
    pilot_groups: int = 3
    repeats: int = 12
    policies: tuple = (ReusePolicy("iid"), ReusePolicy("per_valuation"),
                      ReusePolicy("reuse"), ReusePolicy("threshold", .05))
    quad_order: int = 256
    reference_order: int = 512
    include_quad: bool = True

    def __post_init__(self):
        object.__setattr__(self, "sample_sizes", tuple(self.sample_sizes))
        object.__setattr__(self, "policies", tuple(self.policies))
        for n in self.sample_sizes:
            validate_budget(n)
        if not self.sample_sizes or self.pilot_groups < 1 or self.repeats < 1:
            raise ValueError("Use positive group/repeat counts and sample sizes.")
        if len({p.method for p in self.policies}) != len(self.policies):
            raise ValueError("Use at most one threshold policy per run.")
        if self.pilot_n < 200 or not isinstance(self.pilot_n, int):
            raise ValueError("Use an integer pilot size of at least 200.")
        if min(self.quad_order, self.reference_order) < 16:
            raise ValueError("Quadrature orders must be at least 16.")


def quadrature_price(model, contract, order=256):
    """One quadrature order, without the separate refinement reference."""
    cutoff = math.log(contract.strike/contract.spot)-contract.shift(model)
    total = 0.
    discount = math.exp(-contract.rate*contract.maturity)
    for w, m, a, b in zip(model.weights, model.locations, model.scales, model.shapes):
        top, bound = (cutoff-m)/a, radius(b)
        norm = b/(2*math.gamma(1/b))
        def integrand(z):
            payoff = -np.expm1(np.minimum(m+a*z-cutoff, 0))*contract.strike*discount
            return payoff*norm*np.exp(-np.abs(z)**b)
        total += w*(positive_integral(lambda u: integrand(-u), max(0, -top), bound, order)
                    + positive_integral(integrand, 0, min(max(top, 0), bound), order))
    return float(total)


def _coerce(case, config):
    return (ReuseCase(**case) if isinstance(case, dict) else case,
            ReuseConfig(**config) if isinstance(config, dict) else config)


def run_reuse_case(case, config=ReuseConfig()):
    """Run independent valuation histories, charging each history its pilots.

    A pilot group shares pilot seeds across repeats and budgets. Every fitting
    event reruns those pilot draws and measures its own elapsed cost.
    Production streams include policy, budget, repeat and valuation keys.
    """
    from .workbench_common import stable_seed, trial_record
    case, config = _coerce(case, config)
    setup_start = perf_counter()
    _ = case.model.log_mgf_one
    moment_seconds = perf_counter()-setup_start
    setup_start = perf_counter()
    rule(config.quad_order)
    quadrature_cache_seconds = perf_counter()-setup_start
    rule(config.reference_order)
    rule(2*config.reference_order)
    references = []
    for index, contract in enumerate(case.contracts):
        start = perf_counter()
        value, refinement = contract.reference(case.model, config.reference_order)
        references.append(dict(valuation_index=index, reference=value,
                          refinement_diagnostic=refinement, seconds=perf_counter()-start))
    tuning_rows, trials, ordering = [], [], []
    for group in range(config.pilot_groups):
        for repeat in range(config.repeats):
            tasks = [(policy, n) for policy in config.policies for n in config.sample_sizes]
            if config.include_quad:
                tasks.append((None, 0))
            order_seed = stable_seed(config.root_seed, "C", case.case_id, "ordering", group, repeat)
            np.random.default_rng(order_seed).shuffle(tasks)
            ordering.append(dict(group=group, repeat=repeat,
                            methods=[(p.method if p else "QUAD", n) for p, n in tasks]))
            for policy, n in tasks:
                fit, anchor, cumulative = None, None, 0.
                method = policy.method if policy else "QUAD"
                for index, contract in enumerate(case.contracts):
                    ref = references[index]["reference"]
                    decision_start = perf_counter()
                    retuned = policy is not None and needs_retune(policy, case.model, contract, fit, anchor)
                    decision_seconds = perf_counter()-decision_start
                    tuning_seconds, tuning_seed_key, proposal_id = 0., None, None
                    if retuned:
                        tuning_seed = stable_seed(config.root_seed, "C", case.case_id, "tune", group, index)
                        fit = tune_proposal(case.model, contract, tuning_seed, config.pilot_n)
                        anchor = contract.strike
                        tuning_rows.append(dict(group=group, repeat=repeat, method=method,
                           n_requested=n, valuation_index=index, anchor_strike=contract.strike,
                           seed_key=fit.seed_key, seconds=fit.seconds,
                           payoff_evaluations=fit.payoff_evaluations,
                           candidates=[dict(row) for row in fit.candidate_rows]))
                        tuning_seconds = fit.seconds
                    if fit is not None:
                        tuning_seed_key = fit.seed_key
                        proposal_id = f"shift={fit.proposal.shift:g};inflation={fit.proposal.inflation:g};pilot={fit.seed_key}"
                    seed = stable_seed(config.root_seed, "C", case.case_id, "production", group,
                                       repeat, method, n, index)
                    start = perf_counter()
                    if policy is None:
                        value = quadrature_price(case.model, contract, config.quad_order)
                        standard_error, independent, evaluations = None, 0, 0
                    else:
                        result = estimate("IID" if policy.mode == "iid" else "IS", case.model,
                                          contract, n, np.random.default_rng(seed),
                                          proposal=None if fit is None else fit.proposal)
                        value, standard_error = result.price, result.standard_error
                        independent, evaluations = result.observations, result.payoff_evaluations
                    seconds = perf_counter()-start
                    cumulative += seconds+tuning_seconds+decision_seconds
                    trials.append(trial_record(suite="C", case_id=case.case_id, split=case.split,
                        workload_id=case.workload_id, valuation_index=index, method=method, group=group,
                        repeat=repeat, n_requested=n, independent_samples=independent,
                        path_evaluations=None, fixing_evaluations=None, payoff_evaluations=evaluations,
                        quantity="price", estimate=value, standard_error=standard_error,
                        reference=ref, reference_standard_error=None, reference_kind="quadrature_refinement",
                        covered_95=None if standard_error is None else abs(value-ref) <= 1.96*standard_error,
                        combined_uncertainty_covered_95=None, seed_key=str(seed),
                        tuning_seed_key=tuning_seed_key, production_seconds=seconds,
                        tuning_seconds=tuning_seconds, decision_seconds=decision_seconds,
                        cumulative_seconds=cumulative, peak_memory_bytes=None, memory_kind="not_measured",
                        proposal_id=proposal_id, retuned=bool(retuned), coefficient_id=None,
                        block_size=None, status="ok"))
    return dict(trials=trials, diagnostics=dict(references=references, tuning=tuning_rows,
        setup_seconds=moment_seconds+quadrature_cache_seconds,
        setup=dict(model_moment_seconds=moment_seconds, quadrature_cache_seconds=quadrature_cache_seconds,
                   scope="once per run, outside warmed deployment histories"),
        ordering=ordering, tuning_cost_accounting="each fitting event executed and timed in its actual history",
        production_seed_scope="case/group/repeat/policy/budget/valuation",
        quadrature_reference_uncertainty="refinement diagnostic; not a stochastic standard error"))


def run_timing_pilot(case, config=ReuseConfig()):
    """A bounded eight-call TRAIN pilot; callers retain/freeze its evidence."""
    case, config = _coerce(case, config)
    pilot_case = replace(case, case_id=case.case_id+"-timing-pilot", split="train",
                         contracts=case.contracts[:8])
    pilot_config = replace(config, pilot_groups=1, repeats=2, pilot_n=1000,
                           sample_sizes=(2000, 8000, 32000))
    start = perf_counter()
    result = run_reuse_case(pilot_case, pilot_config)
    result["diagnostics"]["wall_seconds"] = perf_counter()-start
    result["diagnostics"]["purpose"] = "runtime grid choice before final freeze"
    return result


def select_reuse_threshold(training_cases, config=ReuseConfig(), candidates=(.02, .05, .10, math.inf)):
    """Choose on TRAIN only, minimising measured MSE times total cost."""
    config = ReuseConfig(**config) if isinstance(config, dict) else config
    cases = [ReuseCase(**c) if isinstance(c, dict) else c for c in training_cases]
    if not cases or any(c.split != "train" for c in cases):
        raise ValueError("Threshold selection requires nonempty TRAIN cases only.")
    evidence, all_trials, diagnostics = [], [], []
    for threshold in candidates:
        policy = ReusePolicy("threshold", threshold)
        rows = []
        for case in cases:
            result = run_reuse_case(case, replace(config, policies=(policy,), include_quad=False))
            rows.extend(result["trials"])
            diagnostics.append(dict(threshold="infinity" if math.isinf(threshold) else threshold,
                                    case_id=case.case_id, **result["diagnostics"]))
        mse = float(np.mean([(r["estimate"]-r["reference"])**2 for r in rows]))
        cost = float(sum(r["production_seconds"]+r["tuning_seconds"]+r["decision_seconds"] for r in rows))
        evidence.append(dict(threshold="infinity" if math.isinf(threshold) else threshold, mse=mse, total_seconds=cost,
                             objective=mse*cost, retunes=sum(r["retuned"] for r in rows)))
        all_trials.extend(dict(r, threshold="infinity" if math.isinf(threshold) else threshold) for r in rows)
    if not evidence:
        raise ValueError("At least one threshold candidate is required.")
    def numeric_threshold(row):
        return math.inf if row["threshold"] == "infinity" else row["threshold"]
    best = min(evidence, key=lambda r: (r["objective"], r["retunes"], -numeric_threshold(r)))
    return dict(policy=ReusePolicy("threshold", numeric_threshold(best)), evidence=evidence,
                trials=all_trials, diagnostics=diagnostics, selection_split="train",
                objective="empirical MSE * measured total cost")


def _history_arrays(rows, horizon):
    histories = {}
    for r in rows:
        if r["valuation_index"] < horizon:
            histories.setdefault((r["group"], r["repeat"]), []).append(r)
    complete = {k: sorted(v, key=lambda r: r["valuation_index"]) for k, v in histories.items()
                if len(v) == horizon and {r["valuation_index"] for r in v} == set(range(horizon))}
    keys = sorted(complete)
    errors = np.array([[(r["estimate"]-r["reference"])**2 for r in complete[k]] for k in keys])
    costs = np.array([complete[k][-1]["cumulative_seconds"] for k in keys])
    return keys, errors, costs


def summarize_reuse(trials, target_rmse=.05, horizons=(1, 2, 4, 8, 16, 32), bootstrap_repeats=1000,
                    seed=991, inner_bootstrap_repeats=100):
    """Qualify finite-horizon sizes using a hierarchical history bootstrap.

    Entire histories are resampled; groups are resampled first, then repeat
    histories inside each group. The fastest qualifying size is reselected in
    every bootstrap replicate. Unqualified baselines cannot demonstrate a
    crossing. Input rows should be a single case/workload/split.
    """
    if target_rmse <= 0 or bootstrap_repeats < 100 or inner_bootstrap_repeats < 100:
        raise ValueError("Use positive RMSE target and at least 100 bootstrap replicates.")
    if not trials:
        return dict(horizons=[], target_rmse=target_rmse, observed_break_even=None)
    scopes = {(r["case_id"], r["workload_id"], r["split"]) for r in trials}
    if len(scopes) != 1:
        raise ValueError("Summarize one case/workload/split at a time.")
    # QUAD has a distinct deterministic accuracy criterion, not a stochastic SE.
    grouped = {}
    for r in trials:
        if r["method"] != "QUAD":
            grouped.setdefault((r["method"], r["n_requested"]), []).append(r)
    valuation_rows = []
    for (method, n), rows in grouped.items():
        for index in sorted({r["valuation_index"] for r in rows}):
            local = [r for r in rows if r["valuation_index"] == index]
            rmse = math.sqrt(float(np.mean([(r["estimate"]-r["reference"])**2 for r in local])))
            coverage = [r["covered_95"] for r in local if r.get("covered_95") is not None]
            valuation_rows.append(dict(method=method, n_requested=n, valuation_index=index,
                rmse=rmse, exceeds_target=rmse > target_rmse,
                coverage_95=None if not coverage else float(np.mean(coverage))))
    quad = [r for r in trials if r["method"] == "QUAD"]
    quad_summary = None if not quad else dict(
        max_absolute_reference_difference=max(abs(r["estimate"]-r["reference"]) for r in quad),
        mean_production_seconds=float(np.mean([r["production_seconds"] for r in quad])),
        stochastic_standard_error=None, uncertainty_kind="quadrature refinement diagnostic")
    summaries, crossing = [], None
    rng = np.random.default_rng(seed)
    for horizon in horizons:
        arrays = {key: _history_arrays(rows, horizon) for key, rows in grouped.items()}
        arrays = {key: value for key, value in arrays.items() if len(value[0])}
        if not arrays:
            continue
        key_sets = {tuple(value[0]) for value in arrays.values()}
        if len(key_sets) != 1:
            raise ValueError("Every method/budget must have matching complete histories.")
        history_keys = next(iter(arrays.values()))[0]
        group_ids = sorted({k[0] for k in history_keys})
        group_indices = {g: np.array([i for i, k in enumerate(history_keys) if k[0] == g]) for g in group_ids}
        counts = {len(indices) for indices in group_indices.values()}
        if len(counts) != 1:
            raise ValueError("Pilot groups must have equal numbers of complete repeat histories.")
        repeats_per_group = counts.pop()
        draws = []
        for _ in range(bootstrap_repeats):
            indices = []
            for g in rng.choice(group_ids, size=len(group_ids), replace=True):
                indices.extend(rng.choice(group_indices[g], size=len(group_indices[g]), replace=True))
            draws.append(indices)
        draws = np.asarray(draws)
        # Each outer sample has group slots, preserving the group clustering
        # even when the same original pilot group was sampled twice.
        nested = np.empty((bootstrap_repeats, inner_bootstrap_repeats, len(history_keys)), dtype=np.int64)
        for b, outer in enumerate(draws):
            slots = outer.reshape(len(group_ids), repeats_per_group)
            slot_draws = rng.integers(len(group_ids), size=(inner_bootstrap_repeats, len(group_ids), 1))
            repeat_draws = rng.integers(repeats_per_group,
                                      size=(inner_bootstrap_repeats, len(group_ids), repeats_per_group))
            nested[b] = slots[slot_draws, repeat_draws].reshape(inner_bootstrap_repeats, -1)
        choices, size_rows = {}, []
        for key, (_, squared_errors, costs) in arrays.items():
            history_mse = squared_errors.mean(axis=1)
            boot_rmse = np.sqrt(history_mse[draws].mean(axis=1))
            boot_cost = costs[draws].mean(axis=1)
            nested_upper = np.quantile(np.sqrt(history_mse[nested].mean(axis=2)), .975, axis=1)
            upper = float(np.quantile(boot_rmse, .975))
            qualified = upper <= target_rmse
            row = dict(method=key[0], n_requested=key[1], horizon=horizon,
                rmse=math.sqrt(float(squared_errors.mean())), rmse_upper_95=upper,
                mean_cumulative_seconds=float(costs.mean()),
                status="qualified" if qualified else "candidate", qualified=qualified)
            size_rows.append(row)
            choices[key] = (row, nested_upper, boot_cost)
        selected, selection_boot = {}, {}
        for method in sorted({key[0] for key in choices}):
            eligible = [key for key in choices if key[0] == method and choices[key][0]["qualified"]]
            selected[method] = min((choices[key][0] for key in eligible),
                                  key=lambda row: row["mean_cumulative_seconds"], default=None)
            method_draws = np.full(bootstrap_repeats, np.nan)
            for b in range(bootstrap_repeats):
                eligible_b = [key for key in choices if key[0] == method and choices[key][1][b] <= target_rmse]
                if eligible_b:
                    method_draws[b] = min(choices[key][2][b] for key in eligible_b)
            selection_boot[method] = method_draws
        comparisons = []
        for method, row in selected.items():
            if method == "IID":
                continue
            baseline = selected.get("IID")
            diff = selection_boot[method]-selection_boot.get("IID", np.full(bootstrap_repeats, np.nan))
            valid = np.isfinite(diff)
            # Missing qualified selections are failures, not discarded wins.
            qualified_fraction = float(valid.mean())
            padded = np.where(valid, diff, np.inf)
            upper = float(np.quantile(padded, .975, method="higher")) if valid.any() else None
            observed = row is not None and baseline is not None and qualified_fraction >= .975 and upper is not None and upper < 0
            comparisons.append(dict(method=method, status="qualified" if row is not None else "none",
                selected_n=None if row is None else row["n_requested"],
                iid_selected_n=None if baseline is None else baseline["n_requested"],
                timing_difference_upper_95=upper if upper is not None and math.isfinite(upper) else None,
                bootstrap_joint_qualification_fraction=qualified_fraction,
                demonstrated_crossing=bool(observed)))
            if observed and method in ("IS_REUSE", "IS_THRESHOLD") and crossing is None:
                crossing = horizon
        summaries.append(dict(horizon=horizon, sizes=size_rows, comparisons=comparisons,
            iid_status="qualified" if selected.get("IID") is not None else "none"))
    return dict(horizons=summaries, valuations=valuation_rows, quadrature=quad_summary,
                target_rmse=target_rmse, observed_break_even=crossing,
                bootstrap_unit="pilot group then full repeat history",
                qualification="upper 95% bootstrap RMSE <= target",
                selection="fastest measured qualifying size; reselected in bootstrap",
                bootstrap_repeats=bootstrap_repeats, inner_bootstrap_repeats=inner_bootstrap_repeats,
                conclusion="observed crossing" if crossing is not None else "no observed crossing")
