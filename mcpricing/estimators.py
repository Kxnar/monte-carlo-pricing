"""Sampling estimators and independent-pilot tuning."""
import math
import numpy as np
from .model import Proposal
from dataclasses import dataclass
from time import perf_counter


METHODS = ("IID", "IS", "AV", "CV", "MH")


def validate_budget(n):
    if not isinstance(n, (int, np.integer)) or n < 200 or n % 2:
        raise ValueError("Use an even payoff budget of at least 200.")


@dataclass(frozen=True)
class Estimate:
    price: float
    standard_error: float
    observations: int
    payoff_evaluations: int
    acceptance: float | None = None
    weight_ess: float | None = None
    payoff_ess: float | None = None


def antithetic_values(model, contract, n, rng):
    """n payoff evaluations return n/2 independent pair averages.

    Reflect inside the sampled component. Reflecting the entire asymmetric
    mixture about zero would not preserve its distribution.
    """
    validate_budget(n)
    components = rng.choice(2, size=n//2, p=model.weights)
    shape = np.asarray(model.shapes)[components]
    radius = rng.gamma(1/shape)**(1/shape)
    signs = 2*rng.integers(0, 2, size=n//2)-1
    centre = np.asarray(model.locations)[components]
    offset = np.asarray(model.scales)[components]*signs*radius
    return (contract.payoff(model, centre+offset)+contract.payoff(model, centre-offset))/2


def control_values(model, contract, coefficient, n, rng):
    x = model.sample(rng, n)
    return contract.payoff(model,x)-coefficient*(contract.discounted_stock(model,x)-contract.spot)


def estimate(method, model, contract, n, rng, proposal=None, coefficient=None, mh_step=None):
    validate_budget(n)
    acceptance = weight_ess = payoff_ess = None
    if method == "IID":
        values = iid_values(model, contract, n, rng)
    elif method == "IS":
        if proposal is None:
            raise ValueError("IS needs a frozen proposal from an independent pilot.")
        x = proposal.sample(rng, n)
        weights = np.exp(model.logpdf(x)-proposal.logpdf(x))
        values = contract.payoff(model,x)*weights
        weight_ess = float(weights.sum()**2/np.dot(weights,weights))
    elif method == "AV":
        values = antithetic_values(model,contract,n,rng)
    elif method == "CV":
        if coefficient is None:
            raise ValueError("CV needs a coefficient fitted on an independent pilot.")
        values = control_values(model,contract,coefficient,n,rng)
    elif method == "MH":
        if mh_step is None:
            raise ValueError("MH needs a proposal step.")
        values, acceptance = mh_values(model,contract,n,rng,mh_step)
    else:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")
    error = standard_error(values, correlated=method == "MH")
    if method == "MH" and error > 0:
        payoff_ess = float(np.var(values,ddof=1)/(error*error))
    return Estimate(float(values.mean()), error, len(values), n, acceptance, weight_ess, payoff_ess)

def iid_values(model, contract, n, rng):
    return contract.payoff(model, model.sample(rng, n))


def importance_values(model, contract, proposal, n, rng):
    x = proposal.sample(rng, n)
    weights = np.exp(model.logpdf(x)-proposal.logpdf(x))
    return contract.payoff(model, x)*weights


def mh_values(model, contract, n, rng, step=0.25, burn=1000):
    if not isinstance(n, (int, np.integer)) or n < 1 or not isinstance(burn,int) or burn < 0:
        raise ValueError("MH needs a positive sample count and nonnegative integer burn-in.")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("MH step must be finite and positive.")
    # Exact target initialisation is available for this benchmark model.
    # This avoids favouring IID/IS through a poorly initialised MH chain.
    state = float(model.sample(rng, 1)[0])
    density = model.scalar_logpdf(state)
    increments = rng.normal(0, step, n+burn)
    log_uniform = np.log(rng.random(n+burn))
    samples = np.empty(n)
    accepted = 0
    for i in range(n+burn):
        candidate = state + increments[i]
        candidate_density = model.scalar_logpdf(candidate)
        accept = log_uniform[i] < candidate_density-density
        if accept:
            state, density = candidate, candidate_density
        if i >= burn:
            accepted += accept
            samples[i-burn] = state
    return contract.payoff(model, samples), accepted/n


def standard_error(values, correlated=False):
    if len(values) < 100:
        raise ValueError("Use at least 100 observations for error estimation.")
    if not correlated:
        return float(np.std(values, ddof=1)/math.sqrt(len(values)))
    # Non-overlapping batch means, not an IID standard error for an MCMC chain.
    batches = max(10, int(math.sqrt(len(values))))
    length = len(values)//batches
    means = values[:batches*length].reshape(batches, length).mean(axis=1)
    return float(np.std(means, ddof=1)/math.sqrt(batches))


def tune(model, contract, seed=1729, pilot_n=8000):
    """Choose proposals on independent pilot draws, never benchmark draws."""
    if pilot_n < 200:
        raise ValueError("Pilot size must be at least 200.")
    children = np.random.SeedSequence(seed).spawn(19)
    candidates, diagnostics = [], []
    costs = {}
    index = 0
    start = perf_counter()
    for shift in (0.0, -0.1, -0.2, -0.35):
        for inflation in (1.0, 1.25, 1.5):
            proposal = Proposal(model, shift, inflation)
            values = importance_values(model, contract, proposal, pilot_n, np.random.default_rng(children[index]))
            variance = float(np.var(values, ddof=1))
            candidates.append((variance, proposal))
            diagnostics.append(dict(method="IS", shift=shift, inflation=inflation, pilot_variance=variance))
            index += 1
    costs["IS"] = perf_counter()-start
    start = perf_counter()
    steps = []
    for step in (0.06, 0.12, 0.2, 0.35, 0.55, 0.9):
        values, acceptance = mh_values(model, contract, pilot_n, np.random.default_rng(children[index]), step)
        variance = standard_error(values, correlated=True)**2*pilot_n
        steps.append((variance, step))
        diagnostics.append(dict(method="MH", step=step, pilot_asymptotic_variance=variance, acceptance=acceptance))
        index += 1
    costs["MH"] = perf_counter()-start
    start = perf_counter()
    x = model.sample(np.random.default_rng(children[18]), pilot_n)
    stock = contract.discounted_stock(model,x)
    payoff = contract.payoff(model,x)
    centred_stock, centred_payoff = stock-stock.mean(), payoff-payoff.mean()
    coefficient = float(np.dot(centred_stock,centred_payoff)/np.dot(centred_stock,centred_stock))
    costs["CV"] = perf_counter()-start
    costs["IID"] = costs["AV"] = 0.0
    return dict(proposal=min(candidates,key=lambda x:x[0])[1], mh_step=min(steps)[1],
                coefficient=coefficient, costs=costs, candidates=diagnostics)
