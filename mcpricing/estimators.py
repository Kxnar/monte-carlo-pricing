"""Sampling estimators and independent-pilot tuning."""
import math
import numpy as np
from .model import Proposal

def iid_values(model, contract, n, rng):
    return contract.payoff(model, model.sample(rng, n))


def importance_values(model, contract, proposal, n, rng):
    x = proposal.sample(rng, n)
    weights = np.exp(model.logpdf(x)-proposal.logpdf(x))
    return contract.payoff(model, x)*weights


def mh_values(model, contract, n, rng, step=0.25, burn=1000):
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
    children = np.random.SeedSequence(seed).spawn(18)
    candidates, diagnostics = [], []
    index = 0
    for shift in (0.0, -0.1, -0.2, -0.35):
        for inflation in (1.0, 1.25, 1.5):
            proposal = Proposal(model, shift, inflation)
            values = importance_values(model, contract, proposal, pilot_n, np.random.default_rng(children[index]))
            variance = float(np.var(values, ddof=1))
            candidates.append((variance, proposal))
            diagnostics.append(dict(method="IS", shift=shift, inflation=inflation, pilot_variance=variance))
            index += 1
    steps = []
    for step in (0.06, 0.12, 0.2, 0.35, 0.55, 0.9):
        values, acceptance = mh_values(model, contract, pilot_n, np.random.default_rng(children[index]), step)
        variance = standard_error(values, correlated=True)**2*pilot_n
        steps.append((variance, step))
        diagnostics.append(dict(method="MH", step=step, pilot_asymptotic_variance=variance, acceptance=acceptance))
        index += 1
    return min(candidates, key=lambda x:x[0])[1], min(steps)[1], diagnostics
