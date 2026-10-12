"""Arithmetic Asian puts with exact geometric controls and independent pilots."""
from dataclasses import dataclass
import math
import numpy as np
from .gbm import AsianPut, GBM, PathSummary, path_summaries, summary_from_brownian, validate_times


def geometric_put(model, contract):
    times = np.asarray(contract.fixing_times)
    m = len(times)
    a = float(times.mean())
    b = float(np.dot(times, 2*m-2*np.arange(m)-1)/m**2)
    v = model.volatility**2*b
    mu = math.log(model.spot)+(model.rate-model.dividend-0.5*model.volatility**2)*a
    root_v = math.sqrt(v)
    d1 = (mu-math.log(contract.strike)+v)/root_v
    d2 = d1-root_v
    cdf = lambda x: 0.5*math.erfc(-x/math.sqrt(2))
    return math.exp(-model.rate*contract.maturity)*(contract.strike*cdf(-d2)-math.exp(mu+0.5*v)*cdf(-d1))


def model_signature(model):
    return (model.spot, model.rate, model.dividend, model.volatility)


def contract_signature(contract):
    return (contract.strike, contract.fixing_times, contract.maturity)


@dataclass(frozen=True)
class ControlFit:
    coefficient: float
    pilot_paths: int
    seed: int
    variance: float
    covariance: float
    model_signature: tuple
    contract_signature: tuple
    diagnostic: str = "ok"

    @property
    def provenance(self):
        return {"model": self.model_signature, "contract": self.contract_signature,
                "pilot_paths": self.pilot_paths, "seed": self.seed}


@dataclass(frozen=True)
class AsianEstimate:
    price: float
    standard_error: float
    independent_samples: int
    path_evaluations: int
    fixing_evaluations: int
    coefficient: float | None = None


class _Moments:
    """Parallel/Welford merge for one or more columns of observations."""
    def __init__(self, dimension=1):
        self.n = 0
        self.mean = np.zeros(dimension)
        self.m2 = np.zeros((dimension, dimension))

    def update(self, values):
        values = np.asarray(values, dtype=float).reshape(-1, len(self.mean))
        n = len(values)
        if not n:
            return
        mean = values.mean(axis=0)
        centred = values-mean
        delta = mean-self.mean
        total = self.n+n
        self.m2 += centred.T @ centred + np.outer(delta, delta)*(self.n*n/total)
        self.mean += delta*(n/total)
        self.n = total

    @property
    def covariance(self):
        return self.m2/(self.n-1)


def _validate_count(n, name, minimum=2):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")


def fit_control(model, contract, *, pilot_paths, seed, block_size=4096):
    _validate_count(pilot_paths, "Pilot paths")
    _validate_count(block_size, "Block size", 1)
    rng = np.random.default_rng(seed)
    moments = _Moments(2)
    for start in range(0, pilot_paths, block_size):
        n = min(block_size, pilot_paths-start)
        summary = path_summaries(model, contract.fixing_times, rng.standard_normal((n, len(contract.fixing_times))))
        moments.update(np.column_stack((contract.payoff(model, summary.arithmetic), contract.payoff(model, summary.geometric))))
    covariance = float(moments.covariance[0, 1])
    variance = float(moments.covariance[1, 1])
    coefficient = covariance/variance if variance > 0 else 0.0
    return ControlFit(coefficient, int(pilot_paths), seed, variance, covariance,
                      model_signature(model), contract_signature(contract),
                      "ok" if variance > 0 else "zero_control_variance")


def price_asian(model, contract, *, n_paths, seed, method="iid", control=None, block_size=4096):
    _validate_count(n_paths, "Paths")
    _validate_count(block_size, "Block size", 1)
    if method not in ("iid", "antithetic", "geometric_cv"):
        raise ValueError("Method must be iid, antithetic or geometric_cv.")
    if method == "antithetic" and (n_paths % 2 or n_paths < 4):
        raise ValueError("Antithetic paths must be even and at least four.")
    coefficient = None
    known = None
    if method == "geometric_cv":
        if not isinstance(control, ControlFit):
            raise ValueError("Geometric CV requires a frozen independent ControlFit.")
        if control.model_signature != model_signature(model) or control.contract_signature != contract_signature(contract):
            raise ValueError("Control provenance does not match the model and contract.")
        if control.seed == seed:
            raise ValueError("Pilot and production seeds must be independent.")
        if not math.isfinite(control.coefficient):
            raise ValueError("Control coefficient must be finite.")
        coefficient, known = control.coefficient, geometric_put(model, contract)
    rng = np.random.default_rng(seed)
    independent = n_paths//2 if method == "antithetic" else n_paths
    moments = _Moments()
    for start in range(0, independent, block_size):
        n = min(block_size, independent-start)
        shocks = rng.standard_normal((n, len(contract.fixing_times)))
        summary = path_summaries(model, contract.fixing_times, shocks)
        values = contract.payoff(model, summary.arithmetic)
        if method == "antithetic":
            opposite = path_summaries(model, contract.fixing_times, -shocks)
            values = 0.5*(values+contract.payoff(model, opposite.arithmetic))
        elif method == "geometric_cv":
            values = values-coefficient*(contract.payoff(model, summary.geometric)-known)
        moments.update(values)
    error = math.sqrt(max(float(moments.covariance[0, 0]), 0)/moments.n)
    return AsianEstimate(float(moments.mean[0]), error, moments.n, int(n_paths),
                         int(n_paths)*len(contract.fixing_times), coefficient)
