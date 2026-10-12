"""Exact risk-neutral GBM paths and bounded-memory Asian summaries."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GBM:
    spot: float = 100.0
    rate: float = 0.03
    dividend: float = 0.0
    volatility: float = 0.2

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.spot, self.rate, self.dividend, self.volatility)):
            raise ValueError("GBM parameters must be finite.")
        if self.spot <= 0 or self.volatility <= 0:
            raise ValueError("Spot and volatility must be positive.")


@dataclass(frozen=True)
class PathSummary:
    arithmetic: np.ndarray
    geometric: np.ndarray
    arithmetic_vega: np.ndarray
    geometric_vega: np.ndarray


@dataclass(frozen=True)
class AsianPut:
    strike: float
    fixing_times: tuple
    maturity: float

    def __post_init__(self):
        times = validate_times(self.fixing_times)
        if not math.isfinite(self.strike) or self.strike <= 0:
            raise ValueError("Strike must be finite and positive.")
        if not math.isfinite(self.maturity) or self.maturity <= 0 or times[-1] != self.maturity:
            raise ValueError("The final fixing must equal the finite positive maturity.")
        object.__setattr__(self, "fixing_times", tuple(float(t) for t in times))

    def payoff(self, model, arithmetic):
        return math.exp(-model.rate*self.maturity)*np.maximum(self.strike-np.asarray(arithmetic), 0)


def validate_times(fixing_times):
    times = np.asarray(fixing_times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError("Fixing times must be a nonempty finite vector.")
    if times[0] <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Fixing times must be positive and strictly increasing.")
    return times


def _validate_matrix(values, times, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(times) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must have finite shape (paths, fixings).")
    return values


def summary_from_brownian(model, fixing_times, W):
    """Summarise supplied Brownian levels, including pathwise volatility derivatives."""
    times = validate_times(fixing_times)
    W = _validate_matrix(W, times, "Brownian levels")
    n = len(W)
    arithmetic = np.zeros(n)
    arithmetic_vega = np.zeros(n)
    log_sum = np.zeros(n)
    derivative_sum = np.zeros(n)
    drift = model.rate - model.dividend - 0.5 * model.volatility**2
    for i, t in enumerate(times):
        log_stock = math.log(model.spot) + drift*t + model.volatility*W[:, i]
        stock = np.exp(log_stock)
        derivative = W[:, i] - model.volatility*t
        arithmetic += stock
        arithmetic_vega += stock*derivative
        log_sum += log_stock
        derivative_sum += derivative
    m = len(times)
    geometric = np.exp(log_sum/m)
    return PathSummary(arithmetic/m, geometric, arithmetic_vega/m,
                       geometric*derivative_sum/m)


def path_summaries(model, fixing_times, z):
    """Use exact GBM transitions from independent standard normal increments.

    Only one stock column is materialised at a time; no path-by-fixing stock
    matrix is retained. Shocks are caller supplied to permit common paths.
    """
    times = validate_times(fixing_times)
    z = _validate_matrix(z, times, "Shocks")
    brownian = np.zeros(len(z))
    arithmetic = np.zeros(len(z))
    arithmetic_vega = np.zeros(len(z))
    log_sum = np.zeros(len(z))
    derivative_sum = np.zeros(len(z))
    drift = model.rate-model.dividend-0.5*model.volatility**2
    previous = 0.0
    for i, t in enumerate(times):
        brownian += math.sqrt(t-previous)*z[:, i]
        log_stock = math.log(model.spot)+drift*t+model.volatility*brownian
        stock = np.exp(log_stock)
        derivative = brownian-model.volatility*t
        arithmetic += stock
        arithmetic_vega += stock*derivative
        log_sum += log_stock
        derivative_sum += derivative
        previous = t
    m = len(times)
    geometric = np.exp(log_sum/m)
    return PathSummary(arithmetic/m, geometric, arithmetic_vega/m,
                       geometric*derivative_sum/m)
