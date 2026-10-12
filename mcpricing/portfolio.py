"""Joint Brownian Asian-put pricing and first-order portfolio risk.

Physical paths are streamed; statistical observations are whole paths (IID/CV)
or whole antithetic pairs. All portfolio errors retain cross-leg covariance.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence
import numpy as np

from .asian import GBM, AsianPut, geometric_put

METRICS = ("price", "delta", "vega")


@dataclass(frozen=True)
class GreekEstimate:
    delta: float
    vega: float

    def __iter__(self):
        return iter((self.delta, self.vega))


@dataclass(frozen=True)
class LegEstimate:
    price: float
    delta: float
    vega: float
    price_se: float
    delta_se: float
    vega_se: float


@dataclass(frozen=True)
class PortfolioControlFit:
    betas: np.ndarray
    pilot_paths: int
    pilot_seed: int
    model: GBM
    contracts: tuple[AsianPut, ...]
    simulation_times: tuple[float, ...]
    block_size: int

    def __post_init__(self):
        betas = np.array(self.betas, dtype=float, copy=True)
        betas.setflags(write=False)
        object.__setattr__(self, "betas", betas)

    @property
    def provenance(self):
        return {"model": self.model, "contracts": self.contracts,
                "simulation_times": self.simulation_times, "pilot_paths": self.pilot_paths,
                "pilot_seed": self.pilot_seed, "block_size": self.block_size}


@dataclass(frozen=True)
class PortfolioEstimate:
    legs: tuple[LegEstimate, ...]
    portfolio: LegEstimate
    quantities: tuple[float, ...]
    method: str
    n_paths: int
    n_observations: int
    simulation_times: tuple[float, ...]
    unique_schedule_count: int
    block_size: int
    max_block_paths: int
    control: PortfolioControlFit | None = None


class _Moments:
    """Parallel Welford updates, with memory independent of the total paths."""
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.m2 = np.zeros(shape)

    def add(self, values):
        n = len(values)
        mean = values.mean(axis=0)
        m2 = ((values - mean) ** 2).sum(axis=0)
        if not self.n:
            self.n, self.mean, self.m2 = n, mean, m2
            return
        total = self.n + n
        delta = mean - self.mean
        self.m2 += m2 + delta**2 * (self.n * n / total)
        self.mean += delta * (n / total)
        self.n = total

    @property
    def se(self):
        return np.sqrt(np.maximum(self.m2, 0) / ((self.n - 1) * self.n))


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _setup(model, contracts, block_size, simulation_times):
    try:
        contracts = tuple(contracts)
    except TypeError as exc:
        raise ValueError("contracts must be a nonempty sequence of AsianPut contracts") from exc
    if not isinstance(model, GBM) or not contracts or any(not isinstance(c, AsianPut) for c in contracts):
        raise ValueError("a GBM model and a nonempty sequence of AsianPut contracts are required")
    _positive_integer(block_size, "block_size")
    required = {float(t) for c in contracts for t in c.fixing_times}
    try:
        grid = tuple(sorted(required)) if simulation_times is None else tuple(float(t) for t in simulation_times)
    except (TypeError, ValueError) as exc:
        raise ValueError("simulation_times must be a one-dimensional numeric sequence") from exc
    if not grid or not all(math.isfinite(t) and t > 0 for t in grid) or any(a >= b for a, b in zip(grid, grid[1:])):
        raise ValueError("simulation_times must be finite, positive and strictly increasing")
    if not required.issubset(grid):
        raise ValueError("simulation_times must contain every absolute fixing date")
    return contracts, grid


def _quantities(quantities, count):
    q = np.ones(count) if quantities is None else np.asarray(quantities, dtype=float)
    if q.shape != (count,) or not np.all(np.isfinite(q)):
        raise ValueError("quantities must be a finite vector with one entry per contract")
    return q


def geometric_put_greeks(model: GBM, contract: AsianPut) -> GreekEstimate:
    """Analytic delta and vega for the discrete geometric-average put."""
    t = np.asarray(contract.fixing_times, dtype=float)
    a = float(t.mean())
    b = float(np.minimum.outer(t, t).mean())
    sigma = model.volatility
    mu = math.log(model.spot) + (model.rate - model.dividend - .5 * sigma**2) * a
    mean = math.exp(mu + .5 * sigma**2 * b)
    s = sigma * math.sqrt(b)
    d1 = (mu - math.log(contract.strike)) / s + s
    cdf = .5 * math.erfc(d1 / math.sqrt(2))
    pdf = math.exp(-.5 * d1**2) / math.sqrt(2 * math.pi)
    discount = math.exp(-model.rate * contract.maturity)
    return GreekEstimate(-discount * mean / model.spot * cdf,
                         discount * mean * (pdf * math.sqrt(b) - sigma * (b - a) * cdf))


def _expected_controls(model, contracts):
    result = np.empty((len(contracts), 3))
    for i, c in enumerate(contracts):
        g = geometric_put_greeks(model, c)
        result[i] = geometric_put(model, c), g.delta, g.vega
    return result


def _values(model, contracts, grid, z):
    """Accumulate each distinct schedule while consuming absolute-date shocks."""
    schedules = tuple(dict.fromkeys(tuple(c.fixing_times) for c in contracts))
    schedule_index = {s: i for i, s in enumerate(schedules)}
    date_members = {t: [] for t in grid}
    for i, schedule in enumerate(schedules):
        for t in schedule:
            date_members[t].append(i)
    sums = np.zeros((len(z), len(schedules), 4))
    brownian = np.zeros(len(z))
    previous = 0.0
    for j, t in enumerate(grid):
        brownian += math.sqrt(t - previous) * z[:, j]
        previous = t
        members = date_members[t]
        if not members:
            continue
        logspot = math.log(model.spot) + (model.rate - model.dividend - .5 * model.volatility**2) * t + model.volatility * brownian
        spot = np.exp(logspot)
        sensitivity = brownian - model.volatility * t
        for i in members:
            sums[:, i, 0] += spot
            sums[:, i, 1] += logspot
            sums[:, i, 2] += spot * sensitivity
            sums[:, i, 3] += sensitivity
    raw = np.empty((len(z), len(contracts), 3))
    geometric = np.empty_like(raw)
    for i, contract in enumerate(contracts):
        mean = sums[:, schedule_index[tuple(contract.fixing_times)], :] / len(contract.fixing_times)
        arithmetic = mean[:, 0]
        g = np.exp(mean[:, 1])
        discount = math.exp(-model.rate * contract.maturity)
        for value, vega, target in ((arithmetic, mean[:, 2], raw), (g, g * mean[:, 3], geometric)):
            indicator = value < contract.strike
            target[:, i, 0] = discount * np.maximum(contract.strike - value, 0)
            target[:, i, 1] = -discount * indicator * value / model.spot
            target[:, i, 2] = -discount * indicator * vega
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(geometric)):
        raise ValueError("simulation produced nonfinite values")
    return raw, geometric


def _blocks(n_paths, seed, block_size, grid, method):
    _positive_integer(seed, "seed", minimum=0)
    rng = np.random.default_rng(seed)
    count = n_paths // 2 if method == "antithetic" else n_paths
    for start in range(0, count, block_size):
        size = min(block_size, count - start)
        yield rng.standard_normal((size, len(grid)))


def fit_portfolio_control(model, contracts, pilot_paths, seed, *, block_size=4096, simulation_times=None):
    """Fit separate coefficients for price/delta/vega on an independent pilot."""
    pilot_paths = _positive_integer(pilot_paths, "pilot_paths", 2)
    contracts, grid = _setup(model, contracts, block_size, simulation_times)
    joint = _Moments((len(contracts), 2, 3))
    cross = np.zeros((len(contracts), 3))
    for z in _blocks(pilot_paths, seed, block_size, grid, "iid"):
        x, y = _values(model, contracts, grid, z)
        block_mean_x, block_mean_y = x.mean(axis=0), y.mean(axis=0)
        cross += ((x - block_mean_x) * (y - block_mean_y)).sum(axis=0)
        if joint.n:
            cross += (block_mean_x - joint.mean[:, 0]) * (block_mean_y - joint.mean[:, 1]) * (joint.n * len(z) / (joint.n + len(z)))
        joint.add(np.stack((x, y), axis=2))
    # joint dimensions are (legs, raw-or-control, metric).
    variance = joint.m2[:, 1, :]
    betas = np.divide(cross, variance, out=np.zeros_like(cross), where=variance > 0)
    return PortfolioControlFit(betas, pilot_paths, int(seed), model, contracts, grid, block_size)


def _validate_control(control, model, contracts, grid, seed):
    if not isinstance(control, PortfolioControlFit):
        raise ValueError("geometric_cv requires a fitted PortfolioControlFit")
    if control.model != model or control.contracts != contracts or control.simulation_times != grid:
        raise ValueError("control provenance does not match model, ordered contracts and grid")
    if control.pilot_seed == seed:
        raise ValueError("evaluation seed must differ from the pilot seed")
    if control.betas.shape != (len(contracts), 3) or not np.all(np.isfinite(control.betas)):
        raise ValueError("control coefficients must have finite shape (contracts, 3)")
    _positive_integer(control.pilot_paths, "control pilot_paths", 2)
    _positive_integer(control.pilot_seed, "control pilot_seed", 0)


def price_portfolio(model, contracts, n_paths, seed, *, quantities=None, method="iid", control=None, block_size=4096, simulation_times=None):
    n_paths = _positive_integer(n_paths, "n_paths", 2)
    contracts, grid = _setup(model, contracts, block_size, simulation_times)
    q = _quantities(quantities, len(contracts))
    if method not in ("iid", "antithetic", "geometric_cv"):
        raise ValueError("method must be iid, antithetic or geometric_cv")
    if method == "antithetic" and (n_paths % 2 or n_paths < 4):
        raise ValueError("antithetic n_paths must be even and >= 4")
    expected = None
    if method == "geometric_cv":
        _validate_control(control, model, contracts, grid, seed)
        expected = _expected_controls(model, contracts)
    elif control is not None:
        raise ValueError("control is only valid with method geometric_cv")
    moments = _Moments((len(contracts) + 1, 3))
    max_paths = 0
    for z in _blocks(n_paths, seed, block_size, grid, method):
        values, g = _values(model, contracts, grid, z)
        max_paths = max(max_paths, len(z) * (2 if method == "antithetic" else 1))
        if method == "antithetic":
            opposite, _ = _values(model, contracts, grid, -z)
            values = .5 * (values + opposite)
        elif method == "geometric_cv":
            values -= control.betas * (g - expected)
        portfolio = np.sum(values * q[None, :, None], axis=1)
        moments.add(np.concatenate((values, portfolio[:, None, :]), axis=1))
    estimates = tuple(LegEstimate(*map(float, mean), *map(float, se)) for mean, se in zip(moments.mean, moments.se))
    return PortfolioEstimate(estimates[:-1], estimates[-1], tuple(map(float, q)), method, n_paths, moments.n, grid,
                             len({tuple(c.fixing_times) for c in contracts}), block_size, max_paths, control)


def fd_convergence(model, contracts, n_paths, seed, *, quantities=None, block_size=4096, simulation_times=None,
                   spot_bumps=(.01, .003, .001, .0003, .0001), volatility_bumps=(.01, .003, .001, .0003, .0001)):
    """Raw common-random-number central differences vs raw pathwise Greeks.

Spot bumps are relative; volatility bumps are absolute. Paired discrepancy
errors use FD minus pathwise values on each identical shock observation.
"""
    n_paths = _positive_integer(n_paths, "n_paths", 2)
    contracts, grid = _setup(model, contracts, block_size, simulation_times)
    q = _quantities(quantities, len(contracts))
    rows = []
    for parameter, bumps, metric in (("spot", spot_bumps, 1), ("volatility", volatility_bumps, 2)):
        for bump in bumps:
            bump = float(bump)
            amount = bump * model.spot if parameter == "spot" else bump
            base = getattr(model, parameter)
            if not math.isfinite(amount) or amount <= 0 or base - amount <= 0:
                raise ValueError("finite positive bumps must preserve positive spot and volatility")
            plus, minus = replace(model, **{parameter: base + amount}), replace(model, **{parameter: base - amount})
            moments = _Moments((len(contracts) + 1, 3))
            for z in _blocks(n_paths, seed, block_size, grid, "iid"):
                raw, _ = _values(model, contracts, grid, z)
                upper, _ = _values(plus, contracts, grid, z)
                lower, _ = _values(minus, contracts, grid, z)
                finite = (upper[:, :, 0] - lower[:, :, 0]) / (2 * amount)
                pathwise = raw[:, :, metric]
                comparisons = np.stack((finite, pathwise, finite - pathwise), axis=2)
                portfolio = np.sum(comparisons * q[None, :, None], axis=1)
                moments.add(np.concatenate((comparisons, portfolio[:, None, :]), axis=1))
            rows.append({"parameter": parameter, "bump": bump, "absolute_bump": amount, "seed": int(seed),
                         "n_paths": n_paths, "legs": [{"finite_difference": float(m[0]), "pathwise": float(m[1]),
                         "difference": float(m[2]), "difference_se": float(s[2])} for m, s in zip(moments.mean[:-1], moments.se[:-1])],
                         "portfolio": {"finite_difference": float(moments.mean[-1, 0]), "pathwise": float(moments.mean[-1, 1]),
                         "difference": float(moments.mean[-1, 2]), "difference_se": float(moments.se[-1, 2])}})
    return rows
