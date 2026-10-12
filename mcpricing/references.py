"""Independent QuantLib validation with explicit stochastic-reference uncertainty."""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import numpy as np


QUANTLIB_CONTROL_LIMITATION = "QuantLib clamps negative control-variate price estimates to zero."


def _quantlib():
    try:
        import QuantLib as ql
    except ImportError as exc:
        raise ImportError("QuantLib is required for independent Asian references; install the optional reference dependency.") from exc
    return ql


def _day_offsets(contract):
    offsets = []
    for t in contract.fixing_times:
        days = round(t*365)
        if days <= 0 or not math.isclose(t, days/365, rel_tol=0, abs_tol=1e-12):
            raise ValueError("QuantLib reference fixings must be exactly representable as integer days / 365.")
        offsets.append(days)
    if len(set(offsets)) != len(offsets):
        raise ValueError("QuantLib fixing dates must be distinct.")
    return offsets


@contextmanager
def _option(model, contract, geometric):
    ql = _quantlib()
    offsets = _day_offsets(contract)
    settings = ql.Settings.instance()
    original = settings.evaluationDate
    date = ql.Date(12, ql.October, 2026)
    try:
        settings.evaluationDate = date
        day_count, calendar = ql.Actual365Fixed(), ql.NullCalendar()
        rate = ql.YieldTermStructureHandle(ql.FlatForward(date, model.rate, day_count))
        dividend = ql.YieldTermStructureHandle(ql.FlatForward(date, model.dividend, day_count))
        volatility = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(date, calendar, model.volatility, day_count))
        process = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(model.spot)), dividend, rate, volatility)
        fixings = [date+d for d in offsets]
        option = ql.DiscreteAveragingAsianOption(ql.Average.Geometric if geometric else ql.Average.Arithmetic,
                                                1.0 if geometric else 0.0, 0, fixings,
                                                ql.PlainVanillaPayoff(ql.Option.Put, contract.strike),
                                                ql.EuropeanExercise(fixings[-1]))
        yield ql, process, option
    finally:
        settings.evaluationDate = original


def quantlib_geometric_put(model, contract):
    with _option(model, contract, True) as (ql, process, option):
        option.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process))
        return float(option.NPV())


@dataclass(frozen=True)
class ReferenceRun:
    estimate: float
    standard_error: float
    independent_samples: int
    path_evaluations: int
    seed: int
    reference_kind: str = "quantlib_arithmetic_mc"
    limitation: str = QUANTLIB_CONTROL_LIMITATION

    @property
    def price(self):
        return self.estimate


def quantlib_arithmetic_put(model, contract, *, n_samples, seed):
    """n_samples is independent antithetic pairs, requiring 2*n_samples paths."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("Reference samples must be an integer of at least two.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 1 <= seed <= 2**31-1:
        raise ValueError("Portable QuantLib seed range is integers 1 to 2**31-1; zero requests nondeterministic seeding.")
    with _option(model, contract, False) as (ql, process, option):
        option.setPricingEngine(ql.MCDiscreteArithmeticAPEngine(process, "pseudorandom", requiredSamples=int(n_samples),
                                                              seed=int(seed), antitheticVariate=True,
                                                              controlVariate=True, brownianBridge=False))
        value, error = float(option.NPV()), float(option.errorEstimate())
    return ReferenceRun(value, error, int(n_samples), 2*int(n_samples), int(seed))


@dataclass(frozen=True)
class ReferenceEstimate:
    estimate: float
    standard_error: float
    between_run_standard_error: float
    runs: int
    independent_samples: int
    path_evaluations: int
    reference_kind: str = "quantlib_arithmetic_mc"
    limitation: str = QUANTLIB_CONTROL_LIMITATION

    @property
    def price(self):
        return self.estimate


def aggregate_references(runs):
    runs = tuple(runs)
    if not runs:
        raise ValueError("At least one reference run is required.")
    if len({r.seed for r in runs}) != len(runs):
        raise ValueError("Reference runs must have distinct independent seeds.")
    if any(not math.isfinite(r.estimate) or not math.isfinite(r.standard_error) or r.standard_error < 0 for r in runs):
        raise ValueError("Reference estimates and standard errors must be finite.")
    values = [r.estimate for r in runs]
    j = len(runs)
    return ReferenceEstimate(float(np.mean(values)), math.sqrt(sum(r.standard_error**2 for r in runs))/j,
                             float(np.std(values, ddof=1)/math.sqrt(j)) if j > 1 else 0.0, j,
                             sum(r.independent_samples for r in runs), sum(r.path_evaluations for r in runs))
