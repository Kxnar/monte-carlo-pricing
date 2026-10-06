"""European put estimation under a synthetic terminal risk-neutral mixture."""
from dataclasses import dataclass, replace
from functools import cached_property
import math
import numpy as np
from .quadrature import positive_integral, radius, tail_bound


@dataclass(frozen=True)
class Mixture:
    weights: tuple = (0.8, 0.2)
    locations: tuple = (0.02, -0.08)
    scales: tuple = (0.16, 0.30)
    shapes: tuple = (2.0, 1.5)

    def __post_init__(self):
        groups = (self.weights, self.locations, self.scales, self.shapes)
        if any(len(g) != 2 for g in groups):
            raise ValueError("Exactly two components are required.")
        if not all(math.isfinite(v) for g in groups for v in g):
            raise ValueError("Parameters must be finite.")
        if any(w <= 0 for w in self.weights) or not math.isclose(sum(self.weights), 1):
            raise ValueError("Positive mixture weights must sum to one.")
        if any(a <= 0 for a in self.scales) or any(b <= 1 for b in self.shapes):
            raise ValueError("Scales must be positive; shapes > 1 ensure finite exponential moments.")

    @cached_property
    def log_constants(self):
        return tuple(math.log(w*b/(2*a))-math.lgamma(1/b)
                     for w, a, b in zip(self.weights, self.scales, self.shapes))

    def logpdf(self, x):
        x = np.asarray(x)
        terms = [c - np.abs((x-m)/a)**b for c, m, a, b in
                 zip(self.log_constants, self.locations, self.scales, self.shapes)]
        return np.logaddexp(*terms)

    def scalar_logpdf(self, x):
        c0, c1 = self.log_constants
        a = c0 - abs((x-self.locations[0])/self.scales[0])**self.shapes[0]
        b = c1 - abs((x-self.locations[1])/self.scales[1])**self.shapes[1]
        hi, lo = max(a, b), min(a, b)
        return hi + math.log1p(math.exp(lo-hi))

    def sample(self, rng, n):
        components = rng.choice(2, size=n, p=self.weights)
        shapes = np.asarray(self.shapes)[components]
        radii = rng.gamma(1/shapes)**(1/shapes)
        signs = 2*rng.integers(0, 2, size=n)-1
        return np.asarray(self.locations)[components] + np.asarray(self.scales)[components]*signs*radii

    @cached_property
    def log_mgf_one(self):
        # Standardise each component before quadrature; integrate both sides of its cusp.
        total = 0.0
        for w, m, a, b in zip(self.weights, self.locations, self.scales, self.shapes):
            normaliser = b/(2*math.gamma(1/b))
            cutoff = radius(b, a)
            halves = sum(positive_integral(lambda u: np.exp(sign*a*u-u**b), 0, cutoff)
                         for sign in (-1, 1))
            total += w*math.exp(m)*normaliser*halves
        return math.log(total)


@dataclass(frozen=True)
class Contract:
    spot: float = 100.0
    strike: float = 100.0
    rate: float = 0.03
    maturity: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.spot, self.strike, self.rate, self.maturity)):
            raise ValueError("Contract parameters must be finite.")
        if min(self.spot, self.strike, self.maturity) <= 0:
            raise ValueError("Spot, strike and maturity must be positive.")

    def shift(self, model):
        return self.rate*self.maturity - model.log_mgf_one

    def payoff(self, model, x):
        # Evaluate only in-the-money terms, avoiding exp overflow for large positive x.
        log_ratio = np.asarray(x) + self.shift(model) + math.log(self.spot/self.strike)
        return math.exp(-self.rate*self.maturity)*self.strike*(-np.expm1(np.minimum(log_ratio, 0)))

    def reference(self, model, order=256):
        cutoff = math.log(self.strike/self.spot)-self.shift(model)
        value, error = 0.0, 0.0
        discount = math.exp(-self.rate*self.maturity)
        for w, m, a, b in zip(model.weights, model.locations, model.scales, model.shapes):
            top = (cutoff-m)/a
            norm = b/(2*math.gamma(1/b))
            def integrand(z):
                log_ratio = m+a*z-cutoff
                payoff = -np.expm1(np.minimum(log_ratio, 0))*self.strike*discount
                return payoff*norm*np.exp(-np.abs(z)**b)
            bound = radius(b)
            def integrate(n):
                left = positive_integral(lambda u: integrand(-u), max(0, -top), bound, n)
                right = positive_integral(integrand, 0, min(max(top, 0), bound), n)
                return left+right
            result, refined = integrate(order), integrate(order*2)
            value += w*refined
            # Refinement difference is a diagnostic, not a rigorous quadrature bound.
            error += w*(abs(result-refined)+2*self.strike*discount*norm*tail_bound(bound,b))
        return value, error


@dataclass(frozen=True)
class Proposal:
    target: Mixture
    shift: float = -0.2
    inflation: float = 1.25
    defensive_weight: float = 0.1

    def __post_init__(self):
        if not math.isfinite(self.shift) or not math.isfinite(self.inflation) or self.inflation <= 0:
            raise ValueError("Proposal shift must be finite and inflation positive.")
        if not 0 < self.defensive_weight < 1:
            raise ValueError("Defensive weight must be between zero and one.")

    @cached_property
    def tilted(self):
        return replace(self.target, locations=tuple(m+self.shift for m in self.target.locations),
                       scales=tuple(a*self.inflation for a in self.target.scales))

    def logpdf(self, x):
        d = self.defensive_weight
        return np.logaddexp(math.log(d)+self.target.logpdf(x),
                           math.log1p(-d)+self.tilted.logpdf(x))

    def sample(self, rng, n):
        defensive = rng.random(n) < self.defensive_weight
        x = np.empty(n)
        x[defensive] = self.target.sample(rng, int(defensive.sum()))
        x[~defensive] = self.tilted.sample(rng, int((~defensive).sum()))
        return x

