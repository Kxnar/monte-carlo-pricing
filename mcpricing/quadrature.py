"""Gauss-Legendre integration, with a fourth-power map for endpoint cusps."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=8)
def rule(order):
    return np.polynomial.legendre.leggauss(order)


def positive_integral(function, lower, upper, order=256):
    """Integrate on a nonnegative interval; z=t^4 smooths gennorm's cusp."""
    if upper <= lower:
        return 0.0
    nodes, weights = rule(order)
    a, b = lower**.25, upper**.25
    t = (a+b)/2 + (b-a)*nodes/2
    return float((b-a)/2*np.dot(weights, function(t**4)*4*t**3))


def radius(shape, scale=0.0):
    # Convexity supplies an exponential tangent bound for the omitted tail.
    r = max(70**(1/shape), (2*scale/shape)**(1/(shape-1)) if scale else 0)
    while r**shape-scale*r < 60:
        r *= 1.2
    return r


def tail_bound(r, shape, tilt=0.0):
    return float(np.exp(tilt*r-r**shape)/(shape*r**(shape-1)-tilt))
