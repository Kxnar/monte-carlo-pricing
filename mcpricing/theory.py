"""Numerical population moments: a second check on empirical variance claims."""
import math
import numpy as np
from .quadrature import positive_integral, radius


def component_expectation(model, j, function, breakpoints=(), order=256):
    m, a, b = model.locations[j], model.scales[j], model.shapes[j]
    bound = radius(b, 2*a)
    total = 0.0
    for sign in (-1, 1):
        cuts = [0.0,bound]+[sign*(x-m)/a for x in breakpoints if 0 < sign*(x-m)/a < bound]
        cuts = sorted(set(cuts))
        for lower, upper in zip(cuts[:-1],cuts[1:]):
            total += positive_integral(lambda z: function(m+sign*a*z)*np.exp(-z**b),lower,upper,order)
    return total*b/(2*math.gamma(1/b))


def expectation(model, function, breakpoints=(), order=256):
    return sum(w*component_expectation(model,j,function,breakpoints,order) for j,w in enumerate(model.weights))


def variance_constants(model, contract, proposal, coefficient, order=256):
    """Return constants V such that Var(price estimate) = V / payoff budget.

    Conditional on frozen pilot parameters; exact up to quadrature. MH has no
    population constant here because its autocovariance sum is not integrated.
    """
    price = contract.reference(model)[0]
    cutoff = math.log(contract.strike/contract.spot)-contract.shift(model)
    cuts = (*model.locations,*proposal.tilted.locations,cutoff)
    payoff = lambda x: contract.payoff(model,x)
    second = expectation(model,lambda x:payoff(x)**2,cuts,order)
    iid = max(0.0,second-price*price)
    is_second = expectation(model,lambda x:payoff(x)**2*np.exp(model.logpdf(x)-proposal.logpdf(x)),cuts,order)
    control = lambda x:contract.discounted_stock(model,x)-contract.spot
    covariance = expectation(model,lambda x:payoff(x)*control(x),cuts,order)
    stock_variance = contract.spot**2*math.expm1(model.log_mgf(2)-2*model.log_mgf_one)
    cv = iid-2*coefficient*covariance+coefficient**2*stock_variance
    antithetic_second = 0.0
    for j,w in enumerate(model.weights):
        centre = model.locations[j]
        pair = lambda x:(payoff(x)+payoff(2*centre-x))/2
        antithetic_second += w*component_expectation(model,j,lambda x:pair(x)**2,
                                                    (cutoff,2*centre-cutoff),order)
    return dict(IID=iid,IS=max(0.0,is_second-price*price),AV=max(0.0,2*(antithetic_second-price*price)),
                CV=max(0.0,cv),optimal_cv_coefficient=covariance/stock_variance,
                optimal_cv_constant=max(0.0,iid-covariance**2/stock_variance))


def reference_delta(model, contract):
    cutoff = math.log(contract.strike/contract.spot)-contract.shift(model)
    return expectation(model,lambda x:contract.delta_values(model,x),(*model.locations,cutoff))
