import math
import unittest
import numpy as np
from mcpricing.quadrature import positive_integral, radius
from mcpricing import Mixture, Contract, Proposal
from mcpricing.estimators import iid_values, importance_values, mh_values, standard_error


class PricingTests(unittest.TestCase):
    def test_density_and_sampler(self):
        model = Mixture()
        # Integrate each component separately so all density cusps are interval endpoints.
        area = sum(w*b/math.gamma(1/b)*positive_integral(lambda z: np.exp(-z**b),0,radius(b))
                   for w,b in zip(model.weights,model.shapes))
        self.assertAlmostEqual(area, 1, places=8)
        x = np.linspace(-3, 3, 51)
        np.testing.assert_allclose(model.logpdf(x), [model.scalar_logpdf(v) for v in x], atol=1e-12)
        draws = model.sample(np.random.default_rng(4), 300000)
        mean = sum(w*m for w, m in zip(model.weights, model.locations))
        second = sum(w*(m*m+a*a*math.gamma(3/b)/math.gamma(1/b)) for w,m,a,b in
                     zip(model.weights, model.locations, model.scales, model.shapes))
        self.assertLess(abs(draws.mean()-mean), 5*draws.std()/math.sqrt(len(draws)))
        self.assertLess(abs(draws.var()-(second-mean**2)), .001)

    def test_black_scholes_limit(self):
        sigma, maturity = .2, 1.4
        scale = math.sqrt(2*maturity)*sigma
        model = Mixture(locations=(0, 0), scales=(scale, scale), shapes=(2, 2))
        for strike in (70, 100, 130):
            c = Contract(strike=strike, maturity=maturity)
            d1 = (math.log(c.spot/strike)+(c.rate+sigma*sigma/2)*maturity)/(sigma*math.sqrt(maturity))
            d2 = d1-sigma*math.sqrt(maturity)
            cdf = lambda x: .5*math.erfc(-x/math.sqrt(2))
            expected = strike*math.exp(-c.rate*maturity)*cdf(-d2)-c.spot*cdf(-d1)
            self.assertAlmostEqual(c.reference(model)[0], expected, places=8)
        self.assertAlmostEqual(model.log_mgf_one, sigma*sigma*maturity/2, places=10)

    def test_martingale_and_put_call_parity(self):
        model, c = Mixture(), Contract()
        draws = model.sample(np.random.default_rng(5), 300000)
        terminal = c.spot*np.exp(c.shift(model)+draws)
        self.assertLess(abs(terminal.mean()-c.spot*math.exp(c.rate*c.maturity)), 5*terminal.std()/math.sqrt(len(draws)))
        put = c.payoff(model, draws)
        call = math.exp(-c.rate*c.maturity)*np.maximum(terminal-c.strike, 0)
        np.testing.assert_allclose(call-put, math.exp(-c.rate*c.maturity)*(terminal-c.strike), atol=1e-12)

    def test_proposal_support_weights_and_estimators(self):
        model, c = Mixture(), Contract()
        p = Proposal(model)
        x = p.sample(np.random.default_rng(7), 100000)
        weights = np.exp(model.logpdf(x)-p.logpdf(x))
        self.assertLessEqual(weights.max(), 1/p.defensive_weight+1e-10)
        self.assertLess(abs(weights.mean()-1), 5*weights.std()/math.sqrt(len(x)))
        reference = c.reference(model)[0]
        for values in (iid_values(model,c,150000,np.random.default_rng(8)),
                       importance_values(model,c,p,150000,np.random.default_rng(9))):
            self.assertLess(abs(values.mean()-reference), 5*standard_error(values))
        values, acceptance = mh_values(model,c,80000,np.random.default_rng(10))
        self.assertTrue(0 < acceptance < 1)
        self.assertLess(abs(values.mean()-reference), 5*standard_error(values,correlated=True))

    def test_reference_and_price_bounds(self):
        model = Mixture()
        prices = []
        for k in (50, 80, 100, 120, 150):
            c = Contract(strike=k)
            value, error = c.reference(model)
            self.assertLess(abs(value-c.reference(model,order=512)[0]), 1e-8)
            self.assertGreaterEqual(value, max(k*math.exp(-c.rate*c.maturity)-c.spot, 0)-1e-9)
            self.assertLessEqual(value, k*math.exp(-c.rate*c.maturity))
            self.assertLess(error, 1e-7)
            prices.append(value)
        self.assertEqual(prices, sorted(prices))

    def test_invalid_inputs(self):
        for kwargs in ({"weights":(.4,.4)}, {"shapes":(.8,2)}, {"scales":(-1,1)}):
            with self.assertRaises(ValueError):
                Mixture(**kwargs)
        with self.assertRaises(ValueError):
            Contract(strike=-1)


if __name__ == "__main__":
    unittest.main()
