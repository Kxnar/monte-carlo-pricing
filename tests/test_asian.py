import math
import unittest
from dataclasses import replace
import numpy as np
from mcpricing.gbm import GBM, AsianPut, path_summaries, summary_from_brownian
from mcpricing.asian import geometric_put, fit_control, price_asian


class AsianTests(unittest.TestCase):
    def setUp(self):
        self.model = GBM()
        self.contract = AsianPut(100, (90/365, 180/365, 270/365, 1.0), 1.0)

    def test_manual_two_step_paths_and_derivatives(self):
        times = (0.25, 1.0)
        z = np.array([[1.0, -0.5], [-0.2, 0.8]])
        W = np.cumsum(z*np.sqrt([0.25, 0.75]), axis=1)
        S = self.model.spot*np.exp((self.model.rate-self.model.dividend-0.5*self.model.volatility**2)*np.array(times)+self.model.volatility*W)
        result = path_summaries(self.model, times, z)
        supplied = summary_from_brownian(self.model, times, W)
        np.testing.assert_allclose(result.arithmetic, S.mean(axis=1), atol=1e-12)
        np.testing.assert_allclose(result.geometric, np.sqrt(S.prod(axis=1)), atol=1e-12)
        np.testing.assert_allclose(result.arithmetic_vega, (S*(W-self.model.volatility*np.array(times))).mean(axis=1), atol=1e-12)
        for name in ("arithmetic", "geometric", "arithmetic_vega", "geometric_vega"):
            np.testing.assert_array_equal(getattr(result, name), getattr(supplied, name))
            eps = 1e-6
            high = path_summaries(replace(self.model, volatility=self.model.volatility+eps), times, z)
            low = path_summaries(replace(self.model, volatility=self.model.volatility-eps), times, z)
            if name.endswith("vega"):
                base = name.removesuffix("_vega")
                np.testing.assert_allclose(getattr(result, name), (getattr(high, base)-getattr(low, base))/(2*eps), rtol=1e-8, atol=1e-8)

    def test_one_fixing_black_scholes_and_identity(self):
        for strike in (70, 100, 130):
            c = AsianPut(strike, (1.2,), 1.2)
            model = GBM(rate=-0.01, dividend=0.02)
            d1 = (math.log(model.spot/strike)+(model.rate-model.dividend+model.volatility**2/2)*c.maturity)/(model.volatility*math.sqrt(c.maturity))
            d2 = d1-model.volatility*math.sqrt(c.maturity)
            cdf = lambda x: 0.5*math.erfc(-x/math.sqrt(2))
            expected = strike*math.exp(-model.rate*c.maturity)*cdf(-d2)-model.spot*math.exp(-model.dividend*c.maturity)*cdf(-d1)
            self.assertAlmostEqual(geometric_put(model, c), expected, places=12)
        summary = path_summaries(model, c.fixing_times, np.array([[1.0], [-1.0]]))
        np.testing.assert_array_equal(summary.arithmetic, summary.geometric)
        np.testing.assert_array_equal(summary.arithmetic_vega, summary.geometric_vega)

    def test_antithetic_pair_standard_error(self):
        n, seed = 2000, 42
        z = np.random.default_rng(seed).standard_normal((n//2, len(self.contract.fixing_times)))
        positive = self.contract.payoff(self.model, path_summaries(self.model, self.contract.fixing_times, z).arithmetic)
        negative = self.contract.payoff(self.model, path_summaries(self.model, self.contract.fixing_times, -z).arithmetic)
        values = (positive+negative)/2
        estimate = price_asian(self.model, self.contract, n_paths=n, seed=seed, method="antithetic", block_size=137)
        self.assertAlmostEqual(estimate.price, values.mean(), places=12)
        self.assertAlmostEqual(estimate.standard_error, values.std(ddof=1)/math.sqrt(n//2), places=12)
        self.assertEqual(estimate.independent_samples, n//2)
        self.assertEqual(estimate.path_evaluations, n)
        self.assertEqual(estimate.fixing_evaluations, 4*n)

    def test_control_manual_fit_provenance_and_unclamped_mean(self):
        fit = fit_control(self.model, self.contract, pilot_paths=2000, seed=22, block_size=113)
        z = np.random.default_rng(22).standard_normal((2000, 4))
        summary = path_summaries(self.model, self.contract.fixing_times, z)
        x = self.contract.payoff(self.model, summary.arithmetic)
        y = self.contract.payoff(self.model, summary.geometric)
        expected = np.cov(x, y, ddof=1)
        self.assertAlmostEqual(fit.variance, expected[1, 1], places=12)
        self.assertAlmostEqual(fit.coefficient, expected[0, 1]/expected[1, 1], places=12)
        self.assertEqual(fit.provenance["seed"], 22)
        with self.assertRaises(ValueError):
            price_asian(self.model, self.contract, n_paths=20, seed=22, method="geometric_cv", control=fit)
        with self.assertRaises(ValueError):
            price_asian(replace(self.model, rate=.02), self.contract, n_paths=20, seed=23, method="geometric_cv", control=fit)
        extreme = replace(fit, coefficient=-10000)
        estimate = price_asian(self.model, self.contract, n_paths=2, seed=0, method="geometric_cv", control=extreme)
        # A deliberately poor frozen beta can produce a negative mean; it must be retained.
        summary = path_summaries(self.model, self.contract.fixing_times, np.random.default_rng(0).standard_normal((2, 4)))
        values = self.contract.payoff(self.model, summary.arithmetic)-extreme.coefficient*(self.contract.payoff(self.model, summary.geometric)-geometric_put(self.model, self.contract))
        self.assertAlmostEqual(estimate.price, values.mean(), places=10)
        self.assertLess(estimate.price, 0)

    def test_block_size_does_not_change_draws(self):
        fit = fit_control(self.model, self.contract, pilot_paths=1000, seed=8)
        small = fit_control(self.model, self.contract, pilot_paths=1000, seed=8, block_size=7)
        self.assertAlmostEqual(small.coefficient, fit.coefficient, places=13)
        for method in ("iid", "antithetic", "geometric_cv"):
            a = price_asian(self.model, self.contract, n_paths=2000, seed=9, method=method, control=fit, block_size=7)
            b = price_asian(self.model, self.contract, n_paths=2000, seed=9, method=method, control=fit, block_size=1024)
            self.assertAlmostEqual(a.price, b.price, places=12)
            self.assertAlmostEqual(a.standard_error, b.standard_error, places=12)

    def test_invalid_inputs(self):
        for kwargs in ({"spot": 0}, {"volatility": -1}, {"rate": float("nan")}, {"dividend": float("inf")}):
            with self.assertRaises(ValueError):
                GBM(**kwargs)
        for times, maturity in (((), 1), ((0, 1), 1), ((1, .5), .5), ((.5, 1), 2), ((float("nan"),), 1)):
            with self.assertRaises(ValueError):
                AsianPut(100, times, maturity)
        for z in (np.ones((3, 2)), np.array([[1, 2, 3, np.nan]]), np.ones(4)):
            with self.assertRaises(ValueError):
                path_summaries(self.model, self.contract.fixing_times, z)
        for n in (True, 0, 1, 2.5):
            with self.assertRaises(ValueError):
                price_asian(self.model, self.contract, n_paths=n, seed=1)
        for n in (2, 3, 5):
            with self.assertRaises(ValueError):
                price_asian(self.model, self.contract, n_paths=n, seed=1, method="antithetic")
        with self.assertRaises(ValueError):
            price_asian(self.model, self.contract, n_paths=10, seed=1, method="geometric_cv")
        with self.assertRaises(ValueError):
            fit_control(self.model, self.contract, pilot_paths=10, seed=1, block_size=0)


if __name__ == "__main__":
    unittest.main()
