import importlib.util
import math
import unittest
try:
    import QuantLib as ql
except ImportError:
    ql = None
from mcpricing.gbm import GBM, AsianPut
from mcpricing.asian import geometric_put, fit_control, price_asian
from mcpricing.references import quantlib_geometric_put, quantlib_arithmetic_put, aggregate_references


@unittest.skipIf(importlib.util.find_spec("QuantLib") is None, "optional QuantLib dependency unavailable")
class ReferenceTests(unittest.TestCase):
    def test_geometric_analytic_regular_irregular_rates_dividends(self):
        for days in ((365,), (90, 180, 270, 365), (7, 31, 156, 400)):
            c = AsianPut(102, tuple(t/365 for t in days), days[-1]/365)
            for model in (GBM(), GBM(rate=-.02, dividend=.04, volatility=.35)):
                native = geometric_put(model, c)
                independent = quantlib_geometric_put(model, c)
                self.assertLessEqual(abs(native-independent), 1e-10*max(1, native))

    def test_arithmetic_combined_uncertainty_and_pair_accounting(self):
        model = GBM(dividend=.01)
        c = AsianPut(100, tuple(t/365 for t in (30, 100, 190, 365)), 1)
        fit = fit_control(model, c, pilot_paths=5000, seed=550)
        native = price_asian(model, c, n_paths=80000, seed=551, method="geometric_cv", control=fit)
        reference = quantlib_arithmetic_put(model, c, n_samples=80000, seed=552)
        combined = math.hypot(native.standard_error, reference.standard_error)
        self.assertLess(abs(native.price-reference.estimate), 5*combined)
        self.assertEqual(reference.independent_samples, 80000)
        self.assertEqual(reference.path_evaluations, 160000)
        self.assertIn("clamps", reference.limitation)

    def test_global_date_restored_and_day_grid_rejected(self):
        original = ql.Settings.instance().evaluationDate
        temporary = ql.Date(2, ql.January, 2025)
        ql.Settings.instance().evaluationDate = temporary
        try:
            quantlib_geometric_put(GBM(), AsianPut(100, (1,), 1))
            self.assertEqual(ql.Settings.instance().evaluationDate, temporary)
            with self.assertRaises(ValueError):
                quantlib_geometric_put(GBM(), AsianPut(100, (.12345,), .12345))
            self.assertEqual(ql.Settings.instance().evaluationDate, temporary)
        finally:
            ql.Settings.instance().evaluationDate = original

    def test_reference_aggregation(self):
        from mcpricing.references import ReferenceRun
        runs = [ReferenceRun(1, .2, 20, 40, 1), ReferenceRun(2, .3, 20, 40, 2)]
        combined = aggregate_references(runs)
        self.assertEqual(combined.estimate, 1.5)
        self.assertAlmostEqual(combined.standard_error, math.sqrt(.2**2+.3**2)/2)
        self.assertAlmostEqual(combined.between_run_standard_error, .5)
        with self.assertRaises(ValueError):
            aggregate_references([runs[0], runs[0]])

    def test_reference_requires_nonzero_reproducible_seed(self):
        model = GBM()
        c = AsianPut(100, (90/365, 180/365, 270/365, 1), 1)
        for seed in (0, -1, 2**31, 2**32-1, True, 1.5):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "Portable QuantLib seed range"):
                    quantlib_arithmetic_put(model, c, n_samples=200, seed=seed)
        first = quantlib_arithmetic_put(model, c, n_samples=200, seed=2**31-1)
        second = quantlib_arithmetic_put(model, c, n_samples=200, seed=2**31-1)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
