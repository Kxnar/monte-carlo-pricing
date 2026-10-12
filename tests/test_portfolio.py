import math
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np

from mcpricing.asian import GBM, AsianPut, geometric_put
from mcpricing.gbm import summary_from_brownian
from mcpricing.portfolio import (fit_portfolio_control, price_portfolio,
    geometric_put_greeks, fd_convergence, PortfolioControlFit, _values)


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.model = GBM(100, .03, .01, .25)
        self.contracts = (AsianPut(90, (.1, .3), .3), AsianPut(105, (.1, .2, .6), .6),
                          AsianPut(110, (.2, .6), .6))
        self.grid = (.1, .2, .3, .6)

    def test_same_union_singletons_all_metrics_and_errors(self):
        fit = fit_portfolio_control(self.model, self.contracts, 1003, 44, block_size=111)
        for method in ("iid", "antithetic", "geometric_cv"):
            batch = price_portfolio(self.model, self.contracts, 2002, 45, method=method,
                                    control=fit if method == "geometric_cv" else None, block_size=113)
            for i, c in enumerate(self.contracts):
                single_fit = fit_portfolio_control(self.model, [c], 1003, 44, block_size=111, simulation_times=self.grid)
                np.testing.assert_allclose(single_fit.betas, fit.betas[i:i+1], atol=1e-11, rtol=0)
                single = price_portfolio(self.model, [c], 2002, 45, method=method,
                    control=single_fit if method == "geometric_cv" else None, block_size=113, simulation_times=self.grid)
                np.testing.assert_allclose(list(vars(single.legs[0]).values()), list(vars(batch.legs[i]).values()), atol=1e-11, rtol=0)

    def test_union_mapping_oracle_and_pathwise_values(self):
        z = np.random.default_rng(61).normal(size=(71, 4))
        W = np.cumsum(z * np.sqrt(np.diff((0,) + self.grid)), axis=1)
        raw, geo = _values(self.model, self.contracts, self.grid, z)
        for i, c in enumerate(self.contracts):
            summary = summary_from_brownian(self.model, c.fixing_times, W[:, [self.grid.index(t) for t in c.fixing_times]])
            d = math.exp(-self.model.rate*c.maturity)
            for average, vega, values in ((summary.arithmetic, summary.arithmetic_vega, raw),
                                         (summary.geometric, summary.geometric_vega, geo)):
                expected = np.stack((d*np.maximum(c.strike-average, 0),
                    -d*(average<c.strike)*average/self.model.spot, -d*(average<c.strike)*vega), axis=1)
                np.testing.assert_allclose(values[:, i], expected, atol=1e-12, rtol=1e-13)

    def test_signed_duplicates_cancel_and_double_errors(self):
        c = self.contracts[1]
        contracts = (c, c)
        fit = fit_portfolio_control(self.model, contracts, 503, 62)
        for method in ("iid", "antithetic", "geometric_cv"):
            kw = dict(method=method, control=fit if method == "geometric_cv" else None)
            zero = price_portfolio(self.model, contracts, 1000, 63, quantities=[1, -1], **kw)
            self.assertEqual(list(vars(zero.portfolio).values()), [0]*6)
            doubled = price_portfolio(self.model, contracts, 1000, 63, quantities=[1, 1], **kw)
            np.testing.assert_allclose(list(vars(doubled.portfolio).values()), 2*np.array(list(vars(doubled.legs[0]).values())), rtol=1e-14)
            self.assertEqual(doubled.unique_schedule_count, 1)

    def test_antithetic_whole_pair_oracle(self):
        z = np.random.default_rng(64).normal(size=(101, 4))
        plus, _ = _values(self.model, self.contracts, self.grid, z)
        minus, _ = _values(self.model, self.contracts, self.grid, -z)
        pairs = .5*(plus+minus)
        actual = price_portfolio(self.model, self.contracts, 202, 64, method="antithetic", block_size=17)
        self.assertEqual(actual.n_observations, 101)
        for i, leg in enumerate(actual.legs):
            np.testing.assert_allclose([leg.price,leg.delta,leg.vega], pairs[:,i].mean(axis=0), atol=1e-12)
            np.testing.assert_allclose([leg.price_se,leg.delta_se,leg.vega_se], pairs[:,i].std(axis=0,ddof=1)/math.sqrt(101), atol=1e-12)

    def test_block_invariance_remainders_and_bounded_shapes(self):
        for method in ("iid", "antithetic", "geometric_cv"):
            fits = [fit_portfolio_control(self.model, self.contracts, 1003, 65, block_size=b) for b in (7, 128)]
            np.testing.assert_allclose(fits[0].betas, fits[1].betas, atol=1e-11, rtol=0)
            calls = []
            def observe(model, contracts, grid, z):
                calls.append(z.shape)
                return _values(model, contracts, grid, z)
            with patch("mcpricing.portfolio._values", side_effect=observe):
                small = price_portfolio(self.model, self.contracts, 2004, 66, method=method,
                        block_size=7, control=fits[0] if method == "geometric_cv" else None)
            large = price_portfolio(self.model, self.contracts, 2004, 66, method=method,
                        block_size=128, control=fits[1] if method == "geometric_cv" else None)
            self.assertTrue(all(n<=7 and m==4 for n,m in calls))
            self.assertTrue(any(n<7 for n,m in calls))
            for a,b in zip(small.legs+(small.portfolio,), large.legs+(large.portfolio,)):
                np.testing.assert_allclose(list(vars(a).values()), list(vars(b).values()), atol=1e-11, rtol=0)

    def test_analytic_geometric_greeks_and_one_fixing(self):
        for c in self.contracts + (AsianPut(105, (.6,), .6),):
            g = geometric_put_greeks(self.model, c)
            for name, amount, expected in (("spot", 1e-3, g.delta), ("volatility", 1e-5, g.vega)):
                fd = (geometric_put(replace(self.model, **{name:getattr(self.model,name)+amount}),c)-
                      geometric_put(replace(self.model, **{name:getattr(self.model,name)-amount}),c))/(2*amount)
                self.assertAlmostEqual(expected, fd, places=7)
        c = AsianPut(105, (.6,), .6)
        t=c.maturity; sigma=self.model.volatility
        d1=(math.log(self.model.spot/c.strike)+(self.model.rate-self.model.dividend+.5*sigma*sigma)*t)/(sigma*math.sqrt(t))
        g=geometric_put_greeks(self.model,c)
        self.assertAlmostEqual(g.delta,-math.exp(-self.model.dividend*t)*.5*math.erfc(d1/math.sqrt(2)),places=13)
        self.assertAlmostEqual(g.vega,self.model.spot*math.exp(-self.model.dividend*t)*math.sqrt(t)*math.exp(-.5*d1*d1)/math.sqrt(2*math.pi),places=13)

    def test_raw_fd_ladder_paired_discrepancies(self):
        rows=fd_convergence(self.model,self.contracts,4001,67,block_size=127,
                            quantities=[1,-2,3],spot_bumps=(.01,.0001),volatility_bumps=(.01,.0001))
        self.assertEqual(len(rows),4)
        for row in rows:
            result=row['portfolio']
            self.assertAlmostEqual(result['finite_difference']-result['pathwise'],result['difference'],places=11)
            self.assertGreaterEqual(result['difference_se'],0)
        for metric in ('spot','volatility'):
            selected=[r for r in rows if r['parameter']==metric]
            fine=selected[-1]['portfolio']
            self.assertLess(abs(fine['difference']),.003+5*fine['difference_se'])

    def test_no_negative_control_clamp(self):
        fit=fit_portfolio_control(self.model,self.contracts,23,68)
        forced=replace(fit,betas=np.full((3,3),100.))
        result=price_portfolio(self.model,self.contracts,31,69,method='geometric_cv',control=forced)
        z=np.random.default_rng(69).normal(size=(31,4)); raw,g=_values(self.model,self.contracts,self.grid,z)
        from mcpricing.portfolio import _expected_controls
        expected=(raw-forced.betas*(g-_expected_controls(self.model,self.contracts))).mean(axis=0)
        np.testing.assert_allclose([list(vars(leg).values())[:3] for leg in result.legs],expected,atol=1e-11)
        self.assertTrue(np.any(expected[:,0]<0))

    def test_validation_and_fit_is_frozen(self):
        fit=fit_portfolio_control(self.model,self.contracts,33,70)
        self.assertFalse(fit.betas.flags.writeable)
        invalid=({}, {'simulation_times':(.1,.2,.3)}, {'simulation_times':(.1,.2,.3,.3,.6)},
                 {'simulation_times':(.1,.2,.3,float('nan'),.6)}, {'simulation_times':[[.1,.2],[.3,.6]]},
                 {'quantities':[1]}, {'quantities':[1,2,float('inf')]},
                 {'block_size':0}, {'method':'bad'}, {'method':'antithetic'}, {'method':'geometric_cv','control':fit})
        for kw in invalid[1:]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                price_portfolio(self.model,self.contracts,33,70,**kw)
        for badfit in (replace(fit,betas=np.ones((3,2))), replace(fit,betas=np.full((3,3),np.nan)),
                       replace(fit,model=replace(self.model,spot=101)), replace(fit,contracts=self.contracts[::-1]),
                       replace(fit,simulation_times=(.1,.2,.3,.6,.7))):
            with self.assertRaises(ValueError):
                price_portfolio(self.model,self.contracts,32,71,method='geometric_cv',control=badfit)
        for n in (0,1,2.5,True):
            with self.assertRaises(ValueError): price_portfolio(self.model,self.contracts,n,72)

    def test_demo_report_schema_and_overwrite_refusal(self):
        from examples.portfolio_risk_demo import run_evaluation
        from mcpricing.workbench_common import TRIAL_FIELDS
        import json
        import tempfile
        from pathlib import Path
        result=run_evaluation(output=None,model=self.model,contracts=self.contracts,quantities=(1,-2,3),
                              n_paths=258,pilot_paths=129,pilot_seed=73,evaluation_seed=74,block_size=32,
                              spot_bumps=(.001,),volatility_bumps=(.001,))
        json.dumps(result,allow_nan=False)
        self.assertTrue(result['validation']['same_union_singleton_pass'])
        self.assertEqual(len(result['trials']),36)
        self.assertTrue(all(tuple(row)==TRIAL_FIELDS for row in result['trials']))
        self.assertEqual(result['seeds'],{'pilot':73,'evaluation':74})
        self.assertEqual(result['results']['antithetic']['n_observations'],129)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'already.json'
            output.write_text('preserve',encoding='utf-8')
            with self.assertRaises(FileExistsError):run_evaluation(output=output)
            self.assertEqual(output.read_text(encoding='utf-8'),'preserve')


if __name__ == '__main__':
    unittest.main()
