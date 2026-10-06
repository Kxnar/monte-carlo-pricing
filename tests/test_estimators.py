import math
import unittest
from dataclasses import replace
import numpy as np
from mcpricing import Mixture, Contract, Proposal
from mcpricing.estimators import estimate, antithetic_values, tune, METHODS
from mcpricing.theory import variance_constants, reference_delta
from mcpricing.scenarios import SCENARIOS


class EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=Mixture()
        cls.contract=Contract()
        cls.pilot=tune(cls.model,cls.contract,seed=441,pilot_n=2000)

    def test_all_methods_and_reproducibility(self):
        p=self.pilot
        reference=self.contract.reference(self.model)[0]
        for method in METHODS:
            kw=dict(proposal=p['proposal'],coefficient=p['coefficient'],mh_step=p['mh_step'])
            result=estimate(method,self.model,self.contract,20000,np.random.default_rng(220),**kw)
            repeat=estimate(method,self.model,self.contract,20000,np.random.default_rng(220),**kw)
            self.assertEqual(result,repeat)
            self.assertLess(abs(result.price-reference),6*result.standard_error)
            self.assertEqual(result.payoff_evaluations,20000)
            self.assertEqual(result.observations,10000 if method=='AV' else 20000)

    def test_antithetic_asymmetric_mixture_is_unbiased(self):
        model=SCENARIOS['stress']
        c=Contract(strike=120)
        v=antithetic_values(model,c,300000,np.random.default_rng(221))
        self.assertLess(abs(v.mean()-c.reference(model)[0]),5*v.std()/math.sqrt(len(v)))

    def test_population_variance_matches_repeated_simulations(self):
        p=self.pilot
        constants=variance_constants(self.model,self.contract,p['proposal'],p['coefficient'])
        for method in ('IID','IS','AV','CV'):
            estimates=[estimate(method,self.model,self.contract,4000,np.random.default_rng([222,i]),
                       proposal=p['proposal'],coefficient=p['coefficient']).price for i in range(150)]
            empirical=np.var(estimates,ddof=1)*4000
            self.assertLess(abs(empirical/constants[method]-1),.45)

    def test_exact_target_proposal_recovers_iid_variance(self):
        constants=variance_constants(self.model,self.contract,Proposal(self.model,0,1),0)
        self.assertAlmostEqual(constants['IID'],constants['IS'],places=9)
        self.assertAlmostEqual(constants['IID'],constants['CV'],places=9)

    def test_delta_matches_common_random_numbers_and_reference(self):
        model,c=self.model,self.contract
        x=model.sample(np.random.default_rng(223),200000)
        delta=c.delta_values(model,x)
        eps=1e-4
        finite=(replace(c,spot=c.spot+eps).payoff(model,x)-replace(c,spot=c.spot-eps).payoff(model,x))/(2*eps)
        self.assertLess(abs(delta.mean()-finite.mean()),2e-5)
        self.assertLess(abs(delta.mean()-reference_delta(model,c)),5*delta.std()/math.sqrt(len(x)))

    def test_theory_refinement_all_scenarios(self):
        for model in SCENARIOS.values():
            c=Contract()
            p=Proposal(model)
            coarse=variance_constants(model,c,p,-.4,128)
            fine=variance_constants(model,c,p,-.4,256)
            for method in ('IID','IS','AV','CV'):
                self.assertAlmostEqual(coarse[method],fine[method],places=6)

    def test_invalid_budgets(self):
        for n in (0,199,201,200.5):
            with self.assertRaises(ValueError):
                estimate('IID',self.model,self.contract,n,np.random.default_rng(1))


if __name__=='__main__':
    unittest.main()
