"""Run after installing the package: python examples/first_experiment.py."""
import math
import numpy as np
from mcpricing import Mixture, Contract, Proposal
from mcpricing.estimators import estimate
from mcpricing.theory import variance_constants

model = Mixture()
contract = Contract(strike=100)
proposal = Proposal(model, shift=-0.2, inflation=1.25)
constants = variance_constants(model, contract, proposal, coefficient=0)
ratio = constants['IID'] / constants['IS']
print(f'Reference price: {contract.reference(model)[0]:.8f}')
print(f'Population variance reduction: {ratio:.3f}x')
print(f'Expected standard-error reduction: {math.sqrt(ratio):.3f}x')
for n in (1000, 5000, 20000):
    for index, method in enumerate(('IID', 'IS')):
        result = estimate(method, model, contract, n,
                          np.random.default_rng([42, n, index]), proposal=proposal)
        print(f'{method:3s} N={n:6,d} price={result.price:.5f} SE={result.standard_error:.5f}')
print('This uses a fixed proposal, so no tuning cost is measured here.')
