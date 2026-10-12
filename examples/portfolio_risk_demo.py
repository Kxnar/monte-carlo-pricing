"""Reproducible Asian-put portfolio prices and first-order risk; no VaR claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import tracemalloc
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcpricing.asian import GBM, AsianPut
from mcpricing.portfolio import (fit_portfolio_control, price_portfolio,
                                  fd_convergence, PortfolioControlFit)
from mcpricing.workbench_common import environment_metadata, stable_seed, trial_record


def default_contracts():
    """Nine legs, with integer calendar-day offsets / 365 and 3/6/12 fixings."""
    result = []
    for days, count in ((91, 3), (182, 6), (365, 12)):
        offsets = tuple(int(round(days*i/count)) for i in range(1, count+1))
        times = tuple(day/365 for day in offsets)
        result.extend(AsianPut(strike, times, days/365) for strike in (90., 100., 110.))
    return tuple(result)


def _serialize_fit(fit):
    return {"betas": fit.betas.tolist(), "pilot_paths": fit.pilot_paths,
            "pilot_seed": fit.pilot_seed, "block_size": fit.block_size,
            "model": asdict(fit.model), "contracts": [asdict(c) for c in fit.contracts],
            "simulation_times": list(fit.simulation_times)}


def _serialize_estimate(result):
    return {"legs": [asdict(leg) for leg in result.legs], "portfolio": asdict(result.portfolio),
            "quantities": list(result.quantities), "method": result.method,
            "n_paths": result.n_paths, "n_observations": result.n_observations,
            "simulation_times": list(result.simulation_times), "unique_schedule_count": result.unique_schedule_count,
            "block_size": result.block_size, "max_block_paths": result.max_block_paths}


def run_evaluation(*, output=None, model=None, contracts=None, quantities=None, n_paths=16384,
                   pilot_paths=4096, pilot_seed=81011, evaluation_seed=81012, block_size=4096,
                   spot_bumps=(.01,.003,.001,.0003,.0001),
                   volatility_bumps=(.01,.003,.001,.0003,.0001)):
    """Run an explicitly sized evaluation, optionally writing a new JSON artifact.

No overwrites. Pilot and evaluation seeds must differ. Memory profiling is an
untimed extra run. Spot and volatility FD ladders use raw estimators and CRN.
"""
    destination = None if output is None else Path(output)
    if destination is not None and destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    model = GBM(100., .03, .01, .25) if model is None else model
    default_book = contracts is None
    contracts = default_contracts() if default_book else tuple(contracts)
    if quantities is None:
        quantities = (1.,-2.,1.,-1.,2.,-1.,1.,-1.,.5) if default_book else (1.,)*len(contracts)
    else:
        quantities = tuple(quantities)
    if pilot_seed == evaluation_seed:
        raise ValueError("pilot and evaluation seeds must differ")
    grid = tuple(sorted({t for c in contracts for t in c.fixing_times}))
    setup_started = time.perf_counter()
    metadata = environment_metadata()
    setup_seconds = time.perf_counter()-setup_started
    started = time.perf_counter()
    control = fit_portfolio_control(model, contracts, pilot_paths, pilot_seed, block_size=block_size)
    pilot_seconds = time.perf_counter()-started
    estimates = {}; timing = {}
    for method in ("iid", "antithetic", "geometric_cv"):
        started = time.perf_counter()
        result = price_portfolio(model, contracts, n_paths, evaluation_seed, quantities=quantities,
                    method=method, control=control if method=="geometric_cv" else None, block_size=block_size)
        timing[method] = {"seconds": time.perf_counter()-started,
                          "physical_paths": n_paths, "payoff_evaluations": n_paths*len(contracts),
                          "statistical_observations": result.n_observations}
        estimates[method] = result
    # Keep the same union grid to compare identical random variables. Own-grid
    # singleton timings are a different workload and produce different paths.
    singleton_started = time.perf_counter()
    singletons = []
    for i, c in enumerate(contracts):
        singleton_fit = PortfolioControlFit(control.betas[i:i+1], control.pilot_paths, control.pilot_seed,
                            model, (c,), grid, control.block_size)
        singletons.append(price_portfolio(model, (c,), n_paths, evaluation_seed, method="geometric_cv",
                          control=singleton_fit, block_size=block_size, simulation_times=grid))
    singleton_seconds = time.perf_counter()-singleton_started
    discrepancy = max(abs(getattr(single.legs[0], field)-getattr(estimates['geometric_cv'].legs[i], field))
        for i,single in enumerate(singletons) for field in ("price","delta","vega","price_se","delta_se","vega_se"))
    own_started=time.perf_counter()
    for c in contracts:
        price_portfolio(model, (c,), n_paths, evaluation_seed, block_size=block_size)
    own_seconds=time.perf_counter()-own_started
    ladders = fd_convergence(model, contracts, n_paths, evaluation_seed, quantities=quantities,
                  block_size=block_size, simulation_times=grid, spot_bumps=spot_bumps, volatility_bumps=volatility_bumps)
    # The same run is repeated solely for allocation measurement, never timing.
    tracemalloc.start()
    try:
        price_portfolio(model, contracts, n_paths, evaluation_seed, quantities=quantities,
                        method="geometric_cv", control=control, block_size=block_size)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    total = estimates['geometric_cv'].portfolio
    coefficient_id = f"portfolio-control-{stable_seed(pilot_seed, 'frozen-coefficients', control.betas):08x}"
    trials=[]
    for method, estimate in estimates.items():
        for metric in ("price", "delta", "vega"):
            for i, leg in enumerate(estimate.legs + (estimate.portfolio,)):
                is_portfolio = i == len(contracts)
                trials.append(trial_record(suite="B", case_id=f"{'portfolio' if is_portfolio else 'leg-'+str(i)}:{metric}",
                    split="evaluation", workload_id="joint-absolute-union-three-metrics", valuation_index=0,
                    method=method, group=0, repeat=0, n_requested=n_paths, independent_samples=estimate.n_observations,
                    path_evaluations=n_paths, payoff_evaluations=n_paths*(len(contracts) if is_portfolio else 1),
                    fixing_evaluations=n_paths*len(grid), quantity=None if is_portfolio else quantities[i],
                    estimate=getattr(leg,metric), standard_error=getattr(leg,metric+'_se'),
                    seed_key=f"portfolio/evaluation/{evaluation_seed}",
                    tuning_seed_key=f"portfolio/pilot/{pilot_seed}" if method=='geometric_cv' else None,
                    production_seconds=timing[method]['seconds'], tuning_seconds=pilot_seconds if method=='geometric_cv' else 0.,
                    setup_seconds=setup_seconds, cumulative_seconds=setup_seconds+timing[method]['seconds']+(pilot_seconds if method=='geometric_cv' else 0.),
                    peak_memory_bytes=peak_bytes if method=='geometric_cv' else None,
                    memory_kind='tracemalloc separate untimed run' if method=='geometric_cv' else None,
                    coefficient_id=coefficient_id if method=='geometric_cv' else None, retuned=False,block_size=block_size))
    payload = {"schema_version": "asian-workbench-v1", "experiment": "portfolio_risk",
        "environment":metadata,"trials":trials,
        "model": asdict(model), "contracts": [asdict(c) for c in contracts], "quantities": list(quantities),
        "seeds": {"pilot":pilot_seed,"evaluation":evaluation_seed},
        "control": _serialize_fit(control), "results": {m:_serialize_estimate(e) for m,e in estimates.items()},
        "validation": {"same_union_singleton_max_absolute_difference":discrepancy,
                       "same_union_singleton_tolerance":1e-11, "same_union_singleton_pass":discrepancy<=1e-11,
                       "raw_crn_fd_convergence":ladders},
        "timing": {"setup_seconds":setup_seconds,"pilot_seconds":pilot_seconds,"methods":timing,
            "common_union_singleton_cv_seconds":singleton_seconds,
            "common_union_batch_cv_seconds":timing['geometric_cv']['seconds'],
            "common_union_speedup":singleton_seconds/timing['geometric_cv']['seconds'],
            "own_grid_singleton_iid_seconds":own_seconds,
            "note":"Batch and common-union singletons calculate all three metrics; own-grid IID singletons are a different workload. Single timings are descriptive."},
        "memory": {"current_traced_bytes":current_bytes,"peak_traced_bytes":peak_bytes,
            "block_size":block_size,"max_block_paths":estimates['geometric_cv'].max_block_paths,
            "measurement":"untimed tracemalloc run; traced allocations, not process RSS"},
        "risk": {"delta_hedge_underlying_units":-total.delta,
            "scenarios":[{"spot_change":ds,"volatility_change":dv,
                          "first_order_pnl":total.delta*ds+total.vega*dv}
                          for ds,dv in ((-1.,0.),(1.,0.),(0.,-.01),(0.,.01),(-1.,.01))],
            "label":"First-order delta/vega P&L under GBM; no VaR, gamma or market calibration claim"}}
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload,handle,indent=2,allow_nan=False)
            handle.write("\n")
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--paths',type=int,default=16384)
    parser.add_argument('--pilot-paths',type=int,default=4096)
    parser.add_argument('--block-size',type=int,default=4096)
    args=parser.parse_args()
    result=run_evaluation(output=args.output,n_paths=args.paths,pilot_paths=args.pilot_paths,block_size=args.block_size)
    print("Geometric-control leg estimates (vega per unit absolute volatility)")
    print("leg    K  days quantity     price (SE)       delta (SE)        vega (SE)")
    for i,(contract,quantity,leg) in enumerate(zip(result['contracts'],result['quantities'],result['results']['geometric_cv']['legs'])):
        print(f"{i:3} {contract['strike']:4.0f} {round(contract['maturity']*365):5} {quantity:8.2f} "
              f"{leg['price']:8.4f} ({leg['price_se']:.4f}) {leg['delta']:8.4f} ({leg['delta_se']:.4f}) "
              f"{leg['vega']:8.4f} ({leg['vega_se']:.4f})")
    print("\nSigned portfolio totals")
    print("method          price (SE)          delta (SE)          vega (SE)")
    for method,estimate in result['results'].items():
        p=estimate['portfolio']
        print(f"{method:14} {p['price']:9.4f} ({p['price_se']:.4f})  {p['delta']:9.4f} ({p['delta_se']:.4f})  {p['vega']:9.4f} ({p['vega_se']:.4f})")
    print(f"Underlying delta hedge: {result['risk']['delta_hedge_underlying_units']:.5f} units")
    print(result['risk']['label'])
    print("spot change   volatility change   first-order P&L")
    for scenario in result['risk']['scenarios']:
        print(f"{scenario['spot_change']:11.2f} {scenario['volatility_change']:19.3f} {scenario['first_order_pnl']:17.5f}")


if __name__ == '__main__':
    main()
