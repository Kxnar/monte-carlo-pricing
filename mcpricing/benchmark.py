"""Prespecified multi-scenario experiments, with pilot groups held separate."""
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter
import numpy as np
from .model import Contract
from .estimators import METHODS, estimate, tune, validate_budget
from .scenarios import SCENARIOS
from .theory import variance_constants


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hierarchical_variance_ratio(iid, method, rng, draws=1000):
    """Resample pilot groups, then independent replicates within each group.

    Groups are shared between methods; individual draws are independent. With
    few pilot groups these percentile intervals remain exploratory.
    """
    groups,repeats=iid.shape
    choices=rng.integers(groups,size=(draws,groups,1))
    left=rng.integers(repeats,size=(draws,groups,repeats))
    right=rng.integers(repeats,size=(draws,groups,repeats))
    a=iid[choices,left].reshape(draws,-1).var(axis=1,ddof=1)
    b=method[choices,right].reshape(draws,-1).var(axis=1,ddof=1)
    ratios=np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)
    return [float(x) for x in np.nanquantile(ratios,[.025,.975])]


def run(output, names=('gaussian','mixture','stress'), strikes=(80.,100.,120.),
        sizes=(1000,5000,20000), groups=5, repeats=20, pilot_n=8000,
        seed=20261006, bootstrap=1000):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty; choose a new output folder to preserve earlier results.')
    if groups<2 or repeats<2 or bootstrap<100:
        raise ValueError('Use at least two pilot groups, two repeats and 100 bootstrap resamples.')
    for n in sizes:
        validate_budget(n)
    if len(set(names))!=len(names) or len(set(strikes))!=len(strikes) or len(set(sizes))!=len(sizes):
        raise ValueError('Scenario names, strikes and budgets must be unique.')
    for name in names:
        if name not in SCENARIOS:
            raise ValueError(f'Unknown scenario {name!r}.')
    for strike in strikes:
        Contract(strike=strike)
    if pilot_n<200:
        raise ValueError('Pilot size must be at least 200.')
    output.mkdir(parents=True,exist_ok=True)
    raw, pilots, population, summaries=[],[],[],[]
    start_all=perf_counter()
    for name in names:
        scenario_index=list(SCENARIOS).index(name)
        model=SCENARIOS[name]
        # Warm up quadrature and the Python/NumPy estimator paths before timing.
        model.log_mgf_one
        for strike in strikes:
            c=Contract(strike=strike)
            price,quad_diagnostic=c.reference(model)
            # Floating-point bit pattern makes streams stable under grid reordering.
            strike_key=int(np.asarray(strike,dtype=np.float64).view(np.uint64))
            for group in range(groups):
                pilot_seed=[seed,scenario_index,strike_key,group,111]
                p=tune(model,c,seed=pilot_seed,pilot_n=pilot_n)
                constants=variance_constants(model,c,p['proposal'],p['coefficient'])
                refined=variance_constants(model,c,p['proposal'],p['coefficient'],order=512)
                if any(abs(constants[m]-refined[m])>1e-7*max(1,abs(refined[m])) for m in METHODS if m!='MH'):
                    raise ArithmeticError('Population-variance quadrature did not converge.')
                population.append(dict(scenario=name,strike=strike,group=group,**refined))
                pilots.append(dict(scenario=name,strike=strike,group=group,seed=pilot_seed,
                    proposal=dict(shift=p['proposal'].shift,inflation=p['proposal'].inflation,
                                  defensive_weight=p['proposal'].defensive_weight),
                    coefficient=p['coefficient'],mh_step=p['mh_step'],costs=p['costs'],candidates=p['candidates']))
                kwargs=dict(proposal=p['proposal'],coefficient=p['coefficient'],mh_step=p['mh_step'])
                for method in METHODS:
                    estimate(method,model,c,200,np.random.default_rng([seed,scenario_index,group,444]),**kwargs)
                for n in sizes:
                    for repeat in range(repeats):
                        children=np.random.SeedSequence([seed,scenario_index,strike_key,group,n,repeat,222]).spawn(6)
                        order=np.random.default_rng(children[-1]).permutation(len(METHODS))
                        for method_index in order:
                            method=METHODS[method_index]
                            rng=np.random.default_rng(children[method_index])
                            start=perf_counter()
                            result=estimate(method,model,c,n,rng,**kwargs)
                            seconds=perf_counter()-start
                            raw.append(dict(scenario=name,strike=strike,group=group,n=n,repeat=repeat,
                                method=method,**asdict(result),seconds=seconds,pilot_seconds=p['costs'][method],
                                reference=price,covered_95=int(abs(result.price-price)<=1.96*result.standard_error)))
                print(f'{name:8s} K={strike:g} pilot {group+1}/{groups}: ref={price:.8f}, IS={p["proposal"].shift:+.2f}/{p["proposal"].inflation:g}, CV={p["coefficient"]:.3f}',flush=True)
            for n in sizes:
                base=[r for r in raw if r['scenario']==name and r['strike']==strike and r['n']==n and r['method']=='IID']
                iid=np.array([r['price'] for r in base]).reshape(groups,repeats)
                iid_var=float(iid.var(ddof=1))
                iid_time=float(np.mean([r['seconds'] for r in base]))
                for method_index,method in enumerate(METHODS):
                    block=[r for r in raw if r['scenario']==name and r['strike']==strike and r['n']==n and r['method']==method]
                    estimates=np.array([r['price'] for r in block]).reshape(groups,repeats)
                    variance=float(estimates.var(ddof=1))
                    runtime=float(np.mean([r['seconds'] for r in block]))
                    pilot_cost=float(np.mean([r['pilot_seconds'] for r in block]))
                    average_se2=float(np.mean([r['standard_error']**2 for r in block]))
                    ci=[1.,1.] if method=='IID' else hierarchical_variance_ratio(iid,estimates,
                        np.random.default_rng([seed,scenario_index,strike_key,n,method_index,333]),bootstrap)
                    theory_values=[r[method] for r in population if r['scenario']==name and r['strike']==strike] if method!='MH' else []
                    theory=float(np.mean(theory_values)) if theory_values else None
                    iid_theory=float(np.mean([r['IID'] for r in population if r['scenario']==name and r['strike']==strike]))
                    saving_per_call=iid_time*iid_var/variance-runtime if variance>0 else -1
                    # An asymptotic cost projection, not an observed equal-error experiment.
                    break_even=math.ceil(pilot_cost/saving_per_call) if saving_per_call>0 else None
                    summaries.append(dict(scenario=name,strike=strike,n=n,method=method,pilot_groups=groups,
                        repeats_per_group=repeats,reference=price,quadrature_diagnostic=quad_diagnostic,
                        mean_price=float(estimates.mean()),bias=float(estimates.mean()-price),
                        estimator_variance=variance,rmse=float(np.sqrt(np.mean((estimates-price)**2))),
                        mean_reported_se=float(np.mean([r['standard_error'] for r in block])),
                        reported_se2_over_empirical_variance=average_se2/variance if variance else None,
                        coverage_95=float(np.mean([r['covered_95'] for r in block])),
                        mean_seconds=runtime,median_seconds=float(np.median([r['seconds'] for r in block])),
                        mean_pilot_seconds=pilot_cost,variance_ratio_vs_iid=iid_var/variance if variance else None,
                        variance_ratio_ci_low=ci[0],variance_ratio_ci_high=ci[1],
                        population_variance_constant=theory,population_variance_ratio=iid_theory/theory if theory else None,
                        empirical_over_population_variance=variance*n/theory if theory else None,
                        efficiency_vs_iid=(iid_var*iid_time)/(variance*runtime) if variance else None,
                        efficiency_including_one_pilot=(iid_var*iid_time)/(variance*(runtime+pilot_cost)) if variance else None,
                        projected_calls_to_amortise_pilot=break_even))
    write_csv(output/'trials.csv',raw)
    write_csv(output/'summary.csv',summaries)
    write_csv(output/'population.csv',population)
    (output/'pilots.json').write_text(json.dumps(pilots,indent=2),encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=seed,scenarios={k:asdict(SCENARIOS[k]) for k in names},
        strikes=list(strikes),sizes=list(sizes),pilot_groups=groups,repeats_per_group=repeats,pilot_n=pilot_n,
        bootstrap_resamples=bootstrap,total_estimates=len(raw),elapsed_seconds=perf_counter()-start_all,
        python=sys.version,numpy=np.__version__,platform=platform.platform(),machine=platform.machine(),
        source_sha256=hashes,timing='Includes estimator sampling, payoff and diagnostics, including MH burn-in. Excludes separately reported pilots, shared model/reference setup, imports and output generation.')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    from .report import render_report
    render_report(output)
    return metadata
