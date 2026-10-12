"""Cross-platform command line interface: python -m mcpricing --help."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .model import Contract
from .scenarios import SCENARIOS
from .estimators import METHODS, estimate, tune
from .theory import reference_delta, variance_constants


def main():
    parser=argparse.ArgumentParser(description='Monte Carlo pricing experiments under synthetic terminal laws.')
    subs=parser.add_subparsers(dest='command',required=True)
    price=subs.add_parser('price',help='Estimate a put price, compare to quadrature and estimate delta.')
    price.add_argument('--scenario',choices=SCENARIOS,default='mixture')
    price.add_argument('--strike',type=float,default=100.)
    price.add_argument('--samples',type=int,default=20000)
    price.add_argument('--seed',type=int,default=42)
    bench=subs.add_parser('benchmark',help='Run prespecified experiments and generate a report.')
    bench.add_argument('--output',type=Path,default=Path('local-results'))
    bench.add_argument('--scenarios',choices=SCENARIOS,nargs='+',default=list(SCENARIOS))
    bench.add_argument('--strikes',type=float,nargs='+',default=[80,100,120])
    bench.add_argument('--sizes',type=int,nargs='+',default=[1000,5000,20000])
    bench.add_argument('--groups',type=int,default=5)
    bench.add_argument('--repeats',type=int,default=20)
    bench.add_argument('--pilot-samples',type=int,default=8000)
    bench.add_argument('--bootstrap',type=int,default=1000)
    bench.add_argument('--seed',type=int,default=20261006)
    report=subs.add_parser('report',help='Regenerate report and figures from stored CSV files.')
    report.add_argument('directory',type=Path)
    wp=subs.add_parser('workbench-pilot',help='Measure the finite A/B/C timing design before freezing.')
    wp.add_argument('--output',type=Path,required=True)
    wf=subs.add_parser('workbench-freeze',help='Select on TRAIN and freeze a source-hashed finite campaign.')
    wf.add_argument('--pilot',type=Path,required=True);wf.add_argument('--output',type=Path,required=True)
    wr=subs.add_parser('workbench-run',help='Execute all final cases from the frozen configuration.')
    wr.add_argument('--configuration',type=Path,required=True);wr.add_argument('--output',type=Path,required=True)
    rr=subs.add_parser('workbench-report',help='Regenerate the workbench report from saved evidence.')
    rr.add_argument('directory',type=Path)
    ap=subs.add_parser('asian-price',help='Price a GBM arithmetic Asian put with independent geometric control.')
    ap.add_argument('--strike',type=float,default=100.);ap.add_argument('--days',type=int,default=365)
    ap.add_argument('--fixings',type=int,default=12)
    pd=subs.add_parser('portfolio-demo',help='Nine signed Asian legs, covariance-aware risk, hedge and first-order P&L.')
    for command in (ap,pd):
        command.add_argument('--paths',type=int,default=16384);command.add_argument('--pilot-paths',type=int,default=4096)
        command.add_argument('--seed',type=int,default=42);command.add_argument('--block-size',type=int,default=4096)
    args=parser.parse_args()
    try:
        if args.command.startswith('workbench-'):
            if args.command=='workbench-report':
                from .workbench_report import render_workbench_report
                render_workbench_report(args.directory)
            else:
                from .workbench import run_pilot,freeze_configuration,run_workbench
                if args.command=='workbench-pilot':result=run_pilot(args.output)
                elif args.command=='workbench-freeze':result=freeze_configuration(args.pilot,args.output)
                else:result=run_workbench(args.configuration,args.output)
                print(json.dumps({k:result[k] for k in ('schema_version','actual_counts','validation','forecast','adaptation') if k in result},indent=2))
        elif args.command=='portfolio-demo':
            from .workbench import portfolio_demo
            portfolio_demo(args.paths,args.pilot_paths,args.seed,args.block_size)
        elif args.command=='asian-price':
            from .asian import GBM,AsianPut,fit_control,price_asian
            if args.fixings<1 or args.days<args.fixings:raise ValueError('Use positive fixings and days >= fixings')
            model=GBM();contract=AsianPut(args.strike,tuple(round(args.days*i/args.fixings)/365 for i in range(1,args.fixings+1)),args.days/365)
            fit=fit_control(model,contract,pilot_paths=args.pilot_paths,seed=args.seed+1,block_size=args.block_size)
            print(json.dumps(dict(model=asdict(model),contract=asdict(contract),control=asdict(fit),estimates={m:asdict(price_asian(model,contract,n_paths=args.paths,seed=args.seed,method=m,control=fit if m=='geometric_cv' else None,block_size=args.block_size)) for m in ('iid','antithetic','geometric_cv')}),indent=2))
        elif args.command=='benchmark':
            from .benchmark import run
            meta=run(args.output,args.scenarios,args.strikes,args.sizes,args.groups,args.repeats,
                     args.pilot_samples,args.seed,args.bootstrap)
            print(f'Saved {meta["total_estimates"]:,} estimates to {args.output.resolve()}')
        elif args.command=='report':
            from .report import render_report
            render_report(args.directory)
        else:
            model=SCENARIOS[args.scenario]
            c=Contract(strike=args.strike)
            model.log_mgf_one  # Shared setup is not a method-specific tuning cost.
            p=tune(model,c,args.seed)
            constants=variance_constants(model,c,p['proposal'],p['coefficient'])
            output=dict(scenario=args.scenario,contract=asdict(c),reference=c.reference(model)[0],
                theoretical_variance_constants=constants,pilot_seconds=p['costs'],estimates={})
            for i,method in enumerate(METHODS):
                result=estimate(method,model,c,args.samples,np.random.default_rng([args.seed,111,i]),
                    proposal=p['proposal'],coefficient=p['coefficient'],mh_step=p['mh_step'])
                output['estimates'][method]=asdict(result)
            x=model.sample(np.random.default_rng([args.seed,222]),args.samples)
            delta=c.delta_values(model,x)
            output['delta']=dict(pathwise_estimate=float(delta.mean()),standard_error=float(delta.std(ddof=1)/len(x)**.5),reference=reference_delta(model,c))
            print(json.dumps(output,indent=2))
    except (ValueError,ArithmeticError) as exc:
        parser.error(str(exc))


if __name__=='__main__':
    main()
