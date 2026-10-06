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
    args=parser.parse_args()
    try:
        if args.command=='benchmark':
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
