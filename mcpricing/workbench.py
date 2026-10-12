"""Finite, source-frozen workbench campaigns; legacy experiments remain separate."""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import hashlib
import json
import math
import os
import traceback
import tracemalloc
import numpy as np

from .asian import GBM, AsianPut, fit_control, geometric_put, price_asian
from .portfolio import (PortfolioControlFit, fit_portfolio_control, price_portfolio,
                        fd_convergence, geometric_put_greeks)
from .references import quantlib_geometric_put, quantlib_arithmetic_put, aggregate_references
from .reuse import (ReuseCase, ReuseConfig, ReusePolicy, run_timing_pilot,
                    run_reuse_case, select_reuse_threshold, summarize_reuse)
from .model import Contract, Mixture
from .scenarios import SCENARIOS
from .workbench_common import (environment_metadata, refuse_nonempty_output, stable_seed,
                               trial_record, write_json, write_csv)

METHODS = ("iid", "antithetic", "geometric_cv")
ROOT_SEED = 20261012


def quantlib_seed(root,*keys):
    """Nonzero signed-31-bit seeds also fit Windows SWIG BigInteger."""
    return 1+stable_seed(root,*keys)%(2**31-1)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_contracts():
    return tuple(AsianPut(k, tuple(round(days*i/count)/365 for i in range(1,count+1)), days/365)
                 for days,count in ((91,3),(182,6),(365,12)) for k in (90.,100.,110.))


def _asian_case(case_id, model, strike, days):
    return dict(case_id=case_id, model=asdict(model), strike=strike, fixing_days=list(days))


def asian_cases():
    monthly = [round(365*i/12) for i in range(1,13)]
    return [_asian_case(f"monthly-K{k}", GBM(), float(k), monthly) for k in (80,100,120)] + [
        _asian_case("stress-52", GBM(100.,-.01,.02,.45), 100., [round(730*i/52) for i in range(1,53)]),
        _asian_case("irregular", GBM(100.,.03,0.,.3), 105., [7,31,80,121,220,365]),
        _asian_case("one-fixing", GBM(), 100., [365])]


def _decode_asian(case):
    times = tuple(day/365 for day in case["fixing_days"])
    return GBM(**case["model"]), AsianPut(case["strike"], times, times[-1])


def _serialize_contract(c):
    return dict(strike=c.strike, fixing_days=[round(t*365) for t in c.fixing_times])


def portfolio_cases():
    return [dict(case_id="signed-nine-leg", model=asdict(GBM(100.,.03,.01,.25)),
                 contracts=[_serialize_contract(c) for c in default_contracts()],
                 quantities=[1.,-2.,1.,-1.,2.,-1.,1.,-1.,.5]),
            dict(case_id="irregular-book", model=asdict(GBM(100.,.03,.01,.25)),
                 contracts=[dict(strike=105., fixing_days=[7,31,80,121,220,365])], quantities=[-1.5])]


def _decode_book(case):
    model = GBM(**case["model"])
    contracts = tuple(AsianPut(c["strike"],tuple(d/365 for d in c["fixing_days"]),c["fixing_days"][-1]/365)
                      for c in case["contracts"])
    return model, contracts, tuple(case["quantities"])


def reuse_cases(stress=True):
    sequences = {"repeated": [95.]*32, "nearby": [95.,97.,99.,102.,105.,103.,100.,98.]*4,
                 "jump": [105.,107.,109.,115.,117.,113.,89.,92.]*4}
    return [dict(case_id=f"heldout-{name}-{workload}", split="heldout", workload_id=workload,
                 model=asdict(SCENARIOS[name]), contracts=[asdict(Contract(strike=k)) for k in ks])
            for name in (("mixture","stress") if stress else ("mixture",)) for workload,ks in sequences.items()]


def _decode_reuse(case):
    return ReuseCase(case["case_id"],Mixture(**{k:tuple(v) for k,v in case["model"].items()}),
                     tuple(Contract(**c) for c in case["contracts"]),case["split"],case["workload_id"])


def _timed(call):
    start=perf_counter(); result=call()
    return result,perf_counter()-start


def _memory(call):
    start=perf_counter(); tracemalloc.start()
    try:
        call(); _,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return dict(peak_memory_bytes=peak, seconds=perf_counter()-start,
                memory_kind="tracemalloc separate untimed run; not process RSS")


def _failure(output, error, stage, completed):
    write_json(output/"failure.json",dict(stage=stage,error_type=type(error).__name__,
               error=str(error),traceback=traceback.format_exc(),completed=completed))
    write_json(output/"status.json",dict(status="failed",stage=stage,completed=completed))


def run_pilot(output):
    output=refuse_nonempty_output(output); start=perf_counter(); raw=[]; memories=[];trials=[];controls=[];reference_runs=[]
    write_json(output/"environment.json",environment_metadata())
    try:
        for case in (asian_cases()[1],asian_cases()[3]):
            model,contract=_decode_asian(case)
            for block in (512,4096):
                tune_seed=stable_seed(ROOT_SEED,"pilot","A",case["case_id"],"tune",block)
                production_seed=stable_seed(ROOT_SEED,"pilot","A",case["case_id"],"production",block)
                control,seconds=_timed(lambda:fit_control(model,contract,pilot_paths=8192,seed=tune_seed,block_size=block))
                controls.append(dict(suite="A",case_id=case["case_id"],block_size=block,seconds=seconds,**asdict(control)))
                raw.append(dict(suite="A",case_id=case["case_id"],kind="tuning",block_size=block,seconds=seconds,n=8192))
                for method in METHODS:
                    result,seconds=_timed(lambda:price_asian(model,contract,n_paths=65536,seed=production_seed,method=method,
                                                control=control if method=="geometric_cv" else None,block_size=block))
                    raw.append(dict(suite="A",case_id=case["case_id"],kind=method,block_size=block,seconds=seconds,n=65536))
                    trials.append(trial_record(suite="A",case_id=case["case_id"],split="train",workload_id="timing-pilot",valuation_index=0,
                        method=method,group=0,repeat=0,n_requested=65536,independent_samples=result.independent_samples,path_evaluations=result.path_evaluations,
                        payoff_evaluations=result.path_evaluations,fixing_evaluations=result.fixing_evaluations,quantity="price",estimate=result.price,
                        standard_error=result.standard_error,seed_key=str(production_seed),tuning_seed_key=str(tune_seed) if method=="geometric_cv" else None,
                        production_seconds=seconds,tuning_seconds=0.,block_size=block,coefficient_id=f"pilot-A-{case['case_id']}-{block}" if method=="geometric_cv" else None))
                memories.append(dict(suite="A",case_id=case["case_id"],block_size=block,
                                     seed=production_seed,**_memory(lambda:price_asian(model,contract,n_paths=65536,seed=production_seed,block_size=block))))
            for j in range(2):
                ref_seed=quantlib_seed(ROOT_SEED,"pilot-QL",case["case_id"],j)
                ref,seconds=_timed(lambda:quantlib_arithmetic_put(model,contract,n_samples=16384,seed=ref_seed))
                reference_runs.append(dict(case_id=case["case_id"],seconds=seconds,**asdict(ref)))
                raw.append(dict(suite="A",case_id=case["case_id"],kind="reference",block_size=0,seconds=seconds,n=16384))
        model,contracts,q=_decode_book(portfolio_cases()[0])
        for block in (512,4096):
            tune_seed=stable_seed(ROOT_SEED,"pilot","B","tune",block);production_seed=stable_seed(ROOT_SEED,"pilot","B","production",block)
            fit,seconds=_timed(lambda:fit_portfolio_control(model,contracts,4096,tune_seed,block_size=block))
            controls.append(dict(suite="B",case_id="signed-nine-leg",seconds=seconds,**asdict(fit)))
            raw.append(dict(suite="B",case_id="signed-nine-leg",kind="tuning",block_size=block,seconds=seconds,n=4096))
            for method in METHODS:
                result,seconds=_timed(lambda:price_portfolio(model,contracts,16384,production_seed,quantities=q,method=method,
                    control=fit if method=="geometric_cv" else None,block_size=block))
                raw.append(dict(suite="B",case_id="signed-nine-leg",kind=method,block_size=block,seconds=seconds,n=16384))
                for metric in ("price","delta","vega"):
                    trials.append(trial_record(suite="B",case_id=f"signed-nine-leg:portfolio:{metric}",split="train",workload_id="timing-pilot",valuation_index=0,
                        method=method,group=0,repeat=0,n_requested=16384,independent_samples=result.n_observations,path_evaluations=16384,
                        payoff_evaluations=16384*len(contracts),fixing_evaluations=16384*len(result.simulation_times),quantity=metric,
                        estimate=getattr(result.portfolio,metric),standard_error=getattr(result.portfolio,metric+"_se"),seed_key=str(production_seed),
                        tuning_seed_key=str(tune_seed) if method=="geometric_cv" else None,production_seconds=seconds if metric=="price" else 0.,
                        tuning_seconds=0.,block_size=block,coefficient_id=f"pilot-B-{block}" if method=="geometric_cv" else None))
            memories.append(dict(suite="B",block_size=block,seed=production_seed,**_memory(lambda:price_portfolio(model,contracts,16384,production_seed,block_size=block))))
        train=ReuseCase("train-timing-mixture",SCENARIOS["mixture"],tuple(Contract(strike=k) for k in [90,91,93,96,105,110,111,115]),"train","timing")
        c=run_timing_pilot(train,ReuseConfig(root_seed=stable_seed(ROOT_SEED,"timing"),repeats=2))
        write_csv(output/"C-timing-trials.csv",c["trials"]); write_json(output/"C-timing-diagnostics.json",c["diagnostics"])
        write_csv(output/"trials.csv",trials+c["trials"]);write_json(output/"pilot-controls.json",controls);write_json(output/"pilot-reference-runs.json",reference_runs)
        block=min((512,4096),key=lambda b:sum(r["seconds"] for r in raw if r["block_size"]==b and r["kind"] in METHODS))
        # Forecast includes every physical retune; use direct pilot wall extrapolation.
        af=sum(r["seconds"] for r in raw if r["suite"]=="A" and r["block_size"]==block and r["kind"] in METHODS)*3*24*(4096+16384+65536)/65536
        rf=sum(r["seconds"] for r in raw if r["kind"]=="reference")/4*24*262144/16384
        bf=sum(r["seconds"] for r in raw if r["suite"]=="B" and r["block_size"]==block and r["kind"] in METHODS)*8*10
        cf=c["diagnostics"]["wall_seconds"]*6*3*4*4
        forecast=dict(A_production_seconds=af,A_reference_seconds=rf,B_validation_seconds=bf,C_seconds=cf,
                      total_seconds=af+rf+bf+cf)
        payload=dict(schema_version="workbench-pilot-v1",chosen_block_size=block,raw=raw,memory=memories,
                     C_wall_seconds=c["diagnostics"]["wall_seconds"],forecast=forecast,wall_seconds=perf_counter()-start)
        write_json(output/"pilot.json",payload);write_json(output/"status.json",dict(status="complete"))
        return payload
    except Exception as exc:
        write_csv(output/"trials-partial.csv",trials);write_json(output/"pilot-controls-partial.json",controls);write_json(output/"pilot-reference-runs-partial.json",reference_runs)
        write_json(output/"pilot-partial.json",dict(raw=raw,memory=memories));_failure(output,exc,"pilot",len(raw));raise


def expected_counts(configuration):
    a=configuration["A"];b=configuration["B"];c=configuration["C"]
    return dict(A=len(a["cases"])*len(a["sizes"])*len(a["methods"])*a["groups"]*a["repeats"],
                A_controls=len(a["cases"])*a["groups"],A_reference_runs=len(a["cases"])*a["reference_runs"],
                B=sum((len(case["contracts"])+1)*3*len(b["methods"]) for case in b["cases"]),
                C=sum(len(case["contracts"])*(len(c["sizes"])*4+1)*c["groups"]*c["repeats"] for case in c["cases"]))


def validate_configuration(configuration):
    if configuration.get("schema_version")!="workbench-config-v1":raise ValueError("Unknown configuration schema")
    if expected_counts(configuration)!=configuration["expected_counts"]:raise ValueError("Expected counts do not match frozen design")
    train_ids=set(configuration["training"]["case_ids"])
    if train_ids & {c["case_id"] for c in configuration["C"]["cases"]}:raise ValueError("TRAIN and heldout identifiers overlap")
    if any(c["split"]!="heldout" for c in configuration["C"]["cases"]):raise ValueError("Final C cases must be heldout")
    train_strikes={c["strike"] for case in configuration["training"].get("cases",[]) for c in case["contracts"]}
    final_strikes={c["strike"] for case in configuration["C"]["cases"] for c in case["contracts"]}
    if train_strikes & final_strikes:raise ValueError("TRAIN and heldout strikes overlap")
    if configuration["seeds"]["train"]==configuration["seeds"]["final"]:raise ValueError("TRAIN and final seed roots overlap")
    json.dumps(configuration,allow_nan=False)


def freeze_configuration(pilot, output):
    pilot=Path(pilot);output=refuse_nonempty_output(output);start=perf_counter()
    p=json.loads((pilot/"pilot.json").read_text());env=environment_metadata()
    if env["source_hashes"]!=json.loads((pilot/"environment.json").read_text())["source_hashes"]:
        raise ValueError("Sources changed after timing pilot; run a new pilot")
    try:
        train=[]
        for anchor,ks in ((90.,[90.,91.,93.,96.,108.,114.,91.,90.]),
                          (110.,[110.,111.,112.,116.,128.,134.,111.,110.])):
            train.append(ReuseCase(f"train-threshold-{int(anchor)}",SCENARIOS["mixture"],tuple(Contract(strike=k) for k in ks),"train","threshold-selection"))
        seed_train=stable_seed(ROOT_SEED,"TRAIN");seed_final=stable_seed(ROOT_SEED,"FINAL")
        result=select_reuse_threshold(train,ReuseConfig(root_seed=seed_train,sample_sizes=(2000,8000),pilot_groups=2,repeats=3),(.02,.05,.10,math.inf))
        write_csv(output/"training-trials.csv",result["trials"])
        write_json(output/"training-evidence.json",dict(evidence=result["evidence"],diagnostics=result["diagnostics"],
                   selection_split=result["selection_split"],objective=result["objective"]))
        threshold="infinity" if math.isinf(result["policy"].threshold) else result["policy"].threshold
        # Adaptations are allowed only here, based solely on the timing forecast.
        adjusted=p["forecast"]["total_seconds"]>900
        stress=not (p["forecast"]["C_seconds"]>650)
        configuration=dict(schema_version="workbench-config-v1",root_seed=ROOT_SEED,seeds=dict(train=seed_train,final=seed_final,quantlib_mapping="1 + semantic_seed % (2**31 - 1), nonzero Windows SWIG-compatible"),
            block_size=p["chosen_block_size"],limits=dict(production_seconds=900,tracked_memory_bytes=512*1024**2),
            adaptation=dict(reduced_A=adjusted,include_stress_C=stress,reason="pre-freeze timing forecast",forecast=p["forecast"]),
            A=dict(cases=asian_cases(),sizes=[4096,16384,32768 if adjusted else 65536],methods=list(METHODS),groups=3,repeats=8,
                   pilot_paths=8192,reference_runs=4,reference_pairs=131072 if adjusted else 262144),
            B=dict(cases=portfolio_cases(),methods=list(METHODS),n_paths=131072,pilot_paths=8192,timing_repeats=3,
                   bumps=[.01,.003,.001,.0003,.0001]),
            C=dict(cases=reuse_cases(stress),sizes=[2000,8000,32000],groups=3,repeats=8,pilot_n=1000,
                   threshold=threshold,quad_order=256,reference_order=512,target_rmse=.05,
                   horizons=[1,2,4,8,16,32],bootstrap_repeats=300,inner_bootstrap_repeats=100),
            gates=dict(geometric_price_relative=1e-10,pooled_sigma=5.,pooled_floor=1e-10,singleton_absolute=1e-11,
                       delta_fd_floor=2e-5,vega_fd_floor=2e-3,geometric_delta_floor=1e-7,geometric_vega_floor=1e-6,
                       B_memory_absolute=64*1024**2,B_memory_ratio=1.5,B_memory_slack=1024**2,
                       reference_resolution_ratio=.5,reference_resolution_floor=1e-8),
            training=dict(case_ids=[c.case_id for c in train],cases=[dict(case_id=c.case_id,split=c.split,workload_id=c.workload_id,
                model=asdict(c.model),contracts=[asdict(x) for x in c.contracts]) for c in train],threshold_candidates=[.02,.05,.1,"infinity"],
                sizes=[2000,8000],groups=2,repeats=3,pilot_n=1000,selected_threshold=threshold,wall_seconds=perf_counter()-start))
        prechecks=[]
        for case in configuration["B"]["cases"]:
            model,contracts,_=_decode_book(case)
            _geometric_greek_checks(model,contracts,prechecks,case["case_id"],configuration["gates"])
        write_json(output/"pre-freeze-validation.json",prechecks)
        if not all(c["passed"] for c in prechecks):raise ArithmeticError("Deterministic geometric Greek gates failed before freeze")
        configuration["expected_counts"]=expected_counts(configuration);validate_configuration(configuration)
        write_json(output/"configuration.json",configuration)
        write_json(output/"manifest.json",dict(configuration_sha256=canonical_hash(configuration),source_hashes=env["source_hashes"],
                   pilot_directory=os.path.relpath(pilot.resolve(),output.resolve()),pilot_path_relative_to_manifest=True,
                   pilot_hashes={str(f.relative_to(pilot)):file_hash(f) for f in pilot.iterdir() if f.is_file()},
                   training_hashes={f.name:file_hash(f) for f in output.glob("training-*")},training_directory=str(output.resolve())))
        write_json(output/"environment.json",env);write_json(output/"status.json",dict(status="frozen"));return configuration
    except Exception as exc:
        _failure(output,exc,"freeze",0);raise


def load_configuration(path):
    path=Path(path);configuration=json.loads(path.read_text());manifest=json.loads((path.parent/"manifest.json").read_text())
    validate_configuration(configuration)
    if canonical_hash(configuration)!=manifest["configuration_sha256"]:raise ValueError("Configuration hash mismatch")
    if environment_metadata()["source_hashes"]!=manifest["source_hashes"]:raise ValueError("Frozen source hash mismatch")
    pilot_path=Path(manifest["pilot_directory"])
    if manifest.get("pilot_path_relative_to_manifest"):pilot_path=path.parent/pilot_path
    for folder,key in ((path.parent,"training_hashes"),(pilot_path,"pilot_hashes")):
        for name,digest in manifest[key].items():
            if file_hash(folder/name)!=digest:raise ValueError(f"Frozen {key} mismatch: {name}")
    return configuration,manifest


def _check(checks,name,passed,**details):
    checks.append(dict(name=name,passed=bool(passed),**details))


def _wilson(values):
    n=len(values);p=sum(values)/n;z=1.96;den=1+z*z/n;centre=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(fraction=p,lower=centre-half,upper=centre+half,n=n)


def pooled_asian_summary(rows, reference, groups, repeats, sigma=5., floor=1e-10):
    values=np.array([[r["estimate"] for r in rows if r["group"]==g] for g in range(groups)])
    if values.shape!=(groups,repeats):raise ValueError("Incomplete A independent-group/repeat design")
    # Within-group empirical repeat variance + between-pilot group heterogeneity.
    within=float(np.mean(np.var(values,axis=1,ddof=1)))/repeats
    between=max(0.,float(np.var(values.mean(axis=1),ddof=1))-within)
    sampling=math.sqrt((within+between)/groups)
    combined=math.sqrt(sampling**2+reference.standard_error**2)
    error=float(values.mean())-reference.estimate
    return dict(mean=float(values.mean()),pooled_standard_error=sampling,reference=reference.estimate,
                reference_standard_error=reference.standard_error,combined_standard_error=combined,
                difference=error,passed=abs(error)<=sigma*combined+floor,
                observed_deviation_rmse=math.sqrt(float(np.mean((values-reference.estimate)**2))),
                reference_corrected_mse=float(np.mean((values-reference.estimate)**2))-reference.standard_error**2,
                estimator_coverage=_wilson([r["covered_95"] for r in rows]),
                combined_uncertainty_coverage=_wilson([r["combined_uncertainty_covered_95"] for r in rows]),
                mean_production_seconds=float(np.mean([r["production_seconds"] for r in rows])))


def _asian_performance(rows,controls,a,seed):
    """Descriptive group-then-repeat bootstrap, including independent pilot costs."""
    reports=[];rng=np.random.default_rng(seed);groups=a["groups"];repeats=a["repeats"]
    pilot_cost=np.array([c["seconds"] for c in controls])
    for n in a["sizes"]:
        local={m:[[r for r in rows if r["method"]==m and r["n_requested"]==n and r["group"]==g] for g in range(groups)] for m in a["methods"]}
        values={m:np.array([[r["estimate"] for r in rr] for rr in histories]) for m,histories in local.items()}
        timing={m:np.array([[r["production_seconds"] for r in rr] for rr in histories]) for m,histories in local.items()}
        def ratios(gidx,ridx,baseline):
            def take(x):return np.array([x[g,r] for g,r in zip(gidx,ridx)]).reshape(-1)
            cv=take(values["geometric_cv"]);base=take(values[baseline]);den=float(np.var(cv,ddof=1))
            variance=float(np.var(base,ddof=1))/max(den,1e-30)
            basecost=float(take(timing[baseline]).mean());prod=float(take(timing["geometric_cv"]).mean())
            fit=float(pilot_cost[gidx].mean())
            return variance,variance*basecost/max(prod+fit,1e-30),variance*basecost/max(prod+fit/(len(a["sizes"])*repeats),1e-30)
        for baseline in ("iid","antithetic"):
            draws=[]
            for _ in range(300):
                gidx=rng.integers(0,groups,size=groups);ridx=rng.integers(0,repeats,size=(groups,repeats))
                draws.append(ratios(gidx,ridx,baseline))
            observed=ratios(np.arange(groups),np.tile(np.arange(repeats),(groups,1)),baseline)
            bounds=np.quantile(draws,[.025,.975],axis=0)
            reports.append(dict(n_requested=n,baseline=baseline,variance_reduction=observed[0],variance_reduction_lower=bounds[0,0],variance_reduction_upper=bounds[1,0],
                cold_call_efficiency_ratio=observed[1],cold_call_lower=bounds[0,1],cold_call_upper=bounds[1,1],
                reused_pilot_efficiency_ratio=observed[2],reused_pilot_lower=bounds[0,2],reused_pilot_upper=bounds[1,2],
                bootstrap_unit="pilot group then repeat",interpretation="exploratory descriptive; three pilot groups; higher ratio favors CV",bootstrap_repeats=300))
    return reports


def _run_a(cfg, output, rows, checks, costs, memory, checkpoint):
    a=cfg["A"];block=cfg["block_size"];root=cfg["seeds"]["final"];controls=[];refs=[];summaries=[];performance=[]
    for case in a["cases"]:
        cid=case["case_id"];model,contract=_decode_asian(case)
        actual=geometric_put(model,contract);independent=quantlib_geometric_put(model,contract)
        tolerance=cfg["gates"]["geometric_price_relative"]*max(1.,abs(actual))
        _check(checks,f"A:{cid}:geometric-analytic",abs(actual-independent)<=tolerance,difference=actual-independent,tolerance=tolerance)
        runs=[]
        for j in range(a["reference_runs"]):
            seed=quantlib_seed(root,"A",cid,"reference",j)
            result,seconds=_timed(lambda:quantlib_arithmetic_put(model,contract,n_samples=a["reference_pairs"],seed=seed))
            costs["reference_seconds"]+=seconds;runs.append(result);refs.append(dict(case_id=cid,run=j,seconds=seconds,**asdict(result)))
        reference=aggregate_references(runs);case_rows=[]
        for group in range(a["groups"]):
            tune_seed=stable_seed(root,"A",cid,"control",group)
            control,tuning_seconds=_timed(lambda:fit_control(model,contract,pilot_paths=a["pilot_paths"],seed=tune_seed,block_size=block))
            costs["tuning_seconds"]+=tuning_seconds
            coefficient_id=f"A:{cid}:{group}"
            controls.append(dict(case_id=cid,group=group,coefficient_id=coefficient_id,seconds=tuning_seconds,**asdict(control)))
            for n in a["sizes"]:
                for repeat in range(a["repeats"]):
                    order=list(a["methods"]);np.random.default_rng(stable_seed(root,"A",cid,"order",group,n,repeat)).shuffle(order)
                    for method in order:
                        seed=stable_seed(root,"A",cid,"production",group,n,repeat,method)
                        result,seconds=_timed(lambda:price_asian(model,contract,n_paths=n,seed=seed,method=method,
                            control=control if method=="geometric_cv" else None,block_size=block))
                        costs["production_seconds"]+=seconds
                        row=trial_record(suite="A",case_id=cid,split="evaluation",workload_id="asian-price",valuation_index=0,
                            method=method,group=group,repeat=repeat,n_requested=n,independent_samples=result.independent_samples,
                            path_evaluations=result.path_evaluations,payoff_evaluations=result.path_evaluations,
                            fixing_evaluations=result.fixing_evaluations,quantity="price",estimate=result.price,standard_error=result.standard_error,
                            reference=reference.estimate,reference_standard_error=reference.standard_error,reference_kind=reference.reference_kind,
                            covered_95=abs(result.price-reference.estimate)<=1.96*result.standard_error,
                            combined_uncertainty_covered_95=abs(result.price-reference.estimate)<=1.96*math.hypot(result.standard_error,reference.standard_error),
                            seed_key=str(seed),tuning_seed_key=str(tune_seed) if method=="geometric_cv" else None,
                            production_seconds=seconds,tuning_seconds=0.,coefficient_id=coefficient_id if method=="geometric_cv" else None,
                            block_size=block,memory_kind="separate diagnostic",retuned=False)
                        rows.append(row);case_rows.append(row)
        for method in a["methods"]:
            for n in a["sizes"]:
                summary=pooled_asian_summary([r for r in case_rows if r["method"]==method and r["n_requested"]==n],reference,a["groups"],a["repeats"],cfg["gates"]["pooled_sigma"],cfg["gates"]["pooled_floor"])
                summaries.append(dict(case_id=cid,method=method,n_requested=n,**summary))
                _check(checks,f"A:{cid}:{method}:{n}:pooled",summary["passed"],difference=summary["difference"],combined_standard_error=summary["combined_standard_error"])
        cv_errors=[r["standard_error"] for r in case_rows if r["method"]=="geometric_cv" and r["n_requested"]==max(a["sizes"])]
        resolution_limit=max(cfg["gates"]["reference_resolution_floor"],cfg["gates"]["reference_resolution_ratio"]*math.sqrt(float(np.mean(np.square(cv_errors)))))
        _check(checks,f"A:{cid}:reference-resolution",reference.standard_error<=resolution_limit,reference_standard_error=reference.standard_error,limit=resolution_limit)
        perf,elapsed=_timed(lambda:_asian_performance(case_rows,[c for c in controls if c["case_id"]==cid],a,stable_seed(root,"A",cid,"performance-bootstrap")))
        costs["summary_seconds"]+=elapsed;performance.extend(dict(case_id=cid,**p) for p in perf)
        write_json(output/"A-controls.json",controls);write_json(output/"A-reference-runs.json",refs);write_json(output/"A-validation.json",summaries);checkpoint("A",cid)
    model,contract=_decode_asian(a["cases"][3] if len(a["cases"])>3 else a["cases"][0])
    measured=dict(suite="A",n_paths=max(a["sizes"]),block_size=block,seed=51,**_memory(lambda:price_asian(model,contract,n_paths=max(a["sizes"]),seed=51,block_size=block)))
    memory.append(measured);costs["memory_seconds"]+=measured["seconds"]
    _check(checks,"A:tracked-memory-limit",measured["peak_memory_bytes"]<=cfg["limits"]["tracked_memory_bytes"],**measured)
    _check(checks,"A:independent-reference-seeds",len({r["seed"] for r in refs})==len(refs),count=len(refs))
    return dict(groups=summaries,performance=performance,controls=len(controls),reference_runs=len(refs),cost_note="Control fits charged once per case/group; trial tuning_seconds is zero. Cold call cost includes one full fit; reused fit cost amortized over sizes x repeats.")


def _geometric_greek_checks(model, contracts, checks, cid, gates):
    for i,c in enumerate(contracts):
        analytic=geometric_put_greeks(model,c)
        for param,step,field,floor in (("spot",model.spot*1e-5,"delta",gates["geometric_delta_floor"]),
                                       ("volatility",1e-5,"vega",gates["geometric_vega_floor"])):
            fd=(geometric_put(replace(model,**{param:getattr(model,param)+step}),c)-geometric_put(replace(model,**{param:getattr(model,param)-step}),c))/(2*step)
            _check(checks,f"B:{cid}:leg{i}:geometric-{field}",abs(fd-getattr(analytic,field))<=floor,difference=fd-getattr(analytic,field),tolerance=floor)


def _run_b(cfg,output,rows,checks,costs,memory,checkpoint):
    b=cfg["B"];root=cfg["seeds"]["final"];block=cfg["block_size"];allcontrols=[];allfd=[];alltiming=[];allmemory=[];summaries=[]
    for case in b["cases"]:
        cid=case["case_id"];model,contracts,q=_decode_book(case);n=b["n_paths"]
        grid=tuple(sorted({t for c in contracts for t in c.fixing_times}));seed=stable_seed(root,"B",cid,"production");pilot_seed=stable_seed(root,"B",cid,"control")
        fit,tuning=_timed(lambda:fit_portfolio_control(model,contracts,b["pilot_paths"],pilot_seed,block_size=block));costs["tuning_seconds"]+=tuning
        allcontrols.append(dict(case_id=cid,seconds=tuning,**asdict(fit)))
        _geometric_greek_checks(model,contracts,checks,cid,cfg["gates"])
        results={};timings={}
        for method in b["methods"]:
            result,seconds=_timed(lambda:price_portfolio(model,contracts,n,seed,quantities=q,method=method,
                    control=fit if method=="geometric_cv" else None,block_size=block));costs["production_seconds"]+=seconds
            results[method]=result;timings[method]=seconds
            for i,leg in enumerate(result.legs+(result.portfolio,)):
                for metric in ("price","delta","vega"):
                    rows.append(trial_record(suite="B",case_id=f"{cid}:{'portfolio' if i==len(contracts) else 'leg'+str(i)}:{metric}",
                        split="evaluation",workload_id="joint-union-three-metrics",valuation_index=0,method=method,group=0,repeat=0,
                        n_requested=n,independent_samples=result.n_observations,path_evaluations=n,payoff_evaluations=n*len(contracts),
                        fixing_evaluations=n*len(grid),quantity=None if i==len(contracts) else q[i],estimate=getattr(leg,metric),
                        standard_error=getattr(leg,metric+"_se"),seed_key=str(seed),tuning_seed_key=str(pilot_seed) if method=="geometric_cv" else None,
                        production_seconds=seconds if i==0 and metric=="price" else 0.,tuning_seconds=0.,block_size=block,retuned=False,
                        coefficient_id=f"B:{cid}" if method=="geometric_cv" else None))
            discrepancies=[]
            for i,c in enumerate(contracts):
                sliced=PortfolioControlFit(fit.betas[i:i+1],fit.pilot_paths,fit.pilot_seed,model,(c,),grid,block)
                single,elapsed=_timed(lambda:price_portfolio(model,(c,),n,seed,method=method,control=sliced if method=="geometric_cv" else None,
                                                            block_size=block,simulation_times=grid))
                costs["validation_seconds"]+=elapsed
                discrepancies.extend(abs(getattr(single.legs[0],field)-getattr(result.legs[i],field)) for field in ("price","delta","vega","price_se","delta_se","vega_se"))
            _check(checks,f"B:{cid}:{method}:same-union-singleton",max(discrepancies)<=cfg["gates"]["singleton_absolute"],max_absolute_difference=max(discrepancies))
        fd,elapsed=_timed(lambda:fd_convergence(model,contracts,n,seed,quantities=q,block_size=block,spot_bumps=b["bumps"],volatility_bumps=b["bumps"]))
        costs["validation_seconds"]+=elapsed;allfd.append(dict(case_id=cid,rows=fd))
        for row in fd:
            if row["bump"] not in b["bumps"][-2:]:continue
            floor=cfg["gates"]["delta_fd_floor" if row["parameter"]=="spot" else "vega_fd_floor"]
            for i,entry in enumerate(row["legs"]+[row["portfolio"]]):
                _check(checks,f"B:{cid}:{row['parameter']}:{row['bump']}:entry{i}:paired-FD",abs(entry["difference"])<=5*entry["difference_se"]+floor,**entry)
        def batch():
            return price_portfolio(model,contracts,n,seed,quantities=q,method="geometric_cv",control=fit,block_size=block)
        def singles(own=False):
            for i,c in enumerate(contracts):
                sliced=PortfolioControlFit(fit.betas[i:i+1],fit.pilot_paths,fit.pilot_seed,model,(c,),grid,block)
                price_portfolio(model,(c,),n,seed,method="iid" if own else "geometric_cv",control=None if own else sliced,
                                block_size=block,simulation_times=None if own else grid)
        for repeat in range(b["timing_repeats"]):
            order=["batch","common_union_singletons","own_grid_singletons_iid"];np.random.default_rng(stable_seed(root,"B",cid,"timing-order",repeat)).shuffle(order)
            for kind in order:
                _,seconds=_timed(batch if kind=="batch" else lambda:singles(kind=="own_grid_singletons_iid"))
                alltiming.append(dict(case_id=cid,repeat=repeat,kind=kind,seconds=seconds,order=order));costs["validation_seconds"]+=seconds
        peaks=[]
        for count in (n,4*n):
            measurement=dict(suite="B",case_id=cid,n_paths=count,block_size=block,seed=seed,**_memory(lambda:price_portfolio(model,contracts,count,seed,quantities=q,method="geometric_cv",control=fit,block_size=block)))
            peaks.append(measurement["peak_memory_bytes"]);allmemory.append(measurement);memory.append(measurement);costs["memory_seconds"]+=measurement["seconds"]
        gates=cfg["gates"]
        _check(checks,f"B:{cid}:bounded-memory",max(peaks)<gates["B_memory_absolute"] and peaks[1]<=gates["B_memory_ratio"]*peaks[0]+gates["B_memory_slack"],small_peak=peaks[0],large_peak=peaks[1])
        total=results["geometric_cv"].portfolio
        summaries.append(dict(case_id=cid,results={m:dict(legs=[asdict(x) for x in r.legs],portfolio=asdict(r.portfolio)) for m,r in results.items()},
            risk=dict(delta_hedge_underlying_units=-total.delta,scenarios=[dict(spot_change=ds,volatility_change=dv,first_order_pnl=total.delta*ds+total.vega*dv) for ds,dv in ((-1.,0.),(1.,0.),(0.,-.01),(0.,.01),(-1.,.01))],label="First-order delta/vega GBM P&L; no VaR claim"),timing=timings))
        write_json(output/"B-controls.json",allcontrols);write_json(output/"B-fd.json",allfd);write_json(output/"B-timings.json",alltiming);write_json(output/"B-memory.json",allmemory);checkpoint("B",cid)
    return dict(cases=summaries,timing=alltiming,cost_note="Only first price leg carries each method call's production_seconds; metric rows are never summed as replicated call costs.")


def _run_c(cfg,output,rows,checks,costs,memory,checkpoint):
    c=cfg["C"];threshold=math.inf if c["threshold"]=="infinity" else c["threshold"]
    config=ReuseConfig(root_seed=cfg["seeds"]["final"],sample_sizes=tuple(c["sizes"]),pilot_n=c["pilot_n"],pilot_groups=c["groups"],repeats=c["repeats"],
        policies=(ReusePolicy("iid"),ReusePolicy("per_valuation"),ReusePolicy("reuse"),ReusePolicy("threshold",threshold)),quad_order=c["quad_order"],reference_order=c["reference_order"])
    summaries=[]
    for serialized in c["cases"]:
        case=_decode_reuse(serialized);result=run_reuse_case(case,config);local=result["trials"];rows.extend(local)
        for row in local:
            for field in ("production_seconds","tuning_seconds","decision_seconds"):costs[field]+=row[field] or 0.
        diag=result["diagnostics"];costs["setup_seconds"]+=diag["setup_seconds"];costs["reference_seconds"]+=sum(r["seconds"] for r in diag["references"])
        write_json(output/f"C-diagnostics-{case.case_id}.json",diag)
        summary,elapsed=_timed(lambda:summarize_reuse(local,target_rmse=c["target_rmse"],horizons=tuple(c["horizons"]),bootstrap_repeats=c["bootstrap_repeats"],inner_bootstrap_repeats=c["inner_bootstrap_repeats"],seed=stable_seed(cfg["root_seed"],"C",case.case_id,"summary")))
        costs["summary_seconds"]+=elapsed;summaries.append(dict(case_id=case.case_id,**summary))
        write_json(output/"C-summary.json",summaries);checkpoint("C",case.case_id)
    # One largest production allocation pass, independent of campaign timing.
    from .estimators import estimate
    from .reuse import tune_proposal
    case=_decode_reuse(c["cases"][0]);fit=tune_proposal(case.model,case.contracts[0],9191,c["pilot_n"])
    measurement=dict(suite="C",n_requested=max(c["sizes"]),seed=9192,tuning_seed=9191,**_memory(lambda:estimate("IS",case.model,case.contracts[0],max(c["sizes"]),np.random.default_rng(9192),proposal=fit.proposal)))
    memory.append(measurement);costs["memory_seconds"]+=measurement["seconds"]
    _check(checks,"C:tracked-memory-limit",measurement["peak_memory_bytes"]<=cfg["limits"]["tracked_memory_bytes"],**measurement)
    return dict(cases=summaries,threshold=c["threshold"],training_split="train",evaluation_split="heldout")


def run_workbench(configuration, output):
    cfg,manifest=load_configuration(configuration);output=refuse_nonempty_output(output);start=perf_counter()
    rows=[];checks=[];memory=[];completed=[]
    costs={key:0. for key in ("production_seconds","tuning_seconds","decision_seconds","setup_seconds","reference_seconds","validation_seconds","memory_seconds","summary_seconds","report_seconds")}
    write_json(output/"configuration.json",cfg);write_json(output/"manifest.json",manifest);write_json(output/"environment.json",environment_metadata())
    def checkpoint(suite,case_id):
        completed.append(dict(suite=suite,case_id=case_id));write_csv(output/"trials.csv",rows)
        write_json(output/"validation.json",dict(passed=all(c["passed"] for c in checks),checks=checks))
        write_json(output/"costs.json",costs);write_json(output/"status.json",dict(status="running",completed=completed,wall_seconds=perf_counter()-start))
        print(f"Checkpoint {suite}/{case_id}: {len(rows):,} rows, wall {perf_counter()-start:.1f}s, production {costs['production_seconds']:.1f}s",flush=True)
        if costs["production_seconds"]>cfg["limits"]["production_seconds"]:raise RuntimeError("Frozen production runtime limit exceeded")
    try:
        a=_run_a(cfg,output,rows,checks,costs,memory,checkpoint)
        b=_run_b(cfg,output,rows,checks,costs,memory,checkpoint)
        c=_run_c(cfg,output,rows,checks,costs,memory,checkpoint)
        actual={suite:sum(r["suite"]==suite for r in rows) for suite in ("A","B","C")}
        actual.update(A_controls=a["controls"],A_reference_runs=a["reference_runs"])
        _check(checks,"frozen-counts",actual==cfg["expected_counts"],expected=cfg["expected_counts"],actual=actual)
        validation=dict(passed=all(c["passed"] for c in checks),checks=checks)
        costs["wall_seconds_before_report"]=perf_counter()-start
        summary=dict(schema_version="workbench-summary-v1",configuration_sha256=manifest["configuration_sha256"],
            expected_counts=cfg["expected_counts"],actual_counts=actual,validation=validation,A=a,B=b,C=c,costs=costs,
            limitations=["Synthetic laws and uncalibrated GBM; no market accuracy claim.","Three pilot groups: hierarchical performance intervals are exploratory.",
                        "QuantLib arithmetic reference is stochastic; shared reference uncertainty is never divided by repeat count.",
                        "Coverage Wilson intervals describe observed histories, not established calibration.","tracemalloc tracks allocations, not process RSS."])
        write_csv(output/"trials.csv",rows);write_json(output/"validation.json",validation);write_json(output/"memory.json",memory);write_json(output/"summary.json",summary)
        from .workbench_report import render_workbench_report
        _,elapsed=_timed(lambda:render_workbench_report(output));costs["report_seconds"]=elapsed;costs["wall_seconds"]=perf_counter()-start
        write_json(output/"summary.json",summary);write_json(output/"costs.json",costs)
        render_workbench_report(output)
        write_json(output/"status.json",dict(status="complete" if validation["passed"] else "completed_with_failed_gates",completed=completed,wall_seconds=costs["wall_seconds"]))
        return summary
    except Exception as exc:
        write_csv(output/"trials.csv",rows);write_json(output/"validation.json",dict(passed=False,checks=checks));write_json(output/"costs.json",costs)
        _failure(output,exc,"campaign",completed);raise


def portfolio_demo(n_paths=16384,pilot_paths=4096,seed=81012,block_size=4096):
    model,contracts,q=_decode_book(portfolio_cases()[0]);fit=fit_portfolio_control(model,contracts,pilot_paths,seed+1,block_size=block_size)
    results={method:price_portfolio(model,contracts,n_paths,seed,quantities=q,method=method,control=fit if method=="geometric_cv" else None,block_size=block_size) for method in METHODS}
    print("leg K days qty price(SE) delta(SE) vega(SE)")
    for i,(c,quantity,leg) in enumerate(zip(contracts,q,results["geometric_cv"].legs)):
        print(f"{i} {c.strike:g} {round(c.maturity*365)} {quantity:g} {leg.price:.5f}({leg.price_se:.5f}) {leg.delta:.5f}({leg.delta_se:.5f}) {leg.vega:.5f}({leg.vega_se:.5f})")
    for method,result in results.items():
        p=result.portfolio;print(f"{method}: price {p.price:.5f} (SE {p.price_se:.5f}), delta {p.delta:.5f} (SE {p.delta_se:.5f}), vega {p.vega:.5f} (SE {p.vega_se:.5f})")
    total=results["geometric_cv"].portfolio;print(f"Underlying delta hedge: {-total.delta:.5f} units")
    print("First-order GBM P&L; vega per absolute volatility unit; no VaR claim")
    for ds,dv in ((-1.,0.),(1.,0.),(0.,-.01),(0.,.01),(-1.,.01)):
        print(f"spot change {ds:+g}, volatility change {dv:+g}: P&L {total.delta*ds+total.vega*dv:+.5f}")
    return results
