import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from mcpricing import workbench as w
from mcpricing.workbench_common import environment_metadata,write_json,trial_record
from mcpricing.references import ReferenceEstimate


def tiny_configuration():
    return dict(schema_version="workbench-config-v1",root_seed=123,seeds=dict(train=11,final=12),block_size=32,
        limits=dict(production_seconds=900,tracked_memory_bytes=512*1024**2),
        A=dict(cases=[w.asian_cases()[-1]],sizes=[32],methods=list(w.METHODS),groups=2,repeats=2,pilot_paths=32,reference_runs=2,reference_pairs=32),
        B=dict(cases=[w.portfolio_cases()[-1]],methods=list(w.METHODS),n_paths=64,pilot_paths=32,timing_repeats=1,bumps=[.01,.003,.001,.0003,.0001]),
        C=dict(cases=[dict(w.reuse_cases(False)[0],contracts=w.reuse_cases(False)[0]["contracts"][:2])],sizes=[200],groups=2,repeats=2,pilot_n=200,
               threshold="infinity",quad_order=32,reference_order=64,target_rmse=.05,horizons=[1,2],bootstrap_repeats=100,inner_bootstrap_repeats=100),
        gates=dict(geometric_price_relative=1e-10,pooled_sigma=5.,pooled_floor=1e-10,singleton_absolute=1e-11,delta_fd_floor=2e-5,
                   vega_fd_floor=2e-3,geometric_delta_floor=1e-7,geometric_vega_floor=1e-6,B_memory_absolute=64*1024**2,
                   B_memory_ratio=1.5,B_memory_slack=1024**2,reference_resolution_ratio=.5,reference_resolution_floor=1e-8),
        training=dict(case_ids=["train-fixture"]))


def save_configuration(folder,configuration):
    configuration["expected_counts"]=w.expected_counts(configuration)
    write_json(folder/"configuration.json",configuration)
    write_json(folder/"manifest.json",dict(configuration_sha256=w.canonical_hash(configuration),
         source_hashes=environment_metadata()["source_hashes"],pilot_directory=str(folder),pilot_hashes={},training_hashes={}))
    return folder/"configuration.json"


class WorkbenchIntegrationTests(unittest.TestCase):
    def test_quantlib_seed_portable_nonzero_and_distinct(self):
        seeds=[w.quantlib_seed(w.ROOT_SEED,"A",case["case_id"],"reference",j) for case in w.asian_cases() for j in range(4)]
        self.assertEqual(len(set(seeds)),24)
        self.assertTrue(all(0<seed<2**31 for seed in seeds))
    def test_default_design_counts_and_serialized_infinity(self):
        c=tiny_configuration();c["A"].update(cases=w.asian_cases(),sizes=[4096,16384,65536],groups=3,repeats=8,reference_runs=4)
        c["B"].update(cases=w.portfolio_cases());c["C"].update(cases=w.reuse_cases(),sizes=[2000,8000,32000],groups=3,repeats=8)
        self.assertEqual(w.expected_counts(c),dict(A=1296,A_controls=18,A_reference_runs=24,B=108,C=59904))
        self.assertEqual(json.loads(json.dumps(c,allow_nan=False))["C"]["threshold"],"infinity")

    def test_hash_roundtrip_tamper_source_and_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);c=tiny_configuration();path=save_configuration(folder,c)
            self.assertEqual(w.load_configuration(path)[0],json.loads(json.dumps(c)))
            changed=dict(c,block_size=64);write_json(path,changed)
            with self.assertRaisesRegex(ValueError,"hash mismatch"):w.load_configuration(path)
            write_json(path,c)
            with patch.object(w,"environment_metadata",return_value={"source_hashes":{}}):
                with self.assertRaisesRegex(ValueError,"source hash"):w.load_configuration(path)
            c["C"]["cases"][0]["case_id"]="train-fixture"
            with self.assertRaisesRegex(ValueError,"overlap"):w.validate_configuration(c)

    def test_expected_counts_detection_and_runtime_failure_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);path=save_configuration(folder,tiny_configuration());out=folder/"campaign"
            with patch.object(w,"_run_a",side_effect=ArithmeticError("retained failure")):
                with self.assertRaisesRegex(ArithmeticError,"retained failure"):w.run_workbench(path,out)
            self.assertEqual(json.loads((out/"failure.json").read_text())["error_type"],"ArithmeticError")
            self.assertTrue((out/"trials.csv").exists())
            with self.assertRaises(FileExistsError):w.run_workbench(path,out)
            c=json.loads(path.read_text());c["expected_counts"]["A"]+=1
            with self.assertRaisesRegex(ValueError,"Expected counts"):w.validate_configuration(c)

    def test_shared_reference_uncertainty_not_divided_by_repeats(self):
        ref=ReferenceEstimate(10.,.3,.1,4,400,800)
        rows=[trial_record(estimate=10.,group=g,covered_95=True,combined_uncertainty_covered_95=True,production_seconds=.01) for g in range(3) for r in range(8)]
        s=w.pooled_asian_summary(rows,ref,3,8)
        self.assertEqual(s["combined_standard_error"],.3)

    def test_freeze_train_selection_infinity_and_strike_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);pilot=folder/"pilot";pilot.mkdir();output=folder/"frozen"
            write_json(pilot/"pilot.json",dict(chosen_block_size=512,forecast=dict(total_seconds=1.,C_seconds=1.)))
            write_json(pilot/"environment.json",environment_metadata())
            selected=dict(policy=w.ReusePolicy("threshold"),trials=[],evidence=[],diagnostics=[],selection_split="train",objective="fixture")
            with patch.object(w,"select_reuse_threshold",return_value=selected) as select:
                frozen=w.freeze_configuration(pilot,output)
            self.assertEqual(frozen["C"]["threshold"],"infinity")
            self.assertEqual(w.load_configuration(output/"configuration.json")[0],json.loads(json.dumps(frozen)))
            train_cases=select.call_args.args[0]
            self.assertTrue(all(c.split=="train" for c in train_cases))
            strikes={c.strike for case in train_cases for c in case.contracts}
            self.assertFalse(strikes & {c["strike"] for case in frozen["C"]["cases"] for c in case["contracts"]})
            frozen["training"]["cases"][0]["contracts"][0]["strike"]=95.
            with self.assertRaisesRegex(ValueError,"strikes overlap"):w.validate_configuration(frozen)

    @unittest.skipUnless(importlib.util.find_spec("QuantLib"), "Optional QuantLib reference dependency is absent")
    def test_tiny_complete_campaign_costs_and_report_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);path=save_configuration(folder,tiny_configuration());out=folder/"campaign"
            result=w.run_workbench(path,out)
            self.assertEqual(result["actual_counts"],result["expected_counts"])
            self.assertEqual(result["actual_counts"]["B"],18)
            self.assertTrue((out/"report.html").exists())
            self.assertFalse((out/"failure.json").exists())
            # B metrics share one call; only one record carries each method cost.
            import csv
            with (out/"trials.csv").open(newline="") as f:rows=list(csv.DictReader(f))
            self.assertEqual(sum(float(r["production_seconds"] or 0)>0 for r in rows if r["suite"]=="B"),3)
            before=w.file_hash(out/"trials.csv")
            from mcpricing.workbench_report import render_workbench_report
            with patch.object(w,"price_asian",side_effect=AssertionError("report resampled")):
                render_workbench_report(out)
            self.assertEqual(before,w.file_hash(out/"trials.csv"))


if __name__=="__main__":unittest.main()
