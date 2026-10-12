import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mcpricing.workbench_report import render_workbench_report, _asian_costs


class ReportTests(unittest.TestCase):
    def saved_run(self, directory, failed=False):
        values = {
            "configuration.json": {"stage": "evaluation", "label": "<script>alert(1)</script>"},
            "environment.json": {"python": "3.14.6", "source_hashes": {"mcpricing/asian.py": "abc"}},
            "validation.json": {"passed": not failed, "checks": {"math": {"passed": not failed}}},
            "summary.json": {"expected_counts": {"A": 1}, "actual_counts": {"A": 1},
                "A": {"groups": [{"case_id": "<script>alert(1)</script>", "rmse": .2, "reference_standard_error": .01}]},
                "C": {"cases": [{"case_id": "test", "observed_break_even": None, "conclusion": "no observed crossing"}]},
                "costs": {"production_seconds": .2, "tuning_seconds": .1}},
        }
        for name, value in values.items():
            (directory/name).write_text(json.dumps(value), encoding="utf-8")
        with (directory/"A-trials.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("suite", "case_id", "status"))
            writer.writeheader()
            writer.writerow({"suite": "A", "case_id": "test", "status": "failed gate" if failed else "ok"})

    def test_saved_only_regeneration_escaping_and_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.saved_run(directory)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
            with patch("numpy.random.default_rng", side_effect=AssertionError("report must not sample")):
                result = render_workbench_report(directory)
                rendered = (directory/"report.html").read_bytes()
                render_workbench_report(directory)
            self.assertEqual(result["validation_status"], "PASS")
            self.assertEqual(result["raw_counts"], {"A": 1})
            self.assertEqual(rendered, (directory/"report.html").read_bytes())
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.name in before})
            output = rendered.decode()
            self.assertNotIn("<script>", output)
            self.assertIn("&lt;script&gt;", output)
            self.assertIn("Shared reference SE", output)
            self.assertIn("Observed deviation RMSE", output)
            self.assertIn("not a stochastic SE", output)
            self.assertIn("no observed crossing", output)
            self.assertIn("not process RSS", output)

    def test_failed_gate_and_raw_status_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.saved_run(directory, failed=True)
            result = render_workbench_report(directory)
            self.assertEqual(result["validation_status"], "FAIL")
            text = (directory/"report.md").read_text(encoding="utf-8")
            self.assertIn("failed gate", text)
            self.assertIn("validation.checks.math.passed", text)

    def test_count_mismatch_cannot_be_reported_as_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.saved_run(directory)
            summary = json.loads((directory/"summary.json").read_text())
            summary["expected_counts"]["A"] = 2
            (directory/"summary.json").write_text(json.dumps(summary))
            self.assertEqual(render_workbench_report(directory)["validation_status"], "FAIL")

    def test_missing_summary_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                render_workbench_report(tmp)

    def test_full_pilot_and_amortized_cost_use_saved_actual_calls(self):
        controls = [{"case_id": "case", "group": 0, "coefficient_id": "fit", "seconds": .8}]
        calls = {("fit", "100"): [.1, .3], ("fit", "200"): [.3, .5]}
        rows = _asian_costs(controls, calls)
        self.assertEqual(rows[0]["production_calls_sharing_fit"], 4)
        self.assertAlmostEqual(rows[0]["single_call_including_full_pilot_seconds"], 1.)
        self.assertAlmostEqual(rows[0]["amortized_seconds_per_call"], .4)


if __name__ == "__main__":
    unittest.main()
