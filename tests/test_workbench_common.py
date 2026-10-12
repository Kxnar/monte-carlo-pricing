import json
from pathlib import Path
import tempfile
import unittest
from mcpricing.workbench_common import stable_seed, trial_record, refuse_nonempty_output, write_json, write_csv, TRIAL_FIELDS


class WorkbenchTests(unittest.TestCase):
    def test_semantic_seeds(self):
        self.assertEqual(stable_seed(12, {"b": 2, "a": 1}), stable_seed(12, {"a": 1, "b": 2}))
        self.assertNotEqual(stable_seed(12, "a", "bc"), stable_seed(12, "ab", "c"))
        self.assertGreaterEqual(stable_seed(12, "production", 2), 0)
        self.assertGreater(stable_seed(12, "production", 2), 2**32)
        self.assertLess(stable_seed(12, "production", 2), 2**128)

    def test_record_validation(self):
        record = trial_record(suite="asian", estimate=-.5, path_evaluations=10, independent_samples=5)
        self.assertEqual(tuple(record), TRIAL_FIELDS)
        self.assertIsNone(record["reference"])
        self.assertEqual(record["schema_version"], 1)
        for fields in ({"oops": 1}, {"estimate": float("nan")}, {"production_seconds": -1}, {"path_evaluations": 1.5}, {"covered_95": 1}):
            with self.assertRaises(ValueError):
                trial_record(**fields)

    def test_outputs_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = refuse_nonempty_output(Path(folder)/"run")
            write_json(output/"a.json", {"x": 1})
            write_csv(output/"a.csv", [trial_record(suite="A")])
            self.assertEqual(json.loads((output/"a.json").read_text())["x"], 1)
            self.assertIn("schema_version,suite", (output/"a.csv").read_text())
            with self.assertRaises(FileExistsError):
                refuse_nonempty_output(output)


if __name__ == "__main__":
    unittest.main()
