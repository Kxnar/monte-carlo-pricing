import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from mcpricing.benchmark import hierarchical_variance_ratio, run


class BenchmarkTests(unittest.TestCase):
    def test_ratio_bootstrap(self):
        rng=np.random.default_rng(71)
        a=rng.normal(size=(5,30))
        low,high=hierarchical_variance_ratio(a,a/2,np.random.default_rng(72),1000)
        self.assertLess(low,4)
        self.assertGreater(high,4)

    def test_end_to_end_results_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'results'
            meta=run(out,names=['mixture'],strikes=[100],sizes=[200],groups=2,repeats=2,pilot_n=200,bootstrap=100)
            self.assertEqual(meta['total_estimates'],20)
            for name in ('trials.csv','summary.csv','population.csv','pilots.json','metadata.json','REPORT.md','report.html','mixture-convergence.svg'):
                self.assertTrue((out/name).is_file())
            self.assertEqual(json.loads((out/'metadata.json').read_text())['seed'],20261006)
            with self.assertRaises(ValueError):
                run(out)


if __name__=='__main__':
    unittest.main()
