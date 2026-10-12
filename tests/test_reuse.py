import math
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch
import numpy as np

from mcpricing.model import Contract, Mixture
from mcpricing.estimators import estimate, tune
from mcpricing.reuse import (ProposalFit, ReuseCase, ReuseConfig, ReusePolicy,
    needs_retune, quadrature_price, run_reuse_case, run_timing_pilot,
    select_reuse_threshold, summarize_reuse, tune_proposal)


class ProposalReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Mixture()
        cls.contract = Contract(strike=95)
        cls.fit = tune_proposal(cls.model, cls.contract, seed=771, pilot_n=200)

    def test_only_original_is_grid_and_identical_selection(self):
        repeat = tune_proposal(self.model, self.contract, seed=771, pilot_n=200)
        original = tune(self.model, self.contract, seed=771, pilot_n=200)
        self.assertEqual(self.fit.proposal, original["proposal"])
        self.assertEqual(self.fit.proposal, repeat.proposal)
        self.assertEqual(self.fit.candidate_rows, repeat.candidate_rows)
        self.assertEqual(len(self.fit.candidate_rows), 12)
        self.assertEqual(self.fit.payoff_evaluations, 2400)
        with self.assertRaises(FrozenInstanceError):
            self.fit.seconds = 0
        with self.assertRaises(TypeError):
            self.fit.candidate_rows[0]["pilot_variance"] = 0

    def test_threshold_boundary_last_tuning_anchor_and_model_guard(self):
        contract = Contract(strike=100)
        exact = abs(math.log(100/95))
        self.assertFalse(needs_retune(ReusePolicy("threshold", exact), self.model,
                                     contract, self.fit, 95))
        self.assertTrue(needs_retune(ReusePolicy("threshold", exact/2), self.model,
                                    contract, self.fit, 95))
        # Comparing to the last call (97) would miss this trigger; anchor stays95.
        self.assertTrue(needs_retune(ReusePolicy("threshold", .04), self.model,
                                    Contract(strike=100), self.fit, 95))
        self.assertFalse(needs_retune(ReusePolicy("reuse"), self.model,
                                     Contract(strike=140), self.fit, 95))
        with self.assertRaises(ValueError):
            needs_retune(ReusePolicy("reuse"), replace(self.model, locations=(.03, -.08)),
                         contract, self.fit, 95)

    def test_reused_importance_weights_agree_with_quadrature_at_new_strikes(self):
        for strike in (80, 105, 130):
            contract = Contract(strike=strike)
            result = estimate("IS", self.model, contract, 100000,
                              np.random.default_rng(strike), proposal=self.fit.proposal)
            reference = contract.reference(self.model)[0]
            self.assertLess(abs(result.price-reference), 6*result.standard_error)
            self.assertAlmostEqual(quadrature_price(self.model, contract), reference, places=7)

    def small_case(self):
        return ReuseCase("test", self.model,
                        tuple(Contract(strike=k) for k in (95, 97, 100, 99)), workload_id="nearby")

    def small_config(self):
        return ReuseConfig(root_seed=901, sample_sizes=(200,), pilot_n=200,
                           pilot_groups=1, repeats=2, policies=(ReusePolicy("iid"),
                           ReusePolicy("per_valuation"), ReusePolicy("reuse"),
                           ReusePolicy("threshold", .04)), quad_order=64, reference_order=128)

    def test_retuning_charges_anchor_updates_and_complete_records(self):
        def fixed_tune(model, contract, seed, pilot_n):
            fit = tune_proposal(model, contract, seed, pilot_n)
            return replace(fit, seconds=.75)
        with patch("mcpricing.reuse.tune_proposal", side_effect=fixed_tune) as tuner:
            result = run_reuse_case(self.small_case(), self.small_config())
        rows = result["trials"]
        self.assertEqual(tuner.call_count, sum(r["retuned"] for r in rows))
        self.assertEqual(len(result["diagnostics"]["tuning"]), tuner.call_count)
        self.assertEqual(tuner.call_count, 14)
        self.assertEqual(len(rows), 40)
        for event in result["diagnostics"]["tuning"]:
            corresponding = [r for r in rows if r["method"] == event["method"]
                and r["group"] == event["group"] and r["repeat"] == event["repeat"]
                and r["n_requested"] == event["n_requested"]
                and r["valuation_index"] == event["valuation_index"]]
            self.assertEqual(len(corresponding), 1)
            self.assertEqual(corresponding[0]["tuning_seconds"], event["seconds"])
        for method, expected in (("IID", 0), ("IS_REUSE", 1),
                                  ("IS_PER_VALUATION", 4), ("IS_THRESHOLD", 2)):
            history = sorted((r for r in rows if r["method"] == method and r["repeat"] == 0),
                             key=lambda r:r["valuation_index"])
            self.assertEqual(sum(r["retuned"] for r in history), expected)
            self.assertEqual(sum(r["tuning_seconds"] for r in history), expected*.75)
            cumulative = 0.
            for row in history:
                cumulative += row["production_seconds"]+row["tuning_seconds"]+row["decision_seconds"]
                self.assertAlmostEqual(row["cumulative_seconds"], cumulative)
                self.assertTrue({"schema_version", "payoff_evaluations", "seed_key",
                    "tuning_seed_key", "reference_kind", "proposal_id", "retuned"}.issubset(row))
                if method != "IID":
                    self.assertNotEqual(row["seed_key"], row["tuning_seed_key"])
        stochastic = [r for r in rows if r["method"] != "QUAD"]
        self.assertEqual(len({r["seed_key"] for r in stochastic}), len(stochastic))
        for row in rows:
            self.assertIsNone(row["path_evaluations"])
            self.assertIsNone(row["fixing_evaluations"])
            if row["method"] == "QUAD":
                self.assertIsNone(row["standard_error"])
                self.assertIsNone(row["covered_95"])
                self.assertIsNone(row["reference_standard_error"])
                self.assertEqual(row["payoff_evaluations"], 0)
        threshold = sorted((r for r in rows if r["method"] == "IS_THRESHOLD" and r["repeat"] == 0),
                           key=lambda r:r["valuation_index"])
        self.assertEqual([r["retuned"] for r in threshold], [True, False, True, False])
        self.assertEqual(threshold[2]["proposal_id"], threshold[3]["proposal_id"])
        self.assertGreaterEqual(result["diagnostics"]["setup_seconds"], 0)

    def test_timing_instrumentation_does_not_change_draws(self):
        original = run_reuse_case(self.small_case(), self.small_config())["trials"]
        with patch("mcpricing.reuse.perf_counter", side_effect=iter(np.arange(1000)*.01)):
            instrumented = run_reuse_case(self.small_case(), self.small_config())["trials"]
        fields = ("method", "repeat", "valuation_index", "seed_key", "estimate", "standard_error", "proposal_id")
        self.assertEqual([tuple(r[f] for f in fields) for r in original],
                         [tuple(r[f] for f in fields) for r in instrumented])

    def test_threshold_selection_train_only_and_infinity_json_safe(self):
        with self.assertRaises(ValueError):
            select_reuse_threshold([self.small_case()], self.small_config(), candidates=(.04,))
        trained = select_reuse_threshold([replace(self.small_case(), split="train")],
                    replace(self.small_config(), repeats=1), candidates=(math.inf,))
        self.assertEqual(trained["evidence"][0]["threshold"], "infinity")
        self.assertEqual(trained["policy"].threshold, math.inf)
        self.assertEqual(trained["evidence"][0]["retunes"], 1)

    def test_timing_pilot_forces_train_and_eight_call_bound(self):
        with patch("mcpricing.reuse.run_reuse_case", return_value={"trials": [], "diagnostics": {}}) as runner:
            run_timing_pilot(replace(self.small_case(), contracts=(self.contract,)*12), self.small_config())
        case, config = runner.call_args.args
        self.assertEqual(case.split, "train")
        self.assertEqual(len(case.contracts), 8)
        self.assertEqual(config.repeats, 2)
        self.assertEqual(config.sample_sizes, (2000, 8000, 32000))

    def synthetic_summary_rows(self, error=.01, iid_cost=2., reuse_cost=1.):
        rows=[]
        for group in range(2):
            for repeat in range(4):
                for method, cost in (("IID", iid_cost), ("IS_REUSE", reuse_cost)):
                    for index in range(4):
                        rows.append(dict(case_id="summary", workload_id="same", split="heldout",
                            group=group, repeat=repeat, method=method, n_requested=200,
                            valuation_index=index, estimate=1+error, reference=1,
                            cumulative_seconds=cost*(index+1)))
        return rows

    def test_summary_only_demonstrates_crossing_with_qualified_iid(self):
        result = summarize_reuse(self.synthetic_summary_rows(), horizons=(1,2,4),
                                bootstrap_repeats=100, inner_bootstrap_repeats=100)
        self.assertEqual(result["observed_break_even"], 1)
        self.assertEqual(result["horizons"][0]["iid_status"], "qualified")
        failed = summarize_reuse(self.synthetic_summary_rows(error=.2), horizons=(4,),
                                bootstrap_repeats=100, inner_bootstrap_repeats=100)
        self.assertIsNone(failed["observed_break_even"])
        self.assertEqual(failed["horizons"][0]["iid_status"], "none")
        self.assertEqual(failed["horizons"][0]["sizes"][0]["status"], "candidate")


if __name__ == "__main__":
    unittest.main()
