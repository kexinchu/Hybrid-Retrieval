import argparse
import unittest

from experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner import (
    Config,
    bootstrap_mean_ci,
    consolidate_final,
    configs_for_mode,
    parse_targets,
    percentile,
    select_row,
)


class TargetRecallRunnerTests(unittest.TestCase):
    def test_parse_targets_sorts_and_deduplicates(self):
        self.assertEqual(parse_targets("0.99, 0.90,0.99,1"), [0.90, 0.99, 1.0])

        with self.assertRaises(argparse.ArgumentTypeError):
            parse_targets("0,0.95")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_targets("0.95,1.01")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_targets(" , ")

    def test_percentile_uses_sorted_floor_index_and_empty_is_zero(self):
        self.assertEqual(percentile([], 0.95), 0.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.0), 1.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.5), 2.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 1.0), 3.0)

    def test_bootstrap_mean_ci_is_reproducible_for_a_seed(self):
        values = [10.0, 20.0, 40.0, 80.0]
        first = bootstrap_mean_ci(values, samples=250, seed=1234)
        second = bootstrap_mean_ci(values, samples=250, seed=1234)

        self.assertEqual(first, second)
        self.assertEqual(bootstrap_mean_ci([], samples=250, seed=1234), (0.0, 0.0))
        self.assertEqual(bootstrap_mean_ci([7.0], samples=250, seed=1234), (7.0, 7.0))

    def test_select_row_meeting_target_chooses_lowest_latency(self):
        rows = [
            {"config": "slow", "ok": 2, "errors": 0, "recall_mean": 0.99, "latency_mean_ms": 30.0},
            {"config": "fast", "ok": 2, "errors": 0, "recall_mean": 0.95, "latency_mean_ms": 10.0},
            {"config": "failed", "ok": 0, "errors": 0, "recall_mean": 1.0, "latency_mean_ms": 1.0},
            {"config": "errored", "ok": 2, "errors": 1, "recall_mean": 1.0, "latency_mean_ms": 1.0},
        ]

        selected, met = select_row(rows, target=0.95)

        self.assertTrue(met)
        self.assertEqual(selected["config"], "fast")

    def test_select_row_falls_back_to_highest_recall_then_lowest_latency(self):
        rows = [
            {"config": "best-recall-slow", "ok": 2, "errors": 0, "recall_mean": 0.90, "latency_mean_ms": 50.0},
            {"config": "best-recall-fast", "ok": 2, "errors": 0, "recall_mean": 0.90, "latency_mean_ms": 20.0},
            {"config": "faster-lower-recall", "ok": 2, "errors": 0, "recall_mean": 0.85, "latency_mean_ms": 1.0},
        ]

        selected, met = select_row(rows, target=0.95)

        self.assertFalse(met)
        self.assertEqual(selected["config"], "best-recall-fast")

    def test_configs_for_mode_deduplicates_stock_but_not_sqlens(self):
        configs = [
            Config(100, 1000, 8.0, "strict_order", 100),
            Config(100, 1000, 8.0, "strict_order", 200),
            Config(200, 1000, 8.0, "strict_order", 100),
        ]

        stock = configs_for_mode(configs, "original")
        sqlens = configs_for_mode(configs, "design1_bloom")

        self.assertEqual(len(stock), 2)
        self.assertEqual([config.guided_collect_target for config in stock], [100, 100])
        self.assertEqual(sqlens, configs)

    def test_consolidate_final_calculates_speedup_vs_stock(self):
        selected = [
            {
                "target_recall": 0.95,
                "target_met_in_calibration": True,
                "filter_name": "popular_ge1000",
                "mode": "original",
                "config": "stock-config",
                "ef_search": 100,
                "guided_collect_target": 100,
                "max_scan_tuples": 1000,
                "scan_mem_multiplier": 8.0,
                "iterative_scan": "strict_order",
            },
            {
                "target_recall": 0.95,
                "target_met_in_calibration": True,
                "filter_name": "popular_ge1000",
                "mode": "design1_bloom",
                "config": "sqlens-config",
                "ef_search": 100,
                "guided_collect_target": 100,
                "max_scan_tuples": 1000,
                "scan_mem_multiplier": 8.0,
                "iterative_scan": "strict_order",
            },
        ]
        final_results = {
            ("popular_ge1000", "original", "stock-config"): {
                "recall_mean": 0.96,
                "latency_mean_ms": 100.0,
            },
            ("popular_ge1000", "design1_bloom", "sqlens-config"): {
                "recall_mean": 0.94,
                "latency_mean_ms": 40.0,
            },
        }

        rows = consolidate_final(selected, final_results)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["speedup_vs_stock"], 1.0)
        self.assertEqual(rows[1]["speedup_vs_stock"], 2.5)
        self.assertTrue(rows[0]["target_met_in_final"])
        self.assertFalse(rows[1]["target_met_in_final"])


if __name__ == "__main__":
    unittest.main()
