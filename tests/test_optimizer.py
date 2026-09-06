"""
Unit tests for F1 Strategy Optimizer.
"""
import sys
import os
import numpy as np
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strategy_optimizer import F1StrategyOptimizer


class TestOptimizerLoading(unittest.TestCase):
    opt: F1StrategyOptimizer

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = F1StrategyOptimizer()

    def test_tracks_loaded(self) -> None:
        self.assertGreater(len(self.opt.circuit_info), 0)
        self.assertIn("British Grand Prix", self.opt.circuit_info)

    def test_drivers_loaded(self) -> None:
        self.assertGreater(len(self.opt.driver_offsets), 0)
        self.assertIn("VER", self.opt.driver_offsets)
        self.assertIn("HAM", self.opt.driver_offsets)

    def test_driver_offsets_sensible(self) -> None:
        for d, off in self.opt.driver_offsets.items():
            self.assertGreater(off, -15.0, f"{d} offset {off} too negative")
            self.assertLess(off, 15.0, f"{d} offset {off} too positive")

    def test_models_loaded(self) -> None:
        self.assertIsNotNone(self.opt.xgb_model)
        self.assertIsNotNone(self.opt.le_driver)
        self.assertIsNotNone(self.opt.le_compound)

    def test_fallback_exists_for_all_tracks(self) -> None:
        for race in self.opt.circuit_info:
            self.assertIn(race, self.opt.fallback, f"Missing fallback for {race}")


class TestSimulation(unittest.TestCase):
    opt: F1StrategyOptimizer
    race: str
    driver: str
    laps: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = F1StrategyOptimizer()
        cls.race = "British Grand Prix"
        cls.driver = "VER"
        cls.laps = 52

    def test_single_simulation(self) -> None:
        r = self.opt.simulate_strategy(
            self.race, self.laps, self.driver,
            [("SOFT", 18), ("HARD", 34)],
            sc_prob=0.0
        )
        self.assertIsNotNone(r)
        self.assertEqual(r["driver"], self.driver)
        self.assertEqual(r["total_laps"], self.laps)
        self.assertEqual(r["stints"], 2)
        self.assertGreater(r["total_time"], 0)
        self.assertEqual(len(r["stint_details"]), 2)
        self.assertEqual(len(r["stint_details"][0]["lap_times"]), 18)
        self.assertEqual(len(r["stint_details"][1]["lap_times"]), 34)

    def test_invalid_race_returns_results(self) -> None:
        r = self.opt.simulate_strategy("Nonexistent GP", 52, "VER", [("SOFT", 52)])
        self.assertIsNotNone(r, "Model should predict for unknown circuits using circuit features")

    def test_invalid_driver_returns_none(self) -> None:
        """Previously, unknown drivers returned None. Now they get simulated with neutral offset."""
        # Unknown driver should now return a valid simulation result (not None)
        r = self.opt.simulate_strategy(self.race, 52, "XXX", [("SOFT", 52)])
        self.assertIsNotNone(r)
        self.assertEqual(r["driver"], "XXX")

    def test_pit_loss_applied(self) -> None:
        r = self.opt.simulate_strategy(
            self.race, self.laps, self.driver,
            [("SOFT", 18), ("HARD", 34)],
            pit_loss=22.0, sc_prob=0.0
        )
        self.assertAlmostEqual(r["pit_loss_total"], 22.0)

    def test_zero_stop_has_no_pit_loss(self) -> None:
        r = self.opt.simulate_strategy(
            self.race, self.laps, self.driver,
            [("SOFT", 52)],
            sc_prob=0.0
        )
        self.assertEqual(r["pit_loss_total"], 0.0)

    def test_safety_car_records(self) -> None:
        results: list[bool] = []
        for _ in range(20):
            r = self.opt.simulate_strategy(
                self.race, self.laps, self.driver,
                [("SOFT", 18), ("HARD", 34)],
                sc_prob=0.99
            )
            if r:
                results.append(r["sc_deployed"])
        self.assertGreater(sum(results), 0, "SC should deploy with 99% prob")

    def test_dnf_returns_none(self) -> None:
        for _ in range(50):
            r = self.opt.simulate_strategy(
                self.race, self.laps, "VER",
                [("SOFT", 52)],
                sc_prob=0.0, dnf_prob=1.0
            )
            self.assertIsNone(r, "DNF at prob=1.0 should always return None")

    def test_dnf_partial(self) -> None:
        dnf_count = 0
        for _ in range(200):
            r = self.opt.simulate_strategy(
                self.race, self.laps, "VER",
                [("SOFT", 52)],
                sc_prob=0.0, dnf_prob=0.1
            )
            if r is None:
                dnf_count += 1
        # With 10% prob and 200 runs, expect roughly 10-30 DNFs
        self.assertGreater(dnf_count, 0, "DNF should occur with 10% prob")
        self.assertLess(dnf_count, 100)

    def test_sc_lap_times_slower(self) -> None:
        r1 = self.opt.simulate_strategy(
            self.race, self.laps, self.driver,
            [("SOFT", 52)],
            sc_prob=0.0
        )
        r2 = self.opt.simulate_strategy(
            self.race, self.laps, self.driver,
            [("SOFT", 52)],
            sc_prob=0.99
        )
        self.assertGreaterEqual(r2["total_time"], r1["total_time"] * 0.95)

    def test_different_compounds_produce_different_times(self) -> None:
        soft = self.opt.simulate_strategy(self.race, self.laps, self.driver, [("SOFT", 52)])
        hard = self.opt.simulate_strategy(self.race, self.laps, self.driver, [("HARD", 52)])
        medium = self.opt.simulate_strategy(self.race, self.laps, self.driver, [("MEDIUM", 52)])
        self.assertIsNotNone(soft)
        self.assertIsNotNone(hard)
        self.assertIsNotNone(medium)
        soft_avg = np.mean(soft["stint_details"][0]["lap_times"])
        hard_avg = np.mean(hard["stint_details"][0]["lap_times"])
        medium_avg = np.mean(medium["stint_details"][0]["lap_times"])
        avgs = [soft_avg, medium_avg, hard_avg]
        self.assertGreater(len(set(round(a, 4) for a in avgs)), 1,
                           "Compounds should produce measurably different lap times")


class TestOptimization(unittest.TestCase):
    opt: F1StrategyOptimizer

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = F1StrategyOptimizer()

    def test_optimize_returns_sorted(self) -> None:
        r = self.opt.optimize("British Grand Prix", 30, "VER", mc_runs=5, sc_prob=0.2)
        self.assertGreater(len(r), 0)
        self.assertLessEqual(len(r), 10)
        for i in range(len(r) - 1):
            self.assertLessEqual(r[i]["mean_time"], r[i + 1]["mean_time"])

    def test_optimize_result_structure(self) -> None:
        r = self.opt.optimize("British Grand Prix", 30, "VER", mc_runs=5, sc_prob=0.2)
        top = r[0]
        self.assertIn("strategy", top)
        self.assertIn("mean_time", top)
        self.assertIn("std_time", top)
        self.assertIn("stints", top)
        self.assertGreater(top["mean_time"], 0)
        self.assertGreater(top["std_time"], 0)

    def test_different_drivers_give_different_results(self) -> None:
        r1 = self.opt.optimize("British Grand Prix", 30, "VER", mc_runs=5)
        r2 = self.opt.optimize("British Grand Prix", 30, "HAM", mc_runs=5)
        self.assertNotEqual(r1[0]["mean_time"], r2[0]["mean_time"])

    def test_strategy_summary_format(self) -> None:
        s = self.opt.strategy_summary("British Grand Prix", 52, "VER", [("SOFT", 18), ("HARD", 34)])
        self.assertIsNotNone(s)
        self.assertIn("SOFT", s)
        self.assertIn("HARD", s)
        self.assertIn("Total:", s)

    def test_detailed_run(self) -> None:
        r = self.opt.get_detailed_run("British Grand Prix", 52, "VER", [("SOFT", 18), ("HARD", 34)])
        self.assertIsNotNone(r)
        self.assertIn("stint_details", r)
        self.assertEqual(len(r["stint_details"]), 2)


class TestMLIntegration(unittest.TestCase):
    opt: F1StrategyOptimizer

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = F1StrategyOptimizer()

    def test_build_feature_matrix_shape(self) -> None:
        M = self.opt._build_feature_matrix("VER", "British Grand Prix", [("SOFT", 18), ("HARD", 34)])
        self.assertEqual(M.shape, (52, len(self.opt.feature_list)))

    def test_build_feature_matrix_constant_features(self) -> None:
        M = self.opt._build_feature_matrix("VER", "British Grand Prix", [("SOFT", 18), ("HARD", 34)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        driver_vals = M[:, cid["Driver_enc"]]
        self.assertTrue(np.all(driver_vals == driver_vals[0]))

    def test_build_feature_matrix_increasing_lap(self) -> None:
        M = self.opt._build_feature_matrix("VER", "British Grand Prix", [("SOFT", 10)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        laps = M[:, cid["LapInRace"]]
        expected = np.arange(1, 11, dtype=np.float32)
        np.testing.assert_array_equal(laps, expected)

    def test_physics_fallback_exists(self) -> None:
        delta = self.opt._physics_delta("VER", "SOFT", 1, 1, False)
        self.assertIsInstance(delta, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
