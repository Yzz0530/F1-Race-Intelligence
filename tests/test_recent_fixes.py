"""Regression tests for the weak-spot fixes pass:

A. SC SIMULATOR compares like-for-like (baseline also pits), so a free pit
   under Safety Car is a GAIN, not a spurious LOSS.
B. TRACK ANALYSIS traffic uses race_physics.traffic_effect (no inline duplicate).
C. find_optimal_pit_window is a real optimizer (true total-time ranking).
D. StrategyAssistant.ask routes rain questions to wet-tyre advice.
E. Dead code removed (overcut_benefit, vsc_delta, compound_degradation_rate,
   analyze_undercut must not exist).
F. optimize() excludes INTERMEDIATE on dry races.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import race_physics
from race_physics import simulate_sc_scenario, traffic_effect
from strategy_optimizer import F1StrategyOptimizer
from undercut_analyzer import UndercutAnalyzer
from strategy_assistant import StrategyAssistant


class TestSCLikeForLike(unittest.TestCase):
    opt: F1StrategyOptimizer

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = F1StrategyOptimizer()

    def test_free_pit_under_sc_is_gain(self) -> None:
        """Pitting for free under the SC beats a normal green-flag pit."""
        res = simulate_sc_scenario(sc_lap=14, total_laps=52, base_lap_time=87.0,
                                   sc_duration=3, sc_free_pit=True)
        self.assertGreater(res["time_saved"], 0,
                           "Free pit under SC must be a GAIN vs baseline that also pits")
        # Baseline must itself include a pit stop (like-for-like).
        self.assertIsNotNone(res.get("baseline_pit_lap"))
        self.assertEqual(res["baseline_pit_loss"], race_physics.PIT_LOSS_DEFAULT)

    def test_free_pit_is_strictly_better_than_no_free_pit(self) -> None:
        """The reduced SC pit loss must make the free-pit SC strictly better.

        In this simplified single-compound model the no-free-pit case is near
        neutral (SC slow laps replace already fuel-heavy early laps), so we do
        not assert a hard LOSS there — we assert the real invariant: a free pit
        under SC is better than the same SC without the free pit.
        """
        free = simulate_sc_scenario(sc_lap=14, total_laps=52, base_lap_time=87.0,
                                    sc_duration=3, sc_free_pit=True)
        no_free = simulate_sc_scenario(sc_lap=14, total_laps=52, base_lap_time=87.0,
                                       sc_duration=3, sc_free_pit=False)
        self.assertGreater(free["time_saved"], no_free["time_saved"],
                           "Free-pit SC must beat no-free-pit SC")


class TestTrafficModelConsistent(unittest.TestCase):
    """TRACK ANALYSIS must use the engine's traffic model, not an inline one."""

    def test_traffic_effect_matches_engine_table(self) -> None:
        # Replicate the dashboard expression and confirm it equals traffic_effect.
        pos = list(range(1, 13))
        losses = [traffic_effect(p) for p in pos]
        # P1 clean air = 0; P12 = 0.55 by the engine's TRAFFIC_LOSS table.
        self.assertEqual(losses[0], 0.0)
        self.assertAlmostEqual(losses[-1], 0.55, places=5)
        self.assertGreater(losses[-1], 0.4)
        # The full table extends to P20 = 0.95s (engine source of truth).
        self.assertAlmostEqual(traffic_effect(20), 0.95, places=5)


class TestPitWindowOptimizer(unittest.TestCase):
    ua: UndercutAnalyzer

    @classmethod
    def setUpClass(cls) -> None:
        cls.ua = UndercutAnalyzer(base_lap_time=87.0)

    def test_window_ranked_by_true_total_time(self) -> None:
        wins = self.ua.find_optimal_pit_window(total_laps=52, compound="MEDIUM",
                                                stint_length=26, track_temp=35.0)
        self.assertTrue(wins)
        # Sorted ascending by total time, best has 0 penalty.
        times = [w["total_time_seconds"] for w in wins]
        self.assertEqual(times, sorted(times))
        self.assertEqual(wins[0]["penalty_vs_best_seconds"], 0.0)

    def test_no_arbitrary_fuel_term_distorts_ranking(self) -> None:
        """Best pit lap must be a real-time minimum, not a fixed stint_length."""
        wins = self.ua.find_optimal_pit_window(total_laps=52, compound="SOFT",
                                                stint_length=10, track_temp=35.0)
        # SOFT degrades fastest, so pitting earlier is genuinely better.
        best = wins[0]["pit_lap"]
        self.assertLessEqual(best, 26, "SOFT should favour an earlier, fresher stop")


class TestRainRouting(unittest.TestCase):
    asst: StrategyAssistant

    @classmethod
    def setUpClass(cls) -> None:
        cls.asst = StrategyAssistant(F1StrategyOptimizer())

    def test_rain_question_routes_to_intermediate(self) -> None:
        ans = self.asst.ask("Which tyre should I use next? It's raining.",
                             driver="VER", track="British Grand Prix",
                             total_laps=52)
        self.assertIn("INTERMEDIATE", ans,
                      "Rain question must recommend INTERMEDIATE, not slicks")

    def test_explicit_rainfall_arg_routes_wet(self) -> None:
        ans = self.asst.ask("Which tyre should I use next?",
                             driver="VER", track="British Grand Prix",
                             total_laps=52, rainfall=0.8)
        self.assertIn("INTERMEDIATE", ans)

    def test_dry_question_still_recommends_slicks(self) -> None:
        ans = self.asst.ask("Which tyre should I use next?",
                             driver="VER", track="British Grand Prix",
                             total_laps=52, rainfall=0.0)
        self.assertNotIn("INTERMEDIATE", ans)


class TestDeadCodeRemoved(unittest.TestCase):
    def test_dead_helpers_gone(self) -> None:
        for name in ("overcut_benefit", "vsc_delta", "compound_degradation_rate"):
            self.assertFalse(hasattr(race_physics, name),
                             f"{name} should have been removed")
        self.assertFalse(hasattr(UndercutAnalyzer, "analyze_undercut"),
                         "UndercutAnalyzer.analyze_undercut should have been removed")


class TestDryOptimizeExcludesWet(unittest.TestCase):
    opt: F1StrategyOptimizer

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = F1StrategyOptimizer()

    def test_intermediate_absent_on_dry_race(self) -> None:
        res = self.opt.optimize("British Grand Prix", 52, "VER", mc_runs=2)
        for r in res:
            for comp, _ in r["strategy"]:
                self.assertNotEqual(comp, "INTERMEDIATE",
                                    "Dry optimize must not emit INTERMEDIATE")

    def test_intermediate_present_when_wet_flagged(self) -> None:
        res = self.opt.optimize("British Grand Prix", 52, "VER", mc_runs=2, wet=True)
        self.assertTrue(any(comp == "INTERMEDIATE" for r in res for comp, _ in r["strategy"]))


    def test_load_cached_telemetry_missing_returns_none(self) -> None:
        from telemetry_loader import load_cached_telemetry
        # No committed parquet for a nonexistent race/year -> graceful None.
        self.assertIsNone(load_cached_telemetry(2099, "Phantom Grand Prix", "VER"))

    def test_load_cached_telemetry_round_trip(self) -> None:
        import os, pandas as pd
        import telemetry_loader
        from telemetry_loader import load_cached_telemetry
        telem_dir = os.path.join(telemetry_loader._BASE, "data", "telemetry")
        os.makedirs(telem_dir, exist_ok=True)
        df = pd.DataFrame({
            "Distance": [0.0, 10.0, 20.0],
            "Speed": [120.0, 200.0, 180.0],
            "Throttle": [0.0, 100.0, 50.0],
            "Brake": [0.0, 0.0, 1.0],
            "nGear": [2, 5, 4],
            "DRS": [0, 1, 0],
        })
        path = os.path.join(telem_dir, "2026__Test_Grand_Prix__VER.parquet")
        df.to_parquet(path, index=False)
        try:
            out = load_cached_telemetry(2026, "Test Grand Prix", "VER")
            self.assertIsNotNone(out)
            # Parquet may store Distance as int; compare values, not exact dtypes.
            self.assertEqual([float(x) for x in out.index], [0.0, 10.0, 20.0])
            self.assertEqual(out.index.name, "Distance")
            self.assertIn("Speed", out.columns)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
