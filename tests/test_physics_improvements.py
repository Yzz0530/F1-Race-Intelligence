"""Tests for the engineering-improvement pass:
- Fuel model gives realistic race-long fuel decay (~2.5-4s).
- INTERMEDIATE wet tyre is supported end-to-end (encoding + ranking).
- optimize() is deterministic when called identically.
- undercut_benefit returns a positive fresh-tyre gain (corrected sign).
"""
import os
import sys
import numpy as np
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strategy_optimizer import F1StrategyOptimizer
from race_physics import undercut_benefit, COMPOUND_DELTA


class TestImprovements(unittest.TestCase):
    opt: F1StrategyOptimizer

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = F1StrategyOptimizer()
        cls.race = "British Grand Prix"
        cls.driver = "VER"
        cls.laps = 52

    def test_fuel_decay_is_realistic(self) -> None:
        """Fuel makes early laps slower than late laps by a realistic margin.

        The optimizer blends the physics fuel overlay at 60% (ML_WEIGHT=0.4).
        Over a 52-lap race the raw physics fuel effect is ~1.8s; MEDIUM
        degradation (~0.011s/lap) partly offsets it in a single-stint sim, so
        we assert on the isolated physics fuel delta, which must be >1.2s.
        """
        # Isolated fuel physics: lap 1 vs lap 52, no tyre degradation, no ML.
        d1 = self.opt._physics_delta(self.driver, "MEDIUM", 1, 1, 52, False)
        d52 = self.opt._physics_delta(self.driver, "MEDIUM", 1, 52, 52, False)
        fuel_decay = d52 - d1  # negative = faster at end of race
        self.assertLess(fuel_decay, -2.5, "Fuel effect should exceed 2.5s/race")
        self.assertGreater(fuel_decay, -4.5, "Fuel effect should not exceed ~4.5s/race")

    def test_intermediate_is_distinct_and_slower(self) -> None:
        """INTERMEDIATE must simulate separately and slower than dry slicks."""
        med = self.opt.simulate_strategy(self.race, self.laps, self.driver,
                                         [("MEDIUM", self.laps)], sc_prob=0.0)
        wet = self.opt.simulate_strategy(self.race, self.laps, self.driver,
                                         [("INTERMEDIATE", self.laps)], sc_prob=0.0)
        self.assertIsNotNone(med)
        self.assertIsNotNone(wet)
        med_avg = float(np.mean(med["stint_details"][0]["lap_times"]))
        wet_avg = float(np.mean(wet["stint_details"][0]["lap_times"]))
        # INTERMEDIATE carries a +1.2s physics penalty vs MEDIUM baseline.
        self.assertGreater(wet_avg - med_avg, 0.5,
                           "INTERMEDIATE should be slower than MEDIUM")

    def test_intermediate_encoding_is_wet(self) -> None:
        """Feature matrix must flag INTERMEDIATE as wet, not hardcode DRY."""
        M = self.opt._build_feature_matrix(self.driver, self.race,
                                           [("INTERMEDIATE", 10)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        is_wet = M[:, cid["IsWet"]]
        fam = M[:, cid["CompoundFamily_enc"]]
        self.assertTrue(np.all(is_wet == 1.0), "IsWet must be 1 for INTERMEDIATE")
        wet_enc = int(self.opt.le_family.transform(["WET"])[0])
        self.assertTrue(np.all(fam == wet_enc),
                         "CompoundFamily_enc must be WET for INTERMEDIATE")

    def test_optimize_is_deterministic(self) -> None:
        """Identical optimizer calls must return identical top-10 strategies."""
        a = self.opt.optimize(self.race, 30, self.driver, mc_runs=5, sc_prob=0.2)
        b = self.opt.optimize(self.race, 30, self.driver, mc_runs=5, sc_prob=0.2)
        self.assertEqual([tuple(s["strategy"]) for s in a],
                         [tuple(s["strategy"]) for s in b])

    def test_undercut_fresh_gain_positive(self) -> None:
        """Fresh tyres should yield a positive time gain vs old tyres."""
        uc = undercut_benefit(pitting_lap=10, total_laps=52,
                              tyre_age_before=15, fresh_compound="SOFT",
                              pit_loss=22.0)
        self.assertGreater(uc["gain_from_fresh"], 0,
                           "Fresh-tyre gain must be positive")
        self.assertGreater(uc["crossover_lap"], 0,
                           "Crossover lap should be found")

    def test_compound_delta_preserved(self) -> None:
        self.assertAlmostEqual(COMPOUND_DELTA["MEDIUM"], 0.0)
        self.assertLess(COMPOUND_DELTA["SOFT"], COMPOUND_DELTA["MEDIUM"])
        self.assertGreater(COMPOUND_DELTA["HARD"], COMPOUND_DELTA["MEDIUM"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
