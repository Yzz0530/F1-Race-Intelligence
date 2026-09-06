"""Tests for P2 features (TyreLifePhase, StopNumber, TrackEvoProxy) and fallback encodings."""
import os
import sys
import numpy as np
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strategy_optimizer import F1StrategyOptimizer
from race_physics import TYRE_DEG_RATE, COMPOUND_DELTA


class TestP2Features(unittest.TestCase):
    """Test that P2 features are correctly computed in the feature matrix."""

    @classmethod
    def setUpClass(cls):
        cls.opt = F1StrategyOptimizer()

    def test_tyre_life_phase_soft(self):
        """TyreLifePhase should be 0/1/2 based on lap and compound."""
        M = self.opt._build_feature_matrix("VER", "British Grand Prix", [("SOFT", 25)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        phase = M[:, cid["TyreLifePhase"]]
        # SOFT: phase 0 for laps 1-12, phase 1 for 13-18, phase 2 for 19-25
        self.assertEqual(phase[0], 0.0)   # lap 1
        self.assertEqual(phase[11], 0.0)  # lap 12
        self.assertEqual(phase[12], 1.0)  # lap 13
        self.assertEqual(phase[17], 1.0)  # lap 18
        self.assertEqual(phase[18], 2.0)  # lap 19
        self.assertEqual(phase[24], 2.0)  # lap 25

    def test_tyre_life_phase_medium(self):
        """TyreLifePhase for MEDIUM compound."""
        M = self.opt._build_feature_matrix("VER", "British Grand Prix", [("MEDIUM", 35)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        phase = M[:, cid["TyreLifePhase"]]
        # MEDIUM: phase 0 for laps 1-20, phase 1 for 21-30, phase 2 for 31-35
        self.assertEqual(phase[0], 0.0)   # lap 1
        self.assertEqual(phase[19], 0.0)  # lap 20
        self.assertEqual(phase[20], 1.0)  # lap 21
        self.assertEqual(phase[29], 1.0)  # lap 30
        self.assertEqual(phase[30], 2.0)  # lap 31
        self.assertEqual(phase[34], 2.0)  # lap 35

    def test_tyre_life_phase_hard(self):
        """TyreLifePhase for HARD compound."""
        M = self.opt._build_feature_matrix("VER", "British Grand Prix", [("HARD", 45)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        phase = M[:, cid["TyreLifePhase"]]
        # HARD: phase 0 for laps 1-28, phase 1 for 29-40, phase 2 for 41-45
        self.assertEqual(phase[0], 0.0)   # lap 1
        self.assertEqual(phase[27], 0.0)  # lap 28
        self.assertEqual(phase[28], 1.0)  # lap 29
        self.assertEqual(phase[39], 1.0)  # lap 40
        self.assertEqual(phase[40], 2.0)  # lap 41
        self.assertEqual(phase[44], 2.0)  # lap 45

    def test_stop_number_increases_per_stint(self):
        """StopNumber should increment with each new stint."""
        M = self.opt._build_feature_matrix("VER", "British Grand Prix", [
            ("SOFT", 18),
            ("HARD", 18),
            ("MEDIUM", 16),
        ])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        stops = M[:, cid["StopNumber"]]
        # First 18 laps: stop 0, next 18: stop 1, last 16: stop 2
        self.assertTrue(np.all(stops[:18] == 0.0))
        self.assertTrue(np.all(stops[18:36] == 1.0))
        self.assertTrue(np.all(stops[36:] == 2.0))

    def test_track_evo_proxy_increases(self):
        """TrackEvoProxy should increase monotonically with lap number."""
        M = self.opt._build_feature_matrix("VER", "British Grand Prix", [("SOFT", 10)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        evo = M[:, cid["TrackEvoProxy"]]
        # Should be lap_number / 1000
        expected = np.arange(1, 11, dtype=np.float32) / 1000.0
        np.testing.assert_array_almost_equal(evo, expected)

    def test_p2_features_in_feature_list(self):
        """P2 features must be present in the model's feature list."""
        self.assertIn("TyreLifePhase", self.opt.feature_list)
        self.assertIn("StopNumber", self.opt.feature_list)
        self.assertIn("TrackEvoProxy", self.opt.feature_list)

    def test_p2_features_not_zero_in_matrix(self):
        """P2 features should have non-zero values in a realistic strategy."""
        M = self.opt._build_feature_matrix("VER", "British Grand Prix", [
            ("SOFT", 20),
            ("HARD", 32),
        ])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        # At least some laps should have non-zero TyreLifePhase
        self.assertGreater(M[:, cid["TyreLifePhase"]].sum(), 0)
        # StopNumber should have both 0 and 1 values
        self.assertIn(0.0, M[:, cid["StopNumber"]])
        self.assertIn(1.0, M[:, cid["StopNumber"]])
        # TrackEvoProxy should vary
        evo_values = M[:, cid["TrackEvoProxy"]]
        self.assertGreater(evo_values.max() - evo_values.min(), 0.001)


class TestFallbackEncodings(unittest.TestCase):
    """Test fallback encoding for unknown drivers and compounds."""

    @classmethod
    def setUpClass(cls):
        cls.opt = F1StrategyOptimizer()

    def test_unknown_driver_gets_incremental_id(self):
        """Unknown drivers should get incremental encoding IDs."""
        M1 = self.opt._build_feature_matrix("NONEXISTENT_DRIVER", "British Grand Prix", [("SOFT", 5)])
        M2 = self.opt._build_feature_matrix("ANOTHER_DRIVER", "British Grand Prix", [("SOFT", 5)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        # Both should get non-zero encoding (different from known drivers)
        enc1 = M1[0, cid["Driver_enc"]]
        enc2 = M2[0, cid["Driver_enc"]]
        self.assertNotEqual(enc1, enc2, "Different unknown drivers should get different encodings")
        self.assertGreater(enc1, max(self.opt._encoded_drivers.values()),
                          "Unknown driver encoding should exceed known range")

    def test_unknown_driver_consistent_encoding(self):
        """Same unknown driver should get consistent encoding across calls."""
        M1 = self.opt._build_feature_matrix("UNKNOWN_DRIVER_X", "British Grand Prix", [("SOFT", 5)])
        M2 = self.opt._build_feature_matrix("UNKNOWN_DRIVER_X", "British Grand Prix", [("HARD", 5)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        enc1 = M1[0, cid["Driver_enc"]]
        enc2 = M2[0, cid["Driver_enc"]]
        self.assertEqual(enc1, enc2, "Same unknown driver should get consistent encoding")

    def test_wet_compound_maps_to_intermediate(self):
        """WET compound should map to INTERMEDIATE fallback."""
        M = self.opt._build_feature_matrix("VER", "British Grand Prix", [("WET", 10)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        # IsWet should be 1 because WET is treated as wet
        is_wet = M[0, cid["IsWet"]]
        self.assertEqual(is_wet, 1.0, "WET compound should flag IsWet=1")

    def test_wet_compound_family_is_wet(self):
        """WET compound should have CompoundFamily_enc = WET encoding."""
        M = self.opt._build_feature_matrix("VER", "British Grand Prix", [("WET", 10)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        fam = M[0, cid["CompoundFamily_enc"]]
        wet_enc = int(self.opt.le_family.transform(["WET"])[0])
        self.assertEqual(fam, float(wet_enc), "WET compound should use WET family encoding")

    def test_simulate_with_unknown_driver(self):
        """Simulation should not fail with unknown driver."""
        result = self.opt.simulate_strategy(
            "British Grand Prix", 52, "UNKNOWN_DRIVER_123",
            [("SOFT", 26), ("HARD", 26)]
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["driver"], "UNKNOWN_DRIVER_123")
        self.assertAlmostEqual(result["total_laps"], 52)

    def test_simulate_with_wet_compound(self):
        """Simulation should handle WET compound correctly."""
        result = self.opt.simulate_strategy(
            "British Grand Prix", 52, "VER",
            [("WET", 52)]
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["total_laps"], 52)
        # WET should be slower than dry due to physics delta
        dry_result = self.opt.simulate_strategy(
            "British Grand Prix", 52, "VER",
            [("MEDIUM", 52)]
        )
        self.assertGreater(result["total_time"], dry_result["total_time"],
                          "WET should be slower than MEDIUM")


class TestCalibratedConstants(unittest.TestCase):
    """Test that calibrated constants match expected values."""

    def test_tyre_deg_rate_soft(self):
        """SOFT degradation rate should be ~0.036 s/lap."""
        self.assertAlmostEqual(TYRE_DEG_RATE["SOFT"], 0.036, places=3)

    def test_tyre_deg_rate_medium(self):
        """MEDIUM degradation rate should be ~0.011 s/lap."""
        self.assertAlmostEqual(TYRE_DEG_RATE["MEDIUM"], 0.011, places=3)

    def test_tyre_deg_rate_hard_negative(self):
        """HARD degradation rate should be negative (gains pace over time)."""
        self.assertLess(TYRE_DEG_RATE["HARD"], 0, "HARD should have negative degradation")
        self.assertAlmostEqual(TYRE_DEG_RATE["HARD"], -0.030, places=3)

    def test_compound_delta_soft_vs_medium(self):
        """SOFT should be faster than MEDIUM (negative delta)."""
        self.assertLess(COMPOUND_DELTA["SOFT"], COMPOUND_DELTA["MEDIUM"])
        self.assertAlmostEqual(COMPOUND_DELTA["SOFT"], -3.0, places=1)

    def test_compound_delta_hard_vs_medium(self):
        """HARD should be slower than MEDIUM (positive delta)."""
        self.assertGreater(COMPOUND_DELTA["HARD"], COMPOUND_DELTA["MEDIUM"])
        self.assertAlmostEqual(COMPOUND_DELTA["HARD"], 1.8, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
