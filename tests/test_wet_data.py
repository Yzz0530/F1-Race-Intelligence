"""Verify the wet-race ingestion produced usable wet training data and that
the retrained model actually differentiates wet laps from dry.

These tests depend on all_races_master.csv containing INTERMEDIATE laps
(ingested from the 2025 Belgian GP wet race). Run prepare_enhanced_data.py
+ train.py before relying on them.
"""
import os
import sys
import numpy as np
import pandas as pd
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strategy_optimizer import F1StrategyOptimizer
from train import FEATURES  # confirms train.py still builds the same schema


class TestWetData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = F1StrategyOptimizer()
        cls.master = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "data", "all_races_master.csv")
        )

    def test_intermediate_laps_present(self) -> None:
        """Ingestion must keep a meaningful number of wet laps."""
        n_inter = int((self.master["Compound"] == "INTERMEDIATE").sum())
        self.assertGreater(n_inter, 100, "Wet laps were dropped by cleaning")

    def test_le_compound_knows_intermediate(self) -> None:
        """LabelEncoder must include INTERMEDIATE so wet is encodable."""
        self.assertIn("INTERMEDIATE", list(self.opt.le_compound.classes_))

    def test_wet_is_flagged_in_feature_matrix(self) -> None:
        M = self.opt._build_feature_matrix("VER", "British Grand Prix",
                                           [("INTERMEDIATE", 10)])
        cid = {n: i for i, n in enumerate(self.opt.feature_list)}
        self.assertTrue(np.all(M[:, cid["IsWet"]] == 1.0))

    def test_model_predicts_wet_slower_than_dry(self) -> None:
        """ML delta for an INTERMEDIATE lap should be slower than MEDIUM."""
        base = self.opt.race_baselines.get("British Grand Prix", self.opt.overall_baseline)
        for compound in ["MEDIUM", "INTERMEDIATE"]:
            M = self.opt._build_feature_matrix("VER", "British Grand Prix",
                                               [(compound, 20)])
            preds = self.opt.xgb_model.predict(M)
            avg = float(np.mean(preds))
            setattr(self, f"_avg_{compound}", avg)
        self.assertGreater(self._avg_INTERMEDIATE, self._avg_MEDIUM,
                           "Retrained model should rank wet laps slower than dry")


if __name__ == "__main__":
    unittest.main(verbosity=2)
