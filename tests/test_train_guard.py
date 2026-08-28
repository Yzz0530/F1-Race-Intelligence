"""Guard test: importing train.py must NOT retrain or overwrite model artifacts.

Background
----------
train.py used to execute its entire Optuna training pipeline (and joblib.dump the
pickles) at *import time*, with no `if __name__ == "__main__"` guard. Because
test_wet_data.py did `from train import FEATURES`, every `pytest` run silently
re-trained the model and overwrote xgb_master.pkl. That is a dangerous, silent
regression: the committed model could change without anyone running `train.py`.

This test locks the fix in: importing `train` must leave every file under models/
byte-for-byte untouched (we compare mtimes) and must still expose FEATURES.
"""

import os
import sys
import unittest

# Make `src` importable without relying on pytest rootdir config
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

MODELS_DIR = os.path.join(_HERE, "..", "models")


def _snapshot_mtimes() -> dict[str, float]:
    snap: dict[str, float] = {}
    for name in os.listdir(MODELS_DIR):
        path = os.path.join(MODELS_DIR, name)
        if os.path.isfile(path):
            snap[name] = os.path.getmtime(path)
    return snap


class TestTrainImportGuard(unittest.TestCase):
    def test_import_train_does_not_write_models(self) -> None:
        """Importing train.py must not modify any committed model artifact."""
        before = _snapshot_mtimes()
        self.assertIn(
            "xgb_master.pkl", before,
            "xgb_master.pkl missing from models/ — test setup is wrong",
        )

        import train  # noqa: F401  (this is the action under test)

        after = _snapshot_mtimes()
        self.assertEqual(
            set(before), set(after),
            "A model file appeared/disappeared during import of train.py",
        )
        changed = [n for n in before if before[n] != after.get(n)]
        self.assertEqual(
            changed, [],
            f"Importing train.py modified model artifacts (side-effect training!): {changed}",
        )

    def test_train_exposes_features_without_training(self) -> None:
        """The legitimate use (`from train import FEATURES`) still works."""
        from train import FEATURES
        self.assertIsInstance(FEATURES, list)
        self.assertGreater(len(FEATURES), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
