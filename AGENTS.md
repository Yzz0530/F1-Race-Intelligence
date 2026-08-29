# F1 Race Intelligence — Project Context

## Setup
- **Location:** `C:\Users\tyz20\OneDrive\Desktop\F1\F1-Race-Intelligence`
- **System Python:** `C:\Users\tyz20\AppData\Local\Programs\Python\Python312\python.exe` (NOT Hermes venv)
- **Dashboard:** `src/dashboard.py` — Streamlit, 11 tabs, port 8501
- **Run:** `python -m streamlit run src/dashboard.py --server.port 8501`

## Model & Data
- **Model:** `models/xgb_master.pkl` — XGBoost, **450 trees** (Optuna-tuned), 31 features. See `models/TRAINING_MANIFEST.json` for the authoritative current values (n_estimators, val MAE, training-data sha256, git commit).
- **Feature list:** `models/feature_list_master.pkl`
- **Training data:** `data/all_races_master.csv` (~31,700 laps, 2025–2026, 24 races, 24 drivers) incl. 806 wet (INTERMEDIATE) laps from 2025 Belgian GP
- **Target:** delta from race baseline (not absolute lap time)
- **Current MAE:** **0.8706s** (val-split, per `models/TRAINING_MANIFEST.json`). The sidebar reads this dynamically from the manifest — do NOT hard-code a different number in the UI.
- **Wet handling:** model learns wet/dry from data (CompoundFamily_enc + IsWet top features). Dry slick spacing still reinforced by physics overlay.

## Pipeline
```
download_all_races.py  →  prepare_enhanced_data.py  →  train.py  →  dashboard.py / optimizer
(fastf1 ingestion)        (feature engineering)        (XGBoost + Optuna)
```
- **Cleaning bug fixed:** 107% rule is now per (Race, Compound) + excludes TrackStatus!=1 laps, so wet laps survive instead of being 100% deleted.

## Physics Constants (`race_physics.py`)
- PIT_LOSS: 22s (12s under SC)
- Fuel: 110kg start, 2.5kg/lap, 0.035s/kg
- Compound delta: SOFT -0.35s, MEDIUM 0s, HARD +0.20s
- Degradation: SOFT 0.080s/lap, MEDIUM 0.045s/lap, HARD 0.025s/lap

## Strategy Optimizer (`strategy_optimizer.py`)
- ML_WEIGHT: 0.4 (40% ML, 60% physics blend)
- DriverForm scaled to 0.15× in feature matrix
- Race baselines stored in `self.race_baselines` dict

## Tests
- `tests/test_optimizer.py` — 24 tests (strategy simulation, optimization, ML integration)
- `tests/test_physics_improvements.py` — 6 tests (fuel, wet encoding, determinism, undercut)
- `tests/test_wet_data.py` — 4 tests (wet-lap ingestion + retrained wet prediction)
- `tests/test_train_guard.py` — 2 tests (train.py import does NOT overwrite model artifacts)
- **Total: 36 tests, all must pass.** Run: `python -m pytest tests/ -v`

## Offline / Deployment Behavior (Streamlit Cloud)
The app is deployed via share.streamlit.io (auto-redeploys on push to `master`).
Cold-cache live FastF1 calls are the main cause of slow loads and CPU throttling,
so tabs are built offline-first wherever committed data permits:
- **RESULTS** — track list + per-race session list come from committed
  `data/race_results.csv` (instant, offline). FastF1 used only as a fallback for
  races not yet in the CSV.
- **STANDINGS** — computed from committed `race_results.csv` via
  `_csv_covers_completed()` (offline). No live schedule call on the hot path.
- **STINT TELEMETRY** — already fully offline (simulator: `strategy_optimizer`
  over `all_races_master.csv` + model pickles). No FastF1 involved.
- **CAR TELEMETRY** — **live-only** by design. Per-metre Speed/Throttle/Brake/Gear/
  DRS traces are never committed to the data pipeline. On FastF1 failure it shows a
  friendly error + a *clearly-labeled* offline sector-speed comparison (committed
  `all_races_master.csv` S1/S2/S3/AvgSpeed) — NOT full car telemetry. Persisting
  FastF1 car telemetry to CSV is a known larger pipeline change, deferred (see
  docs/PLAN_TELEMETRY_PIPELINE.md).
- **TRACK ANALYSIS / STRATEGY / SC SIM / DRIVER BATTLE / AI ASSISTANT** — fully
  offline (committed CSV + pickles).
- **Scheduled data refresh ALREADY EXISTS:** `.github/workflows/update-data.yml`
  runs weekly (Mon 09:00 UTC) + manual dispatch, and re-ingests results, retrains,
  and commits. `.github/workflows/ci.yml` runs `pytest` on push/PR. Keep both in
  sync when changing the pipeline.

## Model Provenance (IMPORTANT)
- `train.py` writes `models/TRAINING_MANIFEST.json` on every real train run: data
  sha256 + git commit + MAE + n_estimators + feature list. The committed model is
  therefore a reproducible artifact, not an opaque pickle. Read the manifest
  before trusting `xgb_master.pkl`.

## AI Assistant honesty
- The strategy engine is a **physics + ML blend**. Pace/degradation/wet come from
  the XGBoost model; compound speed deltas (SOFT/MEDIUM/HARD) are a physics
  overlay, NOT a pure ML prediction. The AI tab shows a disclaimer and the assistant
  fallback states this explicitly. Never present overlay numbers as "model output".

## train.py guard (IMPORTANT)
- `train.py` MUST run its Optuna training + `joblib.dump` only under
  `if __name__ == "__main__"`. `test_wet_data.py` does `from train import FEATURES`;
  importing train must NOT retrain or overwrite `xgb_master.pkl`.
- A previous session had no guard: every `pytest` silently re-trained and overwrote
  the model. `tests/test_train_guard.py` locks this in. Never reintroduce import-time
  side effects in train.py.

## Session Split Recommendation
- **ML & Model** — train.py, retraining, features, MAE work
- **Dashboard UI** — dashboard.py, tabs, styling, layout
- **Strategy Engine** — strategy_optimizer.py, race_physics.py, undercut_analyzer.py
- **Data Pipeline** — download_all_races.py, prepare_enhanced_data.py
- **AI Assistant** — strategy_assistant.py, Q&A routing
- **General** — overview, planning, cross-cutting
