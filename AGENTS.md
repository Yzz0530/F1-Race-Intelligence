# F1 Race Intelligence — Project Context

## Setup
- **Location:** `C:\Users\tyz20\OneDrive\Desktop\F1\F1-Race-Intelligence`
- **System Python:** `C:\Users\tyz20\AppData\Local\Programs\Python\Python312\python.exe` (NOT Hermes venv)
- **Dashboard:** `src/dashboard.py` — Streamlit, 11 tabs, port 8501
- **Run:** `python -m streamlit run src/dashboard.py --server.port 8501`

## Model & Data
- **Model:** `models/xgb_master.pkl` — XGBoost, 250 trees, max_depth=5, lr=0.0123, 31 features
- **Feature list:** `models/feature_list_master.pkl`
- **Training data:** `data/all_races_master.csv` (~29,813 laps, 2025–2026, 24 races, 24 drivers)
- **Target:** delta from race baseline (not absolute lap time)
- **Current MAE:** 0.88s (validation) — dashboard sidebar claims 0.73s (UNFIXED, per user request)

## Pipeline
```
download_all_races.py  →  prepare_enhanced_data.py  →  train.py  →  dashboard.py / optimizer
(fastf1 ingestion)        (feature engineering)        (XGBoost + Optuna)
```

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
- `tests/test_optimizer.py` — 24 tests, all must pass
- Run: `python -m pytest tests/ -v`

## Dashboard Tabs (11 total)
STRATEGY, DRIVER BATTLE, STINT TELEMETRY, TRACK ANALYSIS, SC SIMULATOR, PIT ANALYSIS, CAR TELEMETRY, RESULTS, STANDINGS, RACE TIMELINE, AI ASSISTANT

## Session Split Recommendation
- **ML & Model** — train.py, retraining, features, MAE work
- **Dashboard UI** — dashboard.py, tabs, styling, layout
- **Strategy Engine** — strategy_optimizer.py, race_physics.py, undercut_analyzer.py
- **Data Pipeline** — download_all_races.py, prepare_enhanced_data.py
- **AI Assistant** — strategy_assistant.py, Q&A routing
- **General** — overview, planning, cross-cutting
