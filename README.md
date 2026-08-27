# F1 Race Intelligence

Predicts F1 lap times and optimizes race strategy using **XGBoost + Monte Carlo simulation + Physics Engine**.

Built as a portfolio project targeting motorsport analytics — combining ML, data science, simulation, and interactive visualization.

## Features

| Tab | Description |
|-----|-------------|
| **STRATEGY** | Monte Carlo pit‑strategy simulation (1‑stop / 2‑stop / undercut) with safety‑car & DNF probability |
| **DRIVER BATTLE** | Head‑to‑head driver comparison across quali, race pace, stint length |
| **STINT TELEMETRY** | Stint‑level telemetry: speed, throttle, brake, gear, DRS per lap |
| **TRACK ANALYSIS** | Per‑circuit sector breakdown, lap time distribution, and track characteristics |
| **SC SIMULATOR** | Safety‑Car "what‑if" scenarios — timing and impact analysis |
| **UNDERCUT** | Undercut / overcut analysis with gap modeling and pit‑window optimization |
| **CAR TELEMETRY** | Per‑driver car data visualization (speed, RPM, gear, DRS) |
| **AI ASSISTANT** | Natural‑language strategy Q&A ("Should I pit?", "What's the fastest strategy?") |
| **RACE TIMELINE** | Stint‑by‑stint race timeline, degradation chart, key events & actionable insights |
| **RESULTS** | Track session results — FP1, FP2, FP3, Qualifying, Sprint Qualifying, Sprint Race, Race |
| **STANDINGS** | Driver & Constructor championship standings with team‑color styled tables |

## Performance

- **XGBoost (delta-from-baseline):** 0.88s MAE — 31 features, Optuna‑tuned (25 trials)
- **Training data:** 29,800+ laps across 2025–2026 F1 seasons (24 races, 24 drivers)
- **Physics‑ML blend:** hybrid prediction (ML_WEIGHT=0.4) for known and unseen circuits
- **Monte Carlo strategy engine:** configurable runs with SC/DNF modelling
- **Compound differentiation:** model now ranks SOFT < MEDIUM < HARD correctly (fixed Aug 2026)

## Pipeline

```text
download_all_races.py  →  prepare_enhanced_data.py  →  train.py  →  dashboard.py / optimizer
(fastf1 ingestion)        (feature engineering)        (XGBoost + Optuna)
```

## Usage

### Run the dashboard

```bash
# Requires system Python 3.12 (streamlit not in Hermes venv)
python -m streamlit run src/dashboard.py --server.port 8501
```

Dashboard available at `http://localhost:8501`.

### Retrain the model

```bash
python src/train.py
```

Trains XGBoost with Optuna hyperparameter search (25 trials), saves model + encoders + feature list to `models/`.

### Run tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
F1-Race-Intelligence/
├── src/
│   ├── dashboard.py          # Streamlit dashboard (11 tabs)
│   ├── train.py              # XGBoost training with Optuna
│   ├── strategy_optimizer.py # Monte Carlo strategy simulation
│   ├── strategy_assistant.py # AI strategy Q&A
│   ├── race_physics.py       # Physics engine (fuel, tyres, SC)
│   ├── undercut_analyzer.py  # Undercut/overcut analysis
│   ├── results.py            # Live race results (fastf1)
│   ├── standings.py          # Championship standings
│   ├── telemetry_loader.py   # Car telemetry (fastf1)
│   ├── race_timeline.py      # Race timeline visualization
│   ├── style.css             # Dashboard stylesheet
│   └── __init__.py
├── models/                   # Trained model + artifacts
│   ├── xgb_master.pkl
│   ├── le_driver_master.pkl
│   ├── le_compound_master.pkl
│   ├── le_family_master.pkl
│   ├── feature_list_master.pkl
│   ├── fallback_features.pkl
│   ├── driver_form_proxy.pkl
│   ├── circuit_info.pkl
│   └── race_baselines.pkl
├── data/
│   ├── all_races_master.csv  # Combined training data
│   ├── all_races_2025.csv
│   ├── all_races_2026.csv
│   └── circuits_metadata.csv
├── tests/
│   └── test_optimizer.py     # 24 unit tests
├── AGENTS.md                 # Project context for AI sessions
└── README.md
```

## Model Details

The model predicts **delta from race baseline** (how much faster/slower a lap is vs the race average), not absolute lap time. This forces the model to learn compound/tyre effects rather than memorizing per-driver-per-race means.

**Feature importance (top 10):**
1. FuelWeightEffect
2. Position_normalized
3. LapInRace
4. LapInRace_sq
5. Stint
6. StintPhase
7. CircuitLength_km
8. IsPersonalBest_int
9. CircuitAvgSpeed
10. DriverForm

Compound_enc and CompoundOrdinal now have meaningful importance (~186 and ~170 gain), enabling proper strategy differentiation.

## Known Issues

- **Dashboard sidebar (line 241):** claims "MAE 0.73s" — actual validation MAE is 0.88s. Deferred per user request.
- **README performance section:** previously claimed 0.48s MAE — now updated to 0.88s.

## License

Portfolio project — free to use and modify.
