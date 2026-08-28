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
- **Training data:** 31,700+ laps across 2025–2026 F1 seasons (24 races, 24 drivers), including 800+ wet (INTERMEDIATE) laps from the 2025 Belgian GP
- **Physics‑ML blend:** hybrid prediction (ML_WEIGHT=0.4) for known and unseen circuits
- **Monte Carlo strategy engine:** configurable runs with SC/DNF modelling
- **Compound & wet differentiation:** model learns wet/dry separation from data (CompoundFamily_enc 0.43, IsWet 0.35 importance); dry slick spacing reinforced by physics overlay

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

The model predicts **delta from race baseline** (how much faster/slower a lap is vs the race average), not absolute lap time. This forces the model to learn driver/car/race pace rather than memorizing per-driver-per-race means.

**Important — compound/tyre effects are now partly learned from data.** As of the
2025 Belgian GP wet-race ingestion, `all_races_master.csv` contains 800+ INTERMEDIATE
(wet) laps. After retraining, the model assigns real importance to
`CompoundFamily_enc` (0.43) and `IsWet` (0.35), so wet/dry separation is data-driven.
Dry slick compounds (SOFT/MEDIUM/HARD) still rely partly on the physics overlay for
their ±0.35/0/0.20s spacing, but the ML delta now also responds to compound/wet
context. The training set is still mostly dry, so dry-compound separation is weaker
in the learned signal than the physics overlay — both contribute in the 60% physics /
40% ML blend.

**Feature importance (top 10):** the ML model assigns weight to fuel/position/lap-progress/driver-form features. `Compound_enc` and `CompoundOrdinal` appear in the feature list but carry little learned signal on dry data — they are placeholders the physics overlay acts on.

**Known limitation:** because the model targets a per-race delta, laps are predicted relative to that race's mean. On a circuit where the model's baseline is off, every lap shifts together; the *spread* between strategies (the thing the optimizer ranks on) is dominated by the physics overlay and is robust to that offset.

## Known Issues

- **Dry-compound spacing is physics-led:** the model's learned delta still leans on the physics overlay for SOFT/MEDIUM/HARD ±0.35/0/0.20s spacing, since the training set is >95% dry. Wet separation is fully data-driven.
- **FastF1 rate limits:** re-running `prepare_enhanced_data.py` fetches weather live; the FastF1 HTTP cache (`cache/`) is gitignored, so a cold run re-downloads.

## License

Portfolio project — free to use and modify.
