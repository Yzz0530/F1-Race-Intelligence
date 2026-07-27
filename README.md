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

- **XGBoost (absolute lap‑time)**: 0.48s MAE — 27 features, Optuna‑tuned
- **Training data**: 28,000+ laps across 2025–2026 F1 seasons
- **Physics‑ML blend**: hybrid prediction for known and unseen circuits
- **Monte Carlo strategy engine**: configurable runs with SC/DNF modelling

## Pipeline

```
download_all_races.py    → Downloads F1 session data via fastf1
prepare_enhanced_data.py → Cleans data & engineers features (circuit, weather, sector speeds)
train.py                 → Optuna‑tuned XGBoost (absolute lap‑time regression)
strategy_optimizer.py    → Monte Carlo strategy simulation (physics + ML blend)
dashboard.py             → Streamlit UI (11 tabs, interactive controls)
```

## Tech Stack

| Component | Technology |
|---|---|
| **Models** | XGBoost, Optuna, scikit‑learn |
| **Simulation** | Monte Carlo, custom physics engine |
| **UI** | Streamlit 1.59.0, custom CSS (carbon‑fibre theme) |
| **Data** | fastf1, pandas, numpy |
| **CI/CD** | GitHub Actions (auto‑update pipeline) |
| **Deployment** | Streamlit Cloud |

## Automation

Data pipeline runs automatically every Monday at 9am UTC via GitHub Actions:

1. **Download** → Fetches new race data from fastf1
2. **Prepare** → Cleans data & engineers features
3. **Train** → Retrains XGBoost with Optuna hyperparameter optimization
4. **Deploy** → Commits updated data + model, Streamlit Cloud auto‑redeploys

Manual trigger available in GitHub Actions tab.

## Setup

```powershell
pip install -r requirements.txt
```

## Usage

```powershell
# 1. Download race data
python src/download_all_races.py

# 2. Prepare features
python src/prepare_enhanced_data.py

# 3. Train the XGBoost model
python src/train.py

# 4. Launch the dashboard
streamlit run src/dashboard.py
```

## Project Structure

```
src/
├── dashboard.py             # Streamlit UI (11 tabs)
├── style.css                # Custom F1‑themed CSS
├── train.py                 # Optuna + XGBoost training pipeline
├── download_all_races.py    # fastf1 data ingestion
├── prepare_enhanced_data.py # Feature engineering
├── strategy_optimizer.py    # Monte Carlo strategy simulation
├── race_physics.py          # Physics engine (fuel, tyre wear, lap modelling)
├── undercut_analyzer.py     # Undercut / overcut analysis
├── telemetry_loader.py      # Fast‑lap & stint telemetry
├── strategy_assistant.py    # AI Assistant (natural‑language Q&A)
├── race_timeline.py         # Stint‑by‑stint race timeline & degradation
├── results.py               # Track session results (FP/Q/Race)
├── standings.py             # Championship standings tables
tests/
└── test_optimizer.py        # 24 unit tests
assets/
└── f1_theme.mp3             # Background theme music
data/
├── all_races_master.csv     # Combined 2025+2026 training data
├── all_races_2026.csv       # 2026 season data
├── all_races_2025.csv       # 2025 season data
└── circuits_metadata.csv    # Circuit characteristics
```

## Disclaimer

This project is an independent portfolio work and is not affiliated with, endorsed by, or associated with Formula 1, FIA, or any of their subsidiaries. All data is sourced from publicly available APIs and is for educational purposes only.

---

© 2026 Tang Yi Zhe. F1 Race Intelligence. All rights reserved.
