"""
Production training pipeline for F1 lap time prediction.
Trains XGBoost with Optuna hyperparameter optimization.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib
import optuna
import logging
import os
import sys
import json
import hashlib
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log: logging.Logger = logging.getLogger(__name__)

BASE: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE, "data")
MODEL_DIR: str = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

log.info("Loading data...")
master_path: str = os.path.join(DATA_DIR, "all_races_master.csv")
if not os.path.exists(master_path):
    raise FileNotFoundError(
        "all_races_master.csv not found in data/. The 'Prepare features' step "
        "(prepare_enhanced_data.py) must run before training and produce it from "
        "the downloaded all_races_*.csv files."
    )
df: pd.DataFrame = pd.read_csv(master_path)
log.info(f"Raw shape: {df.shape}, races: {df['Race'].nunique()}, drivers: {df['Driver'].nunique()}")

# --- Feature engineering ---
log.info("Engineering features...")

# Categorical encodings
le_driver: LabelEncoder = LabelEncoder()
le_compound: LabelEncoder = LabelEncoder()
df["Driver_enc"] = le_driver.fit_transform(df["Driver"])
df["Compound_enc"] = le_compound.fit_transform(df["Compound"].astype(str))

# Compound family (all dry in current data)
compound_map = {"SOFT": "DRY", "MEDIUM": "DRY", "HARD": "DRY", "INTERMEDIATE": "WET"}
df["CompoundFamily"] = df["Compound"].fillna("UNKNOWN").map(compound_map).fillna("UNKNOWN")
le_family: LabelEncoder = LabelEncoder()
df["CompoundFamily_enc"] = le_family.fit_transform(df["CompoundFamily"])

# Compound ordinal
compound_order: dict[str, int] = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4}
df["CompoundOrdinal"] = df["Compound"].fillna("MEDIUM").map(compound_order).fillna(2)
df["IsWet"] = (df["Compound"] == "INTERMEDIATE").astype(int)

# Stint phase (lap within stint, not global lap)
df["LapInStint"] = df.groupby(["Race", "Driver", "Stint"]).cumcount() + 1
stint_len = df.groupby(["Race", "Driver", "Stint"])["LapInStint"].transform("max")
progress = df["LapInStint"] / stint_len.clip(lower=1)
df["StintPhase"] = pd.cut(progress, bins=[0, 0.33, 0.66, 1], labels=[0, 1, 2]).fillna(0).astype(int)

# Target: delta from per-race baseline — forces the model to learn
# compound/tyre effects rather than memorizing per-driver-per-race means.
# With absolute lap time as target, DriverForm (per-driver-per-race mean)
# absorbs nearly all variance and compound features get zero importance.
df["RaceBaseline"] = df.groupby("Race")["LapTime"].transform("mean")
df["Target"] = (df["LapTime"] - df["RaceBaseline"]).astype(np.float32)

# Feature columns (must stay in this order for prediction)
misc: list[str] = ["Position_normalized", "IsPersonalBest_int", "FreshTire_int", "IsStartLap"]
sectors: list[str] = ["S1_speed", "S2_speed", "S3_speed", "AvgSpeed"]
weather: list[str] = ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]
circuit: list[str] = ["CircuitLength_km", "CircuitCorners", "CircuitAvgSpeed", "CircuitType_enc"]
FEATURES: list[str] = [
    "Driver_enc", "Compound_enc", "CompoundFamily_enc", "CompoundOrdinal", "IsWet",
    "TyreLife", "TyreLife_sq", "Stint", "StintPhase", "TrackStatus",
    "LapInRace", "LapInRace_sq", "FuelWeightEffect", "DriverForm",
] + misc + sectors + weather + circuit
N_FEATURES: int = len(FEATURES)

# Weather (and any other) columns are supplementary — they may be absent when the
# weather fetch is rate-limited (common on a fresh CI cache). Backfill missing
# features with neutral zeros instead of crashing on dropna.
_missing_feats = [c for c in FEATURES if c not in df.columns]
if _missing_feats:
    log.warning(f"Missing feature columns (backfilled with 0): {_missing_feats}")
    for c in _missing_feats:
        df[c] = 0
log.info(f"Features ({N_FEATURES}): {FEATURES}")

# Drop NaN
before = len(df)
df = df.dropna(subset=FEATURES + ["Target"])
log.info(f"Rows after dropna: {len(df)} (removed {before - len(df)})")

X: pd.DataFrame = df[FEATURES].astype(np.float32)
y: pd.Series = df["Target"].astype(np.float32)

# --- Train/val split by race ---
races: np.ndarray = df["Race"].unique()
rng: np.random.RandomState = np.random.RandomState(42)
rng.shuffle(races)
split = int(len(races) * 0.8)
train_races, val_races = races[:split], races[split:]
log.info(f"Train races ({len(train_races)}): {list(train_races)}")
log.info(f"Val races ({len(val_races)}): {list(val_races)}")

mask_train = df["Race"].isin(train_races)
X_train, y_train = X[mask_train], y[mask_train]
X_val, y_val = X[~mask_train], y[~mask_train]
log.info(f"Train: {len(X_train)} rows, Val: {len(X_val)} rows")

# --- Optuna hyperparameter search ---
log.info("Starting Optuna hyperparameter search (25 trials)...")

def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42,
    }
    m = xgb.XGBRegressor(**params)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return float(mean_absolute_error(y_val, m.predict(X_val)))

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _write_manifest(val_mae: float, train_mae: float, best_trial_mae: float,
                    best_params: dict, rows: int) -> None:
    """Write models/TRAINING_MANIFEST.json recording exactly what produced the
    committed model. Makes the artifact reproducible, not an opaque pickle."""
    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "git_commit": _git_commit(),
        "training_data": {
            "path": os.path.relpath(master_path, BASE),
            "sha256": _sha256(master_path),
            "rows": int(rows),
            "races": int(df["Race"].nunique()),
            "drivers": int(df["Driver"].nunique()),
        },
        "model": {
            "path": os.path.relpath(os.path.join(MODEL_DIR, "xgb_master.pkl"), BASE),
            "sha256": _sha256(os.path.join(MODEL_DIR, "xgb_master.pkl")),
            "n_estimators": int(best_params.get("n_estimators", 0)),
            "learning_rate": float(best_params.get("learning_rate", 0.0)),
            "max_depth": int(best_params.get("max_depth", 0)),
        },
        "features": FEATURES,
        "metrics": {
            "train_mae": float(train_mae),
            "val_mae": float(val_mae),
            "best_trial_mae": float(best_trial_mae),
        },
        "target": "delta from per-race baseline lap time (s)",
    }
    with open(os.path.join(MODEL_DIR, "TRAINING_MANIFEST.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)
    log.info(f"Wrote provenance manifest: {os.path.join(MODEL_DIR, 'TRAINING_MANIFEST.json')}")


def _establish_baseline_manifest() -> None:
    """Evaluate the ALREADY-COMMITTED model on the SAME 20% held-out race split
    used by main() (the split uses RandomState(42), so it is deterministic) and
    write a baseline manifest. This gives the CI quality gate a like-for-like MAE
    to compare against. Does NOT retrain or overwrite the model."""
    model_path = os.path.join(MODEL_DIR, "xgb_master.pkl")
    if not os.path.exists(model_path):
        log.warning("No committed xgb_master.pkl — skipping baseline manifest.")
        return
    model = joblib.load(model_path)
    # Reuse the deterministic val split computed at module load (X_val, y_val).
    val_pred = model.predict(X_val)
    val_mae = float(mean_absolute_error(y_val, val_pred))
    try:
        n_est = int(model.n_estimators)
    except Exception:
        n_est = 0
    _write_manifest(val_mae, val_mae, val_mae,
                    {"n_estimators": n_est}, len(df))
    log.info(f"Baseline manifest written. Committed model val-split MAE: {val_mae:.4f}s")


def main(force_train: bool = False) -> None:
    """Train the model. If force_train is False (CI quality-gate mode), the model
    is only (re)written when it beats the MAE recorded in the existing manifest."""
    # Establish a baseline reference if none exists (pre-existing model w/o manifest)
    manifest_path = os.path.join(MODEL_DIR, "TRAINING_MANIFEST.json")
    if not os.path.exists(manifest_path):
        _establish_baseline_manifest()

    study = optuna.create_study(direction="minimize", study_name="f1_laptime")
    study.optimize(objective, n_trials=25)
    log.info(f"Best MAE: {study.best_value:.4f}s")
    log.info(f"Best params: {study.best_params}")

    # --- Train final model ---
    log.info("Training final model with best params...")
    best_params = {**study.best_params, "random_state": 42}
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    log.info(f"Validation MAE: {val_mae:.4f}s")

    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    log.info(f"Train MAE: {train_mae:.4f}s")

    # --- Quality gate ------------------------------------------------------
    # Compare against the baseline manifest. Only overwrite the committed model
    # if the new val MAE is genuinely better (Optuna is randomized, so a fresh
    # train can be worse). This prevents the weekly CI job from regressing the
    # deployed model. Pass force_train=True to always overwrite (manual retrain).
    baseline_mae = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as mf:
                baseline_mae = float(json.load(mf)["metrics"]["val_mae"])
        except Exception:
            baseline_mae = None

    if (not force_train) and (baseline_mae is not None) and (val_mae >= baseline_mae - 1e-6):
        log.info(
            f"Quality gate: new val MAE {val_mae:.4f}s not better than baseline "
            f"{baseline_mae:.4f}s — keeping committed model, not overwriting."
        )
        # Still record provenance of the run that was attempted (no model change).
        _write_manifest(val_mae, train_mae, study.best_value, best_params, len(df))
        return

    # --- Save artifacts ---
    log.info("Saving artifacts...")
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_master.pkl"))
    joblib.dump(le_driver, os.path.join(MODEL_DIR, "le_driver_master.pkl"))
    joblib.dump(le_compound, os.path.join(MODEL_DIR, "le_compound_master.pkl"))
    joblib.dump(le_family, os.path.join(MODEL_DIR, "le_family_master.pkl"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "feature_list_master.pkl"))

    # Per-track fallback features (weather, sector speeds, position, circuit)
    fallback_cols = weather + sectors + ["Position_normalized"] + circuit
    fallback = df.groupby("Race")[fallback_cols].mean().to_dict("index")
    joblib.dump(fallback, os.path.join(MODEL_DIR, "fallback_features.pkl"))

    # Per-race DriverForm proxy
    form_proxy = df.groupby(["Race", "Driver"])["LapTime"].mean().to_dict()
    joblib.dump(form_proxy, os.path.join(MODEL_DIR, "driver_form_proxy.pkl"))

    # Race baselines (per-race average lap time) - kept for backward compat
    baselines = df.groupby("Race")["LapTime"].mean().to_dict()
    joblib.dump(baselines, os.path.join(MODEL_DIR, "race_baselines.pkl"))

    # Circuit metadata for new tracks
    circuits_path = os.path.join(DATA_DIR, "circuits_metadata.csv")
    circuits_df = pd.read_csv(circuits_path)
    race_to_circuit = {
        "Bahrain Grand Prix": "Bahrain International Circuit",
        "Saudi Arabian Grand Prix": "Jeddah Corniche Circuit",
        "Australian Grand Prix": "Albert Park Circuit",
        "Azerbaijan Grand Prix": "Baku City Circuit",
        "Barcelona Grand Prix": "Circuit de Barcelona-Catalunya",
        "Monaco Grand Prix": "Circuit de Monaco",
        "Canadian Grand Prix": "Circuit Gilles Villeneuve",
        "British Grand Prix": "Silverstone Circuit",
        "Austrian Grand Prix": "Red Bull Ring",
        "Hungarian Grand Prix": "Hungaroring",
        "Belgian Grand Prix": "Circuit de Spa-Francorchamps",
        "Dutch Grand Prix": "Circuit Zandvoort",
        "Italian Grand Prix": "Monza",
        "Singapore Grand Prix": "Marina Bay Street Circuit",
        "Japanese Grand Prix": "Suzuka International Racing Course",
        "Qatar Grand Prix": "Losail International Circuit",
        "United States Grand Prix": "Circuit of the Americas",
        "Mexico City Grand Prix": "Autodromo Hermanos Rodriguez",
        "São Paulo Grand Prix": "Interlagos",
        "Las Vegas Grand Prix": "Las Vegas Strip Circuit",
        "Abu Dhabi Grand Prix": "Yas Marina Circuit",
        "Miami Grand Prix": "Miami International Autodrome",
        "Emilia Romagna Grand Prix": "Imola",
        "Chinese Grand Prix": "Shanghai International Circuit",
    }
    circuit_info = {}
    for race, circuit_name in race_to_circuit.items():
        row = circuits_df[circuits_df["Circuit"] == circuit_name]
        if not row.empty:
            circuit_info[race] = {
                "Length_km": row["Length_km"].values[0],
                "Corners": row["Corners"].values[0],
                "AvgSpeed": row["AvgSpeed_kmh"].values[0],
                "Type_enc": {"Permanent": 0, "Street": 1, "Street/Permanent": 2}.get(row["Type"].values[0], 0),
            }
    joblib.dump(circuit_info, os.path.join(MODEL_DIR, "circuit_info.pkl"))

    log.info(f"Saved {len(os.listdir(MODEL_DIR))} files to {MODEL_DIR}")

    _write_manifest(val_mae, train_mae, study.best_value, best_params, len(df))

    # --- Feature importance ---
    fi = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    log.info(f"\nTop 10 features:\n{fi.head(10).to_string(index=False)}")

    log.info(f"\n{'='*50}")
    log.info(f"Train MAE:   {train_mae:.4f}s")
    log.info(f"Val MAE:     {val_mae:.4f}s")
    log.info(f"Best trial:  {study.best_value:.4f}s")
    log.info(f"{'='*50}")
    log.info("Training complete.")


if __name__ == "__main__":
    # Usage:
    #   python src/train.py              -> quality-gated train (CI default)
    #   python src/train.py --force-train -> always overwrite (manual retrain)
    #   python src/train.py --baseline    -> seed baseline manifest, no retrain
    force = "--force-train" in sys.argv
    if "--baseline" in sys.argv:
        _establish_baseline_manifest()
    else:
        main(force_train=force)
