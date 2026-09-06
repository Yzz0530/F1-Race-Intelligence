"""
F1 data preparation pipeline — loads raw lap data (2025–2026), enriches with
weather, circuit metadata, and engineered features to produce a clean CSV
for model training and strategy optimisation.

Usage:
    python src/prepare_enhanced_data.py [--years 2025 2026]
    # If raw CSVs don't exist for a year, run download_all_races.py <year> first.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Any

import fastf1
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR: str = os.path.join(BASE, "cache")
DATA_DIR: str = os.path.join(BASE, "data")

fastf1.Cache.enable_cache(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Race → circuit name map (hard-coded for correctness across years) ────────
# Updated for 2026 season: Spanish GP moved to Madrid
RACE_TO_CIRCUIT: dict[tuple[int, str], str] = {
    # 2024 season
    (2024, "Bahrain Grand Prix"): "Bahrain International Circuit",
    (2024, "Saudi Arabian Grand Prix"): "Jeddah Corniche Circuit",
    (2024, "Australian Grand Prix"): "Albert Park Circuit",
    (2024, "Japanese Grand Prix"): "Suzuka International Racing Course",
    (2024, "Chinese Grand Prix"): "Shanghai International Circuit",
    (2024, "Miami Grand Prix"): "Miami International Autodrome",
    (2024, "Emilia Romagna Grand Prix"): "Imola",
    (2024, "Monaco Grand Prix"): "Circuit de Monaco",
    (2024, "Canadian Grand Prix"): "Circuit Gilles Villeneuve",
    (2024, "Spanish Grand Prix"): "Circuit de Barcelona-Catalunya",
    (2024, "Austrian Grand Prix"): "Red Bull Ring",
    (2024, "British Grand Prix"): "Silverstone Circuit",
    (2024, "Hungarian Grand Prix"): "Hungaroring",
    (2024, "Belgian Grand Prix"): "Circuit de Spa-Francorchamps",
    (2024, "Dutch Grand Prix"): "Circuit Zandvoort",
    (2024, "Italian Grand Prix"): "Monza",
    (2024, "Azerbaijan Grand Prix"): "Baku City Circuit",
    (2024, "Singapore Grand Prix"): "Marina Bay Street Circuit",
    (2024, "United States Grand Prix"): "Circuit of the Americas",
    (2024, "Mexico City Grand Prix"): "Autodromo Hermanos Rodriguez",
    (2024, "São Paulo Grand Prix"): "Interlagos",
    (2024, "Las Vegas Grand Prix"): "Las Vegas Strip Circuit",
    (2024, "Qatar Grand Prix"): "Losail International Circuit",
    (2024, "Abu Dhabi Grand Prix"): "Yas Marina Circuit",
    # 2025 season
    (2025, "Bahrain Grand Prix"): "Bahrain International Circuit",
    (2025, "Saudi Arabian Grand Prix"): "Jeddah Corniche Circuit",
    (2025, "Australian Grand Prix"): "Albert Park Circuit",
    (2025, "Azerbaijan Grand Prix"): "Baku City Circuit",
    (2025, "Barcelona Grand Prix"): "Circuit de Barcelona-Catalunya",
    (2025, "Monaco Grand Prix"): "Circuit de Monaco",
    (2025, "Canadian Grand Prix"): "Circuit Gilles Villeneuve",
    (2025, "British Grand Prix"): "Silverstone Circuit",
    (2025, "Austrian Grand Prix"): "Red Bull Ring",
    (2025, "Hungarian Grand Prix"): "Hungaroring",
    (2025, "Belgian Grand Prix"): "Circuit de Spa-Francorchamps",
    (2025, "Dutch Grand Prix"): "Circuit Zandvoort",
    (2025, "Italian Grand Prix"): "Monza",
    (2025, "Singapore Grand Prix"): "Marina Bay Street Circuit",
    (2025, "Japanese Grand Prix"): "Suzuka International Racing Course",
    (2025, "Qatar Grand Prix"): "Losail International Circuit",
    (2025, "United States Grand Prix"): "Circuit of the Americas",
    (2025, "Mexico City Grand Prix"): "Autodromo Hermanos Rodriguez",
    (2025, "São Paulo Grand Prix"): "Interlagos",
    (2025, "Las Vegas Grand Prix"): "Las Vegas Strip Circuit",
    (2025, "Abu Dhabi Grand Prix"): "Yas Marina Circuit",
    (2025, "Miami Grand Prix"): "Miami International Autodrome",
    (2025, "Emilia Romagna Grand Prix"): "Imola",
    (2025, "Chinese Grand Prix"): "Shanghai International Circuit",
    # 2026 season (partial — only completed races)
    (2026, "Australian Grand Prix"): "Albert Park Circuit",
    (2026, "Chinese Grand Prix"): "Shanghai International Circuit",
    (2026, "Japanese Grand Prix"): "Suzuka International Racing Course",
    (2026, "Miami Grand Prix"): "Miami International Autodrome",
    (2026, "Canadian Grand Prix"): "Circuit Gilles Villeneuve",
    (2026, "Monaco Grand Prix"): "Circuit de Monaco",
    (2026, "Barcelona Grand Prix"): "Circuit de Barcelona-Catalunya",
    (2026, "Austrian Grand Prix"): "Red Bull Ring",
    (2026, "British Grand Prix"): "Silverstone Circuit",
    (2026, "Belgian Grand Prix"): "Circuit de Spa-Francorchamps",
    (2026, "Hungarian Grand Prix"): "Hungaroring",
    (2026, "Dutch Grand Prix"): "Circuit Zandvoort",
    (2026, "Italian Grand Prix"): "Monza",
    (2026, "Spanish Grand Prix"): "Madrid Street Circuit",
    (2026, "Azerbaijan Grand Prix"): "Baku City Circuit",
    (2026, "Singapore Grand Prix"): "Marina Bay Street Circuit",
    (2026, "United States Grand Prix"): "Circuit of the Americas",
    (2026, "Mexico City Grand Prix"): "Autodromo Hermanos Rodriguez",
    (2026, "São Paulo Grand Prix"): "Interlagos",
    (2026, "Las Vegas Grand Prix"): "Las Vegas Strip Circuit",
    (2026, "Qatar Grand Prix"): "Losail International Circuit",
    (2026, "Abu Dhabi Grand Prix"): "Yas Marina Circuit",
}

# fastf1 short name → location for weather loading (built from schedule below)
_RACE_MAP_CACHE: dict[int, dict[str, str]] = {}


def _get_race_map(year: int) -> dict[str, str]:
    """Build {race_name: location} from the fastf1 schedule for a given year."""
    if year not in _RACE_MAP_CACHE:
        schedule: pd.DataFrame = fastf1.get_event_schedule(year)
        _RACE_MAP_CACHE[year] = {
            r["EventName"]: r["Location"]
            for _, r in schedule.iterrows()
            if r["EventFormat"] != "testing"
        }
    return _RACE_MAP_CACHE[year]


def _load_circuits_metadata(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _parse_lap_timedeltas(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col]).dt.total_seconds()
    if "Time" in df.columns:
        df["Time_sec"] = pd.to_timedelta(df["Time"]).dt.total_seconds()
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to lap data."""
    # Sector speeds (rough km/h from time over ~1 km)
    for s, name in [(1, "S1"), (2, "S2"), (3, "S3")]:
        col = f"Sector{s}Time"
        if col in df.columns:
            df[f"{name}_speed"] = 1000 / df[col].clip(lower=0.001) * 3.6

    # Average speed from traps
    speed_cols = [c for c in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"] if c in df.columns]
    if speed_cols:
        df["AvgSpeed"] = df[speed_cols].mean(axis=1)

    # Normalised position
    df["Position_normalized"] = df["Position"] / df.groupby("Race")["Position"].transform("max")

    # Personal best flag
    df["IsPersonalBest_int"] = df.get("IsPersonalBest", pd.Series(False)).fillna(False).astype(int)

    # Fresh tyre indicator
    if "FreshTyre" in df.columns:
        df["FreshTire_int"] = df["FreshTyre"].fillna(False).astype(int)
    else:
        df["FreshTire_int"] = 0

    # Start lap
    df["IsStartLap"] = (df["TyreLife"] <= 1).astype(int) if "TyreLife" in df.columns else 1

    # ── P2: Non-linear tyre features ──────────────────────────────────
    # Tyre life phase: 0=early linear, 1=mid transition, 2=cliff region
    # Based on calibrated cliff points: SOFT~18, MEDIUM~30, HARD~40
    def _tyre_life_phase(row):
        tl = row.get("TyreLife", 0)
        comp = row.get("Compound", "")
        if comp == "SOFT":
            if tl <= 12: return 0.0
            elif tl <= 18: return 1.0
            else: return 2.0
        elif comp == "MEDIUM":
            if tl <= 20: return 0.0
            elif tl <= 30: return 1.0
            else: return 2.0
        elif comp == "HARD":
            if tl <= 28: return 0.0
            elif tl <= 40: return 1.0
            else: return 2.0
        else:
            return 0.0
    
    df["TyreLifePhase"] = df.apply(_tyre_life_phase, axis=1)

    # Stop number (which pit stop this stint represents)
    # Stint 1 = stop 0 (start), stint 2 = stop 1, stint 3 = stop 2, etc.
    if "Stint" in df.columns:
        df["StopNumber"] = (df["Stint"] - 1).clip(lower=0)

    return df


def _add_circuit_features(df: pd.DataFrame, circuits_df: pd.DataFrame) -> pd.DataFrame:
    """Merge circuit metadata by (year, race_name)."""
    # Normalise circuit names
    circuits_df["Circuit_norm"] = circuits_df["Circuit"].str.strip().str.lower()

    def _lookup(year: int, race: str) -> dict[str, float]:
        circuit_name = RACE_TO_CIRCUIT.get((year, race))
        if circuit_name is None:
            return {"Length_km": np.nan, "Corners": np.nan,
                    "AvgSpeed_kmh": np.nan, "Type": np.nan}
        row = circuits_df[circuits_df["Circuit_norm"] == circuit_name.strip().lower()]
        if row.empty:
            return {"Length_km": np.nan, "Corners": np.nan,
                    "AvgSpeed_kmh": np.nan, "Type": np.nan}
        return {
            "Length_km": row["Length_km"].values[0],
            "Corners": row["Corners"].values[0],
            "AvgSpeed_kmh": row["AvgSpeed_kmh"].values[0],
            "Type": row["Type"].values[0],
        }

    feats = df.apply(lambda r: _lookup(int(r["Year"]), r["Race"]), axis=1, result_type="expand")
    df["CircuitLength_km"] = feats["Length_km"]
    df["CircuitCorners"] = feats["Corners"]
    df["CircuitAvgSpeed"] = feats["AvgSpeed_kmh"]
    df["CircuitType"] = feats["Type"]

    type_map = {"Permanent": 0, "Street": 1, "Street/Permanent": 2}
    df["CircuitType_enc"] = df["CircuitType"].map(type_map).fillna(0).astype(int)
    return df


def _load_weather_for_year(year: int, df: pd.DataFrame) -> pd.DataFrame:
    """Load and merge weather data for all races in `df` (all from same `year`)."""
    # Aliases for normalised race names → fastf1 schedule names (e.g. 2025 Spain)
    RACE_NAME_ALIASES = {"Barcelona Grand Prix": "Spanish Grand Prix"}
    all_races: pd.Index = df["Race"].unique()
    race_map = _get_race_map(year)
    weather_dfs: list[pd.DataFrame] = []

    for i, race in enumerate(all_races):
        short_name = race_map.get(race) or race_map.get(RACE_NAME_ALIASES.get(race, ""))
        if short_name is None:
            print(f"  [{i+1}/{len(all_races)}] {race}: no location in {year} schedule, skipping weather")
            continue
        try:
            session = fastf1.get_session(year, short_name, "R")
            session.load(laps=False, telemetry=False, weather=True)
            wd = session.weather_data.copy()
            wd["Race"] = race
            wd["Year"] = year
            weather_dfs.append(wd)
            print(f"  [{i+1}/{len(all_races)}] {race} ({year}): {len(wd)} weather rows")
        except Exception as e:
            print(f"  [{i+1}/{len(all_races)}] {race} ({year}): weather FAILED - {e}")

    if not weather_dfs:
        return df

    all_weather = pd.concat(weather_dfs, ignore_index=True)
    all_weather["Time_sec"] = pd.to_timedelta(all_weather["Time"]).dt.total_seconds()

    merged_dfs = []
    for race in all_races:
        race_laps = df[df["Race"] == race].sort_values("Time_sec").copy()
        race_wx = all_weather[(all_weather["Race"] == race)].sort_values("Time_sec").copy()
        if len(race_wx) == 0:
            merged_dfs.append(race_laps)
            continue
        merged = pd.merge_asof(
            race_laps,
            race_wx[["Time_sec", "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]],
            on="Time_sec", direction="nearest",
        )
        merged_dfs.append(merged)
    return pd.concat(merged_dfs, ignore_index=True)


def _clean_and_featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules and final feature engineering."""
    # Convert LapTime from timedelta string to float seconds (if not already numeric)
    if not pd.api.types.is_numeric_dtype(df["LapTime"]):
        df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

    df = df.dropna(subset=["LapTime"])
    df = df[df["LapTime"] > 60]

    # 107% rule — computed per (Race, Compound), NOT per race.
    # A wet race's fastest lap (~105s on INTERMEDIATE) is far slower than a
    # dry slick lap, so comparing all compounds against the single race-fastest
    # lap wrongly deletes 100% of wet laps (they sit well outside 107% of the
    # dry best). Grouping by compound keeps each tyre's laps valid relative to
    # its own fastest lap. TrackStatus!=1 (SC/VSC/red-flag) laps are excluded
    # from the baseline minimum so a stoppage lap can't poison the cutoff.
    # We mask LapTime to NaN on non-green laps, then groupby().transform("min")
    # returns a full-index Series aligned to df (NaN where no green lap exists
    # for that group, falling back to the overall compound min).
    lap_green = df["LapTime"].where(df.get("TrackStatus", 1) == 1)
    green_min = df.assign(_lg=lap_green).groupby(["Race", "Compound"])["_lg"].transform("min")
    any_min = df.groupby(["Race", "Compound"])["LapTime"].transform("min")
    safe_min = green_min.where(~green_min.isna(), any_min)
    df = df[df["LapTime"] <= safe_min * 1.07]

    # Deduplicate — keep the fastest lap per (Race, Driver, Year, Stint, TyreLife).
    # Same-stint duplicates with identical tyre config add no information and
    # would bias the model toward races with more duplicate records.
    df = df.sort_values("LapTime").drop_duplicates(
        subset=["Race", "Driver", "Year", "Stint", "TyreLife"],
        keep="first",
    )

    # Lap number within race
    df["LapInRace"] = df.groupby(["Race", "Driver"]).cumcount() + 1

    # Fuel weight effect
    df["FuelWeightEffect"] = df["LapInRace"] * -0.03

    # Quadratic features
    df["TyreLife_sq"] = df["TyreLife"] ** 2
    df["LapInRace_sq"] = df["LapInRace"] ** 2

    # Driver form (rolling avg of last 5 laps)
    df["DriverForm"] = df.groupby(["Race", "Driver"])["LapTime"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["DriverForm"] = df["DriverForm"].fillna(df["LapTime"])

    # ── P2: Track evolution proxy (rubbering in) ──────────────────────
    # Session-wide lap counter — laps completed in this race by all drivers.
    # More laps = more rubber laid down = faster track.
    df["TrackEvoProxy"] = df.groupby("Race").cumcount() / 1000.0  # scale to ~0-0.1

    return df


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns used by the model and dashboard."""
    keep_cols = [
        "Year", "Driver", "LapTime", "Compound", "TyreLife", "TyreLife_sq",
        "Stint", "TrackStatus", "Race", "LapInRace", "LapInRace_sq",
        "FuelWeightEffect", "DriverForm", "Position_normalized",
        "S1_speed", "S2_speed", "S3_speed", "AvgSpeed",
        "IsPersonalBest_int", "FreshTire_int", "IsStartLap",
        "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed",
        "CircuitLength_km", "CircuitCorners", "CircuitAvgSpeed", "CircuitType_enc",
        # P2: New non-linear tyre features
        "TyreLifePhase", "StopNumber", "TrackEvoProxy",
    ]
    keep = [c for c in keep_cols if c in df.columns]
    df = df[keep]
    df = df.dropna()
    return df


def process_year(year: int, circuits_df: pd.DataFrame) -> pd.DataFrame:
    """Load raw CSV for a year, process and return clean dataframe."""
    raw_path = os.path.join(DATA_DIR, f"all_races_{year}.csv")
    if not os.path.exists(raw_path):
        print(f"[{year}] Raw data not found at {raw_path}")
        print(f"[{year}] Run: python src/download_all_races.py --year {year}")
        return pd.DataFrame()

    print(f"\n[{year}] Loading {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"[{year}] Raw shape: {df.shape}")

    df["Year"] = year
    # Normalise race naming: 2025 "Spanish Grand Prix" → "Barcelona Grand Prix"
    df["Race"] = df["Race"].replace("Spanish Grand Prix", "Barcelona Grand Prix")
    df = _parse_lap_timedeltas(df)
    df = _engineer_features(df)
    df = _add_circuit_features(df, circuits_df)

    print(f"[{year}] Loading weather data...")
    df = _load_weather_for_year(year, df)

    print(f"[{year}] Cleaning and featurising...")
    df = _clean_and_featurize(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare F1 lap data for model training.")
    parser.add_argument("--years", nargs="+", type=int, default=[2025, 2026],
                        help="Years to process (default: 2025 2026)")
    parser.add_argument("--output", default="all_races_master.csv",
                        help="Output filename in data/ (default: all_races_master.csv)")
    args = parser.parse_args()

    circuits_path = os.path.join(DATA_DIR, "circuits_metadata.csv")
    circuits_df = _load_circuits_metadata(circuits_path)
    print(f"Loaded {len(circuits_df)} circuits from metadata")

    all_frames: list[pd.DataFrame] = []
    for year in sorted(args.years):
        frame = process_year(year, circuits_df)
        if not frame.empty:
            all_frames.append(frame)

    if not all_frames:
        print("No data processed. Exiting.")
        sys.exit(1)

    df = pd.concat(all_frames, ignore_index=True)
    df = _select_columns(df)

    out_path = os.path.join(DATA_DIR, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nDone — saved to {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Years: {sorted(df['Year'].unique())}")
    print(f"Races: {df['Race'].nunique()}")
    print(f"Drivers: {df['Driver'].nunique()}")
    print(f"Laps: {len(df)}")


if __name__ == "__main__":
    main()
