"""
Build a committed telemetry cache from FastF1 — fastest lap per driver per race.

Scope (v1): one fastest lap's distance-indexed telemetry per (Year, Race, Driver),
written to data/telemetry/<Year>__<Race>__<Driver>.parquet.

Resumable + incremental: if a parquet already exists for a (Year, Race, Driver),
it is skipped. Safe to run in short bursts (e.g. time-boxed CI, or weekend batches).

The cache is small enough to commit directly to the repo (fastest-lap sample, not
full stint traces), so the CAR TELEMETRY tab can render offline on Streamlit Cloud.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import fastf1
import numpy as np
import pandas as pd

warnings = __import__("warnings")
warnings.filterwarnings("ignore")

BASE: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE, "data")
CACHE_DIR: str = os.path.join(BASE, "cache")
TELEM_DIR: str = os.path.join(DATA_DIR, "telemetry")

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
os.makedirs(TELEM_DIR, exist_ok=True)

# Which car-data columns to persist (distance is always present after add_distance).
_TELEmA_COLS = ["Speed", "Throttle", "Brake", "nGear", "DRS", "RPM"]

YEARS: list[int] = [2025, 2026]
SESSION_CODE: str = "R"


def _cache_key(year: int, race: str, driver: str) -> str:
    safe_race = race.replace(" ", "_").replace("/", "_")
    safe_driver = driver.replace(" ", "_")
    return f"{year}__{safe_race}__{safe_driver}.parquet"


def _list_cached_keys() -> set[tuple[int, str, str]]:
    """Return the set of (Year, Race, Driver) already cached on disk."""
    present: set[tuple[int, str, str]] = set()
    if not os.path.isdir(TELEM_DIR):
        return present
    for name in os.listdir(TELEM_DIR):
        if not name.endswith(".parquet"):
            continue
        parts = name[:-8].split("__")
        if len(parts) != 3:
            continue
        try:
            y = int(parts[0])
        except ValueError:
            continue
        present.add((y, parts[1].replace("_", " "), parts[2].replace("_", " ")))
    return present


def _build_one_driver(session: Any, driver_code: str) -> pd.DataFrame | None:
    """Return distance-indexed telemetry for a driver's fastest available lap."""
    try:
        driver_laps = session.laps.pick_driver(driver_code)
        if driver_laps.empty:
            return None
        # Prefer the fastest lap; fall back to median-time lap if fastest is missing data.
        lap = driver_laps.pick_fastest()
        if lap.empty:
            return None
        car_data = lap.get_car_data()
        if car_data is None or car_data.empty:
            # fastest lap has no telemetry — try the median-time lap as a fallback
            median_time = driver_laps["LapTime"].median()
            if pd.isna(median_time):
                return None
            idx = (driver_laps["LapTime"] - median_time).abs().argsort().iloc[:1]
            lap = driver_laps.iloc[idx]
            if lap.empty:
                return None
            car_data = lap.get_car_data()
            if car_data is None or car_data.empty:
                return None
        car_data = car_data.add_distance()
        return car_data
    except Exception:
        return None


def _write_cache(df: pd.DataFrame, year: int, race: str, driver: str) -> str:
    """Persist a distance-indexed telemetry frame to a committed parquet.

    Distance is stored as a regular column (not the index) so it round-trips cleanly
    across pandas versions; load_cached_telemetry() restores it as the index.
    """
    df = df.copy()
    if "Distance" not in df.columns and "Distance" not in df.index.name:
        # add_distance() puts Distance in the index — promote it.
        if "Distance" in df.index.names():
            df = df.reset_index()  # Distance becomes a column named "Distance"
        elif df.index.name == "Distance":
            df = df.reset_index()
    elif "Distance" not in df.columns:
        # Last resort: there is a numeric index we can call Distance.
        df = df.copy()
        df["Distance"] = df.index.astype(float)

    # Keep only columns that actually exist (RPM is not always emitted).
    cols = ["Distance"] + [c for c in _TELEmA_COLS if c in df.columns]
    df = df[cols].copy()
    # Coerce to float to avoid object dtypes that bloat the parquet.
    for c in df.columns:
        if c != "Distance":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    path = os.path.join(TELEM_DIR, _cache_key(year, race, driver))
    df.to_parquet(path, index=False)
    return path


def _decode_cache_key(name: str) -> tuple[int, str, str] | None:
    """Reverse of _cache_key: '2026__British_Grand_Prix__VER.parquet' -> (2026, 'British Grand Prix', 'VER')."""
    if not name.endswith(".parquet"):
        return None
    parts = name[:-8].split("__")
    if len(parts) != 3:
        return None
    try:
        y = int(parts[0])
    except ValueError:
        return None
    race = parts[1].replace("_", " ")
    driver = parts[2].replace("_", " ")
    return (y, race, driver)


def build_cache(
    years: list[int] | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build / extend the telemetry cache for `years`.

    Returns a summary dict with keys: years, races_attempted, drivers_attempted,
    cached (newly written), skipped (already present), failed.
    """
    years = years or YEARS
    cached_keys = _list_cached_keys() if resume else set()
    skipped: list[tuple[int, str, str]] = []
    cached: list[tuple[int, str, str]] = []
    failed: list[tuple[int, str, str, str]] = []

    total_races = 0
    total_drivers = 0

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            if verbose:
                print(f"  [{year}] cannot fetch schedule: {e}")
            continue
        schedule = schedule[schedule["EventFormat"] != "testing"]
        races = sorted(schedule["EventName"].tolist())
        total_races += len(races)

        for race in races:
            # Resolve session
            try:
                session = fastf1.get_session(year, race, SESSION_CODE)
                session.load(laps=True, telemetry=True, weather=False)
            except Exception as e:
                if verbose:
                    print(f"  [{year}] {race}: session load failed — {e}")
                continue

            laps = session.laps
            if laps is None or laps.empty:
                if verbose:
                    print(f"  [{year}] {race}: no laps")
                continue

            drivers = sorted(laps["DriverNumber"].unique())
            # Map fastf1 DriverNumber -> 3-letter code via session's driver list if possible.
            # fastf1 laps store Driver as the 3-letter code already on most sessions.
            # Fall back to the DriverNumber integer if the column is numeric.
            driver_codes: list[str] = []
            for dn in drivers:
                sample = laps[laps["DriverNumber"] == dn]
                if "Driver" in sample.columns and not sample["Driver"].isna().all():
                    code = str(sample["Driver"].iloc[0])
                else:
                    code = str(int(dn))
                driver_codes.append(code)

            total_drivers += len(driver_codes)

            for code in driver_codes:
                key = (year, race, code)
                if resume and key in cached_keys:
                    skipped.append(key)
                    continue
                if verbose:
                    print(f"  [{year}] {race} {code} ... ", end="", flush=True)
                cdf = _build_one_driver(session, code)
                if cdf is None or cdf.empty:
                    if verbose:
                        print("no telemetry")
                    failed.append((*key, "no telemetry"))
                    continue
                try:
                    _write_cache(cdf, year, race, code)
                    cached.append(key)
                    if verbose:
                        print(f"OK ({len(cdf)} rows)")
                except Exception as e:
                    if verbose:
                        print(f"FAIL — {e}")
                    failed.append((*key, str(e)))

    summary = {
        "years": years,
        "races_attempted": total_races,
        "drivers_attempted": total_drivers,
        "cached": cached,
        "skipped": skipped,
        "failed": failed,
        "telem_dir": TELEM_DIR,
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    s = summary
    print("\n" + "=" * 60)
    print(f"Telemetry cache build — years {s['years']}")
    print(f"Races attempted : {s['races_attempted']}")
    print(f"Driver-slots    : {s['drivers_attempted']}")
    print(f"Newly cached    : {len(s['cached'])}")
    print(f"Already present : {len(s['skipped'])}")
    print(f"Failed          : {len(s['failed'])}")
    if s["failed"]:
        print("\nFailed entries:")
        for y, r, d, why in s["failed"]:
            print(f"  {y} {r} {d} — {why}")
    print(f"\nCache dir: {s['telem_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build F1 car-telemetry cache (fastest-lap sample).")
    ap.add_argument("--years", nargs="+", type=int, default=None, help="Years to process (default: 2025 2026)")
    ap.add_argument("--no-resume", action="store_true", help="Rebuild everything (overwrite existing caches)")
    ap.add_argument("--quiet", action="store_true", help="Only print summary at the end")
    args = ap.parse_args()

    verbose = not args.quiet
    summary = build_cache(years=args.years, resume=not args.no_resume, verbose=verbose)
    if not verbose:
        print_summary(summary)
    else:
        print_summary(summary)
