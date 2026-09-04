"""Build / extend data/race_results.csv — pre-fetched race/session results for
instant local + Streamlit Cloud loads (avoids per-load fastf1 network downloads).

Resumable + incremental: if data/race_results.csv already exists, sessions already
present (matching Year+Race+Session) are SKIPPED, and new sessions are APPENDED.
This lets the build run in many short bursts (e.g. when the process is time-boxed)
and finish unattended in the weekly GitHub Action.

Schema (one row per driver per session):
  Year, Race, Session,
  DriverNumber, FullName, TeamName,
  Position, GridPosition, Status, Time, Points,       # Race/Sprint/Qualifying
  Q1, Q2, Q3,                                          # Qualifying / Sprint Qualifying
  BestLapTime, Laps                                     # FP1/FP2/FP3 / Sprint Qualifying (best-lap rank)
"""
from __future__ import annotations

import os
import sys
import warnings

import fastf1
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE, "cache")
OUT = os.path.join(BASE, "data", "race_results.csv")
COLUMNS = [
    "Year", "Race", "Session", "DriverNumber", "FullName", "TeamName",
    "Position", "GridPosition", "Status", "Time", "Points",
    "Q1", "Q2", "Q3", "BestLapTime", "Laps",
]

fastf1.Cache.enable_cache(CACHE_DIR)

YEARS = [2025, 2026]
SESSION_TYPES = ["FP1", "FP2", "FP3", "Sprint Qualifying", "Sprint", "Q", "R"]
LAPS_NEEDED = {"FP1", "FP2", "FP3", "Sprint Qualifying"}


def _fmt(td) -> str:
    if td is None or (hasattr(td, "isna") and td.isna()) or pd.isna(td):
        return ""
    return str(td)


def _session_results(year: int, race: str, stype: str) -> list[dict]:
    rows = []
    session = fastf1.get_session(year, race, stype)
    if stype in LAPS_NEEDED:
        session.load()
        laps = session.laps
        results = session.results
        if laps is None or laps.empty or results is None or results.empty:
            return rows
        best = laps.groupby("DriverNumber")["LapTime"].min().reset_index()
        best.rename(columns={"LapTime": "BestLapTime"}, inplace=True)
        counts = laps.groupby("DriverNumber")["LapNumber"].count().reset_index()
        counts.rename(columns={"LapNumber": "Laps"}, inplace=True)
        merged = best.merge(counts, on="DriverNumber")
        merged = merged.merge(
            results[["DriverNumber", "FullName", "TeamName"]].drop_duplicates("DriverNumber"),
            on="DriverNumber", how="left",
        )
        merged = merged.sort_values("BestLapTime").reset_index(drop=True)
        for i, r in merged.iterrows():
            rows.append({
                "Year": year, "Race": race, "Session": stype,
                "DriverNumber": r["DriverNumber"], "FullName": r.get("FullName", ""),
                "TeamName": r.get("TeamName", ""),
                "Position": i + 1, "GridPosition": "", "Status": "", "Time": "",
                "Points": "", "Q1": "", "Q2": "", "Q3": "",
                "BestLapTime": _fmt(r.get("BestLapTime")), "Laps": r.get("Laps", ""),
            })
    else:
        # Race / Sprint: fastf1's native 'Time' column is unreliable for
        # non-leaders (often shows garbage). Compute each driver's total race
        # time from the sum of their lap times — this is the true cumulative
        # time and yields correct gaps.
        session.load(telemetry=False)
        res = session.results
        if res is None or res.empty:
            return rows
        total_time = {}
        if session.laps is not None and not session.laps.empty:
            try:
                # Cumulative race time per driver = last lap's cumulative Time.
                # More robust than summing LapTime (can miss laps for some
                # drivers in older seasons, and fastf1's native results Time
                # column is unreliable for non-leaders).
                cum = session.laps.dropna(subset=["Time"]).groupby("DriverNumber")["Time"].max()
                total_time = cum.to_dict()
            except Exception:
                total_time = {}
        for _, r in res.iterrows():
            dn = r.get("DriverNumber", "")
            tval = total_time.get(dn, r.get("Time")) if total_time else r.get("Time")
            rows.append({
                "Year": year, "Race": race, "Session": stype,
                "DriverNumber": _fmt(dn), "FullName": _fmt(r.get("FullName", "")),
                "TeamName": _fmt(r.get("TeamName", "")),
                "Position": _fmt(r.get("Position")),
                "GridPosition": _fmt(r.get("GridPosition")),
                "Status": _fmt(r.get("Status")),
                "Time": _fmt(tval),
                "Points": _fmt(r.get("Points")),
                "Q1": _fmt(r.get("Q1")), "Q2": _fmt(r.get("Q2")), "Q3": _fmt(r.get("Q3")),
                "BestLapTime": "", "Laps": "",
            })
    return rows


def _load_existing() -> pd.DataFrame:
    if os.path.exists(OUT):
        try:
            return pd.read_csv(OUT)
        except Exception:
            return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)


def main() -> None:
    existing = _load_existing()
    done_keys = set(
        zip(existing["Year"], existing["Race"], existing["Session"])
    ) if not existing.empty else set()

    new_rows: list[dict] = []
    for year in YEARS:
            schedule = fastf1.get_event_schedule(year)
            schedule = schedule[schedule["EventFormat"] != "testing"]
            races = sorted(schedule["EventName"].tolist())
            for race in races:
                for stype in SESSION_TYPES:
                    key = (year, race, stype)
                    if key in done_keys:
                        continue  # already in CSV — skip (resumable)
                    try:
                        rows = _session_results(year, race, stype)
                        # Skip if no rows returned (2026 data not on Ergast yet)
                        if not rows:
                            if verbose:
                                print(f"  [SKIP] {year} {race} {stype}: no data available (Ergast lag)")
                            continue
                        new_rows.extend(rows)
                        print(f"  [OK] {year} {race} {stype} ({len(rows)} drivers)")
                    except Exception as e:
                        print(f"  [SKIP] {year} {race} {stype}: {e}")

    if new_rows:
        df_new = pd.DataFrame(new_rows, columns=COLUMNS)
        combined = pd.concat([existing, df_new], ignore_index=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        combined.to_csv(OUT, index=False)
        print(f"\nAppended {len(df_new)} rows -> {OUT} (total {len(combined)} rows)")
    else:
        print("\nNothing new to add (all sessions already in CSV).")


if __name__ == "__main__":
    main()
