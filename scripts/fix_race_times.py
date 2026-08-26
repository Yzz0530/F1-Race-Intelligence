"""One-off: recompute 'Time' for R/Sprint rows in race_results.csv from summed lap
times (fastf1's native Time column is unreliable for non-leaders). Updates CSV
in place. Collects all edits then applies once to avoid chained-assignment loss."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastf1, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fastf1.Cache.enable_cache(os.path.join(BASE, "cache"))
OUT = os.path.join(BASE, "data", "race_results.csv")

def _fmt(td):
    if td is None or (hasattr(td, "isna") and td.isna()) or pd.isna(td):
        return ""
    return str(td)

df = pd.read_csv(OUT)
updates = []  # (index, new_time_str)
years = [int(y) for y in sys.argv[1:]] or [2025, 2026]
for year in years:
    sched = fastf1.get_event_schedule(year)
    sched = sched[sched["EventFormat"] != "testing"]
    for race in sorted(sched["EventName"].tolist()):
        for stype in ("R", "Sprint"):
            mask = (df.Year == year) & (df.Race == race) & (df.Session == stype)
            if not mask.any():
                continue
            try:
                s = fastf1.get_session(year, race, stype)
                s.load(telemetry=False)
                laps = s.laps
                if laps is None or laps.empty:
                    continue
                tot = laps.groupby("DriverNumber")["LapTime"].sum()
                # Cumulative race time per driver = the last lap's cumulative
                # Time (more robust than summing LapTime, which can miss laps
                # for some drivers in older seasons).
                cum = laps.dropna(subset=["Time"]).groupby("DriverNumber")["Time"].max()
                for dn, total in tot.items():
                    dn_str = str(dn)
                    dmask = mask & (df.DriverNumber.astype(str) == dn_str)
                    idx = df.index[dmask]
                    if len(idx):
                        new_time = cum.get(dn, total)
                        updates.append((idx[0], _fmt(new_time)))
                print(f"[OK] {year} {race} {stype} ({len(updates)} so far)")
            except Exception as e:
                print(f"[SKIP] {year} {race} {stype}: {e}")

if updates:
    for i, val in updates:
        df.at[i, "Time"] = val
    df.to_csv(OUT, index=False)
    print(f"Updated {len(updates)} rows -> {OUT}")
else:
    print("Nothing updated.")
