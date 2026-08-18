"""Quick proof-of-concept: build race_results.csv for ONE race only (Australian GP
2026, all 7 sessions) and verify results.py can read it instantly.

Usage: python scripts/poc_one_race.py
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastf1, pandas as pd
warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fastf1.Cache.enable_cache(os.path.join(BASE, "cache"))
OUT = os.path.join(BASE, "data", "race_results.csv")
COLUMNS = ["Year","Race","Session","DriverNumber","FullName","TeamName",
           "Position","GridPosition","Status","Time","Points","Q1","Q2","Q3","BestLapTime","Laps"]
SESSION_TYPES = ["FP1","FP2","FP3","Q","Sprint Qualifying","Sprint","R"]
LAPS_NEEDED = {"FP1","FP2","FP3","Sprint Qualifying"}

def _fmt(td):
    if td is None or (hasattr(td,'isna') and td.isna()) or pd.isna(td): return ""
    return str(td)

year, race = 2026, "Australian Grand Prix"
rows=[]
for stype in SESSION_TYPES:
    try:
        s = fastf1.get_session(year, race, stype)
        if stype in LAPS_NEEDED:
            s.load()
            laps=s.laps; res=s.results
            if laps is None or laps.empty or res is None or res.empty: continue
            best=laps.groupby("DriverNumber")["LapTime"].min().reset_index().rename(columns={"LapTime":"BestLapTime"})
            cnt=laps.groupby("DriverNumber")["LapNumber"].count().reset_index().rename(columns={"LapNumber":"Laps"})
            m=best.merge(cnt,on="DriverNumber").merge(res[["DriverNumber","FullName","TeamName"]].drop_duplicates("DriverNumber"),on="DriverNumber",how="left").sort_values("BestLapTime").reset_index(drop=True)
            for i,r in m.iterrows():
                rows.append({"Year":year,"Race":race,"Session":stype,"DriverNumber":r["DriverNumber"],"FullName":r.get("FullName",""),"TeamName":r.get("TeamName",""),"Position":i+1,"GridPosition":"","Status":"","Time":"","Points":"","Q1":"","Q2":"","Q3":"","BestLapTime":_fmt(r.get("BestLapTime")),"Laps":r.get("Laps","")})
        else:
            s.load(laps=False,telemetry=False); res=s.results
            if res is None or res.empty: continue
            for _,r in res.iterrows():
                rows.append({"Year":year,"Race":race,"Session":stype,"DriverNumber":r.get("DriverNumber",""),"FullName":r.get("FullName",""),"TeamName":r.get("TeamName",""),"Position":_fmt(r.get("Position")),"GridPosition":_fmt(r.get("GridPosition")),"Status":_fmt(r.get("Status")),"Time":_fmt(r.get("Time")),"Points":_fmt(r.get("Points")),"Q1":_fmt(r.get("Q1")),"Q2":_fmt(r.get("Q2")),"Q3":_fmt(r.get("Q3")),"BestLapTime":"","Laps":""})
        print(f"[OK] {stype}")
    except Exception as e:
        print(f"[SKIP] {stype}: {e}")

df=pd.DataFrame(rows,columns=COLUMNS)
df.to_csv(OUT,index=False)
print(f"Wrote {len(df)} rows -> {OUT}")
