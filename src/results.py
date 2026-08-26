"""
F1 Race Results Tab — Session results for any track/session combination.

Supports: FP1, FP2, FP3, Sprint Qualifying, Sprint, Q, R
Data source: fastf1 (live API)
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import fastf1
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

_RESULTS_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"
)
os.makedirs(_RESULTS_CACHE, exist_ok=True)
fastf1.Cache.enable_cache(_RESULTS_CACHE)

TEAM_COLORS: dict[str, str] = {
    "Red Bull Racing": "#3671c6", "Red Bull": "#3671c6",
    "Ferrari": "#e8002d",
    "Mercedes": "#27f4d2",
    "McLaren": "#ff8000",
    "Aston Martin": "#229971",
    "Alpine": "#00a1e8",
    "Haas F1 Team": "#dee1e2", "Haas": "#dee1e2",
    "Racing Bulls": "#6692ff", "RB": "#6692ff",
    "Williams": "#1868db",
    "Audi": "#ff2d00",
    "Cadillac": "#aaaaad",
}

SESSION_TYPES = ["FP1", "FP2", "FP3", "Sprint Qualifying", "Sprint", "Q", "R"]

# Display names for the session radio buttons
SESSION_DISPLAY = {
    "FP1": "FP1", "FP2": "FP2", "FP3": "FP3",
    "Q": "Qualifying", "Sprint Qualifying": "Sprint Qualifying",
    "Sprint": "Sprint", "R": "Race",
}

# Sessions that need laps loaded to get best lap times (results have no position data)
LAPS_NEEDED = {"FP1", "FP2", "FP3", "Sprint Qualifying"}

# ── F1 points system ──────────────────────────────────────────────────────
RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


def _team_color(team: str) -> str:
    return TEAM_COLORS.get(team, "#666666")


def _norm_driver_name(full_name: str) -> str:
    """Normalize driver display names for consistency across the app.

    fastf1 sometimes stores the same driver under different FullName strings
    (e.g. 'Andrea Kimi Antonelli' vs 'Kimi Antonelli'). Collapse known variants
    to a single canonical display name so tables/charts never differ.
    """
    if not full_name:
        return full_name
    name = str(full_name).strip()
    if name in ("Andrea Kimi Antonelli", "Andrea Antonelli"):
        return "Kimi Antonelli"
    return name


def _norm_status(status: str) -> str:
    """Normalize race-status labels for consistent display.

    'Retired' -> 'DNF', 'Did not start' -> 'DNS'. Other values pass through.
    """
    if not status or (hasattr(status, "isna") and status.isna()):
        return status
    s = str(status).strip()
    return {"Retired": "DNF", "Did not start": "DNS", "Disqualified": "DSQ"}.get(s, s)


def _fmt_time(td) -> str:
    """Format a timedelta to a readable lap time string."""
    if pd.isna(td) or td is None:
        return "—"
    if isinstance(td, (int, float)):
        mins = int(td // 60)
        secs = td - mins * 60
        return f"{mins}:{secs:06.3f}"
    total = td.total_seconds() if hasattr(td, "total_seconds") else float(td)
    mins = int(total // 60)
    secs = total - mins * 60
    return f"{mins}:{secs:06.3f}"


def _fmt_gap(time_val, pole_time) -> str:
    """Format gap to pole/leader."""
    if pd.isna(time_val) or pd.isna(pole_time):
        return "—"
    if isinstance(time_val, (int, float)) and isinstance(pole_time, (int, float)):
        diff = time_val - pole_time
    else:
        diff = time_val.total_seconds() - pole_time.total_seconds()
    if diff == 0:
        return "Pole"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.3f}s"


def _safe_gap(time_val, leader_time) -> str:
    """Gap to leader, but only when both are plausible full-session times.

    FastF1's Sprint 'Time' field is unreliable: only the winner carries a real
    cumulative race time (~25+ min); everyone else gets a garbage sub-minute value.
    Computing a gap from those yields nonsense (e.g. -1597s), so fall back to a
    blank gap unless both times are valid full-race durations (>60s)."""
    if pd.isna(time_val) or pd.isna(leader_time):
        return "—"
    try:
        tv = time_val.total_seconds()
        lt = leader_time.total_seconds()
    except AttributeError:
        return "—"
    if tv <= 60 or lt <= 60:
        # Unreliable data — don't show a misleading number
        return "" if tv > 0 else "—"
    diff = tv - lt
    if diff == 0:
        return "Pole"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.3f}s"


def _load_session_results(year: int, race: str, session_type: str) -> pd.DataFrame | None:
    """Load session results from fastf1. For FP/Sprint Quali, loads laps and
    computes best lap times per driver (these sessions have no position data)."""
    try:
        session = fastf1.get_session(year, race, session_type)
        if session_type in LAPS_NEEDED:
            session.load()  # full load including laps
            laps = session.laps
            results = session.results
            if laps is None or laps.empty:
                return None
            # Compute best lap per driver
            best = laps.groupby("DriverNumber")["LapTime"].min().reset_index()
            best.rename(columns={"LapTime": "BestLapTime"}, inplace=True)
            lap_counts = laps.groupby("DriverNumber")["LapNumber"].count().reset_index()
            lap_counts.rename(columns={"LapNumber": "Laps"}, inplace=True)
            merged = best.merge(lap_counts, on="DriverNumber")
            merged = merged.merge(
                results[["DriverNumber", "FullName", "TeamName"]].drop_duplicates("DriverNumber"),
                on="DriverNumber", how="left"
            )
            merged = merged.sort_values("BestLapTime").reset_index(drop=True)
            return merged
        else:
            session.load(laps=False, telemetry=False)
            return session.results.copy()
    except Exception as e:
        warnings.warn(f"Failed to load {race} {session_type} ({year}): {e}")
        return None


def _build_results_table(results: pd.DataFrame, session_type: str) -> pd.DataFrame:
    """Build a clean results table from fastf1 results."""
    table = pd.DataFrame()

    if session_type in LAPS_NEEDED:
        # Practice/Sprint Quali sessions: sorted by best lap time, no position
        df = results.copy()
        table["Rank"] = range(1, len(df) + 1)
        table["Driver"] = df["FullName"].apply(_norm_driver_name)
        table["No"] = df["DriverNumber"]
        table["Team"] = df["TeamName"]
        table["Best Lap"] = df["BestLapTime"].apply(_fmt_time) if "BestLapTime" in df.columns else "—"
        table["Laps"] = df["Laps"].apply(lambda x: int(x) if pd.notna(x) else 0)
        table["Gap"] = ""
        if "BestLapTime" in df.columns and len(df) > 0:
            best = df["BestLapTime"].iloc[0]
            table["Gap"] = df["BestLapTime"].apply(lambda x: _fmt_gap(x, best))

    else:
        # Race/Sprint/Qualifying sessions: use Position
        table["Pos"] = results["Position"].astype("Int64")
        table["Driver"] = results["FullName"].apply(_norm_driver_name)
        table["No"] = results["DriverNumber"]
        table["Team"] = results["TeamName"]

        if session_type in ("Q", "Sprint Qualifying"):
            table["Q1"] = results["Q1"].apply(_fmt_time)
            table["Q2"] = results["Q2"].apply(_fmt_time)
            table["Q3"] = results["Q3"].apply(_fmt_time)
            pole_q3 = results["Q3"].iloc[0] if len(results) > 0 else None
            table["Gap"] = results["Q3"].apply(lambda x: _fmt_gap(x, pole_q3))

        elif session_type == "R":
            table["Grid"] = results["GridPosition"].astype("Int64")
            table["Points"] = results["Points"].astype(int)
            table["Status"] = results["Status"].apply(_norm_status)
            leader_time = results["Time"].iloc[0] if len(results) > 0 else None
            # Gap only makes sense for drivers who completed the race.
            # Non-finishers (DNF/DNS/DSQ) get their status instead of a bogus gap.
            finished_mask = results["Status"].isin(["Finished", "Lapped"])
            table["Gap"] = [
                _fmt_gap(t, leader_time) if fin else _norm_status(st)
                for t, fin, st in zip(results["Time"], finished_mask, results["Status"])
            ]

        elif session_type == "Sprint":
            table["Grid"] = results["GridPosition"].astype("Int64") if "GridPosition" in results.columns else "—"
            table["Points"] = results["Points"].astype(int)
            table["Status"] = results["Status"].apply(_norm_status)
            leader_time = results["Time"].iloc[0] if len(results) > 0 else None
            finished_mask = results["Status"].isin(["Finished", "Lapped"])
            table["Gap"] = [
                _safe_gap(t, leader_time) if fin else _norm_status(st)
                for t, fin, st in zip(results["Time"], finished_mask, results["Status"])
            ]

        table = table.sort_values("Pos").reset_index(drop=True)

    return table


def _render_position_chart(results: pd.DataFrame, session_type: str, race: str, year: int):
    """Render a horizontal bar chart of positions. Practice/Sprint Qualifying
    sessions are ranked by best lap time (no official positions exist)."""
    if session_type in LAPS_NEEDED:
        # Practice / Sprint Qualifying: rank by best lap time
        df = results.dropna(subset=["BestLapTime"]).copy()
        df = df.sort_values("BestLapTime").reset_index(drop=True)
        df["Position"] = range(1, len(df) + 1)
    else:
        df = results.dropna(subset=["Position"]).copy()
        df = df.sort_values("Position")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    teams = df["TeamName"].values
    colors = [_team_color(t) for t in teams]
    y_pos = range(len(df))

    bars = ax.barh(y_pos, [1] * len(df), color=colors, height=0.7, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["FullName"].apply(_norm_driver_name), color="#eeeeee", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_xlim(0, 1.2)

    # Add position numbers
    for i, (_, row) in enumerate(df.iterrows()):
        pos = int(row["Position"])
        ax.text(0.05, i, f"P{pos}", va="center", ha="left", color="white",
                fontsize=9, fontweight="bold")

    ax.tick_params(axis="x", colors="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")

    plt.title(f"{race} {year} — {SESSION_DISPLAY.get(session_type, session_type)}", color="#eeeeee", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_results_tab():
    """Render the race results tab."""
    # ── Get available sessions for selected year+race ─────────────────────
    # Derive from the event schedule (single cached fetch via _get_schedule)
    # instead of downloading all 7 sessions. FastF1's Session1..Session5
    # columns already list which sessions a weekend actually has.
    SESSION_NAME_TO_KEY = {
        "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
        "Qualifying": "Q", "Sprint Qualifying": "Sprint Qualifying",
        "Sprint": "Sprint", "Race": "R",
    }

    @st.cache_data(ttl=3600, show_spinner=False)
    def _get_available_sessions(year: int, race: str) -> list[str]:
        schedule = _get_schedule(year)
        if race not in schedule:
            return []
        try:
            event = fastf1.get_event(year, race)
        except Exception:
            return []
        present = []
        for i in range(1, 6):
            sname = event.get(f"Session{i}")
            if not sname:
                continue
            key = SESSION_NAME_TO_KEY.get(sname)
            if key and key in SESSION_TYPES:
                present.append(key)
        # Preserve canonical order
        return [s for s in SESSION_TYPES if s in present]

    _RESULTS_CSV = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "race_results.csv",
    )

    def _csv_mtime() -> float:
        """Modification time of the pre-fetched results CSV.

        Used as a cache-busting key so the RESULTS tab never serves a stale
        (empty) cached result after the CSV is topped up by the auto-update.
        """
        try:
            return os.path.getmtime(_RESULTS_CSV)
        except OSError:
            return 0.0

    # Bump this whenever results.py logic changes, to force-clear the Streamlit
    # cache on next load (code changes alone don't change the CSV mtime, so the
    # mtime key wouldn't otherwise invalidate a stale cached result on Cloud).
    RESULTS_CACHE_VERSION = 6

    @st.cache_data(ttl=3600, show_spinner=False)
    def _get_results(year: int, race: str, session_type: str, _mtime: float = 0.0, _ver: int = RESULTS_CACHE_VERSION) -> pd.DataFrame | None:
        # 1) Try the pre-fetched results CSV (instant, works on Cloud too)
        df = _read_results_csv(year, race, session_type, _mtime=_mtime, _ver=_ver)
        if df is not None and not df.empty:
            return df
        # 2) Fallback to live fastf1 (future races not yet in the CSV)
        return _load_session_results(year, race, session_type)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _read_results_csv(year: int, race: str, session_type: str, _mtime: float = 0.0, _ver: int = 0) -> pd.DataFrame | None:
        if not os.path.exists(_RESULTS_CSV):
            return None
        try:
            csv = pd.read_csv(_RESULTS_CSV)
        except Exception:
            return None
        sub = csv[(csv["Year"] == year) & (csv["Race"] == race) & (csv["Session"] == session_type)]
        if sub.empty:
            return None
        # Reconstruct the same column shapes _build_results_table expects
        out = sub.copy()
        out = out.reset_index(drop=True)  # clean 0-based index so _build_results_table column alignment works
        for col in ("Position", "GridPosition", "Points", "Laps"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        # Parse timedelta-like strings back to timedelta for _fmt_gap/_fmt_time
        for col in ("Time", "BestLapTime", "Q1", "Q2", "Q3"):
            if col in out.columns:
                out[col] = pd.to_timedelta(out[col].replace("", pd.NA), errors="coerce")
        return out

    @st.cache_data(ttl=3600, show_spinner=False)
    def _get_schedule(year: int) -> list[str]:
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule["EventFormat"] != "testing"]
        return sorted(schedule["EventName"].tolist())

    # ── Layout ────────────────────────────────────────────────────────────
    # Year first, then tracks adapt to selected year
    c1, c2 = st.columns(2)
    with c2:
        r_year = st.selectbox("Year", [2026, 2025], key="res_year")
    tracks_for_year = _get_schedule(r_year)
    with c1:
        r_track = st.selectbox("Track", tracks_for_year, key="res_track")

    st.markdown("<br>", unsafe_allow_html=True)

    # Reset if track/year changed since last load
    loaded_key = st.session_state.get("_loaded_track_key")
    current_key = f"{r_year}_{r_track}"
    if loaded_key != current_key:
        st.session_state["_show_results"] = False

    if st.button("Load Results", key="load_results_btn"):
        st.session_state["_show_results"] = True
        st.session_state["_loaded_track_key"] = current_key
    if not st.session_state.get("_show_results"):
        return

    # ── Load available sessions ───────────────────────────────────────────
    with st.spinner(f"Loading sessions for {r_track}..."):
        available = _get_available_sessions(r_year, r_track)

    if not available:
        st.info("No session data available for this race.")
        return

    # ── Session selector ──────────────────────────────────────────────────
    display_names = [SESSION_DISPLAY.get(s, s) for s in available]
    name_to_key = {SESSION_DISPLAY.get(s, s): s for s in available}
    chosen_display = st.radio("Session", display_names, horizontal=True, key="res_session")
    r_session = name_to_key[chosen_display]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Load and display results ──────────────────────────────────────────
    with st.spinner(f"Loading {SESSION_DISPLAY.get(r_session, r_session)} results..."):
        results = _get_results(r_year, r_track, r_session, _mtime=_csv_mtime())

    if results is None or results.empty:
        st.warning(
            f"No results available for **{r_track} — {SESSION_DISPLAY.get(r_session, r_session)}** ({r_year}). "
            "The session may not have been held yet, or the fastf1 API rate limit was hit — "
            "try again in a few minutes."
        )
        return

    table = _build_results_table(results, r_session)

    # ── Summary metrics ───────────────────────────────────────────────────
    if r_session == "R":
        # Race: 4 boxes — Drivers, Finishers, DNFs, Best Mover
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Drivers", len(results))
        if "Status" in results.columns:
            # A finisher is anyone who completed the race: Finished OR Lapped.
            # DNF = Retired / Did not start / Disqualified / anything else.
            finished = results["Status"].isin(["Finished", "Lapped"])
            finishers = int(finished.sum())
            m2.metric("Finishers", finishers)
            m3.metric("DNFs", len(results) - finishers)
        if "GridPosition" in results.columns and "Position" in results.columns:
            best_mover = results.copy()
            best_mover["Gain"] = best_mover["GridPosition"] - best_mover["Position"]
            if len(best_mover) > 0:
                mover = best_mover.loc[best_mover["Gain"].idxmax()]
                m4.metric("Best Mover", f"{_norm_driver_name(mover['FullName'])}", f"+{int(mover['Gain'])} places")
    elif r_session == "Sprint":
        # Sprint: 3 boxes — Drivers, Finishers, DNFs
        m1, m2, m3 = st.columns(3)
        m1.metric("Drivers", len(results))
        if "Status" in results.columns:
            finished = results["Status"].isin(["Finished", "Lapped"])
            finishers = int(finished.sum())
            m2.metric("Finishers", finishers)
            m3.metric("DNFs", len(results) - finishers)
    # FP1 / FP2 / FP3 / Qualifying / Sprint Qualifying: no metric boxes

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Position chart ────────────────────────────────────────────────────
    _render_position_chart(results, r_session, r_track, r_year)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results table ─────────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>{chosen_display} Results</div>", unsafe_allow_html=True)

    # Color the team column
    def _color_team(row):
        team = row.get("Team", "")
        c = _team_color(team)
        return [f"background-color: {c}22"] * len(row)

    styled = table.style.apply(_color_team, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=33 * (len(table) + 1) + 4)
