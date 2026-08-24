"""
F1 Championship Standings Tab — Driver and Constructor standings with progression.

Data source: fastf1 (live API) — computes points from race/sprint results.
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

_STANDINGS_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"
)
os.makedirs(_STANDINGS_CACHE, exist_ok=True)
fastf1.Cache.enable_cache(_STANDINGS_CACHE)

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

# Points systems
RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
FASTEST_LAP_POINT = 1  # P1-P10 with fastest lap gets +1


def _team_color(team: str) -> str:
    return TEAM_COLORS.get(team, "#666666")


@st.cache_data(ttl=3600, show_spinner=False)
def _get_schedule(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year)
    return schedule[schedule["EventFormat"] != "testing"].copy()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_all_standings(year: int) -> pd.DataFrame:
    """Load all race/sprint results for a year and compute cumulative standings."""
    schedule = _get_schedule(year)
    all_records = []

    for _, event in schedule.iterrows():
        race_name = event["EventName"]
        is_sprint = "sprint" in str(event.get("EventFormat", "")).lower()

        # ── Race results ──────────────────────────────────────────────────
        try:
            session = fastf1.get_session(year, race_name, "R")
            session.load(laps=False, telemetry=False)
            results = session.results
            for _, row in results.iterrows():
                pos = row.get("Position")
                if pd.isna(pos):
                    continue
                pos = int(pos)
                pts = RACE_POINTS.get(pos, 0)
                all_records.append({
                    "Race": race_name,
                    "Session": "Race",
                    "Driver": row.get("FullName", ""),
                    "Team": row.get("TeamName", ""),
                    "Position": pos,
                    "Points": pts,
                    "Status": row.get("Status", ""),
                })
        except Exception as e:
            warnings.warn(f"Standings: failed to load Race {race_name} ({year}): {e}")
            pass

        # ── Sprint race results ───────────────────────────────────────────
        if is_sprint:
            try:
                sprint = fastf1.get_session(year, race_name, "Sprint")
                sprint.load(laps=False, telemetry=False)
                results = sprint.results
                for _, row in results.iterrows():
                    pos = row.get("Position")
                    if pd.isna(pos):
                        continue
                    pos = int(pos)
                    pts = SPRINT_POINTS.get(pos, 0)
                    all_records.append({
                        "Race": race_name,
                        "Session": "Sprint",
                        "Driver": row.get("FullName", ""),
                        "Team": row.get("TeamName", ""),
                        "Position": pos,
                        "Points": pts,
                        "Status": row.get("Status", ""),
                    })
            except Exception as e:
                warnings.warn(f"Standings: failed to load Sprint {race_name} ({year}): {e}")
                pass

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame(all_records)


def _compute_driver_standings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute driver standings with cumulative points per race.

    Grouped by DRIVER only (not Driver+Team): a driver who changes teams
    mid-season (e.g. Lawson Red Bull -> Racing Bulls) must appear as ONE entry,
    with all their points summed. The display team is the driver's most recent
    race team.
    """
    if df.empty:
        return pd.DataFrame()

    # Sort races by order in schedule
    race_order = df["Race"].unique().tolist()
    race_idx = {r: i for i, r in enumerate(race_order)}
    df = df.copy()
    df["RaceIdx"] = df["Race"].map(race_idx)

    # Most recent team per driver (last race they entered)
    last_team = (
        df.sort_values("RaceIdx").groupby("Driver").tail(1).set_index("Driver")["Team"]
    )

    # Cumulative points progression, indexed by Driver ONLY
    progression = df.pivot_table(
        index="Driver", columns="Race", values="Points", aggfunc="sum"
    ).fillna(0)

    # Reorder columns by race order
    ordered = [r for r in race_order if r in progression.columns]
    progression = progression[ordered]
    progression = progression.cumsum(axis=1)

    # Total = sum of all points for the driver
    progression["Total"] = df.groupby("Driver")["Points"].sum().values

    # Display team = most recent
    progression["Team"] = progression.index.map(last_team)

    # Sort by total
    progression = progression.sort_values("Total", ascending=False).reset_index()

    return progression


def _compute_constructor_standings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute constructor standings."""
    if df.empty:
        return pd.DataFrame()

    race_order = df["Race"].unique().tolist()

    # Group by team
    constructor_totals = df.groupby("Team").agg(
        TotalPoints=("Points", "sum"),
        Wins=("Position", lambda x: (x == 1).sum()),
        Podiums=("Position", lambda x: (x <= 3).sum()),
    ).reset_index()

    # Cumulative progression
    progression = df.pivot_table(
        index="Team", columns="Race", values="Points", aggfunc="sum"
    ).fillna(0)

    ordered = [r for r in race_order if r in progression.columns]
    progression = progression[ordered]
    progression = progression.cumsum(axis=1)

    progression["Total"] = df.groupby("Team")["Points"].sum().values

    progression = progression.sort_values("Total", ascending=False).reset_index()

    return progression


    """Render cumulative points progression chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    # Get race columns (exclude identity/total columns)
    id_cols = ["Driver", "Team", "Total"] if not is_constructor else ["Team", "Total"]
    race_cols = [c for c in standings.columns if c not in id_cols and c != "index"]

    if not race_cols:
        plt.close(fig)
        return

    # Plot top drivers/teams
    top_n = min(10, len(standings))
    for i in range(top_n):
        row = standings.iloc[i]
        name = row.get("Driver") if not is_constructor else row.get("Team")
        team = row.get("Team", "") if not is_constructor else row.get("Team", "")
        color = _team_color(team)
        y_vals = [0] + [row[c] for c in race_cols]

        ax.plot(range(len(y_vals)), y_vals, marker="o", markersize=4,
                color=color, linewidth=2, label=name, alpha=0.9)

    # X-axis labels
    x_labels = ["Pre"] + [r.replace(" Grand Prix", " GP").replace("São Paulo", "SPA")[:10] for r in race_cols]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, color="#999999", fontsize=8, rotation=45, ha="right")

    ax.tick_params(axis="y", colors="#999999")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")

    ax.legend(loc="upper left", fontsize=8, facecolor="#1a1a1a", edgecolor="#333333",
              labelcolor="#eeeeee", ncol=2)

    plt.title(title, color="#eeeeee", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Cumulative Points", color="#999999", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_standings_table(standings: pd.DataFrame, is_constructor: bool = False):
    """Render a styled standings table — dataframe look, no scroll."""
    if standings.empty:
        st.info("No standings data available.")
        return

    table = pd.DataFrame()
    table["Pos"] = range(1, len(standings) + 1)

    if is_constructor:
        table["Team"] = standings["Team"]
        table["Points"] = standings["Total"].astype(int)
    else:
        table["Driver"] = standings["Driver"]
        table["Team"] = standings["Team"]
        table["Points"] = standings["Total"].astype(int)

    def _color_row(row):
        c = _team_color(row.get("Team", ""))
        return [f"background-color: {c}22"] * len(row)

    styled = table.style.apply(_color_row, axis=1)

    # Disable scroll via CSS override on the dataframe container
    st.markdown("<style>div[data-testid='stDataFrame'] div[role='gridcell'], div[data-testid='stDataFrame'] div[tabindex] { overflow: visible !important; max-height: none !important; } div[data-testid='stDataFrame'] { overflow: visible !important; }</style>", unsafe_allow_html=True)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=33 * (len(table) + 1) + 4)


def render_standings_tab():
    """Render the championship standings tab."""
    # ── Year selector ─────────────────────────────────────────────────────
    s_year = st.selectbox("Season", [2026, 2025], key="standings_year")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tab toggle: Driver / Constructor ──────────────────────────────────
    standings_type = st.radio("Standings", ["Driver Championship", "Constructor Championship"],
                              horizontal=True, key="standings_type")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Load standings ────────────────────────────────────────────────────
    with st.spinner(f"Loading {s_year} championship data..."):
        raw = _load_all_standings(s_year)

    if raw.empty:
        st.warning(
            f"No {s_year} race data available yet — "
            "either no races have been held, or the fastf1 API rate limit was hit. "
            "Try again in a few minutes."
        )
        return

    # ── Summary metrics ───────────────────────────────────────────────────
    races_completed = raw["Race"].nunique()
    total_points = raw["Points"].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Races Completed", races_completed)
    m2.metric("Total Points Awarded", int(total_points))

    if standings_type == "Driver Championship":
        driver_standings = _compute_driver_standings(raw)
        if not driver_standings.empty:
            leader = driver_standings.iloc[0]
            m3.metric("Leader", leader["Driver"], f"{int(leader['Total'])} pts")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            _render_standings_table(driver_standings, is_constructor=False)
        else:
            m3.metric("Leader", "—", "—")
    else:
        constructor_standings = _compute_constructor_standings(raw)
        if not constructor_standings.empty:
            leader = constructor_standings.iloc[0]
            m3.metric("Leader", leader["Team"], f"{int(leader['Total'])} pts")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            _render_standings_table(constructor_standings, is_constructor=True)
        else:
            m3.metric("Leader", "—", "—")
