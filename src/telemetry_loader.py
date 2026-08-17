"""
F1 Telemetry Pipeline — loads, caches, and visualizes car telemetry.

Provides access to speed, throttle, brake, gear, DRS, RPM, and
sector data per lap per driver via fastf1.
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import fastf1
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_BASE, "cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(_CACHE_DIR)

# Legacy export — kept for backwards compatibility only.
# Do NOT use for new code. resolve_session() resolves via the event schedule.
RACE_SHORT_NAMES: dict[str, str] = {}

# Cross-year event name aliases (dashboard track names → fastf1 EventName)
_EVENT_ALIASES: dict[str, dict[int, str]] = {
    "Barcelona Grand Prix": {2026: "Spanish Grand Prix"},
    "Spanish Grand Prix":   {2025: "Barcelona Grand Prix"},
}


def _resolve_event_name(year: int, race_name: str) -> str | None:
    """Match a dashboard track name to the correct fastf1 EventName for *year*.

    Resolution order:
    1. Exact match against the fastf1 event schedule (EventName column)
    2. Alias lookup (e.g. 2026 "Barcelona Grand Prix" → "Spanish Grand Prix")
    3. Partial / substring match on EventName and Location columns
    """
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception:
        return None

    valid = schedule[schedule["EventFormat"] != "testing"]
    event_names = valid["EventName"].tolist()
    locations = valid["Location"].tolist()

    # 1. Exact EventName match
    if race_name in event_names:
        return race_name

    # 2. Alias lookup
    alias_for_year = _EVENT_ALIASES.get(race_name, {})
    if year in alias_for_year:
        candidate = alias_for_year[year]
        if candidate in event_names:
            return candidate

    # 3. Fuzzy: match cleaned input against EventName or Location
    clean = race_name.lower().replace(" grand prix", "").replace(" gp", "").strip()
    for en, loc in zip(event_names, locations):
        en_clean = en.lower().replace(" grand prix", "").replace(" gp", "").strip()
        loc_clean = loc.lower().strip()
        if (clean in en_clean or en_clean in clean
                or clean in loc_clean or loc_clean in clean):
            return en

    return None


def resolve_session(year: int, race_name: str) -> Any | None:
    """Resolve and load a race session from fastf1."""
    event_name = _resolve_event_name(year, race_name)
    if event_name is None:
        warnings.warn(f"telemetry_loader: no {year} event matched '{race_name}'")
        return None
    try:
        session = fastf1.get_session(year, event_name, "R")
        session.load(laps=True, telemetry=True, weather=True)
        return session
    except Exception as e:
        warnings.warn(f"telemetry_loader: failed to load {event_name} {year}: {e}")
        return None


def get_driver_lap_telemetry(
    session: Any, driver_code: str,
    fastest_only: bool = True,
) -> pd.DataFrame | None:
    """Get telemetry for a driver's fastest lap (or all laps)."""
    try:
        driver_laps = session.laps.pick_driver(driver_code)
        if driver_laps.empty:
            return None
        if fastest_only:
            lap = driver_laps.pick_fastest()
        else:
            # Use the most representative lap (median)
            median_time = driver_laps["LapTime"].median()
            lap = driver_laps.iloc[(driver_laps["LapTime"] - median_time).abs().argsort()[:1]]
        if lap.empty:
            return None
        # Get telemetry for this specific lap
        lap_telemetry = lap.get_car_data().add_distance()
        return lap_telemetry
    except Exception:
        return None


def get_driver_sector_times(
    session: Any, driver_code: str,
) -> list[dict[str, Any]] | None:
    """Get sector times for each lap by a driver."""
    try:
        driver_laps = session.laps.pick_driver(driver_code)
        if driver_laps.empty:
            return None
        result = []
        for _, lap in driver_laps.iterrows():
            s = lap.to_dict()
            if pd.notna(lap.get("Sector1Time")) and pd.notna(lap.get("Sector2Time")):
                result.append({
                    "lap": s.get("LapNumber", 0),
                    "s1": s["Sector1Time"].total_seconds(),
                    "s2": s["Sector2Time"].total_seconds(),
                    "s3": s["Sector3Time"].total_seconds() if pd.notna(s.get("Sector3Time")) else 0,
                    "compound": s.get("Compound", ""),
                    "tyre_life": s.get("TyreLife", 0),
                })
        return result
    except Exception:
        return None


def plot_telemetry_comparison(
    telemetry_a: pd.DataFrame | None,
    telemetry_b: pd.DataFrame | None,
    label_a: str = "Driver A",
    label_b: str = "Driver B",
    figsize: tuple[int, int] = (14, 8),
) -> plt.Figure | None:
    """
    Side-by-side telemetry comparison: speed, throttle, brake, gear, DRS.
    Returns matplotlib Figure for Streamlit embedding.
    """
    if telemetry_a is None and telemetry_b is None:
        return None

    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True, gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor("#0d0d0d")

    colors = {"a": "#e10600", "b": "#3793ff"}
    styles = {"a": "-", "b": "--"}

    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#444", labelsize=7)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#222")
        ax.spines["bottom"].set_color("#222")
        ax.grid(True, alpha=0.06, color="#fff")

    data_sets = [
        (telemetry_a, label_a, "a"),
        (telemetry_b, label_b, "b"),
    ]
    plotted = False

    for telemetry, label, key in data_sets:
        if telemetry is None or telemetry.empty:
            continue
        distance = telemetry.get("Distance", telemetry.index)

        # Speed
        if "Speed" in telemetry.columns:
            axes[0].plot(distance, telemetry["Speed"], color=colors[key], linestyle=styles[key],
                         linewidth=1.2, label=label)
            axes[0].set_ylabel("Speed (km/h)", color="#555", fontsize=8)

        # Throttle (0-100)
        if "Throttle" in telemetry.columns:
            axes[1].fill_between(distance, telemetry["Throttle"], alpha=0.3, color=colors[key])
            axes[1].plot(distance, telemetry["Throttle"], color=colors[key], linestyle=styles[key],
                         linewidth=0.8, label=label)
            axes[1].set_ylabel("Throttle (%)", color="#555", fontsize=8)
            axes[1].set_ylim(-5, 105)

        # Brake (0 or 1 typically)
        if "Brake" in telemetry.columns:
            brake_data = telemetry["Brake"].astype(float) * 100
            axes[2].fill_between(distance, brake_data, alpha=0.4, color=colors[key])
            axes[2].plot(distance, brake_data, color=colors[key], linestyle=styles[key],
                         linewidth=0.8, label=label)
            axes[2].set_ylabel("Brake (%)", color="#555", fontsize=8)
            axes[2].set_ylim(-5, 105)

        # Gear
        if "nGear" in telemetry.columns:
            axes[3].plot(distance, telemetry["nGear"], color=colors[key], linestyle=styles[key],
                         linewidth=1.2, label=label)
            axes[3].set_ylabel("Gear", color="#555", fontsize=8)
            axes[3].set_ylim(0, 9)

        # DRS (0/1)
        if "DRS" in telemetry.columns:
            drs = telemetry["DRS"].astype(float) * 100
            axes[4].fill_between(distance, drs, alpha=0.5, color=colors[key])
            axes[4].set_ylabel("DRS (%)", color="#555", fontsize=8)
            axes[4].set_ylim(-5, 105)

        plotted = True

    if plotted:
        axes[4].set_xlabel("Distance (m)", color="#555", fontsize=8)
        # Style legend on top subplot
        axes[0].legend(facecolor="#151515", labelcolor="#aaa", fontsize=7,
                       framealpha=0.95, edgecolor="#2a2a2a", handlelength=1.2)
        # Add compound badges / lap info as text
        return fig

    plt.close(fig)
    return None


def plot_sector_comparison(
    sectors_a: list[dict[str, Any]] | None,
    sectors_b: list[dict[str, Any]] | None,
    label_a: str = "Driver A",
    label_b: str = "Driver B",
) -> plt.Figure | None:
    """
    Sector time comparison across all laps.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor("#0d0d0d")

    data_sets = [
        (sectors_a, label_a, "#e10600"),
        (sectors_b, label_b, "#3793ff"),
    ]

    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#444", labelsize=7)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#222")
        ax.spines["bottom"].set_color("#222")

    sector_names = ["Sector 1", "Sector 2", "Sector 3"]
    sector_keys = ["s1", "s2", "s3"]

    plotted = False
    for idx, (ax, sec_key, sec_name) in enumerate(zip(axes, sector_keys, sector_names)):
        for sectors, label, color in data_sets:
            if sectors is None:
                continue
            laps = [s["lap"] for s in sectors]
            times = [s[sec_key] for s in sectors]
            ax.plot(laps, times, color=color, linewidth=1, alpha=0.7, marker=".", markersize=2, label=label if idx == 0 else "")
            ax.set_ylabel(f"{sec_name} (s)", color="#555", fontsize=8)
        plotted = True

    if plotted:
        axes[0].legend(facecolor="#151515", labelcolor="#aaa", fontsize=7,
                       framealpha=0.95, edgecolor="#2a2a2a")
        axes[2].set_xlabel("Lap", color="#555", fontsize=8)
        return fig

    plt.close(fig)
    return None


def get_session_weather(session: Any) -> list[dict[str, Any]]:
    """Extract weather data from a session."""
    try:
        w = session.weather_data
        if w is None or w.empty:
            return []
        result = []
        for _, row in w.iterrows():
            result.append({
                "time": row.get("Time", 0),
                "air_temp": row.get("AirTemp", 25),
                "track_temp": row.get("TrackTemp", 30),
                "humidity": row.get("Humidity", 50),
                "rainfall": row.get("Rainfall", 0),
                "wind_speed": row.get("WindSpeed", 2),
            })
        return result
    except Exception:
        return []
