"""Shared constants, helpers, and cached resources for dashboard tabs.

This module is the single import surface for every tab under src/tabs/.
It guarantees src/ is on sys.path (so engine modules resolve) and exposes
the team/compound styling, plot styling, and the Streamlit cache_resource
loaders used across tabs.
"""
from __future__ import annotations

import os
import sys

# Ensure src/ is importable from a tab module (src/tabs/*.py).
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from strategy_optimizer import F1StrategyOptimizer  # noqa: E402
from race_physics import PIT_LOSS_DEFAULT, simulate_sc_scenario, undercut_benefit, fuel_effect, traffic_effect  # noqa: E402
from undercut_analyzer import UndercutAnalyzer  # noqa: E402
from strategy_assistant import StrategyAssistant  # noqa: E402

F1_RED = "#e10600"
COMPOUND_COLORS: dict[str, str] = {
    "SOFT": "#e10600", "MEDIUM": "#ffb800", "HARD": "#a0a0a0",
}
TEAMS: dict[str, tuple[str, str]] = {
    "VER": ("#3671c6", "Red Bull"), "HAD": ("#3671c6", "Red Bull"),
    "LEC": ("#e8002d", "Ferrari"), "HAM": ("#e8002d", "Ferrari"),
    "RUS": ("#27f4d2", "Mercedes"), "ANT": ("#27f4d2", "Mercedes"),
    "NOR": ("#ff8000", "McLaren"), "PIA": ("#ff8000", "McLaren"),
    "ALO": ("#229971", "Aston Martin"), "STR": ("#229971", "Aston Martin"),
    "GAS": ("#00a1e8", "Alpine"), "COL": ("#00a1e8", "Alpine"),
    "OCO": ("#dee1e2", "Haas"), "BEA": ("#dee1e2", "Haas"),
    "LAW": ("#6692ff", "Racing Bulls"), "LIN": ("#6692ff", "Racing Bulls"),
    "TSU": ("#6692ff", "Racing Bulls"),
    "ALB": ("#1868db", "Williams"), "SAI": ("#1868db", "Williams"),
    "HUL": ("#ff2d00", "Audi"), "BOR": ("#ff2d00", "Audi"),
    "PER": ("#aaaaad", "Cadillac"), "BOT": ("#aaaaad", "Cadillac"),
}
DRIVERS_LIST = sorted(TEAMS.keys())


# ── Styling helpers ───────────────────────────────────────────────
def _team_color(driver: str) -> str:
    return TEAMS.get(driver, ("#666", ""))[0]


def _team_name(driver: str) -> str:
    return TEAMS.get(driver, ("", ""))[1]


def _compound_badge(cpd: str) -> str:
    cls = {"SOFT": "cpd-soft-bg", "MEDIUM": "cpd-medium-bg", "HARD": "cpd-hard-bg"}.get(cpd, "")
    return f"<span class='compound-badge {cls}'>{cpd}</span>"


def _driver_tag(driver: str) -> str:
    c = _team_color(driver)
    return (
        f"<span style='display:inline-flex;align-items:center;gap:4px;'>"
        f"<span class='team-dot' style='color:{c};'></span>"
        f"<b style='color:var(--text-primary);'>{driver}</b>"
        f"<span style='color:var(--text-dim);font-size:0.7rem;'>{_team_name(driver)}</span></span>"
    )


def style_ax(ax: plt.Axes, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor("#0d0d0d")
    ax.set_xlabel(xlabel, color="#555", fontsize=8, labelpad=6)
    ax.set_ylabel(ylabel, color="#555", fontsize=8, labelpad=6)
    ax.tick_params(colors="#444", labelsize=7)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#222")
    ax.yaxis.label.set_color("#555")
    ax.xaxis.label.set_color("#555")


def style_legend(ax: plt.Axes) -> None:
    ax.legend(
        facecolor="#151515", labelcolor="#aaa", fontsize=7,
        framealpha=0.95, edgecolor="#2a2a2a", borderpad=0.6,
        handlelength=1.2, handletextpad=0.6,
    )


# ── Cached resources ─────────────────────────────────────────────
@st.cache_resource
def load_optimizer() -> F1StrategyOptimizer:
    return F1StrategyOptimizer()


@st.cache_resource
def load_assistant() -> StrategyAssistant:
    return StrategyAssistant(load_optimizer())


@st.cache_resource
def load_undercut() -> UndercutAnalyzer:
    return UndercutAnalyzer(base_lap_time=load_optimizer().overall_baseline)


@st.cache_data(ttl=600, show_spinner=False)
def run_opt(track: str, laps: int, driver: str, mc_runs: int,
            sc_prob: float, dnf_prob: float = 0.05) -> list[dict]:
    return load_optimizer().optimize(track, laps, driver, mc_runs=mc_runs, sc_prob=sc_prob, dnf_prob=dnf_prob)


@st.cache_data(ttl=600, show_spinner=False)
def run_detailed(track: str, laps: int, driver: str,
                 strategy_tuple: tuple[tuple[str, int], ...]) -> dict | None:
    return load_optimizer().get_detailed_run(track, laps, driver, list(strategy_tuple))
