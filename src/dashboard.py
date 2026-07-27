"""
F1 Strategy Optimizer Dashboard V2 — 8-tab full-feature UI.

Tabs: STRATEGY | DRIVER BATTLE | STINT TELEMETRY | TRACK ANALYSIS |
      SC SIMULATOR | UNDERCUT | TELEMETRY | AI ASSISTANT
"""
from __future__ import annotations

import os
import time
from typing import Any
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from strategy_optimizer import F1StrategyOptimizer
from race_physics import PIT_LOSS_DEFAULT, simulate_sc_scenario, undercut_benefit, fuel_effect
from undercut_analyzer import UndercutAnalyzer
from strategy_assistant import StrategyAssistant
from race_timeline import render_race_timeline
from results import render_results_tab
from standings import render_standings_tab

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
    "ALB": ("#1868db", "Williams"), "SAI": ("#1868db", "Williams"),
    "HUL": ("#ff2d00", "Audi"), "BOR": ("#ff2d00", "Audi"),
    "PER": ("#aaaaad", "Cadillac"), "BOT": ("#aaaaad", "Cadillac"),
}
DRIVERS_LIST = sorted(TEAMS.keys())
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


# ── Config ───────────────────────────────────────────────────────

st.set_page_config(page_title="F1 Race Intelligence", page_icon="assets/favicon.ico", layout="wide")

_css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(_css_path, encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)


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
            sc_prob: float, dnf_prob: float = 0.05) -> list[dict[str, Any]]:
    return load_optimizer().optimize(track, laps, driver, mc_runs=mc_runs, sc_prob=sc_prob, dnf_prob=dnf_prob)


@st.cache_data(ttl=600, show_spinner=False)
def run_detailed(track: str, laps: int, driver: str,
                 strategy_tuple: tuple[tuple[str, int], ...]) -> dict[str, Any] | None:
    return load_optimizer().get_detailed_run(track, laps, driver, list(strategy_tuple))


# ── Init ─────────────────────────────────────────────────────────

try:
    opt = load_optimizer()
    tracks = sorted(opt.circuit_info.keys())
    base_lap = opt.overall_baseline
    ua = load_undercut()
    assistant = load_assistant()
except Exception as e:
    st.error(f"Failed to load optimizer: {e}")
    st.stop()


# ACTIVE CIRCUIT ROTATION (sidebar — auto-cycles every 10s)
# ══════════════════════════════════════════════════════════════════

_prev_circuit = None

@st.fragment(run_every=10)
def _render_active_circuit():
    global _prev_circuit
    idx = int(time.time() / 10) % len(tracks)
    name = tracks[idx]
    prev = _prev_circuit
    _prev_circuit = name

    info = opt.circuit_info.get(name, {})
    length = info.get("Length_km", 0)
    corners = info.get("Corners", 0)
    speed = info.get("AvgSpeed", 0)

    p_len = p_corners = p_speed = 0
    if prev:
        pi = opt.circuit_info.get(prev, {})
        p_len, p_corners, p_speed = pi.get("Length_km", 0), pi.get("Corners", 0), pi.get("AvgSpeed", 0)

    old_html = ""
    if prev:
        old_html = f'''<div class="old" style="position:absolute;inset:0;text-align:center;">
            <div style="color:rgba(255,255,255,0.9);font-weight:600;font-size:0.85rem;">{prev}</div>
            <div style="color:rgba(255,255,255,0.5);font-size:0.6rem;margin-top:0.75rem;">{p_len} km · {p_corners} corners · {p_speed} km/h</div>
        </div>'''

    new_cls = "new" if prev else ""

    html = f"""<div style="
        font-family:Inter,'Segoe UI',sans-serif;
        text-align:left;
    ">
        <style>
            @keyframes oldFade {{0%{{opacity:1;}}100%{{opacity:0;}}}}
            @keyframes newFade {{0%{{opacity:0;}}100%{{opacity:1;}}}}
            .old{{animation:oldFade 0.35s ease-out forwards;}}
            .new{{animation:newFade 0.45s ease-out 0.35s both;}}
            body{{margin:0;background:transparent;}}
        </style>
        <div style="color:rgba(255,255,255,0.35);font-size:0.6rem;letter-spacing:0.5px;text-transform:uppercase;">Active Circuit</div><br>
        <div style="position:relative;min-height:1rem;text-align:center;">
            {old_html}
            <div class="{new_cls}">
                <div style="color:rgba(255,255,255,0.9);font-weight:600;font-size:0.85rem;">{name}</div>
                <div style="color:rgba(255,255,255,0.5);font-size:0.6rem;margin-top:0.75rem;">{length} km · {corners} corners · {speed} km/h</div>
            </div>
        </div>
    </div>"""

    components.html(html, height=72)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div class='f1-sidebar-header'>"
        "<span class='f1-logo-mark'>F1</span>"
        "<div>"
        "<div style='color:var(--text-primary);font-weight:700;font-size:0.95rem;letter-spacing:1.2px;'>"
        "Race Intelligence</div>"
        "<div style='color:var(--text-dim);font-size:0.62rem;letter-spacing:0.5px;'>PREDICT · SIMULATE · OPTIMIZE</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span class='badge' style='margin-bottom:0.8rem;display:inline-block;'>2026 SEASON · 24 RACES</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='tech-dot-group'>"
        "<div class='tech-dot-item'><div class='tech-dot'></div><span class='tech-dot-label'>FastF1</span></div>"
        "<div class='tech-dot-item'><div class='tech-dot'></div><span class='tech-dot-label'>XGBoost</span></div>"
        "<div class='tech-dot-item'><div class='tech-dot'></div><span class='tech-dot-label'>Streamlit</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='display:flex;gap:1.2rem;'>"
        f"<div class='stat-box'><div style='font-family:var(--font-mono);font-size:1.2rem;font-weight:700;color:var(--text-primary);'>{len(DRIVERS_LIST)}</div>"
        f"<div style='color:var(--text-dim);font-size:0.55rem;letter-spacing:0.8px;text-transform:uppercase;margin-top:2px;'>Drivers</div></div>"
        f"<div class='stat-box'><div style='font-family:var(--font-mono);font-size:1.2rem;font-weight:700;color:var(--f1-red);'>{len(tracks)}</div>"
        f"<div style='color:var(--text-dim);font-size:0.55rem;letter-spacing:0.8px;text-transform:uppercase;margin-top:2px;'>Tracks</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    _render_active_circuit()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:var(--text-muted);font-size:0.7rem;line-height:1.7;'>"
        "<span style='color:var(--text-dim);font-size:0.6rem;letter-spacing:0.5px;text-transform:uppercase;'>Technology Stack</span><br>"
        "XGBoost · Monte Carlo · Physics Engine<br>"
        "<span style='color:var(--text-dim);font-size:0.6em;'>MAE 0.73s · 27 features · 28K laps</span></div>",
        unsafe_allow_html=True,

    )
    # Background music playlist (invisible, auto-plays, rotates)
    _gh_base = "https://raw.githubusercontent.com/Yzz0530/F1-Race-Intelligence/master/assets"
    _songs = [
        f"{_gh_base}/f1_theme.mp3",
        f"{_gh_base}/don_toliver_lose_my_mind.mp3",
        f"{_gh_base}/tate_mcrae_just_keep_watching.mp3",
        f"{_gh_base}/rose_messy.mp3",
        f"{_gh_base}/ed_sheeran_drive.mp3",
    ]
    _js_songs = ",".join(""" + s + """ for s in _songs)
    st.components.v1.html(
        f"""<div style="position:absolute;opacity:0;width:0;height:0;overflow:hidden">
<audio id="f1audio"></audio>
<script>
var playlist = [{_js_songs}];
var idx = 0;
var a = document.getElementById("f1audio");
if (a) a.volume = 0.5;
function playNext() {{ idx = (idx + 1) % playlist.length; a.src = playlist[idx]; a.load(); a.play().catch(function(){{}}); }}
if (a) {{ a.addEventListener("ended", playNext); a.src = playlist[0]; a.load(); a.play().catch(function(){{}}); }}
document.addEventListener("click", function handler() {{ var a = document.getElementById("f1audio"); if (a && a.paused) a.play().catch(function(){{}}); document.removeEventListener("click", handler); }});
</script></div>""",
        height=1
    )
    st.markdown("<hr style='margin-top:2rem;opacity:0.3;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:var(--text-dim);font-size:0.55rem;text-align:center;letter-spacing:0.5px;line-height:1.6;padding-bottom:0.5rem;'>"
        "© 2026 Tang Yi Zhe. F1 Race Intelligence. All rights reserved.<br><br>"
        "<span style='font-size:0.48rem;opacity:0.6;'>"
        "Disclaimer: This project is an independent portfolio work and is not affiliated with, "
        "endorsed by, or associated with Formula 1, FIA, or any of their subsidiaries. "
        "All data is sourced from publicly available APIs and is for educational purposes only.</span></div>",
        unsafe_allow_html=True,
    )
