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
    "TSU": ("#6692ff", "Racing Bulls"),
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


# ACTIVE CIRCUIT ROTATION (sidebar — auto-cycles every 10s, JS fade transition)
# ══════════════════════════════════════════════════════════════════

import json


def _render_active_circuit():
    # Build circuit data once — embedded in JS, no fragment re-runs
    circuit_data = []
    for name in tracks:
        info = opt.circuit_info.get(name, {})
        circuit_data.append([
            name,
            float(info.get("Length_km", 0)),
            int(info.get("Corners", 0)),
            float(info.get("AvgSpeed", 0)),
        ])

    data_json = json.dumps(circuit_data)
    first = circuit_data[0]

    html = (
        '<div style="font-family:Inter,\'Segoe UI\',sans-serif;text-align:center;">'
        '<div id="cn" style="color:rgba(255,255,255,0.9);font-weight:600;font-size:0.85rem;transition:opacity 0.4s;">'
        + str(first[0]) +
        '</div>'
        '<div id="ci" style="color:rgba(255,255,255,0.5);font-size:0.6rem;margin-top:0.75rem;transition:opacity 0.4s;">'
        + str(first[1]) + ' km · ' + str(first[2]) + ' corners · ' + str(first[3]) + ' km/h'
        '</div>'
        '<script>'
        'var d=' + data_json + ';'
        'var i=1;'
        'setInterval(function(){'
        'var n=document.getElementById("cn"),f=document.getElementById("ci");'
        'n.style.opacity="0";f.style.opacity="0";'
        'setTimeout(function(){'
        'n.textContent=d[i][0];'
        'f.textContent=d[i][1]+" km \\u00b7 "+d[i][2]+" corners \\u00b7 "+d[i][3]+" km/h";'
        'n.style.opacity="1";f.style.opacity="1";'
        '},400);'
        'i=(i+1)%d.length;'
        '},10000);'
        '</script>'
        '</div>'
    )

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

    st.markdown(
        "<div style='color:rgba(255,255,255,0.35);font-size:0.6rem;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:0.35rem;'>Active Circuit</div>",
        unsafe_allow_html=True,
    )
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
    _gh = "https://raw.githubusercontent.com/Yzz0530/F1-Race-Intelligence/master/assets"
    _songs = [
        _gh + "/f1_theme.mp3",
        _gh + "/don_toliver_lose_my_mind.mp3",
        _gh + "/tate_mcrae_just_keep_watching.mp3",
        _gh + "/rose_messy.mp3",
        _gh + "/ed_sheeran_drive.mp3",
        _gh + "/f1_additional.mp3",
    ]
    # Part 1: audio element in main DOM (survives Streamlit reruns)
    st.markdown(
        '<div style="position:absolute;opacity:0;width:0;height:0;overflow:hidden">'
        '<audio id="f1audio"><source src="' + _songs[0] + '" type="audio/mpeg"></audio>'
        '</div>',
        unsafe_allow_html=True,
    )
    # Part 2: JS controller in iframe (not stripped by DOMPurify)
    _js_songs = ",".join('"' + s + '"' for s in _songs)
    components.html(
        '<script>'
        'var a=parent.document.getElementById("f1audio");'
        'var pl=[' + _js_songs + '];'
        'var si=0;'
        'if(a)a.volume=0.5;'
        'function pn(){si=(si+1)%pl.length;a.src=pl[si];a.load();a.play().catch(function(){}); }'
        'if(a){a.addEventListener("ended",pn);a.src=pl[0];a.load();a.play().catch(function(){}); }'
        'document.addEventListener("click",function h(){var x=parent.document.getElementById("f1audio");'
        'if(x&&x.paused)x.play().catch(function(){});document.removeEventListener("click",h);});'
        '</script>',
        height=0
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


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown(
    "<div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.3rem;'>"
    "<span style='background:var(--f1-red);width:4px;height:1.6rem;display:inline-block;border-radius:2px;'></span>"
    "<h1 style='margin:0;font-size:1.7rem;letter-spacing:1.5px;'>"
    "<span style='color:var(--f1-red);'>F1</span>"
    "<span style='color:var(--text-primary);'> Race Intelligence</span></h1>"
    "<span style='color:var(--text-muted);font-size:0.72rem;font-weight:400;margin-left:0.3rem;'>"
    "Predict · Simulate · Optimize</span></div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════

TAB_NAMES = ["RESULTS", "STANDINGS", "RACE TIMELINE", "DRIVER BATTLE",
             "TRACK ANALYSIS", "STINT TELEMETRY", "CAR TELEMETRY", "STRATEGY",
             "SC SIMULATOR", "PIT ANALYSIS", "AI ASSISTANT"]
active_tab = st.radio("tab_nav", TAB_NAMES, horizontal=True, label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════
# TAB 1 — STRATEGY (existing refactored)
# ══════════════════════════════════════════════════════════════════

if active_tab == "STRATEGY":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sd1 = st.selectbox("Driver", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"),
                           format_func=lambda d: f"{d}  ·  {_team_name(d)}",
                           key="s_driver")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-top:4px;padding-left:4px;'>"
            f"<span class='driver-dot' style='color:{_team_color(sd1)};'></span>"
            f"<span style='color:var(--text-primary);font-weight:700;font-size:0.85rem;'>{sd1}</span>"
            f"<span style='color:var(--text-dim);font-size:0.7rem;'>{_team_name(sd1)}</span></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st1 = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="s_track")
    with c3:
        sl1 = st.number_input("Race Laps", 10, 80, 52, step=1, key="s_laps")
    with c4:
        sm1 = st.number_input("Simulations", 10, 500, 30, step=10, key="s_mc")

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    sc_col, dnf_col, btn_col = st.columns([1, 1, 1.5])
    with sc_col:
        sc_prob = st.slider("Safety Car Probability", 0.0, 0.5, 0.20, 0.05)
    with dnf_col:
        dnf_prob = st.slider("DNF Probability", 0.0, 0.3, 0.05, 0.01, format="%.2f")
    with btn_col:
        st.markdown("<div style='padding-top:1.2rem;'>", unsafe_allow_html=True)
        run_btn = st.button("RUN OPTIMIZATION", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if run_btn:
        with st.spinner(f"Running {sm1} simulations across strategies..."):
            t0 = time.time()
            results = run_opt(st1, sl1, sd1, sm1, sc_prob, dnf_prob)
            elapsed = time.time() - t0
        if not results:
            st.warning("No valid strategies found.")
        else:
            ci = opt.circuit_info.get(st1, {})
            info = []
            if ci.get("Length_km"): info.append(f"<b>{ci['Length_km']:.1f}</b><span style='color:var(--text-dim);font-size:0.65rem;'> km</span>")
            if ci.get("Corners"): info.append(f"<b>{int(ci['Corners'])}</b><span style='color:var(--text-dim);font-size:0.65rem;'> corners</span>")
            if ci.get("AvgSpeed"): info.append(f"<b>{ci['AvgSpeed']:.0f}</b><span style='color:var(--text-dim);font-size:0.65rem;'> km/h avg</span>")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.8rem;padding:0.7rem 1rem;"
                f"background:linear-gradient(135deg,var(--bg-card),var(--bg-elevated));"
                f"border:1px solid var(--border-subtle);border-radius:6px;margin:0.5rem 0;'>"
                f"<span class='team-dot' style='color:{_team_color(sd1)};'></span>"
                f"<b style='color:var(--text-primary);'>{sd1}</b> {_team_name(sd1)}"
                f"<span style='color:var(--text-muted);'>·</span> {st1}"
                f"<span style='color:var(--text-muted);'>·</span> <span style='font-family:var(--font-mono);'>{sl1} laps</span>"
                f"<span style='color:var(--text-muted);'>·</span> {' '.join(info)}"
                f"<div style='margin-left:auto;color:var(--text-muted);font-size:0.7rem;font-family:var(--font-mono);'>{elapsed:.1f}s</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='section-label'>Top Strategies</div>", unsafe_allow_html=True)
            best_time = results[0]["mean_time"]
            for i, r in enumerate(results[:6]):
                parts = " <span style='color:var(--text-muted);font-size:0.75rem;'>→</span> ".join(
                    [f"{_compound_badge(c)} <span style='font-family:var(--font-mono);'>{l}l</span>" for c, l in r["strategy"]]
                )
                diff = r["mean_time"] - best_time
                st.markdown(
                    f"<div class='sr'><span class='sr-rank'>#{i+1}</span>"
                    f"<span class='sr-strat'>{parts}</span>"
                    f"<span class='sr-time'>{r['mean_time']:.1f}s</span>"
                    f"<span class='sr-std'>+{diff:.2f}</span></div>", unsafe_allow_html=True)

            st.markdown("<div class='section-label'>Race Time Comparison</div>", unsafe_allow_html=True)
            labels = [" → ".join([f"{c[:3]}-{l}" for c, l in r["strategy"]]) for r in results[:8]]
            means = [r["mean_time"] / 60 for r in results[:8]]
            stds = [r["std_time"] / 60 for r in results[:8]]
            fig, ax = plt.subplots(figsize=(12, 3.2))
            fig.patch.set_facecolor("none")
            bars = ax.barh(range(len(labels))[::-1], means, xerr=stds,
                           color=_team_color(sd1), capsize=2, height=0.45)
            if bars:
                bars[0].set_color(F1_RED)
            ax.set_yticks(range(len(labels))[::-1])
            ax.set_yticklabels(labels, fontsize=7.5, color="#888")
            style_ax(ax, "minutes")
            ax.margins(y=0.15)
            st.pyplot(fig, clear_figure=True)

            st.markdown("<div class='section-label'>Stint Degradation</div>", unsafe_allow_html=True)
            best_strat = results[0]["strategy"]
            run = run_detailed(st1, sl1, sd1, tuple((c, l) for c, l in best_strat))
            if run and run.get("stint_details"):
                fig2, ax2 = plt.subplots(figsize=(12, 3.2))
                fig2.patch.set_facecolor("none")
                lg = 1
                for s in run["stint_details"]:
                    xs = list(range(lg, lg + s["laps"]))
                    c = COMPOUND_COLORS.get(s["compound"], "#fff")
                    ax2.plot(xs, s["lap_times"], color=c, linewidth=1.5, marker=".", markersize=3)
                    if len(s["lap_times"]) > 1:
                        z = np.polyfit(range(len(s["lap_times"])), s["lap_times"], 1)
                        ax2.plot(xs, np.poly1d(z)(range(len(s["lap_times"]))), color=c, linestyle="--", alpha=0.3)
                    lg += s["laps"]
                pit_x = []
                for idx in range(len(best_strat) - 1):
                    pit_x.append(sum(sl for _, sl in best_strat[:idx + 1]))
                for px in pit_x:
                    ax2.axvline(x=px, color=F1_RED, linestyle=":", alpha=0.3)
                if pit_x:
                    ax2.annotate("PIT", (pit_x[0], ax2.get_ylim()[1] * 0.95), color=F1_RED, ha="center", fontsize=6.5)
                style_ax(ax2, "Lap", "Lap Time (s)")
                style_legend(ax2)
                st.pyplot(fig2, clear_figure=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — DRIVER BATTLE (existing, preserved)
# ══════════════════════════════════════════════════════════════════

elif active_tab == "DRIVER BATTLE":
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        d1 = st.selectbox("Driver 1", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"), key="d1",
                          format_func=lambda d: f"{d}  ·  {_team_name(d)}")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-top:4px;padding-left:4px;'>"
            f"<span class='driver-dot' style='color:{_team_color(d1)};'></span>"
            f"<span style='color:var(--text-primary);font-weight:700;font-size:0.85rem;'>{d1}</span>"
            f"<span style='color:var(--text-dim);font-size:0.7rem;'>{_team_name(d1)}</span></div>",
            unsafe_allow_html=True,
        )
    with col_d2:
        d2 = st.selectbox("Driver 2", DRIVERS_LIST, index=DRIVERS_LIST.index("HAM"), key="d2",
                          format_func=lambda d: f"{d}  ·  {_team_name(d)}")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-top:4px;padding-left:4px;'>"
            f"<span class='driver-dot' style='color:{_team_color(d2)};'></span>"
            f"<span style='color:var(--text-primary);font-weight:700;font-size:0.85rem;'>{d2}</span>"
            f"<span style='color:var(--text-dim);font-size:0.7rem;'>{_team_name(d2)}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)
    if d1 == d2:
        st.info("Select two different drivers.")
    else:
        col_track, col_laps, col_sims, col_dnf = st.columns(4)
        with col_track:
            tc = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="bc")
        with col_laps:
            lc = st.number_input("Laps", 10, 80, 52, key="bl")
        with col_sims:
            mc = st.number_input("Sims", 10, 500, 30, step=10, key="bm")
        with col_dnf:
            dc = st.slider("DNF", 0.0, 0.3, 0.05, 0.01, format="%.2f", key="bd")

        if st.button("COMPARE DRIVERS", type="primary"):
            with st.spinner("Running..."):
                r1 = run_opt(tc, lc, d1, mc, 0.2, dc)
                r2 = run_opt(tc, lc, d2, mc, 0.2, dc)
            if not r1 or not r2:
                st.warning("No results.")
            else:
                b1, b2 = r1[0], r2[0]
                diff = b1["mean_time"] - b2["mean_time"]
                winner, loser = (d1, d2) if diff < 0 else (d2, d1)
                # Fresh side-by-side columns for the comparison cards
                res_d1, res_d2 = st.columns(2)
                for d, res, col in [(d1, b1, res_d1), (d2, b2, res_d2)]:
                    strat = " → ".join([f"{c} ({l}l)" for c, l in res["strategy"]])
                    mins, secs = divmod(int(res["mean_time"]), 60)
                    col.markdown(
                        f"<div class='card'><div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;'>"
                        f"<span class='team-dot' style='color:{_team_color(d)};'></span>"
                        f"<b style='color:var(--text-primary);'>{d}</b> <span style='color:var(--text-dim);font-size:0.7rem;'>{_team_name(d)}</span></div>"
                        f"<div style='color:var(--text-secondary);font-size:0.78rem;'>{strat}</div>"
                        f"<div style='margin-top:0.5rem;'><span style='font-size:1.15rem;font-weight:700;font-family:var(--font-mono);color:var(--text-primary);'>{res['mean_time']:.1f}s</span>"
                        f"<span style='color:var(--text-dim);font-size:0.75rem;margin-left:0.5rem;font-family:var(--font-mono);'>({mins}:{secs:02d}) ±{res['std_time']:.2f}s</span></div></div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(f"<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='winner-badge'>🏆 {_driver_tag(winner)} beats {_driver_tag(loser)} "
                    f"by <b style='color:var(--f1-red);font-family:var(--font-mono);'>{abs(diff):.1f}s</b></div>",
                    unsafe_allow_html=True,
                )
                run1 = run_detailed(tc, lc, d1, tuple((c, l) for c, l in b1["strategy"]))
                run2 = run_detailed(tc, lc, d2, tuple((c, l) for c, l in b2["strategy"]))
                if run1 and run2:
                    fig, ax = plt.subplots(figsize=(12, 3.2))
                    fig.patch.set_facecolor("none")
                    for d, run, sty, mk in [(d1, run1, "-", "o"), (d2, run2, "--", "s")]:
                        lg = 1
                        for s in run["stint_details"]:
                            xs = list(range(lg, lg + s["laps"]))
                            c = COMPOUND_COLORS.get(s["compound"], "#fff")
                            ax.plot(xs, s["lap_times"], color=c, linestyle=sty, linewidth=1.5, marker=mk, markersize=2, label=f"{d} {s['compound']}")
                            lg += s["laps"]
                    style_ax(ax, "Lap", "Lap Time (s)")
                    style_legend(ax)
                    st.pyplot(fig, clear_figure=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — STINT TELEMETRY (enhanced stint sim)
# ══════════════════════════════════════════════════════════════════

elif active_tab == "STINT TELEMETRY":
    ca, cb, cc, cd = st.columns(4)
    with ca:
        sd3 = st.selectbox("Driver", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"), key="s3",
                           format_func=lambda d: f"{d}  ·  {_team_name(d)}")
    with cb:
        sc3 = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"])
    with cc:
        sl3 = st.number_input("Stint Length", 5, 50, 20, key="s3l")
    with cd:
        st3 = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="s3t")

    st.markdown("<br>", unsafe_allow_html=True)

    stint_run = run_detailed(st3, sl3, sd3, ((sc3, sl3),))
    if stint_run and stint_run.get("stint_details"):
        sd = stint_run["stint_details"][0]
        times = sd["lap_times"]
        laps_arr = list(range(1, sl3 + 1))
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Lap", f"{sd['avg_time']:.3f}s")
        c2.metric("Degradation", f"{times[-1] - times[0]:+.3f}s")
        c3.metric("First Lap", f"{times[0]:.3f}s")

        # Main plot: lap times with fuel & deg breakdown
        fig, ax = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={"height_ratios": [2, 1], "hspace": 0.35})
        fig.patch.set_facecolor("none")
        tc = COMPOUND_COLORS.get(sc3, F1_RED)
        ax[0].plot(laps_arr, times, color=tc, linewidth=2, marker="o", markersize=3)
        if len(times) > 2:
            z = np.polyfit(laps_arr, times, 2)
            ax[0].plot(laps_arr, np.poly1d(z)(laps_arr), linestyle="--", color="#555", alpha=0.4, label="Quadratic trend")
        style_ax(ax[0], "", "Lap Time (s)")
        style_legend(ax[0])

        # Fuel delta overlay
        fuel_deltas = [fuel_effect(lap, sl3) for lap in laps_arr]
        ax[1].fill_between(laps_arr, fuel_deltas, alpha=0.3, color="#3793ff")
        ax[1].plot(laps_arr, fuel_deltas, color="#3793ff", linewidth=1.5)
        style_ax(ax[1], "Lap", "Fuel delta (s)")
        st.pyplot(fig, clear_figure=True)

        st.markdown("<div class='section-label'>Lap Times</div>", unsafe_allow_html=True)
        tbl = pd.DataFrame({
            "Lap": laps_arr,
            "Time": [f"{t:.3f}s" for t in times],
            "Delta": [f"{(t - times[0]):+.3f}s" for t in times],
            "Fuel": [f"{fuel_effect(l, sl3):+.3f}s" for l in laps_arr],
        })
        st.dataframe(tbl, hide_index=True)
    else:
        st.info("Run a stint to see data.")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — TRACK ANALYSIS (existing, preserved with weather viz)
# ══════════════════════════════════════════════════════════════════

elif active_tab == "TRACK ANALYSIS":
    ca, cb, cc = st.columns(3)
    with ca:
        ta_track = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="ta_track")
    with cb:
        ta_year = st.selectbox("Year", ["All", "2026", "2025"], key="ta_year")
    with cc:
        ta_compound = st.selectbox("Compound", ["All", "SOFT", "MEDIUM", "HARD"], key="ta_compound")

    st.markdown("<br>", unsafe_allow_html=True)

    df_raw = pd.read_csv(os.path.join(_BASE, "data", "all_races_master.csv"))
    df_track = df_raw[df_raw["Race"] == ta_track].copy()
    if ta_year != "All": df_track = df_track[df_track["Year"] == int(ta_year)]
    if ta_compound != "All": df_track = df_track[df_track["Compound"] == ta_compound]
    if df_track.empty:
        st.info("No data for these filters.")
    else:
        ci_t4 = opt.circuit_info.get(ta_track, {})
        t4_info = []
        if ci_t4.get("Length_km"): t4_info.append(f"<span class='data-tag'>Length <b>{ci_t4['Length_km']:.1f} km</b></span>")
        if ci_t4.get("Corners"): t4_info.append(f"<span class='data-tag'>Corners <b>{int(ci_t4['Corners'])}</b></span>")
        if ci_t4.get("AvgSpeed"): t4_info.append(f"<span class='data-tag'>Avg Speed <b>{ci_t4['AvgSpeed']:.0f} km/h</b></span>")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem;flex-wrap:wrap;'>"
            f"<span style='color:var(--text-primary);font-weight:600;font-size:1rem;'>{ta_track}</span>{' '.join(t4_info)}</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Laps", len(df_track))
        c2.metric("Drivers", df_track["Driver"].nunique())
        c3.metric("Avg Lap", f"{df_track['LapTime'].mean():.3f}s")
        c4.metric("Median Lap", f"{df_track['LapTime'].median():.3f}s")
        yrs = sorted(df_track["Year"].unique())
        c5.metric("Year(s)", "+".join(str(int(y)) for y in yrs) if len(yrs) <= 2 else f"{int(min(yrs))}-{int(max(yrs))}")

        st.markdown("<div class='section-label'>Lap Time by Compound</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 3.2))
        fig.patch.set_facecolor("none")
        for cpd in ["SOFT", "MEDIUM", "HARD"]:
            sub = df_raw[(df_raw["Race"] == ta_track) & (df_raw["Compound"] == cpd)]
            if not sub.empty:
                ax.hist(sub["LapTime"], bins=40, alpha=0.45, color=COMPOUND_COLORS.get(cpd, "#888"), label=cpd, density=True)
        style_ax(ax, "Lap Time (s)", "Density")
        style_legend(ax)
        st.pyplot(fig, clear_figure=True)

        st.markdown("<div class='section-label'>Weather Impact (Air Temp vs Lap Time)</div>", unsafe_allow_html=True)
        if "AirTemp" in df_track.columns:
            fig2, ax2 = plt.subplots(figsize=(10, 3.2))
            fig2.patch.set_facecolor("none")
            for cpd in ["SOFT", "MEDIUM", "HARD"]:
                sub = df_track[df_track["Compound"] == cpd]
                if not sub.empty and sub["AirTemp"].nunique() > 2:
                    ax2.scatter(sub["AirTemp"], sub["LapTime"], alpha=0.3, s=4, color=COMPOUND_COLORS.get(cpd, "#888"), label=cpd)
            style_ax(ax2, "Air Temp (°C)", "Lap Time (s)")
            style_legend(ax2)
            st.pyplot(fig2, clear_figure=True)

        st.markdown("<div class='section-label'>Tyre Degradation</div>", unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(10, 3.2))
        fig3.patch.set_facecolor("none")
        df_trk = df_raw[df_raw["Race"] == ta_track]
        for cpd in ["SOFT", "MEDIUM", "HARD"]:
            sub = df_trk[df_trk["Compound"] == cpd]
            if not sub.empty and sub["TyreLife"].nunique() > 2:
                deg = sub.groupby("TyreLife")["LapTime"].mean().reset_index()
                ax3.plot(deg["TyreLife"], deg["LapTime"], color=COMPOUND_COLORS.get(cpd, "#888"), linewidth=2, label=cpd, marker="o", markersize=3)
        style_ax(ax3, "Tyre Life (laps)", "Avg Lap Time (s)")
        style_legend(ax3)
        st.pyplot(fig3, clear_figure=True)

        st.markdown("<div class='section-label'>Driver Ranking</div>", unsafe_allow_html=True)
        driver_avg = df_trk.groupby("Driver")["LapTime"].mean().sort_values()
        n = len(driver_avg)
        fig4, ax4 = plt.subplots(figsize=(10, max(3.2, n * 0.3)))
        fig4.patch.set_facecolor("none")
        cols = [_team_color(d) for d in driver_avg.index]
        ax4.barh(range(n), driver_avg.values, color=cols, height=0.65)
        ax4.set_yticks(range(n))
        ax4.set_yticklabels(driver_avg.index, color="#aaa", fontsize=7.5)
        for sp in ["top", "right", "bottom"]:
            ax4.spines[sp].set_visible(False)
        ax4.spines["left"].set_color("#222")
        ax4.tick_params(colors="#444", labelsize=7)
        ax4.set_xlabel("Avg Lap Time (s)", color="#555", fontsize=8)
        for i, v in enumerate(driver_avg.values):
            ax4.text(v + 0.02, i, f"{v:.3f}s", color="#777", fontsize=6.5, va="center")
        st.pyplot(fig4, clear_figure=True)

        if df_trk["Year"].nunique() > 1:
            st.markdown("<div class='section-label'>Year-over-Year</div>", unsafe_allow_html=True)
            fig5, ax5 = plt.subplots(figsize=(10, 3.2))
            fig5.patch.set_facecolor("none")
            for yr in sorted(df_trk["Year"].unique()):
                sub = df_trk[df_trk["Year"] == yr]
                d = sub.groupby("Driver")["LapTime"].mean().reindex(driver_avg.index)
                ax5.plot(d.values, range(len(d)), marker="o", markersize=4, linewidth=1.5, label=str(int(yr)))
            style_ax(ax5, "Avg Lap Time (s)", "Driver")
            ax5.set_yticks(range(len(driver_avg)))
            ax5.set_yticklabels(driver_avg.index, color="#aaa", fontsize=7)
            style_legend(ax5)
            st.pyplot(fig5, clear_figure=True)

        # Traffic effect simulation
        st.markdown("<div class='section-label'>Traffic Impact (Dirty Air Simulation)</div>", unsafe_allow_html=True)
        avg_lt = df_trk["LapTime"].mean()
        pos = list(range(1, 13))
        traffic_losses = [max(0, 0.08 * (p - 1) * 0.5) for p in pos]  # simple model
        fig6, ax6 = plt.subplots(figsize=(10, 2.5))
        fig6.patch.set_facecolor("none")
        ax6.fill_between(pos, [avg_lt + t for t in traffic_losses], avg_lt, alpha=0.2, color=F1_RED)
        ax6.plot(pos, [avg_lt + t for t in traffic_losses], color=F1_RED, linewidth=2, marker="o", markersize=5)
        for p, t in zip(pos, traffic_losses):
            ax6.annotate(f"+{t:.2f}s", (p, avg_lt + t), textcoords="offset points", xytext=(0, 5),
                         ha="center", color="#777", fontsize=7)
        style_ax(ax6, "Position", "Estimated Lap Time (s)")
        st.pyplot(fig6, clear_figure=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5 — SC SIMULATOR
# ══════════════════════════════════════════════════════════════════

elif active_tab == "SC SIMULATOR":
    ca, cb, cc, cd = st.columns(4)
    with ca:
        sc_driver = st.selectbox("Driver", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"), key="sc_drv",
                                 format_func=lambda d: f"{d}  ·  {_team_name(d)}")
    with cb:
        sc_track = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="sc_trk")
    with cc:
        sc_total = st.number_input("Total Laps", 30, 80, 52, key="sc_tot")
    with cd:
        sc_lap = st.number_input("SC Deployed on Lap", 3, 78, 14, key="sc_lap_num")

    ca, cb = st.columns(2)
    with ca:
        sc_dur = st.slider("SC Duration (laps)", 1, 6, 3)
    with cb:
        sc_free = st.checkbox("Free Pit Under SC", True)

    sc_show = st.button("RUN SC SIMULATION", type="primary")

    if sc_show:
        base_lt = base_lap
        res = simulate_sc_scenario(
            sc_lap, sc_total, base_lt,
            sc_duration=sc_dur, sc_free_pit=sc_free,
        )
        saved = res["time_saved"]
        verdict = "GAIN" if saved > 0 else "LOSS"

        st.markdown(
            f"<div class='card' style='display:flex;gap:2rem;flex-wrap:wrap;'>"
            f"<div><span style='color:var(--text-dim);font-size:0.6rem;'>GREEN FLAG TOTAL</span><br>"
            f"<span style='font-family:var(--font-mono);font-size:1.3rem;color:var(--text-primary);'>{res['base_total']}s</span></div>"
            f"<div><span style='color:var(--text-dim);font-size:0.6rem;'>SC TOTAL</span><br>"
            f"<span style='font-family:var(--font-mono);font-size:1.3rem;color:var(--text-primary);'>{res['sc_total']}s</span></div>"
            f"<div><span style='color:var(--text-dim);font-size:0.6rem;'>DELTA</span><br>"
            f"<span style='font-family:var(--font-mono);font-size:1.3rem;color:var(--f1-red);'>{saved:+.1f}s</span></div>"
            f"</div>", unsafe_allow_html=True,
        )

        adv = []
        if res["free_pit"] and saved > 0:
            adv.append("✅ **Pitting under SC is advantageous.** Reduced pit loss (12s vs 22s) + field bunches. Best move: pit immediately for fresh tyres.")
        elif saved < -5:
            adv.append("⚠️ **SC hurts your race** — you lose the gap you built. Stay out if track position matters.")
        else:
            adv.append("⏱️ SC has minimal net effect. Consider matching opponents' strategy.")

        adv.append(f"🚨 Lap {sc_lap} | {sc_dur}-lap SC | {'Free pit available' if sc_free else 'No free pit'}")
        st.markdown("  \n".join(adv))

        # Visual timeline
        fig, ax = plt.subplots(figsize=(12, 2.5))
        fig.patch.set_facecolor("none")
        laps_x = list(range(1, sc_total + 1))
        base_curve = [base_lt + 0.03 * (l - 1) * 0.02 for l in laps_x]  # simplified baseline
        sc_curve = []
        for l in laps_x:
            if sc_lap <= l < sc_lap + sc_dur:
                sc_curve.append(base_lt + 3.0)
            else:
                sc_curve.append(base_lt + 0.03 * (l - 1) * 0.02)
        ax.plot(laps_x, base_curve, color="#555", linewidth=1, alpha=0.5, label="Green flag")
        ax.plot(laps_x, sc_curve, color=F1_RED, linewidth=1.5, label="SC scenario")
        ax.axvspan(sc_lap, sc_lap + sc_dur - 0.5, alpha=0.1, color=F1_RED, label="SC period")
        style_ax(ax, "Lap", "Lap Time (s)")
        style_legend(ax)
        st.pyplot(fig, clear_figure=True)


# ══════════════════════════════════════════════════════════════════
# TAB 6 — PIT ANALYSIS
# ══════════════════════════════════════════════════════════════════

elif active_tab == "PIT ANALYSIS":
    ca, cb, cc, cd = st.columns(4)
    with ca:
        u_driver = st.selectbox("Driver", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"), key="u_drv")
    with cb:
        u_track = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="u_trk")
    with cc:
        u_laps = st.number_input("Total Laps", 30, 80, 52, key="u_lap")
    with cd:
        u_pit_lap = st.number_input("Planned Pit Lap", 5, 75, 18, key="u_pit")

    ca, cb, cc = st.columns(3)
    with ca:
        u_tyre_age = st.number_input("Current Tyre Age (laps)", 0, 30, 8, key="u_age")
    with cb:
        u_compound = st.selectbox("Fresh Compound", ["SOFT", "MEDIUM", "HARD"], key="u_cpd")
    with cc:
        u_gap = st.slider("Gap to Car Ahead (s)", 0.0, 5.0, 2.0, 0.1, key="u_gap")

    if st.button("ANALYZE PIT", type="primary"):
        uc = undercut_benefit(u_pit_lap, u_laps, u_tyre_age, u_compound, pit_loss=PIT_LOSS_DEFAULT)
        effective_gap = u_gap + PIT_LOSS_DEFAULT - uc["gain_from_fresh"]
        success = effective_gap < 0

        st.markdown(
            f"<div class='card' style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;'>"
            f"<div><span style='color:var(--text-dim);font-size:0.6rem;'>PIT LOSS</span><br>"
            f"<span style='font-family:var(--font-mono);font-size:1rem;color:var(--text-primary);'>{PIT_LOSS_DEFAULT:.1f}s</span></div>"
            f"<div><span style='color:var(--text-dim);font-size:0.6rem;'>FRESH TYRE GAIN (5 laps)</span><br>"
            f"<span style='font-family:var(--font-mono);font-size:1rem;color:var(--text-primary);'>{uc['gain_from_fresh']:+.3f}s</span></div>"
            f"<div><span style='color:var(--text-dim);font-size:0.6rem;'>CROSSOVER LAP</span><br>"
            f"<span style='font-family:var(--font-mono);font-size:1rem;color:var(--text-primary);'>#{uc['crossover_lap']}</span></div>"
            f"<div><span style='color:var(--text-dim);font-size:0.6rem;'>EFFECTIVE GAP OUT</span><br>"
            f"<span style='font-family:var(--font-mono);font-size:1rem;color:{'#00e701' if success else F1_RED};'>{effective_gap:+.2f}s</span></div>"
            f"</div>", unsafe_allow_html=True,
        )

        if success:
            st.success(f"✅ **Undercut succeeds!** You come out ~{abs(effective_gap):.1f}s ahead of the car in front.")
        else:
            st.warning(f"❌ **Undercut fails.** You'd be ~{effective_gap:.1f}s behind after the stop. "
                       "Try pitting later or using a softer compound.")

        # Optimal pit window
        st.markdown("<div class='section-label'>Optimal Pit Window</div>", unsafe_allow_html=True)
        opt_windows = ua.find_optimal_pit_window(u_laps, u_compound, u_laps - u_pit_lap, track_temp=35.0)
        if opt_windows:
            windows_df = pd.DataFrame(opt_windows[:8])
            windows_df.columns = ["Pit Lap", "Penalty (s)"]
            st.dataframe(windows_df, hide_index=True)

    # Strategy comparison
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Strategy vs Strategy Battle</div>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("<span style='color:var(--text-dim);font-size:0.6rem;'>DRIVER A</span>", unsafe_allow_html=True)
        sa_cpd = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], key="sa_cpd")
        sa_pits = st.text_input("Pit laps (comma-separated)", "18, 36")
    with cb:
        st.markdown("<span style='color:var(--text-dim);font-size:0.6rem;'>DRIVER B</span>", unsafe_allow_html=True)
        sb_cpd = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], key="sb_cpd")
        sb_pits = st.text_input("Pit laps (comma-separated)", "22")

    if st.button("COMPARE STRATEGIES", type="primary", key="cmp_strat"):
        try:
            a_laps = [int(x.strip()) for x in sa_pits.split(",")]
            b_laps = [int(x.strip()) for x in sb_pits.split(",")]
            cmp_result = ua.compare_strategies(sa_cpd, a_laps, sb_cpd, b_laps, u_laps)
            st.markdown(
                f"<div style='display:flex;gap:2rem;padding:0.5rem;'>"
                f"<div><b style='color:var(--text-primary);'>A ({sa_cpd})</b> pit @ {', '.join(map(str, a_laps))}</div>"
                f"<div><b style='color:var(--text-primary);'>B ({sb_cpd})</b> pit @ {', '.join(map(str, b_laps))}</div>"
                f"<div style='margin-left:auto;'><b style='color:{F1_RED};'>Δ {cmp_result['final_delta']:.2f}s</b>"
                f"{' (A ahead)' if cmp_result['final_delta'] < 0 else ' (B ahead)'}</div></div>",
                unsafe_allow_html=True,
            )
            events = cmp_result["events"]
            fig, ax = plt.subplots(figsize=(12, 2.5))
            fig.patch.set_facecolor("none")
            laps_e = [e["lap"] for e in events]
            deltas = [e["delta_a_to_b"] for e in events]
            ax.plot(laps_e, deltas, color=F1_RED, linewidth=2, marker="s", markersize=4)
            ax.axhline(y=0, color="#444", linewidth=0.5, linestyle="--")
            for e in events:
                if e["a_pitted"]:
                    ax.axvline(x=e["lap"], color=_team_color(d1) if 'd1' in dir() else F1_RED, linestyle=":", alpha=0.3)
                if e["b_pitted"]:
                    ax.axvline(x=e["lap"], color=_team_color(d2) if 'd2' in dir() else "#3793ff", linestyle=":", alpha=0.3)
            style_ax(ax, "Lap", "Δ A−B (s)")
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Parse error: {e}. Use format: 18, 36")


# ══════════════════════════════════════════════════════════════════
# TAB 7 — CAR TELEMETRY
# ══════════════════════════════════════════════════════════════════

elif active_tab == "CAR TELEMETRY":
    ca, cb, cc, cd = st.columns(4)
    with ca:
        telem_year = st.selectbox("Year", [2026, 2025], key="telem_yr")
    with cb:
        telem_track = st.selectbox("Track", tracks,
                                   index=tracks.index("British Grand Prix"), key="telem_trk")
    with cc:
        telem_d1 = st.selectbox("Driver 1", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"), key="telem_d1")
    with cd:
        telem_d2 = st.selectbox("Driver 2", DRIVERS_LIST, index=DRIVERS_LIST.index("HAM"), key="telem_d2")

    if st.button("LOAD TELEMETRY", type="primary", key="telem_btn"):
        with st.spinner("Loading telemetry from fastf1 (cached after first load)..."):
            from telemetry_loader import (
                resolve_session, get_driver_lap_telemetry,
                get_driver_sector_times, plot_telemetry_comparison,
                plot_sector_comparison, get_session_weather,
            )
            session = resolve_session(telem_year, telem_track)
            if session is None:
                st.error(f"Could not load {telem_track} {telem_year}. "
                         "The race may not be available yet or fastf1 may not have data.")
            else:
                tel1 = get_driver_lap_telemetry(session, telem_d1, fastest_only=True)
                tel2 = get_driver_lap_telemetry(session, telem_d2, fastest_only=True)
                sec1 = get_driver_sector_times(session, telem_d1)
                sec2 = get_driver_sector_times(session, telem_d2)

                if tel1 is None and tel2 is None:
                    st.warning("No telemetry available for these drivers.")
                else:
                    fig = plot_telemetry_comparison(tel1, tel2, telem_d1, telem_d2)
                    if fig:
                        st.pyplot(fig, clear_figure=True)
                    else:
                        st.info("Telemetry plot not available.")

                    if sec1 or sec2:
                        st.markdown("<div class='section-label'>Sector Time Comparison</div>", unsafe_allow_html=True)
                        fig2 = plot_sector_comparison(sec1, sec2, telem_d1, telem_d2)
                        if fig2:
                            st.pyplot(fig2, clear_figure=True)

                    # Weather data from session
                    wx = get_session_weather(session)
                    if wx:
                        st.markdown("<div class='section-label'>Session Weather</div>", unsafe_allow_html=True)
                        wdf = pd.DataFrame(wx)
                        wdf["time"] = wdf["time"].astype(str)
                        st.dataframe(wdf, hide_index=True)

                    st.info("💡 Tip: First load may take 30-60s (downloading from fastf1). "
                            "Subsequent loads are instant from cache.")


# ══════════════════════════════════════════════════════════════════
# TAB 8 — AI STRATEGY ASSISTANT
# ══════════════════════════════════════════════════════════════════

elif active_tab == "AI ASSISTANT":
    ca, cb, cc, cd = st.columns(4)
    with ca:
        ai_driver = st.selectbox("Driver", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"), key="ai_drv")
    with cb:
        ai_track = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="ai_trk")
    with cc:
        ai_laps = st.number_input("Total Laps", 30, 80, 52, key="ai_lap")
    with cd:
        ai_current = st.number_input("Current Lap", 1, 79, 15, key="ai_cur")

    ca, cb, cc = st.columns(3)
    with ca:
        ai_compound = st.selectbox("Current Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"], key="ai_cpd")
    with cb:
        ai_tyre_age = st.number_input("Tyre Age (laps)", 0, 30, 10, key="ai_age")
    with cc:
        ai_temp = st.slider("Track Temp (°C)", 15, 55, 35, key="ai_temp")

    # Quick prompts
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:0.5rem;margin:0.5rem 0;'>"
        "<span style='color:var(--text-dim);font-size:0.65rem;letter-spacing:0.5px;'>QUICK QUESTIONS:</span></div>",
        unsafe_allow_html=True,
    )

    q_cols = st.columns(4)
    quick_qs = [
        "Should I pit this lap?",
        "What's the fastest strategy?",
        "What if a Safety Car appears now?",
        "Which tyre should I use next?",
    ]
    ai_query = ""
    for i, (col, q) in enumerate(zip(q_cols, quick_qs)):
        with col:
            if st.button(q, key=f"qq_{i}", use_container_width=True):
                ai_query = q

    # Custom question
    custom_q = st.text_input(
        "Or type your own question:",
        placeholder="e.g. How much time will I lose if I stay out 3 more laps?",
        label_visibility="collapsed",
    )
    if custom_q:
        ai_query = custom_q

    if ai_query:
        with st.spinner("Analyzing..."):
            result = assistant.ask(
                ai_query, driver=ai_driver, track=ai_track,
                total_laps=ai_laps, current_lap=ai_current,
                current_compound=ai_compound, tyre_age=ai_tyre_age,
                track_temp=ai_temp,
            )
        st.markdown(
            f"<div class='card' style='white-space:pre-wrap;'>"
            f"<div style='color:var(--text-dim);font-size:0.6rem;letter-spacing:0.5px;margin-bottom:0.3rem;'>"
            f"{ai_driver} · {ai_track} · Lap {ai_current}/{ai_laps} · {ai_compound} (age {ai_tyre_age})</div>"
            f"{result}</div>",
            unsafe_allow_html=True,
        )

    # Example questions
    with st.expander("💡 Example questions you can ask"):
        st.markdown("""
- *Should I pit this lap?*
- *What's the fastest strategy for this race?*
- *What if a Safety Car appears on lap 14?*
- *How much time will I lose if I stay out 3 more laps?*
- *Which tyre should I use for the next stint?*
- *Simulate a 20-lap stint on SOFT*
- *What's the optimal pit window?*
""")

# ══════════════════════════════════════════════════════════════════
# TAB 9 — RACE TIMELINE
# ══════════════════════════════════════════════════════════════════

elif active_tab == "RACE TIMELINE":
    render_race_timeline(opt, tracks, DRIVERS_LIST, _team_name, COMPOUND_COLORS, plt, pd, st)

elif active_tab == "RESULTS":
    render_results_tab()

elif active_tab == "STANDINGS":
    render_standings_tab()
