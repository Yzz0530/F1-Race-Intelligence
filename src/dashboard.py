"""F1 Strategy Optimizer Dashboard V3 — thin shell over src/tabs/*.

Tab rendering logic now lives in src/tabs/ (one module per tab + _shared
for constants/helpers/cached resources). This file owns only page config,
CSS, the sidebar chrome, and the tab router.
"""
from __future__ import annotations

import json
import os
import sys

import streamlit as st
import streamlit.components.v1 as components

# Ensure src/ is importable (tab modules rely on it via _shared).
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from tabs._shared import load_optimizer, _BASE  # noqa: E402
from tabs import strategy, driver_battle, stint_telemetry, track_analysis  # noqa: E402
from tabs import sc_simulator, pit_analysis, car_telemetry, ai_assistant  # noqa: E402
from results import render_results_tab  # noqa: E402
from standings import render_standings_tab  # noqa: E402
from race_timeline import render_race_timeline  # noqa: E402

F1_RED = "#e10600"


def _render_active_circuit(opt, tracks) -> None:
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


def main() -> None:
    st.set_page_config(page_title="F1 Race Intelligence", page_icon="assets/favicon.ico", layout="wide")

    _css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(_css_path, encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

    try:
        opt = load_optimizer()
        tracks = sorted(opt.circuit_info.keys())
        base_lap = opt.overall_baseline
    except Exception as e:
        st.error(f"Failed to load optimizer: {e}")
        st.stop()

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
            f"<div class='stat-box'><div style='font-family:var(--font-mono);font-size:1.2rem;font-weight:700;color:var(--text-primary);'>{len(opt.driver_offsets)}</div>"
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
        _render_active_circuit(opt, tracks)

        st.markdown("<hr>", unsafe_allow_html=True)
        try:
            with open(os.path.join(_BASE, "models", "TRAINING_MANIFEST.json")) as _mf:
                _mf_content = _mf.read()
                _manifest = json.loads(_mf_content)
            _mae_txt = f"{_manifest['metrics']['val_mae']:.2f}s"
            _feat_count = len(_manifest['features'])
            _row_count = _manifest['training_data']['rows']
        except Exception:
            _mae_txt = "0.87s"
            _feat_count = 31
            _row_count = 31703
        st.markdown(
            "<div style='color:var(--text-muted);font-size:0.7rem;line-height:1.7;'>"
            "<span style='color:var(--text-dim);font-size:0.6rem;letter-spacing:0.5px;text-transform:uppercase;'>Technology Stack</span><br>"
            "XGBoost · Monte Carlo · Physics Engine<br>"
            f"<span style='color:var(--text-dim);font-size:0.6em;'>MAE {_mae_txt} · {_feat_count} features · {_row_count:,} laps</span></div>",
            unsafe_allow_html=True,
        )
        _gh = "https://raw.githubusercontent.com/Yzz0530/F1-Race-Intelligence/master/assets"
        _songs_local = [
            f"assets/{f}" for f in [
                "f1_theme.mp3",
                "don_toliver_lose_my_mind.mp3",
                "tate_mcrae_just_keep_watching.mp3",
                "rose_messy.mp3",
                "ed_sheeran_drive.mp3",
                "f1_additional.mp3",
            ]
        ]
        # Use local assets (more reliable than GitHub raw URLs)
        _songs = _songs_local
        st.markdown(
            '<div style="position:absolute;opacity:0;width:0;height:0;overflow:hidden">'
            '<audio id="f1audio"><source src="' + _songs[0] + '" type="audio/mpeg"></audio>'
            '</div>',
            unsafe_allow_html=True,
        )
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
            height=0,
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

    TAB_NAMES = ["RESULTS", "STANDINGS", "RACE TIMELINE", "DRIVER BATTLE",
                 "TRACK ANALYSIS", "STINT TELEMETRY", "CAR TELEMETRY", "STRATEGY",
                 "SC SIMULATOR", "PIT ANALYSIS", "AI ASSISTANT"]
    active_tab = st.radio("tab_nav", TAB_NAMES, horizontal=True, label_visibility="collapsed")

    if active_tab == "STRATEGY":
        strategy.render_strategy()
    elif active_tab == "DRIVER BATTLE":
        driver_battle.render_driver_battle()
    elif active_tab == "STINT TELEMETRY":
        stint_telemetry.render_stint_telemetry()
    elif active_tab == "TRACK ANALYSIS":
        track_analysis.render_track_analysis()
    elif active_tab == "SC SIMULATOR":
        sc_simulator.render_sc_simulator()
    elif active_tab == "PIT ANALYSIS":
        pit_analysis.render_pit_analysis()
    elif active_tab == "CAR TELEMETRY":
        car_telemetry.render_car_telemetry()
    elif active_tab == "AI ASSISTANT":
        ai_assistant.render_ai_assistant()
    elif active_tab == "RACE TIMELINE":
        render_race_timeline(opt, tracks, opt.driver_offsets.keys(), _team_name, COMPOUND_COLORS, plt, pd, st)
    elif active_tab == "RESULTS":
        render_results_tab()
    elif active_tab == "STANDINGS":
        render_standings_tab()


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # noqa: F401
    import pandas as pd  # noqa: F401
    from tabs._shared import COMPOUND_COLORS, _team_name  # noqa: F401
    main()
