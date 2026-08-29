"""AI ASSISTANT tab — natural-language strategy Q&A (physics + ML blend)."""
from __future__ import annotations

import streamlit as st

from ._shared import DRIVERS_LIST, _team_color, _team_name, load_assistant


def render_ai_assistant() -> None:
    assistant = load_assistant()
    tracks = sorted(assistant.opt.circuit_info.keys())
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

    custom_q = st.text_input(
        "Or type your own question:",
        placeholder="e.g. How much time will I lose if I stay out 3 more laps?",
        label_visibility="collapsed",
    )
    if custom_q:
        ai_query = custom_q

    st.caption(
        "ℹ️ Strategy is a **physics + ML blend**. Pace, degradation and wet "
        "conditions come from the trained model; compound speed deltas "
        "(SOFT/MEDIUM/HARD) are a physics overlay, not a pure ML prediction. "
        "Treat outputs as engineering estimates, not certainty."
    )

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

    with st.expander("💡 Example questions you can ask"):
        st.markdown(
            """
- *Should I pit this lap?*
- *What's the fastest strategy for this race?*
- *What if a Safety Car appears on lap 14?*
- *How much time will I lose if I stay out 3 more laps?*
- *Which tyre should I use for the next stint?*
- *Simulate a 20-lap stint on SOFT*
- *What's the optimal pit window?*
"""
        )
