"""PIT ANALYSIS tab — undercut / overcut analysis and strategy vs strategy."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ._shared import F1_RED, COMPOUND_COLORS, DRIVERS_LIST, PIT_LOSS_DEFAULT, _team_color, _team_name, undercut_benefit, style_ax, load_undercut, load_optimizer


def render_pit_analysis() -> None:
    ua = load_undercut()
    tracks = sorted(load_optimizer().circuit_info.keys())
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

        st.markdown("<div class='section-label'>Optimal Pit Window</div>", unsafe_allow_html=True)
        opt_windows = ua.find_optimal_pit_window(u_laps, u_compound, u_laps - u_pit_lap, track_temp=35.0)
        if opt_windows:
            windows_df = pd.DataFrame(opt_windows[:8])
            windows_df.columns = ["Pit Lap", "Total Time (s)", "Penalty vs Best (s)"]
            st.dataframe(windows_df, hide_index=True)

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
        import pandas as pd  # noqa: F401  (windows_df above also needs pd; import once)
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
                    ax.axvline(x=e["lap"], color=F1_RED, linestyle=":", alpha=0.3)
                if e["b_pitted"]:
                    ax.axvline(x=e["lap"], color="#3793ff", linestyle=":", alpha=0.3)
            style_ax(ax, "Lap", "Δ A−B (s)")
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Parse error: {e}. Use format: 18, 36")
