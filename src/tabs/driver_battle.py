"""DRIVER BATTLE tab — head-to-head driver comparison."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._shared import (
    F1_RED, COMPOUND_COLORS, DRIVERS_LIST, _team_color, _team_name,
    _driver_tag, style_ax, style_legend, load_optimizer, run_opt, run_detailed,
)


def render_driver_battle() -> None:
    opt = load_optimizer()
    tracks = sorted(opt.circuit_info.keys())
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
