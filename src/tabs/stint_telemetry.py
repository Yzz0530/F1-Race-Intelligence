"""STINT TELEMETRY tab — stint-level lap-time / fuel / degradation view."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._shared import F1_RED, COMPOUND_COLORS, DRIVERS_LIST, _team_color, _team_name, load_optimizer, run_detailed, fuel_effect


def render_stint_telemetry() -> None:
    opt = load_optimizer()
    tracks = sorted(opt.circuit_info.keys())
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

        fig, ax = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={"height_ratios": [2, 1], "hspace": 0.35})
        fig.patch.set_facecolor("none")
        tc = COMPOUND_COLORS.get(sc3, F1_RED)
        ax[0].plot(laps_arr, times, color=tc, linewidth=2, marker="o", markersize=3)
        if len(times) > 2:
            z = np.polyfit(laps_arr, times, 2)
            ax[0].plot(laps_arr, np.poly1d(z)(laps_arr), linestyle="--", color="#555", alpha=0.4, label="Quadratic trend")
        style_ax(ax[0], "", "Lap Time (s)")
        style_legend(ax[0])

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
