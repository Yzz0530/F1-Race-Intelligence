"""TRACK ANALYSIS tab — per-circuit sector / compound / degradation / traffic view."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ._shared import F1_RED, COMPOUND_COLORS, DRIVERS_LIST, _team_color, _team_name, traffic_effect, style_ax, style_legend, load_optimizer


def render_track_analysis() -> None:
    opt = load_optimizer()
    tracks = sorted(opt.circuit_info.keys())
    ca, cb, cc = st.columns(3)
    with ca:
        ta_track = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="ta_track")
    with cb:
        ta_year = st.selectbox("Year", ["All", "2026", "2025"], key="ta_year")
    with cc:
        ta_compound = st.selectbox("Compound", ["All", "SOFT", "MEDIUM", "HARD"], key="ta_compound")

    st.markdown("<br>", unsafe_allow_html=True)

    _BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df_raw = pd.read_csv(os.path.join(_BASE, "data", "all_races_master.csv"))
    df_track = df_raw[df_raw["Race"] == ta_track].copy()
    if ta_year != "All": df_track = df_track[df_track["Year"] == int(ta_year)]
    if ta_compound != "All": df_track = df_track[df_track["Compound"] == ta_compound]
    if df_track.empty:
        st.info("No data for these filters.")
        return

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

    # Traffic effect simulation — uses the engine's traffic model (race_physics.traffic_effect)
    st.markdown("<div class='section-label'>Traffic Impact (Dirty Air Simulation)</div>", unsafe_allow_html=True)
    avg_lt = df_trk["LapTime"].mean()
    pos = list(range(1, 13))
    traffic_losses = [traffic_effect(p) for p in pos]
    fig6, ax6 = plt.subplots(figsize=(10, 2.5))
    fig6.patch.set_facecolor("none")
    ax6.fill_between(pos, [avg_lt + t for t in traffic_losses], avg_lt, alpha=0.2, color=F1_RED)
    ax6.plot(pos, [avg_lt + t for t in traffic_losses], color=F1_RED, linewidth=2, marker="o", markersize=5)
    for p, t in zip(pos, traffic_losses):
        ax6.annotate(f"+{t:.2f}s", (p, avg_lt + t), textcoords="offset points", xytext=(0, 5),
                     ha="center", color="#777", fontsize=7)
    style_ax(ax6, "Position", "Estimated Lap Time (s)")
    st.pyplot(fig6, clear_figure=True)
