"""STRATEGY tab — Monte Carlo pit-strategy simulation."""
from __future__ import annotations

import time

import streamlit as st

from ._shared import (
    F1_RED, COMPOUND_COLORS, DRIVERS_LIST, _team_color, _team_name,
    _compound_badge, style_ax, style_legend, load_optimizer, run_opt, run_detailed,
)


def render_strategy() -> None:
    opt = load_optimizer()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sd1 = st.selectbox("Driver", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"),
                           format_func=lambda d: f"{d}  ·  {_team_name(d)}", key="s_driver")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-top:4px;padding-left:4px;'>"
            f"<span class='driver-dot' style='color:{_team_color(sd1)};'></span>"
            f"<span style='color:var(--text-primary);font-weight:700;font-size:0.85rem;'>{sd1}</span>"
            f"<span style='color:var(--text-dim);font-size:0.7rem;'>{_team_name(sd1)}</span></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st1 = st.selectbox("Track", sorted(opt.circuit_info.keys()),
                           index=sorted(opt.circuit_info.keys()).index("British Grand Prix"), key="s_track")
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
        # imported lazily to avoid a circular import at module load
        import matplotlib.pyplot as plt
        import numpy as np
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
                f"<div style='display:flex;align-items:center;gap:0.8rem;padding:0.7rem 1rem;'"
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
