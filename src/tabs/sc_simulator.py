"""SC SIMULATOR tab — Safety Car what-if scenarios (now like-for-like)."""
from __future__ import annotations

import matplotlib.pyplot as plt

from ._shared import F1_RED, COMPOUND_COLORS, DRIVERS_LIST, _team_color, _team_name, simulate_sc_scenario, style_ax, load_optimizer


def render_sc_simulator() -> None:
    opt = load_optimizer()
    tracks = sorted(opt.circuit_info.keys())
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
        base_lt = opt.overall_baseline
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
            f"<div><span style='color:var(--text-dim);font-size:0.6rem;'>DELTA (baseline also pits)</span><br>"
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

        fig, ax = plt.subplots(figsize=(12, 2.5))
        fig.patch.set_facecolor("none")
        laps_x = list(range(1, sc_total + 1))
        base_curve = [base_lt + 0.03 * (l - 1) * 0.02 for l in laps_x]
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
