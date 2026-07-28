"""Race Timeline Tab — additive, no existing code modified."""


def render_race_timeline(opt, tracks, DRIVERS_LIST, _team_name, COMPOUND_COLORS, plt, pd, st):

    st.markdown(
        "<h3 style='margin:0 0 0.5rem;'>📊 Race Timeline</h3>"
        "<p style='color:var(--text-dim);font-size:0.8rem;margin-bottom:1rem;'>"
        "See the full race unfold — strategy, pit stops, tyre degradation, and key events.</p>",
        unsafe_allow_html=True,
    )

    rt_c1, rt_c2, rt_c3, rt_c4 = st.columns(4)
    with rt_c1:
        rt_driver = st.selectbox(
            "Driver", DRIVERS_LIST, index=DRIVERS_LIST.index("VER"),
            format_func=lambda d: f"{d}  ·  {_team_name(d)}", key="rt_driver",
        )
    with rt_c2:
        rt_track = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="rt_track")
    with rt_c3:
        rt_laps = st.number_input("Race Laps", 10, 80, 52, step=1, key="rt_laps")
    with rt_c4:
        rt_sc = st.slider("SC Probability", 0.0, 0.5, 0.20, 0.05, key="rt_sc")

    if not st.button("Build Timeline", type="primary", use_container_width=False):
        return

    with st.spinner("Simulating race strategy..."):
        top = opt.optimize(rt_track, rt_laps, rt_driver, mc_runs=50, sc_prob=rt_sc)
        if not top:
            st.warning("No valid strategies found. Try different parameters.")
            return
        best = top[0]
        detail = opt.get_detailed_run(rt_track, rt_laps, rt_driver, best["strategy"])

    if not detail:
        st.warning("Could not simulate. Try different parameters.")
        return

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Overview metrics ──
    mins, secs = divmod(int(best["mean_time"]), 60)
    cols = st.columns([1, 1, 1, 1, 1])
    cols[0].metric("Best Strategy", f"{best['stints']} stop{'s' if best['stints']>1 else ''}")
    cols[1].metric("Total Time", f"{mins}:{secs:02d}")
    cols[2].metric("Avg Lap", f"{best['mean_time']/rt_laps:.3f}s")
    cols[3].metric("Stint Layout", " → ".join(s[0] for s in best["strategy"]))
    cols[4].metric("SC Deployed", "Yes" if detail.get("sc_deployed") else "No")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Visual Timeline ──
    st.markdown("<h4 style='margin:0 0 0.5rem;'>🏎️ Stint Timeline</h4>", unsafe_allow_html=True)

    stint_details = detail["stint_details"]
    pct_per_lap = 100.0 / rt_laps

    html_parts = [
        '<div style="background:var(--bg-card);border:1px solid var(--border-subtle);'
        'border-radius:6px;padding:1rem 1.2rem;">'
    ]

    lap_offset = 0
    for si, stint in enumerate(stint_details):
        cpd = stint["compound"]
        laps = stint["laps"]
        color = COMPOUND_COLORS.get(cpd, "#666")
        width_pct = laps * pct_per_lap
        deg = stint["lap_times"][-1] - stint["lap_times"][0]

        start_lap = lap_offset + 1
        end_lap = lap_offset + laps
        html_parts.append(
            f'<div style="display:flex;align-items:center;margin-bottom:0.4rem;font-size:0.65rem;'
            f'color:var(--text-dim);">'
            f'<span style="min-width:38px;">L{start_lap}</span>'
            f'<div style="width:{width_pct}%;height:1.8rem;'
            f'background:linear-gradient(135deg,{color}99,{color}33);'
            f'border-left:3px solid {color};'
            f'border-radius:0 4px 4px 0;'
            f'display:flex;align-items:center;padding-left:8px;">'
            f'<span style="font-weight:700;font-size:0.7rem;color:#fff;'
            f'text-shadow:0 1px 2px rgba(0,0,0,0.6);">{cpd}</span></div>'
            f'<span style="margin-left:8px;">- L{end_lap}</span>'
            f'<span style="margin-left:auto;font-size:0.6rem;">{laps}l avg {stint["avg_time"]:.3f}s'
            f' deg {deg:+.3f}s</span></div>'
        )

        if si < len(stint_details) - 1:
            pit_lap = lap_offset + laps
            pit_pct = max(0, (pit_lap / rt_laps) * 100 - 2)
            html_parts.append(
                f'<div style="display:flex;align-items:center;margin-bottom:0.4rem;">'
                f'<div style="width:{pit_pct}%;text-align:right;padding-right:6px;">'
                f'<span style="background:#e10600;color:#fff;padding:2px 10px;'
                f'border-radius:10px;font-size:0.6rem;font-weight:600;">'
                f'PIT L{pit_lap}</span></div></div>'
            )

        lap_offset += laps

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Degradation Chart ──
    st.markdown("<h4 style='margin:0 0 0.5rem;'>📉 Tyre Degradation per Stint</h4>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    global_lap = 1
    for stint in stint_details:
        cpd = stint["compound"]
        laps = stint["lap_times"]
        xs = list(range(global_lap, global_lap + len(laps)))
        color = COMPOUND_COLORS.get(cpd, "#666")
        ax.plot(xs, laps, color=color, linewidth=1.8, label=f"{cpd} (avg {stint['avg_time']:.3f}s)")
        ax.fill_between(xs, laps, alpha=0.08, color=color)
        global_lap += len(laps)

    ax.set_xlabel("Lap", color="#888", fontsize=9)
    ax.set_ylabel("Lap Time (s)", color="#888", fontsize=9)
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(framealpha=0.3, fontsize=8, labelcolor="#ccc")
    ax.set_title(f"{rt_driver} @ {rt_track} — Lap Time Progression", color="#ccc", fontsize=10)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Events ──
    st.markdown("<h4 style='margin:0 0 0.5rem;'>📋 Key Events</h4>", unsafe_allow_html=True)

    events = []
    lap_offset = 0
    for si, stint in enumerate(stint_details):
        cpd = stint["compound"]
        laps = stint["laps"]
        deg = stint["lap_times"][-1] - stint["lap_times"][0]
        pit_lap = lap_offset + laps

        events.append((
            lap_offset + 1,
            f"Started stint on <b>{cpd}</b> — lap {stint['avg_time']:.3f}s avg",
            "#ffb800",
        ))

        if deg > 0.5:
            events.append((
                lap_offset + laps // 2,
                f"⚠️ Tyre degradation exceeds <b>+{deg:.2f}s</b> on {cpd}",
                "#e10600",
            ))

        if si < len(stint_details) - 1:
            events.append((
                pit_lap,
                f"⛽ Pit stop — switch from <b>{cpd}</b> to <b>{stint_details[si+1]['compound']}</b>",
                "#a0a0a0",
            ))

        lap_offset += laps

    if detail.get("sc_deployed"):
        events.insert(0, (rt_laps // 3, "🚨 <b>Safety Car deployed</b> — strategy window opens", "#ffb800"))

    seen = set()
    for lap, desc, color in events:
        if lap in seen:
            continue
        seen.add(lap)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;padding:0.3rem 0;font-size:0.8rem;'>"
            f"<span style='background:{color};color:#000;font-weight:700;font-size:0.65rem;"
            f"padding:2px 8px;border-radius:10px;min-width:42px;text-align:center;'>L{lap}</span>"
            f"<span style='color:var(--text-primary);'>{desc}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stint Breakdown Table ──
    st.markdown("<h4 style='margin:0 0 0.5rem;'>📊 Stint Breakdown</h4>", unsafe_allow_html=True)
    rows = []
    for si, stint in enumerate(stint_details):
        rows.append({
            "Stint": si + 1,
            "Compound": stint["compound"],
            "Laps": stint["laps"],
            "Avg Lap": f"{stint['avg_time']:.3f}s",
            "First Lap": f"{stint['lap_times'][0]:.3f}s",
            "Last Lap": f"{stint['lap_times'][-1]:.3f}s",
            "Deg.": f"{stint['lap_times'][-1] - stint['lap_times'][0]:+.3f}s",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Stint"), use_container_width=True)

    # ── Actionable Insight ──
    st.markdown("<br>", unsafe_allow_html=True)
    worst = max(stint_details, key=lambda s: s["avg_time"])
    best = min(stint_details, key=lambda s: s["avg_time"])
    insight = (
        f"💡 <b>Key Insight:</b> The <b>{best['compound']}</b> stint "
        f"(avg {best['avg_time']:.3f}s) was the strongest. "
        f"The <b>{worst['compound']}</b> stint showed "
        f"{'high' if worst['lap_times'][-1] - worst['lap_times'][0] > 0.5 else 'manageable'} "
        f"degradation at +{worst['lap_times'][-1] - worst['lap_times'][0]:.2f}s "
        f"over {worst['laps']} laps."
    )
    st.markdown(
        f"<div style='background:rgba(225,6,0,0.1);border:1px solid #e10600;"
        f"border-radius:6px;padding:0.7rem 1rem;font-size:0.8rem;'>{insight}</div>",
        unsafe_allow_html=True,
    )
