"""CAR TELEMETRY tab — live fastf1 car data with offline sector-speed fallback."""
from __future__ import annotations

import streamlit as st

from ._shared import DRIVERS_LIST, _team_color, _team_name, load_optimizer


def render_car_telemetry() -> None:
    tracks = sorted(load_optimizer().circuit_info.keys())
    ca, cb, cc, cd = st.columns(4)
    with ca:
        telem_year = st.selectbox("Year", [2026, 2025], key="telem_yr")
    with cb:
        telem_track = st.selectbox("Track", tracks, index=tracks.index("British Grand Prix"), key="telem_trk")
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
                get_offline_sector_data, plot_offline_sector_comparison,
                load_cached_telemetry,
            )
            import pandas as pd
            session = resolve_session(telem_year, telem_track)
            if session is None:
                # Offline path 1: committed per-metre telemetry traces built by CI.
                tc1 = load_cached_telemetry(telem_year, telem_track, telem_d1)
                tc2 = load_cached_telemetry(telem_year, telem_track, telem_d2)
                if tc1 is not None or tc2 is not None:
                    st.markdown(
                        "<div class='section-label'>Car Telemetry (offline — committed "
                        "traces from data/telemetry/*.parquet)</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "⚠️ Full Speed / Throttle / Brake / Gear / DRS traces from committed "
                        "data. Shown when live FastF1 telemetry is unavailable."
                    )
                    fig = plot_telemetry_comparison(tc1, tc2, telem_d1, telem_d2)
                    if fig:
                        st.pyplot(fig, clear_figure=True)
                    else:
                        st.info("Telemetry plot not available from committed data.")
                else:
                    st.error(
                        f"**Detailed car telemetry unavailable for {telem_track} "
                        f"({telem_year}).**\n\n"
                        "Per-metre Speed / Throttle / Brake / Gear / DRS traces are only "
                        "fetched live from FastF1 and are **not** stored in the committed "
                        "data pipeline. This session may not be available yet, or the "
                        "FastF1 API may be rate-limited or temporarily unreachable."
                    )
                    off = get_offline_sector_data(telem_year, telem_track, [telem_d1, telem_d2])
                    if off is not None:
                        st.markdown(
                            "<div class='section-label'>Alternative — Sector Speed Comparison "
                            "(offline, committed data)</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "⚠️ This is a sector-speed summary from committed lap data, **not** "
                            "full car telemetry. It is shown only as a comparison aid when live "
                            "telemetry cannot be loaded."
                        )
                        offig = plot_offline_sector_comparison(off, telem_d1, telem_d2)
                        if offig:
                            st.pyplot(offig, clear_figure=True)
                        tbl_rows = []
                        for r in off["drivers"]:
                            tbl_rows.append({
                                "Driver": r["driver"],
                                "Best Lap (s)": (f"{r['best_lap']:.3f}" if r.get("best_lap") == r.get("best_lap") else "—"),
                                "S1 Speed": f"{r.get('S1_speed', 0):.1f}",
                                "S2 Speed": f"{r.get('S2_speed', 0):.1f}",
                                "S3 Speed": f"{r.get('S3_speed', 0):.1f}",
                                "Avg Speed": f"{r.get('AvgSpeed', 0):.1f}",
                            })
                        st.dataframe(pd.DataFrame(tbl_rows), hide_index=True)
                    st.info(
                        "💡 Tip: retry in a few minutes if FastF1 was rate-limited. The first "
                        "successful live load is then cached for instant future loads."
                    )
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

                    wx = get_session_weather(session)
                    if wx:
                        st.markdown("<div class='section-label'>Session Weather</div>", unsafe_allow_html=True)
                        wdf = pd.DataFrame(wx)
                        wdf["time"] = wdf["time"].astype(str)
                        st.dataframe(wdf, hide_index=True)

                    st.info("💡 Tip: First load may take 30-60s (downloading from fastf1). "
                            "Subsequent loads are instant from cache.")
