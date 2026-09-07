"""Regression coverage for the Streamlit dashboard tab router."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "dashboard.py"
TAB_NAMES = [
    "RESULTS",
    "STANDINGS",
    "RACE TIMELINE",
    "DRIVER BATTLE",
    "TRACK ANALYSIS",
    "STINT TELEMETRY",
    "CAR TELEMETRY",
    "STRATEGY",
    "SC SIMULATOR",
    "PIT ANALYSIS",
    "AI ASSISTANT",
]


def test_every_dashboard_tab_renders_without_an_uncaught_exception() -> None:
    """Every routed tab must own the imports it needs to render its default state."""
    app = AppTest.from_file(str(DASHBOARD))
    app.run(timeout=120)

    for tab_name in TAB_NAMES:
        app.radio[0].set_value(tab_name)
        app.run(timeout=120)
        assert not app.exception, (
            f"{tab_name} raised an uncaught exception: "
            f"{[(exception.type, exception.value) for exception in app.exception]}"
        )
