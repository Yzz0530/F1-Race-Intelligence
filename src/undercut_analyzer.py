"""
Undercut / Overcut Analyzer — simulates pit-stop timing battles.

Uses the Physics Engine to model tyre crossover windows and
position delta between two drivers on offset pit strategies.
"""
from __future__ import annotations

import os
import sys
from typing import Any

# Ensure src/ is on sys.path for cross-module imports
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np

from race_physics import (
    PIT_LOSS_DEFAULT,
    compound_delta,
    fuel_effect,
    tyre_degradation,
    track_temp_effect,
    undercut_benefit,
)


class UndercutAnalyzer:
    """Analyze undercut/overcut scenarios between two drivers."""

    def __init__(self, base_lap_time: float = 87.0):
        self.base_lap_time = base_lap_time
        # Typical window: undercut works when gap at pit exit < delta to
        # the car ahead (approx 1.5-2.5s). We model this gap dynamically.

    def compare_strategies(
        self,
        driver_a_compound: str,
        driver_a_pit_laps: list[int],
        driver_b_compound: str,
        driver_b_pit_laps: list[int],
        total_laps: int,
        track_temp: float = 35.0,
    ) -> dict[str, Any]:
        """
        Compare two full race strategies and their cross-over points.
        Returns time deltas at each pit window.
        """
        base = self.base_lap_time
        events: list[dict[str, Any]] = []
        t_a = 0.0
        t_b = 0.0
        stint_a = 0
        stint_b = 0
        lap_a_tyre = 0
        lap_b_tyre = 0

        all_pit_laps = sorted(set(driver_a_pit_laps + driver_b_pit_laps))
        prev_lap = 0

        for pit_lap in all_pit_laps + [total_laps]:
            laps_in_segment = pit_lap - prev_lap
            if laps_in_segment <= 0:
                continue

            # Simulate A's laps
            a_in = driver_a_pit_laps + [total_laps]
            a_seg_end = min(pit_lap, total_laps)
            a_laps = a_seg_end - prev_lap
            for lap in range(a_laps):
                lap_num = prev_lap + lap + 1
                if lap_num in driver_a_pit_laps:
                    t_a += PIT_LOSS_DEFAULT
                    stint_a += 1
                    lap_a_tyre = 0
                lap_a_tyre += 1
                t_a += base + compound_delta(driver_a_compound)
                t_a += tyre_degradation(driver_a_compound, lap_a_tyre)
                t_a += fuel_effect(lap_num, total_laps)

            # Simulate B's laps
            for lap in range(a_laps):
                lap_num = prev_lap + lap + 1
                if lap_num in driver_b_pit_laps:
                    t_b += PIT_LOSS_DEFAULT
                    stint_b += 1
                    lap_b_tyre = 0
                lap_b_tyre += 1
                t_b += base + compound_delta(driver_b_compound)
                t_b += tyre_degradation(driver_b_compound, lap_b_tyre)
                t_b += fuel_effect(lap_num, total_laps)

            delta = t_a - t_b  # positive = A is behind
            events.append({
                "lap": pit_lap,
                "delta_a_to_b": round(delta, 2),
                "t_a": round(t_a, 1),
                "t_b": round(t_b, 1),
                "a_pitted": pit_lap in driver_a_pit_laps,
                "b_pitted": pit_lap in driver_b_pit_laps,
            })
            prev_lap = pit_lap

        return {
            "driver_a": {"compound": driver_a_compound, "pit_laps": driver_a_pit_laps},
            "driver_b": {"compound": driver_b_compound, "pit_laps": driver_b_pit_laps},
            "events": events,
            "final_delta": round(t_a - t_b, 2),
        }

    def _race_time_one_stop(
        self,
        total_laps: int,
        compound: str,
        pit_lap: int,
        tyre_age_at_start: int = 0,
    ) -> float:
        """Total race time for a single-stop race on `compound`/`compound`,
        pitting on `pit_lap` (second stint same compound, fresh tyres)."""
        t = 0.0
        age = tyre_age_at_start + 1
        for lap in range(1, total_laps + 1):
            t += self.base_lap_time + compound_delta(compound)
            t += tyre_degradation(compound, age)
            t += fuel_effect(lap, total_laps)
            if lap == pit_lap:
                t += PIT_LOSS_DEFAULT
                age = 1  # fresh tyres
            else:
                age += 1
        return t

    def find_optimal_pit_window(
        self,
        total_laps: int,
        compound: str,
        stint_length: int,
        tyre_age_at_start: int = 0,
        track_temp: float = 35.0,
    ) -> list[dict[str, Any]]:
        """
        Find pit laps that minimize total race time for a one-stop race on
        `compound`, where the first stint is `stint_length` laps.

        The search evaluates every legal pit lap that yields the requested
        first-stint length and a legal second stint, simulates the full race
        with the real tyre-deg + fuel + pit-loss model, and returns the
        lowest-time options. Earlier vs later pitting is a genuine trade-off:
        pitting early keeps the second stint fresh but wears the first stint
        longer; pitting late does the opposite. The model captures both.

        ``stint_length`` is treated as the *target* first-stint length; the
        optimal pit lap is `stint_length` itself unless there is a secondary
        advantage (there usually isn't for a single compound), in which case
        nearby pit laps are ranked by their true total time.
        """
        results: list[dict[str, Any]] = []
        lo = max(5, stint_length - total_laps // 4)
        hi = min(total_laps - 5, stint_length + total_laps // 4)
        for pit_lap in range(lo, hi + 1):
            if pit_lap < 5 or (total_laps - pit_lap) < 5:
                continue
            total_time = self._race_time_one_stop(
                total_laps, compound, pit_lap, tyre_age_at_start
            )
            results.append({
                "pit_lap": pit_lap,
                "total_time_seconds": round(total_time, 2),
            })

        # Best (minimum) total time = 0 penalty baseline for display.
        best = min((r["total_time_seconds"] for r in results), default=None)
        if best is not None:
            for r in results:
                r["penalty_vs_best_seconds"] = round(r["total_time_seconds"] - best, 2)
        results.sort(key=lambda r: r["total_time_seconds"])
        return results[:10]
