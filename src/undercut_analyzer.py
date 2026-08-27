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
    overcut_benefit,
)


class UndercutAnalyzer:
    """Analyze undercut/overcut scenarios between two drivers."""

    def __init__(self, base_lap_time: float = 87.0):
        self.base_lap_time = base_lap_time
        # Typical window: undercut works when gap at pit exit < delta to
        # the car ahead (approx 1.5-2.5s). We model this gap dynamically.

    def analyze_undercut(
        self,
        pit_lap: int,
        total_laps: int,
        tyre_age_before_pit: int,
        fresh_compound: str = "SOFT",
        old_compound: str = "MEDIUM",
        gap_to_ahead: float = 1.5,  # seconds gap to car ahead at T0
        opponent_pit_lap: int | None = None,
        track_temp: float = 35.0,
    ) -> dict[str, Any]:
        """
        Full undercut analysis.

        Returns dict with crossover lap, net time gain, and whether the
        undercut succeeds given the initial gap.
        """
        uc = undercut_benefit(
            pit_lap, total_laps, tyre_age_before_pit,
            fresh_compound, pit_loss=PIT_LOSS_DEFAULT,
        )
        # Effective gap after pit (gap + pit loss - fresh tyre gain)
        effective_gap = gap_to_ahead + PIT_LOSS_DEFAULT - uc["gain_from_fresh"]

        success = effective_gap < 0  # undercut wins if we come out ahead
        track_positions_gained = 0
        if success:
            # Rough: each ~2s = 1 position
            track_positions_gained = min(int(abs(effective_gap) / 2.0) + 1, 5)

        return {
            "pit_lap": pit_lap,
            "gap_to_ahead_before_pit": round(gap_to_ahead, 2),
            "pit_loss": PIT_LOSS_DEFAULT,
            "fresh_tyre_gain_first_5_laps": uc["gain_from_fresh"],
            "crossover_lap": uc["crossover_lap"],
            "net_benefit": uc["net_benefit"],
            "effective_gap_after_stop": round(effective_gap, 2),
            "undercut_succeeds": success,
            "track_positions_gained_est": track_positions_gained,
        }

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

    def find_optimal_pit_window(
        self,
        total_laps: int,
        compound: str,
        stint_length: int,
        tyre_age_at_start: int = 0,
        track_temp: float = 35.0,
    ) -> list[dict[str, Any]]:
        """
        For a given compound and target stint length, find the optimal
        lap to pit (minimizing time loss from tyre deg + fuel).
        """
        results: list[dict[str, Any]] = []
        for pit_lap in range(5, total_laps - stint_length + 1, 1):
            # Degradation before pit
            before_deg = tyre_degradation(compound, tyre_age_at_start + pit_lap)
            # Degradation after pit
            after_deg = tyre_degradation(compound, stint_length)
            # Fuel effect
            fuel = sum(fuel_effect(lap, total_laps) for lap in range(1, total_laps + 1))
            total_penalty = before_deg * pit_lap + after_deg * stint_length + PIT_LOSS_DEFAULT + fuel * 0.01
            results.append({
                "pit_lap": pit_lap,
                "total_penalty_seconds": round(total_penalty, 2),
            })

        results.sort(key=lambda r: r["total_penalty_seconds"])
        return results[:10]
