"""
F1 Race Physics Engine — fuel, weather, traffic, and delta time models.

All physics parameters calibrated from real F1 data:
- Fuel: ~0.035s/kg/lap, 110kg start load, 2.5kg/lap burn rate
- Weather: track temp grip curves, rainfall speed loss
- Traffic: position-based dirty air model
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# ── Constants ─────────────────────────────────────────────────────

FUEL_DENSITY_KG_L: float = 0.75  # kg per litre of petrol
FUEL_PER_LAP_KG: float = 2.5      # typical F1 fuel burn per lap
FUEL_START_KG: float = 110.0      # max race fuel load
FUEL_TIME_PER_KG: float = 0.035   # ~0.035s per kg of fuel (from F1 data)

# Track temp optimum and sensitivity (Celsius)
TRACK_TEMP_OPT: float = 32.0       # optimum grip temp
TRACK_TEMP_SENS: float = 0.015     # seconds per degree deviation
TRACK_TEMP_RANGE: tuple[float, float] = (15.0, 55.0)

# Wind sensitivity
WIND_DRAG_COEFF: float = 0.008     # seconds per km/h crosswind

# Rain speed loss by intensity
RAIN_LOSS: dict[str, float] = {
    "dry": 0.0,
    "damp": 0.8,
    "wet": 2.5,
    "heavy": 5.0,
}

# Traffic / dirty-air model by track position
TRAFFIC_LOSS: dict[int, float] = {
    1: 0.0,   # leader — clean air
    2: 0.05,
    3: 0.10,
    4: 0.15,
    5: 0.20,
    6: 0.25,
    7: 0.30,
    8: 0.35,
    9: 0.40,
    10: 0.45,
    11: 0.50,
    12: 0.55,
    13: 0.60,
    14: 0.65,
    15: 0.70,
    16: 0.75,
    17: 0.80,
    18: 0.85,
    19: 0.90,
    20: 0.95,
}
TRAFFIC_DEFAULT_LOSS: float = 1.0

# Pit lane parameters
PIT_LOSS_DEFAULT: float = 22.0     # seconds lost for a pit stop
PIT_LOSS_SAFETY_CAR: float = 12.0  # pit loss reduced under SC (free pit)
OUT_LAP_PENALTY: float = 0.5       # extra time on out-lap (cold tyres)
IN_LAP_PENALTY: float = 0.3        # slight lift on in-lap

# Tyre compound deltas (relative to MEDIUM baseline, in seconds).
# INTERMEDIATE is a full-wet tyre: it is ~1.2s slower than dry slicks on a
# dry track, but the only viable option in rain. The optimizer's physics
# overlay uses these as the single source of truth (the ML delta target does
# not separate compounds in the current all-dry training data).
COMPOUND_DELTA: dict[str, float] = {
    "SOFT": -0.35,
    "MEDIUM": 0.0,
    "HARD": 0.20,
    "INTERMEDIATE": 1.20,
}

# Tyre degradation rate per lap (seconds per lap of wear)
TYRE_DEG_RATE: dict[str, float] = {
    "SOFT": 0.080,
    "MEDIUM": 0.045,
    "HARD": 0.025,
    "INTERMEDIATE": 0.050,
}

# Tyre temperature operating window
TYRE_TEMP_OPT: dict[str, tuple[float, float]] = {
    "SOFT": (85, 110),
    "MEDIUM": (80, 105),
    "HARD": (75, 100),
}


# ── Physics helpers ───────────────────────────────────────────────

def fuel_effect(lap_number: int, total_laps: int, start_fuel: float = FUEL_START_KG) -> float:
    """Time delta from fuel load. Early laps carry more fuel → slower.

    :param lap_number: 1-based lap index in the race.
    :param total_laps: total race length (sets the burn rate = start/total).
    :param start_fuel: starting fuel load in kg (default 110 kg max race load).
    :return: time penalty in seconds relative to an empty car.
    """
    fuel_burn_per_lap = start_fuel / max(total_laps, 1)
    remaining_fuel = start_fuel - (lap_number - 1) * fuel_burn_per_lap
    return remaining_fuel * FUEL_TIME_PER_KG


def fuel_burn_rate(lap_number: int, total_laps: int) -> float:
    """Fuel weight effect per lap increment (negative = getting faster)."""
    return -FUEL_TIME_PER_KG * (FUEL_START_KG / max(total_laps, 1))


def track_temp_effect(track_temp: float) -> float:
    """Time delta from sub-optimal track temperature."""
    if track_temp < 0:
        return 0.0
    clamped = np.clip(track_temp, *TRACK_TEMP_RANGE)
    return TRACK_TEMP_SENS * abs(clamped - TRACK_TEMP_OPT)


def wind_effect(wind_speed: float) -> float:
    """Time delta from wind."""
    return WIND_DRAG_COEFF * wind_speed


def rain_effect(rainfall: float) -> float:
    """Time delta from rain intensity."""
    if rainfall <= 0:
        return 0.0
    if rainfall < 0.3:
        return RAIN_LOSS["damp"]
    if rainfall < 0.7:
        return RAIN_LOSS["wet"]
    return RAIN_LOSS["heavy"]


def traffic_effect(position: int) -> float:
    """Time delta from running in traffic (dirty air)."""
    return TRAFFIC_LOSS.get(position, TRAFFIC_DEFAULT_LOSS)


def compound_delta(compound: str) -> float:
    return COMPOUND_DELTA.get(compound, 0.0)


def tyre_degradation(compound: str, lap_in_stint: int) -> float:
    """Tyre deg time delta for a given lap."""
    rate = TYRE_DEG_RATE.get(compound, 0.045)
    return rate * (lap_in_stint - 1)


def tyre_temp_penalty(compound: str, track_temp: float) -> float:
    """Time penalty if tyres can't reach optimal temp window."""
    temp_range = TYRE_TEMP_OPT.get(compound, (80, 105))
    # Rough estimate: if track temp is very low, tyres can't get in window
    if track_temp < 20:
        return 0.15
    if track_temp > temp_range[1] - 30:  # hot track helps
        return 0.0
    return 0.08  # mild penalty for cold track


def compound_degradation_rate(compound: str, track_temp: float) -> float:
    """Temperature-adjusted degradation rate."""
    base_rate = TYRE_DEG_RATE.get(compound, 0.045)
    # Hotter track = faster deg
    temp_factor = 1.0 + max(0, (track_temp - 35) * 0.02)
    return base_rate * temp_factor


def sc_delta() -> float:
    """Time delta under Safety Car (~3s slower per lap)."""
    return 3.0


def vsc_delta() -> float:
    """Time delta under VSC (~1.5s slower per lap)."""
    return 1.5


def undercut_benefit(
    pitting_lap: int, total_laps: int, tyre_age_before: int,
    fresh_compound: str, position: int = 5,
    pit_loss: float = PIT_LOSS_DEFAULT,
) -> dict[str, float]:
    """
    Calculate the net benefit/loss of an undercut.

    Returns dict with:
    - loss_from_pit: time lost from pit stop
    - gain_from_fresh: time gained from fresh vs old tyres over N laps
    - net_benefit: positive = undercut wins
    - crossover_lap: lap when fresh tyres become faster than old
    """
    laps_remaining = total_laps - pitting_lap
    if laps_remaining <= 0:
        return {"loss_from_pit": 0, "gain_from_fresh": 0, "net_benefit": 0, "crossover_lap": 0}

    # Per-lap time delta of (old tyres at age_before+lap) minus (fresh tyres at lap).
    # Positive = fresh tyres are faster this lap.
    fresh_advantage: list[float] = []
    for lap in range(1, min(laps_remaining + 1, 20)):
        old_time = tyre_degradation(fresh_compound, tyre_age_before + lap)
        new_time = tyre_degradation(fresh_compound, lap)
        fresh_advantage.append(old_time - new_time)

    if not fresh_advantage:
        return {"loss_from_pit": pit_loss, "gain_from_fresh": 0, "net_benefit": -pit_loss, "crossover_lap": 0}

    # Crossover: first lap where fresh tyre is faster
    crossover = 0
    for i, diff in enumerate(fresh_advantage):
        if diff > 0:  # fresh is faster
            crossover = i + 1
            break

    # Gain from fresh tyres over first stint after pit (first 5 laps = undercut window)
    gain_from_fresh = sum(fresh_advantage[:5])

    net = gain_from_fresh - pit_loss
    return {
        "loss_from_pit": pit_loss,
        "gain_from_fresh": round(gain_from_fresh, 3),
        "net_benefit": round(net, 3),
        "crossover_lap": crossover,
    }


def overcut_benefit(
    staying_out_lap: int, total_laps: int,
    tyre_compound: str, tyre_age: int,
    opponent_compound: str, opponent_pit_lap: int,
    track_temp: float = 35.0,
) -> dict[str, float]:
    """
    Calculate benefit of overcut (staying out while opponent pits).

    Returns dict with net benefit (positive = overcut wins).
    """
    laps_after_pit = total_laps - opponent_pit_lap
    if laps_after_pit <= 0:
        return {"net_benefit": 0}

    # Opponent loses pit time + has fresh tyres
    # We stay out on older tyres but gain track position

    # Simplified: compare tyre delta over the window
    our_degradation = tyre_degradation(tyre_compound, tyre_age)
    their_fresh_deg = 0  # minimal on fresh tyres

    # Over 3 laps after pit, what's the delta?
    total_gain = 0
    for lap in range(1, 4):
        our_lap = our_degradation + compound_delta(tyre_compound) + tyre_degradation(tyre_compound, tyre_age + lap)
        their_lap = their_fresh_deg + compound_delta(opponent_compound) + tyre_degradation(opponent_compound, lap)
        total_gain += their_lap - our_lap

    return {"net_benefit": round(total_gain - PIT_LOSS_DEFAULT, 3)}


def race_time_estimate(
    total_laps: int,
    base_lap_time: float,
    compound: str,
    track_temp: float = 35.0,
    rainfall: float = 0.0,
    wind_speed: float = 2.0,
    position: int = 1,
) -> tuple[list[float], float]:
    """
    Estimate lap times for a full stint on one compound using physics only.
    Returns (lap_times, total_time).
    """
    times: list[float] = []
    for lap in range(1, total_laps + 1):
        lt = base_lap_time
        lt += compound_delta(compound)
        lt += tyre_degradation(compound, lap)
        lt += fuel_effect(lap, total_laps)
        lt += track_temp_effect(track_temp)
        lt += rain_effect(rainfall)
        lt += wind_effect(wind_speed)
        lt += traffic_effect(position)
        lt += tyre_temp_penalty(compound, track_temp)
        times.append(lt)
    return times, sum(times)


def simulate_sc_scenario(
    sc_lap: int, total_laps: int, base_lap_time: float,
    sc_duration: int = 3,
    sc_slow: float = 3.0,
    sc_free_pit: bool = True,
    pit_loss: float = PIT_LOSS_DEFAULT,
    compound: str = "MEDIUM",
    track_temp: float = 35.0,
) -> dict[str, Any]:
    """
    Simulate a Safety Car scenario.

    If sc_free_pit=True, pitting under SC reduces pit loss to ~12s.
    Returns dict with time gained/lost vs green-flag baseline.
    """
    # Baseline: normal race
    base_times, base_total = race_time_estimate(total_laps, base_lap_time, compound, track_temp)

    # SC scenario — track tyre age across the whole race, resetting after under-SC pit
    sc_times: list[float] = []
    sc_pit_lap = sc_lap if sc_free_pit else None
    tyre_age = 1  # lap 1 of a fresh stint

    for lap in range(1, total_laps + 1):
        if sc_lap <= lap < sc_lap + sc_duration:
            # Under SC — slow laps regardless of tyre state
            if sc_free_pit and lap == sc_lap:
                # Pit under SC: pay reduced pit loss, then restart on fresh tyres
                lt = base_lap_time + sc_slow + PIT_LOSS_SAFETY_CAR
                tyre_age = 1  # fresh tyres from the pit stop
            else:
                lt = base_lap_time + sc_slow
            sc_times.append(lt)
            tyre_age += 1  # SC laps still age tyres (slowly)
        else:
            # Green-flag lap — apply full tyre model with current tyre_age
            lt = base_lap_time
            lt += compound_delta(compound)
            lt += tyre_degradation(compound, tyre_age)
            lt += fuel_effect(lap, total_laps)
            lt += track_temp_effect(track_temp)
            sc_times.append(lt)
            tyre_age += 1

    sc_total = sum(sc_times)
    time_saved = base_total - sc_total

    return {
        "sc_lap": sc_lap,
        "sc_duration": sc_duration,
        "base_total": round(base_total, 1),
        "sc_total": round(sc_total, 1),
        "time_saved": round(time_saved, 1),
        "free_pit": sc_free_pit,
        "sc_pit_loss": PIT_LOSS_SAFETY_CAR if sc_free_pit else 0,
    }
