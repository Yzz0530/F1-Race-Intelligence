"""
F1 Strategy Optimizer — XGBoost-powered race strategy simulation.

Predicts absolute lap times using a trained XGBoost model with
physics-based fallback. Supports Monte Carlo simulation with
safety car / DNF probability for strategy comparison.

Compound separation is driven by the physics overlay (race_physics
COMPOUND_DELTA), because the all-dry training data does not teach the
ML delta target to distinguish SOFT/MEDIUM/HARD. The overlay is the
single source of truth for compound/fuel/track effects; the ML delta
captures driver/race/car-level pace relative to the per-race baseline.
"""
from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from race_physics import fuel_effect as _race_fuel_effect

warnings = __import__("warnings")
warnings.filterwarnings("ignore")

_BASE: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_Strategy = list[tuple[str, int]]
_RunResult = dict[str, Any]
_OptResult = dict[str, Any]

COMPOUND_ORDER: dict[str, int] = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4}
COMPOUND_MAP: dict[str, str] = {"SOFT": "DRY", "MEDIUM": "DRY", "HARD": "DRY", "INTERMEDIATE": "WET"}


class F1StrategyOptimizer:
    COMPOUND_SPEED: dict[str, float] = {
        "SOFT": -2.8,
        "MEDIUM": 0.0,
        "HARD": 1.5,
        "INTERMEDIATE": 1.20,
    }
    TIRE_DEG_RATE: dict[str, float] = {
        "SOFT": 0.080,
        "MEDIUM": 0.045,
        "HARD": 0.025,
        "INTERMEDIATE": 0.050,
    }
    OUT_LAP_PENALTY: float = 0.5
    PIT_LOSS: float = 22.0
    ML_WEIGHT: float = 0.4  # blend between ML and physics; 1.0 = pure ML

    def __init__(self) -> None:
        models_dir = os.path.join(_BASE, "models")
        self.le_driver: Any = joblib.load(os.path.join(models_dir, "le_driver_master.pkl"))
        self.le_compound: Any = joblib.load(os.path.join(models_dir, "le_compound_master.pkl"))
        self.le_family: Any = joblib.load(os.path.join(models_dir, "le_family_master.pkl"))
        self.xgb_model: Any = joblib.load(os.path.join(models_dir, "xgb_master.pkl"))
        self.feature_list: list[str] = joblib.load(os.path.join(models_dir, "feature_list_master.pkl"))
        self.fallback: dict[str, dict[str, float]] = joblib.load(os.path.join(models_dir, "fallback_features.pkl"))
        self.driver_form_proxy: dict[tuple[str, str], float] = joblib.load(os.path.join(models_dir, "driver_form_proxy.pkl"))
        self.circuit_info: dict[str, dict[str, float]] = joblib.load(os.path.join(models_dir, "circuit_info.pkl"))

        df = pd.read_csv(os.path.join(_BASE, "data", "all_races_master.csv"))
        overall_avg = df["LapTime"].mean()
        self.overall_baseline: float = float(overall_avg)
        self.driver_offsets: dict[str, float] = {}
        for d in df["Driver"].unique():
            avg = df[df["Driver"] == d]["LapTime"].mean()
            self.driver_offsets[d] = avg - overall_avg

        # Per-race baseline (mean lap time per race) — used to convert delta
        # predictions back to absolute lap times
        self.race_baselines: dict[str, float] = df.groupby("Race")["LapTime"].mean().to_dict()

        self._encoded_drivers: dict[str, int] = {
            d: int(self.le_driver.transform([d])[0]) for d in self.le_driver.classes_
        }
        self._encoded_compounds: dict[str, int] = {
            c: int(self.le_compound.transform([c])[0]) for c in self.le_compound.classes_
        }
        self._family_enc: int = int(self.le_family.transform(["DRY"])[0])
        self._n_features: int = len(self.feature_list)
        self._cid: dict[str, int] = {n: i for i, n in enumerate(self.feature_list)}

    def _build_feature_matrix(self, driver: str, race_name: str, strategy: _Strategy) -> np.ndarray:
        n_laps = sum(sl for _, sl in strategy)
        M = np.zeros((n_laps, self._n_features), dtype=np.float32)
        fb = self.fallback.get(race_name, {})
        circuit = self.circuit_info.get(race_name, {})
        driver_enc = self._encoded_drivers.get(driver, 0)
        # Tyre family / wet flag are race-wide constants in the feature matrix.
        # Any INTERMEDIATE stint makes the whole run a wet race.
        compounds_in_strategy = [c for c, _ in strategy]
        is_wet = any(c == "INTERMEDIATE" for c in compounds_in_strategy)
        fam_encoded = int(self.le_family.transform(["WET"])[0]) if is_wet else self._family_enc
        form_key = (race_name, driver)
        driver_form = self.driver_form_proxy.get(form_key, self.driver_offsets.get(driver, 0.0))

        const_vals = {
            "Driver_enc": driver_enc,
            "CompoundFamily_enc": fam_encoded,
            "IsWet": 1.0 if is_wet else 0.0,
            "TrackStatus": 1.0,
            "DriverForm": driver_form,
            "Position_normalized": fb.get("Position_normalized", 0.5),
            "IsPersonalBest_int": 0.0,
            "S1_speed": fb.get("S1_speed", 120.0),
            "S2_speed": fb.get("S2_speed", 200.0),
            "S3_speed": fb.get("S3_speed", 95.0),
            "AvgSpeed": fb.get("AvgSpeed", 270.0),
            "AirTemp": fb.get("AirTemp", 25.0),
            "TrackTemp": fb.get("TrackTemp", 35.0),
            "Humidity": fb.get("Humidity", 50.0),
            "Rainfall": fb.get("Rainfall", 0.0),
            "WindSpeed": fb.get("WindSpeed", 2.0),
            "CircuitLength_km": circuit.get("Length_km", 5.0),
            "CircuitCorners": circuit.get("Corners", 15),
            "CircuitAvgSpeed": circuit.get("AvgSpeed", 210.0),
            "CircuitType_enc": circuit.get("Type_enc", 0),
        }
        for name, val in const_vals.items():
            if name in self._cid:
                M[:, self._cid[name]] = val

        idx = 0
        for stint_idx, (compound, stint_laps) in enumerate(strategy):
            compound_enc = self._encoded_compounds.get(compound, 0)
            compound_ord = float(COMPOUND_ORDER.get(compound, 2))
            stint = float(stint_idx + 1)
            for lis in range(1, stint_laps + 1):
                lap_number = idx + 1
                tyre_life = float(lis)
                stint_progress = lis / max(stint_laps, 1)
                if stint_progress <= 0.33:
                    stint_phase = 0.0
                elif stint_progress <= 0.66:
                    stint_phase = 1.0
                else:
                    stint_phase = 2.0
                M[idx, self._cid["Compound_enc"]] = compound_enc
                M[idx, self._cid["CompoundOrdinal"]] = compound_ord
                M[idx, self._cid["TyreLife"]] = tyre_life
                M[idx, self._cid["TyreLife_sq"]] = tyre_life ** 2
                M[idx, self._cid["Stint"]] = stint
                M[idx, self._cid["StintPhase"]] = stint_phase
                M[idx, self._cid["LapInRace"]] = float(lap_number)
                M[idx, self._cid["LapInRace_sq"]] = float(lap_number ** 2)
                M[idx, self._cid["FuelWeightEffect"]] = lap_number * -0.03
                M[idx, self._cid["FreshTire_int"]] = 1.0 if lis == 1 else 0.0
                M[idx, self._cid["IsStartLap"]] = 1.0 if lis <= 1 else 0.0
                idx += 1
        return M

    def _physics_delta(self, driver: str, compound: str, lap_in_stint: int,
                       lap_number: int, total_laps: int, sc_active: bool = False) -> float:
        base_speed = self.COMPOUND_SPEED[compound]
        tire_deg = self.TIRE_DEG_RATE[compound] * (lap_in_stint - 1)
        fuel = _race_fuel_effect(lap_number, total_laps)  # properly models 110kg start, 2.5kg/lap burn
        out_lap = self.OUT_LAP_PENALTY if lap_in_stint == 1 else 0.0
        driver_off = self.driver_offsets.get(driver, 0.0)
        sc_delta = 3.0 if sc_active else 0.0
        return base_speed + tire_deg + fuel + out_lap + driver_off + sc_delta

    def _common_simulate(
        self,
        race_name: str,
        total_laps: int,
        driver: str,
        strategy: _Strategy,
        ml_preds: np.ndarray | None = None,
        pit_loss: float | None = None,
        rng: np.random.RandomState | None = None,
        sc_prob: float = 0.0,
        dnf_prob: float = 0.0,
    ) -> _RunResult | None:
        """Shared simulation core used by both simulate_strategy and _simulate_from_ml.

        When ml_preds is None, the caller is responsible for providing a feature
        matrix and the fallback physics path is used.
        """
        if driver not in self.driver_offsets:
            return None
        if rng is None:
            rng = np.random.RandomState()
        if pit_loss is None:
            pit_loss = self.PIT_LOSS
        if rng.random() < dnf_prob:
            return None

        total_time = 0.0
        stint_details: list[dict[str, Any]] = []
        lap_number = 1
        sc_deployed = False
        sc_next_lap: int | None = None
        if rng.random() < sc_prob:
            sc_next_lap = rng.randint(5, max(15, total_laps // 3))

        flat_idx = 0
        for stint_idx, (compound, stint_laps) in enumerate(strategy):
            times: list[float] = []
            for lis in range(1, stint_laps + 1):
                sc_active = (sc_next_lap == lap_number)
                if sc_active:
                    sc_deployed = True
                    sc_next_lap = None
                noise = rng.normal(0, 0.08)
                if ml_preds is not None:
                    ml_lt = float(ml_preds[flat_idx])
                    # Convert delta prediction back to absolute lap time
                    base_for_race = self.race_baselines.get(race_name, self.overall_baseline)
                    ml_lt += base_for_race
                    phys_delta = self._physics_delta(driver, compound, lis, lap_number, total_laps, sc_active)
                    lt = ml_lt + (1.0 - self.ML_WEIGHT) * phys_delta + noise
                else:
                    lt = self.overall_baseline + self._physics_delta(driver, compound, lis, lap_number, total_laps, sc_active) + noise
                times.append(lt)
                total_time += lt
                lap_number += 1
                flat_idx += 1
            stint_details.append({
                "compound": compound,
                "laps": stint_laps,
                "avg_time": float(np.mean(times)),
                "lap_times": times,
            })
            if stint_idx < len(strategy) - 1:
                total_time += pit_loss

        return {
            "race": race_name,
            "driver": driver,
            "total_laps": total_laps,
            "stints": len(strategy),
            "strategy": [(c, l) for c, l in strategy],
            "total_time": total_time,
            "pit_loss_total": pit_loss * (len(strategy) - 1),
            "stint_details": stint_details,
            "sc_deployed": sc_deployed,
        }

    def simulate_strategy(
        self,
        race_name: str,
        total_laps: int,
        driver: str,
        strategy: _Strategy,
        pit_loss: float | None = None,
        rng: np.random.RandomState | None = None,
        sc_prob: float = 0.0,
        dnf_prob: float = 0.0,
    ) -> _RunResult | None:
        """Run a single strategy simulation (ML + physics blend)."""
        feat_mat = self._build_feature_matrix(driver, race_name, strategy)
        ml_preds: np.ndarray | None = None
        try:
            ml_preds = self.xgb_model.predict(feat_mat)
        except Exception:
            ml_preds = None
        return self._common_simulate(race_name, total_laps, driver, strategy,
                                      ml_preds=ml_preds, pit_loss=pit_loss,
                                      rng=rng, sc_prob=sc_prob, dnf_prob=dnf_prob)

    def _simulate_from_ml(
        self,
        race_name: str,
        total_laps: int,
        driver: str,
        strategy: _Strategy,
        ml_preds: np.ndarray,
        pit_loss: float | None = None,
        rng: np.random.RandomState | None = None,
        sc_prob: float = 0.0,
        dnf_prob: float = 0.0,
    ) -> _RunResult | None:
        """Run a single strategy simulation using pre-computed ML predictions."""
        return self._common_simulate(race_name, total_laps, driver, strategy,
                                      ml_preds=ml_preds, pit_loss=pit_loss,
                                      rng=rng, sc_prob=sc_prob, dnf_prob=dnf_prob)

    def optimize(
        self,
        race_name: str,
        total_laps: int,
        driver: str,
        mc_runs: int = 1,
        sc_prob: float = 0.0,
        dnf_prob: float = 0.0,
        wet: bool = False,
    ) -> list[_OptResult]:
        # On a dry race, INTERMEDIATE is ~1.2s/lap slower than slicks and never
        # wins — enumerating 1/2/3-stop INTERMEDIATE combinations just wastes
        # Monte Carlo compute (they all rank last and get filtered out). Keep it
        # only when the caller flags the race wet (e.g. rain parsed by the AI
        # assistant). The physics overlay still prices the wet delta correctly
        # for any INTERMEDIATE stint that IS passed in.
        compounds = ["INTERMEDIATE"] if wet else ["SOFT", "MEDIUM", "HARD"]
        step = max(3, total_laps // 15)
        strategies: list[_Strategy] = []
        for c1 in compounds:
            for c2 in compounds:
                for s1 in range(step, total_laps - step, step):
                    strategies.append([(c1, s1), (c2, total_laps - s1)])
        for c1 in compounds[:2]:
            for c2 in compounds[:2]:
                for c3 in compounds:
                    for s1 in range(step, total_laps - 2 * step, 2 * step):
                        for s2 in range(s1 + step, total_laps - step, 2 * step):
                            s3 = total_laps - s1 - s2
                            if s3 >= step:
                                strategies.append([(c1, s1), (c2, s2), (c3, s3)])

        strat_matrices: dict[str, np.ndarray | None] = {}
        for strat in strategies:
            key = str(strat)
            try:
                strat_matrices[key] = self._build_feature_matrix(driver, race_name, strat)
            except Exception:
                strat_matrices[key] = None

        results: list[_OptResult] = []
        for strat in strategies:
            key = str(strat)
            feat_mat = strat_matrices.get(key)
            if feat_mat is not None:
                try:
                    ml_preds = self.xgb_model.predict(feat_mat)
                except Exception:
                    ml_preds = None
            else:
                ml_preds = None
            times: list[float] = []
            for _ in range(mc_runs):
                rng = np.random.RandomState(1234 + _)
                if ml_preds is not None:
                    r = self._simulate_from_ml(race_name, total_laps, driver, strat,
                                                ml_preds, pit_loss=None, rng=rng, sc_prob=sc_prob, dnf_prob=dnf_prob)
                else:
                    r = self.simulate_strategy(race_name, total_laps, driver, strat,
                                               rng=rng, sc_prob=sc_prob, dnf_prob=dnf_prob)
                if r:
                    times.append(r["total_time"])
            if times:
                results.append({
                    "strategy": strat,
                    "mean_time": float(np.mean(times)),
                    "std_time": float(np.std(times)),
                    "min_time": float(np.min(times)),
                    "max_time": float(np.max(times)),
                    "stints": len(strat),
                })

        results.sort(key=lambda r: r["mean_time"])
        return results[:10]

    def strategy_summary(self, race_name: str, total_laps: int,
                         driver: str, strategy: _Strategy) -> str | None:
        r = self.simulate_strategy(race_name, total_laps, driver, strategy)
        if not r:
            return None
        total_sec = r["total_time"]
        mins, secs = divmod(int(total_sec), 60)
        lines = [
            f"Strategy: {' -> '.join([f'{c} ({l}l)' for c, l in strategy])}",
            f"Total: {total_sec:.1f}s ({mins}:{secs:02d}) | Stops: {r['stints']}",
        ]
        for s in r["stint_details"]:
            first, last = s["lap_times"][0], s["lap_times"][-1]
            deg = last - first
            lines.append(
                f"  {s['compound']:8s} {s['laps']:2d}l avg {s['avg_time']:.3f}s  "
                f"{first:.3f}->{last:.3f} deg:{deg:+.3f}"
            )
        return "\n".join(lines)

    def get_detailed_run(self, race_name: str, total_laps: int,
                         driver: str, strategy: _Strategy) -> _RunResult | None:
        return self.simulate_strategy(race_name, total_laps, driver, strategy)


if __name__ == "__main__":
    import time
    t0 = time.time()
    opt = F1StrategyOptimizer()
    print("Drivers:", sorted(opt.driver_offsets.keys()))
    race, laps, driver = "British Grand Prix", 52, "VER"
    print(f"\nOptimizing (MC=100, SC=20%) for {driver} @ {race} ({laps} laps)...")
    t1 = time.time()
    top = opt.optimize(race, laps, driver, mc_runs=100, sc_prob=0.20)
    print(f"Done in {time.time()-t1:.1f}s")
    print(f"\nTOP 5")
    print("=" * 70)
    for i, r in enumerate(top[:5], 1):
        strat = " -> ".join([f"{c} ({l}l)" for c, l in r["strategy"]])
        mins, secs = divmod(int(r["mean_time"]), 60)
        print(f"#{i} {strat}")
        print(f"   Time: {r['mean_time']:.1f}s ({mins}:{secs:02d}) +/- {r['std_time']:.2f}s")
    print(f"\nTotal: {time.time()-t0:.1f}s")
