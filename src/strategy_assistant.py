"""
AI Strategy Assistant — answers natural language race strategy questions.

Uses the optimizer, physics engine, and data to provide actionable
insights like "Should I pit this lap?", "What's the fastest strategy?",
"What if a Safety Car appears now?"
"""
from __future__ import annotations

import re
import sys
import os
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
    fuel_burn_rate,
    race_time_estimate,
    sc_delta,
    simulate_sc_scenario,
    track_temp_effect,
    tyre_degradation,
    undercut_benefit,
)
from strategy_optimizer import F1StrategyOptimizer


class StrategyAssistant:
    """AI race engineer that answers strategy questions."""

    def __init__(self, optimizer: F1StrategyOptimizer):
        self.opt = optimizer

    # ── Public Q&A methods ───────────────────────────────────────

    def should_i_pit(self, driver: str, track: str, current_lap: int,
                     total_laps: int, current_compound: str,
                     tyre_age: int, track_temp: float = 35.0,
                     gap_to_next_pit: float = 2.0,
                     position: int = 5) -> str:
        """Answer: should I pit this lap?"""
        laps_remaining = total_laps - current_lap
        if laps_remaining <= 0:
            return "⚠️ Race is over — no need to pit."

        # Current tyre degradation loss
        current_loss = tyre_degradation(current_compound, tyre_age)
        # Degradation in next 3 laps if staying out
        future_loss = tyre_degradation(current_compound, tyre_age + 3) - current_loss

        # Fuel delta (lighter = faster) over next 3 laps
        fuel_delta = fuel_effect(current_lap + 3, total_laps) - fuel_effect(current_lap, total_laps)

        # Cost of staying out 5 more laps (marginal degradation)
        time_lost_staying_out = tyre_degradation(current_compound, tyre_age + 5) - tyre_degradation(current_compound, tyre_age)

        # Benefit of fresh tyres over old tyres for next 5 laps
        fresh_benefit = sum(
            tyre_degradation(current_compound, tyre_age + lap) - tyre_degradation(current_compound, lap)
            for lap in range(1, 6)
        )

        # Net benefit of pitting now vs waiting 5 laps (positive = pit now is better)
        # Pit loss is a sunk cost (eventual pit), so not included here
        net = fresh_benefit - time_lost_staying_out

        if current_compound == "INTERMEDIATE":
            return "💧 Wet tyres — pit only if dry line appears or tyres are grained."

        # Check if undercut is viable
        undercut_check = undercut_benefit(current_lap, total_laps, tyre_age, current_compound, position)
        undercut_works = undercut_check["net_benefit"] > -PIT_LOSS_DEFAULT + 2

        advice = []
        if net > 5:
            advice.append(f"✅ **YES** — pit now. {current_compound} degradation is costing {time_lost_staying_out:.2f}s over 5 more laps.")
        elif net > 0:
            advice.append(f"✅ **PIT SUGGESTED** — fresh tyres will offset the stop loss within ~{PIT_LOSS_DEFAULT / max(fresh_benefit, 0.01):.0f} laps.")
        elif net > -5:
            if tyre_age > 15:
                advice.append("⚠️ **BORDERLINE** — tyres are old but track position may be valuable.")
            else:
                advice.append("⚠️ **NO** — you'd lose more in the pit than you'd gain from fresh tyres right now.")
        else:
            advice.append("❌ **STAY OUT** — pit loss outweighs degradation gain over the next 5 laps.")

        if undercut_works and gap_to_next_pit < 3:
            advice.append(f"📊 Undercut window open — {gap_to_next_pit:.1f}s gap to car ahead.")

        fuel_delta = fuel_effect(current_lap + 3, total_laps) - fuel_effect(current_lap, total_laps)
        if fuel_delta < -0.3:
            advice.append(f"⛽ Fuel load decreasing — you'll gain ~{abs(fuel_delta):.2f}s from fuel burn over 3 laps.")

        deg_info = f"Degradation: +{future_loss:.2f}s in next 3 laps on {current_compound} "
        deg_info += f"(age {tyre_age} laps)"

        answer = (
            f"## Should you pit? (Lap {current_lap} of {total_laps})\n\n"
            + "\n".join(advice)
            + f"\n\n**{deg_info}**"
            + f"\n**Pit window:** ~{PIT_LOSS_DEFAULT}s lost vs ~{fresh_benefit:.1f}s gained from fresh tyres"
        )
        return answer

    def fastest_strategy(self, driver: str, track: str, total_laps: int,
                         mc_runs: int = 50) -> str:
        """What's the fastest strategy for this race?"""
        results = self.opt.optimize(track, total_laps, driver, mc_runs=mc_runs)
        if not results:
            return "⚠️ Could not compute optimal strategy. Try different parameters."

        top = results[0]
        strat = " → ".join([f"{c} ({l}l)" for c, l in top["strategy"]])
        mins, secs = divmod(int(top["mean_time"]), 60)

        lines = [
            f"## 🏆 Optimal Strategy for {driver} @ {track}\n",
            f"**{strat}**",
            f"**Total time:** {top['mean_time']:.1f}s ({mins}min {secs}s) "
            f"(±{top['std_time']:.1f}s across {mc_runs} simulations)",
            f"**Pit stops:** {len(top['strategy']) - 1}\n",
        ]

        if len(results) > 1:
            runner_up = results[1]
            r2_strat = " → ".join([f"{c} ({l}l)" for c, l in runner_up["strategy"]])
            diff = runner_up["mean_time"] - top["mean_time"]
            lines.append(f"**Next best:** {r2_strat} (+{diff:.1f}s)")
            if len(results) > 2:
                r3 = results[2]
                r3_strat = " → ".join([f"{c} ({l}l)" for c, l in r3["strategy"]])
                d3 = r3["mean_time"] - top["mean_time"]
                lines.append(f"**Alt:** {r3_strat} (+{d3:.1f}s)")

        # Stint details
        run = self.opt.get_detailed_run(track, total_laps, driver, top["strategy"])
        if run and run.get("stint_details"):
            lines.append("\n**Stint analysis:**")
            for s in run["stint_details"]:
                first, last = s["lap_times"][0], s["lap_times"][-1]
                lines.append(f"  • {s['compound']:7s} {s['laps']:2d} laps  "
                             f"avg {s['avg_time']:.3f}s  "
                             f"deg {last - first:+.3f}s")

        return "\n".join(lines)

    def what_if_sc(self, driver: str, track: str, total_laps: int,
                   sc_lap: int, sc_duration: int = 3,
                   base_lap_time: float | None = None) -> str:
        """What happens if a Safety Car appears now?"""
        if base_lap_time is None:
            # Get a baseline from the optimizer
            base = self.opt.overall_baseline
        else:
            base = base_lap_time

        result = simulate_sc_scenario(sc_lap, total_laps, base, sc_duration)
        saved = result["time_saved"]

        verdict = "GAIN" if saved > 0 else "LOSS"
        lines = [
            f"## 🚨 Safety Car on Lap {sc_lap}\n",
            f"**Duration:** {sc_duration} laps under SC",
            f"**Green-flag total:** {result['base_total']}s",
            f"**SC total:** {result['sc_total']}s",
            f"**Time delta:** **{saved:+.1f}s {verdict}**\n",
        ]

        if result["free_pit"] and saved > 0:
            lines.append("✅ **Pitting under SC is advantageous** — reduced pit loss "
                         f"({PIT_LOSS_DEFAULT}s → 12s) + field bunches up.")
            lines.append("💡 Best move: pit immediately for fresh tyres and "
                         f"gain track position on the restart.")
        elif saved < -5:
            lines.append("⚠️ SC hurts your race — you lose the gap you built. "
                         "Stay out if track position matters.")
        else:
            lines.append("⏱️ SC has minimal net effect on your race time.")
            lines.append("Consider: pitting to match opponents' strategy.")

        lines.append(f"\n*Based on {sc_duration}-lap SC, base lap {base:.2f}s*")
        return "\n".join(lines)

    def time_loss_if_stay_out(self, driver: str, current_lap: int,
                              total_laps: int, current_compound: str,
                              tyre_age: int, extra_laps: int = 3) -> str:
        """How much time will I lose if I stay out for N more laps?"""
        total_extra = 0
        breakdown: list[str] = []
        for i in range(1, extra_laps + 1):
            lap = current_lap + i
            age = tyre_age + i
            deg_loss = tyre_degradation(current_compound, age) - tyre_degradation(current_compound, tyre_age)
            fuel_gain = fuel_effect(lap, total_laps) - fuel_effect(current_lap, total_laps)
            net = deg_loss + fuel_gain
            total_extra += net
            breakdown.append(f"  Lap {lap}: | {deg_loss:.3f}s deg | {fuel_gain:+.3f}s fuel = {net:+.3f}s")

        # Compare to pitting and coming out on fresh tyres
        fresh_gain_first_laps = compound_delta(current_compound)

        lines = [
            f"## Time cost of {extra_laps} more laps on {current_compound}\n",
            f"**Starting lap:** {current_lap} (tyre age {tyre_age})",
            f"**Total extra time:** {total_extra:+.3f}s\n",
            "**Breakdown:**",
        ]
        lines.extend(breakdown)
        lines.append(f"\n**Verdict:** staying out {extra_laps} more laps costs ~{total_extra:.2f}s total.")
        lines.append(f"Pitting would cost ~{PIT_LOSS_DEFAULT}s but gains back ~{abs(fresh_gain_first_laps):.2f}s/lap initially.")

        if total_extra > PIT_LOSS_DEFAULT * 0.3:
            lines.append("→ ⚠️ **PIT NOW** — degradation exceeds pit loss efficiency.")
        else:
            lines.append("→ ✅ Stretch it a few more laps if track position matters.")

        return "\n".join(lines)

    def which_tyre_next(self, driver: str, track: str, total_laps: int,
                        remaining_laps: int, track_temp: float = 35.0,
                        rainfall: float = 0.0) -> str:
        """Which tyre should I use for the next stint?

        :param rainfall: 0.0 = dry. >0.0 (0.3 damp, 0.7 wet, 1.0+ heavy) forces
            INTERMEDIATE as the only viable option regardless of slick ranking.
        """
        if remaining_laps <= 0:
            return "Race is over — no tyre needed."

        # Rain → intermediates are mandatory; slicks are unsafe.
        if rainfall > 0.3:
            compound = "INTERMEDIATE"
            stint_laps = min(remaining_laps, 20)
            if remaining_laps < 5:
                stint_laps = remaining_laps
            theory = race_time_estimate(stint_laps, self.opt.overall_baseline,
                                        compound, track_temp, rainfall=rainfall)
            times, total = theory
            avg = sum(times) / len(times) if times else 999
            deg = times[-1] - times[0] if len(times) > 1 else 0
            return (
                f"## Next tyre choice ({remaining_laps} laps remaining)\n\n"
                f"🌧️ **Rainfall {rainfall:.1f} — INTERMEDIATE tyres required.**\n\n"
                f"⭐ **INTERMEDIATE** — avg {avg:.3f}s, deg {deg:+.3f}s over "
                f"{stint_laps} laps\n\n"
                f"Dry slicks (SOFT/MEDIUM/HARD) are not viable in these "
                f"conditions — aquaplaning risk and zero grip."
            )

        candidates = []
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            # Simulate a stint on this compound
            stint_laps = min(remaining_laps, 20)
            if remaining_laps < 5:
                stint_laps = remaining_laps

            theory = race_time_estimate(stint_laps, self.opt.overall_baseline,
                                        compound, track_temp)
            times, total = theory
            avg = sum(times) / len(times) if times else 999
            deg = times[-1] - times[0] if len(times) > 1 else 0
            candidates.append({
                "compound": compound,
                "avg_time": avg,
                "total_time": total,
                "deg": deg,
                "max_laps": stint_laps,
            })

        candidates.sort(key=lambda c: c["avg_time"])
        best = candidates[0]
        worst = candidates[-1]

        lines = [
            f"## Next tyre choice ({remaining_laps} laps remaining)\n",
        ]

        for c in candidates:
            marker = "⭐" if c["compound"] == best["compound"] else "  "
            lines.append(
                f"{marker} **{c['compound']}** — avg {c['avg_time']:.3f}s, "
                f"deg {c['deg']:+.3f}s over {c['max_laps']} laps"
            )

        if track_temp > 40:
            lines.append(f"\n🌡️ Track temp {track_temp:.0f}°C — HARD may outperform expectations.")
        elif track_temp < 25:
            lines.append(f"🥶 Track temp {track_temp:.0f}°C — SOFT will struggle to reach temp window.")

        lines.append(f"\n**Recommended:** **{best['compound']}** "
                     f"(avg {best['avg_time']:.3f}s, Δ {best['deg']:+.3f}s deg)")

        return "\n".join(lines)

    def long_run_prediction(self, driver: str, track: str, total_laps: int,
                            compound: str, stint_laps: int,
                            race_fuel_laps: int | None = None) -> str:
        """Full stint prediction for a given compound.

        race_fuel_laps: if provided, the total race length used for fuel
        burn calculations (default: stint_laps, which assumes a standalone
        stint, not a stint within a longer race).
        """
        if race_fuel_laps is None:
            race_fuel_laps = stint_laps
        strategy = [(compound, stint_laps)]
        result = self.opt.get_detailed_run(track, race_fuel_laps, driver, strategy)
        if not result or not result.get("stint_details"):
            return f"⚠️ Could not simulate {compound} stint for {driver} @ {track}."

        sd = result["stint_details"][0]
        times = sd["lap_times"]
        first, last = times[0], times[-1]
        deg_per_lap = (last - first) / max(len(times) - 1, 1)

        lines = [
            f"## 🔬 {compound} Stint Analysis — {driver} @ {track}\n",
            f"**Laps:** {stint_laps}  |  **Avg:** {sd['avg_time']:.3f}s  |  "
            f"**Deg:** {deg_per_lap:.4f}s/lap",
            f"**First lap:** {first:.3f}s  |  **Last lap:** {last:.3f}s  |  "
            f"**Total loss:** {last - first:+.3f}s\n",
            "**Lap-by-lap:**",
        ]
        for i, t in enumerate(times, 1):
            delta = t - times[0]
            lines.append(f"  L{i:2d}  {t:.3f}s  ({delta:+.3f}s)")

        if deg_per_lap > 0.07 and compound != "SOFT":
            lines.append(f"\n⚠️ High degradation for {compound} — "
                         "consider switching to a harder compound.")
        elif deg_per_lap < 0.03:
            lines.append(f"\n✅ Low degradation — {compound} is well-suited to this track.")

        return "\n".join(lines)

    def ask(self, question: str, driver: str = "VER", track: str = "British Grand Prix",
            total_laps: int = 52, current_lap: int = 15,
            current_compound: str = "MEDIUM", tyre_age: int = 10,
            track_temp: float = 35.0, rainfall: float = 0.0) -> str:
        """
        Route a natural language question to the right handler.

        ``rainfall`` is a 0..1+ intensity (0.3 damp, 0.7 wet, 1.0+ heavy). When
        the caller already knows the weather it can be passed in directly; the
        router also tries to read it out of phrases like "rain", "wet",
        "intermediate" so the natural-language path can reach the wet-tyre
        advice in which_tyre_next().
        """
        q = question.lower()

        # ── Parse rainfall from the question when not (reliably) supplied ──
        parsed_rain = rainfall
        if parsed_rain <= 0.0:
            if any(k in q for k in ("heavy rain", "heavy wet", "torrential")):
                parsed_rain = 1.0
            elif any(k in q for k in ("intermediate", "inters", "wet tyre", "wet tire", "rain")):
                parsed_rain = 0.7
            elif "damp" in q:
                parsed_rain = 0.3
        # Also catch "rainfall 0.X" style explicit numbers in the question.
        rain_num = re.findall(r"rainfall?\s*[:=]?\s*([0-9]*\.?[0-9]+)", q)
        if rain_num:
            try:
                parsed_rain = float(rain_num[0])
            except ValueError:
                pass

        if "pit" in q and ("should" in q or "now" in q or "when" in q):
            return self.should_i_pit(driver, track, current_lap, total_laps,
                                     current_compound, tyre_age, track_temp)
        if "fastest" in q or "best strategy" in q or "optimal" in q:
            return self.fastest_strategy(driver, track, total_laps)
        if ("safety car" in q) or (("sc" in q) and ("lap" in q or "scenario" in q or "deploy" in q)):
            sc_lap = current_lap
            # Try to extract a specific lap number
            nums = re.findall(r"lap (\d+)", q)
            if nums:
                sc_lap = int(nums[0])
            duration = re.findall(r"(\d+) (?:lap|duration)", q)
            sc_dur = int(duration[0]) if duration else 3
            return self.what_if_sc(driver, track, total_laps, sc_lap, sc_dur)
        if "stay out" in q or "stay" in q or "stretch" in q:
            extra = 3
            nums = re.findall(r"(\d+) (?:more|extra|additional)", q)
            if nums:
                extra = int(nums[0])
            return self.time_loss_if_stay_out(driver, current_lap, total_laps,
                                              current_compound, tyre_age, extra)
        # Wet / intermediate tyre routing: if rainfall was parsed (or asked
        # about rain directly) the next-tyre question must respect it.
        if parsed_rain > 0.3:
            return self.which_tyre_next(driver, track, total_laps,
                                        total_laps - current_lap, track_temp,
                                        rainfall=parsed_rain)
        if "tyre" in q or "tire" in q or "next" in q or "compound" in q:
            return self.which_tyre_next(driver, track, total_laps,
                                        total_laps - current_lap, track_temp,
                                        rainfall=parsed_rain)
        if "stint" in q or "long run" in q:
            stint_len = 20
            nums = re.findall(r"(\d+) (?:lap|stint)", q)
            if nums:
                stint_len = int(nums[0])
            return self.long_run_prediction(driver, track, total_laps,
                                            current_compound, stint_len)
        if "compare" in q:
            return self.fastest_strategy(driver, track, total_laps)

        # Fallback: general analysis
        return (
            f"## 🏎️ Strategy Analysis — {driver} @ {track}\n\n"
            f"*Estimates below come from a physics + ML blend (XGBoost pace/degradation "
            f"plus a compound-speed physics overlay), not live timing.*\n\n"
            f"I can answer questions like:\n"
            f"- *Should I pit this lap?*\n"
            f"- *What's the fastest strategy for this race?*\n"
            f"- *What if a Safety Car appears on lap {current_lap}?*\n"
            f"- *How much time will I lose if I stay out 3 more laps?*\n"
            f"- *Which tyre should I use for the next stint?*\n"
            f"- *Simulate a 20-lap stint on SOFT*\n\n"
            f"Current race state: Lap {current_lap}/{total_laps}, "
            f"{current_compound} (age {tyre_age}), {track}"
        )
