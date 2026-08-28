# Plan: Persist FastF1 Car Telemetry to CSV + Dashboard Refactor

> STATUS: Planned, NOT implemented. Scoped 2026-08-28. Deferred from the
> offline-hardening work because it is a multi-part, multi-day effort that
> should not be bolted onto the smaller fixes.

## Problem
- CAR TELEMETRY tab is live-only: `telemetry_loader.resolve_session()` downloads
  per-metre Speed/Throttle/Brake/Gear/DRS traces from FastF1. Those traces are
  never committed, so the tab shows nothing when FastF1 is rate-limited/down.
- Current offline fallback (already shipped) is a *sector-speed comparison* from
  `all_races_master.csv` — explicitly labeled, NOT full car telemetry.
- `dashboard.py` is a single 1026-line `elif`-per-tab file: works, but blocks
  per-tab testing and is a maintenance liability as features grow.

## Goal
1. Persist FastF1 car telemetry (per driver, per lap, distance-indexed: Speed,
   Throttle, Brake, Gear, DRS, plus sector times) to a committed CSV/parquet so
   CAR TELEMETRY can load offline.
2. Split `dashboard.py` into per-tab render modules under `src/tabs/` for
   testability and maintainability.

## Proposed approach

### Part A — Telemetry persistence (data pipeline)
- New script `scripts/build_telemetry_cache.py`:
  - For each (Year, Race) in the schedule (resumable, like build_race_results_csv.py),
    load `R` session with `telemetry=True`, for each driver call
    `lap.get_car_data().add_distance()` and store distance-indexed columns.
  - Write to `data/telemetry/<Year>__<Race>__<Driver>.parquet` (parquet to keep
    size sane; gitignore the raw cache but COMMIT a curated subset, OR upload to
    a release artifact / S3 to avoid repo bloat — decide before building).
  - Size caveat: full telemetry for ~24 races × 20 drivers × ~50 laps × ~300
    distance points = large. Commit only a sample (e.g. fastest lap per driver)
    or use Git LFS / external storage. **This is the key decision.**
- `telemetry_loader.py`: add `load_cached_telemetry(year, race, driver)` that
  reads the committed parquet; `resolve_session()` becomes the fallback for
  races not yet cached.
- CAR TELEMETRY tab: prefer cached telemetry, fall back to live, then to the
  sector-speed comparison.

### Part B — Dashboard refactor
- Move each tab's render function into `src/tabs/<tab>.py` (STRATEGY, CAR,
  STINT, RESULTS, STANDINGS, etc.), keeping `dashboard.py` as the router + shared
  imports (TEAMS, DRIVERS_LIST, _team_name, COMPOUND_COLORS).
- Add per-tab unit tests where feasible (esp. RESULTS/STANDINGS offline paths).
- No behavior change for end users; pure structure move.

## Open decisions (resolve before building)
1. Storage: commit sample parquet vs Git LFS vs external (S3/GCS/release asset).
2. Scope: fastest-lap-per-driver sample only, or full stint traces?
3. Refactor sequencing: do Part A first (highest user value), Part B after.

## Verification when built
- `python scripts/build_telemetry_cache.py` idempotent (resumable).
- CAR TELEMETRY renders offline from committed CSV/parquet.
- `pytest tests/` green; dashboard importable.
- Manual: open dashboard, load CAR TELEMETRY for a cached race with FastF1
  stubbed offline → confirms true offline rendering.
