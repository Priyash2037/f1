"""
F1 Oracle — In-Memory State Store
====================================
Thread-safe (asyncio-lock-guarded) in-memory store for the current
session's live data. All state lives here; the poller writes it,
the engine reads it, WebSocket clients consume it.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Optional

from models import (
    Driver,
    DriverLiveState,
    Interval,
    Lap,
    PitStop,
    ProbabilityResult,
    RaceControlEvent,
    Session,
    Stint,
    Weather,
)

# ─────────────────────────────────────────────────────────────────────────────
# Global state containers
# ─────────────────────────────────────────────────────────────────────────────

# Raw data from OpenF1
session:     Optional[Session]          = None
drivers:     dict[int, Driver]          = {}       # driver_number → Driver
positions:   dict[int, int]             = {}       # driver_number → position
laps:        dict[int, list[Lap]]       = {}       # driver_number → [Lap, …]
stints:      dict[int, list[Stint]]     = {}       # driver_number → [Stint, …]
pit_stops:   dict[int, list[PitStop]]   = {}       # driver_number → [PitStop, …]
intervals:   dict[int, Interval]        = {}       # driver_number → Interval
events:      deque[RaceControlEvent]    = deque(maxlen=30)
weather:     Optional[Weather]          = None

# Computed by the probability engine
probabilities: dict[int, ProbabilityResult] = {}

# Timestamps for incremental fetching
last_lap_fetch_time:      Optional[str] = None
last_interval_fetch_time: Optional[str] = None
last_event_fetch_time:    Optional[str] = None
last_updated:             float         = 0.0

# Asyncio lock — all writes must acquire this lock
_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

async def reset() -> None:
    """Clear all state (called when a new session is detected)."""
    global session, drivers, positions, laps, stints, pit_stops, intervals
    global events, weather, probabilities
    global last_lap_fetch_time, last_interval_fetch_time, last_event_fetch_time
    async with _lock:
        session    = None
        drivers    = {}
        positions  = {}
        laps       = {}
        stints     = {}
        pit_stops  = {}
        intervals  = {}
        events     = deque(maxlen=30)
        weather    = None
        probabilities = {}
        last_lap_fetch_time = None
        last_interval_fetch_time = None
        last_event_fetch_time = None


def get_current_lap(driver_number: int) -> int:
    """Return the latest known lap number for a driver."""
    driver_laps = laps.get(driver_number, [])
    if not driver_laps:
        return 0
    return max(lp.lap_number for lp in driver_laps)


def get_clean_lap_times(driver_number: int, n: int = 5) -> list[float]:
    """
    Return the N most recent clean (non-pit-out) lap durations for a driver.
    Clean = is_pit_out_lap is False AND lap_duration is not None.
    """
    driver_laps = sorted(laps.get(driver_number, []), key=lambda x: x.lap_number)
    clean = [
        lp.lap_duration
        for lp in driver_laps
        if not lp.is_pit_out_lap and lp.lap_duration is not None
    ]
    return clean[-n:]


def get_best_lap_time(driver_number: int) -> Optional[float]:
    """Return the personal best lap time for a driver."""
    times = get_clean_lap_times(driver_number, n=100)
    return min(times) if times else None


def get_overall_best_lap_time() -> Optional[float]:
    """Return the absolute fastest lap time across all drivers."""
    all_times = [
        t
        for d_num in drivers
        for t in get_clean_lap_times(d_num, n=100)
    ]
    return min(all_times) if all_times else None


def get_current_stint(driver_number: int) -> Optional[Stint]:
    """Return the driver's current (most recent) tyre stint."""
    driver_stints = stints.get(driver_number, [])
    if not driver_stints:
        return None
    return max(driver_stints, key=lambda s: s.stint_number)


def get_tyre_age(driver_number: int) -> int:
    """Return current tyre age in laps."""
    stint = get_current_stint(driver_number)
    if not stint:
        return 0
    current = get_current_lap(driver_number)
    return max(current - stint.lap_start, 0) + stint.tyre_age_at_start


def get_pit_stop_count(driver_number: int) -> int:
    """Return total pit stops completed by driver."""
    return len(pit_stops.get(driver_number, []))


def _is_qualifying() -> bool:
    """True when current session is qualifying / sprint qualifying / practice."""
    if session is None:
        return False
    s = session.session_type.lower()
    return "qualifying" in s or "practice" in s


def get_qualifying_best_lap_gap(driver_number: int) -> Optional[str]:
    """
    In Qualifying: gap = driver best lap - session best lap.
    Returns None if driver has not set a lap yet (shown as '—' in UI).
    """
    best = get_best_lap_time(driver_number)
    if best is None:
        return None
    # Session best = minimum across all drivers
    all_bests = [
        get_best_lap_time(dn)
        for dn in drivers
        if get_best_lap_time(dn) is not None
    ]
    if not all_bests:
        return None
    session_best = min(all_bests)
    delta = best - session_best
    if delta < 0.001:
        return "POLE"
    return f"+{delta:.3f}s"


def get_gap_to_leader(driver_number: int) -> Optional[str]:
    """
    Return gap to leader as a formatted string.
    In Qualifying → best-lap delta (intervals not available).
    In Race       → real-time gap from /intervals.
    """
    if _is_qualifying():
        return get_qualifying_best_lap_gap(driver_number)

    interval = intervals.get(driver_number)
    if interval is None:
        return None
    gap = interval.gap_to_leader
    if gap is None:
        return "Leader"
    if isinstance(gap, str):
        return gap
    return f"+{gap:.3f}s"


def get_gap_seconds(driver_number: int) -> Optional[float]:
    """Return gap to leader in seconds (numeric). Returns None if a lap down."""
    if _is_qualifying():
        best = get_best_lap_time(driver_number)
        all_bests = [get_best_lap_time(dn) for dn in drivers if get_best_lap_time(dn) is not None]
        if best is None or not all_bests:
            return None
        return max(best - min(all_bests), 0.0)

    interval = intervals.get(driver_number)
    if interval is None:
        return None
    gap = interval.gap_to_leader
    if gap is None:
        return 0.0   # leader
    if isinstance(gap, str):
        return None  # "+1 LAP"
    return float(gap)


def build_live_state() -> list[DriverLiveState]:
    """
    Merge all state sources into a unified DriverLiveState list.

    Sort order:
      Qualifying: by best lap time (ascending). Drivers with no lap → end.
      Race:       by current position from /position endpoint.
    """
    overall_best = get_overall_best_lap_time()
    qual_mode    = _is_qualifying()
    states: list[DriverLiveState] = []

    for driver_number, driver in drivers.items():
        pos       = positions.get(driver_number, 99)
        stint     = get_current_stint(driver_number)
        last_laps = get_clean_lap_times(driver_number, n=1)
        last_lap  = last_laps[-1] if last_laps else None
        best_lap  = get_best_lap_time(driver_number)
        prob      = probabilities.get(driver_number)

        is_personal_best = (
            last_lap is not None
            and best_lap is not None
            and abs(last_lap - best_lap) < 0.001
        )
        is_overall_best = (
            last_lap is not None
            and overall_best is not None
            and abs(last_lap - overall_best) < 0.001
        )

        # In Qualifying, the /position endpoint gives the provisional
        # qualifying position (updated as laps are set) — this is correct.
        # We just make sure not to use /intervals for the gap display.
        gap_display = get_gap_to_leader(driver_number)

        states.append(
            DriverLiveState(
                driver_number      = driver_number,
                name_acronym       = driver.name_acronym,
                full_name          = driver.full_name,
                team_name          = driver.team_name,
                team_colour        = driver.team_colour,
                headshot_url       = driver.headshot_url,
                position           = pos,
                gap_to_leader      = gap_display,
                interval           = (
                    f"+{intervals[driver_number].interval:.3f}s"
                    if driver_number in intervals
                    and isinstance(intervals[driver_number].interval, float)
                    else None
                ),
                current_lap        = get_current_lap(driver_number),
                last_lap_time      = last_lap,
                best_lap_time      = best_lap,
                is_personal_best   = is_personal_best,
                is_overall_best    = is_overall_best,
                compound           = stint.compound if stint else "UNKNOWN",
                tyre_age           = get_tyre_age(driver_number),
                pit_stop_count     = get_pit_stop_count(driver_number),
                win_probability    = prob.win_probability if prob else 0.0,
                podium_probability = prob.podium_probability if prob else 0.0,
                expected_finish    = prob.expected_finish if prob else float(pos),
                dnf_probability    = prob.dnf_probability if prob else 0.05,
                tyre_life_remaining_percent = prob.tyre_life_remaining_percent if prob else 100.0,
            )
        )

    if qual_mode:
        # Primary sort: by position (set correctly by /position in qualifying)
        # Secondary sort: best lap time to tiebreak (None = no lap → last)
        return sorted(
            states,
            key=lambda s: (
                s.position,
                s.best_lap_time if s.best_lap_time is not None else 9999.0,
            )
        )

    return sorted(states, key=lambda s: s.position)


def mark_updated() -> None:
    """Update the last_updated timestamp."""
    global last_updated
    last_updated = time.time()
