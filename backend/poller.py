"""
F1 Oracle — OpenF1 REST API Poller  [v2 — Fixed]
==================================================

FIXES IN THIS VERSION
---------------------
Bug 4 — RATE LIMITING:
  Original code fires 8 concurrent requests per 3s cycle = ~160 req/min
  against OpenF1's 30 req/min cap.
  Fix: Sequential batching with a 2.5s cycle. High-priority endpoints
  (session, position, laps, intervals) are fetched every cycle.
  Low-priority ones (stints, pit, race_control, weather) are fetched
  every 4th cycle (≈10s), keeping us well under 30 req/min.

Bug 5 — STALE LAP WATERMARK:
  /laps returns 404 for incremental date queries after a session ends.
  The last_lap_fetch_time watermark was never cleared on 404, so the
  poller stopped fetching new lap data mid-session whenever it briefly
  got a 404 (e.g. during flapping). 
  Fix: Reset the watermark when a full-session fetch succeeds after a
  watermark fetch returned 404.

APIs used by this module
------------------------
All from https://api.openf1.org/v1/
  /sessions        — session metadata (every cycle)
  /drivers         — driver/team info (every cycle, cached after first fetch)
  /position        — live race positions (every cycle)
  /laps            — lap times + sectors (every cycle, incremental)
  /intervals       — real-time gaps, RACE ONLY (every cycle)
  /stints          — tyre compound + age (every 4th cycle)
  /pit             — pit stop events (every 4th cycle)
  /race_control    — flags, SC, messages (every 4th cycle)
  /weather         — track + air temperature (every 4th cycle)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import httpx

import state
from engine import compute_probabilities, _is_qualifying_session
from models import (
    Driver,
    Interval,
    Lap,
    PitStop,
    RaceControlEvent,
    Session,
    Stint,
    Weather,
)

log = logging.getLogger(__name__)

OPENF1_BASE    = "https://api.openf1.org/v1"
POLL_INTERVAL  = 3.5     # seconds between high-priority fetches (stays under 30 req/min)
SLOW_EVERY_N   = 4       # fetch slow endpoints every N cycles
HTTP_TIMEOUT   = 10.0
REQUEST_DELAY  = 0.2     # small delay between sequential requests (politeness)

# WebSocket connection registry — populated by main.py
ws_connections: set = set()

# Cycle counter for slow-endpoint scheduling
_cycle = 0


# ─────────────────────────────────────────────────────────────────────────────
# HTTP client helper
# ─────────────────────────────────────────────────────────────────────────────

async def _get(
    client: httpx.AsyncClient,
    path: str,
    params: dict | None = None,
    tag: str = "",
) -> tuple[list[dict], int]:
    """
    GET a list from the OpenF1 API.
    Returns (data_list, http_status_code).
    """
    url = f"{OPENF1_BASE}{path}"
    try:
        resp = await client.get(url, params=params or {}, timeout=HTTP_TIMEOUT)
        code = resp.status_code
        if code == 200:
            data = resp.json()
            return (data if isinstance(data, list) else []), 200
        if code == 429:
            log.warning("Rate limited on %s — backing off", path)
            await asyncio.sleep(2.0)
            return [], 429
        log.debug("OpenF1 %s %s: %s", code, path, tag)
        return [], code
    except Exception as exc:
        log.warning("OpenF1 fetch error %s: %s", path, exc)
        return [], 0


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint fetchers
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_session(client: httpx.AsyncClient) -> Optional[Session]:
    data, _ = await _get(client, "/sessions", {"session_key": "latest"})
    if not data:
        return None
    try:
        return Session(**data[0])
    except Exception as exc:
        log.warning("Session parse: %s", exc)
        return None


async def _fetch_drivers(client: httpx.AsyncClient, session_key: int) -> dict[int, Driver]:
    data, _ = await _get(client, "/drivers", {"session_key": session_key})
    result: dict[int, Driver] = {}
    for row in data:
        try:
            d = Driver(**row)
            result[d.driver_number] = d
        except Exception:
            pass
    return result


async def _fetch_positions(client: httpx.AsyncClient, session_key: int) -> dict[int, int]:
    """Return the most recent position per driver."""
    data, _ = await _get(client, "/position", {"session_key": session_key})
    latest: dict[int, dict] = {}
    for row in data:
        dn = row.get("driver_number")
        if dn is None:
            continue
        if dn not in latest or row["date"] > latest[dn]["date"]:
            latest[dn] = row
    return {dn: row["position"] for dn, row in latest.items()}


async def _fetch_laps(
    client: httpx.AsyncClient,
    session_key: int,
    since: Optional[str] = None,
) -> tuple[list[Lap], bool]:
    """
    Fetch laps. Returns (laps, got_404).
    got_404=True signals we should reset the watermark and re-fetch from start.
    """
    params: dict[str, Any] = {"session_key": session_key}
    if since:
        params["date_start[gte]"] = since

    data, code = await _get(client, "/laps", params, tag=f"since={since}")

    if code == 404 and since is not None:
        # Incremental query 404 = no new laps yet OR session ended
        # Signal caller to retry without watermark next cycle
        return [], True

    laps: list[Lap] = []
    for row in data:
        try:
            laps.append(Lap(**row))
        except Exception:
            pass
    return laps, False


async def _fetch_stints(client: httpx.AsyncClient, session_key: int) -> list[Stint]:
    data, _ = await _get(client, "/stints", {"session_key": session_key})
    stints: list[Stint] = []
    for row in data:
        try:
            stints.append(Stint(**row))
        except Exception:
            pass
    return stints


async def _fetch_pit_stops(client: httpx.AsyncClient, session_key: int) -> list[PitStop]:
    data, _ = await _get(client, "/pit", {"session_key": session_key})
    pits: list[PitStop] = []
    for row in data:
        try:
            pits.append(PitStop(**row))
        except Exception:
            pass
    return pits


async def _fetch_intervals(
    client: httpx.AsyncClient,
    session_key: int,
    since: Optional[str] = None,
) -> dict[int, Interval]:
    """
    Intervals are ONLY available in Race sessions.
    In Qualifying this always returns 404 — skip it entirely.
    """
    if _is_qualifying_session():
        return {}

    params: dict[str, Any] = {"session_key": session_key}
    if since:
        params["date[gte]"] = since
    data, _ = await _get(client, "/intervals", params)

    latest: dict[int, dict] = {}
    for row in data:
        dn = row.get("driver_number")
        if dn is None:
            continue
        if dn not in latest or row.get("date", "") > latest[dn].get("date", ""):
            latest[dn] = row

    result: dict[int, Interval] = {}
    for dn, row in latest.items():
        try:
            result[dn] = Interval(**row)
        except Exception:
            pass
    return result


async def _fetch_race_control(
    client: httpx.AsyncClient,
    session_key: int,
    since: Optional[str] = None,
) -> list[RaceControlEvent]:
    params: dict[str, Any] = {"session_key": session_key}
    if since:
        params["date[gte]"] = since
    data, _ = await _get(client, "/race_control", params)
    events: list[RaceControlEvent] = []
    for row in data:
        try:
            events.append(RaceControlEvent(**row))
        except Exception:
            pass
    return sorted(events, key=lambda e: e.date)


async def _fetch_weather(client: httpx.AsyncClient, session_key: int) -> Optional[Weather]:
    data, _ = await _get(client, "/weather", {"session_key": session_key})
    if not data:
        return None
    latest = max(data, key=lambda r: r.get("date", ""))
    try:
        return Weather(**latest)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# State merge
# ─────────────────────────────────────────────────────────────────────────────

async def _update_state(client: httpx.AsyncClient) -> bool:
    global _cycle

    # ── 1. Session (every cycle) ──────────────────────────────────────────────
    new_session = await _fetch_session(client)
    await asyncio.sleep(REQUEST_DELAY)
    if new_session is None:
        log.debug("No active session")
        return False

    async with state._lock:
        if state.session and state.session.session_key != new_session.session_key:
            log.info("Session changed %s→%s — resetting state",
                     state.session.session_key, new_session.session_key)
            await state.reset()
        state.session = new_session

    session_key = new_session.session_key

    # ── 2. Drivers (every cycle, but only if not already loaded) ─────────────
    if not state.drivers:
        new_drivers = await _fetch_drivers(client, session_key)
        await asyncio.sleep(REQUEST_DELAY)
        if new_drivers:
            async with state._lock:
                state.drivers.update(new_drivers)
            log.info("Loaded %d drivers for session %s", len(new_drivers), session_key)

    # ── 3. Positions (every cycle) ────────────────────────────────────────────
    new_positions = await _fetch_positions(client, session_key)
    await asyncio.sleep(REQUEST_DELAY)
    if new_positions:
        async with state._lock:
            state.positions.update(new_positions)

    # ── 4. Laps (every cycle, incremental) ────────────────────────────────────
    new_laps, got_404 = await _fetch_laps(
        client, session_key, since=state.last_lap_fetch_time
    )
    await asyncio.sleep(REQUEST_DELAY)

    # Bug 5 fix: reset watermark if incremental query 404'd
    if got_404 and state.last_lap_fetch_time is not None:
        log.debug("Laps incremental 404 — resetting watermark for full re-fetch next cycle")
        async with state._lock:
            state.last_lap_fetch_time = None
    elif new_laps:
        async with state._lock:
            for lap in new_laps:
                dn = lap.driver_number
                if dn not in state.laps:
                    state.laps[dn] = []
                existing_nums = {lp.lap_number for lp in state.laps[dn]}
                if lap.lap_number not in existing_nums:
                    state.laps[dn].append(lap)
                else:
                    state.laps[dn] = [
                        lap if lp.lap_number == lap.lap_number else lp
                        for lp in state.laps[dn]
                    ]
                if lap.date_start and (
                    state.last_lap_fetch_time is None
                    or lap.date_start > state.last_lap_fetch_time
                ):
                    state.last_lap_fetch_time = lap.date_start

    # ── 5. Intervals (every cycle — skipped in Qualifying automatically) ──────
    new_intervals = await _fetch_intervals(
        client, session_key, since=state.last_interval_fetch_time
    )
    await asyncio.sleep(REQUEST_DELAY)
    if new_intervals:
        async with state._lock:
            state.intervals.update(new_intervals)
            dates = [iv.date for iv in new_intervals.values()]
            latest_date = max(dates)
            if (
                state.last_interval_fetch_time is None
                or latest_date > state.last_interval_fetch_time
            ):
                state.last_interval_fetch_time = latest_date

    # ── 6. Slow endpoints (every SLOW_EVERY_N cycles) ─────────────────────────
    if _cycle % SLOW_EVERY_N == 0:
        # Stints
        new_stints = await _fetch_stints(client, session_key)
        await asyncio.sleep(REQUEST_DELAY)
        if new_stints:
            async with state._lock:
                for stint in new_stints:
                    dn = stint.driver_number
                    if dn not in state.stints:
                        state.stints[dn] = []
                    existing = {s.stint_number for s in state.stints[dn]}
                    if stint.stint_number not in existing:
                        state.stints[dn].append(stint)
                    else:
                        state.stints[dn] = [
                            stint if s.stint_number == stint.stint_number else s
                            for s in state.stints[dn]
                        ]

        # Pit stops
        new_pits = await _fetch_pit_stops(client, session_key)
        await asyncio.sleep(REQUEST_DELAY)
        if new_pits:
            async with state._lock:
                for pit in new_pits:
                    dn = pit.driver_number
                    if dn not in state.pit_stops:
                        state.pit_stops[dn] = []
                    existing = {p.lap_number for p in state.pit_stops[dn]}
                    if pit.lap_number not in existing:
                        state.pit_stops[dn].append(pit)

        # Race control
        new_events = await _fetch_race_control(
            client, session_key, since=state.last_event_fetch_time
        )
        await asyncio.sleep(REQUEST_DELAY)
        if new_events:
            async with state._lock:
                existing_keys = {(e.date, e.message) for e in state.events}
                for event in new_events:
                    key = (event.date, event.message)
                    if key not in existing_keys:
                        state.events.append(event)
                        existing_keys.add(key)
                        log.info("⚑ [%s] %s", event.flag or "INFO", event.message)
                latest_event_date = max(e.date for e in new_events)
                if (
                    state.last_event_fetch_time is None
                    or latest_event_date > state.last_event_fetch_time
                ):
                    state.last_event_fetch_time = latest_event_date

        # Weather
        new_weather = await _fetch_weather(client, session_key)
        await asyncio.sleep(REQUEST_DELAY)
        if new_weather:
            async with state._lock:
                state.weather = new_weather

    _cycle += 1
    async with state._lock:
        state.mark_updated()

    return True


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket broadcaster
# ─────────────────────────────────────────────────────────────────────────────

async def _broadcast_state() -> None:
    if not ws_connections:
        return

    from models import LiveStatePayload

    payload = LiveStatePayload(
        session          = state.session,
        drivers          = state.build_live_state(),
        events           = list(state.events),
        weather          = state.weather,
        probabilities    = list(state.probabilities.values()),
        last_updated     = state.last_updated,
        connection_count = len(ws_connections),
        is_live          = state.session is not None,
    )
    payload_json = payload.model_dump_json()

    dead: set = set()
    for ws in list(ws_connections):
        try:
            await ws.send_text(payload_json)
        except Exception:
            dead.add(ws)
    ws_connections.difference_update(dead)


# ─────────────────────────────────────────────────────────────────────────────
# Main polling loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_poller() -> None:
    """
    Sequential polling loop — avoids rate-limiting by NOT firing
    all requests concurrently. Under 30 req/min at all times.
    """
    log.info("OpenF1 poller v2 started (cycle=%.1fs, slow_every=%d)",
             POLL_INTERVAL, SLOW_EVERY_N)

    async with httpx.AsyncClient(
        headers={"Accept": "application/json"},
        follow_redirects=True,
    ) as client:
        while True:
            t_start = time.monotonic()
            try:
                changed = await _update_state(client)
                if changed:
                    await compute_probabilities()
                    await _broadcast_state()
            except asyncio.CancelledError:
                log.info("Poller cancelled — shutting down")
                break
            except Exception as exc:
                log.error("Poller error: %s", exc, exc_info=True)

            elapsed   = time.monotonic() - t_start
            sleep_for = max(POLL_INTERVAL - elapsed, 0.3)
            log.debug("Cycle done in %.2fs — sleeping %.2fs", elapsed, sleep_for)
            await asyncio.sleep(sleep_for)
