"""
F1 Oracle — Pydantic Data Models
==================================
Defines all data shapes flowing through the system:
  - OpenF1 API response shapes
  - Internal state types
  - WebSocket broadcast payload
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# OpenF1 API response shapes
# ─────────────────────────────────────────────────────────────────────────────

class Session(BaseModel):
    session_key: int
    session_type: str            # "Race", "Qualifying", "Sprint", etc.
    session_name: str
    date_start: str
    date_end: str
    meeting_key: int
    circuit_short_name: str
    country_name: str
    location: str
    year: int
    is_cancelled: bool = False


class Driver(BaseModel):
    meeting_key: int
    session_key: int
    driver_number: int
    broadcast_name: str
    full_name: str
    name_acronym: str
    team_name: str
    team_colour: str = "FFFFFF"
    first_name: str
    last_name: str
    headshot_url: Optional[str] = None
    country_code: Optional[str] = None


class PositionEntry(BaseModel):
    date: str
    session_key: int
    driver_number: int
    position: int
    meeting_key: int


class Lap(BaseModel):
    meeting_key: int
    session_key: int
    driver_number: int
    lap_number: int
    date_start: Optional[str] = None
    lap_duration: Optional[float] = None
    duration_sector_1: Optional[float] = None
    duration_sector_2: Optional[float] = None
    duration_sector_3: Optional[float] = None
    i1_speed: Optional[int] = None
    i2_speed: Optional[int] = None
    st_speed: Optional[int] = None
    is_pit_out_lap: bool = False


class Stint(BaseModel):
    meeting_key: int
    session_key: int
    driver_number: int
    stint_number: int
    lap_start: int
    lap_end: Optional[int] = None
    compound: str = "UNKNOWN"      # SOFT | MEDIUM | HARD | INTERMEDIATE | WET
    tyre_age_at_start: int = 0


class PitStop(BaseModel):
    date: str
    meeting_key: int
    session_key: int
    driver_number: int
    lap_number: int
    pit_duration: Optional[float] = None
    stop_number: Optional[int] = None


class Interval(BaseModel):
    date: str
    session_key: int
    driver_number: int
    gap_to_leader: Optional[float | str] = None   # may be "+1 LAP" string
    interval: Optional[float | str] = None


class RaceControlEvent(BaseModel):
    date: str
    session_key: int
    message: str
    flag: Optional[str] = None      # GREEN | YELLOW | RED | SC | VSC | CHEQUERED
    category: Optional[str] = None
    scope: Optional[str] = None
    sector: Optional[int] = None
    driver_number: Optional[int] = None


class Weather(BaseModel):
    date: str
    session_key: int
    air_temperature: Optional[float] = None
    track_temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    rainfall: Optional[int] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# Computed / enriched types
# ─────────────────────────────────────────────────────────────────────────────

class ProbabilityResult(BaseModel):
    driver_number: int
    win_probability: float = 0.0
    podium_probability: float = 0.0
    points_probability: float = 0.0
    expected_finish: float = 10.0   # Expected finishing position


class DriverLiveState(BaseModel):
    """Merged snapshot of a driver's current state during a session."""
    driver_number: int
    name_acronym: str = ""
    full_name: str = ""
    team_name: str = ""
    team_colour: str = "FFFFFF"
    headshot_url: Optional[str] = None
    position: int = 99
    gap_to_leader: Optional[str] = None
    interval: Optional[str] = None
    current_lap: int = 0
    last_lap_time: Optional[float] = None
    best_lap_time: Optional[float] = None
    is_personal_best: bool = False
    is_overall_best: bool = False
    compound: str = "UNKNOWN"
    tyre_age: int = 0
    pit_stop_count: int = 0
    is_in_pit: bool = False
    win_probability: float = 0.0
    podium_probability: float = 0.0
    expected_finish: float = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket broadcast payload
# ─────────────────────────────────────────────────────────────────────────────

class LiveStatePayload(BaseModel):
    """Full state snapshot broadcast to all WebSocket clients every ~3s."""
    session: Optional[Session] = None
    drivers: list[DriverLiveState] = Field(default_factory=list)
    events: list[RaceControlEvent] = Field(default_factory=list)
    weather: Optional[Weather] = None
    probabilities: list[ProbabilityResult] = Field(default_factory=list)
    last_updated: float = 0.0
    connection_count: int = 0
    is_live: bool = False
