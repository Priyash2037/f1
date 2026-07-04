"""
F1 Oracle — Monte Carlo Probability Engine  [v2 — Fixed]
==========================================================

ROOT CAUSE AUDIT & FIXES
--------------------------
Bug 1 — QUALIFYING: /intervals returns 404 (race-only endpoint).
  Fix: In Qualifying/Sprint Qualifying, derive "gap to leader" from
       each driver's best lap time vs. the session's best lap time,
       rather than from the interval endpoint.

Bug 2 — SESSION TYPE DETECTION: "Sprint Qualifying" contains neither
  "Race" alone nor plain "Qualifying". All branch logic now uses
  `_is_race_session()` and `_is_qualifying_session()` helpers.

Bug 3 — PROBABILITY COLLAPSE: When all gaps are 0 (no interval data),
  the softmax scores are nearly equal and the driver with the
  marginally fastest lap wins 99% of simulations — e.g. Antonelli
  winning even after Hamilton took pole.
  Fix: Use best_lap_time as the definitive ranking signal in Qualifying.

Bug 4 — RATE LIMITING: 8 concurrent requests per 3s cycle = 160 req/min
  against a 30 req/min cap. Fix is in poller.py (sequential batching
  with jitter). Engine is unchanged here but benefits from better data.

Bug 5 — STALE LAP DATA after session ends: /laps returns 404 for
  incremental queries once a session closes. The incremental watermark
  was preventing re-fetching. Fix: reset watermark if 404 received.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import state

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_SIM           = 500     # Monte Carlo simulations per tick
LAMBDA          = 2.0     # Softmax temperature
LAP_NOISE_STD   = 0.3     # Gaussian noise σ in seconds (tightened from 0.5)
ML_BLEND_ALPHA  = 0.35    # Weight of ML prior vs MC (0 = MC only)

# Compound-specific degradation rates (seconds per lap of tyre age)
COMPOUND_DEG: dict[str, float] = {
    "SOFT":         0.080,
    "MEDIUM":       0.045,
    "HARD":         0.020,
    "INTERMEDIATE": 0.030,
    "WET":          0.015,
    "UNKNOWN":      0.045,
}

PIT_TIME_LOSS = 22.0   # Estimated pit stop time loss (seconds)

# Race lap counts per circuit (for Race sessions)
DEFAULT_RACE_LAPS: dict[str, int] = {
    "bahrain": 57, "jeddah": 50, "albert_park": 58, "suzuka": 53,
    "shanghai": 56, "miami": 57, "imola": 63, "monaco": 78,
    "villeneuve": 70, "catalunya": 66, "red_bull_ring": 71,
    "silverstone": 52, "hungaroring": 70, "spa": 44, "zandvoort": 72,
    "monza": 53, "baku": 51, "marina_bay": 62, "americas": 56,
    "rodriguez": 71, "interlagos": 71, "vegas": 50, "losail": 57,
    "yas_marina": 58,
}
DEFAULT_TOTAL_LAPS = 60


# ─────────────────────────────────────────────────────────────────────────────
# Session type helpers  (Bug 2 fix)
# ─────────────────────────────────────────────────────────────────────────────

def _session_type() -> str:
    """Return the raw session_type string, lower-cased, or empty string."""
    if state.session is None:
        return ""
    return state.session.session_type.lower()


def _is_race_session() -> bool:
    """True for Race and Sprint (but NOT Sprint Qualifying / Practice)."""
    s = _session_type()
    # "Sprint Qualifying" must not match "race"
    # "Sprint" (without Qualifying) is a race-format session
    if "race" in s:
        return True
    # bare "sprint" = race-format sprint race (no laps set, gaps matter)
    if s == "sprint":
        return True
    return False


def _is_qualifying_session() -> bool:
    """True for Q1/Q2/Q3/Sprint Qualifying/Practice — no intervals endpoint."""
    s = _session_type()
    return "qualifying" in s or "practice" in s


# ─────────────────────────────────────────────────────────────────────────────
# Optional ML predictor (lazy load)
# ─────────────────────────────────────────────────────────────────────────────

_ml_predictor = None

def _load_ml_predictor():
    global _ml_predictor
    if _ml_predictor is not None:
        return _ml_predictor

    ml_path = Path(__file__).parent.parent / "ml"
    if ml_path.exists() and str(ml_path) not in sys.path:
        sys.path.insert(0, str(ml_path))

    try:
        from predict import MLPredictor  # type: ignore
        model_path  = ml_path / "models" / "f1_win_s2024_model.joblib"
        scaler_path = ml_path / "models" / "f1_win_s2024_scaler.joblib"
        _ml_predictor = MLPredictor.load(model_path, scaler_path)
    except Exception as exc:
        log.debug("ML predictor not available: %s", exc)
        _ml_predictor = False

    return _ml_predictor


# ─────────────────────────────────────────────────────────────────────────────
# Gap computation — RACE vs QUALIFYING (Bug 1 + 3 fix)
# ─────────────────────────────────────────────────────────────────────────────

def _get_qualifying_gap_seconds(driver_number: int, session_best: Optional[float]) -> float:
    """
    In Qualifying: gap = driver's best lap - session best lap.
    Drivers who haven't set a lap yet receive a large penalty (30s).
    This replaces the /intervals endpoint which only works in Races.
    """
    best = state.get_best_lap_time(driver_number)
    if best is None or session_best is None:
        # No lap set yet — penalise heavily so they rank last
        return 30.0
    return max(best - session_best, 0.0)


def _get_race_gap_seconds(driver_number: int) -> float:
    """
    In Race: gap = real-time gap from /intervals endpoint.
    Returns 0.0 for the leader, None-safe.
    """
    gap = state.get_gap_seconds(driver_number)
    if gap is None:
        # Lapped cars or missing data — penalise
        pos = state.positions.get(driver_number, 20)
        return float(pos) * 3.0   # rough penalty: 3s per position
    return float(gap)


# ─────────────────────────────────────────────────────────────────────────────
# Pace estimation
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_pace(driver_number: int, global_median: float) -> float:
    """
    Estimate a driver's expected clean lap time.
    1. Use median of last 5 clean laps if available.
    2. Fall back to global median adjusted by current position.
    """
    clean_times = state.get_clean_lap_times(driver_number, n=5)
    if len(clean_times) >= 2:
        return float(np.median(clean_times))

    pos = state.positions.get(driver_number, 10)
    return global_median + (pos - 1) * 0.08


def _global_pace_median() -> float:
    """Median clean lap time across all drivers — used as fallback pace."""
    all_clean: list[float] = []
    for d_num in state.drivers:
        all_clean.extend(state.get_clean_lap_times(d_num, n=5))
    if all_clean:
        return float(np.median(all_clean))
    return 90.0


def _get_total_laps() -> int:
    if state.session is None:
        return DEFAULT_TOTAL_LAPS
    circuit = state.session.circuit_short_name.lower().replace(" ", "_")
    return DEFAULT_RACE_LAPS.get(circuit, DEFAULT_TOTAL_LAPS)


# ─────────────────────────────────────────────────────────────────────────────
# Per-driver Monte Carlo simulation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_driver_remaining_time(
    driver_number: int,
    current_gap: float,
    global_median: float,
    total_laps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Run N_SIM simulations of the remaining race/qualifying time for one driver.
    Returns shape (N_SIM,).
    """
    current_lap    = state.get_current_lap(driver_number)
    remaining_laps = max(total_laps - current_lap, 0)

    if remaining_laps == 0:
        return np.full(N_SIM, current_gap)

    base_pace = _estimate_pace(driver_number, global_median)

    # Tyre degradation
    compound = "UNKNOWN"
    tyre_age = state.get_tyre_age(driver_number)
    stint    = state.get_current_stint(driver_number)
    if stint:
        compound = stint.compound
    deg_rate = COMPOUND_DEG.get(compound, COMPOUND_DEG["UNKNOWN"])

    # Remaining pit stops (Race only; 0 in Qualifying)
    if _is_race_session():
        pit_count_done  = state.get_pit_stop_count(driver_number)
        planned_pits    = 2 if total_laps > 40 else 1
        remaining_pits  = max(planned_pits - pit_count_done, 0)
    else:
        remaining_pits  = 0

    # Vectorised simulation across N_SIM × remaining_laps
    noise        = rng.normal(0.0, LAP_NOISE_STD, size=(N_SIM, remaining_laps))
    lap_indices  = np.arange(remaining_laps, dtype=float)
    deg_contrib  = deg_rate * (tyre_age + lap_indices)
    lap_times    = base_pace + deg_contrib[np.newaxis, :] + noise
    lap_times    = np.clip(lap_times, 60.0, 200.0)
    total_time   = lap_times.sum(axis=1) + current_gap

    if remaining_pits > 0:
        pit_noise  = rng.normal(0.0, 2.0, size=(N_SIM, remaining_pits))
        total_time = total_time + (PIT_TIME_LOSS + pit_noise).sum(axis=1)

    return total_time


# ─────────────────────────────────────────────────────────────────────────────
# Qualifying-specific probability (direct best-lap ranking)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_qualifying_probabilities(
    driver_numbers: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For Qualifying / Sprint Qualifying / Practice:
      • Win probability  = probability of having the fastest best lap
        (using MC to model remaining flying laps + pace improvement)
      • Podium prob      = probability of finishing top 3

    We do NOT use /intervals at all here. Instead we rank drivers by
    their current best lap time and apply noise to simulate remaining
    flying laps.
    """
    n = len(driver_numbers)
    rng = np.random.default_rng()

    # Current best lap per driver (None if no clean lap yet)
    best_laps = {
        dn: state.get_best_lap_time(dn)
        for dn in driver_numbers
    }

    # Session best = fastest lap set by anyone
    valid_bests = [t for t in best_laps.values() if t is not None]
    session_best = min(valid_bests) if valid_bests else None

    # Global median pace for fallback
    global_median = _global_pace_median()

    # Each driver may set one more flying lap — simulate that improvement
    # Lap improvement ~ Normal(0, noise) — driver may or may not improve
    improved_bests = np.zeros((N_SIM, n))

    for i, dn in enumerate(driver_numbers):
        current_best = best_laps[dn]
        base = _estimate_pace(dn, global_median)

        if current_best is None:
            # No lap set: simulate flying lap from their pace estimate
            noise = rng.normal(0.0, LAP_NOISE_STD * 2, size=N_SIM)
            improved_bests[:, i] = base + noise
        else:
            # They have a best lap: simulate whether they can improve it
            # ~40% of the time a driver sets a personal best on their final run
            potential = base + rng.normal(0.0, LAP_NOISE_STD, size=N_SIM)
            # Take the better of current best or new attempt
            improved_bests[:, i] = np.minimum(current_best, potential)

    # Rank in each simulation (lower lap time = better position)
    ranks = improved_bests.argsort(axis=1).argsort(axis=1) + 1  # 1-indexed

    win_counts    = (ranks == 1).sum(axis=0).astype(float)
    podium_counts = (ranks <= 3).sum(axis=0).astype(float)
    points_counts = (ranks <= 10).sum(axis=0).astype(float)
    mean_ranks    = ranks.mean(axis=0)

    win_probs    = win_counts / N_SIM
    podium_probs = podium_counts / N_SIM
    points_probs = points_counts / N_SIM

    # Normalise win probs to sum to 1
    total = win_probs.sum()
    if total > 0:
        win_probs = win_probs / total

    return win_probs, podium_probs, points_probs, mean_ranks


# ─────────────────────────────────────────────────────────────────────────────
# Race-specific probability (Monte Carlo with gaps + deg + pits)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_race_probabilities(
    driver_numbers: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(driver_numbers)
    rng = np.random.default_rng()
    total_laps    = _get_total_laps()
    global_median = _global_pace_median()

    all_remaining = np.zeros((N_SIM, n))

    for i, dn in enumerate(driver_numbers):
        gap = _get_race_gap_seconds(dn)
        all_remaining[:, i] = _simulate_driver_remaining_time(
            dn, gap, global_median, total_laps, rng
        )

    # Per-simulation finish order
    ranks = all_remaining.argsort(axis=1).argsort(axis=1) + 1

    win_counts    = (ranks == 1).sum(axis=0).astype(float)
    podium_counts = (ranks <= 3).sum(axis=0).astype(float)
    points_counts = (ranks <= 10).sum(axis=0).astype(float)
    mean_ranks    = ranks.mean(axis=0)

    win_probs    = win_counts / N_SIM
    podium_probs = podium_counts / N_SIM
    points_probs = points_counts / N_SIM

    # Blend raw win-count probs with softmax
    mean_remaining = all_remaining.mean(axis=0)
    scores = -LAMBDA * mean_remaining
    scores -= scores.max()
    softmax_probs = np.exp(scores) / np.exp(scores).sum()

    blended = 0.6 * win_probs + 0.4 * softmax_probs
    total = blended.sum()
    if total > 0:
        blended /= total

    return blended, podium_probs, points_probs, mean_ranks


# ─────────────────────────────────────────────────────────────────────────────
# Softmax (kept for reference / ML blend)
# ─────────────────────────────────────────────────────────────────────────────

def _softmax_probabilities(E_T: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    scores = -lam * E_T
    scores -= scores.max()
    exp_s = np.exp(scores)
    return exp_s / exp_s.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

async def compute_probabilities() -> None:
    """
    Run the correct probability model for the current session type
    and update state.probabilities.

    Session routing:
      • Qualifying / Sprint Qualifying / Practice → _compute_qualifying_probabilities()
      • Race / Sprint (race format)               → _compute_race_probabilities()
    """
    from models import ProbabilityResult

    driver_numbers = list(state.drivers.keys())
    if not driver_numbers:
        return

    loop = asyncio.get_event_loop()
    is_qual = _is_qualifying_session()

    def _run():
        if is_qual:
            return _compute_qualifying_probabilities(driver_numbers)
        else:
            return _compute_race_probabilities(driver_numbers)

    win_probs, podium_probs, points_probs, mean_ranks = await loop.run_in_executor(None, _run)

    # ── Optional ML blend (Race sessions only) ────────────────────────────────
    if not is_qual and _is_race_session():
        ml = _load_ml_predictor()
        if ml:
            try:
                from predict import DriverFeatures  # type: ignore
                ml_feats = [
                    DriverFeatures(
                        driver_number          = dn,
                        starting_grid_position = state.positions.get(dn, 20),
                        constructor_points     = 0.0,
                        driver_points_before   = 0.0,
                        track_temperature      = (
                            state.weather.track_temperature
                            if state.weather and state.weather.track_temperature
                            else 26.0
                        ),
                        pit_stop_count         = float(state.get_pit_stop_count(dn)),
                        fastest_lap_rank       = 99.0,
                    )
                    for dn in driver_numbers
                ]
                ml_dict = ml.predict_win_probabilities(ml_feats)
                ml_arr  = np.array([ml_dict.get(dn, 0.0) for dn in driver_numbers])
                win_probs = ML_BLEND_ALPHA * ml_arr + (1 - ML_BLEND_ALPHA) * win_probs
                s = win_probs.sum()
                if s > 0:
                    win_probs /= s
            except Exception as exc:
                log.debug("ML blend skipped: %s", exc)

    # ── Write results ─────────────────────────────────────────────────────────
    async with state._lock:
        for i, dn in enumerate(driver_numbers):
            state.probabilities[dn] = ProbabilityResult(
                driver_number      = dn,
                win_probability    = float(win_probs[i]),
                podium_probability = float(podium_probs[i]),
                points_probability = float(points_probs[i]),
                expected_finish    = float(mean_ranks[i]),
            )

    log.debug(
        "[%s] Probabilities updated for %d drivers",
        "QUAL" if is_qual else "RACE",
        len(driver_numbers),
    )
