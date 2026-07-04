"""
F1 Oracle — Data Preprocessing Pipeline
========================================
Fetches historical F1 race data from the Ergast Developer API,
engineers features, handles missing pit-stop values, and normalises
the numeric feature set ready for XGBoost training.

Usage
-----
    python preprocess.py                  # fetch + save processed CSV
    python preprocess.py --csv data.csv   # use pre-downloaded CSV

Output
------
    processed/features.csv   — model-ready feature matrix (X)
    processed/targets.csv    — win / podium labels (y)
    processed/scaler.joblib  — fitted StandardScaler for inference-time reuse
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ERGAST_BASE = "https://ergast.com/api/f1"
OUTPUT_DIR = Path("processed")

# Seasons to fetch (inclusive). Ergast goes back to 1950; XGBoost works best
# with >= 10 seasons. Adjust as needed.
SEASON_START = 2010
SEASON_END = 2024          # Last COMPLETE season used for test split

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Ergast API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3, backoff: float = 2.0) -> dict:
    """GET with retry / backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = backoff ** attempt
            log.warning("Request failed (%s). Retrying in %.1fs…", exc, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def fetch_race_results(season: int) -> list[dict]:
    """Fetch all race results for a season. Returns flat list of row dicts."""
    url = f"{ERGAST_BASE}/{season}/results.json?limit=1000"
    data = _get(url)
    races = data["MRData"]["RaceTable"]["Races"]
    rows: list[dict] = []
    for race in races:
        round_num = int(race["round"])
        race_name = race["raceName"]
        circuit = race["Circuit"]["circuitId"]
        for result in race["Results"]:
            driver_id = result["Driver"]["driverId"]
            constructor_id = result["Constructor"]["constructorId"]
            grid = int(result.get("grid") or 0)
            position_text = result.get("positionText", "R")
            try:
                finish_pos = int(result["position"])
            except (KeyError, ValueError):
                finish_pos = 99   # DNF / DSQ / DNS
            status = result.get("status", "Unknown")
            fastest_lap_rank = None
            if "FastestLap" in result:
                try:
                    fastest_lap_rank = int(result["FastestLap"].get("rank", 99))
                except ValueError:
                    pass
            rows.append(
                {
                    "season": season,
                    "round": round_num,
                    "race_name": race_name,
                    "circuit_id": circuit,
                    "driver_id": driver_id,
                    "constructor_id": constructor_id,
                    "starting_grid_position": grid,
                    "finish_position": finish_pos,
                    "status": status,
                    "fastest_lap_rank": fastest_lap_rank,
                    "position_text": position_text,
                }
            )
    return rows


def fetch_pit_stops(season: int, round_num: int) -> dict[str, int]:
    """Return {driverId: pit_stop_count} for one race. Empty dict on failure."""
    url = f"{ERGAST_BASE}/{season}/{round_num}/pitstops.json?limit=200"
    try:
        data = _get(url)
        stops = data["MRData"]["RaceTable"]["Races"]
        if not stops:
            return {}
        counts: dict[str, int] = {}
        for stop in stops[0].get("PitStops", []):
            did = stop["driverId"]
            counts[did] = counts.get(did, 0) + 1
        return counts
    except Exception as exc:
        log.warning("Pit stop fetch failed (season=%d round=%d): %s", season, round_num, exc)
        return {}


def fetch_driver_standings_before_round(season: int, round_num: int) -> dict[str, float]:
    """Return {driverId: championship_points} after round_num-1."""
    if round_num <= 1:
        return {}
    url = f"{ERGAST_BASE}/{season}/{round_num - 1}/driverStandings.json"
    try:
        data = _get(url)
        standings = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not standings:
            return {}
        return {
            s["Driver"]["driverId"]: float(s["points"])
            for s in standings[0]["DriverStandings"]
        }
    except Exception as exc:
        log.warning("Driver standings fetch failed: %s", exc)
        return {}


def fetch_constructor_standings_before_round(season: int, round_num: int) -> dict[str, float]:
    """Return {constructorId: championship_points} after round_num-1."""
    if round_num <= 1:
        return {}
    url = f"{ERGAST_BASE}/{season}/{round_num - 1}/constructorStandings.json"
    try:
        data = _get(url)
        standings = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not standings:
            return {}
        return {
            s["Constructor"]["constructorId"]: float(s["points"])
            for s in standings[0]["ConstructorStandings"]
        }
    except Exception as exc:
        log.warning("Constructor standings fetch failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Full data fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_seasons(start: int = SEASON_START, end: int = SEASON_END) -> pd.DataFrame:
    """
    Fetch race results, pit stops, and standings for every season in [start, end].
    Returns a raw DataFrame before any feature engineering.
    """
    all_rows: list[dict] = []
    for season in range(start, end + 1):
        log.info("Fetching season %d…", season)
        results = fetch_race_results(season)

        # Group results by round so we can batch-fetch pit stops / standings
        rounds: dict[int, list[dict]] = {}
        for row in results:
            rounds.setdefault(row["round"], []).append(row)

        for round_num, race_rows in sorted(rounds.items()):
            log.info("  Round %d/%d — %d drivers", round_num, max(rounds), len(race_rows))

            pit_stops = fetch_pit_stops(season, round_num)
            driver_pts = fetch_driver_standings_before_round(season, round_num)
            ctor_pts = fetch_constructor_standings_before_round(season, round_num)

            for row in race_rows:
                row["pit_stop_count"] = pit_stops.get(row["driver_id"])   # None if missing
                row["driver_points_before"] = driver_pts.get(row["driver_id"], 0.0)
                row["constructor_points"] = ctor_pts.get(row["constructor_id"], 0.0)
            all_rows.extend(race_rows)

            time.sleep(0.25)   # Be polite to the Ergast API

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

# Mapping of approximate average track temperatures (°C) by circuit ID.
# Source: historical race-day averages. Used when live weather is unavailable.
TRACK_TEMP_MAP: dict[str, float] = {
    "bahrain": 32.0, "jeddah": 30.0, "albert_park": 22.0, "suzuka": 18.0,
    "shanghai": 20.0, "miami": 34.0, "imola": 24.0, "monaco": 28.0,
    "villeneuve": 25.0, "catalunya": 33.0, "red_bull_ring": 28.0,
    "silverstone": 25.0, "hungaroring": 35.0, "spa": 20.0, "zandvoort": 22.0,
    "monza": 26.0, "baku": 25.0, "marina_bay": 32.0, "americas": 28.0,
    "rodriguez": 22.0, "interlagos": 26.0, "vegas": 15.0, "losail": 28.0,
    "yas_marina": 28.0,
}

# Canonical pit-stop counts per circuit (historical median).
# Fallback when Ergast pit stop data is missing.
PIT_STOP_MEDIAN_MAP: dict[str, float] = {
    "bahrain": 2.0, "jeddah": 2.0, "albert_park": 2.0, "suzuka": 1.0,
    "shanghai": 2.0, "miami": 2.0, "imola": 1.0, "monaco": 2.0,
    "villeneuve": 2.0, "catalunya": 2.0, "red_bull_ring": 2.0,
    "silverstone": 2.0, "hungaroring": 2.0, "spa": 2.0, "zandvoort": 2.0,
    "monza": 1.0, "baku": 2.0, "marina_bay": 3.0, "americas": 2.0,
    "rodriguez": 2.0, "interlagos": 2.0, "vegas": 2.0, "losail": 2.0,
    "yas_marina": 2.0,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature transformations to the raw DataFrame.

    Steps
    -----
    1.  Clip grid position — pit lane starts (pos 0) → 21.
    2.  Add track_temperature from lookup map (imputed for historical data).
    3.  Impute missing pit_stop_count values with three-level strategy:
          a) Use circuit median from PIT_STOP_MEDIAN_MAP.
          b) If circuit unknown, use per-season median.
          c) If still missing, global median.
    4.  Encode circuit_id as integer (label encoding).
    5.  Encode constructor_id as integer.
    6.  Compute win_flag and podium_flag labels.
    7.  Clamp extreme driver/constructor points at 99th percentile.
    8.  Add derived features:
          - grid_squared  (non-linear grid advantage)
          - points_ratio  (driver / (constructor + 1))
          - is_pole       (grid == 1)
          - is_front_row  (grid <= 2)
    """
    df = df.copy()

    # 1. Fix pit-lane starts (grid = 0) → last position proxy (21)
    df["starting_grid_position"] = df["starting_grid_position"].replace(0, 21)
    df["starting_grid_position"] = df["starting_grid_position"].clip(1, 22)

    # 2. Track temperature (imputed from lookup)
    df["track_temperature"] = df["circuit_id"].map(TRACK_TEMP_MAP).fillna(
        df["circuit_id"].map(TRACK_TEMP_MAP).median()   # global fallback
    ).fillna(26.0)

    # 3. Pit stop imputation ── three-level strategy
    circuit_pit_median = df.groupby("circuit_id")["pit_stop_count"].transform(
        lambda s: s.median()
    )
    season_pit_median = df.groupby("season")["pit_stop_count"].transform(
        lambda s: s.median()
    )
    global_pit_median = df["pit_stop_count"].median()
    if pd.isna(global_pit_median):
        global_pit_median = 2.0

    # Level a: circuit lookup map
    df["pit_stop_count"] = df.apply(
        lambda row: (
            row["pit_stop_count"]
            if pd.notna(row["pit_stop_count"])
            else PIT_STOP_MEDIAN_MAP.get(row["circuit_id"])
        ),
        axis=1,
    )
    # Level b: circuit-based median from data
    df["pit_stop_count"] = df["pit_stop_count"].fillna(circuit_pit_median)
    # Level c: season median
    df["pit_stop_count"] = df["pit_stop_count"].fillna(season_pit_median)
    # Level d: global median
    df["pit_stop_count"] = df["pit_stop_count"].fillna(global_pit_median)
    df["pit_stop_count"] = df["pit_stop_count"].astype(float)

    # 4. Label-encode circuit_id
    circuit_codes, _ = pd.factorize(df["circuit_id"])
    df["circuit_encoded"] = circuit_codes.astype(int)

    # 5. Label-encode constructor_id
    ctor_codes, _ = pd.factorize(df["constructor_id"])
    df["constructor_encoded"] = ctor_codes.astype(int)

    # 6. Target labels
    df["win_flag"] = (df["finish_position"] == 1).astype(int)
    df["podium_flag"] = (df["finish_position"] <= 3).astype(int)
    df["points_flag"] = (df["finish_position"] <= 10).astype(int)

    # 7. Clamp outlier points values
    for col in ["driver_points_before", "constructor_points"]:
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=p99)

    # 8. Derived features
    df["grid_squared"] = df["starting_grid_position"] ** 2
    df["points_ratio"] = df["driver_points_before"] / (df["constructor_points"] + 1.0)
    df["is_pole"] = (df["starting_grid_position"] == 1).astype(int)
    df["is_front_row"] = (df["starting_grid_position"] <= 2).astype(int)
    df["fastest_lap_rank"] = df["fastest_lap_rank"].fillna(99).astype(float)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns (used by both preprocessing and inference)
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES: list[str] = [
    "starting_grid_position",
    "grid_squared",
    "constructor_points",
    "driver_points_before",
    "points_ratio",
    "track_temperature",
    "pit_stop_count",
    "fastest_lap_rank",
    "circuit_encoded",
    "constructor_encoded",
    "round",
    "is_pole",
    "is_front_row",
]

TARGET_WIN = "win_flag"
TARGET_PODIUM = "podium_flag"
TARGET_POINTS = "points_flag"


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_scale(
    df: pd.DataFrame, scaler: Optional[StandardScaler] = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Fit (or reuse) a StandardScaler on NUMERIC_FEATURES.
    Returns the scaled DataFrame slice and the fitted scaler.
    """
    X = df[NUMERIC_FEATURES].copy()

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X)

    scaled = scaler.transform(X)
    df_scaled = pd.DataFrame(scaled, columns=NUMERIC_FEATURES, index=df.index)
    return df_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(csv_path: Optional[str] = None) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Load raw data ──────────────────────────────────────────────────────
    if csv_path:
        log.info("Loading raw data from %s", csv_path)
        raw_df = pd.read_csv(csv_path)
    else:
        log.info("Fetching data from Ergast API (seasons %d–%d)…", SEASON_START, SEASON_END)
        raw_df = fetch_all_seasons()
        raw_path = OUTPUT_DIR / "raw_data.csv"
        raw_df.to_csv(raw_path, index=False)
        log.info("Raw data saved → %s  (%d rows)", raw_path, len(raw_df))

    log.info("Raw data shape: %s", raw_df.shape)

    # ── Feature engineering ────────────────────────────────────────────────
    log.info("Engineering features…")
    df = engineer_features(raw_df)

    # Drop rows where we couldn't determine finish position (DNS / DSQ edge cases)
    df = df[df["finish_position"] < 99].copy()
    log.info("After DNF filter: %d rows", len(df))

    # ── Normalise ──────────────────────────────────────────────────────────
    log.info("Fitting StandardScaler on %d numeric features…", len(NUMERIC_FEATURES))
    X_scaled, scaler = fit_and_scale(df)

    # Re-attach metadata columns needed for train/test splitting
    meta_cols = ["season", "round", "race_name", "circuit_id", "driver_id"]
    df_out = pd.concat(
        [df[meta_cols].reset_index(drop=True), X_scaled.reset_index(drop=True)],
        axis=1,
    )

    # ── Save outputs ───────────────────────────────────────────────────────
    features_path = OUTPUT_DIR / "features.csv"
    df_out.to_csv(features_path, index=False)
    log.info("Feature matrix saved → %s  (%s)", features_path, df_out.shape)

    targets_path = OUTPUT_DIR / "targets.csv"
    df[[TARGET_WIN, TARGET_PODIUM, TARGET_POINTS, "season", "round", "driver_id"]].to_csv(
        targets_path, index=False
    )
    log.info("Target labels saved → %s", targets_path)

    scaler_path = OUTPUT_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    log.info("Scaler saved → %s", scaler_path)

    # ── Summary statistics ─────────────────────────────────────────────────
    log.info(
        "Dataset summary:\n"
        "  Seasons   : %d – %d\n"
        "  Total rows: %d\n"
        "  Win labels: %d (%.1f%%)\n"
        "  Podium    : %d (%.1f%%)\n"
        "  Pit NaN   : %d (imputed)",
        df["season"].min(), df["season"].max(),
        len(df),
        df[TARGET_WIN].sum(), 100 * df[TARGET_WIN].mean(),
        df[TARGET_PODIUM].sum(), 100 * df[TARGET_PODIUM].mean(),
        raw_df["pit_stop_count"].isna().sum(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Oracle — Data Preprocessing")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to pre-downloaded raw CSV (skips Ergast fetch)",
    )
    args = parser.parse_args()
    run_pipeline(csv_path=args.csv)
