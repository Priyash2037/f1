"""
F1 Oracle — ML Inference Wrapper
==================================
Loads the trained XGBoost model + scaler at startup and exposes
a `predict_win_probabilities()` function used by the FastAPI backend
to blend ML probabilities with Monte Carlo simulation results.

Usage
-----
    from ml.predict import MLPredictor

    predictor = MLPredictor.load("models/f1_win_s2024_model.joblib",
                                  "models/f1_win_s2024_scaler.joblib")
    probs = predictor.predict_win_probabilities(driver_features_list)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from preprocess import NUMERIC_FEATURES

log = logging.getLogger(__name__)

# Default model paths (created by train.py)
DEFAULT_MODEL_PATH  = Path("models/f1_win_s2024_model.joblib")
DEFAULT_SCALER_PATH = Path("models/f1_win_s2024_scaler.joblib")


@dataclass
class DriverFeatures:
    """
    Feature vector for a single driver at inference time.
    All fields map 1-to-1 with NUMERIC_FEATURES from preprocess.py.
    Fields the backend cannot determine are left as None and imputed.
    """
    driver_number: int
    starting_grid_position: float  # from live position data
    constructor_points: float      # season points so far
    driver_points_before: float    # driver season points
    track_temperature: float = 26.0
    pit_stop_count: float = 2.0    # estimated remaining
    fastest_lap_rank: float = 99.0
    circuit_encoded: int = 0
    constructor_encoded: int = 0
    round: int = 1
    is_pole: int = field(init=False)
    is_front_row: int = field(init=False)
    grid_squared: float = field(init=False)
    points_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        self.is_pole      = int(self.starting_grid_position == 1)
        self.is_front_row = int(self.starting_grid_position <= 2)
        self.grid_squared = self.starting_grid_position ** 2
        self.points_ratio = self.driver_points_before / (self.constructor_points + 1.0)

    def to_vector(self) -> np.ndarray:
        """Return ordered feature vector matching NUMERIC_FEATURES."""
        return np.array([getattr(self, col) for col in NUMERIC_FEATURES], dtype=np.float32)


class MLPredictor:
    """
    Wraps the trained XGBClassifier + StandardScaler for real-time inference.

    The class is designed to be instantiated once at backend startup
    and reused across WebSocket broadcast cycles.
    """

    def __init__(self, model, scaler) -> None:
        self._model  = model
        self._scaler = scaler
        log.info("MLPredictor initialised")

    @classmethod
    def load(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        scaler_path: Path = DEFAULT_SCALER_PATH,
    ) -> "MLPredictor":
        """
        Load model and scaler from disk.
        Returns None if files don't exist (ML features disabled gracefully).
        """
        if not Path(model_path).exists() or not Path(scaler_path).exists():
            log.warning(
                "ML model files not found at %s / %s. "
                "Run ml/train.py first. ML scoring disabled.",
                model_path, scaler_path,
            )
            return None  # type: ignore

        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        log.info("Loaded ML model from %s", model_path)
        return cls(model, scaler)

    def predict_win_probabilities(
        self,
        driver_features: list[DriverFeatures],
    ) -> dict[int, float]:
        """
        Run inference for a list of drivers in the current race context.

        Returns
        -------
        dict[int, float]
            Mapping of driver_number → normalised win probability [0, 1].
            Probabilities are row-normalised across all drivers so they sum to 1.
        """
        if not driver_features:
            return {}

        # Build feature matrix (N × F)
        X_raw = np.vstack([df.to_vector() for df in driver_features])

        # Apply the same scaling used during training
        X_scaled = self._scaler.transform(X_raw)

        # predict_proba → column 1 is P(win)
        raw_probs: np.ndarray = self._model.predict_proba(X_scaled)[:, 1]

        # Normalise across the field so probabilities sum to 1.0
        total = raw_probs.sum()
        if total > 0:
            norm_probs = raw_probs / total
        else:
            n = len(driver_features)
            norm_probs = np.full(n, 1.0 / n)

        return {
            df.driver_number: float(p)
            for df, p in zip(driver_features, norm_probs)
        }

    def predict_podium_probabilities(
        self,
        driver_features: list[DriverFeatures],
        podium_model_path: Optional[Path] = None,
    ) -> dict[int, float]:
        """
        Convenience method: load the podium model if available,
        otherwise approximate podium probs from win probs × 3.
        """
        if podium_model_path and Path(podium_model_path).exists():
            pod_predictor = MLPredictor.load(podium_model_path, DEFAULT_SCALER_PATH)
            if pod_predictor:
                return pod_predictor.predict_win_probabilities(driver_features)

        # Approximation: P(podium) ≈ 3 × P(win) (normalised)
        win_probs = self.predict_win_probabilities(driver_features)
        total = sum(win_probs.values()) * 3
        if total == 0:
            return win_probs
        return {k: min(v * 3, 1.0) for k, v in win_probs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    predictor = MLPredictor.load()
    if predictor is None:
        print("No model found — run `python train.py --no-grid-search` first.")
    else:
        # Simulate a 5-driver mini field
        test_drivers = [
            DriverFeatures(driver_number=44, starting_grid_position=1,  constructor_points=500, driver_points_before=320),
            DriverFeatures(driver_number=1,  starting_grid_position=2,  constructor_points=480, driver_points_before=290),
            DriverFeatures(driver_number=16, starting_grid_position=3,  constructor_points=350, driver_points_before=180),
            DriverFeatures(driver_number=81, starting_grid_position=4,  constructor_points=500, driver_points_before=280),
            DriverFeatures(driver_number=63, starting_grid_position=5,  constructor_points=200, driver_points_before=150),
        ]
        probs = predictor.predict_win_probabilities(test_drivers)
        print("\n── ML Win Probabilities ──")
        for num, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 40)
            print(f"  Driver #{num:>2}  {prob:6.1%}  {bar}")
