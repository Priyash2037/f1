"""
F1 Oracle — XGBoost Training Pipeline
======================================
Trains an XGBoost binary classifier to predict F1 race win probability.

Key design choices
------------------
• Season-based train/test split — model is trained on older seasons and
  tested on the most recent complete season (avoids temporal leakage).
• GridSearchCV over {learning_rate, max_depth, n_estimators, subsample}.
• Calibrated probabilities via CalibratedClassifierCV (Platt scaling).
• Saves three artefacts: model.joblib, scaler.joblib, meta.json.

Usage
-----
    python train.py                            # default paths
    python train.py --test-season 2024        # explicit test season
    python train.py --target podium           # train for podium % instead
    python train.py --no-grid-search          # skip GridSearch (fast dev run)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# Preprocessing constants shared with preprocess.py
from preprocess import NUMERIC_FEATURES, TARGET_PODIUM, TARGET_WIN, TARGET_POINTS

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost hyperparameter search space
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRID = {
    "learning_rate": [0.02, 0.05, 0.10, 0.20],
    "max_depth":     [3, 4, 5, 6],
    "n_estimators":  [200, 400, 600],
    "subsample":     [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.9],
}

# Reduced grid for fast dev runs (--no-grid-search uses best defaults)
BEST_DEFAULTS = {
    "learning_rate":      0.05,
    "max_depth":          4,
    "n_estimators":       400,
    "subsample":          0.85,
    "colsample_bytree":   0.8,
    "min_child_weight":   3,
    "gamma":              0.1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Load preprocessed data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    features_csv: Path = DATA_DIR / "features.csv",
    targets_csv: Path  = DATA_DIR / "targets.csv",
) -> pd.DataFrame:
    """Merge features + targets on (season, round, driver_id)."""
    X_df = pd.read_csv(features_csv)
    y_df = pd.read_csv(targets_csv)
    df = X_df.merge(y_df, on=["season", "round", "driver_id"], how="inner")
    log.info("Loaded %d samples across seasons %d–%d", len(df), df["season"].min(), df["season"].max())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Season-based train / test split  (no data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    test_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train on all seasons < test_season.
    Test  on test_season.
    This mirrors real-world usage: the model is trained on history and
    evaluated on a season it has never seen before.
    """
    train_df = df[df["season"] < test_season].copy()
    test_df  = df[df["season"] == test_season].copy()
    log.info(
        "Train: seasons %d–%d  (%d samples) | Test: %d  (%d samples)",
        train_df["season"].min(), train_df["season"].max(), len(train_df),
        test_season, len(test_df),
    )
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Class-imbalance weight
# ─────────────────────────────────────────────────────────────────────────────

def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    XGBoost's scale_pos_weight balances positive/negative class frequencies.
    Formula: count(negatives) / count(positives)
    """
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    w = n_neg / max(n_pos, 1)
    log.info("Class balance — pos: %d  neg: %d  → scale_pos_weight=%.2f", n_pos, n_neg, w)
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

def build_xgb_model(scale_pos_weight: float, params: dict | None = None) -> XGBClassifier:
    """
    Build an XGBClassifier configured for probability output (predict_proba).

    Parameters
    ----------
    scale_pos_weight : float
        Class-imbalance correction factor.
    params : dict, optional
        Override default hyperparameters (used after GridSearch).
    """
    base_params = {
        **BEST_DEFAULTS,
        "objective":         "binary:logistic",   # sigmoid output → probabilities
        "eval_metric":       "logloss",
        "use_label_encoder": False,
        "scale_pos_weight":  scale_pos_weight,
        "random_state":      42,
        "n_jobs":            -1,
        "tree_method":       "hist",              # fast histogram-based
    }
    if params:
        base_params.update(params)
    return XGBClassifier(**base_params)


# ─────────────────────────────────────────────────────────────────────────────
# GridSearch
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    n_folds: int = 5,
) -> tuple[dict, float]:
    """
    Stratified K-Fold GridSearchCV over PARAM_GRID.
    Returns the best params dict and the best cross-val ROC-AUC score.
    """
    log.info("Starting GridSearchCV  (%d-fold, %d candidates)…",
             n_folds, np.prod([len(v) for v in PARAM_GRID.values()]))
    t0 = time.time()

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    gs = GridSearchCV(
        base,
        PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)

    elapsed = time.time() - t0
    log.info("GridSearch done in %.1fs", elapsed)
    log.info("Best ROC-AUC  : %.4f", gs.best_score_)
    log.info("Best params   : %s", gs.best_params_)
    return gs.best_params_, gs.best_score_


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_model(
    model: XGBClassifier,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> CalibratedClassifierCV:
    """
    Wrap the fitted XGBClassifier in Platt scaling (sigmoid) so that
    predict_proba outputs are better calibrated probabilities.
    This matters for the Monte Carlo engine in production.
    """
    log.info("Calibrating model (Platt scaling)…")
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
) -> dict:
    """Compute and log key classification metrics on the held-out test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, digits=4)

    log.info(
        "\n═══ Test Evaluation — %s ═══\n"
        "  ROC-AUC        : %.4f\n"
        "  Avg Precision  : %.4f\n"
        "─────────────────────────────\n%s",
        target_name, roc_auc, avg_prec, report,
    )
    return {"roc_auc": roc_auc, "avg_precision": avg_prec}


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def log_feature_importance(model: XGBClassifier | CalibratedClassifierCV) -> None:
    """Extract and log feature importances from the underlying XGB estimator."""
    try:
        # Unwrap CalibratedClassifierCV if needed
        xgb: XGBClassifier = (
            model.estimator if hasattr(model, "estimator") else model
        )
        importances = xgb.feature_importances_
        ranked = sorted(zip(NUMERIC_FEATURES, importances), key=lambda x: -x[1])
        lines = [f"  {name:<28} {imp:.4f}" for name, imp in ranked]
        log.info("Feature importances:\n%s", "\n".join(lines))
    except Exception:
        pass   # Not critical


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def save_artefacts(
    model,
    scaler,
    meta: dict,
    prefix: str,
) -> None:
    """
    Save model + scaler + metadata to disk.

    Output files
    ------------
    models/<prefix>_model.joblib  — calibrated XGBClassifier (predict_proba ready)
    models/<prefix>_scaler.joblib — fitted StandardScaler (same as preprocess.py)
    models/<prefix>_meta.json     — training metadata (features, season split, metrics)
    """
    model_path = MODEL_DIR / f"{prefix}_model.joblib"
    scaler_path = MODEL_DIR / f"{prefix}_scaler.joblib"
    meta_path   = MODEL_DIR / f"{prefix}_meta.json"

    joblib.dump(model, model_path, compress=3)
    joblib.dump(scaler, scaler_path, compress=3)
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info("Model   saved → %s", model_path)
    log.info("Scaler  saved → %s", scaler_path)
    log.info("Meta    saved → %s", meta_path)


# ─────────────────────────────────────────────────────────────────────────────
# Full training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(
    test_season: int = 2024,
    target: str = "win",
    run_grid_search_flag: bool = True,
) -> None:
    """
    End-to-end training pipeline.

    Parameters
    ----------
    test_season : int
        The season held out as the test set. All prior seasons = train.
    target : str
        One of "win", "podium", "points" — which label to predict.
    run_grid_search_flag : bool
        If False, uses BEST_DEFAULTS and skips GridSearchCV.
    """
    target_col = {
        "win":    TARGET_WIN,
        "podium": TARGET_PODIUM,
        "points": TARGET_POINTS,
    }[target]
    prefix = f"f1_{target}_s{test_season}"

    # ── Load data ────────────────────────────────────────────────────────────
    df = load_data()

    if test_season not in df["season"].values:
        raise ValueError(f"Test season {test_season} not found in dataset. "
                         f"Available: {sorted(df['season'].unique())}")

    # ── Season split ─────────────────────────────────────────────────────────
    train_df, test_df = temporal_split(df, test_season)

    # Reserve 20% of training data for probability calibration (Platt scaling)
    # Use the most recent 20% by season so calibration is temporally close to test
    cal_cutoff = train_df["season"].quantile(0.80)
    train_only = train_df[train_df["season"] <= cal_cutoff]
    cal_df     = train_df[train_df["season"] > cal_cutoff]

    X_train = train_only[NUMERIC_FEATURES].to_numpy()
    y_train = train_only[target_col].to_numpy()
    X_cal   = cal_df[NUMERIC_FEATURES].to_numpy()
    y_cal   = cal_df[target_col].to_numpy()
    X_test  = test_df[NUMERIC_FEATURES].to_numpy()
    y_test  = test_df[target_col].to_numpy()

    log.info(
        "Split sizes — Train: %d  |  Cal: %d  |  Test: %d",
        len(X_train), len(X_cal), len(X_test),
    )

    # ── Class weight ─────────────────────────────────────────────────────────
    spw = compute_scale_pos_weight(pd.Series(y_train))

    # ── GridSearch or defaults ────────────────────────────────────────────────
    if run_grid_search_flag:
        best_params, best_cv_score = run_grid_search(X_train, y_train, spw)
    else:
        best_params = BEST_DEFAULTS
        best_cv_score = None
        log.info("Skipping GridSearch — using BEST_DEFAULTS")

    # ── Train final model ─────────────────────────────────────────────────────
    log.info("Training final XGBClassifier…")
    model = build_xgb_model(spw, params=best_params)

    # Use early stopping on calibration set for the final fit
    model.fit(
        X_train, y_train,
        eval_set=[(X_cal, y_cal)],
        verbose=False,
    )

    # ── Calibrate probabilities ───────────────────────────────────────────────
    calibrated = calibrate_model(model, X_cal, y_cal)

    # ── Evaluate on held-out test season ─────────────────────────────────────
    metrics = evaluate(calibrated, X_test, y_test, target)
    log_feature_importance(model)

    # ── Load scaler saved by preprocess.py ───────────────────────────────────
    scaler_path = DATA_DIR / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Run preprocess.py first."
        )
    scaler = joblib.load(scaler_path)

    # ── Save artefacts ────────────────────────────────────────────────────────
    meta = {
        "target": target,
        "target_column": target_col,
        "test_season": test_season,
        "train_seasons": sorted(train_only["season"].unique().tolist()),
        "features": NUMERIC_FEATURES,
        "best_params": best_params,
        "best_cv_roc_auc": best_cv_score,
        "test_roc_auc": metrics["roc_auc"],
        "test_avg_precision": metrics["avg_precision"],
        "n_train": len(X_train),
        "n_test":  len(X_test),
    }
    save_artefacts(calibrated, scaler, meta, prefix)

    log.info("Training complete! Model artefacts saved to: %s/", MODEL_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Oracle — XGBoost Training")
    parser.add_argument(
        "--test-season",
        type=int,
        default=2024,
        help="Season to hold out for testing (default: 2024)",
    )
    parser.add_argument(
        "--target",
        choices=["win", "podium", "points"],
        default="win",
        help="Prediction target: win / podium / points (default: win)",
    )
    parser.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Skip GridSearchCV and use pre-tuned defaults (faster)",
    )
    args = parser.parse_args()
    train(
        test_season=args.test_season,
        target=args.target,
        run_grid_search_flag=not args.no_grid_search,
    )
