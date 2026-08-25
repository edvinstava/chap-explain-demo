"""Training and prediction around a fixed gradient boosted tree."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import features as F

RANDOM_STATE = 42
N_SAMPLES = 100
HOLDOUT_PERIODS = 6
MAX_BACKGROUND_ROWS = 500

# Fixed, conservative hyperparameters. No tuning: this model is optimized for
# clear explanations, not leaderboard accuracy.
_HYPERPARAMETERS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)


def _fit(x: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(**_HYPERPARAMETERS)
    model.fit(x, y)
    return model


def train(df: pd.DataFrame) -> dict[str, Any]:
    """Train on a cleaned CHAP training dataframe; return the model payload."""
    covariates = F.discover_covariates(df)
    df = F.build_features(df, covariates)
    names = F.feature_names(covariates, has_population="population" in df.columns)

    df[F.TARGET] = pd.to_numeric(df[F.TARGET], errors="coerce")
    usable = df.dropna(subset=[*names, F.TARGET])
    if len(usable) < 30:
        raise ValueError(f"Too few usable training rows after lagging: {len(usable)}")

    n_periods = usable.groupby("location").size().min()
    holdout = min(HOLDOUT_PERIODS, max(2, int(n_periods) // 5))
    is_holdout = usable.groupby("location").cumcount(ascending=False) < holdout

    x_fit = usable.loc[~is_holdout, names]
    y_fit = usable.loc[~is_holdout, F.TARGET]
    x_val = usable.loc[is_holdout, names]
    y_val = usable.loc[is_holdout, F.TARGET]

    holdout_model = _fit(x_fit, y_fit)
    y_hat = np.clip(holdout_model.predict(x_val), 0, None)
    residuals = (y_val.to_numpy() - y_hat).astype(float)

    per_location_mae = {
        location: float(
            mean_absolute_error(
                group[F.TARGET], np.clip(holdout_model.predict(group[names]), 0, None)
            )
        )
        for location, group in usable.loc[is_holdout].groupby("location")
    }
    metrics = {
        "holdout_periods_per_location": int(holdout),
        "holdout_rows": int(len(y_val)),
        "mae": float(mean_absolute_error(y_val, y_hat)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, y_hat))),
        "per_location_mae": per_location_mae,
    }

    final_model = _fit(usable[names], usable[F.TARGET])

    background = usable[names]
    if len(background) > MAX_BACKGROUND_ROWS:
        background = background.sample(MAX_BACKGROUND_ROWS, random_state=RANDOM_STATE)

    return {
        "model": final_model,
        "covariates": covariates,
        "feature_names": names,
        "feature_means": usable[names].mean(),
        "residuals": residuals,
        "background": background.reset_index(drop=True),
        "metrics": metrics,
        "schema_version": 1,
    }


# The model file is a joblib pickle: it is produced by this repo's own train
# entry point and consumed by its own predict in the same CHAP run directory,
# never loaded from untrusted sources. sklearn estimators require pickle.
def save(payload: dict[str, Any], path: str) -> None:
    joblib.dump(payload, path)


def load(path: str) -> dict[str, Any]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unrecognized model file: {path}")
    return payload


def build_prediction_features(
    payload: dict[str, Any], historic: pd.DataFrame, future: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (future rows in original order, feature matrix for those rows).

    Lags for the first future months reach back into the historic covariates,
    so both frames are combined per location before feature building.
    """
    covariates = payload["covariates"]
    missing = [c for c in covariates if c not in future.columns]
    if missing:
        raise ValueError(f"Future data is missing covariates the model was trained on: {missing}")

    columns = ["location", "time_period", "_period", *covariates]
    if "population" in future.columns:
        columns.append("population")
    combined = pd.concat(
        [historic[[c for c in columns if c in historic.columns]], future[columns]],
        ignore_index=True,
    )
    combined = combined.sort_values(["location", "_period"]).reset_index(drop=True)
    combined = F.build_features(combined, covariates)

    future_keys = set(zip(future["location"], future["time_period"], strict=True))
    pairs = zip(combined["location"], combined["time_period"], strict=True)
    mask = [(loc, tp) in future_keys for loc, tp in pairs]
    rows = combined.loc[mask].sort_values(["location", "_period"]).reset_index(drop=True)

    x = rows[payload["feature_names"]].fillna(payload["feature_means"])
    return rows, x


def predict_samples(payload: dict[str, Any], x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (point predictions, samples[n_rows, N_SAMPLES]) on the case scale.

    Samples are the clipped point prediction plus bootstrapped holdout
    residuals: distribution-free uncertainty that reflects observed error.
    """
    point = payload["model"].predict(x)
    clipped = np.clip(point, 0, None)
    rng = np.random.default_rng(RANDOM_STATE)
    residuals = payload["residuals"]
    if len(residuals) == 0:
        residuals = np.zeros(1)
    noise = rng.choice(residuals, size=(len(x), N_SAMPLES), replace=True)
    samples = np.clip(clipped[:, None] + noise, 0, None)
    if not np.isfinite(samples).all():
        raise ValueError("Non-finite values in prediction samples")
    return point, samples
