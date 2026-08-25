"""Feature engineering shared by train and predict.

The model is purely exogenous: features are current and lagged covariate
values, calendar seasonality, and population. No case-history features, so
explanations are entirely about covariates and counterfactual predictions
respond only to covariate changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "disease_cases"
LAGS = (0, 1, 2, 3)

# Columns that are never used as covariates.
_META_COLUMNS = {"time_period", "location", "parent", TARGET}

# Covariates the MLproject file declares as required. Everything else numeric
# is picked up dynamically (e.g. spray_coverage, bednet coverage).
REQUIRED_COVARIATES = ("rainfall", "mean_temperature")

SEASONALITY_FEATURES = ("month_sin", "month_cos")


def clean_chap_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a CSV as written by CHAP: drop index columns, parse periods."""
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.copy()
    df["time_period"] = df["time_period"].astype(str)
    df["_period"] = pd.to_datetime(df["time_period"], format="%Y-%m")
    return df.sort_values(["location", "_period"]).reset_index(drop=True)


def discover_covariates(df: pd.DataFrame) -> list[str]:
    """All numeric non-meta columns except population, in stable order.

    Required covariates come first, then any additional ones alphabetically.
    """
    numeric = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    extra = sorted(numeric - _META_COLUMNS - set(REQUIRED_COVARIATES) - {"population", "_period"})
    missing = [c for c in REQUIRED_COVARIATES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required covariates: {missing}")
    return list(REQUIRED_COVARIATES) + extra


def feature_names(covariates: list[str], has_population: bool) -> list[str]:
    names = [f"{cov}_lag_{lag}" for cov in covariates for lag in LAGS]
    names += list(SEASONALITY_FEATURES)
    if has_population:
        names.append("population")
    return names


def build_features(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Add feature columns to a cleaned dataframe (one row per location-period).

    Lag k means "value k months before the row's month"; lag 0 is the row's
    own month, which is valid because CHAP supplies future covariates.
    """
    df = df.copy()
    for cov in covariates:
        grouped = df.groupby("location")[cov]
        for lag in LAGS:
            df[f"{cov}_lag_{lag}"] = grouped.shift(lag)
    month = df["_period"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    if "population" in df.columns:
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
    return df


def base_feature_of(feature: str, covariates: list[str]) -> tuple[str, float]:
    """Map a model feature to (base covariate, lag) for the aggregated view.

    Seasonality features collapse into one "seasonality" base feature;
    population maps to itself. Lag is NaN where it does not apply.
    """
    for cov in covariates:
        prefix = f"{cov}_lag_"
        if feature.startswith(prefix):
            return cov, float(feature.removeprefix(prefix))
    if feature in SEASONALITY_FEATURES:
        return "seasonality", float("nan")
    return feature, float("nan")
