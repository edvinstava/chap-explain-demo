"""SHAP and LIME explanations as one tidy dataframe.

Schema (one row per location, period, method, view, feature):

    location, time_period, method (shap|lime), view (lagged|base),
    feature, base_feature, lag, condition, feature_value,
    contribution, baseline, prediction, local_prediction

- contribution/baseline/prediction are on the raw case scale the model
  predicts (before clipping at zero).
- For SHAP, baseline + sum(contributions) == prediction exactly (additivity).
- For LIME, contribution is the local surrogate weight, condition is the
  human-readable rule it applies to (e.g. "rainfall_lag_2 > 45.10"), and
  local_prediction is the surrogate's own output (fidelity check).
- view == "base" aggregates lagged features back to their base covariate
  (exact for SHAP by additivity, approximate for LIME).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from . import features as F
from .model import RANDOM_STATE

LIME_NUM_SAMPLES = 1000

COLUMNS = [
    "location",
    "time_period",
    "method",
    "view",
    "feature",
    "base_feature",
    "lag",
    "condition",
    "feature_value",
    "contribution",
    "baseline",
    "prediction",
    "local_prediction",
]


def shap_explanations(payload: dict[str, Any], rows: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    model = payload["model"]
    names = payload["feature_names"]
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(x))
    baseline = float(np.ravel(explainer.expected_value)[0])
    predictions = baseline + values.sum(axis=1)

    records = []
    for i in range(len(x)):
        for j, feature in enumerate(names):
            base, lag = F.base_feature_of(feature, payload["covariates"])
            records.append(
                {
                    "location": rows["location"].iloc[i],
                    "time_period": rows["time_period"].iloc[i],
                    "method": "shap",
                    "view": "lagged",
                    "feature": feature,
                    "base_feature": base,
                    "lag": lag,
                    "condition": "",
                    "feature_value": float(x.iloc[i, j]),
                    "contribution": float(values[i, j]),
                    "baseline": baseline,
                    "prediction": float(predictions[i]),
                    "local_prediction": np.nan,
                }
            )
    return pd.DataFrame(records, columns=COLUMNS)


def lime_explanations(payload: dict[str, Any], rows: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    model = payload["model"]
    names = payload["feature_names"]
    explainer = LimeTabularExplainer(
        payload["background"].to_numpy(),
        feature_names=names,
        mode="regression",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    def predict_fn(matrix: np.ndarray) -> np.ndarray:
        return model.predict(pd.DataFrame(matrix, columns=names))

    records = []
    for i in range(len(x)):
        instance = x.iloc[i].to_numpy()
        explanation = explainer.explain_instance(
            instance, predict_fn, num_features=len(names), num_samples=LIME_NUM_SAMPLES
        )
        # In regression mode lime stores the true weights under dummy_label (1)
        # and a NEGATED copy under label 0, so never take next(iter(local_exp)).
        label = explanation.dummy_label
        weights = explanation.local_exp[label]
        conditions = [condition for condition, _ in explanation.as_list()]
        baseline = float(explanation.intercept[label])
        local_prediction = float(np.ravel(explanation.local_pred)[0])
        prediction = float(model.predict(x.iloc[[i]])[0])

        for (feature_id, weight), condition in zip(weights, conditions, strict=True):
            feature = names[feature_id]
            base, lag = F.base_feature_of(feature, payload["covariates"])
            records.append(
                {
                    "location": rows["location"].iloc[i],
                    "time_period": rows["time_period"].iloc[i],
                    "method": "lime",
                    "view": "lagged",
                    "feature": feature,
                    "base_feature": base,
                    "lag": lag,
                    "condition": condition,
                    "feature_value": float(x.iloc[i][feature]),
                    "contribution": float(weight),
                    "baseline": baseline,
                    "prediction": prediction,
                    "local_prediction": local_prediction,
                }
            )
    return pd.DataFrame(records, columns=COLUMNS)


def add_base_view(lagged: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lagged-view rows per base covariate and append as view='base'."""
    grouped = (
        lagged.groupby(["location", "time_period", "method", "base_feature"], sort=False)
        .agg(
            contribution=("contribution", "sum"),
            baseline=("baseline", "first"),
            prediction=("prediction", "first"),
            local_prediction=("local_prediction", "first"),
        )
        .reset_index()
    )
    grouped["view"] = "base"
    grouped["feature"] = grouped["base_feature"]
    grouped["lag"] = np.nan
    grouped["condition"] = ""
    grouped["feature_value"] = np.nan
    return pd.concat([lagged, grouped[COLUMNS]], ignore_index=True)


def explain(payload: dict[str, Any], rows: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Full tidy explanation frame: SHAP + LIME, lagged + base views."""
    tidy = pd.concat(
        [shap_explanations(payload, rows, x), lime_explanations(payload, rows, x)],
        ignore_index=True,
    )
    return add_base_view(tidy)


def global_importance(tidy: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute and mean signed contribution per method, view and feature."""
    importance = (
        tidy.groupby(["method", "view", "feature"], sort=False)
        .agg(
            base_feature=("base_feature", "first"),
            lag=("lag", "first"),
            mean_abs_contribution=("contribution", lambda s: float(s.abs().mean())),
            mean_contribution=("contribution", "mean"),
        )
        .reset_index()
    )
    return importance.sort_values(
        ["method", "view", "mean_abs_contribution"], ascending=[True, True, False]
    ).reset_index(drop=True)
