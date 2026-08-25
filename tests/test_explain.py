import numpy as np
import pandas as pd
import pytest

from chap_explain import explain, model


@pytest.fixture(scope="module")
def tidy(payload, train_df, future_df):
    rows, x = model.build_prediction_features(payload, train_df, future_df)
    return explain.explain(payload, rows.head(4), x.head(4))


def test_schema(tidy):
    assert list(tidy.columns) == explain.COLUMNS
    assert set(tidy.method.unique()) == {"shap", "lime"}
    assert set(tidy.view.unique()) == {"lagged", "base"}


def test_shap_additivity(tidy):
    """baseline + sum(contributions) == prediction, exactly, per instance."""
    lagged = tidy[(tidy.method == "shap") & (tidy.view == "lagged")]
    for _, group in lagged.groupby(["location", "time_period"]):
        total = group.baseline.iloc[0] + group.contribution.sum()
        assert np.isclose(total, group.prediction.iloc[0], atol=1e-6)


def test_base_view_sums_lagged_view(tidy):
    for method in ["shap", "lime"]:
        lagged = tidy[(tidy.method == method) & (tidy.view == "lagged")]
        base = tidy[(tidy.method == method) & (tidy.view == "base")]
        key = ["location", "time_period", "base_feature"]
        expected = lagged.groupby(key).contribution.sum()
        actual = base.set_index(key).contribution
        pd.testing.assert_series_equal(
            actual.sort_index(), expected.sort_index(), check_names=False
        )


def test_lime_sign_regression(payload, train_df, future_df):
    """Guards the lime regression-mode label trap: local_exp[0] holds NEGATED
    weights, the true ones live under dummy_label. A feature the model relies
    on positively must get a positive LIME weight when its value is high.

    The synthetic panel generates cases from rainfall two months earlier, so
    the rainfall_lag_2 base contribution must correlate positively with its
    value across explained instances for both methods.
    """
    rows, x = model.build_prediction_features(payload, train_df, future_df)
    tidy = explain.explain(payload, rows, x)
    lagged = tidy[tidy.view == "lagged"]
    for method in ["shap", "lime"]:
        subset = lagged[(lagged.method == method) & (lagged.feature == "rainfall_lag_2")]
        correlation = np.corrcoef(subset.feature_value, subset.contribution)[0, 1]
        assert correlation > 0.3, (
            f"{method} rainfall_lag_2 sign looks inverted: r={correlation:.2f}"
        )


def test_lime_conditions_mention_their_feature(tidy):
    lime_rows = tidy[(tidy.method == "lime") & (tidy.view == "lagged")]
    assert (lime_rows.condition.str.len() > 0).all()
    matches = [row.feature in row.condition for row in lime_rows.itertuples()]
    assert all(matches)


def test_global_importance(tidy):
    importance = explain.global_importance(tidy)
    assert (importance.mean_abs_contribution >= 0).all()
    top = importance[(importance.method == "shap") & (importance.view == "lagged")].iloc[0]
    assert (
        top.mean_abs_contribution
        == importance[
            (importance.method == "shap") & (importance.view == "lagged")
        ].mean_abs_contribution.max()
    )
