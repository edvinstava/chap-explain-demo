"""Tests for plot filtering and the lean default chart set."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chap_explain import explain, model, plotting

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tidy(payload, train_df, future_df):
    rows, x = model.build_prediction_features(payload, train_df, future_df)
    mask = rows.time_period <= "2022-10"
    tidy = explain.explain(payload, rows[mask], x[mask])
    tidy["condition"] = tidy["condition"].fillna("")
    return tidy


@pytest.fixture(scope="module")
def importance(tidy):
    return explain.global_importance(tidy)


def test_filter_by_location(tidy):
    filtered = plotting.filter_explanations(tidy, locations=["North"])
    assert set(filtered.location.unique()) == {"North"}


def test_filter_by_period(tidy):
    filtered = plotting.filter_explanations(tidy, periods=["2022-07", "2022-08"])
    assert set(filtered.time_period.unique()) == {"2022-07", "2022-08"}


def test_filter_no_arguments_keeps_everything(tidy):
    filtered = plotting.filter_explanations(tidy)
    assert len(filtered) == len(tidy)


def test_filter_unknown_location_raises(tidy):
    with pytest.raises(ValueError, match="North"):
        plotting.filter_explanations(tidy, locations=["Nowhere"])


def test_filter_unknown_period_raises(tidy):
    with pytest.raises(ValueError, match="time_period"):
        plotting.filter_explanations(tidy, periods=["1999-01"])


def test_render_all_default_is_lean(tidy, importance, tmp_path):
    plotting.render_all(tidy, importance, tmp_path)
    pngs = sorted(p.name for p in (tmp_path / "png").glob("*.png"))
    assert "global_importance.png" in pngs
    assert sum(name.startswith("local_") for name in pngs) == 2  # one per location
    assert sum(name.startswith("over_time_") for name in pngs) == 2
    assert not any(name.startswith("intervention_") for name in pngs)
    assert (tmp_path / "highcharts" / "index.html").exists()


def test_render_all_intervention_only(tidy, importance, tmp_path):
    plotting.render_all(tidy, importance, tmp_path, charts=["intervention"])
    pngs = sorted(p.name for p in (tmp_path / "png").glob("*.png"))
    assert pngs == [
        "intervention_North_spray_coverage.png",
        "intervention_South_spray_coverage.png",
    ]


def test_render_all_unknown_chart_raises(tidy, importance, tmp_path):
    with pytest.raises(ValueError, match="intervention"):
        plotting.render_all(tidy, importance, tmp_path, charts=["waterfall"])


def test_format_value():
    assert plotting.format_value(26.6) == "26.6"
    assert plotting.format_value(0.8) == "0.8"
    assert plotting.format_value(323122.5) == "323,123"
    assert plotting.format_value(float("nan")) == ""


def test_local_lagged_tooltips_show_feature_values(tidy):
    period = tidy.time_period.min()
    configs = plotting.local_highcharts(tidy, "North", period)
    row = tidy[
        (tidy.method == "shap")
        & (tidy.view == "lagged")
        & (tidy.location == "North")
        & (tidy.time_period == period)
        & (tidy.feature == "rainfall_lag_2")
    ].iloc[0]
    dumped = json.dumps(configs[f"local_North_{period}_shap_lagged"])
    assert f"value: {plotting.format_value(row.feature_value)}" in dumped
    dumped_lime = json.dumps(configs[f"local_North_{period}_lime_lagged"])
    assert "value: " in dumped_lime


def test_local_base_tooltips_show_per_lag_breakdown(tidy):
    period = tidy.time_period.min()
    configs = plotting.local_highcharts(tidy, "North", period)
    dumped = json.dumps(configs[f"local_North_{period}_shap_base"])
    assert "this month" in dumped  # rainfall varies by lag: full breakdown
    assert "value: 100,000" in dumped  # population equal across lags: collapsed
    assert "earlier 100,000" not in dumped


def test_over_time_tooltips_show_values(tidy):
    config = plotting.over_time_highcharts(tidy, "North")["over_time_North"]
    dumped = json.dumps(config)
    assert "this month" in dumped
    assert "Prediction" in dumped


def test_intervention_tooltips_show_coverage(tidy):
    charts = plotting.intervention_highcharts(tidy, "North", "spray_coverage")
    dumped = json.dumps(charts["intervention_North_spray_coverage_effect"])
    assert "coverage: " in dumped


def test_plots_cli_filters(tidy, tmp_path):
    csv = tmp_path / "predictions.explanations.csv"
    tidy.to_csv(csv, index=False)
    out = tmp_path / "plots"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "plots.py"),
            str(csv),
            "-o",
            str(out),
            "--locations",
            "North",
            "--charts",
            "over_time",
        ],
        check=True,
        cwd=REPO,
    )
    pngs = sorted(p.name for p in (out / "png").glob("*.png"))
    assert pngs == ["over_time_North.png"]
