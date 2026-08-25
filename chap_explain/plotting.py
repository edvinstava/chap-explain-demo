"""Render explanation plots from the tidy sidecar files predict writes.

Every chart is drawn twice from the same data: a matplotlib PNG (for printing)
and a Highcharts config JSON (for the web), plus one self-contained HTML page
that renders all Highcharts configs. Requires the "plots" dependency group.

Color roles (validated palette, see repository README):
- Signed contributions follow the SHAP convention: red pushes the prediction
  up, blue pushes it down.
- Base covariates get stable categorical hues so the same covariate has the
  same color in every chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

POSITIVE = "#e34948"  # pushes prediction up
NEGATIVE = "#2a78d6"  # pushes prediction down
NEUTRAL = "#8f8e8a"
INK = "#0b0b0b"
GRID = "#e5e4e0"

_ROLE_COLORS = {
    "rainfall": "#2a78d6",
    "mean_temperature": "#eb6834",
    "seasonality": "#eda100",
    "population": "#e87ba4",
}
_EXTRA_COLORS = ["#1baf7a", "#008300", "#4a3aa7", "#e34948"]

KNOWN_BASE_FEATURES = set(_ROLE_COLORS) | {"rainfall", "mean_temperature"}


def covariate_colors(base_features: list[str]) -> dict[str, str]:
    colors = {}
    extras = [f for f in sorted(base_features) if f not in _ROLE_COLORS]
    for feature in base_features:
        if feature in _ROLE_COLORS:
            colors[feature] = _ROLE_COLORS[feature]
        else:
            colors[feature] = _EXTRA_COLORS[extras.index(feature) % len(_EXTRA_COLORS)]
    return colors


def display_name(feature: str, lag: float) -> str:
    label = feature.replace("_", " ").capitalize()
    if pd.isna(lag):
        return label
    lag = int(lag)
    base = feature.rsplit("_lag_", 1)[0].replace("_", " ").capitalize()
    if lag == 0:
        return f"{base} (this month)"
    if lag == 1:
        return f"{base} (1 month earlier)"
    return f"{base} ({lag} months earlier)"


def find_explanations_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.glob("*.explanations*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No *.explanations*.csv found in {path}")
    return candidates[-1]


def load_explanations(path: Path) -> pd.DataFrame:
    tidy = pd.read_csv(find_explanations_file(path))
    tidy["condition"] = tidy["condition"].fillna("")
    return tidy


def peak_periods(tidy: pd.DataFrame) -> dict[str, str]:
    """The predicted-peak month per location: the 'why is the peak' target."""
    shap_rows = tidy[(tidy.method == "shap") & (tidy.view == "base")]
    per_period = shap_rows.groupby(["location", "time_period"])["prediction"].first().reset_index()
    idx = per_period.groupby("location")["prediction"].idxmax()
    return dict(
        zip(per_period.loc[idx, "location"], per_period.loc[idx, "time_period"], strict=True)
    )


def detect_intervention_covariates(tidy: pd.DataFrame) -> list[str]:
    base = set(tidy[tidy.view == "base"].base_feature.unique())
    return sorted(base - KNOWN_BASE_FEATURES)


def _style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- local chart


def _local_panel(ax, rows: pd.DataFrame, title: str) -> None:
    rows = rows.sort_values("contribution")
    labels = [
        row.condition if row.condition else display_name(row.feature, row.lag)
        for row in rows.itertuples()
    ]
    colors = [POSITIVE if c > 0 else NEGATIVE for c in rows.contribution]
    bars = ax.barh(labels, rows.contribution, color=colors, height=0.62)
    ax.bar_label(bars, fmt="%+.0f", fontsize=7, color=INK, padding=2)
    ax.axvline(0, color=NEUTRAL, linewidth=1)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.margins(x=0.15)
    _style_axes(ax)


def plot_local(tidy: pd.DataFrame, location: str, period: str, out: Path) -> None:
    rows = tidy[(tidy.location == location) & (tidy.time_period == period)]
    prediction = rows[rows.method == "shap"].prediction.iloc[0]
    baseline = rows[rows.method == "shap"].baseline.iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row_i, method in enumerate(["shap", "lime"]):
        for col_i, view in enumerate(["lagged", "base"]):
            panel = rows[(rows.method == method) & (rows.view == view)]
            view_label = "lagged features" if view == "lagged" else "original covariates"
            _local_panel(axes[row_i][col_i], panel, f"{method.upper()} - {view_label}")
    fig.suptitle(
        f"Why does the model predict {prediction:.0f} cases in {location}, {period}?\n"
        f"Red pushes the prediction up, blue pushes it down. "
        f"Model baseline (average prediction): {baseline:.0f} cases.",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def local_highcharts(tidy: pd.DataFrame, location: str, period: str) -> dict[str, dict]:
    rows = tidy[(tidy.location == location) & (tidy.time_period == period)]
    charts = {}
    for method in ["shap", "lime"]:
        for view in ["lagged", "base"]:
            panel = rows[(rows.method == method) & (rows.view == view)]
            prediction = panel.prediction.iloc[0]
            baseline = panel.baseline.iloc[0]
            view_label = "lagged features" if view == "lagged" else "original covariates"
            if method == "shap":
                panel = panel.sort_values("contribution", key=abs, ascending=False)
                data = [{"name": "Baseline", "y": round(float(baseline), 1), "color": NEUTRAL}]
                data += [
                    {
                        "name": display_name(row.feature, row.lag),
                        "y": round(float(row.contribution), 1),
                    }
                    for row in panel.itertuples()
                ]
                data.append({"name": "Prediction", "isSum": True, "color": NEUTRAL})
                config = {
                    "chart": {"type": "waterfall"},
                    "title": {"text": f"SHAP ({view_label}): {location} {period}"},
                    "subtitle": {"text": "Baseline + contributions = prediction"},
                    "xAxis": {"type": "category"},
                    "yAxis": {"title": {"text": "Predicted cases"}},
                    "legend": {"enabled": False},
                    "tooltip": {"pointFormat": "<b>{point.y:,.1f}</b> cases"},
                    "series": [
                        {
                            "upColor": POSITIVE,
                            "color": NEGATIVE,
                            "data": data,
                            "dataLabels": {"enabled": True, "format": "{point.y:,.0f}"},
                        }
                    ],
                }
            else:
                panel = panel.sort_values("contribution")
                config = {
                    "chart": {"type": "bar"},
                    "title": {"text": f"LIME ({view_label}): {location} {period}"},
                    "subtitle": {
                        "text": (
                            f"Local surrogate weights (model prediction {prediction:,.0f} cases)"
                        )
                    },
                    "xAxis": {
                        "categories": [
                            row.condition if row.condition else display_name(row.feature, row.lag)
                            for row in panel.itertuples()
                        ]
                    },
                    "yAxis": {"title": {"text": "Weight (cases)"}},
                    "legend": {"enabled": False},
                    "tooltip": {"pointFormat": "<b>{point.y:,.1f}</b>"},
                    "series": [
                        {
                            "data": [
                                {
                                    "y": round(float(c), 1),
                                    "color": POSITIVE if c > 0 else NEGATIVE,
                                }
                                for c in panel.contribution
                            ],
                            "dataLabels": {"enabled": True, "format": "{point.y:,.0f}"},
                        }
                    ],
                }
            charts[f"local_{location}_{period}_{method}_{view}"] = config
    return charts


# ------------------------------------------------------- global importance


def plot_global_importance(importance: pd.DataFrame, out: Path, top_n: int = 12) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row_i, method in enumerate(["shap", "lime"]):
        for col_i, view in enumerate(["lagged", "base"]):
            panel = importance[(importance.method == method) & (importance.view == view)]
            panel = panel.nlargest(top_n, "mean_abs_contribution").sort_values(
                "mean_abs_contribution"
            )
            ax = axes[row_i][col_i]
            labels = [display_name(row.feature, row.lag) for row in panel.itertuples()]
            bars = ax.barh(labels, panel.mean_abs_contribution, color="#2a78d6", height=0.62)
            ax.bar_label(bars, fmt="%.0f", fontsize=7, color=INK, padding=2)
            view_label = "lagged features" if view == "lagged" else "original covariates"
            ax.set_title(f"{method.upper()} - {view_label}", fontsize=9, color=INK, loc="left")
            ax.margins(x=0.12)
            _style_axes(ax)
    fig.suptitle(
        "Which covariates matter most? Mean absolute contribution (cases) across all predictions",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def global_importance_highcharts(importance: pd.DataFrame, top_n: int = 12) -> dict[str, dict]:
    charts = {}
    for method in ["shap", "lime"]:
        for view in ["lagged", "base"]:
            panel = importance[(importance.method == method) & (importance.view == view)]
            panel = panel.nlargest(top_n, "mean_abs_contribution")
            view_label = "lagged features" if view == "lagged" else "original covariates"
            charts[f"global_{method}_{view}"] = {
                "chart": {"type": "bar"},
                "title": {"text": f"Covariate importance - {method.upper()} ({view_label})"},
                "subtitle": {"text": "Mean absolute contribution across all predictions"},
                "xAxis": {
                    "categories": [display_name(r.feature, r.lag) for r in panel.itertuples()]
                },
                "yAxis": {"title": {"text": "Mean |contribution| (cases)"}},
                "legend": {"enabled": False},
                "tooltip": {"pointFormat": "<b>{point.y:,.1f}</b> cases"},
                "series": [
                    {
                        "color": "#2a78d6",
                        "data": [round(float(v), 1) for v in panel.mean_abs_contribution],
                        "dataLabels": {"enabled": True, "format": "{point.y:,.0f}"},
                    }
                ],
            }
    return charts


# ------------------------------------------------------------- over time


def plot_over_time(tidy: pd.DataFrame, location: str, out: Path) -> None:
    rows = tidy[(tidy.method == "shap") & (tidy.view == "base") & (tidy.location == location)]
    pivot = rows.pivot_table(index="time_period", columns="base_feature", values="contribution")
    predictions = rows.groupby("time_period")["prediction"].first()
    baseline = rows.baseline.iloc[0]
    colors = covariate_colors(list(pivot.columns))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    positive_stack = pd.Series(0.0, index=pivot.index)
    negative_stack = pd.Series(0.0, index=pivot.index)
    for feature in pivot.columns:
        values = pivot[feature]
        bottom = positive_stack.where(values >= 0, negative_stack)
        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            color=colors[feature],
            width=0.62,
            edgecolor="white",
            linewidth=1,
            label=display_name(feature, float("nan")),
        )
        positive_stack += values.clip(lower=0)
        negative_stack += values.clip(upper=0)
    ax.plot(
        predictions.index,
        predictions - baseline,
        color=INK,
        linewidth=2,
        marker="o",
        markersize=5,
        label="Prediction (relative to baseline)",
    )
    ax.axhline(0, color=NEUTRAL, linewidth=1)
    ax.set_title(
        f"What drives the forecast in {location}, month by month?\n"
        f"Stacked SHAP contributions (cases, relative to the {baseline:.0f}-case baseline)",
        fontsize=11,
        color=INK,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    _style_axes(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def over_time_highcharts(tidy: pd.DataFrame, location: str) -> dict[str, dict]:
    rows = tidy[(tidy.method == "shap") & (tidy.view == "base") & (tidy.location == location)]
    pivot = rows.pivot_table(index="time_period", columns="base_feature", values="contribution")
    predictions = rows.groupby("time_period")["prediction"].first()
    baseline = float(rows.baseline.iloc[0])
    colors = covariate_colors(list(pivot.columns))
    series = [
        {
            "type": "column",
            "name": display_name(feature, float("nan")),
            "color": colors[feature],
            "data": [round(float(v), 1) for v in pivot[feature]],
        }
        for feature in pivot.columns
    ]
    series.append(
        {
            "type": "line",
            "name": "Prediction (relative to baseline)",
            "color": INK,
            "data": [round(float(v - baseline), 1) for v in predictions],
        }
    )
    return {
        f"over_time_{location}": {
            "chart": {},
            "title": {"text": f"What drives the forecast in {location}?"},
            "subtitle": {
                "text": f"Stacked SHAP contributions relative to the {baseline:,.0f}-case baseline"
            },
            "xAxis": {"categories": list(pivot.index)},
            "yAxis": {"title": {"text": "Contribution (cases)"}},
            "plotOptions": {"column": {"stacking": "normal", "borderWidth": 1}},
            "tooltip": {"shared": True},
            "series": series,
        }
    }


# ---------------------------------------------------------- intervention


def plot_intervention(tidy: pd.DataFrame, location: str, covariate: str, out: Path) -> None:
    contributions = tidy[
        (tidy.method == "shap")
        & (tidy.view == "base")
        & (tidy.location == location)
        & (tidy.base_feature == covariate)
    ].set_index("time_period")["contribution"]
    coverage = tidy[
        (tidy.method == "shap")
        & (tidy.view == "lagged")
        & (tidy.location == location)
        & (tidy.feature == f"{covariate}_lag_0")
    ].set_index("time_period")["feature_value"]
    color = covariate_colors([covariate])[covariate]
    label = display_name(covariate, float("nan"))

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_top.plot(coverage.index, coverage, color=color, linewidth=2, marker="o", markersize=5)
    ax_top.set_title(
        f"{label} in {location}: coverage vs. its effect", fontsize=11, color=INK, loc="left"
    )
    ax_top.set_ylabel(f"{label} (coverage)", fontsize=8, color=INK)
    if 0 <= coverage.min() and coverage.max() <= 1:
        ax_top.set_ylim(-0.05, 1.05)
    _style_axes(ax_top)
    ax_top.grid(axis="y", color=GRID, linewidth=0.6)
    ax_top.grid(axis="x", visible=False)

    colors = [POSITIVE if c > 0 else NEGATIVE for c in contributions]
    bars = ax_bottom.bar(contributions.index, contributions, color=colors, width=0.62)
    ax_bottom.bar_label(bars, fmt="%+.0f", fontsize=7, color=INK, padding=2)
    ax_bottom.axhline(0, color=NEUTRAL, linewidth=1)
    ax_bottom.set_ylabel("SHAP contribution (cases)", fontsize=8, color=INK)
    _style_axes(ax_bottom)
    ax_bottom.grid(axis="y", color=GRID, linewidth=0.6)
    ax_bottom.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def intervention_highcharts(tidy: pd.DataFrame, location: str, covariate: str) -> dict[str, dict]:
    contributions = tidy[
        (tidy.method == "shap")
        & (tidy.view == "base")
        & (tidy.location == location)
        & (tidy.base_feature == covariate)
    ].set_index("time_period")["contribution"]
    coverage = tidy[
        (tidy.method == "shap")
        & (tidy.view == "lagged")
        & (tidy.location == location)
        & (tidy.feature == f"{covariate}_lag_0")
    ].set_index("time_period")["feature_value"]
    color = covariate_colors([covariate])[covariate]
    label = display_name(covariate, float("nan"))
    periods = list(contributions.index)
    return {
        f"intervention_{location}_{covariate}_coverage": {
            "chart": {"type": "line"},
            "title": {"text": f"{label} coverage in {location}"},
            "xAxis": {"categories": periods},
            "yAxis": {"title": {"text": "Coverage"}, "min": 0, "max": 1},
            "legend": {"enabled": False},
            "series": [
                {"name": label, "color": color, "data": [round(float(v), 3) for v in coverage]}
            ],
        },
        f"intervention_{location}_{covariate}_effect": {
            "chart": {"type": "column"},
            "title": {"text": f"Effect of {label.lower()} on predicted cases in {location}"},
            "xAxis": {"categories": periods},
            "yAxis": {"title": {"text": "SHAP contribution (cases)"}},
            "legend": {"enabled": False},
            "tooltip": {"pointFormat": "<b>{point.y:,.1f}</b> cases"},
            "series": [
                {
                    "data": [
                        {"y": round(float(c), 1), "color": POSITIVE if c > 0 else NEGATIVE}
                        for c in contributions
                    ],
                    "dataLabels": {"enabled": True, "format": "{point.y:,.0f}"},
                }
            ],
        },
    }


# ------------------------------------------------------------------- page

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Explanation charts</title>
<script src="https://cdn.jsdelivr.net/npm/highcharts@12.4.0/highcharts.js"
        integrity="sha384-SuKJbNf5exCoReOrvlG2qOS0m8rykJbV/EpkAZqFMxqGrDETIdBotuMnz746cTPS"
        crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/highcharts@12.4.0/highcharts-more.js"
        integrity="sha384-mi736cvbRbdpq2s9HK36kEpojV9hklPnx1Pi9zDwcFdsqRGZgtn3nUr+Qb3eQ+KC"
        crossorigin="anonymous"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #fcfcfb; color: #0b0b0b; }}
  h1 {{ font-size: 20px; }}
  .grid {{ display: grid; gap: 24px;
           grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); }}
  .chart {{ min-height: 420px; border: 1px solid #e5e4e0; border-radius: 6px; }}
</style>
</head>
<body>
<h1>Explanation charts</h1>
<p>Rendered with Highcharts from the tidy explanation data. Red pushes the
prediction up, blue pushes it down.</p>
<div class="grid" id="charts"></div>
<script>
const CHARTS = {charts_json};
const grid = document.getElementById("charts");
for (const [name, config] of Object.entries(CHARTS)) {{
  const div = document.createElement("div");
  div.className = "chart";
  div.id = name;
  grid.appendChild(div);
  Highcharts.chart(name, config);
}}
</script>
</body>
</html>
"""


def render_all(tidy: pd.DataFrame, importance: pd.DataFrame, out_dir: Path) -> list[Path]:
    png_dir = out_dir / "png"
    hc_dir = out_dir / "highcharts"
    png_dir.mkdir(parents=True, exist_ok=True)
    hc_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    charts: dict[str, dict] = {}

    peaks = peak_periods(tidy)
    for location, period in peaks.items():
        path = png_dir / f"local_{location}_{period}.png"
        plot_local(tidy, location, period, path)
        written.append(path)
        charts.update(local_highcharts(tidy, location, period))

    path = png_dir / "global_importance.png"
    plot_global_importance(importance, path)
    written.append(path)
    charts.update(global_importance_highcharts(importance))

    for location in sorted(tidy.location.unique()):
        path = png_dir / f"over_time_{location}.png"
        plot_over_time(tidy, location, path)
        written.append(path)
        charts.update(over_time_highcharts(tidy, location))

    for covariate in detect_intervention_covariates(tidy):
        for location in sorted(tidy.location.unique()):
            path = png_dir / f"intervention_{location}_{covariate}.png"
            plot_intervention(tidy, location, covariate, path)
            written.append(path)
            charts.update(intervention_highcharts(tidy, location, covariate))

    for name, config in charts.items():
        path = hc_dir / f"{name}.json"
        path.write_text(json.dumps(config, indent=2))
        written.append(path)
    index = hc_dir / "index.html"
    index.write_text(_HTML_TEMPLATE.format(charts_json=json.dumps(charts)))
    written.append(index)
    return written
