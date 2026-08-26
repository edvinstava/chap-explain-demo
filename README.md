# chap-explain-demo

A [CHAP](https://github.com/dhis2-chap/chap-core) external model built for
**explainability visualization research**: every prediction ships with tidy
SHAP and LIME explanation data that can be plotted right away, on paper or
with Highcharts.

This is a study prop, optimized for clear explanations rather than forecast
accuracy. The model is deliberately simple and purely exogenous, so every
explanation is about actual covariates (rainfall, temperature, interventions)
and counterfactual scenarios behave predictably.

## The model

- **Estimator**: one fixed scikit-learn `GradientBoostingRegressor`
  (conservative hyperparameters, no tuning, no model selection).
- **Target**: raw `disease_cases`, so all explanation values are in cases.
- **Features**: current and lagged (0-3 months) values of every numeric
  covariate in the dataset, month-of-year seasonality (sin/cos), and
  population. No case-history features and no recursion: predictions are a
  pure function of covariates.
- **Covariates**: `rainfall` and `mean_temperature` are required; any extra
  numeric column (e.g. `spray_coverage`, bednet coverage) is picked up
  automatically and lagged the same way.
- **Uncertainty**: 100 prediction samples via residual bootstrap. Residuals
  come from a per-location holdout of the last months of training data;
  holdout metrics are written to `{model}.metrics.json`.

## How CHAP runs it

The `MLproject` file declares the standard train/predict entry points with
`uv_env`, so CHAP builds an isolated environment automatically:

```bash
chap eval --model-name /path/to/chap-explain-demo \
  --dataset-csv example_data/synthetic_laos_intervention.csv \
  --output-file results/evaluation.nc
```

`--model-name` also accepts the GitHub URL of this repository.

## Explanation outputs

Every `predict` call writes two sidecar files next to the predictions CSV:

### `<predictions>.explanations.csv`

One tidy row per (location, time_period, method, view, feature):

| column | meaning |
|---|---|
| `method` | `shap` (exact TreeExplainer values) or `lime` (LimeTabularExplainer weights) |
| `view` | `lagged` = per model feature (e.g. `rainfall_lag_2`); `base` = aggregated per original covariate |
| `feature`, `base_feature`, `lag` | feature identity; `base` rows aggregate over lags |
| `condition` | LIME's human-readable rule, e.g. `rainfall_lag_2 > 50.33` |
| `feature_value` | the feature's value for this prediction |
| `contribution` | contribution in **cases** (positive pushes the prediction up) |
| `baseline` | SHAP expected value / LIME surrogate intercept |
| `prediction` | the model output being explained |
| `local_prediction` | LIME's surrogate output (fidelity check; NaN for SHAP) |

SHAP rows satisfy `baseline + sum(contributions) == prediction` exactly in the
`lagged` view, and the `base` view is an exact regrouping of it. LIME weights
are a local approximation and do not add up; compare `local_prediction` with
`prediction` to judge fidelity.

### `<predictions>.global_importance.csv`

Mean absolute and mean signed contribution per method, view, and feature.

Repeated predict calls in one run directory (e.g. `chap causal`, backtest
splits) never overwrite: later sidecars get `_2`, `_3`, ... suffixes.

## Plots

```bash
uv run --group plots python plots.py <run_dir>   # e.g. runs/chap_explain_demo/latest
```

renders, from the tidy CSV alone, the lean default chart set:

- **Local "why this month"** - the predicted-peak month per location, as a
  2x2 grid (SHAP/LIME x lagged/original views) of signed contribution bars.
- **Global importance** - mean |contribution| per feature, both views and
  methods.
- **Contributions over time** - stacked monthly SHAP contributions per
  covariate with the prediction line on top, per location.

Filters keep the output small enough to hand to study participants. To limit
the visualizations to a **single location**, pass its name to `--locations`;
every chart family is then rendered for that location only:

```bash
python plots.py <run_dir> --locations Bokeo             # one location only
python plots.py <run_dir> --locations Bokeo,Vientiane   # or a few
python plots.py <run_dir> --periods 2023-10,2023-11     # only these months
python plots.py <run_dir> --charts local,over_time      # only these families
python plots.py <run_dir> --charts intervention         # opt-in: coverage vs. effect
```

Filters combine, so `--locations Bokeo --charts local` yields exactly one
chart. Unknown location or period names fail with the list of available
values.

`--charts` accepts `local`, `global`, `over_time`, and `intervention`.
**Intervention focus** charts (auto-detected extra covariates such as
`spray_coverage` against their own contribution) are not rendered by default -
request them with `--charts intervention`.

Each chart is written as a print-ready PNG (`plots/png/`) and as a Highcharts
config JSON plus a self-contained `plots/highcharts/index.html`. Signed
contributions follow the SHAP convention: red pushes the prediction up, blue
pushes it down. Interactive chart tooltips show the covariate value behind
each contribution on hover (per lag where the lagged values differ, collapsed
to a single value where they do not).

## Counterfactual analysis (CHAP built-in)

The model needs no special support: CHAP's `chap causal` trains it once and
calls predict twice with modified covariates. A worked scenario ("what if
spray coverage had been 90% from 2022-10?") is in
[`examples/counterfactual_demo.sh`](examples/counterfactual_demo.sh):

```bash
./examples/counterfactual_demo.sh
```

Both predict calls emit explanation sidecars in the same run directory
(original first, counterfactual as `*_2`), so factual and counterfactual
explanations can be compared side by side.

CHAP's model-agnostic LIME (`chap explain-lime`) also works with this model
as a third, perturbation-based explanation method.

## Run as a service with chapkit

`main.py` wraps the same train/predict scripts as a
[chapkit](https://github.com/dhis2-chap/chapkit) REST service with full model
metadata. chapkit does not activate the model's environment itself, so launch
it from inside this project's environment (Python 3.13, matching chapkit's
requirement):

```bash
uv sync
uv run --with chapkit python main.py
```

Train/predict then run as async jobs (`POST /api/v1/ml/$train`,
`POST /api/v1/ml/$predict`); the prediction workspace artifact
(`GET /api/v1/artifacts/{id}/$download`) contains `predictions.csv` together
with the explanation sidecar files.

To surface the model in a running chap-core (and the DHIS2 Modeling App),
start the service with self-registration enabled (single quotes: `$register`
is literal):

```bash
SERVICEKIT_ORCHESTRATOR_URL='http://localhost:8000/v2/services/$register' \
SERVICEKIT_HOST=host.docker.internal \
SERVICEKIT_PORT=9090 \
uv run --with chapkit python main.py
```

`SERVICEKIT_HOST` is the address chap-core calls back on; from Docker
containers on the same machine that is `host.docker.internal`. The service
registers on startup and pings chap-core once a minute to stay listed, so
keep it running while using the model. (The bare
`chapkit mlproject run . --port 9090` runner still works for local REST use,
but serves no model metadata, so chap-core cannot register it.)

## Development

```bash
uv sync --group dev --group plots
uv run pytest
uv run ruff check .
```

Local smoke run against the bundled example data:

```bash
uv run python train.py example_data/synthetic_laos_intervention.csv output/model
uv run python predict.py output/model <historic.csv> <future.csv> output/predictions.csv
```

The test suite pins the invariants the study depends on, including SHAP
additivity, base-view aggregation, deterministic samples, and LIME weight
signs (the `lime` package stores negated weights under label 0 in regression
mode; see `chap_explain/explain.py`).

## Example data

`example_data/synthetic_laos_intervention.csv` is a copy of the synthetic
dataset generated by chap-core's `scripts/generate_synthetic_intervention_data.py`
(six Laos provinces, monthly 2020-2023, with a `spray_coverage` intervention).
It is synthetic; no real health data is included.
