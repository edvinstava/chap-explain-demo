#!/usr/bin/env bash
# Counterfactual demo: what if spray coverage had been 90% from 2022-10 on?
#
# Uses CHAP's built-in causal tooling against this model. The model is trained
# once on the original data, then predicts under both scenarios; the only
# difference between the two runs is the spray_coverage values fed to predict.
#
# Requires the chap CLI (chap-core). Point CHAP at a specific install if it is
# not on PATH, e.g.:
#   CHAP="uv run --project /path/to/chap-core chap" ./examples/counterfactual_demo.sh
#
# Outputs land in counterfactual_output/ (git-ignored):
#   causal.nc / causal_cf.nc      original and counterfactual evaluations
#   causal_original_vs_cf.html    side-by-side comparison plot (from --plot)
# The model's own explanation sidecars for BOTH predict calls are written in
# the CHAP run directory (runs/chap_explain_demo/latest/): the first
# *.explanations.csv is the original scenario, *_2 the counterfactual.

set -euo pipefail

CHAP="${CHAP:-chap}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/counterfactual_output"
mkdir -p "$OUT"

$CHAP causal build-counterfactual \
    "$REPO/example_data/synthetic_laos_intervention.csv" \
    "spray_coverage=x*0+0.9" \
    --start-time-period 2022-10 \
    --output-csv "$OUT/laos_spray90_cf.csv"

$CHAP causal \
    --model-name "$REPO" \
    --dataset-csv "$REPO/example_data/synthetic_laos_intervention.csv" \
    --counterfactual-csv "$OUT/laos_spray90_cf.csv" \
    --counterfactual-columns spray_coverage \
    --split-period 2023-01 \
    --cf-start-period 2022-10 \
    --output-file "$OUT/causal.nc" \
    --plot

echo "Done. Results in $OUT"
