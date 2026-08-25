"""CHAP predict entry point.

Usage: python predict.py <model> <historic_data.csv> <future_data.csv> <out_file.csv>

Writes the predictions CHAP expects (time_period, location, sample_0..99) to
out_file, plus two explanation sidecar files next to it:

    <out_file stem>.explanations.csv       tidy SHAP + LIME contributions
    <out_file stem>.global_importance.csv  mean |contribution| per feature

Sidecars are suffixed _2, _3, ... when the names are taken, so repeated
predict calls in one run directory (e.g. `chap causal`) never overwrite.
"""

import argparse

import pandas as pd

from chap_explain import explain, features, model, outputs


def main(model_path: str, historic_path: str, future_path: str, out_file: str) -> None:
    payload = model.load(model_path)
    historic = features.clean_chap_csv(pd.read_csv(historic_path))
    future = features.clean_chap_csv(pd.read_csv(future_path))

    rows, x = model.build_prediction_features(payload, historic, future)
    _, samples = model.predict_samples(payload, x)

    predictions = rows[["time_period", "location"]].copy()
    for i in range(samples.shape[1]):
        predictions[f"sample_{i}"] = samples[:, i]
    predictions.to_csv(out_file, index=False)

    tidy = explain.explain(payload, rows, x)
    sidecars = outputs.sidecar_paths(out_file)
    tidy.to_csv(sidecars["explanations"], index=False)
    explain.global_importance(tidy).to_csv(sidecars["global_importance"], index=False)

    print(f"Wrote {len(predictions)} prediction rows to {out_file}")
    print(f"Wrote explanations to {sidecars['explanations']} and {sidecars['global_importance']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("historic_path")
    parser.add_argument("future_path")
    parser.add_argument("out_file")
    args = parser.parse_args()
    main(args.model_path, args.historic_path, args.future_path, args.out_file)
