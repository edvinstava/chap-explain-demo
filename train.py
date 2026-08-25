"""CHAP train entry point: python train.py <train_data.csv> <model_out>."""

import argparse
import json

import pandas as pd

from chap_explain import features, model


def main(train_data: str, model_out: str) -> None:
    df = features.clean_chap_csv(pd.read_csv(train_data))
    payload = model.train(df)
    model.save(payload, model_out)

    metrics = payload["metrics"]
    with open(f"{model_out}.metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"Trained on {len(df)} rows, {df['location'].nunique()} locations, "
        f"{len(payload['feature_names'])} features (covariates: {payload['covariates']})"
    )
    print(
        f"Holdout MAE: {metrics['mae']:.1f}, RMSE: {metrics['rmse']:.1f} "
        f"({metrics['holdout_rows']} rows)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_data")
    parser.add_argument("model_out")
    args = parser.parse_args()
    main(args.train_data, args.model_out)
