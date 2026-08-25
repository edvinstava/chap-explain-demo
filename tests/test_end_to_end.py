"""End-to-end smoke test through the CHAP entry-point scripts."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def _run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(REPO / script), *args], check=True, cwd=REPO)


def test_train_predict_via_scripts(panel, tmp_path):
    train_csv = tmp_path / "train.csv"
    future_csv = tmp_path / "future.csv"
    panel[panel.time_period <= "2022-06"].to_csv(train_csv, index=False)
    future = panel[panel.time_period > "2022-06"].drop(columns=["disease_cases"])
    future.to_csv(future_csv, index=False)

    model_file = tmp_path / "model"
    out_file = tmp_path / "predictions.csv"
    _run("train.py", str(train_csv), str(model_file))
    assert model_file.exists()
    assert (tmp_path / "model.metrics.json").exists()

    _run("predict.py", str(model_file), str(train_csv), str(future_csv), str(out_file))
    predictions = pd.read_csv(out_file)
    assert len(predictions) == len(future)
    sample_columns = [c for c in predictions.columns if c.startswith("sample_")]
    assert len(sample_columns) == 100
    assert {"time_period", "location"} <= set(predictions.columns)

    explanations = tmp_path / "predictions.explanations.csv"
    importance = tmp_path / "predictions.global_importance.csv"
    assert explanations.exists() and importance.exists()

    # a second predict call in the same directory must not overwrite sidecars
    _run("predict.py", str(model_file), str(train_csv), str(future_csv), str(out_file))
    assert (tmp_path / "predictions.explanations_2.csv").exists()
    assert explanations.read_bytes() == (tmp_path / "predictions.explanations_2.csv").read_bytes()
