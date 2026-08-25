import numpy as np

from chap_explain import model


def test_train_payload_contents(payload):
    assert payload["covariates"] == ["rainfall", "mean_temperature", "spray_coverage"]
    assert len(payload["residuals"]) > 0
    assert set(payload["metrics"]) >= {"mae", "rmse", "per_location_mae"}
    assert list(payload["background"].columns) == payload["feature_names"]


def test_predict_output_shape_and_determinism(payload, train_df, future_df):
    rows, x = model.build_prediction_features(payload, train_df, future_df)
    assert len(rows) == len(future_df)
    assert not x.isna().any().any()

    point_a, samples_a = model.predict_samples(payload, x)
    point_b, samples_b = model.predict_samples(payload, x)
    assert samples_a.shape == (len(rows), model.N_SAMPLES)
    assert (samples_a >= 0).all() and np.isfinite(samples_a).all()
    np.testing.assert_array_equal(samples_a, samples_b)
    np.testing.assert_array_equal(point_a, point_b)


def test_model_responds_to_covariate_change(payload, train_df, future_df):
    """The counterfactual contract: changing a covariate changes predictions."""
    rows, x = model.build_prediction_features(payload, train_df, future_df)
    point, _ = model.predict_samples(payload, x)

    sprayed = future_df.copy()
    sprayed["spray_coverage"] = 0.9
    _, x_cf = model.build_prediction_features(payload, train_df, sprayed)
    point_cf, _ = model.predict_samples(payload, x_cf)
    assert point_cf.mean() < point.mean()


def test_save_load_roundtrip(payload, tmp_path):
    path = str(tmp_path / "model")
    model.save(payload, path)
    loaded = model.load(path)
    assert loaded["feature_names"] == payload["feature_names"]
