import numpy as np

from chap_explain import features


def test_discover_covariates_picks_up_extras(train_df):
    covariates = features.discover_covariates(train_df)
    assert covariates == ["rainfall", "mean_temperature", "spray_coverage"]


def test_lag_features_shift_within_location(train_df):
    covariates = features.discover_covariates(train_df)
    built = features.build_features(train_df, covariates)
    north = built[built.location == "North"].reset_index(drop=True)
    assert np.isnan(north.loc[0, "rainfall_lag_2"])
    assert north.loc[5, "rainfall_lag_2"] == north.loc[3, "rainfall"]
    # lags never leak across locations
    south = built[built.location == "South"].reset_index(drop=True)
    assert np.isnan(south.loc[0, "rainfall_lag_1"])


def test_base_feature_mapping():
    covariates = ["rainfall", "spray_coverage"]
    assert features.base_feature_of("rainfall_lag_2", covariates) == ("rainfall", 2)
    assert features.base_feature_of("spray_coverage_lag_0", covariates) == ("spray_coverage", 0)
    assert features.base_feature_of("month_sin", covariates)[0] == "seasonality"
    assert features.base_feature_of("population", covariates)[0] == "population"
