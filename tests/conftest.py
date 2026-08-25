import numpy as np
import pandas as pd
import pytest

from chap_explain import features, model


def _synthetic_panel(
    n_months: int = 36, locations: tuple[str, ...] = ("North", "South")
) -> pd.DataFrame:
    """Small deterministic panel in CHAP's harmonized CSV format.

    Cases depend on rainfall two months earlier and on spray coverage, so
    tests can assert explanations point at known drivers.
    """
    rng = np.random.default_rng(0)
    frames = []
    for loc_i, location in enumerate(locations):
        months = pd.period_range("2020-01", periods=n_months, freq="M")
        month_number = months.month.to_numpy()
        rainfall = (
            40 + 35 * np.sin(2 * np.pi * (month_number - 6) / 12) + rng.normal(0, 5, n_months)
        )
        rainfall = np.clip(rainfall, 0, None)
        temperature = (
            24 + 3 * np.sin(2 * np.pi * (month_number - 4) / 12) + rng.normal(0, 0.5, n_months)
        )
        spray = np.zeros(n_months)
        spray[18:24] = 0.8
        population = 100_000 * (1 + loc_i)
        rain_lag2 = np.roll(rainfall, 2)
        cases = 200 + 200 * loc_i + 4 * rain_lag2 - 150 * spray + rng.normal(0, 10, n_months)
        frames.append(
            pd.DataFrame(
                {
                    "time_period": months.strftime("%Y-%m"),
                    "location": location,
                    "rainfall": np.round(rainfall, 1),
                    "mean_temperature": np.round(temperature, 2),
                    "population": population,
                    "spray_coverage": spray,
                    "disease_cases": np.round(np.clip(cases, 0, None)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    return _synthetic_panel()


@pytest.fixture(scope="session")
def train_df(panel: pd.DataFrame) -> pd.DataFrame:
    return features.clean_chap_csv(panel[panel.time_period <= "2022-06"])


@pytest.fixture(scope="session")
def future_df(panel: pd.DataFrame) -> pd.DataFrame:
    future = panel[panel.time_period > "2022-06"].drop(columns=["disease_cases"])
    return features.clean_chap_csv(future)


@pytest.fixture(scope="session")
def payload(train_df: pd.DataFrame) -> dict:
    return model.train(train_df)
