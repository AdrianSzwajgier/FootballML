import numpy as np
import pandas as pd
import pytest

from ml.ml.data_preprocessing import filter_clubs_by_min_matches


@pytest.fixture(name="sample_data")
def fixture_sample_data():
    rng = np.random.default_rng(12345)
    data = {
        "h_name": ["dummy_name01"] * 5 + ["dummy_name02"] * 7
        + ["dummy_name03"] * 2 + ["dummy_name04"] * 2,
        "a_name": ["dummy_name02"] * 5 + ["dummy_name01"] * 7
        + ["dummy_name04"] * 2 + ["dummy_name03"] * 2,
        "h_goals": rng.integers(0,  5, size=16),
        "a_goals": rng.integers(0,  5, size=16),
        "id": list(range(16))
    }
    df = pd.DataFrame(data)
    return df


def test_filter_clubs_by_min_matches(sample_data):
    filtered_df = filter_clubs_by_min_matches(
        sample_data,
        "id",
        "h_name",
        "a_name",
        5
    )

    names_to_remove = ["dummy_name03", "dummy_name04"]

    assert filtered_df.columns.equals(sample_data.columns)
    assert not filtered_df["h_name"].isin(names_to_remove).any()
    assert not filtered_df.empty
