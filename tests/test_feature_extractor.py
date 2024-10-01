import numpy as np
import pandas as pd
import pytest

from model_server.feature_extractor import FeatureExtractor


def test__timedelta_ago_load__0h_ago():
    # Given a df with one 'Actual Load'
    df = pd.DataFrame(
        {
            "24h_later_forecast": [np.nan] * 48,
            "24h_later_load": list(range(48)),
        },
        index=pd.DatetimeIndex(
            pd.date_range(
                start=pd.Timestamp("20240115 12:00", tz="Europe/Zurich"),
                periods=48,
                freq="h",
            )
        ),
    )

    # When
    df["0h_ago_load"] = FeatureExtractor._compute_timedelta_ago_load(
        df, timedelta=pd.Timedelta(0, "h")
    )

    # Then

    # To know this, we would need to have the '24h_later_load' starting at 2024-01-15 11:00, which we don't
    assert np.isnan(
        df.loc[pd.Timestamp("20240116 11:00", tz="Europe/Zurich"), "0h_ago_load"]
    )

    # We know this, since we know the '24h_later_load' starting at 2024-01-15 12:00
    assert (
        df.loc[pd.Timestamp("20240116 12:00", tz="Europe/Zurich"), "0h_ago_load"] == 0
    )
    assert (
        df.loc[pd.Timestamp("20240116 13:00", tz="Europe/Zurich"), "0h_ago_load"] == 1
    )
    assert (
        df.loc[pd.Timestamp("20240116 14:00", tz="Europe/Zurich"), "0h_ago_load"] == 2
    )
    assert (
        df.loc[pd.Timestamp("20240117 00:00", tz="Europe/Zurich"), "0h_ago_load"] == 12
    )
