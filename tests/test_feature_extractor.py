import numpy as np
import pandas as pd
import pytest

from model_server.feature_extractor import FeatureExtractor


def test__n_hours_ago_load__0h_ago():
    """Check that we can enrich the data with the load between now and now+1h."""
    # Given a df 48h of loads
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
    df["0h_ago_load"] = FeatureExtractor._n_hours_ago_load(df, n_hours=0)

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


def test__n_hours_ago_load__1h_ago():
    """Check that we can enrich the data with the load between '1h ago' and 'now'."""
    # Given a df 48h of loads
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
    df["1h_ago_load"] = FeatureExtractor._n_hours_ago_load(df, n_hours=1)

    # Then

    # To know this, we would need to have the '24h_later_load' starting at 2024-01-15 11:00, which we don't
    assert np.isnan(
        df.loc[pd.Timestamp("20240116 12:00", tz="Europe/Zurich"), "1h_ago_load"]
    )

    # We know this, since we know the '24h_later_load' starting at 2024-01-15 12:00
    assert (
        df.loc[pd.Timestamp("20240116 13:00", tz="Europe/Zurich"), "1h_ago_load"] == 0
    )
    assert (
        df.loc[pd.Timestamp("20240116 14:00", tz="Europe/Zurich"), "1h_ago_load"] == 1
    )
    assert (
        df.loc[pd.Timestamp("20240116 15:00", tz="Europe/Zurich"), "1h_ago_load"] == 2
    )
    assert (
        df.loc[pd.Timestamp("20240117 00:00", tz="Europe/Zurich"), "1h_ago_load"] == 11
    )


def test__n_hours_ago_load__8h_ago():
    """Check that we can enrich the data with the load between '8h ago' and '7h ago'."""
    # Given a df 48h of loads
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
    df["8h_ago_load"] = FeatureExtractor._n_hours_ago_load(df, n_hours=8)

    # Then

    # To know this, we would need to have the '24h_later_load' starting at 2024-01-15 11:00, which we don't
    assert np.isnan(
        df.loc[pd.Timestamp("20240116 19:00", tz="Europe/Zurich"), "8h_ago_load"]
    )

    # We know this, since we know the '24h_later_load' starting at 2024-01-15 12:00
    assert (
        df.loc[pd.Timestamp("20240116 20:00", tz="Europe/Zurich"), "8h_ago_load"] == 0
    )
    assert (
        df.loc[pd.Timestamp("20240116 21:00", tz="Europe/Zurich"), "8h_ago_load"] == 1
    )
    assert (
        df.loc[pd.Timestamp("20240116 22:00", tz="Europe/Zurich"), "8h_ago_load"] == 2
    )
    assert (
        df.loc[pd.Timestamp("20240117 00:00", tz="Europe/Zurich"), "8h_ago_load"] == 4
    )


def test__rolling_window__1h_window():
    """Check that computing the rolling window with a window-size of 1h simply gives the 1h_ago_load."""
    # Given a df 48h of loads
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
    df["1h_min"] = FeatureExtractor._rolling_window(df, n_hours=1, stat=np.min)
    df["1h_max"] = FeatureExtractor._rolling_window(df, n_hours=1, stat=np.max)
    df["1h_median"] = FeatureExtractor._rolling_window(df, n_hours=1, stat=np.median)

    # Then
    df["1h_ago_load"] = FeatureExtractor._n_hours_ago_load(df, n_hours=1)
    np.testing.assert_equal(df["1h_min"].values, df["1h_ago_load"].values)
    np.testing.assert_equal(df["1h_max"].values, df["1h_ago_load"].values)
    np.testing.assert_equal(df["1h_median"].values, df["1h_ago_load"].values)


def test__rolling_window__2h_window():
    """Check that computing the rolling window with a window-size of 2h gives the expected values."""
    # Given a df 48h of loads
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
    df["2h_min"] = FeatureExtractor._rolling_window(df, n_hours=2, stat=np.min)
    df["2h_max"] = FeatureExtractor._rolling_window(df, n_hours=2, stat=np.max)
    df["2h_median"] = FeatureExtractor._rolling_window(df, n_hours=2, stat=np.median)

    # Then

    # We know this, since we know the '24h_later_load' starting at 2024-01-15 12:00
    # Hence we know that the load was 0 (12:00 -> 13:00) and then 1 (12:00 -> 13:00)
    assert df.loc[pd.Timestamp("20240116 14:00", tz="Europe/Zurich"), "2h_min"] == 0
    assert df.loc[pd.Timestamp("20240116 14:00", tz="Europe/Zurich"), "2h_max"] == 1
    assert (
        df.loc[pd.Timestamp("20240116 14:00", tz="Europe/Zurich"), "2h_median"] == 0.5
    )
