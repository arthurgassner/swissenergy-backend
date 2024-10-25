import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

from model_server.data_loader import DataLoader

load_dotenv()


def test__query_load_and_forecast__future_ts():
    """Querying the ENTSO-E API with a timestamp 48h in the future should result in an empty df."""

    # given
    load_dotenv()
    data_loader = DataLoader(entsoe_api_key=os.getenv("ENTSOE_API_KEY"))

    # when
    fetched_df = data_loader._query_load_and_forecast(
        start_ts=pd.Timestamp(datetime.now() + timedelta(hours=48), tz="Europe/Zurich")
    )

    # then
    expected_df = pd.DataFrame(
        columns=["Forecasted Load", "Actual Load"],
        dtype=float,
        index=pd.DatetimeIndex([], dtype="datetime64[ns, Europe/Zurich]"),
    )
    assert (expected_df == fetched_df).all().all()  # same values
    assert all(
        c1 == c2 for c1, c2 in zip(expected_df.columns, fetched_df.columns)
    )  # same column names
    assert (expected_df.dtypes == fetched_df.dtypes).all()  # same dtypes
    assert (expected_df.index == fetched_df.index).all()  # same index


def test__query_load_and_forecast__24h_ago_ts():
    """Querying the ENTSO-E API with a timestamp 24h ago."""

    # given
    load_dotenv()
    data_loader = DataLoader(entsoe_api_key=os.getenv("ENTSOE_API_KEY"))

    # when
    fetched_df = data_loader._query_load_and_forecast(
        start_ts=pd.Timestamp(datetime.now() - timedelta(hours=24), tz="Europe/Zurich")
    )

    # then

    # data
    assert len(fetched_df.columns) == 2  # 2 columns
    assert (
        fetched_df.columns[0] == "Forecasted Load"
        and fetched_df.columns[1] == "Actual Load"
    )
    # data is hourly, so we should not have more than 48 datapoints
    assert len(fetched_df) <= 48
    # at least 24 of those datapoints should be NaN
    assert fetched_df["Actual Load"].isna().sum() >= 24

    # index
    assert isinstance(fetched_df.index, pd.DatetimeIndex)
    assert fetched_df.index.is_monotonic_increasing
    assert fetched_df.index.is_unique

    # dtypes
    assert (fetched_df.dtypes == "float64").all()  # correct dtype
    assert fetched_df.index.dtype == "datetime64[ns, Europe/Zurich]"  # correct timezone


def test__query_load_and_forecast__specitic_ts():
    """Querying the ENTSO-E API with a timestamp whose load/forecast is known."""

    # given
    load_dotenv()
    data_loader = DataLoader(entsoe_api_key=os.getenv("ENTSOE_API_KEY"))

    # when
    fetched_df = data_loader._query_load_and_forecast(
        start_ts=pd.Timestamp("20240701 00:30", tz="Europe/Zurich"),
        end_ts=pd.Timestamp("20240701 01:30", tz="Europe/Zurich"),
    )

    # then

    # data
    assert len(fetched_df.columns) == 2  # 2 columns
    assert (
        fetched_df.columns[0] == "Forecasted Load"
        and fetched_df.columns[1] == "Actual Load"
    )

    # data is hourly, so we should have exactly one datapoint
    assert len(fetched_df) == 1
    # And no NaN, as that data should be known
    assert fetched_df["Actual Load"].isna().sum() == 0
    # And the data should match the historically-known data, as seen on the ENTSO-E website
    # Forecasted Load [6.1.A] 01:00 - 02:00 07.10.2024
    assert fetched_df["Forecasted Load"].iloc[0] == 5693
    # Actual Load [6.1.A] 01:00 - 02:00 07.10.2024 --> Note that this can be updated by ENTSO-E
    assert fetched_df["Actual Load"].iloc[0] == 4994

    # index
    assert isinstance(fetched_df.index, pd.DatetimeIndex)
    assert fetched_df.index.is_monotonic_increasing
    assert fetched_df.index.is_unique

    # dtypes
    assert (fetched_df.dtypes == "float64").all()  # correct dtype
    assert fetched_df.index.dtype == "datetime64[ns, Europe/Zurich]"  # correct timezone


def test_enforce_data_quality():
    """Check that a df with no data quality issues goes through without changes."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when
    enforced_data_quality_df = DataLoader.enforce_data_quality(df)

    # then
    assert enforced_data_quality_df.equals(df)


def test_enforce_data_quality__index_type():
    """Check that if not isinstance(df.index, pd.DatetimeIndex), a ValueError is raised."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DataFrame(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when-then
    with pytest.raises(ValueError):
        df = DataLoader.enforce_data_quality(df)


def test_enforce_data_quality__index_tz():
    """Check that if df.index.dtype != "datetime64[ns, Europe/Zurich]", a ValueError is raised."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Berlin"),
                pd.Timestamp("20240201 23:45", tz="Europe/Berlin"),
                pd.Timestamp("20240301 23:45", tz="Europe/Berlin"),
            ]
        ),
    )

    # when-then
    with pytest.raises(ValueError):
        df = DataLoader.enforce_data_quality(df)


def test_enforce_data_quality__two_columns():
    """Check that if len(df.columns) != 2, a ValueError is raised."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
            "Some Third Column": [0, 0, 0],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when-then
    with pytest.raises(ValueError):
        df = DataLoader.enforce_data_quality(df)


def test_enforce_data_quality__column_names():
    """Check that if df.columns != ["Forecasted Load", "Actual Load"], a ValueError is raised."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Wrong column name": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when-then
    with pytest.raises(ValueError):
        df = DataLoader.enforce_data_quality(df)


def test_enforce_data_quality__dtypes():
    """Check that if df.dtypes.to_list() != ['float64', 'float64'], a ValueError is raised."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": ["a", "b", "c"],
            "Actual Load": ["d", "e", "f"],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when-then
    with pytest.raises(ValueError):
        df = DataLoader.enforce_data_quality(df)


def test_enforce_data_quality__index_is_not_monotonic_increasing():
    """Check that a df with an index that is not monotonic increasing gets sorted."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when
    index_monotic_increasing_df = DataLoader.enforce_data_quality(df)

    # then
    assert index_monotic_increasing_df.index.is_monotonic_increasing
    # reorder the rows, so that index is monotonic increasing
    expected_df = df.iloc[[0, 2, 1]]
    assert expected_df.equals(index_monotic_increasing_df)


def test_enforce_data_quality__index_is_not_unique():
    """Check that a df with an index that is not unique gets aggregated."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [100.0, np.nan, 200.0, np.nan],
            "Actual Load": [np.nan, 200.0, 300.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when
    index_unique_increasing_df = DataLoader.enforce_data_quality(df)

    # then
    assert index_unique_increasing_df.index.is_unique
    assert index_unique_increasing_df.index.is_monotonic_increasing
    assert len(index_unique_increasing_df) == df.index.nunique()
    np.testing.assert_array_equal(
        index_unique_increasing_df.iloc[0], [100.0, np.nan]  # 1st row is unchanged
    )
    np.testing.assert_array_equal(
        index_unique_increasing_df.iloc[1], [200.0, 250.0]  # 2nd row is median
    )
    np.testing.assert_array_equal(
        index_unique_increasing_df.iloc[2], [np.nan, np.nan]  # 3rd row is unchanged
    )
