import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

from model_server.data_loader import DataLoader


def test__get_latest_ts_with_actual_load__empty_dataframe():
    """Empty dataframe should return 2014-01-01 00:00"""
    excepted_ts = pd.Timestamp("20140101 00:00", tz="Europe/Zurich")
    assert excepted_ts == DataLoader._get_latest_ts_with_actual_load(df=pd.DataFrame())


def test__get_latest_ts_with_actual_load__missing_actual_load():
    """Dataframe with no 'Actual Load' should return 2014-01-01 00:00"""

    # Given a df with no 'Actual Load' values
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0],
            "Actual Load": [np.nan],
        },
        index=pd.DatetimeIndex([pd.Timestamp("20240101 23:45", tz="Europe/Zurich")]),
    )

    # When
    ts = DataLoader._get_latest_ts_with_actual_load(df=df)

    # Then
    excepted_ts = pd.Timestamp("20140101 00:00", tz="Europe/Zurich")
    assert excepted_ts == ts


def test__get_latest_ts_with_actual_load():
    """Dataframe with an 'Actual Load' should return the timestamp of that row"""

    # Given a df with no 'Actual Load' values
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

    # When
    ts = DataLoader._get_latest_ts_with_actual_load(df=df)

    # Then
    excepted_ts = pd.Timestamp("20240201 23:45", tz="Europe/Zurich")
    assert excepted_ts == ts


def test__query_load_and_forecast__future_ts():
    """Querying the ENTSO-E API with a timestamp in the future should result in an empty df."""

    # given
    load_dotenv()
    data_loader = DataLoader(entsoe_api_key=os.getenv("ENTSOE_API_KEY"))

    # when
    fetched_df = data_loader._query_load_and_forecast(
        start_ts=pd.Timestamp(datetime.now() + timedelta(hours=24), tz="Europe/Zurich")
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
    assert len(fetched_df.columns) == 2  # 2 columns
    assert (
        fetched_df.columns[0] == "Forecasted Load"
        and fetched_df.columns[1] == "Actual Load"
    )
    assert (fetched_df.dtypes == "float64").all()  # correct dtype
    assert isinstance(fetched_df.index, pd.DatetimeIndex)
    assert fetched_df.index.dtype == "datetime64[ns, Europe/Zurich]"  # correct timezone
