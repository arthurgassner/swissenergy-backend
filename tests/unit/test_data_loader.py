import os
from datetime import datetime, timedelta

import pandas as pd
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
    assert all(c1 == c2 for c1, c2 in zip(expected_df.columns, fetched_df.columns))  # same column names
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
    assert fetched_df.columns[0] == "Forecasted Load" and fetched_df.columns[1] == "Actual Load"
    # data is hourly, so we should not have more than 49 -- 49 if hour change happened in the last 48h -- datapoints
    assert len(fetched_df) <= 49
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
    assert fetched_df.columns[0] == "Forecasted Load" and fetched_df.columns[1] == "Actual Load"

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
