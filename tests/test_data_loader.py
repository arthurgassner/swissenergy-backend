import numpy as np
import pandas as pd
import pytest

from model_server.data_loader import DataLoader


def test__get_latest_ts_with_actual_load__empty_dataframe():
    """Empty dataframe should return 2014-01-01 00:00"""
    excepted_ts = pd.Timestamp("20140101 00:00", tz="Europe/Zurich")
    assert excepted_ts == DataLoader.get_latest_ts_with_actual_load(df=pd.DataFrame())


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
    ts = DataLoader.get_latest_ts_with_actual_load(df=df)

    # Then
    excepted_ts = pd.Timestamp("20140101 00:00", tz="Europe/Zurich")
    assert excepted_ts == ts


def test__get_latest_ts_with_actual_load():
    """Dataframe with an 'Actual Load' should return the datetime of that row"""

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
    ts = DataLoader.get_latest_ts_with_actual_load(df=df)

    # Then
    excepted_ts = pd.Timestamp("20240201 23:45", tz="Europe/Zurich")
    assert excepted_ts == ts
