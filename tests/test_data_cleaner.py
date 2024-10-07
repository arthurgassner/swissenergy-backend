import numpy as np
import pandas as pd

from model_server.data_cleaner import DataCleaner


def test__format():
    """Formatting a dateframe sets back its index by 24 and renames its columns."""

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

    # When
    formatted_df = DataCleaner._format(df=df)

    # Then
    assert len(formatted_df.columns) == 2  # 2 columns
    assert (
        formatted_df.columns[0] == "24h_later_forecast"
        and formatted_df.columns[1] == "24h_later_load"
    )
    assert (formatted_df.dtypes == "float64").all()  # correct dtype
    assert isinstance(formatted_df.index, pd.DatetimeIndex)
    # correct timezone
    assert formatted_df.index.dtype == "datetime64[ns, Europe/Zurich]"
    # -24h delay post-formatting
    assert (df.index - formatted_df.index == pd.Timedelta(24, "h")).all()


def test__force_1h_frequency():
    """Data with a missing hour ends up with a row of NaN inplace of that missing hour."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, 7890.0],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 21:00", tz="Europe/Zurich"),
                pd.Timestamp("20240101 22:00", tz="Europe/Zurich"),
                pd.Timestamp("20240102 00:00", tz="Europe/Zurich"),
            ]
        ),
    )

    # When
    enforced_frequency_df = DataCleaner._force_1h_frequency(df=df)

    # Then

    # data
    assert len(enforced_frequency_df.columns) == 2  # 2 columns
    assert (
        enforced_frequency_df.columns[0] == "Forecasted Load"
        and enforced_frequency_df.columns[1] == "Actual Load"
    )
    assert enforced_frequency_df.index.freq == "h"  # freq is now hourly
    assert len(enforced_frequency_df) == len(df) + 1  # a new row has been added
    assert (
        enforced_frequency_df.iloc[2].isna().all()
    )  # that row is the 3rd row, and filled with nan

    # index
    assert isinstance(enforced_frequency_df.index, pd.DatetimeIndex)
    assert enforced_frequency_df.index.is_monotonic_increasing
    assert enforced_frequency_df.index.is_unique

    # dtypes
    assert (enforced_frequency_df.dtypes == "float64").all()  # correct dtype
    assert (
        enforced_frequency_df.index.dtype == "datetime64[ns, Europe/Zurich]"
    )  # correct timezone
